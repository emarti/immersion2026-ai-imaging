#!/usr/bin/env python3
"""Step 4c: WIDTH sample -- wider (4x the channels).

Design 4c -- the baseline 4a widened to three conv blocks 1 -> 32 -> 64 -> 128 (4x the
channels, ~102k params vs the baseline's ~8k), same dropout (0.6 / 0.2). Its first filter
is also 5x5 rather than 3x3 (pad 2), giving block 1 a larger receptive field on the raw
patch. It samples the "more capacity" direction: does a bigger net help on this small
dataset, or just overfit? The gap over 4a is deliberately large -- an earlier 2x version
(16-32-64, ~28k) landed so close to the baseline that the comparison taught nothing.
Blocks 2-3 are Conv(3x3, pad 1) -> BatchNorm -> ReLU -> MaxPool(2), then global average
pooling and a small 2-layer classifier. Binary output (CDR-positive=1 vs CDR-negative=0)
trained with BCEWithLogitsLoss + AdamW.

Each epoch trains on the training split and evaluates on the validation split;
per-epoch train/val loss, accuracy and validation AUC are written to
``outputs/training_log_4c.csv``, and the final weights to ``outputs/model_4c.pt``
(step6 reads them for Grad-CAM). step5 overlays the CSVs from all designs (4a-4d).

This file is self-contained (the step4 designs differ only in ``Net``).

Usage:
    python step4a-train-network.py [--epochs N]
"""
from __future__ import annotations

import argparse
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.metrics import roc_auc_score

from common import load_config, load_yaml, manifest_yaml

CSV_NAME = "training_log_4c.csv"
MODEL_NAME = "model_4c.pt"


class OASISSlices(Dataset):
    """One 2D slice per item: (1 x H x W float tensor in [0,1], float label)."""

    def __init__(self, manifest, outputs_path, split, transform):
        self.rows = [r for r in manifest["slices"] if r["split"] == split]
        self.outputs_path = outputs_path
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        img = Image.open(os.path.join(self.outputs_path, row["png_path"]))
        x = self.transform(img)                                   # 1 x H x W
        y = torch.tensor(float(row["label"]), dtype=torch.float32)
        cdr = torch.tensor(float(row["cdr"]), dtype=torch.float32)   # CDR grade of this patch
        return x, y, cdr


class Net(nn.Module):
    """Design 4c: 3 conv blocks 1 -> 32 -> 64 -> 128 (wider, 5x5 first), moderate dropout (0.6 / 0.2)."""

    def __init__(self):
        super(Net, self).__init__()
        # 5x5 first filter (pad 2 keeps the output the same size as the input): a wider
        # net can afford a larger receptive field on the raw patch, so block 1 sees more
        # context per step than the baseline's 3x3.
        self.conv1 = nn.Conv2d(1, 32, 5, 1, padding=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, 1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout1 = nn.Dropout(0.6)
        self.fc1 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        # Input is N x 1 x 84 x 68 (a left/right hippocampus patch from step3).
        # AdaptiveAvgPool makes the exact spatial size irrelevant.
        x = F.max_pool2d(F.relu(self.bn1(self.conv1(x))), 2)      # -> 32 channels (5x5 filter)
        x = F.max_pool2d(F.relu(self.bn2(self.conv2(x))), 2)      # -> 64 channels
        x = F.max_pool2d(F.relu(self.bn3(self.conv3(x))), 2)      # -> 128 channels
        x = self.gap(x)                                           # -> 128 x 1 x 1
        x = torch.flatten(x, 1)                                   # -> 128
        x = self.dropout1(x)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)                                           # -> 1 logit
        return x.squeeze(1)                                       # -> (N,)


def train(model, device, train_loader, optimizer):
    """Train one epoch; return this epoch's average loss and accuracy."""
    model.train()
    loss_sum = 0.0
    correct = 0
    for data, target, _ in train_loader:              # CDR grade unused in training
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.binary_cross_entropy_with_logits(output, target)  # nn.BCEWithLogitsLoss
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * len(data)
        correct += ((output > 0).float() == target).sum().item()   # logit > 0 <=> prob > 0.5
    n = len(train_loader.dataset)
    return loss_sum / n, correct / n


GRADES = (0.5, 1.0, 2.0)   # CDR-positive severities pooled under label 1


def evaluate(model, device, loader):
    """Evaluate (no training); return loss, accuracy, sensitivity, specificity, AUC, and
    a per-CDR-grade accuracy dict for the CDR-positive grades 0.5 / 1 / 2.

    Positive class = CDR-positive (label 1). sensitivity = TP/(TP+FN) (recall of
    CDR-positive), specificity = TN/(TN+FP) (recall of CDR-negative). AUC is the area under
    the ROC curve -- the chance that a randomly chosen CDR-positive patch scores above a
    randomly chosen CDR-negative one -- and unlike the three above it does NOT depend on the
    0.5 threshold. Per-grade accuracy is the recall within that grade (all its patches are
    label 1); nan if the grade has no patches in this split.
    """
    model.eval()
    loss_sum = 0.0
    tp = tn = fp = fn = 0
    g_correct = {g: 0 for g in GRADES}
    g_total = {g: 0 for g in GRADES}
    all_logits = []        # kept for AUC, which needs the continuous score, not the 0/1 call
    all_targets = []
    with torch.no_grad():
        for data, target, cdr in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss_sum += F.binary_cross_entropy_with_logits(
                output, target, reduction='sum').item()
            pred = (output > 0).float()                          # logit > 0 <=> prob > 0.5
            tp += int(((pred == 1) & (target == 1)).sum())
            tn += int(((pred == 0) & (target == 0)).sum())
            fp += int(((pred == 1) & (target == 0)).sum())
            fn += int(((pred == 0) & (target == 1)).sum())
            correct = (pred.cpu() == target.cpu())               # per-patch correctness
            for g in GRADES:
                mask = cdr == g
                g_total[g] += int(mask.sum())
                g_correct[g] += int((correct & mask).sum())
            all_logits.append(output.cpu())
            all_targets.append(target.cpu())
    n = len(loader.dataset)
    acc = (tp + tn) / n
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    # AUC is rank-based, so the raw logits work directly -- passing them through a sigmoid
    # first would not change the ordering, and skipping it avoids float32 saturation turning
    # very confident logits into exact ties. Undefined if a split holds only one class.
    targets = torch.cat(all_targets).numpy()
    logits = torch.cat(all_logits).numpy()
    auc = roc_auc_score(targets, logits) if len(set(targets.tolist())) == 2 else float("nan")
    grade_acc = {g: (g_correct[g] / g_total[g] if g_total[g] else float("nan"))
                 for g in GRADES}
    return loss_sum / n, acc, sens, spec, auc, grade_acc


def main():
    parser = argparse.ArgumentParser(description='OASIS CNN design 4c (wider 32-64-128, 5x5 first, dropout 0.6/0.2)')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='input batch size (default: 32)')
    parser.add_argument('--epochs', type=int, default=None, metavar='N',
                        help='number of epochs to train (default: from config.yaml)')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate (default: 1e-4)')
    parser.add_argument('--weight-decay', type=float, default=1e-4, metavar='WD',
                        help='AdamW weight decay (default: 1e-4)')
    parser.add_argument('--no-accel', action='store_true',
                        help='disables accelerator')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    args = parser.parse_args()
    start = time.perf_counter()

    use_accel = not args.no_accel and torch.accelerator.is_available()

    torch.manual_seed(args.seed)

    if use_accel:
        device = torch.accelerator.current_accelerator()
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    config = load_config()
    epochs = args.epochs if args.epochs is not None else config["epochs"]
    outputs_path = config["outputs_path"]
    manifest = load_yaml(manifest_yaml(config))

    # step3 already crops each slice to a left/right hippocampus patch, so no crop
    # here (random-crop augmentation within the patch comes later).
    transform = transforms.ToTensor()   # 8-bit PNG -> 1 x H x W float in [0,1]
    train_ds = OASISSlices(manifest, outputs_path, "train", transform)
    val_ds = OASISSlices(manifest, outputs_path, "validate", transform)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    print(f"Training on {len(train_ds)} slices, validating on {len(val_ds)}")

    model = Net().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)

    history = {k: [] for k in ("train_loss", "train_acc", "val_loss", "val_acc",
                               "val_sens", "val_spec", "val_bal", "val_auc",
                               "val_cdr05", "val_cdr10", "val_cdr20")}
    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train(model, device, train_loader, optimizer)
        va_loss, va_acc, va_sens, va_spec, va_auc, va_grade = evaluate(
            model, device, val_loader)
        va_bal = (va_sens + va_spec) / 2
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)
        history["val_sens"].append(va_sens)
        history["val_spec"].append(va_spec)
        history["val_bal"].append(va_bal)
        history["val_auc"].append(va_auc)
        history["val_cdr05"].append(va_grade[0.5])
        history["val_cdr10"].append(va_grade[1.0])
        history["val_cdr20"].append(va_grade[2.0])
        print(f"Epoch {epoch:3d}/{epochs}   "
              f"train loss {tr_loss:.4f} acc {tr_acc:.3f}   "
              f"val loss {va_loss:.4f} acc {va_acc:.3f} "
              f"sens {va_sens:.3f} spec {va_spec:.3f} bal {va_bal:.3f} auc {va_auc:.3f}")

    # Save the per-epoch numbers as a text file (CSV); step5 plots it.
    log_path = os.path.join(outputs_path, CSV_NAME)
    with open(log_path, "w") as f:
        f.write("epoch,train_loss,train_acc,val_loss,val_acc,val_sens,val_spec,val_bal_acc,"
                "val_auc,val_acc_cdr05,val_acc_cdr10,val_acc_cdr20\n")
        for epoch in range(1, epochs + 1):
            i = epoch - 1
            f.write(f"{epoch},{history['train_loss'][i]:.6f},{history['train_acc'][i]:.6f},"
                    f"{history['val_loss'][i]:.6f},{history['val_acc'][i]:.6f},"
                    f"{history['val_sens'][i]:.6f},{history['val_spec'][i]:.6f},"
                    f"{history['val_bal'][i]:.6f},{history['val_auc'][i]:.6f},"
                    f"{history['val_cdr05'][i]:.6f},{history['val_cdr10'][i]:.6f},"
                    f"{history['val_cdr20'][i]:.6f}\n")
    print(f"Saved training log -> {log_path}")
    model_path = os.path.join(outputs_path, MODEL_NAME)
    torch.save(model.state_dict(), model_path)      # step6 (Grad-CAM) reloads this
    print(f"Saved model weights -> {model_path}")
    elapsed = time.perf_counter() - start
    print(f"Run time: {elapsed:.1f} s "
          f"({epochs} epochs on {device}, {elapsed / epochs:.2f} s/epoch)")


if __name__ == '__main__':
    main()
