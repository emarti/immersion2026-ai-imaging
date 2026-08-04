#!/usr/bin/env python3
"""Step 5: plot the training + validation curves from step4.

Reads the CSV log written by step4 (``outputs/training_log_4.csv``) and plots training
loss, training accuracy, validation loss, validation accuracy, validation balanced
accuracy, and validation AUC, each vs epoch, in a 2x3 grid. The validation panels also
draw a running average (a smoothed line over the noisy raw curve) to show the trend.
Saves ``outputs/step5-training_comparison.png``. Also prints -- and saves to
``outputs/step5-training_summary.txt`` -- the average validation accuracy / sensitivity /
specificity / balanced accuracy / AUC over the last N epochs (``avg_last_epochs`` in
config.yaml), plus a breakdown of validation accuracy by CDR grade (0.5 / 1 / 2, pooled
under label 1).

Also writes ``outputs/step5-roc_curve.png``: the ROC curve for VALIDATE, computed by
re-running the trained checkpoint (``outputs/model_4.pt``) once, fresh -- unlike every
other number in this script, which comes from training_log_4.csv alone. This is a real,
if small, behavior change from earlier versions of step5: even the default (non-
``--reveal``) run now needs the checkpoint and manifest.yaml to exist, not just the CSV
log. The AUC in this figure's legend is computed from that single pass and will not
exactly match the CSV-derived running-tail-mean AUC printed elsewhere in this script's
output (one is one fixed evaluation, the other an average over many training epochs).

TEST is not touched by default. Pass ``--reveal`` to additionally evaluate it ONCE
(reusing the same freshly-loaded checkpoint), add two more panels to the main comparison
plot (headline metrics and per-grade accuracy, validate vs test) plus a printed/saved
test section, and add a second curve to the ROC figure -- meant to be used once,
deliberately, not left on routinely (see step7's docstring for the same reasoning).

Usage:
    python step5-plot-training.py
    python step5-plot-training.py --reveal      # also evaluates TEST -- use once
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import os

import torch
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader
from torchvision import transforms

from common import load_config, load_yaml, manifest_yaml

CSV_NAME = "training_log_4.csv"

NUMERIC_COLS = ("train_loss", "train_acc", "val_loss", "val_acc",
                "val_sens", "val_spec", "val_bal_acc", "val_auc",
                "val_acc_cdr05", "val_acc_cdr10", "val_acc_cdr20")

# CDR-positive grade -> (its log column, a human label)
GRADE_INFO = [
    (0.5, "val_acc_cdr05", "CDR 0.5 (very mild)"),
    (1.0, "val_acc_cdr10", "CDR 1 (mild)"),
    (2.0, "val_acc_cdr20", "CDR 2 (moderate)"),
]


def load_log(path):
    """Read the training_log CSV into a dict of column-name -> list of values.

    Tolerates older logs that lack some columns (e.g. val_auc) -- they stay empty.
    """
    cols = {"epoch": []}
    cols.update({k: [] for k in NUMERIC_COLS})
    with open(path) as f:
        reader = csv.DictReader(f)
        have = set(reader.fieldnames or [])
        for row in reader:
            cols["epoch"].append(int(row["epoch"]))
            for k in NUMERIC_COLS:
                if k in have:
                    cols[k].append(float(row[k]))
    return cols


def tail_mean(values, n=50):
    """Mean of the last n values (fewer if the run was shorter)."""
    tail = values[-n:]
    return sum(tail) / len(tail) if tail else float("nan")


SMOOTH_WINDOW = 20   # running-average window (epochs) for the validation-metric plots


def running_mean(values, window=SMOOTH_WINDOW):
    """Trailing running average over `window` epochs; same length as `values`."""
    out = []
    for i in range(len(values)):
        seg = values[max(0, i - window + 1): i + 1]
        out.append(sum(seg) / len(seg))
    return out


def grade_counts(manifest_path, split):
    """Per CDR-positive grade, how many patches / subjects a given split has."""
    counts = {g: {"patches": 0, "subjects": set()} for g, _, _ in GRADE_INFO}
    if os.path.isfile(manifest_path):
        for r in load_yaml(manifest_path).get("slices", []):
            if r.get("split") == split and r.get("cdr") in counts:
                counts[r["cdr"]]["patches"] += 1
                counts[r["cdr"]]["subjects"].add(r["subject"])
    return counts


def load_step4():
    """Import step4-train-network.py as a module (no architecture/Dataset/eval copy) --
    used by --reveal to build the test set and evaluate the trained checkpoint on it."""
    fname = "step4-train-network.py"
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), fname)
    if not os.path.isfile(path):
        raise SystemExit(f"No such script: {fname}")
    spec = importlib.util.spec_from_file_location("step4_train_network", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_trained_model_and_loader(config, outputs_path, split):
    """Load the step4 checkpoint and a DataLoader for ``split``. Shared by --reveal's
    test evaluation and the ROC-curve figure below -- both need a freshly-run trained
    model, unlike the rest of this script, which only reads training_log_4.csv."""
    step4 = load_step4()
    model_path = os.path.join(outputs_path, step4.MODEL_NAME)
    if not os.path.isfile(model_path):
        raise SystemExit(f"Checkpoint not found: {model_path}\n"
                         f"Run `python step4-train-network.py` first.")
    model = step4.Net()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    manifest = load_yaml(manifest_yaml(config))
    transform = transforms.ToTensor()
    ds = step4.OASISSlices(manifest, outputs_path, split, transform)
    loader = DataLoader(ds, batch_size=32)
    return step4, model, loader


def evaluate_test(config, outputs_path):
    """Load the trained checkpoint and evaluate it ONCE on test. Returns
    ``(loss, acc, sens, spec, auc, grade_acc)`` -- see step4's ``evaluate()``."""
    step4, model, test_loader = load_trained_model_and_loader(config, outputs_path, "test")
    # pos_weight only scales the reported LOSS, not the thresholded predictions this
    # summary actually headlines (acc/sens/spec/bal_acc/auc), so it's left at None here.
    return step4.evaluate(model, torch.device("cpu"), test_loader, pos_weight=None)


def predict_probs(config, outputs_path, split):
    """Raw (sigmoid(logit), true label) arrays, one pair per patch in ``split``, for the
    ROC-curve figure below. step4's own ``evaluate()`` computes exactly these internally
    but only returns the aggregate AUC, not the per-patch values a curve needs."""
    _, model, loader = load_trained_model_and_loader(config, outputs_path, split)
    probs, targets = [], []
    with torch.no_grad():
        for data, target, _ in loader:
            probs.append(torch.sigmoid(model(data)))
            targets.append(target)
    return torch.cat(probs).numpy(), torch.cat(targets).numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reveal", action="store_true",
                        help="also evaluate TEST (default: off -- see module docstring)")
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")                # write a file, no interactive window
    import matplotlib.pyplot as plt

    config = load_config()
    outputs_path = config["outputs_path"]

    log_path = os.path.join(outputs_path, CSV_NAME)
    if not os.path.isfile(log_path):
        raise SystemExit(f"No {CSV_NAME} found -- run step4-train-network.py first.")
    d = load_log(log_path)

    if args.reveal:
        print("*** --reveal is on: TEST will be evaluated below. Use this once. ***")
        te_loss, te_acc, te_sens, te_spec, te_auc, te_grade = evaluate_test(config, outputs_path)
        fig, axes = plt.subplots(2, 4, figsize=(19.5, 9))
        (ax_tl, ax_ta, ax_vl, ax_metrics), (ax_va, ax_bal, ax_auc, ax_grade) = axes
    else:
        fig, ((ax_tl, ax_ta, ax_vl), (ax_va, ax_bal, ax_auc)) = plt.subplots(2, 3, figsize=(15, 9))

    ax_tl.plot(d["epoch"], d["train_loss"], linewidth=1.4)
    ax_ta.plot(d["epoch"], d["train_acc"], linewidth=1.4)
    ax_vl.plot(d["epoch"], d["val_loss"], linewidth=1.4)

    # Validation accuracy, balanced accuracy, and AUC: faint raw + bold running mean.
    for ax, key in ((ax_va, "val_acc"), (ax_bal, "val_bal_acc"), (ax_auc, "val_auc")):
        ax.plot(d["epoch"], d[key], alpha=0.25, linewidth=1, color="steelblue")   # faint raw
        ax.plot(d["epoch"], running_mean(d[key]), color="steelblue",
                alpha=0.95, linewidth=1.9, label=f"{SMOOTH_WINDOW}-epoch running avg")

    ax_tl.set(xlabel="epoch", ylabel="training loss", title="Training loss")
    ax_ta.set(xlabel="epoch", ylabel="accuracy", title="Training accuracy", ylim=(0, 1.02))
    ax_vl.set(xlabel="epoch", ylabel="validation loss", title="Validation loss")
    for ax, ylabel, title in (
            (ax_va, "accuracy", "Validation accuracy"),
            (ax_bal, "balanced accuracy", "Validation balanced accuracy"),
            (ax_auc, "AUC", "Validation AUC")):
        ax.axhline(0.5, color="gray", linestyle="--", label="chance")
        ax.set(xlabel="epoch", ylabel=ylabel, title=title, ylim=(0, 1.02))
        ax.legend(fontsize=8)

    # Text summary: average validation metrics over the last N epochs (+ test, if
    # revealed). Every line is both printed to the console and collected (via emit) so
    # it can be saved to a file.
    n_last = config.get("avg_last_epochs", 50)
    summary_lines = []

    def emit(line=""):
        print(line)
        summary_lines.append(line)

    acc = tail_mean(d["val_acc"], n_last)
    sens = tail_mean(d["val_sens"], n_last)
    spec = tail_mean(d["val_spec"], n_last)
    bal = tail_mean(d["val_bal_acc"], n_last)
    auc = tail_mean(d["val_auc"], n_last)

    emit()
    emit(f"Average validation metrics over the last {n_last} epochs:")
    emit(f"  {'':<10} {'acc':>7} {'sens':>7} {'spec':>7} {'bal_acc':>8} {'auc':>7}")
    emit(f"  {'validate':<10} {acc:>7.3f} {sens:>7.3f} {spec:>7.3f} {bal:>8.3f} {auc:>7.3f}")
    if args.reveal:
        te_bal = (te_sens + te_spec) / 2
        emit(f"  {'test':<10} {te_acc:>7.3f} {te_sens:>7.3f} {te_spec:>7.3f} "
             f"{te_bal:>8.3f} {te_auc:>7.3f}")

        # Subpanel: validate (tail-mean) vs test (single point-in-time), headline metrics.
        names = ["acc", "sens", "spec", "bal_acc", "auc"]
        va_vals = [acc, sens, spec, bal, auc]
        te_vals = [te_acc, te_sens, te_spec, te_bal, te_auc]
        xm = range(len(names))
        w = 0.35
        ax_metrics.bar([i - w / 2 for i in xm], va_vals, w, label="validate", color="steelblue")
        ax_metrics.bar([i + w / 2 for i in xm], te_vals, w, label="test", color="firebrick")
        ax_metrics.axhline(0.5, color="gray", linestyle="--", linewidth=1)
        ax_metrics.set(title="Validate vs test (revealed)", ylim=(0, 1.02))
        ax_metrics.set_xticks(list(xm))
        ax_metrics.set_xticklabels(names, rotation=20, ha="right")
        ax_metrics.legend(fontsize=8)

    # Breakdown of validation (+ test, if revealed) accuracy by CDR grade (0.5/1/2,
    # pooled under label 1), mean of the last N epochs for validate, single point for
    # test. The counts are tiny, so read these as trends, not precise numbers -- more so
    # now that validate/test are drawn from a narrow, age-matched band
    # (cohort.age_eval in config.yaml).
    counts = grade_counts(manifest_yaml(config), "validate")
    emit()
    emit(f"Validation accuracy by CDR grade (mean of last {n_last} epochs):")
    va_grade_vals = []
    for g, col, name in GRADE_INFO:
        c = counts[g]
        m = tail_mean(d.get(col, []), n_last)
        va_grade_vals.append(m)
        cell = "n/a" if m != m else f"{m:.3f}"   # m != m -> nan
        emit(f"  {name:<20} {cell:>7}  ({c['patches']} patches from "
              f"{len(c['subjects'])} subject(s))")

    if args.reveal:
        test_counts = grade_counts(manifest_yaml(config), "test")
        emit()
        emit("Test accuracy by CDR grade (single evaluation):")
        te_grade_vals = []
        for g, col, name in GRADE_INFO:
            c = test_counts[g]
            m = te_grade.get(g, float("nan"))
            te_grade_vals.append(m)
            cell = "n/a" if m != m else f"{m:.3f}"
            emit(f"  {name:<20} {cell:>7}  ({c['patches']} patches from "
                  f"{len(c['subjects'])} subject(s))")

        # Subpanel: validate vs test, per-grade accuracy.
        grade_names = [name for _, _, name in GRADE_INFO]
        xg = range(len(grade_names))
        ax_grade.bar([i - w / 2 for i in xg], va_grade_vals, w, label="validate",
                    color="steelblue")
        ax_grade.bar([i + w / 2 for i in xg], te_grade_vals, w, label="test", color="firebrick")
        ax_grade.set(title="Accuracy by CDR grade: validate vs test", ylim=(0, 1.02))
        ax_grade.set_xticks(list(xg))
        ax_grade.set_xticklabels(grade_names, rotation=20, ha="right", fontsize=8)
        ax_grade.legend(fontsize=8)

    fig.tight_layout()
    out = os.path.join(outputs_path, "step5-training_comparison.png")
    fig.savefig(out, dpi=120)
    print(f"Saved training plot -> {out}")

    # --- ROC curve, as its own figure -- re-runs the trained checkpoint once on
    # validate (and test, if revealed) for the raw per-patch predictions a curve needs;
    # the "Validation AUC" panel above only has the per-epoch scalar from the CSV. The
    # AUC in this figure's legend is computed from THIS single final-checkpoint pass, so
    # it won't exactly match the running-tail-mean AUC printed below (that one averages
    # over the last n_last epochs of training, not one fixed evaluation).
    val_probs, val_targets = predict_probs(config, outputs_path, "validate")
    fig_roc, ax_roc = plt.subplots(figsize=(6, 6))
    fpr, tpr, _ = roc_curve(val_targets, val_probs)
    ax_roc.plot(fpr, tpr, color="steelblue", linewidth=2,
                label=f"validate (AUC={roc_auc_score(val_targets, val_probs):.3f})")
    if args.reveal:
        te_probs, te_targets = predict_probs(config, outputs_path, "test")
        fpr_te, tpr_te, _ = roc_curve(te_targets, te_probs)
        ax_roc.plot(fpr_te, tpr_te, color="firebrick", linewidth=2,
                    label=f"test (AUC={roc_auc_score(te_targets, te_probs):.3f})")
    ax_roc.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1, label="chance")
    ax_roc.set_xlabel("False positive rate")
    ax_roc.set_ylabel("True positive rate")
    ax_roc.set_title("ROC curve: trained CNN (model_4.pt)")
    ax_roc.legend(fontsize=9, loc="lower right")
    fig_roc.tight_layout()
    roc_path = os.path.join(outputs_path, "step5-roc_curve.png")
    fig_roc.savefig(roc_path, dpi=120)
    print(f"Saved ROC curve plot -> {roc_path}")

    # Save the same summary to a text file alongside the plot.
    summary_path = os.path.join(outputs_path, "step5-training_summary.txt")
    with open(summary_path, "w") as f:
        f.write("\n".join(summary_lines).lstrip("\n") + "\n")
    print(f"\nSaved summary -> {summary_path}")


if __name__ == '__main__':
    main()
