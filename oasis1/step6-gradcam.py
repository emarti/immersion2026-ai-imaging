#!/usr/bin/env python3
"""Step 6: Grad-CAM heatmaps -- see WHERE the network looks to call a patch "demented".

Grad-CAM (Gradient-weighted Class Activation Mapping) highlights the regions of an input
that most raise a chosen output score. We compute it for the **demented (label 1)**
direction -- i.e. the raw logit z -- so every heatmap answers the same question: "what in
this hippocampus patch pushes the model toward 'demented'?" The map is

    L = ReLU( sum_k alpha_k * A_k ),   alpha_k = mean over space of  dz/dA_k

where the ``A_k`` are the last conv feature maps (the tensor fed to global-average
pooling). Because we have a single output node, the label-0 map is
``ReLU(-sum_k alpha_k A_k)`` -- the *complementary* lobe (the ReLU makes it NOT a pure
sign flip of the label-1 map).

Grad-CAM explains the *target class you choose*, not the ground truth: a heatmap for a
truly-healthy patch still shows "what would make it look more like label 1".

Method: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via
Gradient-based Localization", ICCV 2017. Structurally inspired by (not copied from) the
original Torch implementation https://github.com/ramprs/grad-cam and the PyTorch
backward-hook approach discussed at
https://discuss.pytorch.org/t/grad-cam-implementation-in-pytorch-backward-on-model/3554/7

Needs a trained checkpoint from step4 (``outputs/model_4{design}.pt``); reads the same
``config.yaml`` / ``manifest.yaml`` as the rest of the pipeline.

Options live in ``config.yaml`` under ``gradcam:`` (which checkpoint, split, sample count,
seed, overlay opacity); any can be overridden on the command line.

Usage:
    python step6-gradcam.py                          # uses config.yaml (defaults to model_4a.pt)
    python step6-gradcam.py --model model_4c.pt --split test --n 16
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import random
import re

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from common import load_config, load_yaml, manifest_yaml


def load_design_net(design: str):
    """Import the ``Net`` class from the matching step4 script (no architecture copy)."""
    fname = f"step4{design}-train-network.py"
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), fname)
    if not os.path.isfile(path):
        raise SystemExit(f"No such design script: {fname}")
    spec = importlib.util.spec_from_file_location(f"step4{design}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Net


def grad_cam(model, x):
    """Grad-CAM map for the demented (label 1) logit of one patch.

    ``x`` is a 1x1xHxW tensor. Returns ``(cam, logit)`` where ``cam`` is an HxW float32
    array in [0,1]. Hooks ``model.gap`` -- present in every design -- to grab the feature
    map ``A`` that GAP pools and its gradient ``dz/dA``.
    """
    captured = {}

    def fwd_hook(module, inp, out):
        a = inp[0]
        a.retain_grad()                 # keep dz/dA around after backward
        captured["A"] = a

    handle = model.gap.register_forward_hook(fwd_hook)
    try:
        model.zero_grad(set_to_none=True)
        z = model(x).squeeze()          # scalar logit; target = the demented direction
        z.backward()
        A = captured["A"]               # 1 x C x h x w  (activations)
        alpha = A.grad.mean(dim=(2, 3), keepdim=True)   # 1 x C x 1 x 1  (channel weights)
        cam = F.relu((alpha * A).sum(dim=1))[0]         # h x w
    finally:
        handle.remove()

    cam = cam.detach()
    cam = cam - cam.min()
    peak = cam.max()
    if peak > 0:
        cam = cam / peak                                # normalize to [0,1]
    h, w = x.shape[-2:]
    cam = F.interpolate(cam[None, None], size=(h, w), mode="bilinear",
                        align_corners=False)[0, 0]      # upsample to the patch size
    return cam.cpu().numpy().astype(np.float32), float(z.detach())


def pick_rows(rows, n, seed):
    """Deterministically pick up to ``n`` rows, interleaving label 0 and label 1."""
    rng = random.Random(seed)
    pools = {0: [r for r in rows if int(r["label"]) == 0],
             1: [r for r in rows if int(r["label"]) == 1]}
    for pool in pools.values():
        rng.shuffle(pool)
    picked, i = [], 0
    while len(picked) < n and (pools[0] or pools[1]):
        cls = i % 2
        if pools[cls]:
            picked.append(pools[cls].pop())
        elif pools[1 - cls]:
            picked.append(pools[1 - cls].pop())
        i += 1
    return picked


def panel_title(row, prob):
    true = int(row["label"])
    return (f"{row['subject']} {row['side']}\n"
            f"true={'dem' if true else 'hlth'}  P(dem)={prob:.2f}")


def main():
    import matplotlib
    matplotlib.use("Agg")                        # write files, no interactive window
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description="Grad-CAM heatmaps for a trained OASIS design")
    parser.add_argument("--model", default=None,
                        help="checkpoint to explain, e.g. model_4c.pt (default: config gradcam.model)")
    parser.add_argument("--split", default=None, choices=["train", "validate", "test"],
                        help="split to sample patches from (default: config gradcam.split)")
    parser.add_argument("--n", type=int, default=None, help="number of patches (default: config)")
    parser.add_argument("--seed", type=int, default=None, help="sampling seed (default: config)")
    parser.add_argument("--alpha", type=float, default=None, help="overlay opacity (default: config)")
    args = parser.parse_args()

    config = load_config()
    outputs_path = config["outputs_path"]
    gc = config.get("gradcam") or {}

    # Resolve each option: CLI override -> config `gradcam.*` -> hardcoded fallback.
    model_name = args.model or gc.get("model", "model_4a.pt")
    split = args.split or gc.get("split", "validate")
    n = args.n if args.n is not None else int(gc.get("n_samples", 12))
    seed = args.seed if args.seed is not None else int(gc.get("seed", 0))
    alpha = args.alpha if args.alpha is not None else float(gc.get("overlay_alpha", 0.45))

    # Infer the architecture from the checkpoint name (model_4X.pt) so config names the
    # file explicitly, not a bare design letter.
    m = re.fullmatch(r"model_4([a-d])\.pt", os.path.basename(model_name))
    if not m:
        raise SystemExit(f"gradcam model must be named like 'model_4a.pt' (got '{model_name}').")
    design = m.group(1)

    model_path = os.path.join(outputs_path, model_name)
    if not os.path.isfile(model_path):
        raise SystemExit(f"Checkpoint not found: {model_path}\n"
                         f"Run `python step4{design}-train-network.py` first.")
    Net = load_design_net(design)
    model = Net()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()                                  # dropout off, BN in eval -> deterministic

    manifest = load_yaml(manifest_yaml(config))
    rows = [r for r in manifest["slices"] if r["split"] == split]
    if not rows:
        raise SystemExit(f"No patches for split '{split}' in the manifest.")
    rows = pick_rows(rows, n, seed)

    out_dir = os.path.join(outputs_path, "gradcam")
    os.makedirs(out_dir, exist_ok=True)

    panels = []
    for row in rows:
        gray = np.asarray(Image.open(os.path.join(outputs_path, row["png_path"])),
                          dtype=np.float32) / 255.0             # H x W in [0,1]
        x = torch.from_numpy(gray)[None, None]                  # 1 x 1 x H x W (matches ToTensor)
        cam, logit = grad_cam(model, x)
        prob = float(1.0 / (1.0 + np.exp(-logit)))              # sigmoid -> P(demented)
        title = panel_title(row, prob)

        fig, ax = plt.subplots(figsize=(2.4, 2.8))
        ax.imshow(gray, cmap="gray")
        ax.imshow(cam, cmap="jet", alpha=alpha)            # red = raises demented score
        ax.set_axis_off()
        ax.set_title(title, fontsize=8)
        fig.tight_layout()
        fname = f"gradcam_4{design}_{row['subject']}_{row['side']}_lbl{int(row['label'])}.png"
        fig.savefig(os.path.join(out_dir, fname), dpi=120, bbox_inches="tight")
        plt.close(fig)
        panels.append((gray, cam, title))

    # Montage of every panel.
    k = len(panels)
    cols = min(4, k)
    nrows = (k + cols - 1) // cols
    fig, axes = plt.subplots(nrows, cols, figsize=(3 * cols, 3.2 * nrows), squeeze=False)
    for ax in axes.flat:
        ax.set_axis_off()
    for (gray, cam, title), ax in zip(panels, axes.flat):
        ax.imshow(gray, cmap="gray")
        ax.imshow(cam, cmap="jet", alpha=alpha)
        ax.set_title(title, fontsize=8)
    fig.suptitle(f"Grad-CAM (design 4{design}) -- red = pushes toward 'demented'", fontsize=11)
    fig.tight_layout()
    grid_path = os.path.join(outputs_path, "gradcam_grid.png")
    fig.savefig(grid_path, dpi=120)
    plt.close(fig)

    print(f"Saved {k} Grad-CAM overlays -> {out_dir}/")
    print(f"Saved montage -> {grid_path}")


if __name__ == "__main__":
    main()
