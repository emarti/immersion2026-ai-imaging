#!/usr/bin/env python3
"""Step 6: Grad-CAM heatmaps -- see WHERE the network looks, in BOTH directions.

Grad-CAM (Gradient-weighted Class Activation Mapping) highlights the regions of an input
that most raise a chosen output score. From one backward pass we build a signed importance
map ``raw = sum_k alpha_k * A_k`` (alpha_k = mean over space of dz/dA_k; A_k = the last conv
feature maps fed to global-average pooling), and show BOTH of its lobes side by side:

    cdr_positive map = ReLU(+raw)   ('jet'):   what pushes this patch toward CDR-positive
    cdr_negative map = ReLU(-raw)   ('cool'):  what pushes it toward CDR-negative

(CDR-negative = CDR 0; CDR-positive = CDR 0.5 / 1 / 2, i.e. any impairment.) The two maps
are *complementary* -- disjoint lobes of the same signed map -- so they look quite
different; that is expected, not a bug. Grad-CAM explains a target you choose, not the ground
truth (a CDR-negative patch's CDR-positive map still shows "what would make it look CDR+").

Method: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via
Gradient-based Localization", ICCV 2017. Structurally inspired by (not copied from) the
original Torch implementation https://github.com/ramprs/grad-cam and the PyTorch
backward-hook approach discussed at
https://discuss.pytorch.org/t/grad-cam-implementation-in-pytorch-backward-on-model/3554/7

This step writes two views: (1) a montage ``step6-gradcam_grid.png`` (plus per-patch overlays in
``outputs/gradcam/``) over a sample of patches; and (2) per-subject **whole-slice context**
overlays in ``outputs/gradcam_context/`` -- the same heatmaps drawn back onto the full axial
slice at the left/right hippocampus crop boxes. The context pass reloads the raw volume, so
it needs the raw data (``DATA_RAW_PATH``).

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


def _load_step3():
    """Import step3's slice helpers (load_volume/normalize_volume/extract_slice) to reuse,
    so the reloaded whole-slice crops are bit-for-bit the validated PNGs."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "step3-generate-slices.py")
    spec = importlib.util.spec_from_file_location("step3_generate_slices", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _finish(cam2d, h, w):
    """A ReLU'd map -> min-max normalized [0,1] float array, upsampled to (h, w)."""
    cam = cam2d.detach()
    cam = cam - cam.min()
    peak = cam.max()
    if peak > 0:
        cam = cam / peak
    cam = F.interpolate(cam[None, None], size=(h, w), mode="bilinear",
                        align_corners=False)[0, 0]
    return cam.cpu().numpy().astype(np.float32)


def grad_cam(model, x):
    """Grad-CAM for BOTH directions of one patch, from a single backward pass.

    ``x`` is a 1x1xHxW tensor. Returns ``(cam_pos, cam_neg, logit)``: the CDR-positive map
    ``ReLU(+raw)`` and the CDR-negative map ``ReLU(-raw)`` where ``raw = sum_k alpha_k A_k``
    is the signed importance map. The two are complementary (disjoint lobes of one map). Hooks
    ``model.gap`` -- present in every design -- to grab the feature map ``A`` and ``dz/dA``.
    """
    captured = {}

    def fwd_hook(module, inp, out):
        a = inp[0]
        a.retain_grad()                 # keep dz/dA around after backward
        captured["A"] = a

    handle = model.gap.register_forward_hook(fwd_hook)
    try:
        model.zero_grad(set_to_none=True)
        z = model(x).squeeze()          # scalar logit z (P(CDR+) = sigmoid(z))
        z.backward()
        A = captured["A"]               # 1 x C x h x w  (activations)
        alpha = A.grad.mean(dim=(2, 3), keepdim=True)   # 1 x C x 1 x 1  (channel weights)
        raw = (alpha * A).sum(dim=1)[0]                 # h x w  (signed importance map)
    finally:
        handle.remove()

    h, w = x.shape[-2:]
    cam_pos = _finish(F.relu(raw), h, w)                # pushes toward CDR-positive  (ReLU(+raw))
    cam_neg = _finish(F.relu(-raw), h, w)               # pushes toward CDR-negative  (ReLU(-raw))
    return cam_pos, cam_neg, float(z.detach())


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
            f"true={POS_SHORT if true else NEG_SHORT}  P(CDR+)={prob:.2f}")


def sigmoid(z):
    return float(1.0 / (1.0 + np.exp(-z)))


def hippocampus_boxes(a0, a1, l0, l1, width):
    """(row0, row1, col0, col1) crop boxes for the left and right hippocampus.

    Matches step3: the left box is ``lr_left``; the right box is it mirrored about the L-R
    centre of a slice ``width`` wide.
    """
    return {"L": (a0, a1, l0, l1), "R": (a0, a1, width - l1, width - l0)}


# Heatmap colormaps: the classic 'jet' for the CDR-positive direction (as before), and a
# distinct 'cool' (cyan->magenta) for CDR-negative, so the two direction panels are easy to
# tell apart. (The panel TITLE colour, separately, encodes the truth: red CDR+ / blue CDR-.)
CMAP_CDR_POS = "jet"
CMAP_CDR_NEG = "cool"

# Compact class names for cramped plot text (full names come from config `labels:`).
POS_SHORT = "CDR+"
NEG_SHORT = "CDR-"


def truth_color(true):
    """Colour for a true label: red = CDR-positive (1), blue = CDR-negative (0)."""
    return "red" if int(true) else "blue"


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
    labels = config.get("labels") or {}
    name_neg = labels.get("cdr_negative", "CDR Negative")   # label 0 display name
    name_pos = labels.get("cdr_positive", "CDR Positive")   # label 1 display name

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
    split_rows = [r for r in manifest["slices"] if r["split"] == split]
    if not split_rows:
        raise SystemExit(f"No patches for split '{split}' in the manifest.")
    rows = pick_rows(split_rows, n, seed)

    out_dir = os.path.join(outputs_path, "step6-gradcam")
    os.makedirs(out_dir, exist_ok=True)

    panels = []                              # (gray, cam_neg, cam_pos, title, colour)
    for row in rows:
        gray = np.asarray(Image.open(os.path.join(outputs_path, row["png_path"])),
                          dtype=np.float32) / 255.0             # H x W in [0,1]
        x = torch.from_numpy(gray)[None, None]                  # 1 x 1 x H x W (matches ToTensor)
        cam_pos, cam_neg, logit = grad_cam(model, x)
        title = panel_title(row, sigmoid(logit))
        colour = truth_color(int(row["label"]))

        # Two panels: left = pushes toward CDR- (blue), right = pushes toward CDR+ (red).
        fig, axs = plt.subplots(1, 2, figsize=(4.6, 2.9))
        for ax, cam, cmap, name in ((axs[0], cam_neg, CMAP_CDR_NEG, f"push -> {NEG_SHORT}"),
                                    (axs[1], cam_pos, CMAP_CDR_POS, f"push -> {POS_SHORT}")):
            ax.imshow(gray, cmap="gray")
            ax.imshow(cam, cmap=cmap, alpha=alpha, vmin=0.0, vmax=1.0)
            ax.set_axis_off()
            ax.set_title(name, fontsize=8)
        fig.suptitle(title, fontsize=8, color=colour)
        fig.tight_layout()
        fname = f"gradcam_4{design}_{row['subject']}_{row['side']}_lbl{int(row['label'])}.png"
        fig.savefig(os.path.join(out_dir, fname), dpi=120, bbox_inches="tight")
        plt.close(fig)
        panels.append((gray, cam_neg, cam_pos, title, colour))

    # Montage: one row per patch -- col 0 = push->CDR- (blue), col 1 = push->CDR+ (red).
    k = len(panels)
    fig, axes = plt.subplots(k, 2, figsize=(5.2, 2.6 * k), squeeze=False)
    for ax in axes.flat:
        ax.set_axis_off()
    for (gray, cam_neg, cam_pos, title, colour), (ax_h, ax_d) in zip(panels, axes):
        for ax, cam, cmap in ((ax_h, cam_neg, CMAP_CDR_NEG), (ax_d, cam_pos, CMAP_CDR_POS)):
            ax.imshow(gray, cmap="gray")
            ax.imshow(cam, cmap=cmap, alpha=alpha, vmin=0.0, vmax=1.0)
        ax_h.set_title(title, fontsize=7, color=colour, loc="left")
    fig.suptitle(f"Grad-CAM (design 4{design})    "
                 f"left/blue = pushes toward {name_neg}    ·    right/red = pushes toward {name_pos}",
                 fontsize=10)
    fig.tight_layout()
    grid_path = os.path.join(outputs_path, "step6-gradcam_grid.png")
    fig.savefig(grid_path, dpi=120)
    plt.close(fig)

    print(f"Saved {k} Grad-CAM overlays (both directions) -> {out_dir}/")
    print(f"Saved montage -> {grid_path}")

    # --- Whole-slice context: draw each subject's L/R heatmaps back onto the full axial
    # slice, reloaded from the raw volume (identical to the validated crops). ---
    from matplotlib.patches import Rectangle
    step3 = _load_step3()
    axis = config["slices"]["slice_axis"]
    hcfg = config["hippocampus"]
    a0, a1 = hcfg["ap"]
    l0, l1 = hcfg["lr_left"]
    data_raw = config["data_raw_path"]

    ctx_dir = os.path.join(outputs_path, "step6-gradcam_context")
    os.makedirs(ctx_dir, exist_ok=True)

    subjects = {}                       # one entry per subject (L/R share img/slice/label)
    for r in split_rows:
        subjects.setdefault(r["subject"], r)
    subjects = list(subjects.values())
    print(f"Whole-slice context: {len(subjects)} '{split}' subject(s) -> {ctx_dir}/")

    ctx_saved = 0
    for i, r in enumerate(subjects, 1):
        subject = r["subject"]
        true = int(r["label"])
        idx = int(r["slice_index"])
        img_abs = os.path.join(data_raw, r["img_path"])
        if not os.path.isfile(img_abs):
            print(f"  [skip] missing volume for {subject}: {img_abs}")
            continue
        sl = step3.extract_slice(step3.normalize_volume(step3.load_volume(img_abs)), axis, idx)
        H, W = sl.shape
        boxes = hippocampus_boxes(a0, a1, l0, l1, W)

        # Per side, compute both direction maps.
        pos, neg, probs = {}, {}, {}
        for side, (r0, r1, c0, c1) in boxes.items():
            crop = sl[r0:r1, c0:c1].astype(np.float32) / 255.0        # == the validated PNG
            cam_pos, cam_neg, logit = grad_cam(model, torch.from_numpy(crop)[None, None])
            pos[side], neg[side] = cam_pos, cam_neg
            probs[side] = sigmoid(logit)

        # Two panels: left = pushes toward CDR- (blue), right = pushes toward CDR+ (red).
        fig, axs = plt.subplots(1, 2, figsize=(9.2, 5.4))
        for ax, cams, cmap, name in ((axs[0], neg, CMAP_CDR_NEG, f"push -> {NEG_SHORT}"),
                                     (axs[1], pos, CMAP_CDR_POS, f"push -> {POS_SHORT}")):
            ax.imshow(sl, cmap="gray", vmin=0, vmax=255)
            for side, (r0, r1, c0, c1) in boxes.items():
                # origin='upper' default: extent = (left, right, bottom, top) = (c0, c1, r1, r0)
                ax.imshow(cams[side], cmap=cmap, alpha=alpha, vmin=0.0, vmax=1.0,
                          extent=(c0, c1, r1, r0))
                ax.add_patch(Rectangle((c0, r0), c1 - c0, r1 - r0, edgecolor="lime",
                                       facecolor="none", linewidth=1.0))
            ax.set_xlim(0, W)
            ax.set_ylim(H, 0)
            ax.set_axis_off()
            ax.set_title(name, fontsize=10)
        fig.suptitle(f"{subject}   true = {name_pos if true else name_neg}   "
                     f"P(CDR+)  L = {probs['L']:.2f}   R = {probs['R']:.2f}",
                     fontsize=11, color=truth_color(true))
        fig.tight_layout()
        out = os.path.join(ctx_dir, f"gradcam_ctx_4{design}_{subject}.png")
        fig.savefig(out, dpi=130, bbox_inches="tight")
        plt.close(fig)
        ctx_saved += 1
        print(f"  [{i}/{len(subjects)}] {subject}  P(CDR+) L={probs['L']:.2f} R={probs['R']:.2f}")

    print(f"Saved {ctx_saved} context overlay(s) -> {ctx_dir}/")


if __name__ == "__main__":
    main()
