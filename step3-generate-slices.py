#!/usr/bin/env python3
"""Step 3: render 2D transverse (axial) PNG slices for every subject in the splits.

Reads ``splits.yaml`` (from step2), loads each subject's brain-masked atlas
volume with nibabel, extracts transverse slices along the I-S axis at the
configured offsets, then (following the DL4MI tutorial) crops a LEFT and a RIGHT
hippocampus patch from each slice, normalizes to 8-bit grayscale, and writes them
as PNGs under ``outputs/<split>/``. A ``manifest.yaml`` lists every patch (with its
output- and raw-relative paths and its ``side``) for the later PyTorch Dataset.

Training subjects yield ``2 x len(offsets_mm) x n_shifts`` patches (left + right
hippocampus per plane, each duplicated with n_shifts random in-plane shifts, per the
tutorial's CropLeftHC augmentation); validation/test subjects yield just ``2`` (left
+ right of the single ``eval_offset_mm`` plane, unshifted) for a deterministic
evaluation. Patches from one subject always live in the same split, so there is no
train/test leakage.

When ``hippocampus.apply_random_reflections_rotations`` is on, each of those TRAIN
patches is ALSO independently reflected left-right and/or top-down (each a fixed 25%
chance, hardcoded as ``REFLECT_LR_PROB``/``REFLECT_UD_PROB`` below) and rotated by a
random multiple of 90 degrees -- a second, independent augmentation layered on top of
the random shifts, never resizing or interpolating (see ``rotate_reflect_patch``).
Validation/test are never reflected or rotated, same as they're never shifted. Every
train row in the manifest records what was drawn (``rotation_deg``, ``reflect_lr``,
``reflect_ud``), 0/False/False on validation/test rows.

For a small sample of subjects per split (``slices.context_samples``), step3 also writes
**context** images to ``outputs/slice_context/<split>/``: the full axial slice with a
rectangle around every crop window actually taken from it (train shows all random-shift
boxes per plane; validation/test show the single box), so the ROI size is visible in situ.

Also copies the config actually used to ``outputs/config.yaml``, so a run's PNGs and
manifest always carry a record of the settings that produced them, independent of
whatever config.yaml (or ``--config`` override) is currently on disk.

Usage:
    python step3-generate-slices.py [--config config.yaml]
"""
from __future__ import annotations

import argparse
import os
import random
import shutil

import numpy as np
import yaml

from common import REPO_ROOT, SPLITS, load_config, load_yaml, manifest_yaml, split_dir, splits_yaml

try:
    import nibabel as nib
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "nibabel is required. Install with: pip install -r requirements.txt"
    ) from exc


def load_volume(path: str) -> np.ndarray:
    """Load an Analyze/NIfTI volume as a 3D array.

    OASIS .img files are stored with a trailing singleton 4th dimension
    ((176, 208, 176, 1)), which numpy.squeeze drops.
    """
    vol = np.squeeze(np.asarray(nib.load(path).dataobj))
    if vol.ndim != 3:
        raise ValueError(f"expected a 3D volume, got shape {vol.shape}")
    return vol


def normalize_volume(vol: np.ndarray) -> np.ndarray:
    """Scale a volume to 0-255 uint8 using robust percentiles of brain voxels.

    Background (exact 0 after brain masking) is preserved as 0.
    """
    vol = vol.astype(np.float32)
    brain = vol[vol > 0]
    if brain.size == 0:
        return np.zeros_like(vol, dtype=np.uint8)
    lo, hi = np.percentile(brain, [1.0, 99.0])
    if hi <= lo:
        hi = lo + 1.0
    scaled = np.clip((vol - lo) / (hi - lo), 0.0, 1.0) * 255.0
    scaled[vol <= 0] = 0.0
    return scaled.astype(np.uint8)


def extract_slice(vol: np.ndarray, axis: int, index: int) -> np.ndarray:
    """Take a 2D slice along ``axis`` and orient it upright.

    For the transverse (I-S) axis this puts anterior at the top, patient-left to
    the right; the same transpose+flip keeps other axes upright too.
    """
    plane = np.take(vol, index, axis=axis)
    return np.flipud(plane.T)


REFLECT_LR_PROB = 0.25   # independent chance of a training patch being flipped left-right
REFLECT_UD_PROB = 0.25   # independent chance of a training patch being flipped top-to-bottom


def rotate_reflect_patch(sl: np.ndarray, r0: int, c0: int, bh: int, bw: int,
                         rng: random.Random):
    """Return a (bh, bw) patch centred on the same window as ``sl[r0:r0+bh, c0:c0+bw]``,
    but rotated by a random multiple of 90 degrees and, independently, possibly flipped
    left-right (probability ``REFLECT_LR_PROB``) and/or top-down (``REFLECT_UD_PROB``).
    Every output is exactly (bh, bw) regardless of which rotation is drawn, and no pixel
    is ever resized or interpolated: a square big enough to contain the (bh, bw) box
    after ANY 90-degree rotation is cropped first, rotated losslessly with ``np.rot90``,
    then centre-cropped back down to (bh, bw).

    Returns ``(patch, rotation_deg, did_lr, did_ud)`` -- the draws are reported back so
    the caller can record them in the manifest.
    """
    H, W = sl.shape
    S = max(bh, bw)
    center_r, center_c = r0 + bh // 2, c0 + bw // 2
    r0_sq = min(max(center_r - S // 2, 0), H - S)
    c0_sq = min(max(center_c - S // 2, 0), W - S)
    square = sl[r0_sq:r0_sq + S, c0_sq:c0_sq + S]

    k = rng.randrange(4)                        # 0/1/2/3 -> 0/90/180/270 degrees
    rotated = np.rot90(square, k)
    rh0, rw0 = (S - bh) // 2, (S - bw) // 2
    patch = rotated[rh0:rh0 + bh, rw0:rw0 + bw]

    did_lr = rng.random() < REFLECT_LR_PROB
    if did_lr:
        patch = np.fliplr(patch)
    did_ud = rng.random() < REFLECT_UD_PROB
    if did_ud:
        patch = np.flipud(patch)

    # rot90/fliplr/flipud all return views -- make the result a contiguous array
    # before it's handed to PIL for saving.
    return np.ascontiguousarray(patch), k * 90, did_lr, did_ud


def rel_to(path: str, root: str) -> str:
    """Path relative to ``root``, with forward slashes so the manifest is portable."""
    return os.path.relpath(path, root).replace(os.sep, "/")


def save_png(arr: np.ndarray, path: str) -> None:
    from PIL import Image
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(arr, mode="L").save(path)


def pick_context_subjects(members: list[dict], n: int, seed: int) -> set:
    """Reproducible, label-interleaved sample of up to ``n`` subject ids from a split.

    Interleaving by label keeps both classes represented in the small context sample.
    """
    if n <= 0 or not members:
        return set()
    rng = random.Random(seed)
    pools = {0: [m for m in members if int(m["label"]) == 0],
             1: [m for m in members if int(m["label"]) == 1]}
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
    return {m["subject"] for m in picked}


# Edge colours for the two hippocampus crop boxes drawn on the context images.
CONTEXT_COLOURS = {"L": "lime", "R": "deepskyblue"}


def save_context_image(sl: np.ndarray, windows: dict, bh: int, bw: int,
                       out_path: str, title: str) -> None:
    """Draw the full axial slice with a rectangle around every crop window.

    ``windows`` maps side ('L'/'R') -> list of (row0, col0) top-left corners; each box is
    ``bh`` tall and ``bw`` wide (the fixed patch size). Multiple boxes (train random shifts)
    are drawn thin and semi-transparent so overlap stays readable; a single box (val/test)
    is drawn solid.
    """
    import matplotlib
    matplotlib.use("Agg")                        # write files, no interactive window
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    H, W = sl.shape
    fig, ax = plt.subplots(figsize=(5.0, 5.0 * H / max(W, 1)))
    ax.imshow(sl, cmap="gray", vmin=0, vmax=255)
    for side, corners in windows.items():
        if not corners:
            continue
        colour = CONTEXT_COLOURS.get(side, "yellow")
        many = len(corners) > 1
        lw = 0.8 if many else 1.6
        alpha = 0.55 if many else 1.0
        for j, (r0, c0s) in enumerate(corners):
            ax.add_patch(Rectangle((c0s, r0), bw, bh, edgecolor=colour, facecolor="none",
                                   linewidth=lw, alpha=alpha,
                                   label=(f"{side} hippocampus" if j == 0 else None)))
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.set_axis_off()
    ax.set_title(title, fontsize=9)
    ax.legend(loc="lower right", fontsize=7, framealpha=0.6)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()

    config = load_config(args.config)
    splits = load_yaml(splits_yaml(config))

    # Copy the config actually used into outputs/, so a run's PNGs/manifest always carry
    # a record of the settings that produced them, even if config.yaml is edited later.
    config_src = args.config if os.path.isabs(args.config) else os.path.join(REPO_ROOT, args.config)
    os.makedirs(config["outputs_path"], exist_ok=True)
    shutil.copy2(config_src, os.path.join(config["outputs_path"], "config.yaml"))

    scfg = config["slices"]
    axis = scfg["slice_axis"]
    middle = scfg["middle_index"]
    offsets = scfg["offsets_mm"]          # 1 mm iso -> voxel offsets
    # Training uses every hippocampus plane (more data); validation/test use only
    # the single `eval_offset_mm` plane, for a deterministic single-slice evaluation.
    # (Later the training offsets become randomized; val/test stay on eval_offset_mm.)
    train_indices = [(off, middle + off) for off in offsets]
    eval_off = scfg["eval_offset_mm"]
    eval_indices = [(eval_off, middle + eval_off)]

    hcfg = config["hippocampus"]          # left/right hippocampus in-plane crop
    ap0, ap1 = hcfg["ap"]
    lc0, lc1 = hcfg["lr_left"]

    # Training-only random-shift augmentation: each train patch is cropped at
    # n_shifts distinct random (dx, dy) offsets; val/test use the unshifted box.
    rng = random.Random(config["cohort"]["seed"])   # reproducible shifts (and, below,
                                                     # reproducible reflections/rotations)
    apply_random_shifts = hcfg.get("apply_random_shifts", False)
    random_shift = hcfg.get("random_shift", 0)
    n_shifts = hcfg.get("n_shifts", 1)
    all_shifts = [(dx, dy) for dx in range(-random_shift, random_shift + 1)
                           for dy in range(-random_shift, random_shift + 1)]

    # Training-only reflection/rotation augmentation -- independent of the shift
    # augmentation above; see config.yaml's comment for the anatomical-plausibility
    # caveat on the top-down flip and the 90-degree rotations. Probabilities are
    # hardcoded (REFLECT_LR_PROB/REFLECT_UD_PROB above), not configurable.
    apply_random_reflections_rotations = hcfg.get("apply_random_reflections_rotations", False)

    # Context images: sample a few subjects per split whose full axial slices we redraw with
    # rectangles around the actual crop windows (train: every random-shift box per plane;
    # val/test: the single box). Sampled reproducibly, with a distinct seed per split.
    seed = config["cohort"]["seed"]
    context_samples = int(scfg.get("context_samples", 3))
    context_subjects = {
        split: pick_context_subjects(splits.get(split, []) or [], context_samples, seed + i)
        for i, split in enumerate(SPLITS)
    }
    context_dir = os.path.join(config["outputs_path"], "slice_context")
    ctx_count = 0

    manifest_path = manifest_yaml(config)
    os.makedirs(os.path.dirname(manifest_path) or ".", exist_ok=True)

    counts = {s: {0: 0, 1: 0} for s in SPLITS}
    rows: list[dict] = []
    subjects = 0
    skipped = 0
    slice_shape = None            # (rows, cols) of the 2D slices; same for all

    for split in SPLITS:
        indices = train_indices if split == "train" else eval_indices
        for member in splits.get(split, []) or []:
            subject = member["subject"]
            label = int(member["label"])
            cdr = member.get("cdr")           # CDR grade (0.5/1/2) behind label 1
            img_path = member["img_path"]
            if not img_path or not os.path.isfile(img_path):
                print(f"  [skip] missing image for {subject}: {img_path}")
                skipped += 1
                continue
            try:
                vol = load_volume(img_path)
            except Exception as e:  # noqa: BLE001
                print(f"  [skip] unreadable {subject}: {e}")
                skipped += 1
                continue

            vol = normalize_volume(vol)
            subjects += 1
            # Source volume path, relative to DATA_RAW_PATH (portable across machines).
            img_rel = rel_to(img_path, config["data_raw_path"])
            for off, idx in indices:
                if not (0 <= idx < vol.shape[axis]):
                    print(f"  [warn] {subject}: index {idx} out of range on axis {axis}")
                    continue
                sl = extract_slice(vol, axis, idx)             # (A-P rows, L-R cols)
                H, W = sl.shape                                # W = L-R width, for mirroring
                bh, bw = ap1 - ap0, lc1 - lc0                  # fixed patch size
                # Right box = left box mirrored about the L-R centre.
                sides = {"L": (lc0, lc1), "R": (W - lc1, W - lc0)}
                # Train with apply_random_shifts on: n_shifts distinct random shifts.
                # Otherwise (off, or val/test): the single unshifted box.
                if split == "train" and apply_random_shifts and random_shift > 0:
                    shifts = rng.sample(all_shifts, min(n_shifts, len(all_shifts)))
                else:
                    shifts = [(0, 0)]
                is_ctx = subject in context_subjects[split]
                windows = {"L": [], "R": []}   # crop corners for this plane's context image
                for side, (c0, c1) in sides.items():
                    seen_positions = set()   # distinct clamped windows -> unique patches
                    a = 0
                    for dx, dy in shifts:
                        # Shift the crop window, clamping so it keeps its size in-bounds.
                        r0 = min(max(ap0 + dy, 0), H - bh)
                        c0s = min(max(c0 + dx, 0), W - bw)
                        # The (dx, dy) offsets are already unique, but near an edge two of
                        # them can clamp to the SAME window; skip the repeat so every saved
                        # slice for this side/plane is genuinely distinct.
                        if (r0, c0s) in seen_positions:
                            continue
                        seen_positions.add((r0, c0s))
                        if is_ctx:
                            # Position only -- faithful to the actual crop window even
                            # when the saved patch below is rotated/reflected for train.
                            windows[side].append((r0, c0s))
                        if split == "train" and apply_random_reflections_rotations:
                            patch, rotation_deg, reflect_lr, reflect_ud = rotate_reflect_patch(
                                sl, r0, c0s, bh, bw, rng)
                        else:
                            patch = sl[r0:r0 + bh, c0s:c0s + bw]   # hippocampus ROI patch
                            rotation_deg, reflect_lr, reflect_ud = 0, False, False
                        slice_shape = patch.shape
                        # Flat per-split folder; label + side + copy index in the name.
                        fname = f"{subject}_lbl{label}_ax{idx:03d}_{side}_a{a:02d}.png"
                        out_path = os.path.join(split_dir(config, split), fname)
                        save_png(patch, out_path)
                        counts[split][label] += 1
                        a += 1
                        rows.append(
                            {
                                # PNG path relative to outputs_path; img_path relative to
                                # DATA_RAW_PATH. Both stay valid if either root moves.
                                "png_path": rel_to(out_path, config["outputs_path"]),
                                "img_path": img_rel,
                                "subject": subject,
                                "split": split,
                                "label": label,
                                "cdr": cdr,
                                "slice_index": idx,
                                "offset_mm": off,
                                "side": side,
                                "shift_x": dx,
                                "shift_y": dy,
                                "rotation_deg": rotation_deg,
                                "reflect_lr": reflect_lr,
                                "reflect_ud": reflect_ud,
                            }
                        )

                if is_ctx:
                    out_ctx = os.path.join(context_dir, split,
                                           f"ctx_{subject}_lbl{label}_ax{idx:03d}.png")
                    nL, nR = len(windows["L"]), len(windows["R"])
                    ttl = (f"{subject}   {split}   ax{idx} (offset {off:+d} mm)   "
                           f"L/R boxes: {nL}/{nR}")
                    save_context_image(sl, windows, bh, bw, out_ctx, ttl)
                    ctx_count += 1

    summary = {
        split: {"cdr_negative": counts[split][0], "cdr_positive": counts[split][1]}
        for split in SPLITS
    }
    with open(manifest_path, "w") as f:
        yaml.safe_dump({"summary": summary, "slices": rows}, f, sort_keys=False)

    print("\nSlices written per split (CDR- / CDR+):")
    for split in SPLITS:
        c = counts[split]
        print(f"  {split:<8} CDR-={c[0]:>4}  CDR+={c[1]:>4}")
    print(f"\nSubjects processed: {subjects}  (skipped: {skipped})")
    print(f"Total slices: {len(rows)}")
    if slice_shape is not None:
        h, w = slice_shape
        print(f"Slice image size: {w} x {h} px (W x H)")
    print(f"Manifest -> {manifest_path}")
    print(f"PNG tree -> {config['outputs_path']}/<split>/")
    if ctx_count:
        print(f"Context images -> {context_dir}/<split>/ ({ctx_count} images)")


if __name__ == "__main__":
    main()
