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

Usage:
    python step3-generate-slices.py [--config config.yaml]
"""
from __future__ import annotations

import argparse
import os
import random

import numpy as np
import yaml

from common import SPLITS, load_config, load_yaml, manifest_yaml, split_dir, splits_yaml

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


def rel_to(path: str, root: str) -> str:
    """Path relative to ``root``, with forward slashes so the manifest is portable."""
    return os.path.relpath(path, root).replace(os.sep, "/")


def save_png(arr: np.ndarray, path: str) -> None:
    from PIL import Image
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(arr, mode="L").save(path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()

    config = load_config(args.config)
    splits = load_yaml(splits_yaml(config))

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
    rng = random.Random(config["cohort"]["seed"])   # reproducible shifts
    apply_random_shifts = hcfg.get("apply_random_shifts", False)
    random_shift = hcfg.get("random_shift", 0)
    n_shifts = hcfg.get("n_shifts", 1)
    all_shifts = [(dx, dy) for dx in range(-random_shift, random_shift + 1)
                           for dy in range(-random_shift, random_shift + 1)]

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
            cdr = member.get("cdr")           # AD grade (0.5/1/2) behind label 1
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
                        patch = sl[r0:r0 + bh, c0s:c0s + bw]   # hippocampus ROI patch
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
                            }
                        )

    summary = {
        split: {"healthy": counts[split][0], "demented": counts[split][1]}
        for split in SPLITS
    }
    with open(manifest_path, "w") as f:
        yaml.safe_dump({"summary": summary, "slices": rows}, f, sort_keys=False)

    print("\nSlices written per split (healthy / demented):")
    for split in SPLITS:
        c = counts[split]
        print(f"  {split:<8} healthy={c[0]:>4}  demented={c[1]:>4}")
    print(f"\nSubjects processed: {subjects}  (skipped: {skipped})")
    print(f"Total slices: {len(rows)}")
    if slice_shape is not None:
        h, w = slice_shape
        print(f"Slice image size: {w} x {h} px (W x H)")
    print(f"Manifest -> {manifest_path}")
    print(f"PNG tree -> {config['outputs_path']}/<split>/")


if __name__ == "__main__":
    main()
