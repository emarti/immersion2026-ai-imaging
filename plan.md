# Project status & decisions

Small CNN that classifies OASIS-1 **transverse hippocampus patches** as CDR-positive (1)
vs CDR-negative (0), built on PyTorch's official MNIST example. This file is the terse
status/decisions tracker; see **readme.md** (how to run + config reference) and
**introduction.md** (the theory and the *why*) for detail.

## Progress

The pipeline (steps 1–6) is built and has been run end to end. Step 7 exists as a
standalone extra (not wired into `process.sh`, `config.yaml`, or the main docs).

- [x] **Step 0 — config**: `config.yaml`, `requirements.txt`, `common.py`, `.env`/`.env.example`.
- [x] **Step 1 — gather metadata**: `step1-gather-metadata.py` → `outputs/metadata.csv`
  (parses each session `.txt`, derives the label, cross-checks the OASIS spreadsheet).
- [x] **Step 2 — cohort + splits**: `step2-assign-groups.py` → `outputs/splits.yaml`.
  Cohort age 62–79, `balance: label` (~110 subjects, ~76/16/18 split); subject-level
  stratified splits (no leakage). Three balance modes: `strict | label | none`.
- [x] **Step 3 — hippocampus patches**: `step3-generate-slices.py` →
  `outputs/<split>/*.png` + `outputs/manifest.yaml`. Left **and** right hippocampus
  crops (separate samples). Training uses all `offsets_mm` planes (× L/R × optional
  random shifts); validation/test use the single `eval_offset_mm` plane, unshifted.
  Also writes **context** images (`outputs/slice_context/<split>/`) for a sample of
  subjects (`slices.context_samples`): the full axial slice with a rectangle around each
  crop box (L=lime, R=deepskyblue; train = every shift box per plane, val/test = single).
- [x] **Step 4 — 4-design sweep**: `step4a…4d-train-network.py`. MNIST-style CNN +
  `OASISSlices` Dataset over the manifest; `BCEWithLogitsLoss`, `AdamW` (lr **1e-4**,
  wd 1e-4), batch 32, GAP head. Each logs per-epoch train + validation
  loss/acc/**sens/spec/balanced-acc** and **per-CDR-grade val accuracy** (0.5/1/2, the
  CDR-positive severities pooled under label 1) (+ run time, s/epoch) to
  `outputs/training_log_4{a,b,c,d}.csv` (11 columns). `cdr` flows subject→
  `splits.yaml`→`manifest.yaml`→Dataset.
- [x] **Step 5 — compare**: `step5-plot-training.py` → `outputs/training_comparison.png`
  (2×2: train loss + train accuracy + validation accuracy + validation balanced-accuracy,
  all designs overlaid; also prints each
  design's mean validation acc/sens/spec/balanced-acc over the last 50 epochs, plus a
  by-AD-grade validation accuracy breakdown with per-grade subject counts).
- [x] **Step 6 — interpretability**: `step6-gradcam.py` → Grad-CAM heatmaps in
  `outputs/gradcam/` (+ `gradcam_grid.png`) plus per-subject whole-slice context overlays
  in `outputs/gradcam_context/` (heatmaps redrawn on the full axial slice). Loads a design's
  checkpoint `outputs/model_4{a,b,c,d}.pt` (now saved by step4), hooks `Net.gap`, and shows
  **both** push directions side by side (CDR-positive in the `jet` colormap, CDR-negative in a
  distinct `cool`), with truth-coloured titles (red CDR-positive / blue CDR-negative). Cites
  Selvaraju et al. (ICCV 2017) / ramprs/grad-cam.
- [x] **Step 7 — interpretability (forward), STANDALONE**: `step7-perturb.py` →
  minimal-perturbation saliency in `outputs/perturb/` (+ `perturb_grid.png`). Gradient-free
  greedy L1-budget counterfactual; reuses the step6 checkpoint-by-name convention. Kept as an
  optional extra — **not** run by `process.sh`, and deliberately left out of `config.yaml` and
  the reader-facing docs (readme/introduction) as of the "up to step 6" commit.
- Also: `process.sh` runs steps 1–6 (`bash process.sh`; `epochs` + `avg_last_epochs`
  live in `config.yaml`).

## The four designs (teaching sweep — sample three directions from a baseline)

A shared **baseline** plus three single-change variants, each isolating one knob (dropout,
width, depth): 4a 8→16→32 @0.6/0.2 (baseline) · 4b 8→16→32, **no dropout** (overfit demo) ·
4c 16→32→64 @0.6/0.2 (**wider**) · 4d 8→16 @0.6/0.2 (**shallower**, 2 blocks). Identical
except the `Net`. (Earlier
optimization sweeps are logged in `lab-notes.md`.)

## Design decisions (current)

- **Label** = CDR: 0 → CDR-negative, 0.5/1/2 → CDR-positive, blank → excluded.
- **Volume** = `PROCESSED/MPRAGE/T88_111/*_t88_masked_gfc.img` (atlas-registered,
  gain-field-corrected, brain-masked); read with nibabel; globbed for the `n3`/`n4` infix.
- **Slices**: transverse (I–S) `slice_axis: 2`, `middle_index: 88`; `offsets_mm` at the
  hippocampus band (training planes), `eval_offset_mm: -24` (single val/test plane).
- **Hippocampus crop**: left/right in-plane boxes (`ap`, `lr_left`), right = mirror.
  Optional training-only random-shift augmentation (`apply_random_shifts`, default off).
- **Balancing**: `strict` (sex×label) / `label` (CDR-negative=CDR-positive, sex free, default) /
  `none`; subject-level stratified splits.
- **Training**: `AdamW` lr 1e-4, `BCEWithLogitsLoss`, dropout + weight decay, GAP head.
- **Artifacts**: everything under `outputs_path` (gitignored); PNGs flat per split,
  `{subject}_lbl{label}_ax{idx}_{L|R}_a{copy}.png`; index of record = `manifest.yaml`.

## Findings

- On this small dataset the bigger designs (4b/4c) overfit — train accuracy → ~100 %,
  validation near chance — while the smallest (4d) underfits. Model size is not the
  bottleneck; **data is**. Correlated slices ⇒ effective sample size ≈ #subjects.
- Direction that helped: hippocampus ROI focus, more subjects (`label` balancing),
  lower learning rate (1e-4). Realistic target ≈ **0.78 balanced accuracy** — the DL4MI
  tutorial's hippocampus CNN in ~30 epochs (≈0.81 with left+right voting).
- Best observed so far: **4c ≈ 0.777** balanced acc (lab-notes Run 3, reproduced in Run 5).
  Results are noisy (~±0.1 on a ~16-subject val set) and sensitive to crop/random-shift
  choices — real risk of validation overfitting from repeated tweaks. See `lab-notes.md`.
- **Age confound**: a low `age_min` lets the model exploit age rather than pathology.
  Raising `age_min` to 70 age-matches the cohort but shrinks it — accuracy collapses toward
  chance (Run 7 ≈ 0.53–0.60). The current `age_min: 60` is more accurate but biased; this is
  a documented, intentional trade-off.
- **Augmentation (`apply_random_shifts`)**: measured — greatly increases training time and
  favors smaller models, but does **not** appreciably improve validation accuracy here.

## TODO / next

- Add early stopping / report the best-validation epoch (not the last).
- Left + right soft-voting to one prediction per subject (like the tutorial).
- To chase the tutorial's numbers: heavier preprocessing (grey-matter segmentation) or
  a 3-D model — not more epochs.
- Spend the **test set** once, at the very end, on the chosen design.
