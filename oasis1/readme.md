# OASIS-1 Alzheimer's CNN — a teaching pipeline

This project trains a small neural network to look at a **single 2-D slice of a brain
MRI** and guess whether the person has **Alzheimer's-type dementia** (label `1`) or is
**healthy** (label `0`). It uses Washington University's
[OASIS-1](https://www.oasis-brains.org) brain-scan dataset.

It is written to be **read and understood**, not to win a competition. If you have
never seen machine learning or a "convolutional neural network" (CNN) before, read
**[introduction.md](introduction.md)** first — it explains the ideas and *why* this project is
built the way it is. This file explains *how to run it* and *what every setting does*.

The whole thing runs as five small Python scripts, in order. Each one reads a single
settings file, **`config.yaml`**, and writes its results into an `outputs/` folder.

```
raw brain scans ─▶ step1 ─▶ step2 ─▶ step3 ─▶ step4 ─▶ step5 ─▶ step6
                  metadata  cohort   slice     train    compare  Grad-CAM
                  .csv      splits   PNGs       curves   plot     heatmaps
```

---

## 1. What you need before you start

1. **The OASIS-1 data.** This is a real dataset of brain MRIs (you download it
   separately; it is not in this repo). It comes as a set of "discs" (`disc1`,
   `disc2`, …), each a folder of scan sessions. Each session `OAS1_XXXX_MRy` has:
   - a text file `OAS1_XXXX_MRy.txt` with the person's age, sex, and clinical scores,
   - a pre-processed 3-D brain image under `PROCESSED/MPRAGE/T88_111/…_masked_gfc.img`
     (already aligned to a standard brain template and with the skull removed).

   The expected folder layout:

   ```
   <your data root>/
     disc1/  disc2/  disc3/  …
       OAS1_XXXX_MRy/
         OAS1_XXXX_MRy.txt
         PROCESSED/MPRAGE/T88_111/
           OAS1_XXXX_MRy_mpr_n{3,4}_anon_111_t88_masked_gfc.img
   ```
   > Discs arrive as `.tar.gz` archives — un-tar the ones you want first. The pipeline
   > uses whatever `disc*` folders it finds.

2. **Python 3.13** (3.10–3.14 all work) and the packages in `requirements.txt`.

---

## 2. Install

**Option A — plain pip + venv:**
```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Option B — conda / mamba:**
```bash
mamba create -n oasis python=3.13
mamba activate oasis
pip install -r requirements.txt
```

The packages: `numpy`, `pyyaml`, `python-dotenv`, `pillow`, `nibabel` (reads the brain
images), `openpyxl` (reads the reference spreadsheet), `torch` + `torchvision` (the
neural network), and `matplotlib` (the comparison plot).

---

## 3. Point the code at your data

The code needs to know where *your* copy of the raw OASIS data lives. That path is
different on every computer, so it is **not** stored in `config.yaml`. Instead, put it
in a private file called `.env`:

```bash
cp .env.example .env
# then edit .env and set the path to your extracted OASIS folder:
DATA_RAW_PATH=/path/to/your/oasis
```

If `DATA_RAW_PATH` is not set, the scripts stop immediately with a clear message.
Everything the pipeline *produces* goes into `./outputs/` (set by `outputs_path` in
the config) and is safe to delete and regenerate.

---

## 4. The pipeline, step by step

| Step | Script | What it does | Produces |
|---|---|---|---|
| 1 | `step1-gather-metadata.py` | Reads every session's `.txt`, works out age/sex/label, finds the image file, and cross-checks against the official OASIS spreadsheet. | `outputs/metadata.csv` (one row per scan session) |
| 2 | `step2-assign-groups.py` | Chooses the study **cohort** (which subjects to use) and splits them into **train / validate / test** groups. | `outputs/splits.yaml` |
| 3 | `step3-generate-slices.py` | For each chosen subject, cuts 2-D **hippocampus patches** (left and right) out of the 3-D brain and saves them as PNG images. | `outputs/<split>/*.png` + `outputs/manifest.yaml` |
| 4 | `step4a … step4d-train-network.py` | Trains **four different CNN designs**, each recording how well it does after every training pass. | `outputs/training_log_4{a,b,c,d}.csv` |
| 5 | `step5-plot-training.py` | Draws all the designs' learning curves on one figure so you can compare them. | `outputs/training_comparison.png` |
| 6 | `step6-gradcam.py` | **Grad-CAM**: overlays a heatmap on sample patches showing *where* a trained design looks to call one "demented". | `outputs/gradcam/` + `gradcam_grid.png` |

**The "label"** (what we're trying to predict) comes from the **CDR** (Clinical
Dementia Rating) in each session's text file: `CDR = 0` → healthy (`0`); `CDR = 0.5, 1,
or 2` → demented (`1`); a blank CDR (young subjects were never assessed) → the subject
is skipped.

---

## 5. How to run it

Run the whole pipeline at once:
```bash
bash process.sh              # number of epochs is set in config.yaml (`epochs`)
```

Or run the steps yourself, in order:
```bash
python step1-gather-metadata.py
python step2-assign-groups.py
python step3-generate-slices.py
python step4a-train-network.py   # baseline 8-16-32, dropout 0.6/0.2
python step4b-train-network.py   # 8-16-32, no dropout
python step4c-train-network.py   # wider 16-32-64, dropout 0.6/0.2
python step4d-train-network.py   # shallower 8-16 (2 blocks), dropout 0.6/0.2
python step5-plot-training.py
python step6-gradcam.py          # Grad-CAM heatmaps (needs a model_4?.pt from step4)
```

(If you re-run step2 or step3 after changing the config, re-run step3 **and** the
step4 scripts so the training uses the new images.)

---

## 6. The config file — `config.yaml` (read this carefully)

**Everything you can change lives in `config.yaml`.** You should not need to edit the
Python to run experiments — change a value here and re-run. Below is every setting,
grouped exactly as in the file, in plain language.

### 6.1 Data locations
```yaml
outputs_path: ./outputs
reference_xlsx: ./docs/oasis_cross-sectional.xlsx
discs: [disc1, disc2, …, disc12]
image_glob: "PROCESSED/MPRAGE/T88_111/*_t88_masked_gfc.img"
```
- **`outputs_path`** — the folder where all generated files go. Everything under it
  can be safely deleted; the pipeline recreates it.
- **`reference_xlsx`** — the official OASIS spreadsheet (shipped in `docs/`). Step 1
  double-checks its own metadata against this and reports any mismatch (it never
  stops the run — it's a sanity check).
- **`discs`** — the list of disc folder names to look inside. Names not present on
  disk are simply skipped.
- **`image_glob`** — the file pattern for the brain image inside each session. The
  `*` matches a small naming difference between scans (`n3` vs `n4`), so you don't
  have to care which one a session used.

### 6.2 Cohort selection — *which subjects to include*
```yaml
cohort:
  age_min: 60
  age_max: 79
  balance: label      # strict | label | none
  seed: 42
  max_subjects: null
```
- **`age_min` / `age_max`** — only include subjects whose age is in this range
  (inclusive). Dementia is age-related, so restricting the age range keeps the healthy
  and demented groups more comparable. Widening it usually gives you more subjects.
  > **Danger — the age (and sex) confound.** Setting `age_min` too **low** pulls in
  > younger subjects, who are mostly healthy, so the healthy and demented groups end up
  > differing systematically in age. The network can then score well by quietly learning
  > to guess **age (or sex)** — which correlate with the label in this sample — instead
  > of reading hippocampal atrophy. Raising `age_min` (e.g. to **70**) better
  > **age-matches** the groups. You get **much less data and lower accuracy**, but that
  > lower number is more *honest*: some of the earlier accuracy was the model exploiting
  > the age/sex confound, not detecting disease. On small clinical data, a lower,
  > well-controlled score usually beats a higher, confounded one.
- **`balance`** — how to even out the groups. This matters a lot (see
  [introduction.md](introduction.md)):
  - `strict` — equal numbers in all four **sex × label** cells (Male/Healthy,
    Male/Demented, Female/Healthy, Female/Demented). Fairest, but throws away the most
    data because it's limited by the rarest cell.
  - `label` — equal **healthy vs demented**, but don't force the sexes to match.
    Roughly **doubles** the usable subjects here, and keeps chance at 50 %.
  - `none` — use every eligible subject (most data, but the groups may be uneven). With
    uneven groups, **raw accuracy can mislead and chance is no longer 0.50**, so read
    **balanced accuracy** (step5 plots it) as the fair headline.
- **`seed`** — a fixed number that makes the random choices (which subjects, which
  train/val/test split) repeatable. Same seed → same cohort every time.
- **`max_subjects`** — a cap for quick test runs (`null` = no cap). Set it small to
  make the whole pipeline finish in seconds while you're experimenting.

### 6.3 Slice geometry — *which flat slices to cut*
```yaml
slices:
  slice_axis: 2
  middle_index: 88
  offsets_mm: [-27, -26, -25, -24, -23, -22]
  eval_offset_mm: -24
```
The 3-D brain is `176 × 208 × 176` voxels (1 voxel = 1 mm). We take **transverse**
(horizontal) slices — imagine a frisbee cutting straight through the head.
- **`slice_axis`** — which of the three axes to slice along. `2` is the
  inferior–superior (bottom-to-top) axis, so each slice is a horizontal cross-section.
- **`middle_index`** — the slice number at the middle of the head (`176 ÷ 2 = 88`).
  Offsets below are measured from here (and because 1 voxel = 1 mm, an offset in mm is
  just an offset in slices).
- **`offsets_mm`** — the slices used for **training**. These six are all at the level
  of the **hippocampus** (the memory structure that shrinks in Alzheimer's), a few mm
  apart. Using several nearby slices gives the network more training examples.
- **`eval_offset_mm`** — the **single** slice used for **validation and test**. Using
  one fixed slice makes evaluation clean and repeatable (no lucky/unlucky slice).

### 6.4 Hippocampus crop + augmentation — *cut a small box, optionally jiggle it*
```yaml
hippocampus:
  ap: [74, 154]
  lr_left: [88, 152]
  apply_random_shifts: true
  random_shift: 4
  n_shifts: 10
```
Instead of feeding the whole slice, we crop a small rectangle around each hippocampus.
There are two (left and right), and we save them as **separate images** — which
doubles the amount of data.
- **`ap`** — the top/bottom (anterior–posterior) rows of the crop box, in voxels.
- **`lr_left`** — the left/right columns of the box for the **left** hippocampus. The
  **right** box is this one mirrored to the other side automatically.
- **`apply_random_shifts`** — turn **data augmentation** on/off. When `true`, each
  **training** crop is made several times, each nudged by a small random amount — this
  teaches the network to not depend on the exact position. When `false`, the
  box is used as-is. Validation/test are **never** shifted.
  > **What we found (see [lab-notes.md](lab-notes.md)).** Because each training image
  > becomes `n_shifts` copies (10 here), the training set and the **time per epoch grow
  > ~10×**. In our runs it **mainly helps smaller / leaner networks** and does **not
  > appreciably raise** the headline balanced accuracy; stacked on very high dropout in
  > a tiny net it can even over-regularize and *hurt* (a model that collapses to always
  > guessing one class). Even so, augmentation is a **good habit** — it's standard
  > practice and pays off more with larger, real datasets.
- **`random_shift`** — the maximum nudge, in voxels (±4). With ±4 there are
  `(2·4+1)² = 81` possible nudges.
- **`n_shifts`** — how many of those 81 nudged copies to make per training image
  (10 here). Only used when `apply_random_shifts` is `true`.

### 6.5 Splits — *how to divide the subjects*
```yaml
splits:
  train:    0.70
  validate: 0.15
  test:     0.15
```
The fractions of subjects put into each group. **Splitting is done per subject**, so
every slice from one person stays in one group — this prevents the network from
"cheating" by seeing the same brain in both training and testing.

### 6.6 Training + reporting
```yaml
epochs: 100
avg_last_epochs: 20
```
- **`epochs`** — how many training passes each step4 design runs. This is the single
  place to change training length; `process.sh` reads it from here. (You can still
  override one script with `python step4a-train-network.py --epochs 5` for a quick test.)
- **`avg_last_epochs`** — how many of the final epochs step5 averages when it prints
  each design's validation summary (accuracy / sensitivity / specificity / balanced
  accuracy, and the by-grade breakdown).

### 6.7 Grad-CAM (step 6)
```yaml
gradcam:
  model: model_4a.pt      # which step4 checkpoint to explain
  split: validate         # patches to sample from: train | validate | test
  n_samples: 12           # how many patches to visualize
  seed: 0                 # sampling seed (reproducible)
  overlay_alpha: 0.45     # heatmap opacity over the grayscale patch (0-1)
```
Step 6 draws heatmaps of **where** a trained model looks to push a patch toward "demented".
- **`model`** — the checkpoint (written by step 4) to explain. Name the file directly; the
  network shape is **inferred** from it (`model_4a.pt` → the 4a design), so you don't pass
  a design letter. Defaults to the 4a baseline.
- **`split`** — which split to sample patches from (`validate` by default).
- **`n_samples`** — how many patches to show (a montage plus one PNG each).
- **`seed`** — makes the random sample of patches repeatable.
- **`overlay_alpha`** — how opaque the colored heatmap is over the grayscale patch.

Any of these can be overridden on the command line, e.g.
`python step6-gradcam.py --model model_4c.pt --split test --n 16`.

---

## 7. What the pipeline produces (files & formats)

All under `outputs/`:

- **`metadata.csv`** — one row per scan session: `subject, disc, age, sex, cdr, mmse,
  label, img_path, img_exists`. This is just the hand-off from step 1 to step 2.
- **`splits.yaml`** — the chosen cohort and its train/validate/test assignment, plus a
  `meta` summary (which balancing mode, total subjects, counts per split).
- **`manifest.yaml`** — the master list of every generated patch. Each entry records
  `png_path, img_path, subject, split, label, cdr, slice_index, offset_mm, side` (L/R),
  and `shift_x/shift_y` (the augmentation nudge, `0` when off). `cdr` is the raw
  dementia grade (0.5/1/2) behind label 1. Step 4 reads this file.
- **`train/  validate/  test/`** — the PNG patches, named
  `{subject}_lbl{label}_ax{sliceindex}_{L|R}_a{copy}.png`. The label and side are in
  the filename so you can eyeball them in a file browser.
- **`training_log_4{a,b,c,d}.csv`** — one per design. Columns: `epoch, train_loss,
  train_acc, val_loss, val_acc, val_sens, val_spec, val_bal_acc`, plus per-severity
  validation accuracy `val_acc_cdr05, val_acc_cdr10, val_acc_cdr20`.
- **`training_comparison.png`** — all designs' curves overlaid in four panels
  (training loss/accuracy on top, validation accuracy and balanced accuracy below).
- **`model_4{a,b,c,d}.pt`** — each design's trained weights (a PyTorch `state_dict`),
  saved by step 4 so step 6 can reload them without retraining.
- **`gradcam/`** + **`gradcam_grid.png`** — step 6's Grad-CAM overlays: per patch, **two
  panels** (push-toward-healthy and push-toward-demented, in two distinct colormaps), plus a
  montage of all sampled patches.
- **`gradcam_context/`** — step 6's whole-slice overlays: one PNG per subject, **two
  side-by-side full axial slices** (healthy | demented) with the L/R hippocampus heatmaps
  drawn at their crop boxes (this pass reloads the raw volume).

### Reading the training logs
Each **epoch** is one full pass over the training images. For each we record, on both
the training set and the (held-out) validation set:
- **loss** — how wrong the network is (lower = better);
- **accuracy** — fraction of patches classified correctly (0.5 = pure guessing *only
  when the two groups are equal-sized*; with `balance: none` they may not be, so lead
  with balanced accuracy below);
- **sensitivity** — of the truly demented, how many were caught;
- **specificity** — of the truly healthy, how many were correctly cleared;
- **balanced accuracy** — the average of sensitivity and specificity (the fair
  headline number). See [introduction.md](introduction.md) for what these mean and why they
  matter.

The healthy sign of learning: training accuracy climbs. The sign that it will
**generalize** to new people: the *validation* numbers improve too. When training keeps
improving but validation stalls or gets worse, the model is **overfitting** — see
[introduction.md](introduction.md).

step5 also prints a **breakdown of validation accuracy by AD severity**. Our label 1
lumps together CDR 0.5 (very mild), 1 (mild), and 2 (moderate) dementia; this breakdown
shows how well each grade is caught — you'd expect more-severe (more atrophied) cases to
be easier. It prints the patch/subject count per grade too, because those counts are
**small** (only a handful of validation subjects per grade), so treat the numbers as
trends, not precise figures.

### Reading the Grad-CAM heatmaps
step 6 shows, for each patch, **two panels**: **left = "pushes toward healthy"** and
**right = "pushes toward demented"**, in two distinct colormaps (`jet` for demented, `cool`
for healthy) so you can tell them apart — each highlighting where changing the image would
move the score that way. Three things to keep in mind:

- **The two directions are complementary, not opposites.** With one output, the demented map
  is `ReLU(+importance)` and the healthy map is `ReLU(−importance)` — the same signed map
  split into its positive and negative lobes — so they light up *different* regions. Looking
  quite different is expected (see [introduction.md](introduction.md) §14).
- **It explains a chosen target, not the truth.** A *healthy* patch's demented panel still
  shows "what would make it look demented." The panel **title** is colored by the true label
  — **blue = healthy, red = demented** — next to the model's `P(dem)`.
- **It's coarse.** The map comes from the last conv layer (a ~10×8 grid here), upsampled —
  so it localizes roughly, not to the pixel. Use it as a sanity check ("is the network
  looking at the hippocampus, or at a border artifact?"), not a precise segmentation.

step 6 also writes **whole-slice context** overlays to `outputs/gradcam_context/` (one per
subject): two side-by-side full axial slices (healthy | demented) with the L/R heatmaps drawn
at their crop boxes, so you can see *where in the brain* the model finds each kind of
evidence. That pass reloads the raw volume, so it needs `DATA_RAW_PATH` set.

Point step 6 at another model or split via `config.yaml` (`gradcam:`) or the command line,
e.g. `--model model_4c.pt --split test --n 16`. Method + citation are in
[introduction.md](introduction.md).

---

## 8. The four CNN designs (step 4)

The four designs are a **teaching sweep**: a shared **baseline** (4a) plus three variants
that each change exactly **one** thing — so comparing a variant against 4a isolates that
effect. Together they sample three different directions: dropout, width, and depth
(shallower). Each is one self-contained file you can read top to bottom.

| Design | Conv blocks | Dropout | Direction sampled |
|---|---|---|---|
| **4a** | 8→16→32 (3) | 0.6 / 0.2 | baseline (~8.2k params) |
| **4b** | 8→16→32 (3) | 0.0 / 0.0 | no dropout → overfitting demo |
| **4c** | 16→32→64 (3) | 0.6 / 0.2 | wider (~28k params) |
| **4d** | 8→16 (2) | 0.6 / 0.2 | shallower (~2.4k params) |

All use one output number (a "logit"), `BCEWithLogitsLoss`, the `AdamW` optimizer with
learning rate `1e-4`, batch size 32, and a global-average-pooling head so the exact
patch size doesn't matter. What each block and term means is in
[introduction.md](introduction.md).

---

## 9. Mini-glossary

- **Voxel** — a 3-D pixel; here 1 voxel = 1 mm of brain.
- **Slice / plane** — one flat 2-D cross-section of the 3-D brain.
- **Patch** — the small cropped rectangle around one hippocampus that we actually feed
  the network.
- **CDR** — Clinical Dementia Rating; the clinical score we turn into the label.
- **Hippocampus** — a brain structure important for memory; it shrinks in Alzheimer's,
  so it's a good place to look.
- **Cohort** — the specific set of subjects chosen for the study.
- **Epoch** — one full pass over the training data during learning.
- **Overfitting** — memorizing the training examples instead of learning a general
  rule (great training scores, poor validation scores).

---

For the *why* behind all of these choices — CNNs, overfitting, the balancing modes,
the hippocampus focus, the design sweep, and how this compares to the tutorial it's
based on — see **[introduction.md](introduction.md)**.
