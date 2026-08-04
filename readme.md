# OASIS-1 Alzheimer's CNN — a teaching pipeline

This project trains a small neural network to look at a **single 2-D slice of a brain
MRI** and guess whether the person is **CDR-positive** (label `1`) or **CDR-negative**
(label `0`). It uses Washington University's
[OASIS-1](https://www.oasis-brains.org) brain-scan dataset.

> **What the labels mean.** The target comes straight from the Clinical Dementia Rating
> (CDR): **CDR-negative = CDR 0** (no impairment); **CDR-positive = CDR 0.5 / 1 / 2** (any
> impairment, severities pooled). These are clinical-severity classes, *not* biomarker-
> confirmed diagnoses — so we deliberately avoid stigmatizing or diagnosis-implying names
> for the two classes and label them by their CDR status instead.
>
> That distinction is worth more than a disclaimer. In biomedical work the thing you actually
> care about is often the thing you *cannot* measure — it needs a biopsy, a PET scan, an
> autopsy, or simply wasn't recorded — so datasets label with an available **proxy** instead.
> Ours is CDR. This model can learn to predict a clinician's rating of impairment; it cannot
> learn to detect Alzheimer's disease, because nothing in this dataset tells it what that is.
> [introduction.md](introduction.md) §1 works through the consequences.

It is written as a basic introduction to convolutional neural networks (CNNs) for biological imaging, while it might not win a competition. Read
**[introduction.md](introduction.md)** to explain the ideas and why this project is
built the way it is. This file explains *how to run it* and *what every setting does*.

The whole project runs as six small Python scripts, in order. Each one reads a single
settings file, **`config.yaml`**, and writes its results into an `outputs/` folder.

```
raw brain scans ─▶ step1 ─▶ step2 ─▶ step3 ─▶ step4 ─▶ step5 ─▶ step6
                  metadata  cohort   slice     train    compare  Grad-CAM
                  .csv      splits   PNGs       curves   plot     heatmaps
```

---

## 1. Before you start

**For this class we use GitHub Codespaces (Option A below)** — it runs the whole project in
your web browser with nothing to install. You are also welcome to download everything and run
it **locally on your own computer** (Option B). Both paths need the two things below.

### Get a GitHub account
Register a free account at https://github.com/signup. You'll use it to open the project in
Codespaces (Option A) or to download the code (Option B).

### Get the OASIS-1 data
The brain scans are **not** included in this repo, and they are **not** an anonymous download —
you request access and agree to a short usage agreement first.

1. Go to https://sites.wustl.edu/oasisbrains/home/oasis-1/ and click **"Request Access to
   Datasets"** (it takes you to https://sites.wustl.edu/oasisbrains/home/access/). Fill in the
   **OASIS Data Access Form** — your name, institutional email, institution, and a short
   research statement (its Aims, Methods, and Variables of interest). **OASIS-1** (and OASIS-2)
   are the open-access tier, so approval is quick; OASIS-3/4 are restricted and not needed here.
2. **Read the Data Use Agreement carefully.** In short: the data is for **academic,
   non-commercial** research only; you may **not redistribute** it to others — so **never
   commit the scans into a repository** (that's why `data/` and `outputs/` are gitignored); you
   may **not** use the images for face recognition or re-identification; and you must **cite
   Marcus et al. 2007** (J. Cognitive Neuroscience; doi:10.1162/jocn.2007.19.9.1498) and include
   the OASIS acknowledgment ("Data were provided by OASIS…").
3. Once you're approved, the dataset comes as **12 archive files**,
   `oasis_cross-sectional_disc1.tar.gz … disc12.tar.gz` (~1.5 GB each), from
   `download.nrg.wustl.edu`. You unzip ("extract") them so you get one `discN/` folder per disc:
   - one file at a time (macOS/Linux): `tar -xzf oasis_cross-sectional_disc1.tar.gz`
   - all at once, from the folder holding the archives:
     `find . -type f -name 'oasis_cross-sectional_disc*.tar.gz' -exec tar -xzf {} \;`
   - **or extract only the files the pipeline actually uses** — much smaller, since it skips the
     raw scans and other extras (~5 GB → ~0.5 GB per disc). All it needs is each session's
     `.txt` plus the masked_gfc volume and its `.hdr`, which you can pull out by pattern. On
     **macOS** (bsdtar, the default):
     ```bash
     tar -xzf oasis_cross-sectional_disc1.tar.gz \
         '*_t88_masked_gfc.img' '*_t88_masked_gfc.hdr' '*_MR[0-9].txt'
     ```
     On **Linux** (GNU tar) add `--wildcards --no-anchored` before the patterns (GNU tar needs
     them to treat the patterns as globs; bsdtar does that by default):
     ```bash
     tar -xzf oasis_cross-sectional_disc1.tar.gz --wildcards --no-anchored \
         '*_t88_masked_gfc.img' '*_t88_masked_gfc.hdr' '*_MR[0-9].txt'
     ```
     (This is exactly what `download-extract-data.sh` does for all 12 discs automatically.)
   - or use the helper script `download-extract-data.sh`, which downloads **and** extracts all
     12 for you and also fetches the OASIS reference spreadsheets + fact sheet into a `docs/`
     folder beside the discs. It reads the data location from `.env`, so copy
     `.env.example` → `.env` first (Option A step 3, or Option B). Only use it after your
     access request is approved and you've agreed to the Data Use Agreement.

     **That script needs `wget`**, a small command-line downloader. Codespaces (Option A) and
     most Linux machines already have it; **macOS does not**. On a Mac, pick whichever is
     easiest for you:
     - **Homebrew** — install Homebrew from https://brew.sh, then run `brew install wget`.
     - **conda / mamba** — if you're setting up an environment anyway (Option B, step 2), run
       `mamba install -c conda-forge wget` (or `conda install -c conda-forge wget`).
     - **Or skip it entirely** — download the 12 archives by hand from the OASIS site and
       extract them with `tar`, as described in the bullets above. Nothing else in this
       project needs `wget`, so this costs you only the convenience of the script.

However you get them, the files must end up in this layout (the pipeline uses whatever `disc*`
folders it finds):

```
<your data root>/          # e.g. the `data/` folder used in the run options below
  disc1/  disc2/  disc3/  …
    OAS1_XXXX_MRy/
      OAS1_XXXX_MRy.txt
      PROCESSED/MPRAGE/T88_111/
        OAS1_XXXX_MRy_mpr_n{3,4}_anon_111_t88_masked_gfc.img
  docs/                    # the OASIS reference materials (optional; step 1 only
    oasis_cross-sectional.xlsx              #   uses the first one, as a sanity check)
    oasis_cross-sectional-reliability.xlsx
    oasis_cross-sectional_facts.pdf
```

---

## 2. Option A — Run in the cloud with GitHub Codespaces

This is the **class default** and the easiest way — everything runs in your browser, with
nothing to install. (These steps assume no coding experience.)

1. **Log into GitHub** (make a free account first — see §1).
2. **Open the project in Codespaces:** click
   https://codespaces.new/emarti/immersion2026-ai-imaging. This opens a full code editor (VS
   Code) inside your browser — nothing to install; the first load takes a minute or two. If a
   dialog asks about trusting the workspace, click **"Trust Folder & Continue"**. On the
   **left** is the *file explorer* (the list of files and folders); along the **bottom** is the
   *terminal* (a panel where you type commands — if you don't see it, open the top menu
   **Terminal → New Terminal**).
3. **Make your settings file, `.env`.** The code reads the data location from a small file
   called `.env`; you create it by copying the example.
   - **Using the menus:** in the file explorer click `.env.example` (it's in the top-level
     list, alongside `config.yaml` and the `step…` files) to open it, then choose
     **File → Save As…** and name the copy `.env`, keeping it in that same top-level folder.
   - **Or in the terminal:** type `cp .env.example .env` and press Enter (`cp` means "copy").
4. **Check the data path.** Click `.env` in the file explorer to open it. It should already say
   `DATA_RAW_PATH=/workspaces/immersion2026-ai-imaging/data/`. That's the correct location for
   Codespaces, so just leave it (if you change anything, save with **Ctrl/Cmd + S**).
5. **Get the brain scans into that folder** (you must have OASIS access first — see §1). Two
   ways:
   - **Automatic (recommended, fastest):** in the terminal, run
     `bash download-extract-data.sh`. It reads your `.env`, creates the `data` folder, and
     downloads + unzips all 12 discs straight from WashU. The download is large (~18 GB total)
     and takes a while — but the script **extracts only the files the pipeline uses** (skipping
     the rest entirely), so the data ends up ~6 GB instead of ~60 GB (important on Codespaces'
     limited disk). Run `EXTRACT_ONLY_MASKED=0 bash download-extract-data.sh` to unpack every
     file.
   - **Manual:** make a folder named `data` at the top level (click the *new-folder* icon at the
     top-left of the file explorer, just to the right of **IMMERSION2026-AI-IMAGING**), then
     drag your files into it. Dragging many files is slow (30 minutes to a few hours).

   Either way, the result must look like `data/disc1/`, `data/disc2/`, ….
6. **Install the packages.** In the terminal, type `pip install -r requirements.txt` and press
   Enter — this fetches the Python libraries the code needs (do it once; **no virtualenv or
   conda is needed in Codespaces**).
7. **Run the code!** Type `bash process.sh` and press Enter to run the whole pipeline (about
   20 minutes), or run the steps one at a time — see **§4** below. Progress prints in the
   terminal, and results appear in `outputs/`.

**Managing your codespace.** You can see and reopen your running codespaces at
https://github.com/codespaces. **Closing the browser tab does *not* stop the session** — it
keeps running (handy for logging back in later), so when you're finished, **stop or delete** it
from that page to avoid using up your free monthly hours.

---

## 3. Option B — Run locally

Optional — for running on your own computer instead of the cloud. It's a bit more setup: you
install git and Python yourself.

1. **Install git and download the code.** Install git from https://git-scm.com/downloads. Then,
   in a terminal, download ("clone") the repo:
   ```bash
   git clone https://github.com/emarti/immersion2026-ai-imaging.git
   cd immersion2026-ai-imaging
   ```
   (If you've set up SSH keys, you can instead clone with
   `git pull git@github.com:emarti/immersion2026-ai-imaging.git`.)

2. **Install Python + the packages.** Use a virtualenv or conda (recommended locally, unlike
   Codespaces):
   ```bash
   # plain pip + venv:
   python3.13 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
   ```bash
   # or conda / mamba:
   mamba create -n oasis python=3.13
   mamba activate oasis
   pip install -r requirements.txt
   ```
   The packages: `numpy`, `pyyaml`, `python-dotenv`, `pillow`, `nibabel` (reads the brain
   images), `openpyxl` (reads the reference spreadsheet), `torch` + `torchvision` (the neural
   network), and `matplotlib` (the comparison plot). (Python 3.13 shown; 3.10–3.14 all work.)

3. **Point the code at your data.** Copy the example settings file and set your data path:
   ```bash
   cp .env.example .env
   # edit .env: the default DATA_RAW_PATH is the Codespaces path, so LOCALLY change it to the
   # folder that holds your extracted OASIS discs, e.g.:
   DATA_RAW_PATH=/path/to/your/oasis
   ```
   If `DATA_RAW_PATH` is not set, the scripts stop immediately with a clear message. You can
   fetch the data with `bash download-extract-data.sh` (it downloads into whatever
   `DATA_RAW_PATH` you set) or manually (see §1). Everything the pipeline *produces* goes into
   `./outputs/` and is safe to delete and regenerate.

4. **Run the code!** Run `bash process.sh` for the whole pipeline, or run the steps one at a
   time — see **§4** below.

---

## 4. The pipeline, step by step

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
python step4c-train-network.py   # wider 32-64-128, 5x5 first filter, dropout 0.6/0.2
python step4d-train-network.py   # shallower 8-16 (2 blocks), dropout 0.4/0.2
python step5-plot-training.py
python step6-gradcam.py          # Grad-CAM heatmaps (needs a model_4?.pt from step4)
```

(If you re-run step2 or step3 after changing the config, re-run step3 **and** the step4
scripts so the training uses the new images.)

The table below is what each step does and produces:

| Step | Script | What it does | Produces |
|---|---|---|---|
| 1 | `step1-gather-metadata.py` | Reads every session's `.txt`, works out age/sex/label, finds the image file, cross-checks against the official OASIS spreadsheet, and plots the age distribution by CDR status for the **whole dataset**. | `outputs/metadata.csv` + `outputs/dataset_age_histograms.png` |
| 2 | `step2-assign-groups.py` | Chooses the study **cohort** (which subjects to use) and splits them into **train / validate / test** groups; also plots the cohort's age distribution by CDR status. | `outputs/splits.yaml` + `outputs/cohort_age_histograms.png` |
| 3 | `step3-generate-slices.py` | For each chosen subject, cuts 2-D **hippocampus patches** (left and right) out of the 3-D brain and saves them as PNG images; also draws a few **context** images showing the crop boxes on the full slice. | `outputs/<split>/*.png` + `outputs/manifest.yaml` + `outputs/slice_context/` |
| 4 | `step4a … step4d-train-network.py` | Trains **four different CNN designs**, each recording how well it does after every training pass. | `outputs/training_log_4{a,b,c,d}.csv` |
| 5 | `step5-plot-training.py` | Draws all the designs' learning curves on one figure so you can compare them. | `outputs/training_comparison.png` |
| 6 | `step6-gradcam.py` | **Grad-CAM**: overlays a heatmap on sample patches showing *where* a trained design looks to call one CDR-positive. | `outputs/gradcam/` + `gradcam_grid.png` |

**The "label"** (what we're trying to predict) comes from the **CDR** (Clinical
Dementia Rating) in each session's text file: `CDR = 0` → CDR-negative (`0`); `CDR = 0.5,
1, or 2` → CDR-positive (`1`); a blank CDR (young subjects were never assessed) → the
subject is skipped.

---

## 5. The config file — `config.yaml` (read this carefully)

**Everything you can change lives in `config.yaml`.** You should not need to edit the
Python to run experiments — change a value here and re-run. Below is every setting,
grouped exactly as in the file, in plain language.

### 5.1 Data locations
```yaml
outputs_path: ./outputs
reference_xlsx: docs/oasis_cross-sectional.xlsx
discs: [disc1, disc2, …, disc12]
image_glob: "PROCESSED/MPRAGE/T88_111/*_t88_masked_gfc.img"
```
- **`outputs_path`** — the folder where all generated files go. Everything under it
  can be safely deleted; the pipeline recreates it.
- **`reference_xlsx`** — the official OASIS spreadsheet. It is **not** shipped in this
  repo (OASIS materials aren't redistributed here); `download-extract-data.sh` fetches
  it into a `docs/` folder **beside your discs**, which is why this path is relative to
  your data root (`DATA_RAW_PATH`) rather than to the code. Step 1 double-checks its own
  metadata against this and reports any mismatch — it never stops the run, and if the
  file isn't there the check is simply skipped.
- **`discs`** — the list of disc folder names to look inside. Names not present on
  disk are simply skipped.
- **`image_glob`** — the file pattern for the brain image inside each session. The
  `*` matches a small naming difference between scans (`n3` vs `n4`), so you don't
  have to care which one a session used.

### 5.2 Cohort selection — *which subjects to include*
```yaml
cohort:
  age_min: 60
  age_max: 79
  balance: label      # strict | label | none
  seed: 42
  max_subjects: null
```
- **`age_min` / `age_max`** — only include subjects whose age is in this range
  (inclusive). Dementia is age-related, so restricting the age range keeps the CDR-negative
  and CDR-positive groups more comparable. Widening it usually gives you more subjects.
  > **Danger — the age (and sex) confound.** Setting `age_min` too **low** pulls in
  > younger subjects, who are mostly CDR-negative, so the CDR-negative and CDR-positive groups end up
  > differing systematically in age. The network can then score well by quietly learning
  > to guess **age (or sex)** — which correlate with the label in this sample — instead
  > of reading hippocampal atrophy. Raising `age_min` (e.g. to **70**) better
  > **age-matches** the groups. You get **much less data and lower accuracy**, but that
  > lower number is more *honest*: some of the earlier accuracy was the model exploiting
  > the age/sex confound, not detecting disease. On small clinical data, a lower,
  > well-controlled score usually beats a higher, confounded one.
- **`balance`** — how to even out the groups. This matters a lot (see
  [introduction.md](introduction.md)):
  - `strict` — equal numbers in all four **sex × label** cells (Male/CDR-,
    Male/CDR+, Female/CDR-, Female/CDR+). Fairest, but throws away the most
    data because it's limited by the rarest cell.
  - `label` — equal **CDR-negative vs CDR-positive**, but don't force the sexes to match.
    Roughly **doubles** the usable subjects here, and keeps chance at 50 %.
  - `none` — use every eligible subject (most data, but the groups may be uneven). With
    uneven groups, **raw accuracy can mislead and chance is no longer 0.50**, so read
    **balanced accuracy** (step5 plots it) as the fair headline.
- **`seed`** — a fixed number that makes the random choices (which subjects, which
  train/val/test split) repeatable. Same seed → same cohort every time.
- **`max_subjects`** — a cap for quick test runs (`null` = no cap). Set it small to
  make the whole pipeline finish in seconds while you're experimenting.

### 5.3 Slice geometry — *which flat slices to cut*
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

### 5.4 Hippocampus crop + augmentation — *cut a small box, optionally jiggle it*
```yaml
hippocampus:
  ap: [80, 148]
  lr_left: [90, 148]
  apply_random_shifts: true
  random_shift: 3
  n_shifts: 10
  apply_random_reflections_rotations: true
```
Instead of feeding the whole slice, we crop a small rectangle around each hippocampus.
There are two (left and right), and we save them as **separate images** — which
doubles the amount of data.
- **`ap`** — the top/bottom (anterior–posterior) rows of the crop box, in voxels.
- **`lr_left`** — the left/right columns of the box for the **left** hippocampus. The
  **right** box is this one mirrored to the other side automatically. Note we mirror the box
  **location** but do **not** flip the pixels, so the two hippocampi reach the network as
  mirror images of each other — which is fine (a CNN copes with both orientations, and it
  doubles as free reflection augmentation; see introduction.md §10 for the why). This is
  about the **static** left/right crop-box mirroring only; the three settings below add a
  *separate*, optional, random augmentation on top of it.
- **`apply_random_shifts`** — turn **data augmentation** on/off. When `true`, each
  **training** crop is made several times, each nudged by a small random amount — this
  teaches the network to not depend on the exact position. When `false`, the
  box is used as-is. Validation/test are **never** shifted.
  > **What we found (see [lab-notes.md](internal/lab-notes.md)).** Because each training image
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
- **`apply_random_reflections_rotations`** — a **second**, independent **training**-only
  augmentation, applied on top of whatever `apply_random_shifts` already did. When `true`,
  each training crop is *also* independently flipped left-right (25% chance), flipped
  top-to-bottom (25% chance, drawn independently of the left-right flip), and rotated by a
  random multiple of 90°. Those probabilities are fixed in code
  (`REFLECT_LR_PROB`/`REFLECT_UD_PROB` in `step3-generate-slices.py`), not configurable
  here. Every saved patch stays exactly `ap` × `lr_left` in size no matter which rotation is
  drawn — step3 crops a slightly larger square first, rotates it losslessly (no resizing or
  blurring), then crops back down to size. Validation/test are **never** reflected or
  rotated, same as they're never shifted.
  > **None of these three transforms is how a brain is actually scanned** — no real
  > acquisition is upside-down or rotated 90° — but that's not really the point. The reason
  > the free left/right pair above works is that **atrophy**, the signal we care about,
  > looks the same reflected; on a small, tightly-cropped, centred patch like ours,
  > orientation carries essentially no diagnostic information either way, so flipping or
  > rotating it is a cheap, standard regularizer for a small dataset (same spirit as
  > dropout, §9 of introduction.md), not a compromise on realism. Whether it empirically
  > helps *this* model is still worth checking; see
  > [lab-notes.md](internal/lab-notes.md) for what running with this on actually showed.

### 5.5 Splits — *how to divide the subjects*
```yaml
splits:
  train:    0.70
  validate: 0.15
  test:     0.15
```
The fractions of subjects put into each group. **Splitting is done per subject**, so
every slice from one person stays in one group — this prevents the network from
"cheating" by seeing the same brain in both training and testing.

### 5.6 Training + reporting
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

### 5.7 Grad-CAM (step 6)
```yaml
gradcam:
  model: model_4a.pt      # which step4 checkpoint to explain
  split: validate         # patches to sample from: train | validate | test
  n_samples: 12           # how many patches to visualize
  seed: 0                 # sampling seed (reproducible)
  overlay_alpha: 0.45     # heatmap opacity over the grayscale patch (0-1)
```
Step 6 draws heatmaps of **where** a trained model looks to push a patch toward CDR-positive.
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

## 6. What the pipeline produces (files & formats)

All under `outputs/`:

- **`metadata.csv`** — one row per scan session: `subject, disc, age, sex, cdr, mmse,
  label, img_path, img_exists`. This is just the hand-off from step 1 to step 2.
- **`dataset_age_histograms.png`** — step 1's age distribution of CDR-negative vs CDR-positive
  over the **entire dataset** (all ages), in three panels (both sexes pooled, sexes separated,
  and by raw CDR grade). Shows the two age clusters and the age/sex confound at a glance.
- **`splits.yaml`** — the chosen cohort and its train/validate/test assignment, plus a
  `meta` summary (which balancing mode, total subjects, counts per split).
- **`cohort_age_histograms.png`** — step 2's version of the same three panels, but restricted
  to the configured **`age_min`–`age_max`** range (the eligible cohort) — useful for checking
  how balanced the classes look within the age band you chose.
- **`manifest.yaml`** — the master list of every generated patch. Each entry records
  `png_path, img_path, subject, split, label, cdr, slice_index, offset_mm, side` (L/R),
  and `shift_x/shift_y` (the augmentation nudge, `0` when off). `cdr` is the raw
  CDR grade (0.5/1/2) behind label 1. Step 4 reads this file.
- **`train/  validate/  test/`** — the PNG patches, named
  `{subject}_lbl{label}_ax{sliceindex}_{L|R}_a{copy}.png`. The label and side are in
  the filename so you can eyeball them in a file browser.
- **`slice_context/<split>/`** — step 3's context images for a sample of subjects
  (`slices.context_samples`, default 3 per split): the full axial slice with a rectangle
  around each crop window — **L** in lime, **R** in deepskyblue. Train images show **every
  random-shift box per plane** (one image per plane); validate/test show the single box.
  Handy for judging whether the ROI is the right size and lands on the hippocampus.
- **`training_log_4{a,b,c,d}.csv`** — one per design. Columns: `epoch, train_loss,
  train_acc, val_loss, val_acc, val_sens, val_spec, val_bal_acc`, plus per-severity
  validation accuracy `val_acc_cdr05, val_acc_cdr10, val_acc_cdr20`.
- **`training_comparison.png`** — all designs' curves overlaid in four panels
  (training loss/accuracy on top, validation accuracy and balanced accuracy below).
- **`model_4{a,b,c,d}.pt`** — each design's trained weights (a PyTorch `state_dict`),
  saved by step 4 so step 6 can reload them without retraining.
- **`gradcam/`** + **`gradcam_grid.png`** — step 6's Grad-CAM overlays: per patch, **two
  panels** (push-toward-CDR− and push-toward-CDR+, in two distinct colormaps), plus a
  montage of all sampled patches.
- **`gradcam_context/`** — step 6's whole-slice overlays: one PNG per subject, **two
  side-by-side full axial slices** (CDR− | CDR+) with the L/R hippocampus heatmaps
  drawn at their crop boxes (this pass reloads the raw volume).

### Reading the training logs
Each **epoch** is one full pass over the training images. For each we record, on both
the training set and the (held-out) validation set:
- **loss** — how wrong the network is (lower = better);
- **accuracy** — fraction of patches classified correctly (0.5 = pure guessing *only
  when the two groups are equal-sized*; with `balance: none` they may not be, so lead
  with balanced accuracy below);
- **sensitivity** — of the truly CDR-positive, how many were caught;
- **specificity** — of the truly CDR-negative, how many were correctly cleared;
- **balanced accuracy** — the average of sensitivity and specificity (the fair
  headline number). See [introduction.md](introduction.md) for what these mean and why they
  matter.

The encouraging sign of learning: training accuracy climbs. The sign that it will
**generalize** to new people: the *validation* numbers improve too. When training keeps
improving but validation stalls or gets worse, the model is **overfitting** — see
[introduction.md](introduction.md).

step5 also prints a **breakdown of validation accuracy by CDR grade**. Our label 1
lumps together CDR 0.5 (very mild), 1 (mild), and 2 (moderate); this breakdown
shows how well each grade is caught — you'd expect more-severe (more atrophied) cases to
be easier. It prints the patch/subject count per grade too, because those counts are
**small** (only a handful of validation subjects per grade), so treat the numbers as
trends, not precise figures.

### Reading the Grad-CAM heatmaps
step 6 shows, for each patch, **two panels**: **left = "pushes toward CDR−"** and
**right = "pushes toward CDR+"**, in two distinct colormaps (`jet` for CDR+, `cool`
for CDR−) so you can tell them apart — each highlighting where changing the image would
move the score that way. Three things to keep in mind:

- **The two directions are complementary, not opposites.** With one output, the CDR-positive
  map is `ReLU(+importance)` and the CDR-negative map is `ReLU(−importance)` — the same signed
  map split into its positive and negative lobes — so they light up *different* regions.
  Looking quite different is expected (see [introduction.md](introduction.md) §14).
- **It explains a chosen target, not the truth.** A *CDR-negative* patch's CDR-positive panel
  still shows "what would make it look CDR-positive." The panel **title** is colored by the
  true label — **blue = CDR-negative, red = CDR-positive** — next to the model's `P(CDR+)`.
- **It's coarse.** The map comes from the last conv layer (a ~10×8 grid here), upsampled —
  so it localizes roughly, not to the pixel. Use it as a sanity check ("is the network
  looking at the hippocampus, or at a border artifact?"), not a precise segmentation.

step 6 also writes **whole-slice context** overlays to `outputs/gradcam_context/` (one per
subject): two side-by-side full axial slices (CDR− | CDR+) with the L/R heatmaps drawn
at their crop boxes, so you can see *where in the brain* the model finds each kind of
evidence. That pass reloads the raw volume, so it needs `DATA_RAW_PATH` set.

Point step 6 at another model or split via `config.yaml` (`gradcam:`) or the command line,
e.g. `--model model_4c.pt --split test --n 16`. Method + citation are in
[introduction.md](introduction.md).

---

## 7. The four CNN designs (step 4)

The four designs are a **teaching sweep**: a shared **baseline** (4a) plus three variants
that each change exactly **one** thing — so comparing a variant against 4a isolates that
effect. Together they sample three different directions: dropout, width, and depth
(shallower). Each is one self-contained file you can read top to bottom.

| Design | Conv blocks | Dropout | Direction sampled |
|---|---|---|---|
| **4a** | 8→16→32 (3) | 0.6 / 0.2 | baseline (~8.2k params) |
| **4b** | 8→16→32 (3) | 0.0 / 0.0 | no dropout → overfitting demo |
| **4c** | 32→64→128 (3), 5×5 first | 0.6 / 0.2 | wider (~102k params, 12× the baseline) |
| **4d** | 8→16 (2), 32-unit head | 0.4 / 0.2 | smaller (~1.9k params) |

All use one output number (a "logit"), `BCEWithLogitsLoss`, the `AdamW` optimizer with
learning rate `1e-4`, batch size 32, and a global-average-pooling head so the exact
patch size doesn't matter. What each block and term means is in
[introduction.md](introduction.md).

---

## 8. Mini-glossary

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

---

## 9. The Jupyter notebooks (optional)

Two extra notebooks live in `jupyter/`. They are **not** part of `process.sh` and they never
write to `outputs/` — they read the OASIS spreadsheet directly and ask questions the pipeline
itself can't:

| Notebook | The question it asks |
|---|---|
| `jupyter/nWBV.ipynb` | How well does a single number — whole-brain volume — predict CDR status? |
| `jupyter/age_confounder.ipynb` | How much of CDR status is predictable from **age alone**, with no brain scan at all? |

They need two packages beyond the pipeline's, `ipykernel` and `scikit-learn`. Both are already
in `requirements.txt`, so if you followed §2 or §3 you have them.

### Opening a notebook in VS Code

Click the `.ipynb` file in the file explorer. It opens as a stack of *cells*; press
**Shift + Enter** to run the cell you're on, or click **Run All** at the top to run everything
in order.

Before any cell runs, VS Code needs to know **which Python** to use — that choice is called the
*kernel*. Click **Select Kernel** at the top right, then:

- **In Codespaces (Option A):** pick the one Python listed under **Python Environments**.
  `pip install -r requirements.txt` already put the packages there, and there is no `oasis`
  environment in Codespaces — so there is nothing else to set up.
- **Locally with conda/mamba (Option B):** pick the environment named **`oasis`**. VS Code
  discovers conda environments on its own, as long as `ipykernel` is installed inside them.

### If `oasis` doesn't show up in that list (local only)

Register it by hand. **Activate the environment first** — this step matters, because the command
registers whichever Python is active *right now* under the name you give it. Run it from the
wrong environment and you get a kernel labelled "oasis" that is actually some other Python:

```bash
mamba activate oasis
python -m ipykernel install --user --name=oasis --display-name="Oasis (Python 3.13.14)"
```

Reopen the notebook and it appears in the list as *Oasis (Python 3.13.14)*. `--name` is the
internal id; `--display-name` is just the label you see, so if you later upgrade the
environment's Python, re-run the command to refresh it — the label is fixed text and won't
update itself.

---

## License & data use

- **Code and written materials** in this repository are released under the **MIT License**
  (see [LICENSE](LICENSE)) — free to use, modify, and share, with attribution. The CNN is
  adapted from PyTorch's MNIST example (BSD-3-Clause).
- **The OASIS-1 data is *not* covered by this license and is *not* included here.** It is
  provided by OASIS under its own **Data Use Agreement** (academic, non-commercial; no
  redistribution). Request access and download it yourself — see §1. The OASIS reference
  spreadsheets and fact sheet are likewise not redistributed in this repo;
  `download-extract-data.sh` fetches them from the OASIS site into `<your data root>/docs/`.
