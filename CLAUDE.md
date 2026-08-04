# CLAUDE.md

Guidance for AI assistants working in this repository. It is also part of the teaching
materials — students are welcome to read it, so it is written as if they will.

## What this is

A **teaching** pipeline: a small PyTorch CNN that classifies 2-D transverse hippocampus
patches from OASIS-1 MRI as CDR-positive vs CDR-negative. Adapted from PyTorch's MNIST
example. MIT licensed.

**The audience are incoming graduate students who may range from little to deep experience
with the command line, python, and deep learning.** That range — not any single level — is the
thing to design for, and it inverts several normal instincts:

- Verbose explanatory comments are **teaching material, not clutter**. Do not "clean up" or
  condense them. When you touch a file, match its existing comment density — which is high.
- Prefer the obvious construction over the clever one, even at some cost in elegance. A clever
  one-liner is a wall to some readers and merely shorter to the rest; the plain version is
  legible to everyone.
- `readme.md` spells out the mechanics — which command, which folder, what a flag does — so
  that nobody is *blocked* on tooling while trying to learn the science. That level of detail
  is a floor for whoever needs it, not a ceiling: someone fluent in a terminal will skim it,
  someone meeting one this week will follow it line by line, and both should get through.
  Keep that register when editing it.

## Non-negotiable: class nomenclature

The two classes are **CDR-negative** (label 0, CDR = 0) and **CDR-positive** (label 1,
CDR = 0.5 / 1 / 2). Identifiers are `cdr_negative` / `cdr_positive`; display strings live in
`config.yaml` under `labels:`; cramped plots use `CDR-` / `CDR+`.

**Never** write "healthy", "demented", "AD", "Alzheimer's patient", or similar as a class
name, variable, plot label, or prose description of a group. Two reasons, and the first is the
one people skip:

1. **It would be wrong.** Our ground truth is **CDR status**, and CDR is not Alzheimer's
   disease. CDR rates *observed cognitive and functional impairment*; Alzheimer's is a specific
   pathology. Impairment has many causes besides Alzheimer's, and Alzheimer's pathology can be
   present in someone rated CDR 0. These are different variables, and OASIS-1 contains only one
   of them. Naming the class "AD" claims something the data cannot support.
2. **It would be stigmatizing.** Labelling a person by a disease they may not have.

Applies to code, comments, docs, plot text, and commit messages alike.

This generalizes, and it is one of the lessons the project exists to teach: **in biomedical
work the ground truth is often hard or impossible to access**, so datasets ship a *proxy* that
is merely available. Elsewhere in machine learning a photo of a cat is labelled "cat" and the
label is correct by definition; here, confirming Alzheimer's needs biomarkers or an autopsy, so
a clinician's rating stands in for it. Everything downstream inherits that substitution.

So: **describe results in terms of what was actually trained on.** "Predicts CDR status" is
supportable. "Detects Alzheimer's" is not — do not silently upgrade the one to the other in a
summary, a plot title, or a commit message. `introduction.md` §1 has the full treatment.

(The one legitimate use of "AD" is describing what CDR clinically indicates, as in
`step1-gather-metadata.py`. "Alzheimer's" in the project title is fine. Naming a *group* is
not.)

## Working conventions

- **Environment is mamba/conda-forge**, env name `oasis` (at
  `~/.local/share/mamba/envs/oasis`; the system `python` does *not* have torch). Do not
  suggest `pip install` for local work. (`requirements.txt` exists for GitHub Codespaces,
  where pip *is* the path.)
- **The user runs everything.** Not just training runs and `process.sh` — this covers *any*
  execution of project code, including one-off diagnostics, `--epochs 1` smoke tests, and
  throwaway scripts that import from the repo. Read-only inspection (`grep`, reading files,
  `git status`, `py_compile`) is fine; anything that executes the project's code is theirs to
  run. When verification needs a run, **write the script and hand it over** — say what it
  checks and what output would mean it passed.
- **The user does all commits and pushes.** Never commit or push. There are two remotes,
  `origin` (emarti) and `origin2` (Marti-Lab-WashU), and both need pushing.
- Make the edits, then stop and report.

## Layout

Everything lives at the repository root — there is no source subfolder, deliberately, so
students never have to `cd` anywhere. Scripts run in numeric order:

| | | |
|---|---|---|
| `step1-gather-metadata.py` | metadata + whole-dataset age histograms | `outputs/metadata.csv` |
| `step2-assign-groups.py` | cohort selection + subject-level splits | `outputs/splits.yaml` |
| `step3-generate-slices.py` | hippocampus patch PNGs + crop-box context images | `outputs/<split>/`, `manifest.yaml` |
| `step4a`–`4d-train-network.py` | four CNN designs (baseline + 3 single-change variants) | `outputs/training_log_4?.csv`, `model_4?.pt` |
| `step5-plot-training.py` | comparison plot + text summary | `outputs/step5-training_comparison.png` |
| `step6-gradcam.py` | Grad-CAM heatmaps | `outputs/gradcam*/` |
| `step7-perturb.py` | perturbation saliency — **standalone** | `outputs/step7-perturb/` |

`process.sh` runs steps 1–6. **Step 7 is intentionally excluded** from `process.sh`,
`config.yaml`, and the reader-facing docs — it's an optional extra. Keep it that way unless
told otherwise.

`common.py` holds shared config loading and path helpers — use them (`metadata_csv`,
`splits_yaml`, `manifest_yaml`, `split_dir`, `reference_xlsx`) rather than rebuilding paths.

The step4 designs are **deliberately near-duplicate files** differing only in `Net`, so a
student can diff any two and see one change. Do not refactor them into a shared module.

## config.yaml is the single source of truth

Every tunable belongs in `config.yaml`, not in the Python. A student should be able to run
every experiment in the course by editing that file alone. If you find yourself adding a
constant to a step script, it probably belongs in the config — and it needs a comment
explaining it in the same plain-language register as its neighbours, plus an entry in
`readme.md` §5.

**Prose that restates a config value will drift from it.** The retired `plan.md` is the worked
example: it claimed `age_min: 60` and "the last 50 epochs" long after the config had moved on.
Prefer describing what a setting *does* over quoting what it currently *is*; where a number
genuinely has to appear in prose, changing the config means checking `readme.md` §5 and
`introduction.md` for the same figure.

## Paths and data

- Raw data root comes from `DATA_RAW_PATH` in a gitignored `.env` (`.env.example` is the
  template; its default is the Codespaces path and should stay that way).
- `outputs_path` is resolved relative to **the repo root** (`common.py`'s `REPO_ROOT`).
- `reference_xlsx` is resolved relative to **the raw data root**, not the repo — the OASIS
  spreadsheets are OASIS materials we do not redistribute, so `download-extract-data.sh`
  fetches them to `<DATA_RAW_PATH>/docs/`. A missing file is non-fatal; step1 skips the check.
- **Never commit OASIS data.** The Data Use Agreement forbids redistribution. `data/` and
  `outputs/` are gitignored.
- Scripts must keep working regardless of the current directory: they rely on `__file__`, not
  `os.getcwd()`. `step6`/`step7` load sibling step files by path via `importlib`.

## The docs, and what goes where

- `readme.md` — how to run it + the full `config.yaml` reference. Student-facing.
- `introduction.md` — the theory and the *why*. Student-facing.
- `internal/lab-notes.md` — the experiment log: runs, results, what was tried. Append, don't rewrite.
- `CLAUDE.md` — this file.

Results, findings, and TODOs go in `internal/lab-notes.md`. Don't start new tracker files.
