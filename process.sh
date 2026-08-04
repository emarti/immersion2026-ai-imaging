#!/usr/bin/env bash
#
# Run the whole OASIS pipeline end to end:
#   metadata -> cohort -> slices -> train the CNN -> plot -> Grad-CAM -> predictor ablation.
#
# Activate the environment first so `python` is the oasis interpreter:
#     mamba activate oasis
#     ./process.sh                 # (or: bash process.sh)
#
# The number of training epochs is set in config.yaml (`epochs`). Override the Python
# interpreter with an env var if needed:
#     PYTHON=/path/to/python ./process.sh
#
# Steps 5 and 7 never touch TEST here -- that needs the deliberate, one-time `--reveal`
# flag (see their own docstrings), which this script does not pass.
set -euo pipefail
cd "$(dirname "$0")"

PYTHON="${PYTHON:-python}"

echo "== Step 1: gather metadata =="
$PYTHON step1-gather-metadata.py

echo "== Step 2: assign groups =="
$PYTHON step2-assign-groups.py

echo "== Step 3: generate slices =="
rm -rf outputs/train outputs/validate outputs/test   # drop stale slices from prior offsets
$PYTHON step3-generate-slices.py

echo "== Step 4: train the CNN (epochs from config.yaml) =="
$PYTHON step4-train-network.py

echo "== Step 5: plot the training curves =="
$PYTHON step5-plot-training.py

echo "== Step 6: Grad-CAM heatmaps =="
$PYTHON step6-gradcam.py

echo "== Step 7: age/nWBV/CNN predictor ablation =="
$PYTHON step7-stack-predictors.py

echo "== Done. See outputs/step5-training_comparison.png, step6-gradcam_grid.png, "
echo "   gradcam_context/, and outputs/step7-stacking_summary.txt =="
