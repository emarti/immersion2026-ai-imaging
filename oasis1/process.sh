#!/usr/bin/env bash
#
# Run the whole OASIS pipeline end to end:
#   metadata -> cohort -> slices -> train the four CNN designs -> compare -> Grad-CAM.
#
# Activate the environment first so `python` is the oasis interpreter:
#     mamba activate oasis
#     ./process.sh                 # (or: bash process.sh)
#
# The number of training epochs is set in config.yaml (`epochs`). Override the Python
# interpreter with an env var if needed:
#     PYTHON=/path/to/python ./process.sh
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

echo "== Step 4: train the four CNN designs (epochs from config.yaml) =="
for design in a b c d; do
    echo "-- step4${design} --"
    $PYTHON "step4${design}-train-network.py"
done

echo "== Step 5: plot the comparison =="
$PYTHON step5-plot-training.py

echo "== Step 6: Grad-CAM heatmaps (baseline design 4a) =="
$PYTHON step6-gradcam.py

echo "== Done. See outputs/training_comparison.png and outputs/gradcam_grid.png =="
