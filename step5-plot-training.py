#!/usr/bin/env python3
"""Step 5: plot the training + validation curves of the step4 designs together.

Reads the per-design CSV logs written by step4a/4b/4c
(``outputs/training_log_4a.csv`` etc.) and overlays them in a 2x2 grid --
training loss, training accuracy, validation loss, validation accuracy, each vs
epoch -- so the designs can be compared. The validation-accuracy panel also draws a
running average (a smoothed line over the noisy raw curve) to show the trend. Saves
``outputs/training_comparison.png``. Also prints -- and saves to
``outputs/training_summary.txt`` -- each design's average validation accuracy /
sensitivity / specificity / balanced accuracy over the last 50 epochs, plus a breakdown of
validation accuracy by CDR grade (0.5 / 1 / 2, pooled under label 1). Designs that haven't
been run yet are simply skipped.

Usage:
    python step5-plot-training.py
"""
from __future__ import annotations

import csv
import os

from common import load_config, load_yaml, manifest_yaml

# CSV file (written by each step4x) -> legend label.
DESIGNS = [
    ("training_log_4a.csv", "4a: 8-16-32, dropout 0.6/0.2 (baseline)"),
    ("training_log_4b.csv", "4b: 8-16-32, no dropout"),
    ("training_log_4c.csv", "4c: wider 32-64-128, 5x5 first, dropout 0.6/0.2"),
    ("training_log_4d.csv", "4d: shallower 8-16 (2 blocks), dropout 0.4/0.2"),
]


NUMERIC_COLS = ("train_loss", "train_acc", "val_loss", "val_acc",
                "val_sens", "val_spec", "val_bal_acc",
                "val_acc_cdr05", "val_acc_cdr10", "val_acc_cdr20")

# CDR-positive grade -> (its log column, a human label)
GRADE_INFO = [
    (0.5, "val_acc_cdr05", "CDR 0.5 (very mild)"),
    (1.0, "val_acc_cdr10", "CDR 1 (mild)"),
    (2.0, "val_acc_cdr20", "CDR 2 (moderate)"),
]


def load_log(path):
    """Read a training_log CSV into a dict of column-name -> list of values.

    Tolerates older logs that lack the sens/spec/bal columns (they stay empty).
    """
    cols = {"epoch": []}
    cols.update({k: [] for k in NUMERIC_COLS})
    with open(path) as f:
        reader = csv.DictReader(f)
        have = set(reader.fieldnames or [])
        for row in reader:
            cols["epoch"].append(int(row["epoch"]))
            for k in NUMERIC_COLS:
                if k in have:
                    cols[k].append(float(row[k]))
    return cols


def tail_mean(values, n=50):
    """Mean of the last n values (fewer if the run was shorter)."""
    tail = values[-n:]
    return sum(tail) / len(tail) if tail else float("nan")


SMOOTH_WINDOW = 20   # running-average window (epochs) for the validation-accuracy plot


def running_mean(values, window=SMOOTH_WINDOW):
    """Trailing running average over `window` epochs; same length as `values`."""
    out = []
    for i in range(len(values)):
        seg = values[max(0, i - window + 1): i + 1]
        out.append(sum(seg) / len(seg))
    return out


def val_grade_counts(manifest_path):
    """Per CDR-positive grade, how many validation patches / subjects there are."""
    counts = {g: {"patches": 0, "subjects": set()} for g, _, _ in GRADE_INFO}
    if os.path.isfile(manifest_path):
        for r in load_yaml(manifest_path).get("slices", []):
            if r.get("split") == "validate" and r.get("cdr") in counts:
                counts[r["cdr"]]["patches"] += 1
                counts[r["cdr"]]["subjects"].add(r["subject"])
    return counts


def main():
    import matplotlib
    matplotlib.use("Agg")                # write a file, no interactive window
    import matplotlib.pyplot as plt

    config = load_config()
    outputs_path = config["outputs_path"]

    fig, ((ax_tl, ax_ta), (ax_va, ax_bal)) = plt.subplots(2, 2, figsize=(11, 9))
    logs = []
    for csv_name, label in DESIGNS:
        path = os.path.join(outputs_path, csv_name)
        if not os.path.isfile(path):
            print(f"  [skip] not found (run its step4 script first): {csv_name}")
            continue
        d = load_log(path)
        ax_tl.plot(d["epoch"], d["train_loss"], alpha=0.6, linewidth=1, label=label)
        ax_ta.plot(d["epoch"], d["train_acc"], alpha=0.6, linewidth=1, label=label)
        # Validation accuracy and balanced accuracy: faint raw + bold running mean.
        for ax, key in ((ax_va, "val_acc"), (ax_bal, "val_bal_acc")):
            raw, = ax.plot(d["epoch"], d[key], alpha=0.18, linewidth=1)       # faint raw
            ax.plot(d["epoch"], running_mean(d[key]), color=raw.get_color(),
                    alpha=0.9, linewidth=1.7, label=label)                    # smoothed
        logs.append((label, d))

    if not logs:
        raise SystemExit("No training_log_4*.csv found -- run step4a/4b/4c first.")

    ax_tl.set(xlabel="epoch", ylabel="training loss", title="Training loss")
    for ax, ylabel, title in (
            (ax_ta, "accuracy", "Training accuracy"),
            (ax_va, "accuracy", f"Validation accuracy ({SMOOTH_WINDOW}-epoch running avg)"),
            (ax_bal, "balanced accuracy",
             f"Validation balanced accuracy ({SMOOTH_WINDOW}-epoch running avg)")):
        ax.axhline(0.5, color="gray", linestyle="--", label="chance")
        ax.set(xlabel="epoch", ylabel=ylabel, title=title, ylim=(0, 1.02))
    for ax in (ax_tl, ax_ta, ax_va, ax_bal):
        ax.legend()
    fig.tight_layout()

    out = os.path.join(outputs_path, "training_comparison.png")
    fig.savefig(out, dpi=120)
    print(f"Saved comparison plot ({len(logs)} designs) -> {out}")

    # Text summary: average validation metrics over the last N epochs, per design. Every line
    # is both printed to the console and collected (via emit) so it can be saved to a file.
    n_last = config.get("avg_last_epochs", 50)
    summary_lines = []

    def emit(line=""):
        print(line)
        summary_lines.append(line)

    emit()
    emit(f"Average validation metrics over the last {n_last} epochs:")
    emit(f"  {'design':<28} {'acc':>7} {'sens':>7} {'spec':>7} {'bal_acc':>8}")
    for label, d in logs:
        acc = tail_mean(d["val_acc"], n_last)
        sens = tail_mean(d["val_sens"], n_last)
        spec = tail_mean(d["val_spec"], n_last)
        bal = tail_mean(d["val_bal_acc"], n_last)
        emit(f"  {label:<28} {acc:>7.3f} {sens:>7.3f} {spec:>7.3f} {bal:>8.3f}")

    # Breakdown of validation accuracy by CDR grade (0.5/1/2, pooled under label 1),
    # mean of the last N epochs. The validation counts are tiny, so read these as
    # trends, not precise numbers.
    counts = val_grade_counts(manifest_yaml(config))
    emit()
    emit(f"Validation accuracy by CDR grade (mean of last {n_last} epochs):")
    for g, _, name in GRADE_INFO:
        c = counts[g]
        emit(f"  {name:<20} {c['patches']:>3} patches from {len(c['subjects'])} subject(s)")
    emit(f"  {'design':<28} {'CDR0.5':>7} {'CDR1':>7} {'CDR2':>7}")
    for label, d in logs:
        cells = []
        for g, col, _ in GRADE_INFO:
            m = tail_mean(d.get(col, []), n_last)
            cells.append(f"{'n/a':>7}" if m != m else f"{m:>7.3f}")   # m != m -> nan
        emit(f"  {label:<28} {cells[0]} {cells[1]} {cells[2]}")

    # Save the same summary to a text file alongside the plot.
    summary_path = os.path.join(outputs_path, "training_summary.txt")
    with open(summary_path, "w") as f:
        f.write("\n".join(summary_lines).lstrip("\n") + "\n")
    print(f"\nSaved summary -> {summary_path}")


if __name__ == '__main__':
    main()
