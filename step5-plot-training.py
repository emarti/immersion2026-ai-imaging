#!/usr/bin/env python3
"""Step 5: plot the training + validation curves of the step4 designs together.

Reads the per-design CSV logs written by step4a/4b/4c/4d
(``outputs/training_log_4a.csv`` etc.) and overlays them in a 2x3 grid -- training loss,
training accuracy, validation loss, validation accuracy, validation balanced accuracy and
validation AUC, each vs epoch -- so the designs can be compared. The four validation-rate
panels also draw a running average (a smoothed line over the noisy raw curve) to show the
trend. Saves ``outputs/step5-training_comparison.png``. Also prints -- and saves to
``outputs/step5-training_summary.txt`` -- each design's average validation accuracy /
sensitivity / specificity / balanced accuracy / AUC over the last N epochs, a breakdown of
validation accuracy by CDR grade (0.5 / 1 / 2, pooled under label 1), and a labelled confusion
matrix (rows = actual, columns = predicted, with totals) from the final checkpoint, so the
rate numbers above it can be checked by hand -- at the default decision threshold (prob > 0.5)
and again at a threshold calibrated on TRAIN for sensitivity ~= specificity (the Equal Error
Rate point), so the effect of threshold-moving is visible directly. Designs that haven't been
run yet are simply skipped.

It then writes two figures that need the model itself rather than the logs, so both re-run the
saved ``outputs/model_4?.pt`` checkpoints over the validation patches:

* ``outputs/step5-logit_by_grade.png`` -- each design's raw (pre-sigmoid) logit output as a
  dot/violin column per CDR grade (0 / 0.5 / 1 / 2). The networks are trained on the BINARY
  label, so nothing ever tells them 2 is worse than 0.5; this checks whether their confidence
  tracks severity anyway.
* ``outputs/step5-roc_curve.png`` -- all designs' ROC curves overlaid, with the area under each
  in the legend. The AUC panel in the comparison grid only has the per-epoch scalar from the
  CSV; a curve needs the raw per-patch predictions.

Usage:
    python step5-plot-training.py
"""
from __future__ import annotations

import csv
import importlib.util
import os

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader
from torchvision import transforms

from common import load_config, load_yaml, manifest_yaml

# design letter -> (CSV file written by that step4x, legend label).
DESIGNS = [
    ("a", "training_log_4a.csv", "4a: 8-16-32, dropout 0.6/0.2 (baseline)"),
    ("b", "training_log_4b.csv", "4b: 8-16-32, no dropout"),
    ("c", "training_log_4c.csv", "4c: wider 32-64-128, 5x5 first, dropout 0.6/0.2"),
    ("d", "training_log_4d.csv", "4d: shallower 8-16 (2 blocks), dropout 0.4/0.2"),
]


NUMERIC_COLS = ("train_loss", "train_acc", "val_loss", "val_acc",
                "val_sens", "val_spec", "val_bal_acc", "val_auc",
                "val_acc_cdr05", "val_acc_cdr10", "val_acc_cdr20")

# CDR-positive grade -> (its log column, a human label)
GRADE_INFO = [
    (0.5, "val_acc_cdr05", "CDR 0.5 (very mild)"),
    (1.0, "val_acc_cdr10", "CDR 1 (mild)"),
    (2.0, "val_acc_cdr20", "CDR 2 (moderate)"),
]

# All FOUR CDR grades, including CDR-negative -- used by the logit-by-grade figure below,
# which (unlike GRADE_INFO's accuracy breakdown) needs the negative class too. In the
# manifest a CDR-negative patch carries cdr == 0.0, not a blank, so all four groups come
# from that one field.
CDR_GROUPS = (0.0, 0.5, 1.0, 2.0)
MIN_VIOLIN_N = 10   # below this many points, dots only -- a violin drawn from a handful of
                    # values implies a distribution shape the data doesn't actually support


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


def load_step4(design: str):
    """Import ``step4{design}-train-network.py`` as a module, the way step6 does.

    Everything the logit figure needs -- ``Net``, the ``OASISSlices`` Dataset and the
    ``MODEL_NAME`` of the checkpoint -- already lives in that file, so nothing is copied.
    """
    fname = f"step4{design}-train-network.py"
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), fname)
    if not os.path.isfile(path):
        raise SystemExit(f"No such design script: {fname}")
    spec = importlib.util.spec_from_file_location(f"step4{design}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_model_and_loader(config, outputs_path, design, split):
    """One design's trained checkpoint plus a DataLoader over ``split``.

    Shared by the two per-patch figures below (logit-by-grade and the ROC curve) -- both
    need a freshly-run trained model, unlike the rest of this script, which only reads the
    CSV logs. Returns ``(None, None)`` if that design has not been trained yet.
    """
    step4 = load_step4(design)
    model_path = os.path.join(outputs_path, step4.MODEL_NAME)
    if not os.path.isfile(model_path):
        return None, None

    model = step4.Net()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    manifest = load_yaml(manifest_yaml(config))
    ds = step4.OASISSlices(manifest, outputs_path, split, transforms.ToTensor())
    return model, DataLoader(ds, batch_size=32)


def predict_logits(config, outputs_path, design, split):
    """Per-patch RAW logits and CDR grades for one design on one split.

    Returns ``(logits, cdrs)`` -- the model's output *before* the sigmoid, not the
    probability, because the logit is what the decision rule actually thresholds
    (``pred = output > 0``) and it spreads the values out instead of squashing them
    toward 0 and 1. Returns ``(None, None)`` if that design has no checkpoint yet.
    """
    model, loader = load_model_and_loader(config, outputs_path, design, split)
    if model is None:
        return None, None
    logits, cdrs = [], []
    with torch.no_grad():
        for data, _, cdr in loader:            # the label is unused; the GRADE is the point
            logits.append(model(data))
            cdrs.append(cdr)
    return torch.cat(logits).numpy(), torch.cat(cdrs).numpy()


def predict_probs(config, outputs_path, design, split):
    """Per-patch PROBABILITIES and true labels for one design on one split.

    The sigmoid twin of ``predict_logits``, for the ROC curve. ROC is rank-based, so the
    raw logits would give an identical curve -- probabilities are used because the ROC
    axes are conventionally read that way and the thresholds it sweeps are then readable
    as "predict CDR-positive above p". Returns ``(None, None)`` if there is no checkpoint.
    """
    model, loader = load_model_and_loader(config, outputs_path, design, split)
    if model is None:
        return None, None
    probs, targets = [], []
    with torch.no_grad():
        for data, target, _ in loader:         # the CDR grade is unused; the LABEL is
            probs.append(torch.sigmoid(model(data)))
            targets.append(target)
    return torch.cat(probs).numpy(), torch.cat(targets).numpy()


def plot_logit_by_grade(ax, logits, cdrs, color, seed=0):
    """Vertical dot plot of raw logit output, one column per CDR grade.

    ALWAYS draws jittered dots (one point per patch) plus a mean +/- SD bar; additionally
    draws a violin body for any grade with at least MIN_VIOLIN_N points. Dots-and-violin is
    the usual preference in biology over a histogram, and the point threshold stops a
    handful of values being smoothed into a shape the data cannot support.
    """
    rng = np.random.default_rng(seed)
    violin_data, violin_pos = [], []
    for i, g in enumerate(CDR_GROUPS):
        vals = logits[cdrs == g]
        if len(vals) == 0:
            continue                            # grade absent from this split
        # The x position is the SLOT INDEX i, not the CDR value: the grades are categorical
        # labels, and plotting them at 0 / 0.5 / 1 / 2 would squash the 0 -> 0.5 gap to a
        # quarter of the 1 -> 2 one, implying a spacing the scale doesn't have.
        jitter = rng.uniform(-0.15, 0.15, size=len(vals))
        ax.scatter(np.full(len(vals), i) + jitter, vals, s=18, color=color, alpha=0.6,
                   edgecolor="white", linewidth=0.4, zorder=3)
        ax.errorbar(i, vals.mean(), yerr=vals.std(ddof=1) if len(vals) > 1 else 0,
                    fmt="_", color="black", markersize=20, capsize=6, elinewidth=1.5,
                    zorder=4)
        if len(vals) >= MIN_VIOLIN_N:
            violin_data.append(vals)
            violin_pos.append(i)
    if violin_data:
        parts = ax.violinplot(violin_data, positions=violin_pos, widths=0.6,
                              showextrema=False)
        for body in parts["bodies"]:
            body.set_facecolor(color)
            body.set_alpha(0.25)
            body.set_zorder(1)

    # The real decision rule, the same one step4's evaluate() uses (`pred = output > 0`,
    # which is exactly p > 0.5): above this line the model says CDR-positive, below it
    # CDR-negative. Labelled via the LEGEND rather than text pinned outside the axes,
    # because "logit > 0 means positive" is not obvious on sight and text placed outside
    # the axes collides with the neighbouring panel.
    ax.axhline(0, color="gray", linestyle="--", linewidth=1.2, zorder=2,
               label="decision line: CDR+ above, CDR- below")
    ax.legend(fontsize=7, loc="upper left")
    # All four slots are reserved even when a grade is missing, so the panels stay aligned.
    ax.set_xticks(range(len(CDR_GROUPS)))
    ax.set_xticklabels([f"{g:g}" for g in CDR_GROUPS])
    ax.set_xlim(-0.6, len(CDR_GROUPS) - 0.4)
    ax.set_xlabel("CDR grade")
    ax.set_ylabel("Raw logit output")


SMOOTH_WINDOW = 20   # running-average window (epochs) for the validation-accuracy plot


def running_mean(values, window=SMOOTH_WINDOW):
    """Trailing running average over `window` epochs; same length as `values`."""
    out = []
    for i in range(len(values)):
        seg = values[max(0, i - window + 1): i + 1]
        out.append(sum(seg) / len(seg))
    return out


def emit_confusion_matrix(emit, label, targets, probs, threshold, threshold_note):
    """Print one design's confusion matrix -- rows = actual, columns = predicted -- with
    row/column totals, so the accuracy/sensitivity/specificity numbers above can be checked
    by hand (e.g. accuracy = (tp+tn) / grand total).

    Predicted CDR-positive is ``prob > threshold`` -- at the default ``threshold=0.5`` this is
    the same rule step4's ``evaluate()`` uses (``logit > 0`` <=> ``prob > 0.5``). These counts
    come from ONE pass over validation with the FINAL checkpoint, so they will not exactly
    match the tail-mean rates in the table above (those average over the last ``n_last``
    epochs of training) -- both are correct, they just measure different points in training.
    """
    preds = (probs > threshold).astype(int)
    targets = targets.astype(int)
    tn = int(((preds == 0) & (targets == 0)).sum())
    fp = int(((preds == 1) & (targets == 0)).sum())
    fn = int(((preds == 0) & (targets == 1)).sum())
    tp = int(((preds == 1) & (targets == 1)).sum())

    emit(f"  {label}  [threshold = {threshold:.3f}, {threshold_note}]")
    emit(f"  {'':<16}{'pred CDR-':>12}{'pred CDR+':>12}{'total':>9}")
    emit(f"  {'actual CDR-':<16}{tn:>12}{fp:>12}{tn + fp:>9}")
    emit(f"  {'actual CDR+':<16}{fn:>12}{tp:>12}{fn + tp:>9}")
    emit(f"  {'total':<16}{tn + fn:>12}{fp + tp:>12}{tn + fp + fn + tp:>9}")
    emit()


def equal_error_rate_threshold(targets, probs):
    """The probability cutoff where sensitivity == specificity -- the point where the ROC
    curve crosses the anti-diagonal from (0,1) to (1,0), a.k.a. the Equal Error Rate (EER)
    point. (This is a different "optimal threshold" than Youden's J statistic, which instead
    maximizes sensitivity + specificity - 1, i.e. the point farthest ABOVE the chance
    diagonal; the two coincide only when the ROC curve happens to be symmetric.)

    sensitivity == specificity  <=>  tpr == 1 - fpr  <=>  tpr + fpr == 1, so this picks
    whichever ROC point minimizes |tpr + fpr - 1| -- the closest the curve actually comes to
    that crossing, since ROC points are discrete (one per distinct score in ``probs``).
    """
    fpr, tpr, thresholds = roc_curve(targets, probs)
    i = int(np.argmin(np.abs(tpr + fpr - 1)))
    return float(thresholds[i])


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

    fig, ((ax_tl, ax_ta, ax_vl),
          (ax_va, ax_bal, ax_auc)) = plt.subplots(2, 3, figsize=(15, 9))
    logs = []
    for _design, csv_name, label in DESIGNS:
        path = os.path.join(outputs_path, csv_name)
        if not os.path.isfile(path):
            print(f"  [skip] not found (run its step4 script first): {csv_name}")
            continue
        d = load_log(path)

        # A log written before a column existed (e.g. val_auc) loads as an empty list, which
        # would not line up with the epoch axis -- so skip any series that is short.
        def has(key):
            return len(d[key]) == len(d["epoch"])

        # The two loss panels: plain lines, no chance level, no 0-1 axis.
        for ax, key in ((ax_tl, "train_loss"), (ax_vl, "val_loss")):
            if has(key):
                ax.plot(d["epoch"], d[key], alpha=0.6, linewidth=1, label=label)
        if has("train_acc"):
            ax_ta.plot(d["epoch"], d["train_acc"], alpha=0.6, linewidth=1, label=label)
        # The validation rate metrics: faint raw curve + bold running mean.
        for ax, key in ((ax_va, "val_acc"), (ax_bal, "val_bal_acc"), (ax_auc, "val_auc")):
            if not has(key):
                continue
            raw, = ax.plot(d["epoch"], d[key], alpha=0.18, linewidth=1)       # faint raw
            ax.plot(d["epoch"], running_mean(d[key]), color=raw.get_color(),
                    alpha=0.9, linewidth=1.7, label=label)                    # smoothed
        logs.append((label, d))

    if not logs:
        raise SystemExit("No training_log_4*.csv found -- run step4a/4b/4c first.")

    # Losses: train and validation side by side is the clearest view of overfitting.
    ax_tl.set(xlabel="epoch", ylabel="training loss", title="Training loss")
    ax_vl.set(xlabel="epoch", ylabel="validation loss", title="Validation loss")
    # Rate metrics: all share the 0-1 axis and a chance line, so they can be read together.
    for ax, ylabel, title in (
            (ax_ta, "accuracy", "Training accuracy"),
            (ax_va, "accuracy", f"Validation accuracy ({SMOOTH_WINDOW}-epoch running avg)"),
            (ax_bal, "balanced accuracy",
             f"Validation balanced accuracy ({SMOOTH_WINDOW}-epoch running avg)"),
            (ax_auc, "AUC", f"Validation AUC ({SMOOTH_WINDOW}-epoch running avg)")):
        ax.axhline(0.5, color="gray", linestyle="--", label="chance")
        ax.set(xlabel="epoch", ylabel=ylabel, title=title, ylim=(0, 1.02))
    for ax in (ax_tl, ax_ta, ax_vl, ax_va, ax_bal, ax_auc):
        ax.legend()
    fig.tight_layout()

    out = os.path.join(outputs_path, "step5-training_comparison.png")
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
    emit(f"  {'design':<28} {'acc':>7} {'sens':>7} {'spec':>7} {'bal_acc':>8} {'auc':>7}")
    for label, d in logs:
        acc = tail_mean(d["val_acc"], n_last)
        sens = tail_mean(d["val_sens"], n_last)
        spec = tail_mean(d["val_spec"], n_last)
        bal = tail_mean(d["val_bal_acc"], n_last)
        auc = tail_mean(d["val_auc"], n_last)          # nan for logs written before AUC
        auc_cell = f"{'n/a':>7}" if auc != auc else f"{auc:>7.3f}"   # x != x -> nan
        emit(f"  {label:<28} {acc:>7.3f} {sens:>7.3f} {spec:>7.3f} {bal:>8.3f} {auc_cell}")

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

    # Confusion matrix per design, from ONE pass over validation with the FINAL checkpoint --
    # cached here (design -> (probs, targets)) so the ROC curve section below reuses the same
    # predictions instead of re-running each checkpoint a second time.
    emit()
    emit("Confusion matrix (validation, final checkpoint):")
    probs_by_design = {}
    for design, _csv_name, label in DESIGNS:
        probs, targets = predict_probs(config, outputs_path, design, "validate")
        probs_by_design[design] = (probs, targets)
        if probs is None:
            continue
        emit_confusion_matrix(emit, label, targets, probs, 0.5, "default")

    # Same validation predictions, but at a different decision threshold: the Equal Error
    # Rate point (sensitivity ~= specificity) found on the TRAIN split, then applied here.
    # The threshold is chosen from train, not validation, on purpose -- picking it on
    # validation and then reporting sensitivity/specificity on that SAME validation data
    # would be leakage (the threshold gets to see the exact numbers it's being judged on),
    # even though it's one scalar rather than a retrained model. Fitting it on train and
    # only ever applying it to validation keeps the same discipline this project already
    # uses for test (introduction.md SS2): whatever you tune on must stay separate from
    # whatever you report on.
    emit("Confusion matrix, threshold calibrated on TRAIN for sensitivity ~= specificity"
         " (applied to the same validation predictions above):")
    for design, _csv_name, label in DESIGNS:
        probs, targets = probs_by_design.get(design, (None, None))
        if probs is None:
            continue
        train_probs, train_targets = predict_probs(config, outputs_path, design, "train")
        if train_probs is None or len(set(train_targets.tolist())) < 2:
            continue                             # can't find a sens=spec point from one class
        t = equal_error_rate_threshold(train_targets, train_probs)
        emit_confusion_matrix(emit, label, targets, probs, t, "train EER")

    # Save the same summary to a text file alongside the plot.
    summary_path = os.path.join(outputs_path, "step5-training_summary.txt")
    with open(summary_path, "w") as f:
        f.write("\n".join(summary_lines).lstrip("\n") + "\n")
    print(f"\nSaved summary -> {summary_path}")

    # ------------------------------------------------------------------
    # Raw logit output by CDR grade -- its own figure.
    #
    # The networks are trained on a BINARY label: every CDR 0.5, 1 and 2 patch is simply
    # "CDR-positive", and nothing in the loss ever tells them that 2 is worse than 0.5. So
    # this asks a question the training objective never posed: does the model's confidence
    # nonetheless line up with severity, 0 < 0.5 < 1 < 2? If it does, the network has picked
    # up something graded from the images rather than just a two-way split. Unlike the rest
    # of this script -- which only reads the CSV logs -- this re-runs each trained
    # checkpoint over the validation patches to get per-patch numbers.
    #
    # Validation only: the test set stays untouched until the very end (introduction.md §2).
    # ------------------------------------------------------------------
    print()
    panels = []
    for design, _csv_name, label in DESIGNS:
        logits, cdrs = predict_logits(config, outputs_path, design, "validate")
        if logits is None:
            print(f"  [skip] no checkpoint for design 4{design} "
                  f"(run step4{design}-train-network.py first)")
            continue
        panels.append((label, logits, cdrs))

    if panels:
        ncols = 2
        nrows = (len(panels) + ncols - 1) // ncols
        fig_lg, axes_lg = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.6 * nrows),
                                       squeeze=False, sharey=True)
        flat = [ax for row in axes_lg for ax in row]
        colours = ["#4C72B0", "#C44E52", "#55A868", "#8172B2"]
        for ax, (label, logits, cdrs), colour in zip(flat, panels, colours):
            plot_logit_by_grade(ax, logits, cdrs, colour)
            ax.set_title(label, fontsize=9)
        for ax in flat[len(panels):]:
            ax.axis("off")                       # unused slot when fewer than 4 designs ran
        fig_lg.suptitle("Raw logit output by CDR grade (validation) — "
                        "training only ever saw the binary label")
        fig_lg.tight_layout()

        logit_path = os.path.join(outputs_path, "step5-logit_by_grade.png")
        fig_lg.savefig(logit_path, dpi=120)
        print(f"Saved logit-by-grade plot ({len(panels)} designs) -> {logit_path}")

    # ------------------------------------------------------------------
    # ROC curve -- its own figure, all designs overlaid.
    #
    # The "Validation AUC" panel above only has the per-epoch SCALAR from the CSV; a curve
    # needs the raw per-patch predictions -- reuses the (probs, targets) pass already cached
    # above for the confusion matrices, rather than re-running each checkpoint again. The
    # curve traces what happens as the decision threshold sweeps from "call everything
    # CDR-negative" (bottom left) to "call everything CDR-positive" (top right); AUC is the
    # area underneath, and the dashed diagonal is a coin flip.
    #
    # NOTE: the AUC in this legend comes from that SINGLE pass over the final checkpoint,
    # so it will not exactly match the `auc` column in the summary above -- that one is a
    # tail mean over the last n_last epochs of training. Both are correct; they measure
    # different things.
    # ------------------------------------------------------------------
    curves = []
    for design, _csv_name, label in DESIGNS:
        probs, targets = probs_by_design.get(design, (None, None))
        if probs is None or len(set(targets.tolist())) < 2:
            continue                             # untrained, or a split with only one class
        curves.append((label, probs, targets))

    if curves:
        fig_roc, ax_roc = plt.subplots(figsize=(6, 6))
        for label, probs, targets in curves:
            fpr, tpr, _ = roc_curve(targets, probs)
            ax_roc.plot(fpr, tpr, linewidth=2,
                        label=f"{label} (AUC={roc_auc_score(targets, probs):.3f})")
        ax_roc.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1,
                    label="chance")
        ax_roc.set(xlabel="False positive rate", ylabel="True positive rate",
                   title="ROC curve on validation (final checkpoints)",
                   xlim=(0, 1), ylim=(0, 1.02))
        ax_roc.legend(fontsize=7, loc="lower right")
        fig_roc.tight_layout()

        roc_path = os.path.join(outputs_path, "step5-roc_curve.png")
        fig_roc.savefig(roc_path, dpi=120)
        print(f"Saved ROC curve plot ({len(curves)} designs) -> {roc_path}")


if __name__ == '__main__':
    main()
