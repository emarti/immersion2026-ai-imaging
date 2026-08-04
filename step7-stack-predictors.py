#!/usr/bin/env python3
"""Step 7: age + nWBV + CNN -- a 5-way predictor ablation (standalone, optional extra).

Answers: does the CNN's hippocampus-patch prediction add anything over two much simpler
numbers -- age and nWBV (normalized whole-brain volume, from the OASIS reference
spreadsheet)? Five predictors are compared, each a logistic regression (z-scored inputs,
fit on TRAIN):

  1. age only
  2. nWBV only
  3. CNN only -- one combined image-based score per subject (mean raw logit over every
     patch belonging to that subject in a split; both sides pooled, and for train, every
     plane/shift pooled too -- not separate left/right features)
  4. age + nWBV
  5. age + nWBV + CNN

That's 5 of the 2**3 - 1 = 7 possible non-empty feature subsets -- age+CNN and nWBV+CNN
are deliberately skipped to bound how many comparisons get made at once (see below).

For each predictor, AUC and balanced accuracy are reported on both TRAIN (in-sample fit
quality) and VALIDATE (held-out generalization), each with a 95% bootstrap confidence
interval (percentile bootstrap over subjects, ~2000 resamples -- the same pattern used in
``internal/jupyter/nWBV.ipynb``). The balanced-accuracy threshold is chosen by sweeping
TRAIN's own predicted probabilities for the best TRAIN balanced accuracy, then the SAME
threshold is applied to VALIDATE -- so validate's balanced accuracy isn't silently
re-optimized on validate itself.

TEST IS NOT TOUCHED BY DEFAULT, ON PURPOSE. Five comparisons already carries real
multiple-hypothesis-testing risk (exactly why 5 of 7 possible subsets were chosen, not
all 7); spending test on top of that would make any single "winning" predictor's
held-out number unreliable. Pass ``--reveal`` to additionally evaluate TEST (using the
SAME train-fit predictors and train-chosen thresholds, never re-fit or re-tuned on
test) -- meant to be used ONCE, deliberately, after you're done comparing on validate,
not as a routine flag you leave on.

Needs a trained checkpoint from step4 (``outputs/model_4.pt``) and the OASIS reference
spreadsheet (for nWBV -- unlike step1's non-fatal cross-check, this script cannot
proceed without it). Writes ``outputs/step7-stacking_summary.txt`` and
``outputs/step7-stacking_comparison.png`` (the AUC/balanced-accuracy ablation above, plus
a third quantity, "bits of information" -- a cross-entropy-based LOWER BOUND on mutual
information with CDR status, letting each predictor's contribution be read in
information-theoretic units instead of only AUC/accuracy; see the text summary for the
full derivation and caveats -- small-sample confidence intervals apply here too, likely
even more so), ``outputs/step7-predictor_correlations.png`` (four scatterplots -- see
below), and ``outputs/step7-roc_curves.png`` (all 5 predictors' ROC curves overlaid, one
panel per split -- the curve view behind the single AUC number in the table above).

As a side note unrelated to the age/nWBV/CNN ablation above: the ablation table treats age,
nWBV, and CNN as three separate signals, but never actually shows whether they *are*
separate. ``step7-predictor_correlations.png`` checks that, with one Pearson correlation
per split in each panel:

  1. Left vs right hippocampus CNN logit (before the two sides are pooled into the single
     CNN predictor). Strong agreement suggests the CNN is picking up a genuinely bilateral
     signal; weak agreement suggests at least one side is closer to noise, which the
     pooled mean would then be diluted by, not helped by.
  2. Age vs nWBV. Brain volume is known to decline with age on its own, so some
     correlation here is expected regardless of CDR status -- a reminder that
     "age+nWBV beats age alone" (if it does) may partly be nWBV re-deriving part of what
     age already said, not adding independent information.
  3. CNN (pooled left+right) vs age. If the CNN's score already tracks age closely, its
     apparent edge over age-alone in the ablation table is worth double-checking -- it may
     be learning age-related anatomy rather than something CDR-specific.
  4. CNN (pooled left+right) vs nWBV, for the same reason as #3.

Panels 2-4 do NOT use age/nWBV/CNN in their raw units. Each is passed through its own
single-variable logistic regression first (the same "age"/"nWBV"/"CNN" rows already in
the ablation table above) and plotted on that model's pre-sigmoid log-odds scale --
putting all three on one common, directly comparable footing, and making the fitted
direction of each effect explicit. This does NOT change how correlated two variables
look: since z-scoring and then applying decision_function = coef * z + intercept is just
an affine transform of the raw variable, the Pearson |r| here is identical to correlating
the raw z-scored variables directly. The one thing that CAN change is the *sign* -- e.g.
nWBV's fitted coefficient is expected to be negative (lower nWBV -> more CDR+), so a
positive raw age-nWBV correlation can show up as a *negative* correlation once nWBV is
expressed as its own predictor's log-odds of CDR+.

Usage:
    python step7-stack-predictors.py [--model model_4.pt] [--seed 0]
    python step7-stack-predictors.py --reveal      # also evaluates TEST -- use once
"""
from __future__ import annotations

import argparse
import importlib.util
import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from torchvision import transforms

from common import load_config, load_yaml, manifest_yaml, reference_xlsx, splits_yaml

N_BOOT = 2000   # bootstrap resamples for each confidence interval

# name -> feature columns. Deliberately 5 of 7 possible non-empty subsets of
# {age, nwbv, cnn} -- age+cnn and nwbv+cnn are skipped (see module docstring).
PREDICTORS = [
    ("age", ["age"]),
    ("nWBV", ["nwbv"]),
    ("CNN", ["cnn"]),
    ("age+nWBV", ["age", "nwbv"]),
    ("age+nWBV+CNN", ["age", "nwbv", "cnn"]),
]


def load_net():
    """Import the ``Net`` class from step4-train-network.py (no architecture copy)."""
    fname = "step4-train-network.py"
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), fname)
    if not os.path.isfile(path):
        raise SystemExit(f"No such script: {fname}")
    spec = importlib.util.spec_from_file_location("step4_train_network", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Net


def load_reference_table(config: dict) -> pd.DataFrame:
    """subject -> nWBV from the OASIS reference spreadsheet.

    Unlike step1's non-fatal cross-check, step7 cannot proceed without this -- nWBV is
    one of its predictors -- so a missing file is a hard error, not a silent skip.
    """
    xlsx = reference_xlsx(config)
    if not os.path.isfile(xlsx):
        raise SystemExit(
            f"Reference spreadsheet not found: {xlsx}\n"
            f"step7 needs it for nWBV -- see download-extract-data.sh / "
            f"readme.md 'Configure data paths'."
        )
    raw = pd.read_excel(xlsx)
    df = raw[raw["ID"].str.endswith("_MR1")].copy()   # drop the repeat scans
    df = df.dropna(subset=["nWBV"])
    return df.set_index("ID")[["nWBV"]]


def cnn_logits_by_subject_side(model, manifest, outputs_path, split, transform):
    """subject -> {"L": mean logit, "R": mean logit} over this split's patches, kept
    separate by side (for train, every plane/shift is still pooled within a side;
    validate/test have exactly one patch per side already, so their means are trivial).
    Used both for the left-vs-right correlation scatter and, pooled across sides below,
    for the main CNN predictor."""
    rows = [r for r in manifest["slices"] if r["split"] == split]
    sums: dict[tuple[str, str], float] = {}
    counts: dict[tuple[str, str], int] = {}
    model.eval()
    with torch.no_grad():
        for r in rows:
            img = Image.open(os.path.join(outputs_path, r["png_path"]))
            x = transform(img)[None]                              # 1 x 1 x H x W
            logit = float(model(x).squeeze().item())
            key = (r["subject"], r["side"])
            sums[key] = sums.get(key, 0.0) + logit
            counts[key] = counts.get(key, 0) + 1
    by_subject: dict[str, dict[str, float]] = {}
    for (subj, side), total in sums.items():
        by_subject.setdefault(subj, {})[side] = total / counts[(subj, side)]
    return by_subject


def pool_sides(by_subject_side: dict) -> dict:
    """{"L": .., "R": ..} per subject -> one both-sides-pooled mean logit per subject
    (what the age/nWBV/CNN predictor table above actually uses)."""
    return {s: sum(sides.values()) / len(sides) for s, sides in by_subject_side.items()}


def lr_arrays(by_subject_side: dict):
    """Paired (left, right) logit arrays, one entry per subject that has both sides
    (should be every subject -- step3 always emits one L and one R per plane/shift)."""
    subjects = [s for s, sides in by_subject_side.items() if "L" in sides and "R" in sides]
    left = np.array([by_subject_side[s]["L"] for s in subjects])
    right = np.array([by_subject_side[s]["R"] for s in subjects])
    return left, right


def pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    """NaN if fewer than 2 points -- a correlation isn't defined for a single subject."""
    return float(np.corrcoef(x, y)[0, 1]) if len(x) >= 2 else float("nan")


def cross_entropy_bits(y_true: np.ndarray, p: np.ndarray) -> float:
    """Average cross-entropy of predicted probabilities ``p`` against ``y_true``, in
    bits (log base 2). Clipped away from exactly 0/1 so log2 never sees a zero."""
    eps = 1e-12
    p = np.clip(p, eps, 1 - eps)
    return float(-np.mean(y_true * np.log2(p) + (1 - y_true) * np.log2(1 - p)))


def bits_gained(y_true: np.ndarray, p: np.ndarray, baseline_ce: float) -> float:
    """How many fewer bits it takes to describe ``y_true`` using ``p`` instead of the
    fixed ``baseline_ce`` (the "always guess TRAIN's prevalence" cross-entropy). See the
    module docstring's "bits of information" section for why this is a LOWER BOUND on
    mutual information, not the true value."""
    return baseline_ce - cross_entropy_bits(y_true, p)


def bits_bootstrap_ci(y_true: np.ndarray, p: np.ndarray, baseline_ce: float, seed: int):
    """95% percentile bootstrap CI, over subjects, for bits_gained at a FIXED baseline
    (never re-derived per resample -- same discipline as bootstrap_ci's fixed threshold)."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    vals = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, n)
        vals.append(bits_gained(y_true[idx], p[idx], baseline_ce))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def build_feature_table(records, cnn_logits, ref):
    """``records``: this split's list of splits.yaml member dicts. Returns a DataFrame
    indexed by subject with columns age/nwbv/cnn/label, plus the subjects dropped for
    missing a CNN logit or an nWBV value (should be rare -- every train/validate subject
    has patches from step3, but nWBV depends on the subject appearing in the reference
    spreadsheet)."""
    rows = []
    dropped_cnn, dropped_nwbv = [], []
    for r in records:
        subj = r["subject"]
        if subj not in cnn_logits:
            dropped_cnn.append(subj)
            continue
        if subj not in ref.index:
            dropped_nwbv.append(subj)
            continue
        rows.append({
            "subject": subj,
            "age": float(r["age"]),
            "nwbv": float(ref.loc[subj, "nWBV"]),
            "cnn": cnn_logits[subj],
            "label": int(r["label"]),
        })
    df = pd.DataFrame(rows).set_index("subject")
    return df, dropped_cnn, dropped_nwbv


def fit_predictor(X_train: np.ndarray, y_train: np.ndarray, seed: int):
    """z-score X_train's columns, fit LogisticRegression(Xz -> y). Returns (mean, std,
    model) so the SAME z-transform (fit on TRAIN) applies to validate later -- refitting
    the z-score on validate would leak validate statistics into a train-only predictor."""
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0, ddof=1)
    std[std == 0] = 1.0
    clf = LogisticRegression(random_state=seed)
    clf.fit((X_train - mean) / std, y_train)
    return mean, std, clf


def predict_proba(mean, std, clf, X: np.ndarray) -> np.ndarray:
    return clf.predict_proba((X - mean) / std)[:, 1]


def best_threshold(y_true: np.ndarray, p: np.ndarray) -> float:
    """The probability threshold on TRAIN that maximizes TRAIN balanced accuracy."""
    best_bal, best_t = -1.0, 0.5
    for t in np.unique(p):
        pred = p >= t
        sens = (pred & y_true).sum() / y_true.sum() if y_true.any() else 0.0
        spec = ((~pred) & (~y_true)).sum() / (~y_true).sum() if (~y_true).any() else 0.0
        bal = (sens + spec) / 2
        if bal > best_bal:
            best_bal, best_t = bal, t
    return best_t


def auc_and_balanced_acc(y_true: np.ndarray, p: np.ndarray, threshold: float):
    pred = p >= threshold
    sens = (pred & y_true).sum() / y_true.sum() if y_true.any() else 0.0
    spec = ((~pred) & (~y_true)).sum() / (~y_true).sum() if (~y_true).any() else 0.0
    bal = (sens + spec) / 2
    auc = roc_auc_score(y_true, p) if len(set(y_true.tolist())) == 2 else float("nan")
    return auc, bal


def bootstrap_ci(y_true: np.ndarray, p: np.ndarray, threshold: float, seed: int):
    """95% percentile bootstrap CI, over subjects, for (AUC, balanced accuracy) at a
    FIXED threshold (never re-optimized per resample -- see auc_and_balanced_acc)."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    aucs, bals = [], []
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, n)
        yb, pb = y_true[idx], p[idx]
        if not yb.any() or yb.all():          # AUC undefined for a single-class resample
            continue
        a, b = auc_and_balanced_acc(yb, pb, threshold)
        aucs.append(a)
        bals.append(b)

    def ci(vals):
        return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) \
            if vals else (float("nan"), float("nan"))

    return ci(aucs), ci(bals)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--model", default=None,
                        help="checkpoint to use for the CNN predictor (default: model_4.pt)")
    parser.add_argument("--seed", type=int, default=0, help="random seed (default: 0)")
    parser.add_argument("--reveal", action="store_true",
                        help="also evaluate TEST (default: off -- see module docstring)")
    args = parser.parse_args()

    config = load_config(args.config)
    outputs_path = config["outputs_path"]
    model_name = args.model or "model_4.pt"

    splits = load_yaml(splits_yaml(config))
    manifest = load_yaml(manifest_yaml(config))
    ref = load_reference_table(config)

    model_path = os.path.join(outputs_path, model_name)
    if not os.path.isfile(model_path):
        raise SystemExit(f"Checkpoint not found: {model_path}\n"
                         f"Run `python step4-train-network.py` first.")
    Net = load_net()
    model = Net()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    transform = transforms.ToTensor()
    train_logits_lr = cnn_logits_by_subject_side(model, manifest, outputs_path, "train", transform)
    val_logits_lr = cnn_logits_by_subject_side(model, manifest, outputs_path, "validate", transform)
    train_logits = pool_sides(train_logits_lr)
    val_logits = pool_sides(val_logits_lr)

    train_df, train_drop_cnn, train_drop_nwbv = build_feature_table(
        splits["train"], train_logits, ref)
    val_df, val_drop_cnn, val_drop_nwbv = build_feature_table(
        splits["validate"], val_logits, ref)

    print(f"Train subjects: {len(train_df)} used, "
          f"{len(train_drop_cnn)} dropped (no CNN patches), "
          f"{len(train_drop_nwbv)} dropped (no nWBV in reference spreadsheet)")
    print(f"Validate subjects: {len(val_df)} used, "
          f"{len(val_drop_cnn)} dropped (no CNN patches), "
          f"{len(val_drop_nwbv)} dropped (no nWBV in reference spreadsheet)")

    y_train = train_df["label"].to_numpy().astype(bool)
    y_val = val_df["label"].to_numpy().astype(bool)

    test_df = None
    test_logits_lr = None
    if args.reveal:
        print("\n*** --reveal is on: TEST will be evaluated below. Use this once. ***")
        test_logits_lr = cnn_logits_by_subject_side(model, manifest, outputs_path, "test", transform)
        test_df, test_drop_cnn, test_drop_nwbv = build_feature_table(
            splits["test"], pool_sides(test_logits_lr), ref)
        print(f"Test subjects: {len(test_df)} used, "
              f"{len(test_drop_cnn)} dropped (no CNN patches), "
              f"{len(test_drop_nwbv)} dropped (no nWBV in reference spreadsheet)")
        y_test = test_df["label"].to_numpy().astype(bool)

    # "No better than TRAIN's own prevalence" baseline for the bits-of-information
    # diagnostic below -- fit on TRAIN (just its label mean), applied out-of-sample, so
    # the "no skill" reference point never leaks an eval split's own label distribution.
    baseline_p = float(y_train.mean())
    baseline_ce_tr = cross_entropy_bits(y_train, np.full(len(y_train), baseline_p))
    baseline_ce_va = cross_entropy_bits(y_val, np.full(len(y_val), baseline_p))
    if test_df is not None:
        baseline_ce_te = cross_entropy_bits(y_test, np.full(len(y_test), baseline_p))

    results = []
    logit_by_name = {}   # single-variable predictor name -> {"train": .., "validate": .., "test": ..}
    for name, cols in PREDICTORS:
        X_train = train_df[cols].to_numpy(dtype=float)
        X_val = val_df[cols].to_numpy(dtype=float)

        mean, std, clf = fit_predictor(X_train, y_train, args.seed)
        p_train = predict_proba(mean, std, clf, X_train)
        p_val = predict_proba(mean, std, clf, X_val)

        threshold = best_threshold(y_train, p_train)   # chosen on TRAIN only

        auc_tr, bal_tr = auc_and_balanced_acc(y_train, p_train, threshold)
        auc_va, bal_va = auc_and_balanced_acc(y_val, p_val, threshold)
        auc_tr_ci, bal_tr_ci = bootstrap_ci(y_train, p_train, threshold, args.seed)
        auc_va_ci, bal_va_ci = bootstrap_ci(y_val, p_val, threshold, args.seed + 1)

        bits_tr = bits_gained(y_train, p_train, baseline_ce_tr)
        bits_va = bits_gained(y_val, p_val, baseline_ce_va)
        bits_tr_ci = bits_bootstrap_ci(y_train, p_train, baseline_ce_tr, args.seed)
        bits_va_ci = bits_bootstrap_ci(y_val, p_val, baseline_ce_va, args.seed + 1)

        result = {
            "name": name, "threshold": threshold,
            "auc_tr": auc_tr, "auc_tr_ci": auc_tr_ci,
            "bal_tr": bal_tr, "bal_tr_ci": bal_tr_ci,
            "auc_va": auc_va, "auc_va_ci": auc_va_ci,
            "bal_va": bal_va, "bal_va_ci": bal_va_ci,
            "bits_tr": bits_tr, "bits_tr_ci": bits_tr_ci,
            "bits_va": bits_va, "bits_va_ci": bits_va_ci,
            # Kept for the ROC-curve figure below -- the raw predicted probabilities,
            # not just the point-threshold metrics derived from them.
            "p_tr": p_train, "p_va": p_val,
        }

        if test_df is not None:
            # SAME train-fit z-score/model and SAME train-chosen threshold as validate --
            # test is only ever applied to, never fit or tuned on.
            X_test = test_df[cols].to_numpy(dtype=float)
            p_test = predict_proba(mean, std, clf, X_test)
            auc_te, bal_te = auc_and_balanced_acc(y_test, p_test, threshold)
            auc_te_ci, bal_te_ci = bootstrap_ci(y_test, p_test, threshold, args.seed + 2)
            bits_te = bits_gained(y_test, p_test, baseline_ce_te)
            bits_te_ci = bits_bootstrap_ci(y_test, p_test, baseline_ce_te, args.seed + 2)
            result.update(auc_te=auc_te, auc_te_ci=auc_te_ci,
                          bal_te=bal_te, bal_te_ci=bal_te_ci,
                          bits_te=bits_te, bits_te_ci=bits_te_ci,
                          p_te=p_test)

        if name in ("age", "nWBV", "CNN"):
            # Pre-sigmoid log-odds from THIS single-variable model -- used by the
            # predictor-correlation panels below instead of the raw age/nWBV/CNN units
            # (see module docstring's "bits of information" / log-odds framing note).
            entry = {
                "train": clf.decision_function((X_train - mean) / std),
                "validate": clf.decision_function((X_val - mean) / std),
            }
            if test_df is not None:
                entry["test"] = clf.decision_function((X_test - mean) / std)
            logit_by_name[name] = entry

        results.append(result)

    # --- text summary ---
    summary_lines = []

    def emit(line=""):
        print(line)
        summary_lines.append(line)

    emit("Age + nWBV + CNN predictor ablation (step7)")
    if test_df is None:
        emit("TEST is not used here -- see the module docstring for why. Re-run with "
             "--reveal to also evaluate it (once, deliberately).")
    else:
        emit("*** --reveal was on: TEST results are included below. ***")
    emit()
    subj_line = f"Train subjects used: {len(train_df)}   Validate subjects used: {len(val_df)}"
    if test_df is not None:
        subj_line += f"   Test subjects used: {len(test_df)}"
    emit(subj_line)
    emit()

    def cell(point, ci):
        return f"{point:.3f} [{ci[0]:.3f},{ci[1]:.3f}]"

    cols = ["AUC (train)", "AUC (validate)", "bal.acc (train)", "bal.acc (validate)"]
    if test_df is not None:
        cols += ["AUC (test)", "bal.acc (test)"]
    emit("  {:<16}".format("predictor") + "".join(f"{c:>20}" for c in cols))
    for r in results:
        row = [cell(r["auc_tr"], r["auc_tr_ci"]), cell(r["auc_va"], r["auc_va_ci"]),
              cell(r["bal_tr"], r["bal_tr_ci"]), cell(r["bal_va"], r["bal_va_ci"])]
        if test_df is not None:
            row += [cell(r["auc_te"], r["auc_te_ci"]), cell(r["bal_te"], r["bal_te_ci"])]
        emit(f"  {r['name']:<16}" + "".join(f"{c:>20}" for c in row))
    emit()
    emit("(95% bootstrap confidence intervals in brackets, ~2000 resamples over subjects. "
         "The balanced-accuracy threshold for each predictor is chosen on TRAIN and reused "
         "unchanged on VALIDATE" + (" and TEST." if test_df is not None else "."))
    emit()

    emit("Bits of information about CDR status, relative to guessing TRAIN's own "
         "prevalence (baseline = 0 bits):")
    bits_cols = ["bits (train)", "bits (validate)"]
    if test_df is not None:
        bits_cols += ["bits (test)"]
    emit("  {:<16}".format("predictor") + "".join(f"{c:>20}" for c in bits_cols))
    for r in results:
        row = [cell(r["bits_tr"], r["bits_tr_ci"]), cell(r["bits_va"], r["bits_va_ci"])]
        if test_df is not None:
            row += [cell(r["bits_te"], r["bits_te_ci"])]
        emit(f"  {r['name']:<16}" + "".join(f"{c:>20}" for c in row))
    emit()
    emit("\"Bits\" = the reduction, in log-base-2 units, between the cross-entropy of "
         "always guessing TRAIN's own CDR+ prevalence and the cross-entropy of this "
         "predictor's actual probabilities -- both measured OUT OF SAMPLE (on the split "
         "named in the column), using a model fit only on TRAIN, exactly like the "
         "AUC/bal.acc numbers above. By Gibbs' inequality (cross-entropy is never "
         "smaller than the true conditional entropy), this is a LOWER BOUND on the true "
         "mutual information between this predictor and CDR status, not the value "
         "itself -- a poorly calibrated model understates it further. It is also not a "
         "clean per-variable attribution: e.g. \"age+nWBV+CNN\"'s bits minus "
         "\"age+nWBV\"'s bits is a meaningful \"what CNN adds on top of age+nWBV\" "
         "number, but not CNN's own independent information content. And like every "
         "other number here, treat the confidence interval as the headline, not the "
         "point estimate -- at this sample size, expect it to be wide and to overlap "
         "across several predictors.")
    emit()

    split_dfs = [("train", train_df), ("validate", val_df)]
    if test_df is not None:
        split_dfs.append(("test", test_df))

    lr_groups = [("train", train_logits_lr), ("validate", val_logits_lr)]
    if test_logits_lr is not None:
        lr_groups.append(("test", test_logits_lr))
    emit("Left vs right hippocampus CNN logit, per subject (before the two sides are "
         "pooled into the single CNN predictor above) -- Pearson correlation:")
    lr_corr = {}
    for split_name, logits_lr in lr_groups:
        left, right = lr_arrays(logits_lr)
        r = pearson_r(left, right)
        lr_corr[split_name] = (left, right, r)
        emit(f"  {split_name:<10} r = {r:.3f}  (n={len(left)} subjects)")
    emit()

    emit("Age vs nWBV vs CNN, per subject -- each expressed as ITS OWN single-variable "
         "predictor's log-odds (pre-sigmoid), not raw units. Pearson |r| is identical to "
         "correlating the raw z-scored variables directly; only the SIGN can differ, if "
         "a fitted coefficient is negative (e.g. nWBV's is expected to be, since lower "
         "nWBV predicts CDR+ -- see module docstring):")

    def logit_corr(name_a, name_b):
        out = {}
        for split_name in [s for s, _ in split_dfs]:
            a = logit_by_name[name_a][split_name]
            b = logit_by_name[name_b][split_name]
            out[split_name] = (a, b, pearson_r(a, b))
        return out

    age_nwbv_corr = logit_corr("age", "nWBV")
    emit("  age vs nWBV:")
    for split_name, (a, _, r) in age_nwbv_corr.items():
        emit(f"    {split_name:<10} r = {r:.3f}  (n={len(a)} subjects)")

    cnn_age_corr = logit_corr("CNN", "age")
    emit("  CNN vs age:")
    for split_name, (a, _, r) in cnn_age_corr.items():
        emit(f"    {split_name:<10} r = {r:.3f}  (n={len(a)} subjects)")

    cnn_nwbv_corr = logit_corr("CNN", "nWBV")
    emit("  CNN vs nWBV:")
    for split_name, (a, _, r) in cnn_nwbv_corr.items():
        emit(f"    {split_name:<10} r = {r:.3f}  (n={len(a)} subjects)")

    summary_path = os.path.join(outputs_path, "step7-stacking_summary.txt")
    with open(summary_path, "w") as f:
        f.write("\n".join(summary_lines) + "\n")
    print(f"\nSaved summary -> {summary_path}")

    # --- comparison plot ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [r["name"] for r in results]
    x = np.arange(len(names))
    groups = [("train", "tr", "0.6")]
    if test_df is not None:
        groups += [("validate", "va", "steelblue"), ("test", "te", "firebrick")]
    else:
        groups += [("validate", "va", "steelblue")]
    width = 0.8 / len(groups)

    fig, (ax_auc, ax_bal, ax_bits) = plt.subplots(1, 3, figsize=(19, 5))
    panels = ((ax_auc, "auc", "AUC", 0.5, (0, 1.02)),
              (ax_bal, "bal", "Balanced accuracy", 0.5, (0, 1.02)),
              (ax_bits, "bits", "Bits of information (lower bound)", 0.0, None))
    for ax, key, title, baseline, ylim in panels:
        for i, (glabel, gkey, gcolor) in enumerate(groups):
            vals = np.array([r[f"{key}_{gkey}"] for r in results])
            ci = np.array([r[f"{key}_{gkey}_ci"] for r in results])
            err = np.abs(np.stack([vals - ci[:, 0], ci[:, 1] - vals], axis=0))
            offset = (i - (len(groups) - 1) / 2) * width
            ax.bar(x + offset, vals, width, yerr=err, capsize=3, label=glabel, color=gcolor)
        ax.axhline(baseline, color="gray", linestyle="--", linewidth=1)
        ax.set(title=title)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=20, ha="right")
        ax.legend(fontsize=8)

    test_note = "test revealed" if test_df is not None else "test not used"
    fig.suptitle(f"Predictor ablation: age / nWBV / CNN, alone and combined ({test_note})")
    fig.tight_layout()
    plot_path = os.path.join(outputs_path, "step7-stacking_comparison.png")
    fig.savefig(plot_path, dpi=120)
    print(f"Saved comparison plot -> {plot_path}")

    # --- predictor correlation scatterplots ---
    split_colors = {"train": "0.6", "validate": "steelblue", "test": "firebrick"}
    fig_corr, ((ax_lr, ax_an), (ax_ca, ax_cn)) = plt.subplots(2, 2, figsize=(12, 11))

    # Panel 1: left vs right hippocampus CNN logit -- the one panel NOT in log-odds
    # space (no single-variable model exists for one side alone). Same units on both
    # axes, so a y=x reference line is meaningful; the other three panels compare two
    # different single-variable models' log-odds and skip it.
    all_vals = []
    for split_name, _ in lr_groups:
        left, right, r = lr_corr[split_name]
        all_vals.append(left)
        all_vals.append(right)
        ax_lr.scatter(left, right, s=18, alpha=0.6, color=split_colors[split_name],
                      label=f"{split_name} (r={r:.2f}, n={len(left)})")
    lo = min(v.min() for v in all_vals)
    hi = max(v.max() for v in all_vals)
    pad = 0.05 * (hi - lo) if hi > lo else 1.0
    ax_lr.plot([lo - pad, hi + pad], [lo - pad, hi + pad],
              color="gray", linestyle="--", linewidth=1, label="left = right")
    ax_lr.set_xlim(lo - pad, hi + pad)
    ax_lr.set_ylim(lo - pad, hi + pad)
    ax_lr.set_xlabel("Left hippocampus CNN logit")
    ax_lr.set_ylabel("Right hippocampus CNN logit")
    ax_lr.set_title("Left vs right hippocampus")
    ax_lr.legend(fontsize=8)

    # Panels 2-4: each variable's OWN single-variable predictor's log-odds (see the
    # module docstring's log-odds framing note -- same |r| as the raw variables, sign
    # can differ).
    def logit_panel(ax, corr, xlabel, ylabel, title):
        for split_name, (a, b, r) in corr.items():
            ax.scatter(a, b, s=18, alpha=0.6, color=split_colors[split_name],
                      label=f"{split_name} (r={r:.2f}, n={len(a)})")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8)

    logit_panel(ax_an, age_nwbv_corr, "Age log-odds", "nWBV log-odds", "Age vs nWBV")
    logit_panel(ax_ca, cnn_age_corr, "CNN log-odds", "Age log-odds", "CNN vs age")
    logit_panel(ax_cn, cnn_nwbv_corr, "CNN log-odds", "nWBV log-odds", "CNN vs nWBV")

    fig_corr.suptitle("Predictor correlation checks: how independent are age / nWBV / CNN?")
    fig_corr.tight_layout()
    corr_plot_path = os.path.join(outputs_path, "step7-predictor_correlations.png")
    fig_corr.savefig(corr_plot_path, dpi=120)
    print(f"Saved predictor correlation plot -> {corr_plot_path}")

    # --- ROC curves, one panel per split, all 5 predictors overlaid ---
    predictor_colors = dict(zip(names, plt.cm.tab10.colors))
    roc_panels = [("train", "tr", y_train), ("validate", "va", y_val)]
    if test_df is not None:
        roc_panels.append(("test", "te", y_test))

    fig_roc, roc_axes = plt.subplots(1, len(roc_panels), figsize=(5.5 * len(roc_panels), 5),
                                     squeeze=False)
    roc_axes = roc_axes[0]
    for ax, (split_label, key, y_true) in zip(roc_axes, roc_panels):
        for r in results:
            fpr, tpr, _ = roc_curve(y_true, r[f"p_{key}"])
            ax.plot(fpr, tpr, color=predictor_colors[r["name"]], linewidth=2,
                    label=f"{r['name']} (AUC={r[f'auc_{key}']:.3f})")
        ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1, label="chance")
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.set_title(split_label)
        ax.legend(fontsize=8, loc="lower right")

    fig_roc.suptitle("ROC curves: age / nWBV / CNN, alone and combined")
    fig_roc.tight_layout()
    roc_plot_path = os.path.join(outputs_path, "step7-roc_curves.png")
    fig_roc.savefig(roc_plot_path, dpi=120)
    print(f"Saved ROC curve plot -> {roc_plot_path}")


if __name__ == "__main__":
    main()
