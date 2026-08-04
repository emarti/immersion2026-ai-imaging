#!/usr/bin/env python3
"""Step 2: choose the study cohort and assign subjects to train/val/test.

Reads ``metadata.csv`` from step1 and produces ``config/splits.yaml``. It also saves an
age-distribution figure ``outputs/step2-cohort_age_histograms.png`` -- CDR-negative vs
CDR-positive counts by age, in three panels (both sexes pooled, sexes separated, and by raw
CDR grade) -- which makes the age (and sex) confound visible and helps pick ``age_min`` /
``age_max``.

Cohort: subjects in the configured age range with a valid CDR and an image on
disk. ``cohort.balance`` then selects how to balance them:
  strict -- equal per (sex x label) cell, capped by the smallest cell.
  label  -- equal cdr_negative/cdr_positive, sex left free (~2x strict).
  none   -- all eligible subjects, no balancing.

Splits are made at the SUBJECT level (no subject appears in two splits) and are
stratified by the balancing strata, so every split keeps the same balance.

Usage:
    python step2-assign-groups.py [--config config.yaml]
"""
from __future__ import annotations

import argparse
import csv
import os
import random

import yaml

from common import SPLITS, load_config, metadata_csv, splits_yaml


def read_metadata(path: str) -> list[dict]:
    with open(path, "r", newline="") as f:
        return list(csv.DictReader(f))


def eligible_rows(rows: list[dict], cohort: dict) -> list[dict]:
    """Keep subjects in the configured age range with a valid label and an image."""
    keep = []
    for r in rows:
        if r["label"] not in ("0", "1"):
            continue
        if r["img_exists"].strip().lower() != "true":
            continue
        if r["age"] == "":
            continue
        age = int(r["age"])
        if not (cohort["age_min"] <= age <= cohort["age_max"]):
            continue
        if r["sex"] not in ("M", "F"):
            continue
        keep.append(r)
    return keep


def cell_key(row: dict) -> tuple[str, str]:
    return (row["sex"], row["label"])


EXPECTED_CELLS = [("M", "0"), ("M", "1"), ("F", "0"), ("F", "1")]


def _cell_sizes(rows: list[dict]) -> dict:
    """(sex, label) cell counts, for the printout (reported in every mode)."""
    cells: dict[tuple[str, str], list[dict]] = {}
    for r in rows:
        cells.setdefault(cell_key(r), []).append(r)
    return {k: len(cells.get(k, [])) for k in EXPECTED_CELLS}


def _take(members: list[dict], rng: random.Random, n=None) -> list[dict]:
    """Deterministically shuffle members (by subject) and optionally cap to n."""
    members = sorted(members, key=lambda r: r["subject"])
    rng.shuffle(members)
    return members if n is None else members[:n]


def select_cohort(rows: list[dict], balance: str, rng: random.Random,
                  max_subjects=None):
    """Group eligible subjects into strata for the stratified subject-level split.

    Returns ``(groups, sizes, info)`` where ``groups`` maps a stratum key -> subject
    list (what ``assign_splits`` divides independently), ``sizes`` is the (sex,label)
    cell counts for reporting, and ``info`` is a one-line description. Modes:

      strict -- equal per (sex x label) cell, capped by the smallest cell.
      label  -- equal cdr_negative/cdr_positive, sex left free.
      none   -- all eligible subjects, no balancing.

    ``max_subjects`` optionally caps the total for fast test runs.
    """
    sizes = _cell_sizes(rows)

    if balance == "strict":
        cells = {k: [r for r in rows if cell_key(r) == k] for k in EXPECTED_CELLS}
        n = min(sizes.values()) if sizes else 0
        if max_subjects is not None:
            n = min(n, max_subjects // 4)
        groups = {k: _take(cells[k], rng, n) for k in EXPECTED_CELLS}
        return groups, sizes, f"strict (sex x label): {n} per cell, {n * 4} total"

    if balance == "label":
        by_label = {lbl: [r for r in rows if r["label"] == lbl] for lbl in ("0", "1")}
        n = min(len(by_label["0"]), len(by_label["1"]))
        if max_subjects is not None:
            n = min(n, max_subjects // 2)
        groups = {lbl: _take(by_label[lbl], rng, n) for lbl in ("0", "1")}
        return groups, sizes, f"label-balanced (sex free): {n} per label, {n * 2} total"

    if balance == "none":
        kept = _take(rows, rng, max_subjects)
        groups: dict[tuple[str, str], list[dict]] = {}
        for r in kept:
            groups.setdefault(cell_key(r), []).append(r)
        return groups, sizes, f"not balanced: {len(kept)} eligible subjects"

    raise SystemExit(f"unknown cohort.balance: {balance!r} (use strict | label | none)")


def split_counts(n: int, ratios: dict) -> dict:
    """Allocate n items to train/validate/test; remainder favours train.

    When a cell has at least 3 subjects we guarantee validate and test each get
    at least 1 (borrowing from train), so small test cohorts still populate all
    three splits. Cells with fewer than 3 can't fill all three, so they're left
    to the ratio-based rounding.
    """
    train = round(n * ratios["train"])
    validate = round(n * ratios["validate"])
    test = n - train - validate
    if test < 0:  # rounding overshoot on tiny cells
        validate = max(0, validate + test)
        test = 0
    if n >= 3:
        if validate == 0:
            validate, train = 1, train - 1
        if test == 0:
            test, train = 1, train - 1
    return {"train": train, "validate": validate, "test": test}


def assign_splits(balanced: dict, ratios: dict, rng: random.Random) -> dict:
    """Stratified subject-level split; each cell is divided independently."""
    result = {split: [] for split in SPLITS}
    for members in balanced.values():
        members = list(members)
        rng.shuffle(members)
        counts = split_counts(len(members), ratios)
        idx = 0
        for split in SPLITS:
            for r in members[idx: idx + counts[split]]:
                result[split].append(
                    {
                        "subject": r["subject"],
                        "sex": r["sex"],
                        "label": int(r["label"]),
                        "cdr": float(r["cdr"]),   # CDR grade (0.5/1/2) behind label 1
                        "img_path": r["img_path"],
                    }
                )
            idx += counts[split]
    for split in result:
        result[split].sort(key=lambda r: r["subject"])
    return result


def build_summary(splits: dict) -> dict:
    summary = {}
    for split, members in splits.items():
        summary[split] = {
            "total": len(members),
            "male": sum(1 for m in members if m["sex"] == "M"),
            "female": sum(1 for m in members if m["sex"] == "F"),
            "cdr_positive": sum(1 for m in members if m["label"] == 1),
            "cdr_negative": sum(1 for m in members if m["label"] == 0),
        }
    return summary


def print_summary(sizes: dict, info: str, summary: dict) -> None:
    print("\nEligible (sex, label) cell sizes:")
    names = {("M", "0"): "Male/CDR-", ("M", "1"): "Male/CDR+",
             ("F", "0"): "Female/CDR-", ("F", "1"): "Female/CDR+"}
    for k, v in sizes.items():
        print(f"  {names[k]:<16} {v}")
    print(f"\nCohort: {info}\n")
    header = f"  {'split':<8} {'total':>5} {'male':>5} {'female':>7} {'CDR+':>9} {'CDR-':>8}"
    print(header)
    for split in SPLITS:
        s = summary[split]
        print(f"  {split:<8} {s['total']:>5} {s['male']:>5} {s['female']:>7} "
              f"{s['cdr_positive']:>9} {s['cdr_negative']:>8}")


def plot_age_histograms(rows: list[dict], cohort: dict, out_path: str, labels_cfg: dict) -> None:
    """Save a 3-panel age histogram of CDR-negative vs CDR-positive for the COHORT band only.

    (Deliberately duplicated from step1's near-identical plot -- this teaching project keeps the
    two steps self-contained. step1 shows the entire dataset; step2 restricts to the configured
    age_min..age_max range, i.e. the subjects eligible for the cohort.) Panels: (1) both sexes
    pooled; (2) sexes separated (line style); (3) raw CDR grade (0 / 0.5 / 1 / 2) separated.

    Note: step2's rows come from metadata.csv, so every field is a string (unlike step1's
    native-typed rows).
    """
    import matplotlib
    matplotlib.use("Agg")                 # write a file, no interactive window
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    import numpy as np

    plt.style.use("seaborn-v0_8-darkgrid")

    amin, amax = cohort["age_min"], cohort["age_max"]
    recs = []
    for r in rows:
        if r["label"] not in ("0", "1"):
            continue
        if r.get("img_exists", "").strip().lower() != "true":
            continue
        if r["age"] == "" or r["sex"] not in ("M", "F"):
            continue
        try:
            age, cdr = int(r["age"]), float(r["cdr"])
        except ValueError:
            continue
        if not (amin <= age <= amax):                # cohort band only
            continue
        recs.append((age, r["sex"], int(r["label"]), cdr))
    if not recs:
        print("  [skip] no in-band subjects to plot age histograms")
        return

    ages = np.array([a for a, _, _, _ in recs])
    sexes = np.array([s for _, s, _, _ in recs])
    labels = np.array([l for _, _, l, _ in recs])
    cdrs = np.array([c for _, _, _, c in recs])

    edges = np.arange(amin, amax + 2, 2)           # 2-year bins across the cohort band
    neg_name = labels_cfg.get("cdr_negative", "CDR Negative")
    pos_name = labels_cfg.get("cdr_positive", "CDR Positive")
    C_NEG, C_POS = "steelblue", "crimson"          # matches CDR- = blue / CDR+ = red elsewhere

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 4.6), sharex=True)

    # EVERY panel draws STACKED, filled regions: each bar's full height is the number of
    # subjects in that age bin, divided into coloured segments. Stacking rather than drawing
    # the groups on top of one another means no group can hide behind another, and all three
    # panels are read the same way -- total height first, then the split within it.
    def stacked_fill(ax, cats, alpha=0.6):
        """``cats``: list of (ages_array, colour, legend_label)."""
        cats = [c for c in cats if len(c[0])]
        if not cats:
            return
        ax.hist([c[0] for c in cats], bins=edges, histtype="stepfilled", stacked=True,
                alpha=alpha, color=[c[1] for c in cats], label=[c[2] for c in cats])

    # (1) CDR- vs CDR+, both sexes pooled.
    cats1 = [(ages[labels == 0], C_NEG, neg_name), (ages[labels == 1], C_POS, pos_name)]
    stacked_fill(ax1, cats1, alpha=0.55)
    ax1.set(title="CDR status vs age (both sexes)", xlabel="age (years)", ylabel="subjects")
    ax1.legend(fontsize=8)

    # (2) CDR- vs CDR+, sexes separated -- colour FAMILY = sex (Male green, Female purple;
    # different hues from panels 1/3 so they don't clash with the CDR-status colouring),
    # SHADE = CDR status (dark = CDR+, light = CDR-).
    MALE_LIGHT, MALE_DARK = "palegreen", "darkgreen"
    FEMALE_LIGHT, FEMALE_DARK = "plum", "indigo"
    cats2 = [(ages[(labels == lbl) & (sexes == sex)], colour, f"{sname} · {name}")
             for sex, (c_neg, c_pos), sname in (("M", (MALE_LIGHT, MALE_DARK), "Male"),
                                                ("F", (FEMALE_LIGHT, FEMALE_DARK), "Female"))
             for lbl, colour, name in ((0, c_neg, neg_name), (1, c_pos, pos_name))]
    stacked_fill(ax2, cats2)
    ax2.set(title="CDR status vs age (sexes separated)", xlabel="age (years)")
    ax2.legend(fontsize=7)

    # (3) Raw CDR grade separated (both sexes pooled) -- each grade already has its own colour.
    grade_colours = {0.0: "steelblue", 0.5: "gold", 1.0: "darkorange", 2.0: "crimson"}
    cats3 = [(ages[cdrs == g], grade_colours[g], f"CDR {g:g}") for g in (0.0, 0.5, 1.0, 2.0)]
    stacked_fill(ax3, cats3, alpha=0.7)
    ax3.set(title="CDR grade vs age (both sexes)", xlabel="age (years)")
    ax3.legend(fontsize=8)

    for ax in (ax1, ax2, ax3):                      # integer counts -> integer y ticks
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax1.set_xlim(amin, amax)                        # shared x -> clamps all panels to the band

    fig.suptitle(f"OASIS-1 age distribution by CDR status -- cohort band (age {amin}-{amax})",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Saved age histograms -> {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()

    config = load_config(args.config)
    rng = random.Random(config["cohort"]["seed"])

    rows = read_metadata(metadata_csv(config))
    elig = eligible_rows(rows, config["cohort"])
    c = config["cohort"]
    print(f"Eligible age {c['age_min']}-{c['age_max']} labelled subjects with images: {len(elig)}")

    groups, sizes, info = select_cohort(
        elig, config["cohort"]["balance"], rng, config["cohort"].get("max_subjects")
    )
    total = sum(len(v) for v in groups.values())
    if total == 0:
        print("\nNo cohort selected (no eligible subjects).")
        print("Widen the cohort (age range / balance) or extract more discs, then re-run.")

    splits = assign_splits(groups, config["splits"], rng)
    summary = build_summary(splits)
    print_summary(sizes, info, summary)

    out = {
        "meta": {
            "cohort": config["cohort"],
            "split_ratios": config["splits"],
            "balance": config["cohort"]["balance"],
            "cohort_total": total,
            "summary": summary,
        },
        **{split: splits[split] for split in SPLITS},
    }
    out_path = splits_yaml(config)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        yaml.safe_dump(out, f, sort_keys=False)
    print(f"\nWrote splits -> {out_path}")

    # Age-distribution figure over all labelled subjects (shows the age/sex confound).
    outputs_path = config["outputs_path"]
    os.makedirs(outputs_path, exist_ok=True)
    plot_age_histograms(rows, config["cohort"],
                        os.path.join(outputs_path, "step2-cohort_age_histograms.png"),
                        config.get("labels") or {})


if __name__ == "__main__":
    main()
