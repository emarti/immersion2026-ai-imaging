#!/usr/bin/env python3
"""Step 2: choose the study cohort and assign subjects to train/val/test.

Reads ``metadata.csv`` from step1 and produces ``config/splits.yaml``. It also saves an
age-distribution figure ``outputs/step2-cohort_age_histograms.png`` -- CDR-negative vs
CDR-positive counts by age, in three panels (both sexes pooled, sexes separated, and by raw
CDR grade) -- which makes the age (and sex) confound visible and shows both age bands below.

TRAIN and VALIDATE/TEST are built from two SEPARATE age bands, not one shared band, and
each of the three splits is capped at up to its own ``splits:`` fraction (config.yaml) of
its own eligible pool -- a ceiling, not a guarantee; you can get fewer:

  1. VALIDATE/TEST come from the narrow ``cohort.age_eval`` band. Up to
     ``(splits.validate + splits.test)`` of the age_eval-eligible subjects are drawn as a
     BALANCED sample (equal cdr_negative/cdr_positive, sex left free), then that pool is
     split subject-level into validate/test per the relative ``splits:`` ratio.
  2. TRAIN comes from the broader ``cohort.age_train`` band: up to ``splits.train`` of the
     age_train-eligible subjects not already claimed by validate/test, as a plain RANDOM
     sample -- UNBALANCED on purpose, deliberately left age-diverse and class-imbalanced,
     so it can be a much bigger pool. (See ``training.balanced_loss`` in config.yaml, which
     is what compensates for that imbalance during training.)

This way validate/test stay narrow, age-matched, and balanced (honest, if noisier,
numbers), while train stays large (more signal to learn from).

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


def eligible_rows(rows: list[dict], age_min: int, age_max: int) -> list[dict]:
    """Keep subjects in [age_min, age_max] with a valid label and an image."""
    keep = []
    for r in rows:
        if r["label"] not in ("0", "1"):
            continue
        if r["img_exists"].strip().lower() != "true":
            continue
        if r["age"] == "":
            continue
        age = int(r["age"])
        if not (age_min <= age <= age_max):
            continue
        if r["sex"] not in ("M", "F"):
            continue
        keep.append(r)
    return keep


def cell_key(row: dict) -> tuple[str, str]:
    return (row["sex"], row["label"])


EXPECTED_CELLS = [("M", "0"), ("M", "1"), ("F", "0"), ("F", "1")]


def _cell_sizes(rows: list[dict]) -> dict:
    """(sex, label) cell counts, for the printout."""
    cells: dict[tuple[str, str], list[dict]] = {}
    for r in rows:
        cells.setdefault(cell_key(r), []).append(r)
    return {k: len(cells.get(k, [])) for k in EXPECTED_CELLS}


def _take(members: list[dict], rng: random.Random, n=None) -> list[dict]:
    """Deterministically shuffle members (by subject) and optionally cap to n."""
    members = sorted(members, key=lambda r: r["subject"])
    rng.shuffle(members)
    return members if n is None else members[:n]


def select_balanced_eval_cohort(rows: list[dict], target_per_label: int, rng: random.Random,
                                max_subjects=None):
    """One split's (validate's or test's) draw from whatever's currently left of the
    age_eval-eligible pool: equal cdr_negative/cdr_positive counts (sex left free),
    hardcoded -- capped at ``target_per_label`` per label and, if set, ``max_subjects``.
    May end up SMALLER than the cap if fewer eligible subjects remain for a label than
    the cap allows -- it's a ceiling, not a guarantee.

    Returns ``(groups, sizes, info)`` where ``groups`` maps label -> subject list (the
    chosen members for THIS split), ``sizes`` is the (sex,label) cell counts of ``rows``
    (what was available before this draw) for reporting, and ``info`` is a one-line
    description.
    """
    sizes = _cell_sizes(rows)
    by_label = {lbl: [r for r in rows if r["label"] == lbl] for lbl in ("0", "1")}
    n = min(len(by_label["0"]), len(by_label["1"]), target_per_label)
    if max_subjects is not None:
        n = min(n, max_subjects // 2)
    groups = {lbl: _take(by_label[lbl], rng, n) for lbl in ("0", "1")}
    return groups, sizes, (f"balanced (sex free): {n} per label, {n * 2} total "
                           f"(target {target_per_label} per label)")


def _record(r: dict) -> dict:
    """A metadata.csv row -> the dict shape stored in splits.yaml."""
    return {
        "subject": r["subject"],
        "sex": r["sex"],
        "label": int(r["label"]),
        "cdr": float(r["cdr"]),   # CDR grade (0.5/1/2) behind label 1
        "img_path": r["img_path"],
        "age": int(r["age"]),
    }


def build_train_split(rows: list[dict], exclude_subjects: set[str], rng: random.Random,
                      max_subjects=None) -> list[dict]:
    """ALL age_train-eligible subjects not already claimed by validate/test -- every one
    of them, UNBALANCED, on purpose. No fraction, no cap, other than the optional
    ``max_subjects`` (for fast test runs)."""
    pool = [r for r in rows if r["subject"] not in exclude_subjects]
    if max_subjects is not None:
        pool = _take(pool, rng, max_subjects)
    return sorted((_record(r) for r in pool), key=lambda r: r["subject"])


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


def print_targets(eval_draws: list[dict], n_train_elig: int, n_eval_elig: int,
                  train_actual: int) -> None:
    """Print each split's TARGET vs ACTUAL count, and the fraction it actually ended up
    being -- of the (larger) age_train-eligible pool its fraction was computed against,
    AND of its own (smaller) age_eval-eligible pool, since those two can differ a lot
    (validate/test's fraction is set against the train-range pool, but drawn only from
    the eval range -- so as a share of ITS OWN range it typically ends up far bigger than
    the configured fraction).

    ``eval_draws``: one dict per validate/test draw, each with keys
    ``split, frac, target_total, target_per_label, sizes, info, actual``.
    """
    print("\nTarget vs actual (validate/test fractions are of the age_train-eligible pool, "
          "but drawn only from age_eval):")
    names = {("M", "0"): "Male/CDR-", ("M", "1"): "Male/CDR+",
             ("F", "0"): "Female/CDR-", ("F", "1"): "Female/CDR+"}
    for d in eval_draws:
        print(f"\n  {d['split']}: fraction {d['frac']:.2f} of age_train-eligible "
              f"({n_train_elig}) -> target {d['target_total']} total "
              f"({d['target_per_label']}/label)")
        print(f"    available (sex, label) before this draw, within age_eval:")
        for k, v in d["sizes"].items():
            print(f"      {names[k]:<16} {v}")
        print(f"    drew: {d['info']}")
        actual = d["actual"]
        pct_train = 100 * actual / n_train_elig if n_train_elig else float("nan")
        pct_eval = 100 * actual / n_eval_elig if n_eval_elig else float("nan")
        print(f"    actual: {actual} subjects = {pct_train:.1f}% of age_train-eligible, "
              f"{pct_eval:.1f}% of ITS OWN age_eval-eligible pool ({n_eval_elig})")
    pct_train_actual = 100 * train_actual / n_train_elig if n_train_elig else float("nan")
    print(f"\n  train: everything else in age_train-eligible, not claimed by validate/test "
          f"-> {train_actual} subjects = {pct_train_actual:.1f}% of age_train-eligible "
          f"({n_train_elig})")


def print_summary(summary: dict) -> None:
    header = f"  {'split':<8} {'total':>5} {'male':>5} {'female':>7} {'CDR+':>9} {'CDR-':>8}"
    print(f"\n{header}")
    for split in SPLITS:
        s = summary[split]
        print(f"  {split:<8} {s['total']:>5} {s['male']:>5} {s['female']:>7} "
              f"{s['cdr_positive']:>9} {s['cdr_negative']:>8}")


def plot_age_histograms(rows: list[dict], cohort: dict, splits: dict, out_path: str,
                        labels_cfg: dict) -> None:
    """Save a 2x2 age histogram of CDR-negative vs CDR-positive across the TRAIN band,
    with the narrower EVAL band (validate+test) shaded so both are visible at once.

    (Deliberately duplicated from step1's near-identical plot -- this teaching project keeps the
    two steps self-contained. step1 shows the entire dataset; step2 restricts to the
    age_train band, i.e. the widest range any subject could be eligible for.) Panels:
    (1) both sexes pooled; (2) sexes separated (line style); (3) raw CDR grade
    (0 / 0.5 / 1 / 2) separated; (4) split membership (train/validate/test) -- since
    validate/test subjects are excluded from train (no leakage, see build_train_split),
    this panel shows whether validate/test are visibly depleting train of subjects inside
    the eval band. Some depletion there is expected and not a problem by itself -- this is
    just visibility into how much.

    Note: step2's rows come from metadata.csv, so every field is a string (unlike step1's
    native-typed rows).
    """
    import matplotlib
    matplotlib.use("Agg")                 # write a file, no interactive window
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    import numpy as np

    plt.style.use("seaborn-v0_8-darkgrid")

    tmin, tmax = cohort["age_train"]["min"], cohort["age_train"]["max"]
    emin, emax = cohort["age_eval"]["min"], cohort["age_eval"]["max"]
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
        if not (tmin <= age <= tmax):                # train band only (the widest one)
            continue
        recs.append((age, r["sex"], int(r["label"]), cdr))
    if not recs:
        print("  [skip] no in-band subjects to plot age histograms")
        return

    ages = np.array([a for a, _, _, _ in recs])
    sexes = np.array([s for _, s, _, _ in recs])
    labels = np.array([l for _, _, l, _ in recs])
    cdrs = np.array([c for _, _, _, c in recs])

    edges = np.arange(tmin, tmax + 2, 2)            # 2-year bins across the train band
    neg_name = labels_cfg.get("cdr_negative", "CDR Negative")
    pos_name = labels_cfg.get("cdr_positive", "CDR Positive")
    C_NEG, C_POS = "steelblue", "crimson"          # matches CDR- = blue / CDR+ = red elsewhere

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12.5, 9.4), sharex=True)

    # Shade the narrow eval band (validate+test) inside the wider train band, on every
    # panel, BEFORE the legend() calls below so the shading gets its own legend entry.
    for ax in (ax1, ax2, ax3, ax4):
        ax.axvspan(emin, emax, color="0.4", alpha=0.15, zorder=0,
                  label="eval band (validate+test)")

    # Every panel is STACKED, filled regions -- two (or more) independent semi-
    # transparent fills drawn on top of each other just blur into an ambiguous blob;
    # stacking keeps each sub-group legible.
    def stacked_fill(ax, cats, alpha=0.6):
        """``cats``: list of (ages_array, colour, legend_label)."""
        cats = [c for c in cats if len(c[0])]
        if not cats:
            return
        ax.hist([c[0] for c in cats], bins=edges, histtype="stepfilled", stacked=True,
                alpha=alpha, color=[c[1] for c in cats], label=[c[2] for c in cats])

    # (1) CDR- vs CDR+, both sexes pooled.
    stacked_fill(ax1, [(ages[labels == 0], C_NEG, neg_name),
                       (ages[labels == 1], C_POS, pos_name)])
    ax1.set(title="CDR status vs age (both sexes)", xlabel="age (years)", ylabel="subjects")
    ax1.legend(fontsize=8)

    # (2) CDR- vs CDR+, sexes separated -- colour FAMILY = sex (Male green, Female purple;
    # different hues from panels 1/3/4 to avoid clashing with their CDR-status colouring),
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

    # (3) Raw CDR grade separated (both sexes pooled) -- each grade already has its own
    # colour.
    grade_colours = {0.0: "steelblue", 0.5: "gold", 1.0: "darkorange", 2.0: "crimson"}
    cats3 = [(ages[cdrs == g], grade_colours[g], f"CDR {g:g}")
            for g in (0.0, 0.5, 1.0, 2.0)]
    stacked_fill(ax3, cats3, alpha=0.7)
    ax3.set(title="CDR grade vs age (both sexes)", xlabel="age (years)")
    ax3.legend(fontsize=8)

    # (4) CDR- vs CDR+, train vs eval (validate+test combined) -- colour FAMILY = split
    # (train blue, validate+test red), SHADE = CDR status (dark = CDR+, light = CDR-).
    # Stacking means, inside the shaded eval band, each bar visibly splits into its train
    # portion (bottom) and validate+test portion (top) -- exactly how much of that age
    # range's subjects validate/test took from train.
    TRAIN_LIGHT, TRAIN_DARK = "lightskyblue", "navy"
    EVAL_LIGHT, EVAL_DARK = "lightsalmon", "darkred"
    groups = (("train", ("train",), (TRAIN_LIGHT, TRAIN_DARK)),
             ("validate+test", ("validate", "test"), (EVAL_LIGHT, EVAL_DARK)))
    cats4 = [(np.array([m["age"] for s in split_names for m in splits.get(s, [])
                       if m["label"] == lbl]), colour, f"{group_name} · {name}")
            for group_name, split_names, (c_neg, c_pos) in groups
            for lbl, colour, name in ((0, c_neg, neg_name), (1, c_pos, pos_name))]
    stacked_fill(ax4, cats4)
    ax4.set(title="CDR status by split (train vs validate+test)", xlabel="age (years)")
    ax4.legend(fontsize=7)

    for ax in (ax1, ax2, ax3, ax4):                 # integer counts -> integer y ticks
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax1.set_xlim(tmin, tmax)                        # shared x -> clamps all panels to the band

    fig.suptitle(f"OASIS-1 age distribution by CDR status -- train band {tmin}-{tmax} "
                f"(eval band {emin}-{emax} shaded)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Saved age histograms -> {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()

    config = load_config(args.config)
    cohort = config["cohort"]
    rng = random.Random(cohort["seed"])

    rows = read_metadata(metadata_csv(config))

    eval_elig = eligible_rows(rows, cohort["age_eval"]["min"], cohort["age_eval"]["max"])
    train_elig = eligible_rows(rows, cohort["age_train"]["min"], cohort["age_train"]["max"])
    print(f"Eligible age_eval {cohort['age_eval']['min']}-{cohort['age_eval']['max']} "
          f"labelled subjects with images: {len(eval_elig)}")
    print(f"Eligible age_train {cohort['age_train']['min']}-{cohort['age_train']['max']} "
          f"labelled subjects with images: {len(train_elig)}")

    ratios = config["splits"]
    n_train_elig = len(train_elig)
    n_eval_elig = len(eval_elig)

    # VALIDATE, then TEST: each is drawn from whatever's currently left of the age_eval-
    # eligible pool, balanced, capped at its OWN fraction of the age_train-eligible pool
    # (not the age_eval pool it's actually drawn from -- see config.yaml `splits:`).
    remaining_eval_pool = list(eval_elig)
    eval_out: dict[str, list[dict]] = {}
    eval_draws = []
    for split in ("validate", "test"):
        frac = ratios[split]
        target_total = round(frac * n_train_elig)
        target_per_label = target_total // 2
        groups, sizes, info = select_balanced_eval_cohort(
            remaining_eval_pool, target_per_label, rng, cohort.get("max_subjects")
        )
        chosen = groups["0"] + groups["1"]
        chosen_ids = {r["subject"] for r in chosen}
        remaining_eval_pool = [r for r in remaining_eval_pool if r["subject"] not in chosen_ids]
        eval_out[split] = sorted((_record(r) for r in chosen), key=lambda r: r["subject"])
        eval_draws.append({"split": split, "frac": frac, "target_total": target_total,
                           "target_per_label": target_per_label, "sizes": sizes,
                           "info": info, "actual": len(chosen)})

    eval_subject_ids = {r["subject"] for r in eval_out["validate"] + eval_out["test"]}
    if not eval_subject_ids:
        print("\nNo eval cohort selected (no eligible subjects in the age_eval band).")
        print("Widen age_eval, raise the validate/test fractions, or extract more discs.")

    # TRAIN: everything else in age_train-eligible, not claimed by validate/test above.
    train_members = build_train_split(train_elig, eval_subject_ids, rng,
                                      cohort.get("max_subjects"))

    splits = {"train": train_members, "validate": eval_out["validate"], "test": eval_out["test"]}
    summary = build_summary(splits)
    print_targets(eval_draws, n_train_elig, n_eval_elig, len(train_members))
    print_summary(summary)

    out = {
        "meta": {
            "cohort": cohort,
            "split_ratios": ratios,
            "n_train_eligible": n_train_elig,
            "n_eval_eligible": n_eval_elig,
            "validate_target": eval_draws[0]["target_total"],
            "test_target": eval_draws[1]["target_total"],
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
    plot_age_histograms(rows, cohort, splits,
                        os.path.join(outputs_path, "step2-cohort_age_histograms.png"),
                        config.get("labels") or {})


if __name__ == "__main__":
    main()
