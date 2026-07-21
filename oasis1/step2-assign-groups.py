#!/usr/bin/env python3
"""Step 2: choose the study cohort and assign subjects to train/val/test.

Reads ``metadata.csv`` from step1 and produces ``config/splits.yaml``.

Cohort: subjects in the configured age range with a valid CDR and an image on
disk. ``cohort.balance`` then selects how to balance them:
  strict -- equal per (sex x label) cell, capped by the smallest cell.
  label  -- equal healthy/demented, sex left free (~2x strict).
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
      label  -- equal healthy/demented, sex left free.
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
                        "cdr": float(r["cdr"]),   # AD grade (0.5/1/2) behind label 1
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
            "demented": sum(1 for m in members if m["label"] == 1),
            "healthy": sum(1 for m in members if m["label"] == 0),
        }
    return summary


def print_summary(sizes: dict, info: str, summary: dict) -> None:
    print("\nEligible (sex, label) cell sizes:")
    names = {("M", "0"): "Male/Healthy", ("M", "1"): "Male/Demented",
             ("F", "0"): "Female/Healthy", ("F", "1"): "Female/Demented"}
    for k, v in sizes.items():
        print(f"  {names[k]:<16} {v}")
    print(f"\nCohort: {info}\n")
    header = f"  {'split':<8} {'total':>5} {'male':>5} {'female':>7} {'demented':>9} {'healthy':>8}"
    print(header)
    for split in SPLITS:
        s = summary[split]
        print(f"  {split:<8} {s['total']:>5} {s['male']:>5} {s['female']:>7} "
              f"{s['demented']:>9} {s['healthy']:>8}")


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


if __name__ == "__main__":
    main()
