#!/usr/bin/env python3
"""Shared configuration loading for the OASIS-1 pipeline (steps 1-3).

Centralizes two things that every step needs:

1. Reading ``config.yaml``.
2. Resolving the two data locations:
   - ``data_raw_path``  -- the machine-specific OASIS dataset root, read from the
     ``DATA_RAW_PATH`` environment variable (loaded from a gitignored ``.env``).
   - ``outputs_path`` -- the repo-relative output root (``./outputs``
     by default), set in ``config.yaml``.

All generated artifacts live under ``outputs_path``; the derived paths for
them (metadata.csv, splits.yaml, manifest.yaml, and the per-split PNG folders) are
computed here so ``config.yaml`` never has to repeat them.
"""
from __future__ import annotations

import os

import yaml
from dotenv import load_dotenv

# Repo root = directory containing this file. Used to resolve the (relative)
# outputs_path so scripts work regardless of the current directory.
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

SPLITS = ("train", "validate", "test")


def load_config(path: str = "config.yaml") -> dict:
    """Load config.yaml and resolve both data paths.

    Reads ``.env`` (if present) via python-dotenv, then injects:
      - ``config["data_raw_path"]``       from ``$DATA_RAW_PATH``
      - ``config["outputs_path"]`` resolved to an absolute path

    Raises SystemExit with a clear message if ``DATA_RAW_PATH`` is unset.
    """
    load_dotenv()  # populates os.environ from a .env file if one exists

    if not os.path.isabs(path):
        path = os.path.join(REPO_ROOT, path)
    with open(path, "r") as f:
        config = yaml.safe_load(f)

    raw = os.environ.get("DATA_RAW_PATH")
    if not raw:
        raise SystemExit(
            "DATA_RAW_PATH is not set. Copy .env.example to .env and set it to your "
            "OASIS dataset root (see the 'Configure data paths' section of readme.md)."
        )
    config["data_raw_path"] = os.path.expanduser(raw)

    processed = config.get("outputs_path", "./outputs")
    if not os.path.isabs(processed):
        processed = os.path.join(REPO_ROOT, processed)
    config["outputs_path"] = os.path.normpath(processed)

    return config


def metadata_csv(config: dict) -> str:
    return os.path.join(config["outputs_path"], "metadata.csv")


def reference_xlsx(config: dict) -> str:
    """Absolute path to the OASIS cross-sectional reference spreadsheet.

    Resolved relative to the RAW DATA root (``data_raw_path``) when the config
    value is not absolute -- the spreadsheet is an OASIS material we do not
    redistribute, so it lives beside the discs (``<DATA_RAW_PATH>/docs/``) where
    download-extract-data.sh puts it, not in the repo.

    A missing file is not an error: step1's cross-check reports it and skips.
    """
    path = config.get("reference_xlsx", "docs/oasis_cross-sectional.xlsx")
    if not os.path.isabs(path):
        path = os.path.join(config["data_raw_path"], path)
    return os.path.normpath(path)


def splits_yaml(config: dict) -> str:
    return os.path.join(config["outputs_path"], "splits.yaml")


def manifest_yaml(config: dict) -> str:
    return os.path.join(config["outputs_path"], "manifest.yaml")


def split_dir(config: dict, split: str) -> str:
    """Directory holding a split's flat PNGs: {outputs_path}/{split}/."""
    return os.path.join(config["outputs_path"], split)


def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)
