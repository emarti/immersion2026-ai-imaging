#!/usr/bin/env bash
#
# download-extract-data.sh -- OPTIONAL: download + extract the OASIS-1 cross-sectional
# discs (1-12) into the folder named by DATA_RAW_PATH in oasis1/.env.
#
# This is NOT part of process.sh. It's a convenience for grabbing the raw data, especially
# inside GitHub Codespaces (pulls straight from WashU, no local upload). Requires `wget` and
# `tar` -- both present on Linux / Codespaces; on macOS install wget via Homebrew, or download
# the discs manually (see readme.md). The download is large and slow (~1.5 GB per disc, 12
# discs); `wget -c` resumes partial files if you re-run. By default it EXTRACTS ONLY the files
# the pipeline uses (~63 GB -> ~6 GB); set EXTRACT_ONLY_MASKED=0 to unpack every file. It also
# fetches the OASIS reference spreadsheets + fact sheet into docs/ (these are OASIS materials,
# not redistributed in the repo).
#
# Use this ONLY after your OASIS access request is approved and you have agreed to the OASIS
# Data Use Agreement (academic / non-commercial use; no redistribution). See readme.md, §1.
#
# Usage (from the oasis1 folder):   bash download-extract-data.sh
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
env_file="$script_dir/.env"
if [[ ! -f "$env_file" ]]; then
    echo "No $env_file found -- copy .env.example to .env first (see readme.md, §1)." >&2
    exit 1
fi

# Read DATA_RAW_PATH from .env (the last matching line wins).
data_dir="$(grep -E '^DATA_RAW_PATH=' "$env_file" | tail -1 | cut -d= -f2-)"
if [[ -z "$data_dir" ]]; then
    echo "DATA_RAW_PATH is not set in $env_file." >&2
    exit 1
fi

# Extract only what the pipeline uses (default ON): instead of unpacking every file, tar pulls
# out just each session's .txt (metadata, read by step1) and the atlas-registered brain-masked
# volume that step3 slices (*_t88_masked_gfc.img + its .hdr; nibabel's Analyze reader needs
# both). The unused files (raw scans, extra processed volumes, segmentations, previews) never
# touch disk -- ~13 MB per subject instead of ~115 MB, i.e. the whole set ~6 GB instead of
# ~63 GB, which is what makes it fit in GitHub Codespaces. Set EXTRACT_ONLY_MASKED=0 to unpack
# everything:  EXTRACT_ONLY_MASKED=0 bash download-extract-data.sh
EXTRACT_ONLY_MASKED="${EXTRACT_ONLY_MASKED:-1}"

# tar member patterns to keep. `*_MR[0-9].txt` matches each session's metadata file (sessions
# are MR1/MR2) but not the FSL_SEG `*_fseg.txt`.
keep_patterns=( '*_t88_masked_gfc.img' '*_t88_masked_gfc.hdr' '*_MR[0-9].txt' )

# macOS and Linux tar differ on selecting members by pattern: GNU tar (Linux / Codespaces)
# needs `--wildcards --no-anchored` to treat the patterns as globs that match anywhere in the
# path; bsdtar (the macOS default) globs by default and rejects those flags. Detect which one
# we have so the same script works on both.
if tar --version 2>/dev/null | grep -qi 'gnu tar'; then
    tar_glob=( --wildcards --no-anchored )   # GNU tar
else
    tar_glob=()                              # bsdtar / libarchive (macOS)
fi

extract_only_masked() {
    # Unpack just the keep_patterns from the archive (tar still reads the whole stream, but only
    # writes the matches, creating just their parent folders). The ${arr[@]+...} guard keeps the
    # empty-array expansion safe under `set -u` on macOS's bash 3.2.
    tar -xzf "$1" ${tar_glob[@]+"${tar_glob[@]}"} "${keep_patterns[@]}"
}

# Fetch the OASIS reference spreadsheets + fact sheet into docs/ (step1 cross-checks
# metadata.csv against oasis_cross-sectional.xlsx). These are OASIS materials we do NOT
# redistribute in the repo (OASIS DUA), so they are downloaded from the OASIS site here. The
# download URLs carry a hash suffix, so save each under the name the pipeline expects.
docs_dir="$script_dir/docs"
mkdir -p "$docs_dir"
docs=(
    "oasis_cross-sectional.xlsx|https://sites.wustl.edu/oasisbrains/files/2024/04/oasis_cross-sectional-5708aa0a98d82080.xlsx"
    "oasis_cross-sectional-reliability.xlsx|https://sites.wustl.edu/oasisbrains/files/2024/04/oasis_cross-sectional-reliability-063c8642b909ee76.xlsx"
    "oasis_cross-sectional_facts.pdf|https://sites.wustl.edu/oasisbrains/files/2024/03/oasis_cross-sectional_facts-bcc7a002dfb104f4.pdf"
)
for entry in "${docs[@]}"; do
    name="${entry%%|*}"; url="${entry#*|}"
    echo "== Downloading docs/${name} =="
    wget -O "$docs_dir/$name" "$url"
done

mkdir -p "$data_dir"
cd "$data_dir"

# Process the 12 discs ONE AT A TIME -- download, extract (only the needed files by default),
# then delete the .tar.gz -- so nothing piles up. Peak extra disk is ~1 archive (~1.5 GB) plus
# the small extracted data; the ~63 GB of unused files never hit disk at all. `wget -c` resumes
# a partial download; `rm` runs only after `tar` succeeds (otherwise `set -e` aborts the run),
# so a failed extraction leaves that archive in place for a re-run.
base_url="https://download.nrg.wustl.edu/data"
for i in {1..12}; do
    f="oasis_cross-sectional_disc${i}.tar.gz"
    echo "== [${i}/12] Downloading ${f} =="
    wget -c "${base_url}/${f}"
    if [[ "$EXTRACT_ONLY_MASKED" == "1" ]]; then
        echo "== [${i}/12] Extracting only the .txt + masked_gfc volume from ${f} =="
        extract_only_masked "$f"
    else
        echo "== [${i}/12] Extracting all of ${f} =="
        tar -xzf "$f"
    fi
    rm -f "$f"
done

echo "== Done. Data is in: $data_dir =="
