#!/usr/bin/env bash
# Fetch the Bottomly et al. 2022 analysis code (the paper this project builds on).
#
# It is NOT vendored into this repo: the upstream repository ships no LICENSE
# file, so redistributing it here would be a rights question we do not need to
# take on. Clone it yourself; we only read it for reference.
#
# What it is good for: R/preprocess.R defines the PC1 cell-type baseline
# (`van.galen.scores`), the WGCNA eigengenes (`predict.modules`) and the
# drug-screen join (`inhibitor.dataset`). src/load.py reproduces that join.
#
# Usage:  bash scripts/get_manuscript_code.sh [dest]
set -euo pipefail

DEST="${1:-../beataml2_manuscript}"
REPO="https://github.com/biodev/beataml2_manuscript.git"

command -v git >/dev/null || { echo "ERROR: git not found."; exit 1; }

if [ -d "$DEST" ]; then
  echo "$DEST already exists -- leaving it alone."
  exit 0
fi

echo "==> Cloning $REPO into $DEST"
git clone --depth 1 "$REPO" "$DEST"

cat <<'NOTE'

Cloned. Note two things if you want to actually RUN it:

  1. It needs R 4.0.3 and `renv::restore()`; the README recommends Docker.
  2. Its data/ directory is not included. It expects the nine Beat AML files
     PLUS van Galen et al. 2019 Table S3, which is not in the Beat AML release:
     https://ars.els-cdn.com/content/image/1-s2.0-S0092867419300947-mmc3.xlsx

Reading it is usually enough -- see reports/schema_notes.md.
NOTE
