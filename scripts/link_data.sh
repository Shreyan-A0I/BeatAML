#!/usr/bin/env bash
# Point data/raw/ at an EXISTING local clone of biodev/beataml2.0_data via symlinks.
#
# Use this instead of get_data.sh when you have already cloned the data repo and
# do not want a second ~430 MB copy. get_data.sh is the path for a fresh machine.
#
# Usage:  bash scripts/link_data.sh [path-to-beataml2.0_data]
#         bash scripts/link_data.sh ../beataml2.0_data
set -euo pipefail

SRC="${1:-../beataml2.0_data}"
DEST="data/raw"

[ -d "$SRC" ] || { echo "ERROR: $SRC is not a directory."; echo "Clone biodev/beataml2.0_data first (with git-lfs), or run scripts/get_data.sh."; exit 1; }

SRC_ABS="$(cd "$SRC" && pwd)"
mkdir -p "$DEST"

FILES=(
  beataml_waves1to4_norm_exp_dbgap.txt
  beataml_waves1to4_counts_dbgap.txt
  beataml_probit_curve_fits_v4_dbgap.txt
  beataml_wv1to4_raw_inhibitor_v4_dbgap.txt
  beataml_wes_wv1to4_mutations_dbgap.txt
  beataml_wv1to4_clinical.xlsx
  beataml_waves1to4_sample_mapping.xlsx
  beataml_drug_families.xlsx
  beataml2_manuscript_vg_cts.xlsx
)

echo "==> Linking $SRC_ABS -> $DEST"
MISSING=0
for f in "${FILES[@]}"; do
  if [ -f "$SRC_ABS/$f" ]; then
    ln -sf "$SRC_ABS/$f" "$DEST/$f"
  else
    echo "    MISSING: $f"
    MISSING=1
  fi
done
[ -d "$SRC_ABS/wgcna" ] && ln -sfn "$SRC_ABS/wgcna" "$DEST/wgcna" || { echo "    MISSING: wgcna/"; MISSING=1; }

echo "==> Verifying the targets are real data, not Git LFS pointers"
BAD=0
for f in "${FILES[@]}"; do
  [ -f "$DEST/$f" ] || continue
  if head -c 50 "$DEST/$f" 2>/dev/null | grep -q "git-lfs.github.com"; then
    echo "    POINTER FILE (not data): $f"; BAD=1
  fi
done

if [ "$BAD" -eq 1 ]; then
  echo "ERROR: the source clone holds LFS pointers. In that clone run:"
  echo "  git lfs install && git lfs pull"
  exit 1
fi
[ "$MISSING" -eq 1 ] && echo "WARNING: some files were missing -- filenames may have changed upstream."

echo
echo "Done. data/raw now points at $SRC_ABS (nothing copied)."
echo "Next:  python scripts/inspect_columns.py > reports/schema.txt"
