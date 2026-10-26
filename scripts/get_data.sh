#!/usr/bin/env bash
# Fetch the Beat AML 2.0 public data release into data/raw/.
#
# IMPORTANT: this repo stores its data files with Git LFS. A plain `git clone`
# without git-lfs installed gives you ~130-byte POINTER FILES, not data. If you
# already cloned it and the files look tiny, that is what happened -- install
# git-lfs and run `git lfs pull` inside that clone.
#
# Usage:  bash scripts/get_data.sh
set -euo pipefail

REPO_URL="https://github.com/biodev/beataml2.0_data.git"
DEST="data/raw"
TMP=".beataml_clone"

command -v git >/dev/null || { echo "ERROR: git not found."; exit 1; }

if ! command -v git-lfs >/dev/null && ! git lfs version >/dev/null 2>&1; then
  cat <<'EOF'
ERROR: git-lfs is not installed.

Without it you will download pointer files instead of data.

  macOS:          brew install git-lfs
  Debian/Ubuntu:  sudo apt-get install git-lfs
  conda:          conda install -c conda-forge git-lfs

Then run:  git lfs install
EOF
  exit 1
fi

mkdir -p "$DEST"

echo "==> Cloning Beat AML 2.0 data (LFS, shallow)"
rm -rf "$TMP"
GIT_LFS_SKIP_SMUDGE=0 git clone --depth 1 "$REPO_URL" "$TMP"
( cd "$TMP" && git lfs pull )

echo "==> Copying data files into $DEST"
# Core files used by this project
for f in \
  beataml_waves1to4_norm_exp_dbgap.txt \
  beataml_waves1to4_counts_dbgap.txt \
  beataml_probit_curve_fits_v4_dbgap.txt \
  beataml_wv1to4_raw_inhibitor_v4_dbgap.txt \
  beataml_wes_wv1to4_mutations_dbgap.txt \
  beataml_wv1to4_clinical.xlsx \
  beataml_waves1to4_sample_mapping.xlsx \
  beataml_drug_families.xlsx \
  beataml2_manuscript_vg_cts.xlsx
do
  if [ -f "$TMP/$f" ]; then
    cp "$TMP/$f" "$DEST/"
  else
    echo "    WARNING: $f not found in the clone -- filename may have changed."
  fi
done

# WGCNA modules (needed to check whether latent dims recover known modules)
if [ -d "$TMP/wgcna" ]; then
  cp -r "$TMP/wgcna" "$DEST/"
fi

rm -rf "$TMP"

echo "==> Verifying that files are real data, not LFS pointers"
BAD=0
while IFS= read -r -d '' file; do
  if head -c 50 "$file" 2>/dev/null | grep -q "git-lfs.github.com"; then
    echo "    POINTER FILE (not data): $file"
    BAD=1
  fi
done < <(find "$DEST" -type f -print0)

if [ "$BAD" -eq 1 ]; then
  echo "ERROR: some files are LFS pointers. Run 'git lfs install' and retry."
  exit 1
fi

echo "==> Recording checksums"
( cd "$DEST" && find . -type f ! -name 'MANIFEST.sha256' \
    -exec shasum -a 256 {} \; | sort -k2 > MANIFEST.sha256 )

echo
echo "Done. Contents of $DEST:"
du -h "$DEST"/* 2>/dev/null | sort -h
echo
echo "Checksums written to $DEST/MANIFEST.sha256 (commit this file -- it is small"
echo "and lets teammates verify they have identical data)."
