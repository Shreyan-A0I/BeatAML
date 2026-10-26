# Beat AML 2.0 — verified schema and data notes

Verified 2026-10-26 against the real files. Raw header dumps are in
`reports/schema.txt` (regenerate with `python scripts/inspect_columns.py`).
This file records what the headers *mean*, which is the part that bit us.

## Corrections to the scaffold's guessed column names

| key | guessed | actual | why the guess was wrong |
|---|---|---|---|
| `timepoint` | `specimenType` | `diseaseStageAtSpecimenCollection` | `specimenType` is the **tissue** (Bone Marrow Aspirate 527 / Peripheral Blood 393 / Leukapheresis 22), not the timepoint. The timepoint is Initial Diagnosis 580 / Residual 202 / Relapse 63 / Remission 47. |
| `drug_id` | `dbgap_dnaseq_sample` | derived `base_samp` | That column is **blank for 2,652 of 63,395** AUC rows. Joining on it silently drops them. |
| `specimen_id` | `dbgap_sample_id` | `dbgap_rnaseq_sample` | No `dbgap_sample_id` exists in the clinical sheet — only the mutations file uses that name, for what clinical calls `dbgap_dnaseq_sample`. |

The other eight keys were correct as guessed.

### The `base_samp` join key
Specimen IDs are `BA3025D` (exome) and `BA3025R` (RNA) for one physical
specimen. The AUC table carries whichever it has, sometimes neither-complete.
Strip the trailing `D`/`R` to get a key that joins both sides. This is what
`inhibitor.dataset()` does in the manuscript's `R/preprocess.R`; `src/load.py`
reproduces it in `_add_base_samp()`.

## File inventory

| file | shape | notes |
|---|---|---|
| `beataml_waves1to4_norm_exp_dbgap.txt` | 22,843 genes × 707 samples | 4 ID cols (`stable_id`, `display_label`, `description`, `biotype`) then samples. **Already normalised and log-scaled** — do not log again. |
| `beataml_waves1to4_counts_dbgap.txt` | same layout | raw counts, for a VAE with a NB likelihood |
| `beataml_probit_curve_fits_v4_dbgap.txt` | 63,395 × 27 | AUC per specimen × inhibitor, 166 inhibitors |
| `beataml_wv1to4_raw_inhibitor_v4_dbgap.txt` | 555,583 × 12 | per-well viability, only if refitting curves |
| `beataml_wes_wv1to4_mutations_dbgap.txt` | 11,721 × 32 | **whole-exome**, not targeted panel (the proposal says "targeted") |
| `beataml_wv1to4_clinical.xlsx` | 942 × 95 | one row per specimen |
| `beataml_waves1to4_sample_mapping.xlsx` | 979 × 12 | ID crosswalk + `rna_control` |
| `beataml_drug_families.xlsx` | 3 sheets | `synonyms` 86, `drug_family` 395, `drug_gene` 651 |
| `beataml2_manuscript_vg_cts.xlsx` | 560 × 7 | **the published PC1 baseline scores** |
| `wgcna/merged_older_wgcna_kme.RData` | 3,879 genes / 14 labels | R workspace — convert with `Rscript scripts/export_wgcna.R` |

## The numbers that actually constrain the project

```
patients                                805
specimens                               942
expression columns                      707
  ├─ manuscript_rnaseq == "yes"         560   (= 560 distinct patients)
  ├─ healthy controls                    36   (pooled CD34+ / BM MNC)
  └─ AML excluded from the manuscript   111
manuscript_inhibitor == "yes"           512
rnaseq AND inhibitor                    430
  └─ after AUC QC filters               429   <-- the real modeling n
       Waves1+2 266 / Waves3+4 163
```

**The modeling set is n=429, not 942 or 805.** The proposal's "805 patients,
942 specimens" describes the release, not the analysable intersection. Put 429
in the write-up.

The `manuscript_rnaseq`/`manuscript_inhibitor` flags already reduce to one
specimen per patient, so the proposal's one-specimen filter is satisfied for
free. Keep `patient_groups()` for any analysis that relaxes the flags.

## Traps found in the data

1. **Healthy controls sit at the left edge of the expression matrix.** The
   first sample columns (`BA2392R`, `BA2611R`, `BA2506R`, …) are all controls.
   Slicing the first N sample columns to "peek at patients" gives you controls.
   Note `rna_control` is the **string "No"** for patients, so `.notna()` matches
   every row — test `!= "No"`.

2. **AUC is on a ~0–286 scale**, not 0–1 and not a percentage.

3. **45% of curves never worked.** `all_gt_50 == True` for 28,758 of 63,395
   rows: viability stayed above 50% at every dose. Mean AUC 245 vs 156 for the
   rest. That is a ceiling/censoring problem, not noise, and it will dominate an
   RMSE comparison if ignored.

4. **8,468 curves are `curve_type == "increasing"`** — viability rises with
   dose. AUC there does not mean resistance. `load_auc_long()` drops them by
   default.

5. **`paper_inclusion` is `True` for every row** — useless as a filter.

6. **Per-drug n varies wildly.** On the n=429 frame the AUC matrix is 46%
   missing; median per-drug n is 289, but 26 inhibitors have <50 specimens and
   one has 0. Use `filter_drugs(y, min_n=…)`: min_n=100 keeps 119 inhibitors,
   min_n=300 keeps 77.

7. **`cohort` has a third level, `Both` (23 specimens).** The manuscript's R
   code filters `cohort == "Waves1+2"` vs `== "Waves3+4"`, so those 23 are
   silently dropped. Decide deliberately.

8. **`specimenType` has three levels, not two.** The paper's PC1 code z-scores
   `== "Bone Marrow Aspirate"` against everything else, lumping the 22
   Leukapheresis specimens with Peripheral Blood.

9. **`BA2563`** is flagged `manuscript_inhibitor == "yes"` but has exactly one
   AUC row in the whole table, and it is an increasing curve — hence 430 → 429.

10. **ELN2017 needs cleaning.** Six of its nine levels are non-calls
    (`NonInitial` 333, `NonAML` 29, `FavorableOrIntermediate` 15, …). Real
    calls: Adverse 234 / Favorable 193 / Intermediate 121. `load_clinical()`
    nulls the rest, matching the paper.

11. **`vitalStatus` has "Unknown" (32)** and `overallSurvival` has non-positive
    values; both are nulled by `load_clinical()`, matching the paper.

12. **WGCNA M0 is the grey catch-all** — 2,091 of 3,879 genes are unassigned.
    There are 13 real modules. Exclude M0 before claiming module recovery.

## The PC1 baseline, and the one real design decision

`beataml2_manuscript_vg_cts.xlsx` holds the published van Galen PC1 scores for
all 560 manuscript RNA-seq specimens — including **all 429** of our modeling
set. Verified: zero-centred, with the expected maturation gradient
(Progenitor-like vs Monocyte-like r = −0.72, HSC-like vs Promono-like r = −0.58).

But those scores were **fit on Waves1+2 and projected onto Waves3+4**. So:

- **Option A — adopt the paper's cohort split.** Fit the VAE/NMF on Waves1+2
  (266), test on Waves3+4 (163). The published PC1 scores are then a valid,
  leak-free baseline exactly as distributed, and the comparison is
  apples-to-apples with the paper. Costs you the 5-fold CV the proposal
  specifies, and the test set is 163 specimens.
- **Option B — patient-grouped 5-fold CV as the proposal says.** Then the
  published PC1 scores leak (they saw all of Waves1+2 during fitting) and must
  be **refit inside each fold** — which needs the van Galen top-30 gene lists
  that are **not in this repo** (see below).

Doing both is also defensible: A as the headline, B as a robustness check.

## Missing from the repo

1. **van Galen et al. 2019 Table S3** — `1-s2.0-S0092867419300947-mmc3.xlsx`.
   This holds the top-30 marker genes per malignant cell type, i.e. the
   definition of the PC1 baseline. Needed for Option B above, and to run the
   manuscript's own pipeline. Public Elsevier supplement, no login:
   `https://ars.els-cdn.com/content/image/1-s2.0-S0092867419300947-mmc3.xlsx`
   Parsing is fiddly — see `parse.van.galen.supp()` in the manuscript's
   `R/preprocess.R`: it drops columns 1–4 and 14, promotes row 1 to the header,
   drops rows 1 and 51, and ranks within column.

2. **TCGA-LAML** — the proposal's independent survival-validation cohort. Not
   present and no fetch script. GDC open tier, no application needed.

3. `beataml2_manuscript-main/` has **no `data/` directory**, so the R pipeline
   cannot run as-is. Its `_targets.R` expects the nine files plus the van Galen
   supplement under `data/`. Only needed if you want to reproduce the paper's
   figures rather than just its baseline.

## Local setup deviations from the scaffold

- `scripts/get_data.sh` was **not** used; the data was already downloaded to
  `../beataml2.0_data` (a complete, LFS-pulled clone — all files are real, not
  pointers). `data/raw/*` are **symlinks** into it, so nothing is duplicated.
  Re-running `get_data.sh` would overwrite those symlinks with copies (+430 MB).
- `environment.yml` is the conda path and **conda is installed**
  (`/opt/anaconda3`, initialised in `.zshrc`), so `conda env create -f
  environment.yml` remains available. We are nonetheless running `.venv` with
  `requirements.txt` (the pip mirror) on **Python 3.13**, where
  environment.yml pins 3.11. All packages resolved, including
  scikit-survival 0.28.0 and torch 2.14.1. Keep `environment.yml` as-is — it is
  the reproducible path for teammates, and the two are not in conflict.
- The README references a `.devcontainer/` that does not exist.
- `src/` contains only `load.py`; the README's layout also lists features,
  models and evaluation modules. No `__init__.py` (namespace package — fine).
- The project is **not a git repo yet** (`git init` not run).
