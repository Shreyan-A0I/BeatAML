"""Loading and joining Beat AML tables.

THE THING THAT MATTERS HERE
---------------------------
Beat AML has three different units of observation and they are easy to confuse:

    patient   (805)   clinical variables, ELN risk, survival
    specimen  (942)   expression, mutations -- several per patient
    specimen x drug   the AUC matrix

Expression is keyed to SPECIMEN, survival to PATIENT. If you cross-validate by
splitting specimens rather than patients, two specimens from one patient can
land on both sides of the split and your performance will be inflated.
`patient_groups()` below exists to prevent that -- pass it to GroupKFold.

In practice the `manuscript_*` flags in the clinical sheet already reduce to one
specimen per patient (manuscript_rnaseq == "yes" gives 560 rows / 560 distinct
patients), so the main analysis frame is patient-level by construction.
`one_specimen_per_patient()` is kept for analyses that relax those flags.

SCHEMA STATUS: verified 2026-10-26 against the real files via
`scripts/inspect_columns.py` (output in reports/schema.txt). Two of the
originally guessed names were wrong -- see the notes on COLS below.
"""

from __future__ import annotations

from pathlib import Path
from typing import TypedDict, cast

import numpy as np
import pandas as pd

RAW = Path("data/raw")

FILES = {
    "expr": RAW / "beataml_waves1to4_norm_exp_dbgap.txt",
    "counts": RAW / "beataml_waves1to4_counts_dbgap.txt",
    "auc": RAW / "beataml_probit_curve_fits_v4_dbgap.txt",
    "raw_inhib": RAW / "beataml_wv1to4_raw_inhibitor_v4_dbgap.txt",
    "mutations": RAW / "beataml_wes_wv1to4_mutations_dbgap.txt",
    "clinical": RAW / "beataml_wv1to4_clinical.xlsx",
    "mapping": RAW / "beataml_waves1to4_sample_mapping.xlsx",
    "drug_families": RAW / "beataml_drug_families.xlsx",
    # Published van Galen PC1 cell-type scores -- the baseline this project
    # benchmarks against. Fit on Waves1+2 and projected onto Waves3+4.
    "vg_scores": RAW / "beataml2_manuscript_vg_cts.xlsx",
    # WGCNA module memberships, exported from the .RData by scripts/export_wgcna.R
    "wgcna": Path("data/interim/wgcna_cur_map.csv"),
}

# ---------------------------------------------------------------------------
# Verified against reports/schema.txt. Corrections from the initial guesses:
#
#   timepoint: was "specimenType" -- WRONG. specimenType is the TISSUE
#       (Bone Marrow Aspirate / Peripheral Blood / Leukapheresis). The
#       disease timepoint lives in diseaseStageAtSpecimenCollection
#       (Initial Diagnosis / Residual / Relapse / Remission / Unknown).
#   drug_id: was "dbgap_dnaseq_sample" -- WRONG as a join key. That column is
#       blank for 2,652 of 63,395 AUC rows. The AUC table must be joined on the
#       suffix-stripped `base_samp` key instead; see `_add_base_samp()`.
#   specimen_id: there is no single "dbgap_sample_id" in the clinical sheet.
#       Only the mutations file uses that name. Expression-keyed work uses
#       dbgap_rnaseq_sample as the specimen identifier.
# ---------------------------------------------------------------------------
COLS = {
    "patient_id": "dbgap_subject_id",                  # clinical, AUC, mapping
    "specimen_id": "dbgap_rnaseq_sample",              # expression-keyed specimen
    "rna_id": "dbgap_rnaseq_sample",                   # expression column names
    "dna_id": "dbgap_dnaseq_sample",                   # mutations (as dbgap_sample_id)
    "drug_id": "base_samp",                            # DERIVED -- see note above
    "inhibitor": "inhibitor",
    "auc": "auc",
    "timepoint": "diseaseStageAtSpecimenCollection",   # corrected
    "tissue": "specimenType",
    "cohort": "cohort",                                # Waves1+2 / Waves3+4 / Both
    "eln": "ELN2017",
    "vital": "vitalStatus",
    "os_days": "overallSurvival",
    "age": "ageAtDiagnosis",
    "sex": "consensus_sex",
}

# Gene-identifier columns at the left edge of the expression/counts matrices.
EXPR_ID_COLS = ["stable_id", "display_label", "description", "biotype"]

# ELN2017 levels that are not real risk calls; the manuscript nulls these out.
ELN_NON_CALLS = [
    "FavorableOrIntermediate", "MissingMutations", "NonInitial",
    "MissingKaryo", "NonAML", "IntermediateOrAdverse",
]

# The six van Galen malignant cell types, ordered along the maturation axis.
CELL_TYPES = [
    "HSC-like", "Progenitor-like", "GMP-like",
    "Promono-like", "Monocyte-like", "cDC-like",
]

_CONFIRMED = True  # verified against reports/schema.txt on 2026-10-26


def _check_confirmed() -> None:
    if not _CONFIRMED:
        raise RuntimeError(
            "COLS in src/load.py has not been verified against the real files.\n"
            "Run `python scripts/inspect_columns.py`, correct COLS, then set "
            "_CONFIRMED = True."
        )


def _require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{path} missing -- run `bash scripts/get_data.sh`")
    # stat() follows symlinks, which is how data/raw is wired on a local clone.
    if path.stat().st_size < 1000 and path.suffix == ".txt":
        raise RuntimeError(
            f"{path} is suspiciously small -- it may be a Git LFS pointer. "
            "Run `git lfs install` and re-fetch."
        )
    return path


def _add_base_samp(df: pd.DataFrame) -> pd.DataFrame:
    """Add the `base_samp` join key used to bridge drug screens and RNA-seq.

    Beat AML specimen IDs look like BA3025D (exome) and BA3025R (RNA) for the
    same physical specimen. The AUC table carries whichever it has -- and for
    2,652 rows the exome ID is blank -- so neither column alone joins cleanly.
    Stripping the trailing D/R gives a key that works on both sides. This
    mirrors `inhibitor.dataset()` in the manuscript's R/preprocess.R.
    """
    dna = df.get(COLS["dna_id"])
    rna = df.get(COLS["rna_id"])
    dna = pd.Series("", index=df.index) if dna is None else dna.astype("string").fillna("").str.strip()
    rna = pd.Series("", index=df.index) if rna is None else rna.astype("string").fillna("").str.strip()
    base = np.where(
        dna.ne("") & dna.ne("nan"),
        dna.str.removesuffix("D"),
        rna.str.removesuffix("R"),
    )
    return df.assign(base_samp=pd.Series(base, index=df.index).astype(str))


def load_expression(log_transform: bool = False) -> pd.DataFrame:
    """Return genes x specimens (22,843 x 707). Transpose before feeding sklearn.

    Values are already library-size normalised and log-scaled in the published
    file, so `log_transform` should stay False unless you know why you want it.

    The 707 columns are NOT all AML patients: 560 carry manuscript_rnaseq=="yes",
    36 are healthy controls (pooled CD34+ / BM MNC), and 111 are AML specimens
    the manuscript excluded. The controls sit at the LEFT edge of the matrix, so
    slicing the first few sample columns gives you controls, not patients. Use
    `drop_controls()` / `manuscript_rna_samples()` to subset deliberately.
    """
    df = pd.read_csv(_require(FILES["expr"]), sep="\t", low_memory=False)
    df = df.set_index([c for c in EXPR_ID_COLS if c in df.columns])
    if log_transform:
        df = df.add(1).transform(np.log2)
    return df


def load_counts() -> pd.DataFrame:
    """Raw gene counts, same layout as `load_expression()`."""
    df = pd.read_csv(_require(FILES["counts"]), sep="\t", low_memory=False)
    return df.set_index([c for c in EXPR_ID_COLS if c in df.columns])


def load_clinical(clean: bool = True) -> pd.DataFrame:
    """One row per specimen (942 x 95), with the manuscript's cleaning applied.

    `clean=True` reproduces `clinical.for.genomic.analysis()` from the paper:
    non-calls in ELN2017 -> NaN, vitalStatus "Unknown" -> NaN, non-positive
    overallSurvival -> NaN, plus a boolean `is_dead`.
    """
    clin = pd.read_excel(_require(FILES["clinical"]))
    if clean:
        clin.loc[clin[COLS["eln"]].isin(ELN_NON_CALLS), COLS["eln"]] = np.nan
        clin.loc[clin[COLS["vital"]].eq("Unknown"), COLS["vital"]] = np.nan
        clin.loc[clin[COLS["os_days"]].le(0), COLS["os_days"]] = np.nan
        clin["is_dead"] = clin[COLS["vital"]].eq("Dead").where(clin[COLS["vital"]].notna())
    return _add_base_samp(clin)


def load_mapping() -> pd.DataFrame:
    """Join key between RNA-seq IDs, drug-screen IDs and patient IDs (979 rows)."""
    return pd.read_excel(_require(FILES["mapping"]))


def control_rna_samples() -> list[str]:
    """The 36 healthy-control RNA-seq columns in the expression matrix.

    `rna_control` is the string "No" for patient samples, so test against "No"
    rather than for null -- `.notna()` matches every row.

    Returns the 37 controls the mapping sheet lists; 36 of them are columns in
    the expression matrix (BA2686R is listed but not released). Intersect with
    `expr.columns` before using these as a column selection.
    """
    m = load_mapping()
    ids = [
        str(rna)
        for rna, flag in zip(m[COLS["rna_id"]], m["rna_control"])
        if pd.notna(rna) and pd.notna(flag) and flag != "No"
    ]
    return sorted(ids)


def manuscript_rna_samples() -> list[str]:
    """The 560 RNA-seq specimens the manuscript analysed (one per patient)."""
    clin = load_clinical()
    return sorted(clin.loc[clin.manuscript_rnaseq.eq("yes"), COLS["rna_id"]].dropna().astype(str))


def drop_controls(expr: pd.DataFrame) -> pd.DataFrame:
    """Remove healthy-control columns from a genes x specimens matrix."""
    return expr.drop(columns=[c for c in control_rna_samples() if c in expr.columns])


def load_auc_long(
    decreasing_only: bool = True,
    converged_only: bool = True,
) -> pd.DataFrame:
    """Long-format probit curve fits, one row per specimen x inhibitor.

    Two filters worth understanding before you turn them off:

    `decreasing_only` drops 8,468 of 63,395 rows whose fitted curve goes UP with
    dose. AUC on an increasing curve does not mean "resistant" the way it does
    elsewhere, so mixing them into a regression target is incoherent.

    `converged_only` drops 14 rows where the probit fit failed.

    One specimen, BA2563, is flagged manuscript_inhibitor == "yes" but has a
    single AUC row in the entire table and that curve is increasing -- so
    `decreasing_only` removes it outright and the honest modeling n is 429,
    not the 430 the clinical flags imply.

    Not filtered here, but check it yourself: `all_gt_50` is True for 45% of
    rows -- the drug never pushed viability below 50% at any tested dose. Those
    AUCs pile up against the ceiling (mean 245 vs 156 for the rest), which is a
    censoring problem, not noise. `paper_inclusion` is True for every row, so it
    is useless as a filter.
    """
    _check_confirmed()
    df = pd.read_csv(_require(FILES["auc"]), sep="\t", low_memory=False)
    keep = pd.Series(True, index=df.index)
    if decreasing_only:
        keep &= df["curve_type"].eq("decreasing")
    if converged_only:
        keep &= df["converged"].astype(str).str.upper().eq("TRUE")
    return _add_base_samp(cast(pd.DataFrame, df[keep]))


def load_auc_matrix(**kwargs) -> pd.DataFrame:
    """Return a specimen x inhibitor AUC matrix (sparse -- many NaNs).

    Indexed by `base_samp`, not by an RNA or exome ID -- see `_add_base_samp()`.
    AUC here is on the published ~0-286 scale, NOT 0-1 and NOT a percentage.
    """
    long = load_auc_long(**kwargs)
    return long.pivot_table(
        index=COLS["drug_id"],
        columns=COLS["inhibitor"],
        values=COLS["auc"],
        aggfunc="mean",
    )


def load_mutations(binary: bool = True) -> pd.DataFrame:
    """Specimen x gene mutation matrix, keyed by base_samp.

    The published variant table uses `dbgap_sample_id` for what the clinical
    sheet calls `dbgap_dnaseq_sample`.
    """
    _check_confirmed()
    muts = pd.read_csv(_require(FILES["mutations"]), sep="\t", low_memory=False)
    muts["base_samp"] = muts.dbgap_sample_id.astype(str).str.removesuffix("D")
    if not binary:
        return muts
    return (
        muts.assign(value=1)
        .pivot_table(index="base_samp", columns="symbol", values="value",
                     aggfunc=lambda x: int(len(x) > 0), fill_value=0)
        .astype("int8")
    )


def load_pc1_baseline() -> pd.DataFrame:
    """The published van Galen PC1 cell-type scores -- the baseline to beat.

    560 specimens x 6 cell types, indexed by RNA-seq ID. Each column is the
    sign-aligned PC1 of the top-30 marker genes for that cell type, z-scored
    within tissue, FIT ON WAVES1+2 AND PROJECTED ONTO WAVES3+4.

    That last point constrains the comparison. These scores saw all of Waves1+2
    during fitting, so re-using them inside a random 5-fold CV leaks. Either
    adopt the paper's Waves1+2 -> Waves3+4 split (then these are valid as
    published), or refit PC1 per fold -- which needs the van Galen top-30 gene
    lists that are NOT in this repo. See reports/schema_notes.md.
    """
    vg = pd.read_excel(_require(FILES["vg_scores"]), sheet_name="Malignant Cell Type Scores")
    return cast(pd.DataFrame, vg.set_index(COLS["rna_id"])[CELL_TYPES])


def load_wgcna_modules() -> pd.DataFrame:
    """Gene -> WGCNA module membership (3,879 genes over 14 labels).

    M0 is the "grey" catch-all of 2,091 unassigned genes, so there are 13 real
    modules. Exclude M0 before claiming a latent dimension "recovered a module".
    """
    path = FILES["wgcna"]
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing -- run `Rscript scripts/export_wgcna.R` to convert "
            "data/raw/wgcna/merged_older_wgcna_kme.RData to CSV."
        )
    return pd.read_csv(path)


def load_drug_families() -> dict[str, pd.DataFrame]:
    """Three sheets: synonyms, drug_family, drug_gene (annotated drug targets)."""
    return pd.read_excel(_require(FILES["drug_families"]), sheet_name=None)


def drug_sample_counts(auc: pd.DataFrame) -> pd.Series:
    """Non-null specimens per inhibitor, descending.

    Run this before anything else. Per-drug n -- not the 942 total -- is the
    real sample size for every per-drug model, and it varies enormously: on the
    n=430 modeling frame the median is 369 but 21 inhibitors have fewer than 50
    specimens and the smallest has 2.
    """
    return auc.notna().sum(axis=0).sort_values(ascending=False)


def filter_drugs(auc: pd.DataFrame, min_n: int = 100) -> pd.DataFrame:
    """Drop inhibitor columns with fewer than `min_n` non-null specimens.

    Needed, not optional. On the n=429 modeling frame the AUC matrix is 46%
    missing, 26 of the 166 inhibitors have fewer than 50 specimens, and some
    have ZERO -- they were screened only on specimens that lack RNA-seq. A
    per-drug model fit on a handful of samples will produce a correlation that
    looks like a result and is not one.
    """
    keep = auc.notna().sum(axis=0) >= min_n
    return auc.loc[:, keep[keep].index]


def one_specimen_per_patient(
    meta: pd.DataFrame,
    prefer: str = "diagnosis",
) -> pd.DataFrame:
    """Keep a single specimen per patient, preferring first-timepoint samples.

    `meta` must carry the patient, specimen and timepoint columns. Matching is
    on diseaseStageAtSpecimenCollection, whose first-timepoint value is
    "Initial Diagnosis" -- so the default `prefer="diagnosis"` works.
    """
    _check_confirmed()
    pid, sid, tp = COLS["patient_id"], COLS["specimen_id"], COLS["timepoint"]
    ranked = meta.assign(
        _rank=meta[tp].astype(str).str.lower().str.contains(prefer).map({True: 0, False: 1})
    )
    return (
        ranked.sort_values([pid, "_rank", sid])
        .groupby(pid, as_index=False)
        .first()
        .drop(columns="_rank")
    )


def patient_groups(meta: pd.DataFrame) -> np.ndarray:
    """Group labels for GroupKFold, so no patient spans a train/test split."""
    _check_confirmed()
    return np.asarray(meta[COLS["patient_id"]].to_numpy())


class ModelingFrame(TypedDict):
    """What `build_modeling_frame()` returns. Keys are fixed so type checkers can see them."""

    expr: pd.DataFrame      # genes x all expression columns (incl. controls)
    auc: pd.DataFrame       # base_samp x inhibitor AUC matrix (sparse)
    clinical: pd.DataFrame  # one row per specimen, cleaned
    meta: pd.DataFrame      # the modeling rows (clinical subset, one per specimen)
    X: pd.DataFrame         # specimens x genes, row-aligned to `meta`
    y: pd.DataFrame         # specimens x inhibitors, row-aligned to `meta`
    groups: np.ndarray      # patient IDs, for GroupKFold


def build_modeling_frame(verbose: bool = True) -> ModelingFrame:
    """Assemble aligned expression / AUC / clinical for modeling.

    Returns a dict so the caller can see exactly how many specimens survived
    each join -- print those numbers in the report, they are a real result.
    """
    _check_confirmed()
    expr = load_expression()
    clin = load_clinical()
    auc = load_auc_matrix()

    rna_yes = clin["manuscript_rnaseq"].eq("yes")
    inh_yes = clin["manuscript_inhibitor"].eq("yes")
    model_meta = cast(pd.DataFrame, clin[rna_yes & inh_yes]).copy()

    expr_cols = set(expr.columns)  # set for fast lookup
    in_expr = model_meta[COLS["rna_id"]].isin(list(expr_cols))
    in_auc = model_meta["base_samp"].isin(list(auc.index))
    usable = cast(pd.DataFrame, model_meta[in_expr & in_auc]).copy()

    if verbose:
        ctrl_in_expr = [c for c in control_rna_samples() if c in expr_cols]
        print(f"expression columns (all)        : {len(expr_cols):,}")
        print(f"  healthy controls              : {len(ctrl_in_expr):,}")
        print(f"  manuscript_rnaseq == yes      : {rna_yes.sum():,}")
        print(f"specimens with drug screen      : {auc.shape[0]:,}")
        print(f"inhibitors                      : {auc.shape[1]:,}")
        print(f"rnaseq AND inhibitor (clinical) : {(rna_yes & inh_yes).sum():,}")
        print(f"overlap present in both matrices: {len(usable):,}")
        dropped = set(model_meta["base_samp"]) - set(usable["base_samp"])
        if dropped:
            print(f"  dropped by AUC QC filters     : {sorted(dropped)}")
        print(f"  distinct patients             : {usable[COLS['patient_id']].nunique():,}")
        print(f"  cohort split                  : {dict(usable[COLS['cohort']].value_counts())}")

    X = expr[usable[COLS["rna_id"]].tolist()].T
    y = auc.loc[usable["base_samp"].tolist()]

    return ModelingFrame(
        expr=expr,
        auc=auc,
        clinical=clin,
        meta=usable,
        X=X,                                  # specimens x genes, aligned
        y=y,                                  # specimens x inhibitors, aligned
        groups=patient_groups(usable),
    )


if __name__ == "__main__":
    build_modeling_frame()
