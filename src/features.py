"""Latent feature builders for the Beat AML drug-response models.

Every builder is FIT ON THE TRAINING COHORT ONLY and then applied to the test
cohort. The default split is the paper's own: fit on Waves1+2, test on
Waves3+4 (see reports/schema_notes.md, Option A). Fitting on all specimens and
then cross-validating would leak test information into the features.

Builders:
    pc1_baseline()          published van Galen PC1 scores (6 cell types), no fitting
    select_genes()          top-variance genes, chosen on the training cohort
    nmf_features()          NMF on non-negative shifted expression, fit on train

Paper features (Bottomly 2022), used by the drug recommender:
    paper_score()           PC1 gene-set score, per-tissue scaling, fit on train
    celltype_scores()       the six van Galen cell-type scores
    module_eigengenes()     WGCNA module eigengenes (13 modules, M0 excluded)
    nonredundant_modules()  drop modules strongly +/- correlated with cell types
    pear1_lsc17()           PEAR1 expression and the LSC17 score
    mutation_matrix()       patient x gene 0/1, clinical consensus overrides
    fusion_matrix()         patient x fusion 0/1
    build_patient_features() all of the above in one patient-indexed table

Everything in the recommender is keyed by PATIENT (`dbgap_subject_id`): the
expression, DNA and drug-screen specimens of one patient are different rows of
the clinical sheet, and the manuscript flags pick one of each per patient.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from sklearn.decomposition import NMF

from .load import (
    CELL_TYPES,
    COLS,
    FILES,
    RAW,
    load_pc1_baseline,
    load_wgcna_modules,
)


def split_by_cohort(
    meta: pd.DataFrame,
    train: str = "Waves1+2",
    test: str = "Waves3+4",
) -> tuple[np.ndarray, np.ndarray]:
    """Return (train_mask, test_mask) boolean arrays over the modeling rows."""
    cohort = meta[COLS["cohort"]].to_numpy()
    return np.asarray(cohort == train, dtype=bool), np.asarray(cohort == test, dtype=bool)


def pc1_baseline(specimen_ids: Iterable[str]) -> pd.DataFrame:
    """Published PC1 cell-type scores, aligned to `specimen_ids` (RNA-seq IDs).

    These were fit by the paper on Waves1+2 and projected onto Waves3+4, so they
    are only leak-free under the Option A split. Do not re-fit them inside CV.
    """
    pc1 = load_pc1_baseline()
    return pc1.reindex(list(specimen_ids))


def select_genes(X_train: pd.DataFrame, n_top: int = 2000) -> list[Any]:
    """Top `n_top` genes by variance, computed on the training rows only.

    Returns the column labels of `X_train` (for this dataset each label is a
    4-part tuple: stable_id, display_label, description, biotype).
    """
    var = np.asarray(X_train.var(axis=0), dtype=float)
    order = np.argsort(-var)[:n_top]
    return [X_train.columns[i] for i in order]


def nmf_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    k: int = 10,
    random_state: int = 0,
    n_top: int = 2000,
) -> tuple[pd.DataFrame, pd.DataFrame, NMF]:
    """Fit NMF on training specimens, transform both cohorts.

    Expression here is log-scaled and therefore has negative values, but NMF
    needs non-negative input. We shift every gene by its training-set minimum
    so the smallest training value is zero. The shift is a per-gene constant,
    so it does not change which genes are variable, only their zero point.
    Test values are shifted by the SAME training minimums, and clipped at zero
    so an unseen low value cannot make the input negative.
    """
    genes = select_genes(X_train, n_top=n_top)
    tr = X_train.loc[:, genes].to_numpy(dtype=float)
    te = X_test.loc[:, genes].to_numpy(dtype=float)

    shift = tr.min(axis=0)
    tr_nn = np.clip(tr - shift, 0.0, None)
    te_nn = np.clip(te - shift, 0.0, None)

    # sklearn's NMF signature has no annotations and infers n_components as str
    # from its "auto" default, so the int is cast for the type checker only.
    model = NMF(
        n_components=cast(Any, k),
        init="nndsvda",
        max_iter=1000,
        random_state=random_state,
    )
    W_tr = np.asarray(model.fit_transform(tr_nn), dtype=float)
    W_te = np.asarray(model.transform(te_nn), dtype=float)

    cols = [f"NMF{i + 1}" for i in range(k)]
    return (
        pd.DataFrame(W_tr, index=X_train.index, columns=pd.Index(cols)),
        pd.DataFrame(W_te, index=X_test.index, columns=pd.Index(cols)),
        model,
    )


# ---------------------------------------------------------------------------
# Paper features (Bottomly et al. 2022)
# ---------------------------------------------------------------------------

VG_SUPP = RAW / "mmc3.xlsx"  # van Galen 2019 Table S3
VG_N_TOP = 30

# 0-based column of each tumour-derived per-cell-type signature in Table S3.
# GMP-like has no per-cell-type list: its cell says "see combined signature",
# so it uses the tumour-derived combined GMP-like column instead.
VG_SIG_COL = {
    "HSC-like": 7,
    "Progenitor-like": 8,
    "GMP-like": 5,
    "Promono-like": 10,
    "Monocyte-like": 11,
    "cDC-like": 12,
}

# LSC17 (Ng et al. 2016) coefficients. SMIM24 is called C19orf77 in this
# expression file (same Ensembl ID), as in the manuscript's compute.lsc17().
LSC17_COEFS = {
    "DNMT3B": 0.0874, "ZBTB46": -0.0347, "NYNRIN": 0.00865, "ARHGAP22": -0.0138,
    "LAPTM4B": 0.00582, "MMRN1": 0.0258, "DPYSL3": 0.0284, "KIAA0125": 0.0196,
    "CDK6": -0.0704, "CPXM1": -0.0258, "SOCS2": 0.0271, "C19orf77": -0.0226,
    "EMP1": 0.0146, "NGFRAP1": 0.0465, "CD34": 0.0338, "AKR1C3": -0.0402,
    "GPR56": 0.0501,
}

# Clinical sheet ELN-style consensus calls that override the exome matrix,
# as in the manuscript's mutation.only.dataset(): PCR assays for NPM1 and
# FLT3-ITD, manually adjudicated calls for RUNX1/ASXL1/TP53.
# The paper's 9 non-redundant WGCNA modules (Fig 6A), used as a FIXED list.
# Bottomly et al. chose them on all 560 RNA-seq specimens, i.e. including
# Waves3+4 EXPRESSION (no drug or outcome data). We adopt the published choice
# rather than re-deriving it, because the pruning rule is unstable at its
# threshold (see nonredundant_modules()). State this in the write-up.
PAPER_MODULES = ["M3", "M7", "M8", "M9", "M12", "M14", "M16", "M21", "M23"]

_CONSENSUS_POSITIVE = {"NPM1": "NPM1", "FLT3_ITD": "FLT3-ITD"}
_CONSENSUS_NOTNULL = ["RUNX1", "ASXL1", "TP53"]


def load_signatures(n_top: int = VG_N_TOP, path: Path = VG_SUPP) -> dict[str, list[str]]:
    """Top-`n_top` gene signature per malignant cell type from van Galen Table S3.

    Returns {cell type: [gene symbols]} in Table S3 order. Raises AssertionError
    if a column header does not match its expected cell type, or a column has
    fewer than `n_top` genes.
    """
    raw = pd.read_excel(path, header=None)
    names = raw.iloc[1]  # row 2 of the sheet holds the column names
    sigs = {}
    for ct, j in VG_SIG_COL.items():
        assert names[j] == ct, f"Table S3 column {j} is {names[j]!r}, expected {ct!r}"
        genes = raw.iloc[2:2 + n_top, j].dropna().astype(str).str.strip().tolist()
        assert len(genes) == n_top, f"{ct}: found {len(genes)} genes"
        sigs[ct] = genes
    return sigs


def paper_score(X: np.ndarray, tissue_bm: np.ndarray, train: np.ndarray) -> np.ndarray:
    """PC1 score of one gene set, with centring and scaling per tissue group.

    Inputs:
        X: float array (n_specimens, n_genes), expression of the gene set only.
        tissue_bm: bool array (n_specimens,), True for bone marrow aspirate.
            Pass all-True for a single global scaling (WGCNA eigengenes).
        train: bool array (n_specimens,), True for specimens used to fit.

    Output: PC1 score per specimen, signed to correlate positively with the
    mean expression of the set. Means, SDs and the rotation come from training
    specimens only; everyone else is projected, so test specimens never
    influence the fit. Reproduces the published scores exactly (r = 1.0).
    """
    Z = np.empty_like(X, dtype=float)
    for group in (tissue_bm, ~tissue_bm):
        if not group.any():
            continue
        mu = X[group & train].mean(axis=0)
        sd = X[group & train].std(axis=0, ddof=1)
        sd[sd == 0] = 1.0
        Z[group] = (X[group] - mu) / sd
    _, _, vt = np.linalg.svd(Z[train], full_matrices=False)
    score = Z @ vt[0]
    # PC1 sign is arbitrary: align it with the average expression of the set.
    if np.corrcoef(score, Z.mean(axis=1))[0, 1] < 0:
        score = -score
    return score


def expression_by_symbol(X: pd.DataFrame) -> pd.DataFrame:
    """Re-label expression columns from 4-tuples to gene symbols (display_label)."""
    if isinstance(X.columns, pd.MultiIndex) or isinstance(X.columns[0], tuple):
        labels = [c[1] for c in X.columns]
        X = X.set_axis(labels, axis=1)
    return X.loc[:, ~X.columns.duplicated()]


def celltype_scores(E: pd.DataFrame, tissue_bm: np.ndarray, train: np.ndarray) -> pd.DataFrame:
    """The six van Galen PC1 cell-type scores. E: patients x gene symbols."""
    sigs = load_signatures()
    out = {}
    for ct in CELL_TYPES:
        genes = [g for g in sigs[ct] if g in E.columns]
        out[ct] = paper_score(E[genes].to_numpy(float), tissue_bm, train)
    return pd.DataFrame(out, index=E.index)


def module_eigengenes(E: pd.DataFrame, train: np.ndarray) -> pd.DataFrame:
    """WGCNA module eigengenes (manuscript's predict.modules()): global scaling.

    M0, the grey catch-all, is excluded. Columns are ordered M1, M3, ... M23.
    """
    wg = load_wgcna_modules()
    one_group = np.ones(len(E), dtype=bool)
    me = {}
    for mod, genes in wg.groupby("cur_labels"):
        if mod == "M0":
            continue
        g = [x for x in genes.display_label if x in E.columns]
        me[mod] = paper_score(E[g].to_numpy(float), one_group, train)
    cols = sorted(me, key=lambda m: int(str(m)[1:]))
    return pd.DataFrame(me, index=E.index)[cols]


def nonredundant_modules(
    me: pd.DataFrame, ct: pd.DataFrame, train: np.ndarray, threshold: float = 0.6
) -> list[str]:
    """Modules to KEEP: drop any with |r| > threshold of both signs vs cell types.

    Same rule as mutation.expression.features() in the manuscript, on the
    patients in `train`. UNSTABLE at threshold 0.6: M12's correlation with
    Promono-like sits at -0.60 to -0.65, so the result depends on who is in
    `train`:
        all 560 RNA-seq (the paper)   drops M1, M4, M6, M11   -> 9 kept
        372 Waves1+2 RNA-seq          drops M1, M4, M6, M12   -> 9 kept
        266 Waves1+2 modeling         drops M1, M4, M6, M11, M12 -> 8 kept
    build_patient_features() therefore uses the paper's published list
    (PAPER_MODULES) by default rather than re-running this rule.
    """
    keep = []
    for m in me.columns:
        r = ct[train].corrwith(me.loc[train, m])
        strong = np.sign(r[r.abs() > threshold])
        if strong.nunique() <= 1:
            keep.append(m)
    return keep


def pear1_lsc17(E: pd.DataFrame) -> pd.DataFrame:
    """PEAR1 expression and the LSC17 score (fixed published weights, no fitting)."""
    missing = [g for g in LSC17_COEFS if g not in E.columns]
    assert not missing, f"LSC17 genes missing from expression: {missing}"
    lsc = E[list(LSC17_COEFS)].to_numpy(float) @ np.array(list(LSC17_COEFS.values()))
    return pd.DataFrame({"PEAR1": E["PEAR1"].to_numpy(float), "LSC17": lsc}, index=E.index)


def mutation_matrix(clinical: pd.DataFrame) -> tuple[pd.DataFrame, set[str]]:
    """Patient x gene 0/1 mutation matrix, indexed by dbgap_subject_id.

    `clinical` is the cleaned clinical sheet (load_clinical()). Uses the one
    DNA-seq specimen per patient (manuscript_dnaseq == "yes"), then overwrites
    NPM1, FLT3_ITD, RUNX1, ASXL1, TP53 with the clinical consensus calls.
    Patients without DNA-seq are absent: reindex and they become NaN.

    Also returns the gene symbols covered by the custom capture panel used for
    Waves3+4. Only those genes are comparable across cohorts, so feature
    selection should be restricted to them.
    """
    pid = COLS["patient_id"]
    dna = clinical[clinical["manuscript_dnaseq"].eq("yes")].drop_duplicates(pid).set_index(pid)
    muts = pd.read_csv(FILES["mutations"], sep="\t", low_memory=False)
    mm = dna.reset_index()[[pid, COLS["dna_id"]]].merge(
        muts, left_on=COLS["dna_id"], right_on="dbgap_sample_id")
    M = pd.crosstab(mm[pid], mm["symbol"]).clip(upper=1).reindex(dna.index, fill_value=0)
    for col, src in _CONSENSUS_POSITIVE.items():
        M[col] = dna[src].eq("positive").astype(int)
    for g in _CONSENSUS_NOTNULL:
        M[g] = dna[g].notna().astype(int)
    panel = set(mm.loc[mm["capture_type"].eq("Nimblegen Custom Capture"), "symbol"])
    panel |= set(_CONSENSUS_POSITIVE) | set(_CONSENSUS_NOTNULL)
    return M.astype("int8"), panel


def fusion_matrix(clinical: pd.DataFrame) -> pd.DataFrame:
    """Patient x consensus AML fusion 0/1, indexed by dbgap_subject_id.

    "N/A" means no karyotype/RNA to call from, so those patients are dropped
    (NaN after reindexing), matching fusion.only.dataset() in the manuscript.
    """
    pid = COLS["patient_id"]
    used = clinical[clinical["used_manuscript_analyses"].eq("yes")]
    fus = used[[pid, "consensusAMLFusions"]].fillna({"consensusAMLFusions": ""})
    fus = fus[fus["consensusAMLFusions"] != "N/A"]
    per_pt = fus.groupby(pid)["consensusAMLFusions"].agg(
        lambda s: ";".join(sorted({x for x in s if x})))
    F = pd.get_dummies(per_pt[per_pt != ""]).reindex(per_pt.index, fill_value=0)
    return F.astype("int8")


def clinical_features(meta: pd.DataFrame) -> pd.DataFrame:
    """Age, de novo status and tissue for the modeling patients."""
    pid = COLS["patient_id"]
    m = meta.set_index(pid)
    return pd.DataFrame({
        "age": m[COLS["age"]].astype(float),
        "is_denovo": m["isDenovo"].eq("TRUE").astype(int).where(m["isDenovo"].isin(["TRUE", "FALSE"])),
        "is_bm": m[COLS["tissue"]].eq("Bone Marrow Aspirate").astype(int),
    })


def build_patient_features(
    meta: pd.DataFrame,
    X: pd.DataFrame,
    clinical: pd.DataFrame,
    train: np.ndarray,
    ref_X: pd.DataFrame | None = None,
    ref_meta: pd.DataFrame | None = None,
    modules: list[str] | None = PAPER_MODULES,
    min_carriers: int = 10,
) -> pd.DataFrame:
    """One patient-indexed table of every feature block.

    Inputs:
        meta, X: from build_modeling_frame() (row-aligned; X is specimens x genes).
        clinical: load_clinical() output (for DNA and fusion specimens).
        train: bool mask over meta rows. Mutation/fusion carrier counts use
            only these patients.
        ref_X, ref_meta: optional row-aligned REFERENCE specimens to fit the
            expression features on (PC1 rotations, centring, scaling). Pass the
            372 Waves1+2 RNA-seq specimens to reproduce the published scores
            exactly; they carry no drug data and no Waves3+4 patient. Without
            them, expression features are fit on the `train` rows of `meta`.
        modules: WGCNA modules to keep. Default = the paper's published 9;
            None = re-derive with nonredundant_modules() on the fit patients.

    Columns are prefixed by block so models can select them:
        ct.*   six cell-type scores          me.*   module eigengenes
        rna.*  PEAR1, LSC17                  clin.* age, de novo, bone marrow
        mut.*  mutations (>= min_carriers training carriers, on the capture panel)
        fus.*  fusions (>= min_carriers training carriers)
    Patients without DNA-seq get NaN in mut.*/fus.*.
    """
    pid = COLS["patient_id"]
    ids = meta[pid].to_numpy()
    E = expression_by_symbol(X).set_axis(ids, axis=0)
    tissue_bm = meta[COLS["tissue"]].eq("Bone Marrow Aspirate").to_numpy()

    # Rows used to fit the expression features: `train` rows of meta, or the
    # reference set (appended below meta for patients not already in it).
    if ref_X is None:
        E_all, bm_all, fit = E, tissue_bm, train
    else:
        assert ref_meta is not None, "ref_meta must accompany ref_X"
        ref_ids = ref_meta[pid].to_numpy()
        R = expression_by_symbol(ref_X).set_axis(ref_ids, axis=0)
        ref_bm = ref_meta[COLS["tissue"]].eq("Bone Marrow Aspirate").to_numpy()
        extra = ~np.isin(ref_ids, ids)
        E_all = pd.concat([E, R.loc[extra, E.columns]])
        bm_all = np.r_[tissue_bm, ref_bm[extra]]
        fit = np.r_[np.isin(ids, ref_ids), np.ones(extra.sum(), dtype=bool)]

    ct_all = celltype_scores(E_all, bm_all, fit)
    me_all = module_eigengenes(E_all, fit)
    keep = modules if modules is not None else nonredundant_modules(me_all, ct_all, fit)
    ct, me = ct_all.iloc[: len(ids)], me_all.iloc[: len(ids)][keep]
    rna = pear1_lsc17(E)
    clin = clinical_features(meta).reindex(ids)

    M, panel = mutation_matrix(clinical)
    M = M.reindex(ids)
    train_ids = ids[train]
    counts = M.loc[train_ids].sum()
    mut_cols = [g for g in M.columns if counts[g] >= min_carriers and g in panel]
    F = fusion_matrix(clinical).reindex(ids)
    fcounts = F.loc[train_ids].sum()
    fus_cols = [f for f in F.columns if fcounts[f] >= min_carriers]

    blocks = [
        ct.add_prefix("ct."), me.add_prefix("me."), rna.add_prefix("rna."),
        clin.add_prefix("clin."), M[mut_cols].add_prefix("mut."), F[fus_cols].add_prefix("fus."),
    ]
    out = pd.concat(blocks, axis=1)
    out.index.name = pid
    return out


def alteration_table(clinical: pd.DataFrame, patient_ids: Iterable[Any]) -> pd.DataFrame:
    """EVERY mutation and fusion per patient (no carrier threshold), prefixed `alt.`.

    For rule-based recommenders: a rare actionable lesion (e.g. IDH1, KIT) matters
    for that patient even if too rare to be a model feature. NaN = no DNA-seq.
    """
    ids = list(patient_ids)
    M, _ = mutation_matrix(clinical)
    F = fusion_matrix(clinical)
    out = pd.concat([M.reindex(ids), F.reindex(ids)], axis=1).add_prefix("alt.")
    out.index.name = COLS["patient_id"]
    return out
