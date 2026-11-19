"""Shared data assembly for the interaction screen (Part 1) and the gap analyses (Part 2).

Everything is keyed by `base_samp` (one specimen per patient, see src/load.py) and
cached to data/interim/analysis_frame.pkl, which is gitignored.

    from analysis.common import load_frame
    F = load_frame()
    F["auc"]      429 x 165 AUC matrix (NaN = missing); the drug with zero values is dropped
    F["states"]   429 x 6 published van Galen PC1 cell-state scores
    F["mes"]      429 x 13 WGCNA module eigengenes (M1..M23, grey M0 excluded)
    F["muts"]     429 x genes, 0/1, NaN where the patient has no exome
    F["meta"]     clinical rows for the 429
    F["X"]        429 x 22,843 expression (columns = gene symbols, first occurrence kept)
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.load import CELL_TYPES, COLS, build_modeling_frame, load_pc1_baseline  # noqa: E402

CACHE = ROOT / "data" / "interim" / "analysis_frame.pkl"
SUPP = ROOT / "data" / "raw" / "bottomly2022_supp"
RESULTS = ROOT / "results" / "tables"
FIGS = ROOT / "results" / "figures"
# Clinical consensus calls override the exome for these genes, as the paper does
# (R/preprocess.R, mutation.only.dataset). FLT3-ITD is not callable from the exome.
CLIN_MUTS = {"NPM1": "NPM1", "FLT3_ITD": "FLT3-ITD", "RUNX1": "RUNX1", "ASXL1": "ASXL1", "TP53": "TP53"}


def tissue_pc1(X: np.ndarray, bm: np.ndarray, fit: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """PC1 score of a gene block, standardised within tissue, fit on `fit` rows only.

    Same procedure as the paper's van Galen scores and module eigengenes: z-score
    each gene separately for bone marrow and for blood/leukapheresis, take PC1 of
    the fit rows, project all rows, and orient the sign by mean expression.
    Returns (score, fraction of variance explained by PC1 on the fit rows).
    """
    Z = np.empty_like(X, dtype=float)
    for g in (bm, ~bm):
        if not g.any():          # global scaling passes an all-True mask
            continue
        mu = X[g & fit].mean(0)
        sd = X[g & fit].std(0, ddof=1)
        sd[sd == 0] = 1.0
        Z[g] = (X[g] - mu) / sd
    _, s, vt = np.linalg.svd(Z[fit], full_matrices=False)
    score = Z @ vt[0]
    if np.corrcoef(score, Z.mean(1))[0, 1] < 0:
        score = -score
    return score, float(s[0] ** 2 / (s ** 2).sum())


def _build() -> dict:
    f = build_modeling_frame(verbose=False)
    meta = f["meta"].set_index("base_samp")
    ids = meta.index

    auc = f["y"].set_axis(ids, axis=0)
    auc = auc.loc[:, auc.notna().any()]

    labels = [c[1] for c in f["X"].columns]
    keep = ~pd.Index(labels).duplicated()
    X = f["X"].loc[:, keep].set_axis(ids, axis=0)
    X.columns = pd.Index(labels)[keep]

    states = load_pc1_baseline().reindex(meta[COLS["rna_id"]]).set_axis(ids, axis=0)[CELL_TYPES]

    # Module eigengenes: fit on Waves1+2, project Waves3+4 (same split as the published scores).
    wg = pd.read_csv(ROOT / "data" / "interim" / "wgcna_cur_map.csv")
    bm = meta[COLS["tissue"]].eq("Bone Marrow Aspirate").to_numpy()
    fit = meta[COLS["cohort"]].eq("Waves1+2").to_numpy()
    mes = {}
    for mod, genes in wg[wg.cur_labels != "M0"].groupby("cur_labels").display_label:
        g = [x for x in genes if x in X.columns]
        # global scaling, as the manuscript's predict.modules() does for eigengenes
        mes[mod] = tissue_pc1(X[g].to_numpy(float), np.ones(len(X), bool), fit)[0]
    mes = pd.DataFrame(mes, index=ids)
    mes = mes[sorted(mes.columns, key=lambda m: int(m[1:]))]

    # Mutations: exome calls, then clinical consensus overrides.
    raw = pd.read_csv(ROOT / "data/raw/beataml_wes_wv1to4_mutations_dbgap.txt", sep="\t", low_memory=False)
    raw["base_samp"] = raw.dbgap_sample_id.astype(str).str.removesuffix("D")
    wes = pd.crosstab(raw.base_samp, raw.symbol).clip(upper=1)
    has_wes = meta[COLS["dna_id"]].notna() & meta.index.isin(wes.index)
    muts = wes.reindex(ids).fillna(0).astype(float)
    muts[~has_wes.to_numpy()] = np.nan
    for gene, col in CLIN_MUTS.items():
        clin = meta[col].map({"positive": 1.0, "negative": 0.0})
        muts[gene] = clin.where(clin.notna(), muts[gene] if gene in muts else np.nan)

    return {"meta": meta, "auc": auc, "X": X, "states": states, "mes": mes, "muts": muts}


def load_frame(rebuild: bool = False) -> dict:
    if CACHE.exists() and not rebuild:
        return pickle.loads(CACHE.read_bytes())
    F = _build()
    CACHE.write_bytes(pickle.dumps(F))
    return F


def known_interactions(q: float = 0.10) -> pd.DataFrame:
    """Drug x cell-state x mutation interactions the paper reported (Table S3, top
    interaction per drug x cell type, q < `q`)."""
    s3 = pd.read_excel(SUPP / "S3_drug_cell_type.xlsx", header=1)
    s3 = s3[s3.inter_qval < q].copy()
    s3["gene"] = s3.inter_gene.str.removeprefix("mut.")
    return s3[["inhibitor", "cell_type", "gene", "inter_est", "inter_muts", "inter_qval"]]


if __name__ == "__main__":
    F = load_frame(rebuild=True)
    for k, v in F.items():
        print(f"{k:7s} {v.shape}")
    m = F["muts"]
    print("patients with exome:", int(m.drop(columns=list(CLIN_MUTS)).notna().all(1).sum()))
    print("top mutation counts:\n", m.sum().sort_values(ascending=False).head(15).astype(int).to_string())
