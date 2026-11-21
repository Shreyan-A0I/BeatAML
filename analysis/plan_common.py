"""Shared pieces for the analysis plan (literature/analysis_plan.pdf).

Scores use ONE definition everywhere: genes z-scored within a cohort (Beat AML:
on Waves1+2 rows), projected on PC1 loadings learned in Beat AML Waves1+2. This
lets the same score be computed in TCGA-LAML.
"""
from __future__ import annotations

import pickle
import sys

import numpy as np
import pandas as pd

from analysis.common import ROOT, load_frame
from analysis.part1_screen import imp_soft

CACHE = ROOT / "data/interim/plan_scores.pkl"

# LSC17 (Ng et al., Nature 2016): weights on log2 expression; alias -> current symbol
LSC17 = {"DNMT3B": 0.0874, "ZBTB46": -0.0347, "NYNRIN": 0.00865, "ARHGAP22": -0.0138, "LAPTM4B": 0.00582,
         "MMRN1": 0.0258, "DPYSL3": 0.0284, "FAM30A": 0.0196, "CDK6": -0.0704, "CPXM1": -0.0258,
         "SOCS2": 0.0271, "SMIM24": -0.0226, "EMP1": 0.0146, "BEX3": 0.0465, "CD34": 0.0338,
         "AKR1C3": -0.0402, "ADGRG1": 0.0501}
ALIASES = {"FAM30A": ["KIAA0125"], "BEX3": ["NGFRAP1"], "ADGRG1": ["GPR56"], "SMIM24": ["C19orf77"],
           "NYNRIN": ["KIAA1305"]}


def _z(X, ref):
    sd = ref.std(ddof=1).replace(0, 1)
    return (X - ref.mean()) / sd


def gene_sets() -> dict[str, list[str]]:
    sys.path.insert(0, str(ROOT / "scripts"))
    from replication_vangalen_scores import load_signatures
    sets = {f"ct.{k}": list(v) for k, v in load_signatures().items()}
    wg = pd.read_csv(ROOT / "data/interim/wgcna_cur_map.csv")
    for mod, g in wg[wg.cur_labels != "M0"].groupby("cur_labels").display_label:
        sets[f"me.{mod}"] = [x for x in g if x != "PEAR1"] if mod == "M3" else list(g)
    return sets


def harmonise(E: pd.DataFrame) -> pd.DataFrame:
    """Rename alias columns to the symbols used in LSC17."""
    ren = {}
    for new, olds in ALIASES.items():
        if new not in E.columns:
            for o in olds:
                if o in E.columns:
                    ren[o] = new
                    break
    return E.rename(columns=ren)


def fit_loadings(E_ref: pd.DataFrame, sets, genes_available=None):
    W = {}
    for name, genes in sets.items():
        g = [x for x in genes if x in E_ref.columns and (genes_available is None or x in genes_available)]
        Z = _z(E_ref[g], E_ref[g]).to_numpy(float)
        _, _, vt = np.linalg.svd(Z, full_matrices=False)
        w = vt[0] * np.sign(np.corrcoef(Z @ vt[0], Z.mean(1))[0, 1])
        W[name] = pd.Series(w, index=g)
    return W


def apply_scores(E: pd.DataFrame, ref: pd.DataFrame, W) -> pd.DataFrame:
    out = {}
    for name, w in W.items():
        out[name] = _z(E[w.index], ref[w.index]).to_numpy(float) @ w.to_numpy()
    S = pd.DataFrame(out, index=E.index)
    S = (S - S.loc[ref.index].mean()) / S.loc[ref.index].std()
    S["PEAR1"] = _z(E[["PEAR1"]], ref[["PEAR1"]])["PEAR1"]
    l = [g for g in LSC17 if g in E.columns]
    S["LSC17"] = E[l].to_numpy(float) @ np.array([LSC17[g] for g in l])
    S["LSC17"] = (S.LSC17 - S.loc[ref.index, "LSC17"].mean()) / S.loc[ref.index, "LSC17"].std()
    return S


def beat_expression() -> pd.DataFrame:
    """All manuscript RNA-seq specimens x genes (symbols), log2 normalized."""
    from src.load import load_clinical, load_expression
    clin = load_clinical()
    clin = clin[clin.manuscript_rnaseq.eq("yes")].set_index("dbgap_rnaseq_sample")
    expr = load_expression()
    E = expr[clin.index].T
    E.columns = expr.index.get_level_values("display_label")
    E = harmonise(E.loc[:, ~E.columns.duplicated()])
    return E, clin


def scores(tcga_genes=None, rebuild=False) -> dict:
    """Frozen-weight scores for Beat AML (560 RNA specimens) and TCGA-LAML."""
    if CACHE.exists() and not rebuild:
        return pickle.loads(CACHE.read_bytes())
    from analysis.tcga_pear1 import load_tcga
    E_b, clin = beat_expression()
    E_t, S_t = load_tcga()
    E_t = harmonise(E_t)
    ref_b = E_b[clin.cohort.eq("Waves1+2").to_numpy()]
    W = fit_loadings(ref_b, gene_sets(), set(E_t.columns))
    sc_b = apply_scores(E_b, ref_b, W)
    sc_t = apply_scores(E_t, E_t, W)
    out = {"beat": sc_b, "beat_clin": clin, "tcga": sc_t, "tcga_clin": S_t,
           "lsc17_genes": ([g for g in LSC17 if g in E_b.columns], [g for g in LSC17 if g in E_t.columns])}
    CACHE.write_bytes(pickle.dumps(out))
    return out


def drug_factors(n_comp=8):
    """Soft-impute completed z-AUC on the n=430 cohort, then PCA: the drug-response view.
    Unsupervised and label-free; uses every patient's drug data (transductive)."""
    F = load_frame()
    auc = F["auc"]
    Y = auc.to_numpy(float)
    M = np.isfinite(Y)
    C = imp_soft(np.nan_to_num(Y), M, lam=5)
    Z = (C - C.mean(0)) / C.std(0)
    U, s, vt = np.linalg.svd(Z, full_matrices=False)
    fac = pd.DataFrame(U[:, :n_comp] * s[:n_comp], index=auc.index, columns=[f"df{i + 1}" for i in range(n_comp)])
    zc = pd.DataFrame(Z, index=auc.index, columns=auc.columns)
    return fac, zc, float((s[:n_comp] ** 2).sum() / (s ** 2).sum())
