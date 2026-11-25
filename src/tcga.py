"""External survival validation of the predicted cytarabine-resistance score in TCGA-LAML.

    python -m src.tcga

Pre-specified test (from the Beat AML result HR 1.33/SD, adjusted for ELN + age):
does the FROZEN M1 model's predicted cytarabine z (fit on Beat AML Waves1+2 only,
exactly as for the Waves3+4 holdout) stratify overall survival in TCGA-LAML,
adjusted for an approximate ELN2017 risk group and age?

Data (public, cBioPortal; see data/raw/tcga_laml/README): PanCanAtlas RNA-seq
(RSEM), mutation calls, OS; karyotype and age from the 2013 study annotation.

Feature mapping (different platform, so nothing is copied across on raw scale):
  ct.*/me.*   TCGA genes z-scored within TCGA, projected on the PC1 loadings
              learned on Beat AML's Waves1+2 reference set (same feature definition)
  all expression features are then moment-matched to the Beat AML training patients
  mut.*       nonsynonymous calls; FLT3_ITD = FLT3 in-frame insertion in codons 570-610
  fus.*       from karyotype: t(8;21), inv(16), t(15;17), t(9;11)
  clin.*      age; de novo = 1 and bone marrow = 1 for all (TCGA-LAML: de novo, marrow)

Writes results/tables/tcga_{features,cox}.csv
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .features import (CELL_TYPES, LSC17_COEFS, PAPER_MODULES, expression_by_symbol, load_signatures)
from .features import alteration_table
from .load import COLS, load_wgcna_modules
from .models import PerDrugRidge
from .run_baselines import OUT
from .targets import load_recommender_data, select_drugs

RAW = Path("data/raw/tcga_laml")
DRUG = "Cytarabine"
NONSYN = {"Missense_Mutation", "Nonsense_Mutation", "Frame_Shift_Ins", "Frame_Shift_Del", "Splice_Site",
          "In_Frame_Ins", "In_Frame_Del", "Nonstop_Mutation", "Translation_Start_Site"}


# ----------------------------------------------------------------- PC1 loadings from Beat AML

def _loading(X: np.ndarray, tissue_bm: np.ndarray) -> np.ndarray:
    """Same fit as features.paper_score (per-tissue z, PC1, sign to mean), returning the loading vector."""
    Z = np.empty_like(X, dtype=float)
    for g in (tissue_bm, ~tissue_bm):
        if g.any():
            sd = X[g].std(axis=0, ddof=1); sd[sd == 0] = 1.0
            Z[g] = (X[g] - X[g].mean(axis=0)) / sd
    v = np.linalg.svd(Z, full_matrices=False)[2][0]
    return -v if np.corrcoef(Z @ v, Z.mean(axis=1))[0, 1] < 0 else v


def gene_sets() -> dict[str, tuple[list[str], bool]]:
    """feature name -> (genes, use per-tissue scaling)."""
    sigs = load_signatures()
    out = {f"ct.{ct}": (sigs[ct], True) for ct in CELL_TYPES}
    wg = load_wgcna_modules()
    for m in PAPER_MODULES:
        out[f"me.{m}"] = (wg.loc[wg.cur_labels.eq(m), "display_label"].tolist(), False)
    return out


def project(E_ref: pd.DataFrame, ref_bm: np.ndarray, E_new: pd.DataFrame) -> pd.DataFrame:
    """Scores for E_new using Beat AML loadings, restricted to genes present in both."""
    out, cov = {}, {}
    for name, (genes, by_tissue) in gene_sets().items():
        g = [x for x in genes if x in E_ref.columns and x in E_new.columns and E_new[x].notna().all()]
        bm = ref_bm if by_tissue else np.ones(len(E_ref), bool)
        v = _loading(E_ref[g].to_numpy(float), bm)
        Zn = E_new[g].to_numpy(float)
        sd = Zn.std(axis=0, ddof=1); sd[sd == 0] = 1.0
        out[name] = ((Zn - Zn.mean(axis=0)) / sd) @ v
        cov[name] = len(g) / len([x for x in genes if x in E_ref.columns])
    print("gene coverage of TCGA vs Beat AML per set:", {k: round(v, 2) for k, v in cov.items()})
    return pd.DataFrame(out, index=E_new.index)


# ----------------------------------------------------------------- TCGA tables

def _clin(study_file: str) -> pd.DataFrame:
    c = json.load(open(RAW / study_file))
    return pd.DataFrame(c).pivot(index="patientId", columns="clinicalAttributeId", values="value")


def load_tcga() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(log2 expression patients x genes, mutation long table, clinical)."""
    rs = pd.read_csv(RAW / "data_mrna_seq_v2_rsem.txt", sep="\t")
    rs = rs.dropna(subset=["Hugo_Symbol"]).drop(columns="Entrez_Gene_Id").groupby("Hugo_Symbol").mean()
    E = np.log2(rs.T.clip(lower=0) + 1)
    E.index = E.index.str[:12]
    E = E[~E.index.duplicated()]
    m = json.load(open(RAW / "mutations_api.json"))
    muts = pd.DataFrame({"patient": [x["patientId"] for x in m], "gene": [x["gene"]["hugoGeneSymbol"] for x in m],
                         "type": [x["mutationType"] for x in m], "aa": [x.get("proteinChange") or "" for x in m]})
    muts = muts[muts["type"].isin(NONSYN)]
    clin = _clin("laml2013_clinical_PATIENT.json")
    pan = _clin("clinical_patient_api.json")[["OS_MONTHS", "OS_STATUS"]].add_prefix("pan_")
    clin = clin.join(pan, how="left")
    return E, muts, clin


def _is_itd(row) -> bool:
    if row.gene != "FLT3" or row.type != "In_Frame_Ins":
        return False
    pos = [int(x) for x in re.findall(r"(\d+)", row.aa)[:1]]
    return bool(pos) and 560 <= pos[0] <= 620


def genomic_features(muts: pd.DataFrame, clin: pd.DataFrame, patients: pd.Index) -> pd.DataFrame:
    M = pd.crosstab(muts["patient"], muts["gene"]).clip(upper=1).reindex(patients, fill_value=0)
    itd = muts.apply(_is_itd, axis=1)
    M["FLT3_ITD"] = muts[itd].groupby("patient").size().reindex(patients).notna().astype(int)
    # Beat AML's mut.FLT3 comes from variant calls with the ITD called separately: count non-ITD FLT3 calls only
    M["FLT3"] = muts[~itd & muts.gene.eq("FLT3")].groupby("patient").size().reindex(patients).notna().astype(int)
    cebpa_n = muts[muts.gene.eq("CEBPA")].groupby("patient").size().reindex(patients, fill_value=0)
    cyto = clin["CYTOGENETIC_ABNORMALITY_TYPE"].reindex(patients).fillna("")
    F = pd.DataFrame({"RUNX1-RUNX1T1": cyto.str.contains(r"t\(8;21\)"), "CBFB-MYH11": cyto.str.contains(r"inv\(16\)"),
                      "PML-RARA": cyto.str.contains(r"t\(15;17\)"), "MLLT3-KMT2A": cyto.str.contains(r"t\(9;11\)")},
                     index=patients).astype(int)
    return M, F, cebpa_n, cyto


def eln_approx(M, F, cebpa_n, cyto) -> pd.Series:
    """Approximate ELN2017 from karyotype + mutations (no FLT3-ITD allelic ratio; del(7q) counted adverse)."""
    adv_cyto = cyto.str.contains(r"Complex|del\(5q\)|del\(7q\)|inv\(3\)|t\(6;9\)")
    fav_cyto = (F["RUNX1-RUNX1T1"] + F["CBFB-MYH11"]) > 0
    g = lambda x: M[x] if x in M else 0
    fav_mut = ((g("NPM1") > 0) & (g("FLT3_ITD") == 0)) | (cebpa_n >= 2)
    adv_mut = (g("TP53") > 0) | (((g("RUNX1") > 0) | (g("ASXL1") > 0)) & ~fav_cyto & ~fav_mut)
    eln = pd.Series("Intermediate", index=M.index)
    eln[fav_cyto | (fav_mut & ~adv_cyto)] = "Favorable"
    eln[(adv_cyto & ~fav_cyto) | (g("TP53") > 0) | (adv_mut & ~fav_cyto)] = "Adverse"
    return eln


def build_features(data, feats_bat: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    E, muts, clin = load_tcga()
    pts = E.index.intersection(clin.index)
    E = E.loc[pts]
    R = expression_by_symbol(data.ref_X)
    ref_bm = data.ref_meta[COLS["tissue"]].eq("Bone Marrow Aspirate").to_numpy()
    expr = project(R, ref_bm, E)
    lsc_genes = [g for g in LSC17_COEFS if g in E.columns]
    print(f"LSC17 genes found in TCGA: {len(lsc_genes)}/17")
    expr["rna.PEAR1"] = E["PEAR1"]
    expr["rna.LSC17"] = E[lsc_genes].to_numpy() @ np.array([LSC17_COEFS[g] for g in lsc_genes])
    # moment-match every expression feature to the Beat AML training patients
    tr = data.patients[data.train]
    for c in expr.columns:
        ref = feats_bat.loc[tr, c]
        expr[c] = (expr[c] - expr[c].mean()) / expr[c].std() * ref.std() + ref.mean()
    M, F, cebpa_n, cyto = genomic_features(muts, clin, pts)
    out = expr.copy()
    for c in [c for c in feats_bat.columns if c.startswith("mut.")]:
        out[c] = M[c[4:]] if c[4:] in M else 0
    for c in [c for c in feats_bat.columns if c.startswith("fus.")]:
        out[c] = F[c[4:]] if c[4:] in F else 0
    out["clin.age"] = pd.to_numeric(clin.loc[pts, "AGE"], errors="coerce")
    out["clin.is_denovo"], out["clin.is_bm"] = 1, 1
    assert out[[c for c in feats_bat.columns if c != "clin.age"]].notna().all().all(), "NaN in mapped TCGA features"
    alts = pd.concat([M, F], axis=1).add_prefix("alt.")
    out = out[feats_bat.columns].join(alts.loc[:, ~alts.columns.isin(out.columns)])
    os_m = pd.to_numeric(clin.loc[pts, "OS_MONTHS"], errors="coerce")
    os_s = clin.loc[pts, "OS_STATUS"].str.startswith("1").astype(float)
    meta = pd.DataFrame({"os_months": os_m, "dead": os_s, "age": out["clin.age"],
                         "eln": eln_approx(M, F, cebpa_n, cyto), "apl": F["PML-RARA"].astype(bool),
                         "fab": clin.loc[pts, "FAB"], "cyto": cyto})
    return out, meta


def main() -> None:
    from lifelines import CoxPHFitter
    pd.set_option("display.width", 200)
    data = load_recommender_data()
    drugs = select_drugs(data.auc, data.train)
    feats_bat = data.features()
    X, meta = build_features(data, feats_bat)
    print(f"TCGA patients with RNA-seq + clinical: {len(X)}; with OS: {meta.os_months.notna().sum()}, deaths {int(meta.dead.sum())}")
    print("approx ELN:", meta.eln.value_counts().to_dict(), "| APL:", int(meta.apl.sum()))

    # sanity: does the mapped Monocyte-like score track FAB M4/M5 as it should?
    mono = meta.fab.isin(["M4", "M5"])
    print("Monocyte-like score, FAB M4/M5 vs other: %.2f vs %.2f" % (X.loc[mono, "ct.Monocyte-like"].mean(), X.loc[~mono, "ct.Monocyte-like"].mean()))

    # frozen model: M1 all features, fit on Waves1+2 (identical to the holdout fit)
    tr = data.patients[data.train]
    m1 = PerDrugRidge().fit(data.auc.loc[tr, drugs], feats_bat.loc[tr])
    pred = m1.predict(X[feats_bat.columns], drugs)
    meta["pred_z"] = pred[DRUG]
    print("predicted cytarabine z in TCGA: mean %.2f sd %.2f" % (meta.pred_z.mean(), meta.pred_z.std()))
    pred.to_csv(OUT / "tcga_predicted_z.csv")

    rows = []
    for cohort, d0 in [("non-APL (primary)", meta[~meta.apl]), ("all incl. APL", meta)]:
        d0 = d0.dropna(subset=["os_months", "dead", "age", "pred_z"])
        d0 = d0[d0.os_months > 0].copy()
        d0["pred_z_sd"] = (d0.pred_z - d0.pred_z.mean()) / d0.pred_z.std()
        d0["eln_int"] = d0.eln.eq("Intermediate").astype(int); d0["eln_adv"] = d0.eln.eq("Adverse").astype(int)
        for name, cols in {"pred_z alone": ["pred_z_sd"], "pred_z + ELN + age": ["pred_z_sd", "age", "eln_int", "eln_adv"],
                           "ELN + age (reference)": ["age", "eln_int", "eln_adv"]}.items():
            f = CoxPHFitter().fit(d0[cols + ["os_months", "dead"]], "os_months", "dead")
            r = {"cohort": cohort, "model": name, "n": len(d0), "deaths": int(d0.dead.sum()), "C-index": f.concordance_index_}
            if "pred_z_sd" in cols:
                ci = f.confidence_intervals_.loc["pred_z_sd"]
                r |= {"HR per SD": f.hazard_ratios_["pred_z_sd"], "lo": np.exp(ci.iloc[0]), "hi": np.exp(ci.iloc[1]),
                      "p": f.summary.loc["pred_z_sd", "p"]}
            rows.append(r)
    res = pd.DataFrame(rows)
    print("\nCox overall survival in TCGA-LAML (HR per SD of predicted cytarabine z; >1 = predicted resistance -> shorter OS)")
    print(res.round(3).to_string(index=False))
    print("\nBeat AML reference: HR 1.33 [1.09, 1.62] adjusted for ELN + age")
    X.join(meta).to_csv(OUT / "tcga_features.csv")
    res.to_csv(OUT / "tcga_cox.csv", index=False)


if __name__ == "__main__":
    main()
