"""External test: is PEAR1 prognostic beyond its WGCNA module (module 3) in TCGA-LAML?

    python scripts/get_tcga_laml.py      # once
    python -m analysis.tcga_pear1

Scores use FROZEN Beat AML weights: genes are z-scored within each cohort, then
projected on the PC1 loadings learned in Beat AML Waves1+2 (global scaling). So
"HSC-like" and "M3 without PEAR1" mean the same gene combination in both cohorts.

Analyses (pre-specified 2026-11-20):
  primary   pooled Beat AML + TCGA, age < 45, Cox stratified by cohort:
            PEAR1 + age + cytogenetic risk + HSC-like + M3_noPEAR1
  secondary TCGA alone (all ages, < 45, >= 45) with the same ladder; PEAR1 x age-group interaction
"""
from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter

from analysis.common import RESULTS, ROOT
from analysis.part2_gaps import rna560, survival_frame

warnings.filterwarnings("ignore")
TC = ROOT / "data/raw/tcga_laml"
LADDER = [("PEAR1 alone", []), ("+ age", ["age"]), ("+ age + risk", ["age", "adv", "fav"]),
          ("+ age + risk + HSC-like", ["age", "adv", "fav", "HSC"]),
          ("+ age + risk + HSC-like + M3 (w/o PEAR1)", ["age", "adv", "fav", "HSC", "M3noP"])]


def zcols(X: pd.DataFrame, ref: pd.DataFrame | None = None) -> pd.DataFrame:
    ref = X if ref is None else ref
    sd = ref.std(ddof=1).replace(0, 1)
    return (X - ref.mean()) / sd


def frozen_scores(E_bea, fit, E_tcga, gene_sets):
    """PC1 loadings from Beat AML fit rows, applied to both cohorts (each z-scored on itself)."""
    out_b, out_t = {}, {}
    for name, genes in gene_sets.items():
        g = [x for x in genes if x in E_bea.columns and x in E_tcga.columns]
        Zb_fit = zcols(E_bea.loc[fit, g])
        _, _, vt = np.linalg.svd(Zb_fit.to_numpy(float), full_matrices=False)
        w = vt[0] * np.sign(np.corrcoef(Zb_fit.to_numpy() @ vt[0], Zb_fit.mean(1))[0, 1])
        out_b[name] = zcols(E_bea[g], E_bea.loc[fit, g]).to_numpy(float) @ w
        out_t[name] = zcols(E_tcga[g]).to_numpy(float) @ w
        print(f"  {name}: {len(g)}/{len(genes)} genes matched")
    return pd.DataFrame(out_b, index=E_bea.index), pd.DataFrame(out_t, index=E_tcga.index)


def load_tcga():
    man = pd.DataFrame(json.loads((TC / "manifest.json").read_text()))
    cols = {}
    for fid, case in zip(man.file_id, man.case):
        d = pd.read_csv(TC / "star" / f"{fid}.tsv", sep="\t", comment="#", usecols=["gene_name", "gene_type", "tpm_unstranded"])
        d = d[d.gene_type.notna()]
        cols[case] = d.groupby("gene_name").tpm_unstranded.sum()
    E = np.log2(pd.DataFrame(cols).T + 1)
    cl = pd.read_csv(TC / "clinical_patient_laml.txt", sep="\t").iloc[2:].set_index("bcr_patient_barcode")
    num = lambda s: pd.to_numeric(s, errors="coerce")
    T = num(cl.death_days_to).fillna(num(cl.last_contact_days_to)) / 365.25
    S = pd.DataFrame({"T": T, "E": cl.vital_status.eq("Dead").astype(int), "age": num(cl.age_at_diagnosis),
                      "risk": cl.cyto_risk_group})
    S["adv"] = S.risk.eq("Poor").astype(int)
    S["fav"] = S.risk.eq("Favorable").astype(int)
    S = S[S["T"].gt(0) & S.age.notna() & S.risk.isin(["Poor", "Favorable", "Intermediate/Normal"])]
    common = E.index.intersection(S.index)
    return E.loc[common], S.loc[common]


def cox(df, cols, strata=None, label=None):
    use = ["T", "E", "PEAR1"] + cols + ([strata] if strata else [])
    f = CoxPHFitter().fit(df[use], "T", "E", strata=[strata] if strata else None)
    s = f.summary.loc["PEAR1"]
    return {**(label or {}), "n": len(df), "events": int(df.E.sum()), "HR_PEAR1_perSD": s["exp(coef)"],
            "lo": s["exp(coef) lower 95%"], "hi": s["exp(coef) upper 95%"], "p": s["p"]}


def main():
    D = rna560()
    wg = pd.read_csv(ROOT / "data/interim/wgcna_cur_map.csv")
    m3 = [g for g in wg[wg.cur_labels == "M3"].display_label if g != "PEAR1"]
    E_t, S_t = load_tcga()
    print(f"TCGA: {len(S_t)} patients, {S_t.E.sum()} deaths; <45: {(S_t.age < 45).sum()} pts, "
          f"{S_t[S_t.age < 45].E.sum()} deaths")
    fit = D["clin"].cohort.eq("Waves1+2")
    sc_b, sc_t = frozen_scores(D["E"], fit, E_t, {"HSC": D["sigs"]["HSC-like"], "M3noP": m3})

    # Beat AML frame; PEAR1 standardised on Waves1+2 rows, TCGA on itself
    def build(S, E, sc, ref_rows=None):
        ref = E.loc[ref_rows] if ref_rows is not None else E
        F = S.join(((E["PEAR1"] - ref["PEAR1"].mean()) / ref["PEAR1"].std()).rename("PEAR1")).join(sc)
        for c in ["HSC", "M3noP"]:
            r = F.loc[ref_rows.index[ref_rows].intersection(F.index), c] if ref_rows is not None else F[c]
            F[c] = (F[c] - r.mean()) / r.std()
        F["age"] = (F.age - 60) / 15          # same scale in both cohorts (per 15 years)
        return F.dropna(subset=["T", "E", "PEAR1", "HSC", "M3noP", "age"])

    F_t = build(S_t, E_t, sc_t)
    print("TCGA correlations:\n", F_t[["PEAR1", "HSC", "M3noP"]].corr().round(2).to_string())

    rows = []
    for denovo in (True, False):
        S_b = survival_frame(D, denovo)
        S_b["adv"] = S_b.ELN.eq("Adverse").astype(int)
        S_b["fav"] = S_b.ELN.eq("Favorable").astype(int)
        S_b = S_b[S_b.ELN.notna()]
        # one specimen per patient
        S_b = S_b[~D["clin"].loc[S_b.index, "dbgap_subject_id"].duplicated()]
        F_b = build(S_b, D["E"], sc_b, fit)
        bea = "Beat AML de novo" if denovo else "Beat AML all AML"
        P = pd.concat([F_b.assign(cohort=0), F_t.assign(cohort=1)])
        for ages, sub in [("age < 45", P[P.age < (45 - 60) / 15]), ("all ages", P)]:
            for name, cols in LADDER:
                rows.append(cox(sub, cols, "cohort", {"analysis": f"pooled ({bea} + TCGA)", "ages": ages, "model": name}))
    for ages, sub in [("all ages", F_t), ("age < 45", F_t[F_t.age < -1]), ("age >= 45", F_t[F_t.age >= -1])]:
        for name, cols in LADDER:
            rows.append(cox(sub, cols, None, {"analysis": "TCGA alone", "ages": ages, "model": name}))
    # PEAR1 x young interaction in TCGA, full model
    G = F_t.assign(young=(F_t.age < -1).astype(int))
    G["PEAR1_x_young"] = G.PEAR1 * G.young
    f = CoxPHFitter().fit(G[["T", "E", "PEAR1", "PEAR1_x_young", "young", "age", "adv", "fav", "HSC", "M3noP"]], "T", "E")
    print("\nTCGA PEAR1 x (age<45) interaction, full model:\n",
          f.summary.loc[["PEAR1", "PEAR1_x_young"], ["exp(coef)", "exp(coef) lower 95%", "exp(coef) upper 95%", "p"]].round(3))

    t = pd.DataFrame(rows)
    t.to_csv(RESULTS / "tcga_pear1.csv", index=False)
    with pd.option_context("display.width", 200):
        print(t.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
