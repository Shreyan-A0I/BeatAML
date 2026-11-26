"""Does the PEAR1 drug-response pattern from Beat AML also show up in the FIMM cohort?

FIMM = Malani et al. 2022 (Helsinki): ex vivo drug screen (sDSS) + RNA-seq on AML samples.

    python -m src.validation.fimm_pear1 [--fimm PATH_TO_FIMM_FOLDER]

The SAME model is fit in both cohorts, one drug at a time:

    z(resistance) ~ PEAR1 + stemness (LSC17) + age + not-at-diagnosis   (+ medium in FIMM)

* resistance: Beat AML = AUC; FIMM = minus sDSS. So for both, higher = more resistant.
* PEAR1 and LSC17 are z-scored inside each cohort, so "effect" = change in resistance
  (in SDs) per 1 SD of PEAR1. Positive = PEAR1-high tumors are more resistant.
* Stemness uses LSC17 (published weights) because it needs only expression; the van Galen
  HSC-like signature file is not available for FIMM.

Pre-stated predictions from the Beat AML results (written before looking at FIMM):
    venetoclax resistant (+), trametinib sensitive (-), panobinostat sensitive (-).

Outputs: results/tables/fimm_pear1_drugs.csv, fimm_pear1_summary.json,
         results/figures/fimm_pear1_replication.png
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

TABLES, FIGS = ROOT / "results" / "tables", ROOT / "results" / "figures"
PEAR1_ID = "ENSG00000187800"
HEADLINE = {"Venetoclax": +1, "Trametinib": -1, "Panobinostat": -1}  # predicted sign of effect
YOUNG_AGE = 45
MIN_N = 60

# LSC17 (Ng et al., Nature 2016); aliases are older gene symbols used in some annotations
LSC17 = {"DNMT3B": 0.0874, "ZBTB46": -0.0347, "NYNRIN": 0.00865, "ARHGAP22": -0.0138, "LAPTM4B": 0.00582,
         "MMRN1": 0.0258, "DPYSL3": 0.0284, "FAM30A": 0.0196, "CDK6": -0.0704, "CPXM1": -0.0258,
         "SOCS2": 0.0271, "SMIM24": -0.0226, "EMP1": 0.0146, "BEX3": 0.0465, "CD34": 0.0338,
         "AKR1C3": -0.0402, "ADGRG1": 0.0501}
ALIASES = {"FAM30A": ["KIAA0125"], "BEX3": ["NGFRAP1"], "ADGRG1": ["GPR56"], "SMIM24": ["C19orf77"],
           "NYNRIN": ["KIAA1305"]}


# ----------------------------------------------------------------------------- helpers
def zscore(s: pd.Series) -> pd.Series:
    return (s - s.mean()) / s.std(ddof=1)


def drug_key(name: str) -> str:
    """'Trametinib (GSK1120212)' -> 'trametinib' so names match across cohorts."""
    return re.sub(r"[^a-z0-9]", "", re.sub(r"\(.*?\)", "", str(name).lower()))


def bh(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float)
    order = np.argsort(p)
    ranked = p[order] * len(p) / (np.arange(len(p)) + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty_like(q)
    out[order] = np.minimum(q, 1)
    return out


def ols_effect(y: np.ndarray, predictors: list[np.ndarray]) -> tuple[float, float, float]:
    """Coefficient, standard error and two-sided p of the FIRST predictor."""
    X = np.column_stack([np.ones(len(y))] + predictors)
    XtXi = np.linalg.pinv(X.T @ X)
    b = XtXi @ X.T @ y
    resid = y - X @ b
    dof = len(y) - X.shape[1]
    se = float(np.sqrt(resid @ resid / dof * XtXi[1, 1]))
    return float(b[1]), se, float(2 * stats.t.sf(abs(b[1] / se), dof))


def drug_effects(resp: pd.DataFrame, cov: pd.DataFrame, adjust: list[str]) -> pd.DataFrame:
    """One regression per drug. resp: samples x drugs (higher = more resistant)."""
    rows = []
    for d in resp.columns:
        ok = resp[d].notna() & cov[["pear1"] + adjust].notna().all(axis=1)   # skip rows with a missing predictor
        if ok.sum() < MIN_N:
            continue
        y = zscore(resp.loc[ok, d]).to_numpy()
        preds = [cov.loc[ok, "pear1"].to_numpy()] + [cov.loc[ok, c].to_numpy() for c in adjust]
        b, se, p = ols_effect(y, preds)
        rows.append({"drug": d, "n": int(ok.sum()), "beta": b, "se": se, "p": p})
    return pd.DataFrame(rows).set_index("drug")


# ----------------------------------------------------------------------------- Beat AML
def load_beat() -> dict:
    from src.load import COLS, build_modeling_frame
    f = build_modeling_frame(verbose=False)
    X, meta, auc = f["X"], f["meta"], f["y"]
    label_to_id = {c[1]: c[0] for c in X.columns}
    X = X.copy()
    X.columns = [c[0] for c in X.columns]
    idx = pd.Index(meta[COLS["patient_id"]].astype(str).to_numpy(), name="patient")
    X.index, auc = idx, auc.set_axis(idx, axis=0)
    cov = pd.DataFrame(index=idx)
    cov["age"] = meta[COLS["age"]].to_numpy()
    cov["nondx"] = (meta[COLS["timepoint"]].to_numpy() != "Initial Diagnosis").astype(float)
    return {"X": X, "auc": auc, "cov": cov, "label_to_id": label_to_id}


# ----------------------------------------------------------------------------- FIMM
def load_fimm(d: Path) -> dict:
    ann = pd.read_excel(d / "File_0_Common_sample_annotation_252S.xlsx")
    ann["Disease status"] = ann["Disease status"].str.strip()
    ann["pid"] = ann["Patient_ID"].str.replace("FPM_", "", regex=False).str.strip()
    clin = pd.read_excel(d / "File_1.1_Clinical_summary_186_Patients.xlsx")
    clin["pid"] = clin["Patient_ID"].str.strip()
    assay = pd.read_excel(d / "File_4_DSRT_assay_details_164S_DM.xlsx")
    sdss = pd.read_excel(d / "File_3.1_Drug_response_sDSS_164S_17Healthy.xlsx")
    lib = pd.read_excel(d / "File_2_Drug_library_515D.xlsx")
    rna = pd.read_csv(d / "File_7_RNA_seq_CPM_163S_4Healthy.csv", index_col=0)

    both = ann[ann["DSRT"].notna() & ann["RNA-seq"].notna()
               & ann["Sample_ID"].isin(sdss.columns) & ann["Sample_ID"].isin(rna.columns)].copy()
    both["not_dx"] = (both["Disease status"] != "Diagnosis").astype(float)
    # one sample per patient: the diagnosis sample if there is one, else the earliest
    both = both.sort_values(["pid", "not_dx", "Sample_ID"]).drop_duplicates("pid")
    both = both.merge(clin[["pid", "Age at diagnosis"]], on="pid", how="left")
    both = both.merge(assay[["Sample_ID", "Medium"]], on="Sample_ID", how="left")
    both = both.dropna(subset=["Age at diagnosis", "Medium"]).set_index("Sample_ID")

    sdss = sdss.drop_duplicates("Drug_name").set_index("Drug_name")
    resp = (-sdss[both.index]).T            # minus sDSS: higher = more resistant
    cov = pd.DataFrame({"age": both["Age at diagnosis"].astype(float), "nondx": both["not_dx"],
                        "cm": (both["Medium"] == "CM").astype(float)}, index=both.index)
    meta = sdss[["Drug_ID"]].reset_index().merge(
        lib.rename(columns={"FIMM batch ID": "Drug_ID"})[["Drug_ID", "Mechanism/Targets", "Drug_Class",
                                                          "High phase/Approval status"]], on="Drug_ID", how="left")
    # File_7 is already log2 CPM (min -1.7, max 16.7), so no further transform
    return {"resp": resp, "cov": cov, "rna": rna[both.index], "drug_info": meta.set_index("Drug_name"),
            "status": both["Disease status"]}


# ----------------------------------------------------------------------------- scores
def add_scores(cov: pd.DataFrame, expr: pd.DataFrame, lsc_ids: dict[str, float]) -> pd.DataFrame:
    """expr: samples x Ensembl id (log scale). Adds z-scored pear1 and lsc columns."""
    w = pd.Series(lsc_ids)
    cov = cov.copy()
    cov["pear1"] = zscore(expr[PEAR1_ID]).to_numpy()
    cov["lsc"] = zscore(expr[w.index] @ w).to_numpy()
    cov["age"] = zscore(cov["age"]).to_numpy()
    return cov


# ----------------------------------------------------------------------------- main
def main(fimm_dir: Path) -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)
    print("loading Beat AML ...")
    beat = load_beat()
    print("loading FIMM ...")
    fimm = load_fimm(fimm_dir)

    # LSC17 genes present in both cohorts, as Ensembl ids (the same gene combination in both)
    ids = {}
    for sym, w in LSC17.items():
        for name in [sym] + ALIASES.get(sym, []):
            gid = beat["label_to_id"].get(name)
            if gid and gid in beat["X"].columns and gid in fimm["rna"].index:
                ids[gid] = w
                break
    print(f"LSC17 genes usable in both cohorts: {len(ids)}/17")
    assert PEAR1_ID in beat["X"].columns and PEAR1_ID in fimm["rna"].index, "PEAR1 missing"

    bcov = add_scores(beat["cov"], beat["X"], ids)
    fcov = add_scores(fimm["cov"], fimm["rna"].T, ids)

    n_f = len(fcov)
    young_f = int((fimm["cov"]["age"] < YOUNG_AGE).sum())
    print(f"Beat AML: {len(bcov)} patients | FIMM: {n_f} patients "
          f"(diagnosis {int((fimm['status'] == 'Diagnosis').sum())}, other {int((fimm['status'] != 'Diagnosis').sum())}; "
          f"age<{YOUNG_AGE}: {young_f})")
    r_b, r_f = bcov[["pear1", "lsc"]].corr().iloc[0, 1], fcov[["pear1", "lsc"]].corr().iloc[0, 1]
    print(f"sanity: corr(PEAR1, LSC17) Beat AML {r_b:.2f} | FIMM {r_f:.2f}")

    # shared drugs
    bmap = {}
    for d in beat["auc"].columns:
        bmap.setdefault(drug_key(d), d)
    fmap = {}
    for d in fimm["resp"].columns:
        fmap.setdefault(drug_key(d), d)
    keys = sorted(set(bmap) & set(fmap))
    print(f"drugs: Beat AML {len(bmap)}, FIMM {len(fmap)}, shared by name {len(keys)}")

    results = {}
    for label, adj_b, adj_f in [("primary", ["lsc", "age", "nondx"], ["lsc", "age", "nondx", "cm"]),
                                ("pear1_only", [], [])]:
        eb = drug_effects(beat["auc"][[bmap[k] for k in keys]], bcov, adj_b)
        ef = drug_effects(fimm["resp"][[fmap[k] for k in keys]], fcov, adj_f)
        t = pd.DataFrame({"drug_key": keys, "drug": [bmap[k] for k in keys], "drug_fimm": [fmap[k] for k in keys]})
        t = t.join(eb.add_suffix("_beat"), on="drug").join(ef.add_suffix("_fimm"), on="drug_fimm").dropna()
        t["q_beat"], t["q_fimm"] = bh(t["p_beat"]), bh(t["p_fimm"])
        results[label] = t

    t = results["primary"].copy()
    info = fimm["drug_info"].reindex(t["drug_fimm"])
    t["mechanism"] = info["Mechanism/Targets"].to_numpy()
    t["drug_class"] = info["Drug_Class"].to_numpy()
    t["approval"] = info["High phase/Approval status"].to_numpy()
    t.to_csv(TABLES / "fimm_pear1_drugs.csv", index=False)

    def agreement(tab: pd.DataFrame) -> dict:
        rho, p = stats.spearmanr(tab["beta_beat"], tab["beta_fimm"])
        sig = tab[tab["q_beat"] < 0.10]
        same = int((np.sign(sig["beta_beat"]) == np.sign(sig["beta_fimm"])).sum())
        binom = stats.binomtest(same, len(sig), 0.5, alternative="greater").pvalue if len(sig) else float("nan")
        return {"n_shared_drugs": len(tab), "spearman_effects": float(rho), "spearman_p": float(p),
                "n_beat_significant_q10": int(len(sig)), "same_sign_in_fimm": same, "sign_test_p": float(binom)}

    headline = []
    for name, pred in HEADLINE.items():
        row = t[t["drug_key"] == drug_key(name)]
        if row.empty:
            headline.append({"drug": name, "predicted_sign": pred, "note": "not tested in both cohorts"})
            continue
        r = row.iloc[0]
        same = bool(np.sign(r.beta_fimm) == pred)
        p_one = r.p_fimm / 2 if same else 1 - r.p_fimm / 2
        headline.append({"drug": name, "predicted_sign": pred, "n_beat": int(r.n_beat), "beta_beat": float(r.beta_beat),
                         "p_beat": float(r.p_beat), "n_fimm": int(r.n_fimm), "beta_fimm": float(r.beta_fimm),
                         "se_fimm": float(r.se_fimm), "p_fimm_two_sided": float(r.p_fimm),
                         "p_fimm_one_sided_predicted_direction": float(p_one),
                         "same_direction_as_predicted": same, "replicated_p_lt_0.05": bool(same and p_one < 0.05)})

    summary = {"cohorts": {"beat_aml_patients": len(bcov), "fimm_patients": n_f,
                           "fimm_diagnosis": int((fimm["status"] == "Diagnosis").sum()),
                           "fimm_relapse_or_refractory": int((fimm["status"] != "Diagnosis").sum()),
                           "fimm_age_under_45": young_f, "lsc17_genes_used": len(ids),
                           "corr_pear1_lsc17_beat": float(r_b), "corr_pear1_lsc17_fimm": float(r_f)},
               "convention": "effect = change in z-scored resistance per SD of PEAR1; positive = PEAR1-high more resistant",
               "headline_drugs": headline,
               "agreement_primary": agreement(results["primary"]),
               "agreement_pear1_only": agreement(results["pear1_only"])}
    (TABLES / "fimm_pear1_summary.json").write_text(json.dumps(summary, indent=2))

    # ---- print
    print("\nHeadline drugs (predicted sign: + resistant, - sensitive):")
    for h in headline:
        if "beta_fimm" in h:
            print(f"  {h['drug']:<13} pred {h['predicted_sign']:+d} | Beat AML {h['beta_beat']:+.2f} (n={h['n_beat']}) "
                  f"| FIMM {h['beta_fimm']:+.2f} +/- {h['se_fimm']:.2f} (n={h['n_fimm']}), one-sided p={h['p_fimm_one_sided_predicted_direction']:.3f}"
                  f" (two-sided p={h['p_fimm_two_sided']:.3f})"
                  f" -> {'REPLICATES' if h['replicated_p_lt_0.05'] and h['p_fimm_two_sided'] < 0.05 else ('borderline: one-sided only' if h['replicated_p_lt_0.05'] else ('right direction, not significant' if h['same_direction_as_predicted'] else 'wrong direction'))}")
        else:
            print(f"  {h['drug']}: {h['note']}")
    for k in ("agreement_primary", "agreement_pear1_only"):
        a = summary[k]
        print(f"\n{k}: {a['n_shared_drugs']} shared drugs | Spearman of effects {a['spearman_effects']:+.2f} (p={a['spearman_p']:.3g}) | "
              f"Beat AML q<0.10 drugs: {a['n_beat_significant_q10']}, same sign in FIMM: {a['same_sign_in_fimm']} (sign test p={a['sign_test_p']:.3g})")

    # ---- figure
    import matplotlib
    if not os.environ.get("DISPLAY") and sys.platform.startswith("linux"):
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.5, 6))
    sig = t["q_beat"] < 0.10
    ax.scatter(t.loc[~sig, "beta_beat"], t.loc[~sig, "beta_fimm"], s=16, color="#BBBBBB", label="Beat AML q >= 0.10")
    ax.scatter(t.loc[sig, "beta_beat"], t.loc[sig, "beta_fimm"], s=22, color="#4C78A8", label="Beat AML q < 0.10")
    ax.axhline(0, color="k", lw=0.6)
    ax.axvline(0, color="k", lw=0.6)
    for name in HEADLINE:
        r = t[t["drug_key"] == drug_key(name)]
        if len(r):
            ax.scatter(r.beta_beat, r.beta_fimm, s=60, facecolor="none", edgecolor="#D62728", lw=1.6)
            ax.annotate(name, (r.beta_beat.iloc[0], r.beta_fimm.iloc[0]), xytext=(5, 5), textcoords="offset points", fontsize=8)
    a = summary["agreement_primary"]
    ax.set(xlabel="PEAR1 effect in Beat AML (SD resistance per SD PEAR1)", ylabel="PEAR1 effect in FIMM",
           title=f"{a['n_shared_drugs']} shared drugs, Spearman {a['spearman_effects']:+.2f}")
    ax.legend(fontsize=8, loc="lower right")
    plt.tight_layout()
    plt.savefig(FIGS / "fimm_pear1_replication.png", dpi=150)
    plt.close()
    print(f"\nwrote {TABLES / 'fimm_pear1_drugs.csv'}, fimm_pear1_summary.json and the figure")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fimm", default=os.environ.get("FIMM_DIR", str(ROOT.parent / "Functional_Precision_Medicine_Tumor_Board_AML")))
    main(Path(ap.parse_args().fimm))
