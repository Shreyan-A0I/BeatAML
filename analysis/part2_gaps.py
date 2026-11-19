"""Part 2: what did Bottomly et al. (2022) leave on the table?

    python -m analysis.part2_gaps            # all gaps
    python -m analysis.part2_gaps pear1 kme  # selected gaps

Each gap writes results/tables/part2_<gap>*.csv and prints a short summary.
"""

from __future__ import annotations

import pickle
import re
import sys
import warnings

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from scipy import stats
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import adjusted_rand_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from analysis.common import RESULTS, ROOT, SUPP, load_frame, tissue_pc1
from src.load import CELL_TYPES, load_auc_long, load_clinical, load_expression, load_pc1_baseline

warnings.filterwarnings("ignore")
RAW = ROOT / "data" / "raw"
CACHE560 = ROOT / "data" / "interim" / "analysis_rna560.pkl"
RESULTS.mkdir(parents=True, exist_ok=True)


def bh(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    q = p[o] * len(p) / (np.arange(len(p)) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q)
    out[o] = np.minimum(q, 1)
    return out


def by(p):
    return np.minimum(bh(p) * np.sum(1.0 / np.arange(1, len(p) + 1)), 1)


# ---------------------------------------------------------------- 560-specimen RNA cohort
def rna560() -> dict:
    """All 560 manuscript RNA-seq specimens: clinical, published states, eigengenes, mutations,
    and expression of the genes the gap analyses need."""
    if CACHE560.exists():
        return pickle.loads(CACHE560.read_bytes())
    clin = load_clinical()
    clin = clin[clin.manuscript_rnaseq.eq("yes")].set_index("dbgap_rnaseq_sample")
    expr = load_expression()   # genes x specimens, MultiIndex (stable_id, display_label, ...)
    E = expr[clin.index].T
    E.columns = expr.index.get_level_values("display_label")
    E = E.loc[:, ~E.columns.duplicated()]

    wg = pd.read_csv(ROOT / "data/interim/wgcna_cur_map.csv")
    bm = clin.specimenType.eq("Bone Marrow Aspirate").to_numpy()
    fit = clin.cohort.eq("Waves1+2").to_numpy()
    mes = {}
    for mod, genes in wg[wg.cur_labels != "M0"].groupby("cur_labels").display_label:
        g = [x for x in genes if x in E.columns]
        # Module eigengenes use ONE global scaling (manuscript predict.modules()), unlike the
        # per-tissue cell-type scores.
        mes[mod] = tissue_pc1(E[g].to_numpy(float), np.ones(len(E), bool), fit)[0]
        if mod == "M3":   # also an eigengene WITHOUT PEAR1, for the adjustment analysis
            g2 = [x for x in g if x != "PEAR1"]
            mes["M3_noPEAR1"] = tissue_pc1(E[g2].to_numpy(float), np.ones(len(E), bool), fit)[0]
    mes = pd.DataFrame(mes, index=clin.index)

    sys.path.insert(0, str(ROOT / "scripts"))
    from replication_vangalen_scores import load_signatures
    sigs = load_signatures()
    keep_genes = set(wg.display_label) | {g for s in sigs.values() for g in s} | {"PEAR1"}
    E_small = E[[g for g in E.columns if g in keep_genes]]

    raw = pd.read_csv(RAW / "beataml_wes_wv1to4_mutations_dbgap.txt", sep="\t", low_memory=False)
    wes = pd.crosstab(raw.dbgap_sample_id, raw.symbol).clip(upper=1)
    muts = wes.reindex(clin.dbgap_dnaseq_sample).set_axis(clin.index).astype(float)
    for gname, col in {"NPM1": "NPM1", "FLT3_ITD": "FLT3-ITD", "RUNX1": "RUNX1", "ASXL1": "ASXL1", "TP53": "TP53"}.items():
        c = clin[col].map({"positive": 1.0, "negative": 0.0})
        muts[gname] = c.where(c.notna(), muts[gname] if gname in muts else np.nan)

    states = load_pc1_baseline().reindex(clin.index)
    D = {"clin": clin, "E": E_small, "mes": mes, "states": states, "muts": muts, "sigs": sigs}
    CACHE560.write_bytes(pickle.dumps(D))
    return D


def survival_frame(D, denovo_only=True) -> pd.DataFrame:
    c = D["clin"]
    os_ = pd.to_numeric(c.overallSurvival, errors="coerce")
    dead = c.vitalStatus.map({"Dead": 1, "Alive": 0})
    keep = os_.gt(0) & dead.notna()
    if denovo_only:
        keep &= c.isDenovo.astype(str).eq("TRUE")
    df = pd.DataFrame({"T": os_ / 365.25, "E": dead, "age": c.ageAtDiagnosis,
                       "denovo": c.isDenovo.astype(str), "transformed": c.isTransformed.astype(str),
                       "therapy": c.specificDxAtAcquisition.astype(str).str.contains("Therapy", na=False),
                       "ELN": c.ELN2017})[keep]
    return df


# ---------------------------------------------------------------- G1 ex vivo vs clinical
TREAT_SYNONYMS = {
    "Venetoclax": ["venetoclax", "abt-199"], "Sorafenib": ["sorafenib"], "Midostaurin": ["midostaurin", "pkc412"],
    "Gilteritinib": ["gilteritinib"], "Quizartinib (AC220)": ["quizartinib", "ac220"], "Crenolanib": ["crenolanib"],
    "Cytarabine": ["cytarabine", "7+3", "hidac", "midac", "mec ", "ldac", "flag", "clag", "ara-c"],
    "Azacytidine": ["azacitidine", "azacytidine"], "Decitabine": ["decitabine"],
    "Dasatinib": ["dasatinib"], "Ibrutinib (PCI-32765)": ["ibrutinib"], "Ruxolitinib (INCB018424)": ["ruxolitinib"],
    "Selinexor": ["selinexor"], "Trametinib (GSK1120212)": ["trametinib"], "Panobinostat": ["panobinostat"],
    "Idelalisib": ["idelalisib"], "Elesclomol": ["elesclomol"], "Lenalidomide": ["lenalidomide"],
}


def gap_exvivo_clinical():
    clin = load_clinical()
    auc = load_auc_long()
    am = auc.pivot_table(index="base_samp", columns="inhibitor", values="auc")
    # one diagnosis specimen per patient with a drug screen
    c = clin[clin.base_samp.isin(am.index)].copy()
    c["dx"] = c.diseaseStageAtSpecimenCollection.eq("Initial Diagnosis")
    c = c.sort_values(["dbgap_subject_id", "dx"], ascending=[True, False]).groupby("dbgap_subject_id").head(1)
    reg = c.cumulativeTreatmentRegimens.fillna("").str.lower() + "|" + c.currentRegimen.fillna("").str.lower()

    rows = []
    for drug, syns in TREAT_SYNONYMS.items():
        cols = [x for x in am.columns if x.lower().startswith(drug.lower().split(" ")[0])]
        if not cols:
            rows.append({"drug": drug, "panel_name": None, "treated": int(reg.apply(lambda r: any(s in r for s in syns)).sum())})
            continue
        col = cols[0]
        treated = reg.apply(lambda r: any(s in r for s in syns))
        has_auc = c.base_samp.map(am[col]).notna()
        dx = c.dx
        rows.append({"drug": drug, "panel_name": col, "treated": int(treated.sum()),
                     "auc": int(has_auc.sum()), "treated_and_auc": int((treated & has_auc).sum()),
                     "treated_and_auc_dx_specimen": int((treated & has_auc & dx).sum())})
    counts = pd.DataFrame(rows)
    counts.to_csv(RESULTS / "part2_exvivo_clinical_counts.csv", index=False)
    print(counts.to_string(index=False))

    # (a) Induction response after standard chemotherapy vs ex vivo AUC of each drug.
    resp = c.responseToInductionTx.map({"Complete Response": 1, "Complete Response i": 1, "Refractory": 0})
    ok = c.typeInductionTx.eq("Standard Chemotherapy") & resp.notna() & c.dx
    cc = c[ok].assign(y=resp[ok])
    res = []
    for col in am.columns:
        a = cc.base_samp.map(am[col])
        m = a.notna()
        if m.sum() < 80 or cc.y[m].nunique() < 2:
            continue
        auroc = roc_auc_score(cc.y[m], -a[m])        # lower AUC (sensitive) -> CR
        X = pd.DataFrame({"auc": (a[m] - a[m].mean()) / a[m].std(), "age": (cc.ageAtDiagnosis[m] - 60) / 15}).fillna(0)
        lr = LogisticRegression(C=1e6).fit(X, cc.y[m])
        res.append({"inhibitor": col, "n": int(m.sum()), "n_CR": int(cc.y[m].sum()), "AUROC_CR": auroc,
                    "OR_per_SD_AUC_age_adj": float(np.exp(lr.coef_[0][0]))})
    ind = pd.DataFrame(res).sort_values("AUROC_CR", ascending=False)
    # Permutation-free reference: where does cytarabine rank among all drugs?
    ind["rank"] = np.arange(1, len(ind) + 1)
    ind.to_csv(RESULTS / "part2_exvivo_induction.csv", index=False)
    print(f"\nInduction (standard chemo, diagnosis specimen): {ok.sum()} patients, CR rate {cc.y.mean():.2f}")
    print(ind.head(8).round(3).to_string(index=False))
    print(ind[ind.inhibitor.str.contains("Cytarabine|Idarub|Daunorub|Doxorub")].round(3).to_string(index=False))
    print(f"median AUROC across {len(ind)} drugs: {ind.AUROC_CR.median():.3f}")

    # Bootstrap CI for cytarabine AUROC, and its percentile among drugs
    cy = [x for x in am.columns if x.startswith("Cytarabine")][0]
    a = cc.base_samp.map(am[cy]); m = a.notna()
    rng = np.random.default_rng(0)
    yv, av = cc.y[m].to_numpy(), -a[m].to_numpy()
    bs = [roc_auc_score(yv[i], av[i]) for i in (rng.integers(0, len(yv), len(yv)) for _ in range(2000)) if yv[i].min() != yv[i].max()]
    print(f"cytarabine AUROC CI95 {np.percentile(bs, 2.5):.3f}-{np.percentile(bs, 97.5):.3f}")

    # (b) Treated with drug X: OS by ex vivo AUC of X, vs untreated (interaction).
    os_ = pd.to_numeric(c.overallSurvival, errors="coerce") / 365.25
    dead = c.vitalStatus.map({"Dead": 1, "Alive": 0})
    out = []
    for drug in ["Venetoclax", "Sorafenib", "Midostaurin", "Azacytidine", "Decitabine", "Gilteritinib"]:
        syns = TREAT_SYNONYMS[drug]
        cols = [x for x in am.columns if x.lower().startswith(drug.lower().split(" ")[0])]
        if not cols:
            continue
        a = c.base_samp.map(am[cols[0]])
        tr = reg.apply(lambda r: any(s in r for s in syns)).astype(int)
        df = pd.DataFrame({"T": os_, "E": dead, "z": (a - a.mean()) / a.std(), "tr": tr,
                           "age": (c.ageAtDiagnosis - 60) / 15}).dropna()
        df = df[df["T"] > 0]
        if df.tr.sum() < 15:
            continue
        df["z_tr"] = df.z * df.tr
        f = CoxPHFitter().fit(df[["T", "E", "z", "tr", "z_tr", "age"]], "T", "E")
        ft = CoxPHFitter().fit(df[df.tr == 1][["T", "E", "z", "age"]], "T", "E")
        out.append({"drug": drug, "n_treated": int(df.tr.sum()), "events_treated": int(df.E[df.tr == 1].sum()),
                    "HR_perSD_AUC_treated": ft.hazard_ratios_["z"], "p_treated": ft.summary.loc["z", "p"],
                    "HR_interaction": f.hazard_ratios_["z_tr"], "p_interaction": f.summary.loc["z_tr", "p"]})
    tb = pd.DataFrame(out)
    tb.to_csv(RESULTS / "part2_exvivo_treated_os.csv", index=False)
    print("\nOS among treated, per SD of the drug's own ex vivo AUC (age-adjusted):")
    print(tb.round(3).to_string(index=False))


# ---------------------------------------------------------------- G2 PEAR1 adjustment
def gap_pear1():
    D = rna560()
    out = []
    for cohort, denovo in [("de novo", True), ("all AML", False)]:
        S = survival_frame(D, denovo)
        S = S.join(D["E"]["PEAR1"]).join(D["states"]["HSC-like"]).join(D["mes"][["M3", "M3_noPEAR1"]])
        S["ELN_adv"] = S.ELN.eq("Adverse").astype(int)
        S["ELN_fav"] = S.ELN.eq("Favorable").astype(int)
        for col in ["PEAR1", "HSC-like", "M3", "M3_noPEAR1", "age"]:
            S[col] = (S[col] - S[col].mean()) / S[col].std()
        S = S.dropna(subset=["T", "E", "PEAR1", "HSC-like", "M3", "age"])
        age = D["clin"].loc[S.index, "ageAtDiagnosis"]
        # Age bands follow the paper's cut points (young < 45, middle 45-60).
        for grp, sub in [("all ages", S), ("age < 45", S[age < 45]), ("age 45-60", S[(age >= 45) & (age < 60)]),
                         ("age >= 60", S[age >= 60])]:
            for name, cols in [("PEAR1 alone", []), ("+ age", ["age"]), ("+ age + ELN", ["age", "ELN_adv", "ELN_fav"]),
                               ("+ age + ELN + HSC-like", ["age", "ELN_adv", "ELN_fav", "HSC-like"]),
                               ("+ age + ELN + HSC-like + M3 (w/o PEAR1)",
                                ["age", "ELN_adv", "ELN_fav", "HSC-like", "M3_noPEAR1"])]:
                cols = ["PEAR1"] + cols
                f = CoxPHFitter(penalizer=0.0).fit(sub[["T", "E"] + cols], "T", "E")
                s = f.summary.loc["PEAR1"]
                out.append({"cohort": cohort, "ages": grp, "model": name, "n": len(sub), "events": int(sub.E.sum()),
                            "HR_PEAR1_perSD": s["exp(coef)"], "lo": s["exp(coef) lower 95%"], "hi": s["exp(coef) upper 95%"],
                            "p": s["p"], "c_index": f.concordance_index_})
        if cohort == "de novo":
            r = S[["PEAR1", "HSC-like", "M3", "M3_noPEAR1"]].corr().round(3)
            print("correlations (de novo):\n", r.to_string())
    t = pd.DataFrame(out)
    t.to_csv(RESULTS / "part2_pear1_adjustment.csv", index=False)
    print(t.round(3).to_string(index=False))


# ---------------------------------------------------------------- G3 kME bootstrap
def gap_kme(n_boot=1000):
    D = rna560()
    s7 = pd.read_excel(SUPP / "S7_module_membership.xlsx", header=1)
    m3 = [g for g in s7[s7.module_labels == "Mod3"].symbol if g in D["E"].columns]
    clin = D["clin"]
    bm = clin.specimenType.eq("Bone Marrow Aspirate").to_numpy()
    X = D["E"][m3].to_numpy(float)
    Z = np.empty_like(X)
    for g in (bm, ~bm):                      # tissue-wise standardisation, as for the eigengenes
        Z[g] = (X[g] - X[g].mean(0)) / np.where(X[g].std(0) == 0, 1, X[g].std(0))
    pi = m3.index("PEAR1")

    def kme(Zb):
        Zc = (Zb - Zb.mean(0)) / np.where(Zb.std(0) == 0, 1, Zb.std(0))
        _, _, vt = np.linalg.svd(Zc, full_matrices=False)
        me = Zc @ vt[0]
        if np.corrcoef(me, Zc.mean(1))[0, 1] < 0:
            me = -me
        me = (me - me.mean()) / me.std()
        return (Zc * me[:, None]).mean(0)    # Pearson correlation of each gene with the eigengene

    full = kme(Z)
    rng = np.random.default_rng(0)
    ranks, top1, gap = [], [], []
    for _ in range(n_boot):
        k = kme(Z[rng.integers(0, len(Z), len(Z))])
        order = np.argsort(-k)
        ranks.append(int(np.where(order == pi)[0][0]) + 1)
        top1.append(m3[order[0]])
        gap.append(k[pi] - np.max(np.delete(k, pi)))
    ranks = np.array(ranks)
    t = pd.Series(top1).value_counts()
    res = {"n_genes": len(m3), "n_specimens": len(Z), "full_rank": int((full > full[pi]).sum()) + 1,
           "full_kme": float(full[pi]), "next_best_full": float(np.sort(full)[-2]),
           "p_rank1": float((ranks == 1).mean()), "p_top3": float((ranks <= 3).mean()),
           "median_gap": float(np.median(gap)), "gap_ci": (float(np.percentile(gap, 2.5)), float(np.percentile(gap, 97.5)))}
    # Also within each cohort separately (WGCNA was built on the older waves)
    for coh in ["Waves1+2", "Waves3+4"]:
        m = clin.cohort.eq(coh).to_numpy()
        k = kme(Z[m])
        res[f"rank_in_{coh}"] = int((k > k[pi]).sum()) + 1
    print(res)
    print("top-1 gene across bootstraps:", t.head(5).to_dict())
    pd.DataFrame({"rank": ranks, "gap": gap, "top1": top1}).to_csv(RESULTS / "part2_kme_bootstrap.csv", index=False)
    pd.Series(res).to_csv(RESULTS / "part2_kme_summary.csv")


# ---------------------------------------------------------------- G4 correlation screen FDR
def gap_fdr():
    F = load_frame()
    auc, S = F["auc"], F["states"]
    s3 = pd.read_excel(SUPP / "S3_drug_cell_type.xlsx", header=1)
    rows = []
    for d in auc.columns:
        for ct in CELL_TYPES:
            m = auc[d].notna()
            if m.sum() < 20:
                continue
            r, p = stats.pearsonr(auc.loc[m, d], S.loc[m, ct])
            rows.append({"inhibitor": d, "cell_type": ct, "n": int(m.sum()), "r": r, "p": p})
    t = pd.DataFrame(rows)
    in_paper = t.inhibitor.isin(s3.inhibitor)
    out = {}
    for label, sub in [("paper's 122 drugs", t[in_paper]), ("all drugs n>=20", t)]:
        sub = sub.copy()
        sub["q_bh"], sub["q_by"] = bh(sub.p), by(sub.p)
        out[label] = {"tests": len(sub), "drugs": sub.inhibitor.nunique(),
                      "drugs_raw_p05": sub[sub.p < 0.05].inhibitor.nunique(),
                      "drugs_bonferroni": sub[sub.p < 0.05 / len(sub)].inhibitor.nunique(),
                      "drugs_BH_05": sub[sub.q_bh < 0.05].inhibitor.nunique(),
                      "drugs_BY_05": sub[sub.q_by < 0.05].inhibitor.nunique()}
        if label.startswith("paper"):
            mm = sub.merge(s3, on=["inhibitor", "cell_type"])
            out[label]["r_agreement_with_TableS3"] = float(np.corrcoef(mm.r, mm.cor)[0, 1])
            out[label]["paper_BY_drugs (Table S3)"] = s3[s3.cor_fdr < 0.05].inhibitor.nunique()
            sub.to_csv(RESULTS / "part2_fdr_correlations.csv", index=False)
    o = pd.DataFrame(out)
    o.to_csv(RESULTS / "part2_fdr_summary.csv")
    print(o.to_string())
    # Effect sizes behind the 73
    sig = s3[s3.cor_fdr < 0.05]
    print(f"|r| among paper-significant pairs: median {sig.cor.abs().median():.3f}, "
          f"share with |r|<0.2: {(sig.cor.abs() < 0.2).mean():.2f}")


# ---------------------------------------------------------------- G5 PC1 only
def gap_pc2():
    F = load_frame()
    sys.path.insert(0, str(ROOT / "scripts"))
    from replication_vangalen_scores import load_signatures
    sigs = load_signatures()
    meta, X, auc = F["meta"], F["X"], F["auc"]
    bm = meta.specimenType.eq("Bone Marrow Aspirate").to_numpy()
    tr = meta.cohort.eq("Waves1+2").to_numpy()
    pcs, ve = {}, []
    for ct, genes in sigs.items():
        g = [x for x in genes if x in X.columns]
        B = X[g].to_numpy(float)
        Z = np.empty_like(B)
        for grp in (bm, ~bm):
            mu, sd = B[grp & tr].mean(0), B[grp & tr].std(0, ddof=1)
            Z[grp] = (B[grp] - mu) / np.where(sd == 0, 1, sd)
        _, s, vt = np.linalg.svd(Z[tr], full_matrices=False)
        v = s ** 2 / (s ** 2).sum()
        p1, p2 = Z @ vt[0], Z @ vt[1]
        if np.corrcoef(p1, Z.mean(1))[0, 1] < 0:
            p1 = -p1
        pcs[f"{ct}_PC1"], pcs[f"{ct}_PC2"] = p1, p2
        top2 = pd.Series(vt[1], index=g).abs().sort_values(ascending=False).head(5).index.tolist()
        ve.append({"cell_type": ct, "genes": len(g), "PC1_var": v[0], "PC2_var": v[1], "PC3_var": v[2],
                   "PC2_top_genes": ", ".join(top2)})
    ve = pd.DataFrame(ve)
    P = pd.DataFrame(pcs, index=meta.index)
    print(ve.round(3).to_string(index=False))

    pc1c = [c for c in P if c.endswith("PC1")]
    pc2c = [c for c in P if c.endswith("PC2")]
    rows = []
    for d in auc.columns:
        m = auc[d].notna().to_numpy()
        if m.sum() < 60:
            continue
        y = auc[d].to_numpy(float)
        # in-sample nested F-test: PC1s vs PC1s + PC2s
        X1 = np.column_stack([np.ones(m.sum()), P.loc[m, pc1c]])
        X2 = np.column_stack([X1, P.loc[m, pc2c]])
        rss1 = np.sum((y[m] - X1 @ np.linalg.lstsq(X1, y[m], rcond=None)[0]) ** 2)
        rss2 = np.sum((y[m] - X2 @ np.linalg.lstsq(X2, y[m], rcond=None)[0]) ** 2)
        df1, df2 = len(pc2c), m.sum() - X2.shape[1]
        Fst = ((rss1 - rss2) / df1) / (rss2 / df2)
        # held-out: Waves1+2 -> Waves3+4 ridge
        mt, me = m & tr, m & ~tr
        held = {}
        if mt.sum() >= 50 and me.sum() >= 30:
            for name, cols in [("pc1", pc1c), ("pc12", pc1c + pc2c)]:
                sc = StandardScaler().fit(P.loc[mt, cols])
                rg = RidgeCV(alphas=np.logspace(-2, 4, 25)).fit(sc.transform(P.loc[mt, cols]), y[mt])
                held[name] = stats.spearmanr(rg.predict(sc.transform(P.loc[me, cols])), y[me])[0]
        rows.append({"inhibitor": d, "n": int(m.sum()), "dR2": (rss1 - rss2) / np.sum((y[m] - y[m].mean()) ** 2),
                     "F_p": stats.f.sf(Fst, df1, df2), **{f"heldout_rho_{k}": v for k, v in held.items()}})
    t = pd.DataFrame(rows)
    t["q"] = bh(t.F_p)
    t.to_csv(RESULTS / "part2_pc2_drugs.csv", index=False)
    ve.to_csv(RESULTS / "part2_pc_variance.csv", index=False)
    h = t.dropna(subset=["heldout_rho_pc1"])
    d = h.heldout_rho_pc12 - h.heldout_rho_pc1
    print(f"drugs tested {len(t)}; PC2 block significant at BH q<0.05: {(t.q < 0.05).sum()}; "
          f"median in-sample dR2 {t.dR2.median():.3f}")
    print(f"held-out ({len(h)} drugs): median rho PC1 {h.heldout_rho_pc1.median():.3f} -> PC1+PC2 "
          f"{h.heldout_rho_pc12.median():.3f}; improved on {(d > 0).sum()}/{len(h)}; "
          f"Wilcoxon p={stats.wilcoxon(d).pvalue:.2g}")
    print(t.sort_values("F_p").head(8).round(4).to_string(index=False))


# ---------------------------------------------------------------- G6 missingness
def gap_missing():
    F = load_frame()
    meta, auc = F["meta"], F["auc"]
    raw = pd.read_csv(RAW / "beataml_wv1to4_raw_inhibitor_v4_dbgap.txt", sep="\t", low_memory=False)
    raw["base_samp"] = raw.dbgap_rnaseq_sample.fillna(raw.dbgap_dnaseq_sample).astype(str).str[:-1]
    raw = raw[raw.base_samp.isin(meta.index)]
    assayed = raw.groupby(["base_samp", "inhibitor"]).passed_qc.agg(["size", "mean"]).reset_index()
    fits = pd.read_csv(RAW / "beataml_probit_curve_fits_v4_dbgap.txt", sep="\t", low_memory=False)
    fits["base_samp"] = fits.dbgap_rnaseq_sample.fillna(fits.dbgap_dnaseq_sample).astype(str).str[:-1]
    fits = fits[fits.base_samp.isin(meta.index)]
    fit_set = fits.set_index(["base_samp", "inhibitor"])
    ass_set = set(zip(assayed.base_samp, assayed.inhibitor))
    cats = {"observed": 0, "never assayed": 0, "assayed, no curve fit (QC)": 0,
            "curve increasing (dropped)": 0, "curve not converged": 0}
    cell_cat = pd.DataFrame("observed", index=auc.index, columns=auc.columns)
    for b in auc.index:
        for d in auc.columns:
            if not np.isnan(auc.at[b, d]):
                cats["observed"] += 1
                continue
            if (b, d) in fit_set.index:
                r = fit_set.loc[(b, d)]
                r = r.iloc[0] if isinstance(r, pd.DataFrame) else r
                k = "curve increasing (dropped)" if r.curve_type == "increasing" else "curve not converged"
            elif (b, d) in ass_set:
                k = "assayed, no curve fit (QC)"
            else:
                k = "never assayed"
            cats[k] += 1
            cell_cat.at[b, d] = k
    tot = auc.size
    ct = pd.Series(cats).to_frame("cells").assign(share=lambda x: x.cells / tot)
    print(ct.round(3).to_string())
    ct.to_csv(RESULTS / "part2_missing_categories.csv")

    # Panel version explains "never assayed": which drug set did each patient get?
    panel = raw.groupby("base_samp").inhibitor_panel.agg(lambda s: "|".join(sorted(set(s))))
    print("\npanels per patient (top):", panel.value_counts().head(5).to_dict())

    # Per-patient missingness among drugs that WERE on the patient's panel-era drug set
    miss = auc.isna()
    pp = pd.DataFrame({"frac_missing": miss.mean(1),
                       "frac_increasing": (cell_cat == "curve increasing (dropped)").mean(1),
                       "wave34": meta.cohort.eq("Waves3+4").astype(int),
                       "PB": meta.specimenType.ne("Bone Marrow Aspirate").astype(int),
                       "blasts_BM": pd.to_numeric(meta["%.Blasts.in.BM"], errors="coerce"),
                       "blasts_PB": pd.to_numeric(meta["%.Blasts.in.PB"], errors="coerce"),
                       "wbc": pd.to_numeric(meta.wbcCount, errors="coerce"),
                       "not_dx": meta.diseaseStageAtSpecimenCollection.ne("Initial Diagnosis").astype(int),
                       "monocyte_like": F["states"]["Monocyte-like"], "HSC_like": F["states"]["HSC-like"],
                       "mean_z_auc": ((auc - auc.mean()) / auc.std()).mean(1)})
    rows = []
    for ycol in ["frac_missing", "frac_increasing"]:
        for x in ["wave34", "PB", "blasts_BM", "blasts_PB", "wbc", "not_dx", "monocyte_like", "HSC_like", "mean_z_auc"]:
            m = pp[[ycol, x]].dropna()
            r, p = stats.spearmanr(m[ycol], m[x])
            rows.append({"outcome": ycol, "predictor": x, "n": len(m), "spearman": r, "p": p})
    mt = pd.DataFrame(rows)
    mt["q"] = bh(mt.p)
    mt.to_csv(RESULTS / "part2_missing_predictors.csv", index=False)
    print(mt.round(4).to_string(index=False))

    # Cell-level: is an increasing (dropped) curve for drug d predicted by the patient's AUC on
    # other drugs?  If so the dropped values are informative and imputation from neighbours is biased.
    z = (auc - auc.mean()) / auc.std()
    inc = (cell_cat == "curve increasing (dropped)")
    res = []
    for d in auc.columns:
        mask = inc[d] | auc[d].notna()
        if inc[d].sum() < 15:
            continue
        other = z.drop(columns=d).mean(1)
        res.append({"inhibitor": d, "n_increasing": int(inc[d].sum()),
                    "AUROC_other_drug_resistance": roc_auc_score(inc[d][mask], other[mask])})
    r = pd.DataFrame(res)
    r.to_csv(RESULTS / "part2_missing_increasing_by_drug.csv", index=False)
    if len(r):
        print(f"\nIncreasing-curve drops: {len(r)} drugs with >=15; median AUROC of 'resistant elsewhere' "
              f"for predicting a drop: {r.AUROC_other_drug_resistance.median():.3f}")
        print(r.sort_values("n_increasing", ascending=False).head(8).round(3).to_string(index=False))


# ---------------------------------------------------------------- G7 cohort exclusions (survival RF)
def gap_cohort(n_rep=5):
    from sksurv.ensemble import RandomSurvivalForest
    from sklearn.model_selection import KFold
    D = rna560()
    feats = pd.concat([D["states"], D["mes"].drop(columns=["M3_noPEAR1"])], axis=1)
    mut = D["muts"]
    mcols = [g for g in mut.columns if mut[g].sum() >= 10]
    feats = feats.join(mut[mcols].fillna(0))
    out, imps = [], {}
    for label, denovo in [("de novo (paper)", True), ("all AML incl. transformed/therapy", False)]:
        S = survival_frame(D, denovo).join(feats)
        S = S.dropna(subset=["T", "E", "age"] + list(D["states"].columns))
        Xcols = ["age"] + list(feats.columns)
        X = S[Xcols].fillna(0).to_numpy(float)
        y = np.array(list(zip(S.E.astype(bool), S["T"])), dtype=[("e", bool), ("t", float)])
        cis, imp = [], np.zeros(len(Xcols))
        for rep in range(n_rep):
            for trn, tst in KFold(5, shuffle=True, random_state=rep).split(X):
                rsf = RandomSurvivalForest(n_estimators=300, min_samples_leaf=15, max_features="sqrt",
                                           n_jobs=-1, random_state=rep).fit(X[trn], y[trn])
                base = rsf.score(X[tst], y[tst])
                cis.append(base)
                rng = np.random.default_rng(rep)
                for j in range(len(Xcols)):
                    Xp = X[tst].copy()
                    Xp[:, j] = rng.permutation(Xp[:, j])
                    imp[j] += base - rsf.score(Xp, y[tst])
        imp /= n_rep * 5
        imps[label] = pd.Series(imp, index=Xcols)
        out.append({"cohort": label, "n": len(S), "events": int(S.E.sum()),
                    "transformed": int((S.transformed == "TRUE").sum()), "therapy_related": int(S.therapy.sum()),
                    "cv_cindex": float(np.mean(cis)), "cv_cindex_sd": float(np.std(cis))})
    o = pd.DataFrame(out)
    I = pd.DataFrame(imps)
    I["rank_denovo"] = I.iloc[:, 0].rank(ascending=False)
    I["rank_all"] = I.iloc[:, 1].rank(ascending=False)
    I = I.sort_values(I.columns[0], ascending=False)
    o.to_csv(RESULTS / "part2_cohort_rsf.csv", index=False)
    I.to_csv(RESULTS / "part2_cohort_rsf_importance.csv")
    print(o.round(3).to_string(index=False))
    print(I.head(12).round(4).to_string())
    print("Spearman of importances between cohorts:",
          round(stats.spearmanr(I.iloc[:, 0], I.iloc[:, 1])[0], 3))


# ---------------------------------------------------------------- G8 drug families vs AUC clustering
def gap_families():
    clin = load_clinical()
    auc = load_auc_long()
    am = auc.pivot_table(index="base_samp", columns="inhibitor", values="auc")
    keep = clin[clin.manuscript_inhibitor.eq("yes")].base_samp
    am = am.loc[am.index.isin(keep)]
    am = am.loc[:, am.notna().sum() >= 150]
    z = (am - am.mean()) / am.std()
    resid = z.sub(z.mean(1), axis=0)          # remove each patient's general sensitivity
    fam = pd.read_excel(RAW / "beataml_drug_families.xlsx", sheet_name="drug_family")
    syn = pd.read_excel(RAW / "beataml_drug_families.xlsx", sheet_name="synonyms")
    fam = fam[fam.inhibitor.isin(am.columns)]
    nfam = fam.groupby("inhibitor").family.nunique()
    print(f"{am.shape[1]} drugs with >=150 patients; {fam.inhibitor.nunique()} have a family; "
          f"families per drug: median {nfam.median():.0f}, >1 for {(nfam > 1).sum()} drugs (polypharmacology)")

    out = {}
    for label, M in [("raw z-AUC", z), ("after removing general sensitivity", resid)]:
        C = M.corr(method="spearman", min_periods=60)
        drugs = [d for d in C.index if d in set(fam.inhibitor)]
        C = C.loc[drugs, drugs].fillna(0)
        fams = fam.groupby("inhibitor").family.apply(set)
        same = np.array([[len(fams[a] & fams[b]) > 0 for b in drugs] for a in drugs])
        iu = np.triu_indices(len(drugs), 1)
        w, b = C.to_numpy()[iu][same[iu]], C.to_numpy()[iu][~same[iu]]
        # Permutation test: shuffle family labels across drugs
        rng = np.random.default_rng(0)
        obs = w.mean() - b.mean()
        perm = []
        lab = list(fams[drugs])
        for _ in range(500):
            pl = rng.permutation(len(lab))
            sp = np.array([[len(lab[pl[i]] & lab[pl[j]]) > 0 for j in range(len(drugs))] for i in range(len(drugs))])
            perm.append(C.to_numpy()[iu][sp[iu]].mean() - C.to_numpy()[iu][~sp[iu]].mean())
        # clustering vs primary family
        primary = fam.drop_duplicates("inhibitor").set_index("inhibitor").family.reindex(drugs)
        k = primary.nunique()
        Lk = linkage(squareform(1 - C.to_numpy(), checks=False), "average")
        cl = fcluster(Lk, k, "maxclust")
        out[label] = {"drugs": len(drugs), "within_family_mean_rho": w.mean(), "between_mean_rho": b.mean(),
                      "perm_p": float((np.array(perm) >= obs).mean()), "k_clusters": k,
                      "ARI_vs_primary_family": adjusted_rand_score(primary, cl)}
        if label.startswith("after"):
            # Per-family cohesion, and drugs whose nearest AUC neighbours share no family
            fr = []
            for f in fam.family.unique():
                mem = [d for d in drugs if f in fams[d]]
                if len(mem) < 3:
                    continue
                sub = C.loc[mem, mem].to_numpy()[np.triu_indices(len(mem), 1)]
                fr.append({"family": f, "n_drugs": len(mem), "mean_within_rho": sub.mean()})
            fr = pd.DataFrame(fr).sort_values("mean_within_rho")
            fr.to_csv(RESULTS / "part2_family_cohesion.csv", index=False)
            nn = []
            for d in drugs:
                nb = C.loc[d].drop(d).sort_values(ascending=False).head(3)
                share = [len(fams[d] & fams[x]) > 0 for x in nb.index]
                nn.append({"inhibitor": d, "families": "; ".join(sorted(fams[d])),
                           "nearest": "; ".join(f"{x} ({v:.2f})" for x, v in nb.items()),
                           "neighbours_sharing_family": int(sum(share)), "top_rho": nb.iloc[0]})
            nn = pd.DataFrame(nn)
            nn.to_csv(RESULTS / "part2_family_neighbours.csv", index=False)
            print(fr.round(3).to_string(index=False))
            dis = nn[(nn.neighbours_sharing_family == 0) & (nn.top_rho > 0.4)].sort_values("top_rho", ascending=False)
            print(f"\ndrugs whose 3 nearest AUC neighbours share none of their families, top rho>0.4: {len(dis)} / {len(nn)}")
            print(dis.head(12).to_string(index=False))
    o = pd.DataFrame(out)
    o.to_csv(RESULTS / "part2_family_summary.csv")
    print(o.to_string())


GAPS = {"exvivo": gap_exvivo_clinical, "pear1": gap_pear1, "kme": gap_kme, "fdr": gap_fdr,
        "pc2": gap_pc2, "missing": gap_missing, "cohort": gap_cohort, "families": gap_families}

if __name__ == "__main__":
    for g in (sys.argv[1:] or GAPS):
        print(f"\n{'=' * 20} {g} {'=' * 20}")
        GAPS[g]()
