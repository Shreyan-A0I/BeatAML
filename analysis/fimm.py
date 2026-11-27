"""External tests in FIMM (Malani et al. 2022, Helsinki), all models frozen on Beat AML.

    python -m analysis.fimm [recommender|induction|all]

recommender: Vania's per-drug ridge (src.models.PerDrugRidge), trained on all 429 Beat AML
  patients, ranks each FIMM patient's drugs. Features are recomputed so they mean the same
  thing in both cohorts (frozen Beat AML PC1 loadings, genes z-scored within each cohort).
  Check first: the same features reproduce the Beat AML Waves1+2 -> Waves3+4 holdout.
induction: the induction-failure model (ELN 2017 version, since FIMM has ELN 2017 only),
  trained on all Beat AML induction patients, scored once on FIMM intensive-induction patients.

Outputs: results/tables/fimm_recommender*.csv, fimm_induction.csv
"""
from __future__ import annotations

import sys

import xgboost  # noqa: F401  (import before sklearn/scipy: avoids an OpenMP clash that segfaults)
import numpy as np
import pandas as pd
from scipy import stats

from analysis.common import ROOT
from analysis.scores import LSC17, _z, apply_scores, beat_expression, fit_loadings, gene_sets, harmonise

FIMM = ROOT / "data/raw/fimm/Functional_Precision_Medicine_Tumor_Board_AML"
RESULTS = ROOT / "results/tables"
MODULES = ["me.M3", "me.M7", "me.M8", "me.M9", "me.M12", "me.M14", "me.M16", "me.M21", "me.M23"]  # Vania's 9
FUSIONS = {"fus.RUNX1-RUNX1T1": "AML1_ETO", "fus.CBFB-MYH11": "CBFB", "fus.MLLT3-KMT2A": "MLL_AF9", "fus.PML-RARA": "PML"}


def fimm_samples():
    """One FIMM sample per patient with drug screen + RNA (Arrio's loader). Returns its dict + annotation."""
    from src.validation.fimm_pear1 import load_fimm
    f = load_fimm(FIMM)
    ann = pd.read_excel(FIMM / "File_0_Common_sample_annotation_252S.xlsx").set_index("Sample_ID")
    clin = pd.read_excel(FIMM / "File_1.1_Clinical_summary_186_Patients.xlsx")
    clin["pid"] = clin.Patient_ID.str.strip()
    f["ann"], f["clin"] = ann.loc[f["cov"].index], clin.set_index("pid")
    return f


def fimm_expression(rna: pd.DataFrame) -> pd.DataFrame:
    """FIMM log2 CPM (Ensembl x samples) -> samples x symbols, using Beat AML's id -> symbol map."""
    from src.load import load_expression
    idx = load_expression().index
    sym = pd.Series(idx.get_level_values("display_label"), index=idx.get_level_values("stable_id"))
    sym = sym[~sym.index.duplicated()]
    E = rna.loc[rna.index.intersection(sym.index)].T
    E.columns = sym[E.columns].to_numpy()
    return harmonise(E.loc[:, ~E.columns.duplicated()])


def shared_scores(E_f: pd.DataFrame):
    """Cell-type, module, PEAR1 and LSC17 scores with one definition in both cohorts."""
    E_b, clin_b = beat_expression()
    ref_b = E_b[clin_b.cohort.eq("Waves1+2").to_numpy()]
    W = fit_loadings(ref_b, gene_sets(), set(E_f.columns))
    return apply_scores(E_b, ref_b, W), apply_scores(E_f, E_f, W)


def drug_key(name: str) -> str:
    from src.validation.fimm_pear1 import drug_key as k
    return k(name)


def fimm_targets(f, drugs_beat):
    """FIMM drug-standardised response on drugs shared with the Beat AML panel.
    -sDSS (higher = resistant), medium (CM vs MCM) regressed out per drug, then rank-normal
    scores so z < -0.84 is each drug's most sensitive ~20%, as in Beat AML."""
    resp, cm = f["resp"], f["cov"]["cm"]
    fmap = {}
    for d in resp.columns:
        fmap.setdefault(drug_key(d), d)
    pairs = {b: fmap[drug_key(b)] for b in drugs_beat if drug_key(b) in fmap}
    Z = {}
    for b, fd in pairs.items():
        y = resp[fd].dropna()
        X = np.column_stack([np.ones(len(y)), cm[y.index]])
        r = y - X @ np.linalg.lstsq(X, y.to_numpy(), rcond=None)[0]
        Z[b] = pd.Series(stats.norm.ppf((r.rank() - 0.5) / len(r)), index=y.index)
    return pd.DataFrame(Z)


def recommender():
    from src.features import alteration_table
    from src.models import GenomicRuleBaseline, PerDrugRidge
    from src.rank_eval import per_patient_metrics
    from src.targets import DrugZScaler, load_recommender_data, select_drugs

    data = load_recommender_data()
    drugs = select_drugs(data.auc, data.train)
    vf = data.features()                                   # Vania's features (mutations, fusions, clinical)
    f = fimm_samples()
    sc_b, sc_f = shared_scores(fimm_expression(f["rna"]))
    expr_cols = [c for c in sc_b.columns if c.startswith("ct.")] + MODULES

    # Beat AML features: shared expression scores + Vania's genomic/clinical columns
    Fb = sc_b.loc[data.meta.dbgap_rnaseq_sample, expr_cols].set_axis(data.patients)
    Fb["rna.PEAR1"] = sc_b.loc[data.meta.dbgap_rnaseq_sample, "PEAR1"].to_numpy()
    Fb["rna.LSC17"] = sc_b.loc[data.meta.dbgap_rnaseq_sample, "LSC17"].to_numpy()
    Fb = Fb.join(vf[[c for c in vf.columns if c.startswith(("mut.", "fus.")) or c in ("clin.age", "clin.is_bm")]])
    Fb = Fb.join(alteration_table(data.clinical, data.patients)[[f"alt.{a}" for a in
                 ["FLT3_ITD", "FLT3", "KIT", "NRAS", "KRAS", "PTPN11", "JAK2", "IDH1", "IDH2"] if f"alt.{a}" in
                 alteration_table(data.clinical, data.patients).columns]])

    # FIMM features, same columns
    clin, ann = f["clin"], f["ann"]
    pid = ann.Patient_ID.str.replace("FPM_", "", regex=False).str.strip()
    mut = pd.read_excel(FIMM / "File_6_Binary_mutation_225S_57G.xlsx").set_index("Unnamed: 0").T
    Ff = sc_f[expr_cols].copy()
    Ff["rna.PEAR1"], Ff["rna.LSC17"] = sc_f.PEAR1, sc_f.LSC17
    Ff["clin.age"] = clin.loc[pid, "Age at diagnosis"].to_numpy()
    Ff["clin.is_bm"] = ann.Tissue.eq("Bone marrow").astype(float).to_numpy()
    for c in Fb.columns:
        g = c.split(".", 1)[1]
        if c.startswith(("mut.", "alt.")) and g not in ("FLT3_ITD",):
            Ff[c] = mut[g].reindex(Ff.index).to_numpy() if g in mut.columns else 0.0
    itd = clin.loc[pid, "FLT3_ITD"].to_numpy()
    Ff["mut.FLT3_ITD"] = Ff["alt.FLT3_ITD"] = itd
    for c, col in FUSIONS.items():
        Ff[c] = clin.loc[pid, col].fillna(0).to_numpy()
    Ff = Ff[Fb.columns]
    print(f"FIMM: {len(Ff)} patients; mutation calls missing for {int(Ff['mut.NPM1'].isna().sum())}")

    # step 1: do the shared features reproduce Vania's Beat AML holdout?
    auc_tr, auc_te = data.auc.loc[data.train, drugs], data.auc.loc[data.test, drugs]
    z_te = DrugZScaler().fit(auc_tr).transform(auc_te)
    m = PerDrugRidge().fit(auc_tr, Fb)
    pp = per_patient_metrics(m.predict(Fb.loc[auc_te.index], drugs), z_te)
    print(f"Beat AML holdout with shared features: prec@5 {pp['prec@5'].mean():.3f} (Vania: 0.456), "
          f"chance {pp.prevalence.mean():.3f}, n={len(pp)}")

    # step 2: freeze on all Beat AML, score FIMM
    zf = fimm_targets(f, drugs)
    shared = list(zf.columns)
    print(f"drugs shared with Vania's panel: {len(shared)}")
    auc_all = data.auc[drugs]
    models = {"M1 ridge, all features": PerDrugRidge().fit(auc_all, Fb),
              "M1 ridge, mutations + clinical": PerDrugRidge(blocks=("clin", "mut", "fus")).fit(auc_all, Fb),
              "B1 genomic rules": GenomicRuleBaseline().fit(auc_all, Fb)}
    rows, per = [], {}
    for name, mod in models.items():
        p = per_patient_metrics(mod.predict(Ff, shared), zf)
        per[name] = p
    rng = np.random.default_rng(0)
    for name, p in per.items():
        d = p["prec@5"] - per["B1 genomic rules"].loc[p.index, "prec@5"]
        boot = [d.sample(len(d), replace=True, random_state=int(s)).mean() for s in rng.integers(0, 1e9, 1000)]
        rows.append({"model": name, "n_patients": len(p), "prec@5": p["prec@5"].mean(), "chance": p.prevalence.mean(),
                     "spearman": p.spearman.mean(), "ndcg@10": p["ndcg@10"].mean(),
                     "gain_vs_B1": d.mean(), "gain_lo": np.percentile(boot, 2.5), "gain_hi": np.percentile(boot, 97.5)})
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "fimm_recommender.csv", index=False)
    print(out.round(3).to_string(index=False))

    # per drug: predicted vs observed across FIMM patients
    pred = models["M1 ridge, all features"].predict(Ff, shared)
    pd_ = pd.DataFrame([{"drug": d, "n": int(zf[d].notna().sum()),
                         "spearman": stats.spearmanr(pred.loc[zf[d].dropna().index, d], zf[d].dropna()).statistic}
                        for d in shared]).sort_values("spearman", ascending=False)
    pd_.to_csv(RESULTS / "fimm_recommender_per_drug.csv", index=False)
    print("median per-drug Spearman", round(pd_.spearman.median(), 3))
    print(pd_[pd_.drug.str.contains("Venetoclax|Trametinib|Selumetinib|Quizartinib|Gilteritinib|Cytarabine|Panobinostat")].round(3).to_string(index=False))
    # subgroup: no actionable lesion (B1 falls back to population order)
    rule = models["B1 genomic rules"].matched_drugs(Ff).reindex(columns=shared, fill_value=0).sum(axis=1) == 0
    for name in models:
        p = per[name]
        print(f"  no actionable lesion (n={int(rule.loc[p.index].sum())}): {name} prec@5 {p.loc[rule.loc[p.index], 'prec@5'].mean():.3f}")


def induction():
    """Frozen Beat AML induction model (ELN 2017 version, features FIMM also records) -> FIMM."""
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.linear_model import LogisticRegression
    from xgboost import XGBClassifier
    from analysis.induction import cohort

    X, y, info, _ = cohort()
    from analysis.scores import scores
    clin_b = scores()["beat_clin"].loc[X.index]
    X = X.copy()
    X["ELN_adv"] = clin_b.ELN2017.eq("Adverse").astype(float)
    X["ELN_fav"] = clin_b.ELN2017.eq("Favorable").astype(float)
    # FIMM records no peripheral-blood blast % and no RUNX1/ASXL1 status at diagnosis for everyone; keep the rest
    feats = ["age", "ELN_adv", "ELN_fav", "denovo", "female", "NPM1", "FLT3_ITD", "TP53", "RUNX1", "ASXL1",
             "wbcCount", "%.Blasts.in.BM", "LDH", "hemoglobin", "plateletCount", "albumin", "creatinine"]

    c = pd.read_excel(FIMM / "File_1.1_Clinical_summary_186_Patients.xlsx")
    c = c[c.Highdose_cyt_induction_therapy_pt.eq(1)]          # intensive (cytarabine-based) induction
    mut = pd.read_excel(FIMM / "File_6_Binary_mutation_225S_57G.xlsx").set_index("Unnamed: 0").T
    mut.index = mut.index.str.rsplit("_", n=1).str[0]           # sample -> patient
    mut = mut.groupby(level=0).max()
    pid = c.Patient_ID.str.strip()
    F = pd.DataFrame(index=pid.to_numpy())
    F["age"] = c["Age at diagnosis"].to_numpy()
    F["ELN_adv"] = c.eln2017_risk_class_qp.eq(3).astype(float).where(c.eln2017_risk_class_qp.notna()).to_numpy()
    F["ELN_fav"] = c.eln2017_risk_class_qp.eq(1).astype(float).where(c.eln2017_risk_class_qp.notna()).to_numpy()
    F["denovo"] = c.hema_diagnoses.eq("AML").astype(float).to_numpy()
    F["female"] = c.Gender.eq("Female").astype(float).to_numpy()
    F["NPM1"] = c.bm_npm1_d.to_numpy()
    F["FLT3_ITD"] = c.FLT3_ITD.to_numpy()
    for g in ["TP53", "RUNX1", "ASXL1"]:
        F[g] = mut[g].reindex(F.index).to_numpy()
    F["NPM1"] = F.NPM1.fillna(mut["NPM1"].reindex(F.index))
    # unit conversions to Beat AML units (FIMM: g/L, umol/L)
    F["wbcCount"] = c["b-leuk"].to_numpy()
    F["%.Blasts.in.BM"] = c["bm blast p dg"].to_numpy()
    F["LDH"] = c["p-ld"].to_numpy()
    F["hemoglobin"] = c["b-hb"].to_numpy() / 10
    F["plateletCount"] = c["b-trom"].to_numpy()
    F["albumin"] = c["p-alb"].to_numpy() / 10
    F["creatinine"] = c["p-krea"].to_numpy() / 88.4
    yf = c.First_cr_pt.ne(1).astype(int).to_numpy()            # no CR recorded = refractory
    print(f"FIMM intensive induction: {len(F)} patients, {yf.sum()} without CR ({yf.mean():.0%})")
    print("missing per feature in FIMM:", F.isna().mean().round(2)[F.isna().mean() > 0].to_dict())

    rows = []
    for name, cols, model in [
            ("ELN 2017 only, logistic", ["ELN_adv", "ELN_fav"], LogisticRegression(max_iter=5000)),
            ("ELN 2017 + age, logistic", ["ELN_adv", "ELN_fav", "age"], LogisticRegression(max_iter=5000)),
            ("Clinical model, gradient boosting", feats,
             XGBClassifier(n_estimators=200, learning_rate=0.05, subsample=0.8, colsample_bytree=0.7, max_depth=2,
                           eval_metric="logloss", random_state=0)),
            ("Clinical model, logistic", feats, LogisticRegression(C=0.1, max_iter=5000))]:
        pipe = make_pipeline(SimpleImputer(strategy="median"), model) if "logistic" not in name else \
            make_pipeline(SimpleImputer(strategy="median"), __import__("sklearn.preprocessing", fromlist=["x"]).StandardScaler(), model)
        pipe.fit(X[cols], y)
        p = pipe.predict_proba(F[cols])[:, 1]
        auc = roc_auc_score(yf, p)
        rng = np.random.default_rng(0)
        boot = []
        for _ in range(2000):
            i = rng.integers(0, len(yf), len(yf))
            if yf[i].min() != yf[i].max():
                boot.append(roc_auc_score(yf[i], p[i]))
        rows.append({"model": name, "n": len(yf), "events": int(yf.sum()), "AUROC": auc,
                     "lo": np.percentile(boot, 2.5), "hi": np.percentile(boot, 97.5)})
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "fimm_induction.csv", index=False)
    print(out.round(3).to_string(index=False))


def interaction_test(z, g, s):
    """OLS z ~ g + s + g*s. Returns (interaction coefficient, p)."""
    import statsmodels.api as sm
    ok = z.notna() & g.notna() & s.notna()
    X = sm.add_constant(np.column_stack([g[ok], s[ok], g[ok] * s[ok]]))
    r = sm.OLS(z[ok].to_numpy(), X).fit()
    return r.params[3], r.pvalues[3], int(g[ok].sum()), int(ok.sum())


def pairs():
    """Every interaction the shrinkage screen replicated in Beat AML, tested once in FIMM."""
    from analysis.common import load_frame
    h = pd.read_csv(RESULTS / "interact_hits.csv")
    h = h[h.arm.eq("EB global")].groupby(["gene", "state", "drug"]).agg(
        splits=("seed", "nunique"), known=("known_exact", "first")).reset_index()
    F = load_frame()
    zb = (F["auc"] - F["auc"].mean()) / F["auc"].std()
    f = fimm_samples()
    _, sc_f = shared_scores(fimm_expression(f["rna"]))
    zf = fimm_targets(f, list(h.drug.unique()))
    mut = pd.read_excel(FIMM / "File_6_Binary_mutation_225S_57G.xlsx").set_index("Unnamed: 0").T.reindex(sc_f.index)
    pid = f["ann"].Patient_ID.str.replace("FPM_", "", regex=False).str.strip()
    mut["FLT3_ITD"] = f["clin"].loc[pid, "FLT3_ITD"].to_numpy()
    rows = []
    for r in h.itertuples():
        bb, pb, _, _ = interaction_test(zb[r.drug], F["muts"][r.gene], F["states"][r.state])
        row = {"gene": r.gene, "state": r.state, "drug": r.drug, "beat_splits_replicated": r.splits,
               "in_paper": r.known, "beta_beat": bb, "p_beat": pb}
        if r.drug in zf and r.gene in mut:
            bf, pf, nc, n = interaction_test(zf[r.drug], mut[r.gene], sc_f[f"ct.{r.state}"])
            row.update({"fimm_carriers": nc, "fimm_n": n, "beta_fimm": bf, "p_fimm": pf,
                        "same_sign": bool(np.sign(bf) == np.sign(bb))})
        rows.append(row)
    out = pd.DataFrame(rows).sort_values(["beat_splits_replicated", "p_beat"], ascending=[False, True])
    out.to_csv(RESULTS / "fimm_interactions.csv", index=False)
    print(out.round(3).to_string(index=False))
    t = out.dropna(subset=["beta_fimm"])
    k = int(t.same_sign.sum())
    print(f"testable in FIMM: {len(t)}; same sign {k}/{len(t)} (sign test p="
          f"{stats.binomtest(k, len(t), 0.5, alternative='greater').pvalue:.3g}); same sign and p<0.05: "
          f"{int((t.same_sign & t.p_fimm.lt(0.05)).sum())}")


if __name__ == "__main__":
    step = sys.argv[1] if len(sys.argv) > 1 else "all"
    if step in ("recommender", "all"):
        recommender()
    if step in ("induction", "all"):
        induction()
    if step in ("pairs", "all"):
        pairs()
