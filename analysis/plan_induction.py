"""Plan section 4: who fails standard induction? (outcome: refractory vs CR/CRi)

    python -m analysis.plan_induction

Cohort: Beat AML diagnosis RNA-seq specimens of patients given standard induction
chemotherapy with a recorded response. Develop with repeated nested CV in Waves1+2;
the pre-declared winner (best mean outer AUROC) is refit on all Waves1+2 and
scored ONCE on Waves3+4: AUROC (bootstrap CI), calibration, decision curve,
split-conformal prediction sets (90%), subgroup AUROC.
A drug-response feature set is evaluated on the subset with a drug screen.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from analysis.common import RESULTS, load_frame
from analysis.plan_common import drug_factors, scores

warnings.filterwarnings("ignore")
LABS = ["wbcCount", "%.Blasts.in.BM", "%.Blasts.in.PB", "LDH", "hemoglobin", "plateletCount", "albumin", "creatinine"]


def cohort():
    S = scores()
    c, sc = S["beat_clin"], S["beat"]
    ok = (c.typeInductionTx.eq("Standard Chemotherapy") & c.diseaseStageAtSpecimenCollection.eq("Initial Diagnosis")
          & c.responseToInductionTx.isin(["Complete Response", "Complete Response i", "Refractory"]))
    c, sc = c[ok], sc[ok]
    X = pd.DataFrame(index=c.index)
    X["age"] = c.ageAtDiagnosis
    X["ELN_adv"] = c.ELN2017.eq("Adverse").astype(float)
    X["ELN_fav"] = c.ELN2017.eq("Favorable").astype(float)
    X["ELN_missing"] = c.ELN2017.isna().astype(float)
    X["denovo"] = c.isDenovo.astype(str).eq("TRUE").astype(float)
    X["female"] = c.consensus_sex.eq("Female").astype(float)
    for g, col in {"NPM1": "NPM1", "FLT3_ITD": "FLT3-ITD", "TP53": "TP53", "RUNX1": "RUNX1", "ASXL1": "ASXL1"}.items():
        X[g] = c[col].map({"positive": 1.0, "negative": 0.0})
    for l in LABS:
        X[l] = pd.to_numeric(c[l], errors="coerce")
    X = X.join(sc)
    y = c.responseToInductionTx.eq("Refractory").astype(int)
    # drug factors on the subset with a drug screen
    fac, _, ve = drug_factors()
    meta = load_frame()["meta"]
    fac.index = meta.loc[fac.index, "dbgap_rnaseq_sample"]
    X = X.join(fac)
    info = pd.DataFrame({"train": c.cohort.eq("Waves1+2"), "sex": c.consensus_sex,
                         "race": c.reportedRace.where(c.reportedRace.eq("White"), "Non-white/unknown"),
                         "age": c.ageAtDiagnosis}, index=c.index)
    return X, y, info, ve


FEATURE_SETS = {
    "ELN": ["ELN_adv", "ELN_fav", "ELN_missing"],
    "clinical (ELN+age+labs+5 mutations)": None,     # filled below
    "+ expression scores": None,
    "+ expression + drug factors": None,
}

MODELS = {
    "logistic L2": (LogisticRegression(max_iter=5000), {"logisticregression__C": [0.01, 0.03, 0.1, 0.3, 1]}),
    "logistic L1": (LogisticRegression(penalty="l1", solver="liblinear", max_iter=5000),
                    {"logisticregression__C": [0.03, 0.1, 0.3, 1]}),
    "random forest": (RandomForestClassifier(n_estimators=500, random_state=0, n_jobs=-1),
                      {"randomforestclassifier__min_samples_leaf": [5, 15, 30]}),
    "xgboost": (XGBClassifier(n_estimators=200, learning_rate=0.05, subsample=0.8, colsample_bytree=0.7,
                              random_state=0, n_jobs=4, verbosity=0),
                {"xgbclassifier__max_depth": [1, 2, 3]}),
}


def pipe(model):
    return make_pipeline(IterativeImputer(max_iter=10, random_state=0), StandardScaler(), model)


def net_benefit(y, p, thr):
    tp = ((p >= thr) & (y == 1)).sum()
    fp = ((p >= thr) & (y == 0)).sum()
    n = len(y)
    return tp / n - fp / n * thr / (1 - thr)


def main():
    X, y, info, ve = cohort()
    exprs = [c for c in X.columns if c.startswith(("ct.", "me.")) or c in ("PEAR1", "LSC17")]
    drugs = [c for c in X.columns if c.startswith("df")]
    clin = ["age", "ELN_adv", "ELN_fav", "ELN_missing", "denovo", "female", "NPM1", "FLT3_ITD", "TP53", "RUNX1",
            "ASXL1"] + LABS
    FEATURE_SETS["clinical (ELN+age+labs+5 mutations)"] = clin
    FEATURE_SETS["+ expression scores"] = clin + exprs
    FEATURE_SETS["+ expression + drug factors"] = clin + exprs + drugs
    tr = info.train.to_numpy()
    print(f"cohort {len(y)}: train {tr.sum()} ({y[tr].mean():.0%} refractory), test {(~tr).sum()} "
          f"({y[~tr].mean():.0%} refractory); drug screen for {X.df1.notna().sum()}; drug factors explain {ve:.0%}")

    rows = []
    outer = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=0)
    Xtr, ytr = X[tr], y[tr]
    for fs, cols in FEATURE_SETS.items():
        sub = Xtr[cols].notna().any(axis=1) if fs.endswith("drug factors") else pd.Series(True, index=Xtr.index)
        if fs.endswith("drug factors"):
            sub = Xtr["df1"].notna()
        Xs, ys = Xtr.loc[sub, cols], ytr[sub]
        for mname, (model, grid) in MODELS.items():
            if fs == "ELN" and mname != "logistic L2":
                continue
            aucs = []
            for a, b in outer.split(Xs, ys):
                gs = GridSearchCV(pipe(model), grid, cv=StratifiedKFold(4, shuffle=True, random_state=1), scoring="roc_auc")
                gs.fit(Xs.iloc[a], ys.iloc[a])
                aucs.append(roc_auc_score(ys.iloc[b], gs.predict_proba(Xs.iloc[b])[:, 1]))
            rows.append({"features": fs, "model": mname, "n": len(ys), "cv_auroc": np.mean(aucs), "cv_sd": np.std(aucs)})
            print(rows[-1], flush=True)
    cv = pd.DataFrame(rows)
    cv.to_csv(RESULTS / "plan_induction_cv.csv", index=False)

    # pre-declared winner among sets available for every patient (drug set evaluated on a subset)
    full = cv[~cv.features.str.endswith("drug factors")].sort_values("cv_auroc", ascending=False).iloc[0]
    print("winner:", full.to_dict())
    test_rows = []
    for label, fs, mname in [("winner", full.features, full.model), ("ELN only", "ELN", "logistic L2")]:
        cols = FEATURE_SETS[fs]
        model, grid = MODELS[mname]
        gs = GridSearchCV(pipe(model), grid, cv=StratifiedKFold(5, shuffle=True, random_state=1), scoring="roc_auc")
        gs.fit(Xtr[cols], ytr)
        p = gs.predict_proba(X.loc[~tr, cols])[:, 1]
        yt = y[~tr].to_numpy()
        rng = np.random.default_rng(0)
        bs = []
        for _ in range(1000):
            i = rng.integers(0, len(yt), len(yt))
            if yt[i].min() != yt[i].max():
                bs.append(roc_auc_score(yt[i], p[i]))
        lo, hi = np.percentile(bs, [2.5, 97.5])
        lp = np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
        cal = LogisticRegression(C=1e6).fit(lp[:, None], yt)
        rec = {"model": f"{label}: {fs} / {mname}", "n_test": len(yt), "AUROC": roc_auc_score(yt, p), "lo": lo, "hi": hi,
               "cal_slope": cal.coef_[0, 0], "cal_intercept": cal.intercept_[0], "mean_pred": p.mean(), "obs_rate": yt.mean()}
        for t in (0.2, 0.3, 0.4, 0.5):
            rec[f"NB@{t}"] = net_benefit(yt, p, t)
            rec[f"NB_all@{t}"] = net_benefit(yt, np.ones_like(p), t)
        if label == "winner":
            # split conformal: calibration scores = out-of-fold probabilities in Waves1+2
            oof = cross_val_predict(gs.best_estimator_, Xtr[cols], ytr, cv=StratifiedKFold(5, shuffle=True, random_state=2),
                                    method="predict_proba")
            s_cal = 1 - oof[np.arange(len(ytr)), ytr.to_numpy()]
            qhat = np.quantile(s_cal, np.ceil((len(s_cal) + 1) * 0.9) / len(s_cal))
            P = np.c_[1 - p, p]
            sets = (1 - P) <= qhat
            rec["conformal_coverage"] = sets[np.arange(len(yt)), yt].mean()
            rec["conformal_uncertain"] = (sets.sum(1) == 2).mean()
            single = sets.sum(1) == 1
            rec["conformal_confident_n"] = int(single.sum())
            rec["conformal_confident_acc"] = (sets[single].argmax(1) == yt[single]).mean() if single.any() else np.nan
            # subgroup AUROC
            sg = []
            inf = info[~tr]
            for col, val in [("sex", "Male"), ("sex", "Female"), ("race", "White"), ("race", "Non-white/unknown")]:
                m = (inf[col] == val).to_numpy()
                if len(set(yt[m])) == 2:
                    sg.append({"group": f"{col}={val}", "n": int(m.sum()), "AUROC": roc_auc_score(yt[m], p[m])})
            for lab, m in [("age<60", inf.age.to_numpy() < 60), ("age>=60", inf.age.to_numpy() >= 60)]:
                if len(set(yt[m])) == 2:
                    sg.append({"group": lab, "n": int(m.sum()), "AUROC": roc_auc_score(yt[m], p[m])})
            pd.DataFrame(sg).to_csv(RESULTS / "plan_induction_subgroups.csv", index=False)
            print(pd.DataFrame(sg).round(3).to_string(index=False))
            if mname.startswith("logistic"):
                coef = pd.Series(gs.best_estimator_[-1].coef_[0], index=cols).sort_values()
                coef[coef != 0].to_csv(RESULTS / "plan_induction_coefs.csv")
                print("coefficients (refractory +):\n", coef[coef.abs() > 1e-6].round(3).to_string())
        test_rows.append(rec)
    t = pd.DataFrame(test_rows)
    t.to_csv(RESULTS / "plan_induction_test.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(cv.round(3).to_string(index=False))
        print(t.round(3).T.to_string())


if __name__ == "__main__":
    main()
