"""Fair comparison for plan section 4: every feature set on the SAME patients (those with a
drug screen), CV in Waves1+2 and one Waves3+4 score; paired bootstrap of test AUROC
differences vs ELN; xgboost feature importance of the clinical winner.

    python -m analysis.plan_induction_matched
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold

from analysis.common import RESULTS
from analysis.plan_induction import FEATURE_SETS, LABS, MODELS, cohort, main as _unused, pipe  # noqa: F401


def main():
    X, y, info, _ = cohort()
    exprs = [c for c in X.columns if c.startswith(("ct.", "me.")) or c in ("PEAR1", "LSC17")]
    drugs = [c for c in X.columns if c.startswith("df")]
    clin = ["age", "ELN_adv", "ELN_fav", "ELN_missing", "denovo", "female", "NPM1", "FLT3_ITD", "TP53", "RUNX1",
            "ASXL1"] + LABS
    sets = {"ELN": ["ELN_adv", "ELN_fav", "ELN_missing"], "clinical": clin, "clinical + drug factors": clin + drugs,
            "clinical + expression": clin + exprs, "clinical + expression + drug factors": clin + exprs + drugs}
    m = X.df1.notna()
    X, y, info = X[m], y[m], info[m]
    tr = info.train.to_numpy()
    print(f"matched cohort {len(y)}: train {tr.sum()}, test {(~tr).sum()} ({y[~tr].mean():.0%} refractory)")
    outer = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=0)
    rows, preds = [], {}
    yt = y[~tr].to_numpy()
    for fs, cols in sets.items():
        for mname in (["logistic L2"] if fs == "ELN" else ["logistic L2", "xgboost"]):
            model, grid = MODELS[mname]
            aucs = []
            for a, b in outer.split(X[tr], y[tr]):
                gs = GridSearchCV(pipe(model), grid, cv=StratifiedKFold(4, shuffle=True, random_state=1), scoring="roc_auc")
                gs.fit(X[tr].iloc[a][cols], y[tr].iloc[a])
                aucs.append(roc_auc_score(y[tr].iloc[b], gs.predict_proba(X[tr].iloc[b][cols])[:, 1]))
            gs = GridSearchCV(pipe(model), grid, cv=StratifiedKFold(5, shuffle=True, random_state=1), scoring="roc_auc")
            gs.fit(X[tr][cols], y[tr])
            p = gs.predict_proba(X[~tr][cols])[:, 1]
            preds[(fs, mname)] = p
            rows.append({"features": fs, "model": mname, "cv_auroc": np.mean(aucs), "cv_sd": np.std(aucs),
                         "test_auroc": roc_auc_score(yt, p)})
            print(rows[-1], flush=True)
    rng = np.random.default_rng(0)
    base = preds[("ELN", "logistic L2")]
    for r in rows:
        p = preds[(r["features"], r["model"])]
        d = []
        for _ in range(2000):
            i = rng.integers(0, len(yt), len(yt))
            if yt[i].min() != yt[i].max():
                d.append(roc_auc_score(yt[i], p[i]) - roc_auc_score(yt[i], base[i]))
        r["dAUROC_vs_ELN"], r["d_lo"], r["d_hi"] = np.mean(d), *np.percentile(d, [2.5, 97.5])
    t = pd.DataFrame(rows)
    t.to_csv(RESULTS / "plan_induction_matched.csv", index=False)
    print(t.round(3).to_string(index=False))

    # importance in the full-cohort clinical xgboost winner
    X2, y2, info2, _ = cohort()
    tr2 = info2.train.to_numpy()
    model, grid = MODELS["xgboost"]
    gs = GridSearchCV(pipe(model), grid, cv=StratifiedKFold(5, shuffle=True, random_state=1), scoring="roc_auc")
    gs.fit(X2[tr2][clin], y2[tr2])
    from sklearn.inspection import permutation_importance
    pi = permutation_importance(gs.best_estimator_, X2[~tr2][clin], y2[~tr2], scoring="roc_auc", n_repeats=50, random_state=0)
    imp = pd.Series(pi.importances_mean, index=clin).sort_values(ascending=False)
    imp.to_csv(RESULTS / "plan_induction_importance.csv")
    print("held-out permutation importance (AUROC drop):\n", imp.head(10).round(3).to_string())


if __name__ == "__main__":
    main()
