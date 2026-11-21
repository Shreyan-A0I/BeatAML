"""Plan section 5: whom does ELN misclassify? Residual-risk survival model.

    python -m analysis.plan_eln

Train on Beat AML Waves1+2, score once on Waves3+4 and on TCGA-LAML.
  base      Cox: age + ELN2017 (TCGA: age + cytogenetic risk, refit there)
  residual  ridge Cox with the base linear predictor as a fixed OFFSET, features =
            frozen-weight cell-type scores, module eigengenes, PEAR1 (lambda by CV)
  LSC17     single score, coefficient fit with the same offset (the benchmark)
  RSF       random survival forest on age + ELN + all scores (non-linear comparison)
Outputs results/tables/plan_eln_*.csv
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from scipy.optimize import minimize
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import cumulative_dynamic_auc
from sksurv.util import Surv

from analysis.common import RESULTS
from analysis.plan_common import scores

warnings.filterwarnings("ignore")
rng = np.random.default_rng(0)


def cox_offset(X, T, E, off, lam):
    """Ridge Cox partial likelihood (Breslow) with offset; returns beta."""
    o = np.argsort(-T)
    X, T, E, off = X[o], T[o], E[o], off[o]

    def nll(b):
        eta = off + X @ b
        m = eta.max()
        w = np.exp(eta - m)
        cw = np.cumsum(w)
        cx = np.cumsum(w[:, None] * X, 0)
        ll = np.sum(E * (eta - m - np.log(cw)))
        g = (E[:, None] * (X - cx / cw[:, None])).sum(0)
        return -ll + lam * b @ b, -g + 2 * lam * b

    return minimize(nll, np.zeros(X.shape[1]), jac=True, method="L-BFGS-B").x


def cindex(T, E, risk):
    return concordance_index(T, -risk, E)


def survival_frame(S):
    c, sc = S["beat_clin"], S["beat"]
    T = pd.to_numeric(c.overallSurvival, errors="coerce") / 365.25
    E = c.vitalStatus.map({"Dead": 1, "Alive": 0})
    df = pd.DataFrame({"T": T, "E": E, "age": (c.ageAtDiagnosis - 60) / 15,
                       "adv": c.ELN2017.eq("Adverse").astype(int), "fav": c.ELN2017.eq("Favorable").astype(int),
                       "ELN": c.ELN2017, "train": c.cohort.eq("Waves1+2"), "sex": c.consensus_sex,
                       "race": c.reportedRace.where(c.reportedRace.isin(["White"]), "Non-white/unknown"),
                       "denovo": c.isDenovo.astype(str).eq("TRUE")}).join(sc)
    return df[df["T"].gt(0) & df.E.notna() & df.ELN.notna() & df.age.notna()]


def nri(T, E, cat_old, cat_new, horizon=2.0):
    ev = (T <= horizon) & (E == 1)
    ne = T > horizon
    up, down = cat_new > cat_old, cat_new < cat_old
    nri_e = up[ev].mean() - down[ev].mean()
    nri_n = down[ne].mean() - up[ne].mean()
    return nri_e, nri_n, nri_e + nri_n, int(ev.sum()), int(ne.sum())


def main():
    S = scores()
    D = survival_frame(S)
    feats = [c for c in S["beat"].columns if c.startswith(("ct.", "me.")) or c == "PEAR1"]
    tr, te = D[D.train], D[~D.train]
    print(f"train {len(tr)} ({int(tr.E.sum())} deaths), test {len(te)} ({int(te.E.sum())} deaths)")

    base = CoxPHFitter().fit(tr[["T", "E", "age", "adv", "fav"]], "T", "E")
    lp = lambda d: d[["age", "adv", "fav"]].to_numpy() @ base.params_.to_numpy()

    # choose lambda by 5-fold CV C-index of offset + residual within train
    folds = rng.integers(0, 5, len(tr))
    Xtr = tr[feats].to_numpy()
    best = None
    for lam in [0.3, 1, 3, 10, 30, 100]:
        cs = []
        for f in range(5):
            a, b = folds != f, folds == f
            beta = cox_offset(Xtr[a], tr["T"].to_numpy()[a], tr.E.to_numpy()[a], lp(tr)[a], lam)
            r = lp(tr)[b] + Xtr[b] @ beta
            cs.append(cindex(tr["T"].to_numpy()[b], tr.E.to_numpy()[b], r))
        print(f"  lambda {lam}: CV C {np.mean(cs):.3f}")
        if best is None or np.mean(cs) > best[1]:
            best = (lam, np.mean(cs))
    lam = best[0]
    beta = cox_offset(Xtr, tr["T"].to_numpy(), tr.E.to_numpy(), lp(tr), lam)
    coef = pd.Series(beta, index=feats).sort_values()
    coef.to_csv(RESULTS / "plan_eln_residual_coefs.csv")
    print("residual coefficients:\n", coef.round(3).to_string())
    b_lsc = cox_offset(tr[["LSC17"]].to_numpy(), tr["T"].to_numpy(), tr.E.to_numpy(), lp(tr), 0.0)
    b_pear = cox_offset(tr[["PEAR1"]].to_numpy(), tr["T"].to_numpy(), tr.E.to_numpy(), lp(tr), 0.0)

    rsf = RandomSurvivalForest(n_estimators=500, min_samples_leaf=15, max_features="sqrt", random_state=0, n_jobs=-1)
    allf = ["age", "adv", "fav"] + feats + ["LSC17"]
    rsf.fit(tr[allf].to_numpy(), Surv.from_arrays(tr.E.astype(bool), tr["T"]))

    def risks(d):
        return {"ELN + age": lp(d),
                "+ LSC17": lp(d) + d[["LSC17"]].to_numpy() @ b_lsc,
                "+ PEAR1": lp(d) + d[["PEAR1"]].to_numpy() @ b_pear,
                "+ residual (all scores)": lp(d) + d[feats].to_numpy() @ beta,
                "RSF (age, ELN, all scores)": rsf.predict(d[allf].to_numpy())}

    rows = []
    R = risks(te)
    y_tr = Surv.from_arrays(tr.E.astype(bool), tr["T"])
    y_te = Surv.from_arrays(te.E.astype(bool), te["T"])
    boot = [rng.integers(0, len(te), len(te)) for _ in range(500)]
    for name, r in R.items():
        c = cindex(te["T"].to_numpy(), te.E.to_numpy(), r)
        dc = [cindex(te["T"].to_numpy()[i], te.E.to_numpy()[i], r[i]) -
              cindex(te["T"].to_numpy()[i], te.E.to_numpy()[i], R["ELN + age"][i]) for i in boot]
        auc, _ = cumulative_dynamic_auc(y_tr, y_te, r, [1.0, 2.0])
        rows.append({"set": "Beat AML Waves3+4", "model": name, "n": len(te), "events": int(te.E.sum()), "C": c,
                     "dC_vs_ELN": c - cindex(te["T"].to_numpy(), te.E.to_numpy(), R["ELN + age"]),
                     "dC_lo": np.percentile(dc, 2.5), "dC_hi": np.percentile(dc, 97.5),
                     "tdAUC_1y": auc[0], "tdAUC_2y": auc[1]})

    # NRI at 2 years: new categories = residual-model risk cut at train ELN group proportions
    props = tr.ELN.value_counts(normalize=True)
    q = np.cumsum([props["Favorable"], props["Intermediate"]])
    risk_tr = lp(tr) + Xtr @ beta
    cuts = np.quantile(risk_tr, q)
    old = te.ELN.map({"Favorable": 0, "Intermediate": 1, "Adverse": 2}).to_numpy()
    new = np.digitize(R["+ residual (all scores)"], cuts)
    e, n_, tot, ne, nn = nri(te["T"].to_numpy(), te.E.to_numpy(), old, new)
    print(f"\nNRI (2 y, Waves3+4): events {e:+.3f}, non-events {n_:+.3f}, total {tot:+.3f} ({ne} events, {nn} survivors)")

    # Who does ELN miss: ELN favorable/intermediate, residual score top tertile
    resid = te[feats].to_numpy() @ beta
    g = te.assign(resid=resid)
    lowELN = g[g.ELN.isin(["Favorable", "Intermediate"])]
    lowELN = lowELN.assign(high=(lowELN.resid > np.quantile(Xtr @ beta, 2 / 3)).astype(int))
    f = CoxPHFitter().fit(lowELN[["T", "E", "high", "age"]], "T", "E")
    s = f.summary.loc["high"]
    print(f"ELN fav/int in Waves3+4 (n={len(lowELN)}, {int(lowELN.E.sum())} deaths): top-tertile residual score "
          f"HR {s['exp(coef)']:.2f} [{s['exp(coef) lower 95%']:.2f}, {s['exp(coef) upper 95%']:.2f}], p={s['p']:.3g}; "
          f"{lowELN.high.mean():.0%} flagged")
    rows.append({"set": "Waves3+4 ELN fav/int", "model": "top-tertile residual (HR, adj age)", "n": len(lowELN),
                 "events": int(lowELN.E.sum()), "C": np.nan, "HR": s["exp(coef)"], "HR_lo": s["exp(coef) lower 95%"],
                 "HR_hi": s["exp(coef) upper 95%"], "p": s["p"]})

    # TCGA: base refit there (age + cytogenetic risk); frozen residual score / LSC17 / PEAR1 added
    tc = S["tcga_clin"].join(S["tcga"])
    tc["age"] = (tc.age - 60) / 15
    tc["resid"] = tc[feats].to_numpy() @ beta
    tc["resid"] = (tc.resid - tc.resid.mean()) / tc.resid.std()
    b0 = CoxPHFitter().fit(tc[["T", "E", "age", "adv", "fav"]], "T", "E")
    c0 = b0.concordance_index_
    for sc in ["resid", "LSC17", "PEAR1"]:
        f = CoxPHFitter().fit(tc[["T", "E", "age", "adv", "fav", sc]], "T", "E")
        s = f.summary.loc[sc]
        lr = 2 * (f.log_likelihood_ - b0.log_likelihood_)
        rows.append({"set": "TCGA-LAML", "model": f"age + cyto risk + {sc}", "n": len(tc), "events": int(tc.E.sum()),
                     "C": f.concordance_index_, "dC_vs_ELN": f.concordance_index_ - c0, "HR": s["exp(coef)"],
                     "HR_lo": s["exp(coef) lower 95%"], "HR_hi": s["exp(coef) upper 95%"], "p": s["p"], "LR_chi2": lr})
    rows.append({"set": "TCGA-LAML", "model": "age + cyto risk", "n": len(tc), "events": int(tc.E.sum()), "C": c0})

    # fairness: C-index of the residual model by subgroup, Waves3+4
    fair = []
    r = R["+ residual (all scores)"]
    for col, groups in [("sex", ["Male", "Female"]), ("race", ["White", "Non-white/unknown"]),
                        ("ageband", ["<45", "45-60", ">=60"])]:
        lab = te[col] if col != "ageband" else pd.cut(te.age * 15 + 60, [0, 45, 60, 200], right=False,
                                                       labels=["<45", "45-60", ">=60"]).astype(str)
        for g_ in groups:
            m = (lab == g_).to_numpy()
            fair.append({"group": f"{col}={g_}", "n": int(m.sum()), "events": int(te.E.to_numpy()[m].sum()),
                         "C_ELN": cindex(te["T"].to_numpy()[m], te.E.to_numpy()[m], R["ELN + age"][m]),
                         "C_residual": cindex(te["T"].to_numpy()[m], te.E.to_numpy()[m], r[m])})
    fair = pd.DataFrame(fair)
    fair.to_csv(RESULTS / "plan_eln_fairness.csv", index=False)

    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "plan_eln_results.csv", index=False)
    with pd.option_context("display.width", 220, "display.max_columns", 20):
        print(out.round(3).to_string(index=False))
        print(fair.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
