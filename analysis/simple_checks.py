"""Do simpler methods give the same answers as the multi-view clustering and the shrinkage screen?

    python -m analysis.simple_checks [clusters|flt3|all]

clusters: k-means (k=3) on the six cell-type scores vs the multiple-kernel clusters
  (agreement, drugs that differ, survival beyond age + ELN).
flt3: the strongest replicated interaction as a plain t-test. Among FLT3-ITD patients,
  are monocytic-high tumors (above the median monocyte-like score) less sensitive to
  FLT3 / multi-kinase inhibitors? Beat AML and FIMM.

Outputs: results/tables/simple_clusters.csv, simple_flt3.csv
"""
from __future__ import annotations

import xgboost  # noqa: F401  (import before sklearn: avoids an OpenMP clash)
import sys

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

from analysis.common import RESULTS, load_frame

FLT3_DRUGS = ["Sorafenib", "Sunitinib", "Quizartinib (AC220)", "Gilteritinib", "Midostaurin", "Crenolanib",
              "Cabozantinib"]


def bh(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    q = np.minimum.accumulate((p[o] * len(p) / (np.arange(len(p)) + 1))[::-1])[::-1]
    out = np.empty_like(q)
    out[o] = np.minimum(q, 1)
    return out


def describe(lab, F, meta):
    """Drugs differing across clusters (Kruskal-Wallis, BH) and survival LR test beyond age + ELN."""
    auc, k = F["auc"], len(set(lab))
    p = [stats.kruskal(*[auc.loc[lab == c, d].dropna() for c in range(k)]).pvalue for d in auc.columns
         if min(auc.loc[lab == c, d].notna().sum() for c in range(k)) >= 10]
    T = pd.to_numeric(meta.overallSurvival, errors="coerce") / 365.25
    base = pd.DataFrame({"T": T, "E": meta.vitalStatus.map({"Dead": 1, "Alive": 0}),
                         "age": meta.ageAtDiagnosis, "adv": meta.ELN2017.eq("Adverse").astype(int),
                         "fav": meta.ELN2017.eq("Favorable").astype(int)}, index=meta.index)
    ok = base["T"].gt(0) & base.E.notna() & meta.ELN2017.notna() & base.age.notna()
    f0 = CoxPHFitter().fit(base[ok], "T", "E")
    d = base[ok].join(pd.get_dummies(pd.Series(lab, index=meta.index), prefix="c", drop_first=True).astype(int))
    lr = 2 * (CoxPHFitter().fit(d, "T", "E").log_likelihood_ - f0.log_likelihood_)
    return {"drugs_differ_q05": int((bh(p) < 0.05).sum()), "drugs_tested": len(p),
            "survival_p_beyond_age_ELN": stats.chi2.sf(lr, k - 1), "sizes": np.bincount(lab).tolist()}


def clusters():
    F = load_frame()
    meta = F["meta"]
    mk = pd.read_csv(RESULTS / "cluster_labels.csv", index_col=0).cluster.reindex(meta.index).to_numpy()
    S = F["states"].loc[meta.index]
    km = KMeans(3, n_init=50, random_state=0).fit_predict(((S - S.mean()) / S.std()).to_numpy())
    rows = [{"method": "multiple-kernel k-means (3 views)", **describe(mk, F, meta)},
            {"method": "k-means on 6 cell-type scores", **describe(km, F, meta), "ARI_vs_multiview": adjusted_rand_score(mk, km)}]
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "simple_clusters.csv", index=False)
    print(out.to_string(index=False))
    prof = S.groupby(km).mean().round(2)
    prof["n"] = np.bincount(km)
    print(prof.to_string())


def flt3():
    from analysis.fimm import fimm_expression, fimm_samples, fimm_targets, shared_scores, FIMM
    F = load_frame()
    z = (F["auc"] - F["auc"].mean()) / F["auc"].std()
    itd = F["muts"]["FLT3_ITD"].eq(1)
    mono = F["states"]["Monocyte-like"]
    f = fimm_samples()
    _, sc_f = shared_scores(fimm_expression(f["rna"]))
    zf = fimm_targets(f, FLT3_DRUGS)
    pid = f["ann"].Patient_ID.str.replace("FPM_", "", regex=False).str.strip()
    itd_f = pd.Series(f["clin"].loc[pid, "FLT3_ITD"].to_numpy(), index=sc_f.index).eq(1)
    mono_f = sc_f["ct.Monocyte-like"]
    rows = []
    for cohort, Z, I, M in [("Beat AML", z, itd, mono), ("FIMM", zf, itd_f, mono_f)]:
        hi = M > M[I].median()                     # split within FLT3-ITD patients
        for d in FLT3_DRUGS:
            if d not in Z:
                continue
            a, b = Z.loc[I & hi, d].dropna(), Z.loc[I & ~hi, d].dropna()
            if min(len(a), len(b)) < 4:
                continue
            t = stats.ttest_ind(a, b, equal_var=False)
            rows.append({"cohort": cohort, "drug": d, "n_mono_high": len(a), "n_mono_low": len(b),
                         "z_mono_high": a.mean(), "z_mono_low": b.mean(), "diff": a.mean() - b.mean(), "p": t.pvalue})
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "simple_flt3.csv", index=False)
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    step = sys.argv[1] if len(sys.argv) > 1 else "all"
    if step in ("clusters", "all"):
        clusters()
    if step in ("flt3", "all"):
        flt3()
