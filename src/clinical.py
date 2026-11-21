"""Clinical validation: does (observed or predicted) ex vivo cytarabine response
relate to induction outcome and survival?

    python -m src.clinical          # needs models_{cv,holdout}_{predictions,observed_z}.csv from src.run_models

Cohort: modeling patients whose specimen was taken at initial diagnosis, who
got standard-chemotherapy induction, with response CR/CRi vs refractory.
Predicted z is out-of-sample for every patient: out-of-fold CV predictions for
Waves1+2, holdout predictions for Waves3+4. Outcome labels are never used in
training, so the endpoint is independent of the model (the patients are not).

Writes results/tables/clinical_{logistic,auroc,cox}.csv
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .run_baselines import OUT
from .targets import load_recommender_data

DRUG = "Cytarabine"
MODEL = "M1 ridge: all"


def _load_z(kind: str) -> pd.Series:
    parts = []
    for tag in ("cv", "holdout"):
        if kind == "obs":
            z = pd.read_csv(OUT / f"models_{tag}_observed_z.csv", index_col=0)
        else:
            p = pd.read_csv(OUT / f"models_{tag}_predictions.csv", index_col=[0, 1])
            z = p.xs(MODEL, level="model")
        parts.append(z[DRUG])
    return pd.concat(parts)


def build_cohort(intensive_only: bool = False) -> pd.DataFrame:
    data = load_recommender_data()
    m = data.meta
    feats = data.features()
    resp = m["responseToInductionTx"].replace({"Complete Response i": "Complete Response"})
    keep = (m["diseaseStageAtSpecimenCollection"].eq("Initial Diagnosis")
            & m["typeInductionTx"].eq("Standard Chemotherapy")
            & resp.isin(["Complete Response", "Refractory"]))
    if intensive_only:
        keep &= m["cumulativeTreatmentRegimens"].fillna("").str.contains(r"7\+3|HiDAC|Cytarabine, Idarubicin")
    c = pd.DataFrame({
        "refractory": resp.eq("Refractory").astype(int),
        "age": m["ageAtDiagnosis"].astype(float),
        "eln": m["ELN2017"].where(m["ELN2017"].isin(["Favorable", "Intermediate", "Adverse"])),
        "cohort": m["cohort"],
        "os_days": pd.to_numeric(m["overallSurvival"], errors="coerce"),
        "dead": m["vitalStatus"].map({"Dead": 1, "Alive": 0}),
    })[keep]
    c["obs_z"] = _load_z("obs").reindex(c.index)
    c["pred_z"] = _load_z("pred").reindex(c.index)
    c = c.join(feats[["rna.PEAR1", "rna.LSC17"]])
    c["eln_int"] = c["eln"].eq("Intermediate").astype(int)
    c["eln_adv"] = c["eln"].eq("Adverse").astype(int)
    return c


def _logit(c: pd.DataFrame, cols: list[str]):
    X = sm.add_constant(c[cols].astype(float))
    return sm.Logit(c["refractory"], X).fit(disp=0)


def _cv_auroc(c: pd.DataFrame, cols: list[str], seed: int = 0) -> tuple[float, float]:
    """Repeated 5-fold CV AUROC (mean, sd over repeats)."""
    X, y = c[cols].to_numpy(float), c["refractory"].to_numpy()
    rs = []
    for r in range(20):
        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=seed + r)
        p = cross_val_predict(make_pipeline(StandardScaler(), LogisticRegression(C=1e4)), X, y,
                              cv=list(cv.split(X, y)), method="predict_proba")[:, 1]
        rs.append(roc_auc_score(y, p))
    return float(np.mean(rs)), float(np.std(rs))


def analyse(c: pd.DataFrame, label: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = ["age", "eln_int", "eln_adv"]
    specs = {
        "ELN + age": base,
        "ELN + age + observed cytarabine z": base + ["obs_z"],
        "ELN + age + predicted cytarabine z": base + ["pred_z"],
        "ELN + age + PEAR1 + LSC17": base + ["rna.PEAR1", "rna.LSC17"],
        "ELN + age + predicted z + PEAR1 + LSC17": base + ["pred_z", "rna.PEAR1", "rna.LSC17"],
    }
    rows, aucs = [], []
    for name, cols in specs.items():
        d = c.dropna(subset=cols + ["refractory"])
        fit = _logit(d, cols)
        ref = _logit(d, base)
        lr_p = stats.chi2.sf(2 * (fit.llf - ref.llf), len(cols) - len(base)) if len(cols) > len(base) else np.nan
        m, s = _cv_auroc(d, cols)
        aucs.append({"cohort": label, "model": name, "n": len(d), "events (refractory)": int(d["refractory"].sum()),
                     "cv AUROC": m, "sd": s, "LR test p vs ELN+age": lr_p})
        for v in cols:
            if v in base:
                continue
            rows.append({"cohort": label, "model": name, "term": v, "n": len(d),
                         "OR per SD": np.exp(fit.params[v] * d[v].std()),
                         "OR lo": np.exp(fit.conf_int().loc[v, 0] * d[v].std()),
                         "OR hi": np.exp(fit.conf_int().loc[v, 1] * d[v].std()),
                         "p": fit.pvalues[v]})
    return pd.DataFrame(rows), pd.DataFrame(aucs)


def univariate(c: pd.DataFrame, label: str) -> pd.DataFrame:
    rows = []
    for v in ["obs_z", "pred_z"]:
        d = c.dropna(subset=[v])
        rows.append({"cohort": label, "score": v, "n": len(d),
                     "AUROC (higher z -> refractory)": roc_auc_score(d["refractory"], d[v]),
                     "Mann-Whitney p": stats.mannwhitneyu(d.loc[d.refractory == 1, v], d.loc[d.refractory == 0, v]).pvalue})
    return pd.DataFrame(rows)


def cox(c: pd.DataFrame) -> pd.DataFrame:
    from lifelines import CoxPHFitter
    d = c.dropna(subset=["os_days", "dead", "pred_z", "age", "eln"])
    d = d[d["os_days"] > 0]
    out = []
    for name, cols in {"pred_z alone": ["pred_z"], "pred_z + ELN + age": ["pred_z", "age", "eln_int", "eln_adv"],
                       "obs_z + ELN + age": ["obs_z", "age", "eln_int", "eln_adv"]}.items():
        dd = d.dropna(subset=cols)[cols + ["os_days", "dead"]].copy()
        for v in ("pred_z", "obs_z"):
            if v in dd:
                dd[v] = (dd[v] - dd[v].mean()) / dd[v].std()
        f = CoxPHFitter().fit(dd, "os_days", "dead")
        v = cols[0]
        out.append({"model": name, "n": len(dd), "deaths": int(dd["dead"].sum()), "term": v,
                    "HR per SD": f.hazard_ratios_[v], "lo": np.exp(f.confidence_intervals_.loc[v].iloc[0]),
                    "hi": np.exp(f.confidence_intervals_.loc[v].iloc[1]), "p": f.summary.loc[v, "p"],
                    "C-index": f.concordance_index_})
    return pd.DataFrame(out)


def main() -> None:
    pd.set_option("display.width", 250)
    res_or, res_auc, res_uni = [], [], []
    for label, intensive in [("standard-chemo induction", False), ("7+3 / HiDAC confirmed", True)]:
        c = build_cohort(intensive)
        print(f"\n=== {label}: n = {len(c)} ({c.refractory.sum()} refractory), "
              f"Waves1+2 {c.cohort.eq('Waves1+2').sum()} / Waves3+4 {c.cohort.eq('Waves3+4').sum()}; "
              f"with cytarabine tested: {c.obs_z.notna().sum()}")
        o, a = analyse(c, label)
        res_or.append(o); res_auc.append(a); res_uni.append(univariate(c, label))
        if not intensive:
            cx = cox(c)
    OR, AUC, UNI = pd.concat(res_or), pd.concat(res_auc), pd.concat(res_uni)
    print("\nUnivariate: does the score separate refractory from CR?")
    print(UNI.round(3).to_string(index=False))
    print("\nLogistic regression, refractory ~ ... (OR per SD of the added term)")
    print(OR.round(3).to_string(index=False))
    print("\nDiscrimination (20x repeated 5-fold CV AUROC)")
    print(AUC.round(3).to_string(index=False))
    print("\nCox overall survival (standard-chemo cohort)")
    print(cx.round(3).to_string(index=False))
    OR.to_csv(OUT / "clinical_logistic.csv", index=False)
    AUC.to_csv(OUT / "clinical_auroc.csv", index=False)
    UNI.to_csv(OUT / "clinical_univariate.csv", index=False)
    cx.to_csv(OUT / "clinical_cox.csv", index=False)


if __name__ == "__main__":
    main()
