"""Evaluate the recommender baselines.

    python -m src.run_baselines              # 5-fold CV inside Waves1+2 (default; use this while developing)
    python -m src.run_baselines --holdout    # fit on Waves1+2, score Waves3+4. Run ONCE, after models are frozen.

Per fold, everything fitted (drug z-scaling, baseline means) uses that fold's
training patients only. Each patient appears once in the cohort, so plain KFold
over patients is a patient-grouped split.

Writes results/tables/baselines_{cv,holdout}_{per_patient,summary}.csv.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from .features import alteration_table
from .models import GenomicRuleBaseline, PopulationBaseline, RandomRanker
from .rank_eval import compare, per_patient_metrics
from .targets import DrugZScaler, RecommenderData, load_recommender_data, select_drugs

OUT = Path("results/tables")
MODELS: dict[str, Callable[[], object]] = {
    "Random": lambda: RandomRanker(seed=0),
    "B0 population": PopulationBaseline,
    "B1 genomic rules": GenomicRuleBaseline,
}
RAS = ["alt.NRAS", "alt.KRAS", "alt.PTPN11"]


def _score(models, auc_tr, feats_tr, auc_te, feats_te, drugs) -> dict[str, pd.DataFrame]:
    z_te = DrugZScaler().fit(auc_tr).transform(auc_te)
    out = {}
    for name, make in models.items():
        model = make().fit(auc_tr, feats_tr)
        out[name] = per_patient_metrics(model.predict(feats_te, drugs), z_te, raw=auc_te)
    return out


def cross_validate(data: RecommenderData, feats: pd.DataFrame, drugs: list[str],
                   models=MODELS, n_splits: int = 5, seed: int = 0) -> dict[str, pd.DataFrame]:
    ids = data.patients[data.train]
    res: dict[str, list[pd.DataFrame]] = {n: [] for n in models}
    for k, (a, b) in enumerate(KFold(n_splits, shuffle=True, random_state=seed).split(ids)):
        ptr, pte = ids[a], ids[b]
        scored = _score(models, data.auc.loc[ptr, drugs], feats.loc[ptr],
                        data.auc.loc[pte, drugs], feats.loc[pte], drugs)
        for n, pp in scored.items():
            res[n].append(pp.assign(fold=k))
    return {n: pd.concat(v) for n, v in res.items()}


def holdout(data: RecommenderData, feats: pd.DataFrame, drugs: list[str], models=MODELS) -> dict[str, pd.DataFrame]:
    ptr, pte = data.patients[data.train], data.patients[data.test]
    return _score(models, data.auc.loc[ptr, drugs], feats.loc[ptr],
                  data.auc.loc[pte, drugs], feats.loc[pte], drugs)


def subgroup_table(results: dict[str, pd.DataFrame], feats: pd.DataFrame) -> pd.DataFrame:
    """Mean prec@5 / lift@5 by genomic subgroup, to see where rules help."""
    flags = pd.DataFrame(index=feats.index)
    flags["FLT3-ITD"] = feats["alt.FLT3_ITD"].fillna(0) > 0
    flags["RAS pathway"] = feats[[c for c in RAS if c in feats]].fillna(0).sum(axis=1) > 0
    rule = GenomicRuleBaseline()
    actionable = [f"alt.{a}" for a in rule.rules if f"alt.{a}" in feats]
    flags["no actionable lesion"] = feats[actionable].fillna(0).sum(axis=1) == 0
    rows = []
    for g in flags.columns:
        for name, pp in results.items():
            sub = pp[pp.index.isin(flags.index[flags[g]])]
            rows.append({"subgroup": g, "model": name, "n": len(sub),
                         "prec@5": sub["prec@5"].mean(), "lift@5": sub["lift@5"].mean()})
    return pd.DataFrame(rows).pivot(index=["subgroup", "n"], columns="model", values=["prec@5", "lift@5"]).round(3)


def biology_check(data: RecommenderData, feats: pd.DataFrame, drugs: list[str]) -> pd.DataFrame:
    """Do rule-matched drugs actually look better for carriers? (training cohort only)."""
    ptr = data.patients[data.train]
    z = DrugZScaler().fit_transform(data.auc.loc[ptr, drugs])
    rule = GenomicRuleBaseline().fit(data.auc.loc[ptr, drugs])
    rows = []
    for alt, rdrugs in rule.rule_drugs_.items():
        col = f"alt.{alt}"
        if col not in feats or not rdrugs:
            continue
        carrier = feats.loc[ptr, col].fillna(0) > 0
        if carrier.sum() < 5:
            continue
        zt = z[sorted(rdrugs)]
        rows.append({"alteration": alt, "carriers": int(carrier.sum()), "targeted drugs": len(rdrugs),
                     "mean z, carriers": np.nanmean(zt[carrier].to_numpy()),
                     "mean z, non-carriers": np.nanmean(zt[~carrier].to_numpy())})
    return pd.DataFrame(rows).set_index("alteration").round(3)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--holdout", action="store_true", help="score Waves3+4 (run once, after freezing models)")
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    data = load_recommender_data()
    drugs = select_drugs(data.auc, data.train)
    feats = alteration_table(data.clinical, data.patients)
    n_dna = data.clinical.loc[data.clinical["manuscript_dnaseq"].eq("yes"), "dbgap_subject_id"].isin(data.patients).sum()
    print(f"patients: {len(data.patients)} (Waves1+2 {data.train.sum()}, Waves3+4 {data.test.sum()}) | "
          f"drugs with >= 60 training patients: {len(drugs)} | with DNA-seq (manuscript flag): {n_dna}")

    if args.holdout:
        print("\n*** HOLDOUT: scoring Waves3+4. This should happen once, after all modelling choices are frozen. ***")
        results, tag = holdout(data, feats, drugs), "holdout"
    else:
        results, tag = cross_validate(data, feats, drugs, n_splits=args.folds), "cv"
        print(f"\n{args.folds}-fold cross-validation within Waves1+2")

    print("\nMean per-patient metric [95% patient-bootstrap CI]")
    table = compare(results)
    print(table.to_string())
    print("\nBy genomic subgroup:")
    print(subgroup_table(results, feats).to_string())
    print("\nBiology check (Waves1+2): mean drug-standardised z of rule-matched drugs (negative = more sensitive)")
    print(biology_check(data, feats, drugs).to_string())

    OUT.mkdir(parents=True, exist_ok=True)
    pd.concat({n: pp for n, pp in results.items()}, names=["model"]).to_csv(OUT / f"baselines_{tag}_per_patient.csv")
    table.to_csv(OUT / f"baselines_{tag}_summary.csv")
    print(f"\nWrote {OUT}/baselines_{tag}_per_patient.csv and _summary.csv")


if __name__ == "__main__":
    main()
