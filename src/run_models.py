"""Evaluate the learned recommenders (M1 ridge, M2 xgboost) against the baselines.

    python -m src.run_models              # 5-fold CV inside Waves1+2 (development)
    python -m src.run_models --holdout    # fit Waves1+2, score Waves3+4. Run ONCE, after freezing.

Same folds, metrics and leakage rules as src.run_baselines. Writes
results/tables/models_{cv,holdout}_{per_patient,summary,subgroups,per_drug}.csv
and models_{cv,holdout}_predictions.csv (predicted z, used by src.clinical).
"""

from __future__ import annotations

import argparse
from collections.abc import Callable

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import KFold

from .features import alteration_table
from .models import (GenomicRuleBaseline, PerDrugRidge, PooledXGB, PopulationBaseline,
                     RandomRanker)
from .rank_eval import compare, per_patient_metrics
from .run_baselines import OUT, subgroup_table
from .targets import DrugZScaler, RecommenderData, load_recommender_data, select_drugs

MUT = ("mut", "fus", "clin")
MODELS: dict[str, Callable[[], object]] = {
    "Random": lambda: RandomRanker(seed=0),
    "B0 population": PopulationBaseline,
    "B1 genomic rules": GenomicRuleBaseline,
    "M1 ridge: mutations": lambda: PerDrugRidge(MUT),
    "M1 ridge: +cell types": lambda: PerDrugRidge(MUT + ("ct",)),
    "M1 ridge: all": lambda: PerDrugRidge(),
    "M1 ridge: expression only": lambda: PerDrugRidge(("ct", "me", "rna")),
    "M2 xgb: mutations": lambda: PooledXGB(MUT),
    "M2 xgb: all": lambda: PooledXGB(),
    "B1 -> M1 all": lambda: GenomicRuleBaseline(fallback=PerDrugRidge()),
    "B1 -> M2 all": lambda: GenomicRuleBaseline(fallback=PooledXGB()),
    "M1 shuffled labels": lambda: PerDrugRidge(shuffle_seed=0),
}
KEY_DRUGS = ["Venetoclax", "Cytarabine", "Azacytidine", "Gilteritinib", "Quizartinib (AC220)",
             "Sorafenib", "Midostaurin", "Crenolanib", "Trametinib (GSK1120212)", "Selumetinib (AZD6244)"]


def _fold(models, auc_tr, feats_tr, auc_te, feats_te, drugs):
    z_te = DrugZScaler().fit(auc_tr).transform(auc_te)
    pp, preds = {}, {}
    for name, make in models.items():
        m = make().fit(auc_tr, feats_tr)
        s = m.predict(feats_te, drugs)
        pp[name] = per_patient_metrics(s, z_te, raw=auc_te)
        preds[name] = s
    return pp, preds, z_te


def run(data: RecommenderData, feats: pd.DataFrame, drugs: list[str], models=MODELS,
        holdout: bool = False, n_splits: int = 5, seed: int = 0):
    """Returns per-patient metrics, out-of-sample predicted z, and observed z (per model)."""
    if holdout:
        splits = [(data.patients[data.train], data.patients[data.test])]
    else:
        ids = data.patients[data.train]
        splits = [(ids[a], ids[b]) for a, b in KFold(n_splits, shuffle=True, random_state=seed).split(ids)]
    res = {n: [] for n in models}
    pred = {n: [] for n in models}
    zs = []
    for k, (ptr, pte) in enumerate(splits):
        print(f"  fold {k + 1}/{len(splits)}", flush=True)
        pp, pr, z = _fold(models, data.auc.loc[ptr, drugs], feats.loc[ptr],
                          data.auc.loc[pte, drugs], feats.loc[pte], drugs)
        for n in models:
            res[n].append(pp[n].assign(fold=k))
            pred[n].append(pr[n])
        zs.append(z)
    return ({n: pd.concat(v) for n, v in res.items()},
            {n: pd.concat(v) for n, v in pred.items()},
            pd.concat(zs))


def per_drug_table(pred: dict[str, pd.DataFrame], z: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    """Across-patient Spearman of predicted vs observed z, per drug (pooled over folds)."""
    rows = []
    for d in z.columns:
        obs = z[d]
        row = {"drug": d, "n": int(obs.notna().sum())}
        for n in models:
            p = pred[n][d].reindex(obs.index)
            ok = obs.notna() & p.notna()
            row[n] = stats.spearmanr(p[ok], obs[ok]).statistic if ok.sum() >= 20 and p[ok].std() > 0 else np.nan
        rows.append(row)
    return pd.DataFrame(rows).set_index("drug")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--holdout", action="store_true")
    args = ap.parse_args()

    data = load_recommender_data()
    drugs = select_drugs(data.auc, data.train)
    feats = data.features().join(alteration_table(data.clinical, data.patients))
    tag = "holdout" if args.holdout else "cv"
    print(f"{tag}: {len(drugs)} drugs, {data.train.sum()} train / {data.test.sum()} test patients")
    if args.holdout:
        print("*** HOLDOUT: scoring Waves3+4 once, models frozen. ***")

    res, pred, z = run(data, feats, drugs, holdout=args.holdout)
    table = compare(res)
    print("\nMean per-patient metric [95% patient-bootstrap CI]")
    print(table.to_string())
    sub = subgroup_table(res, feats)
    print("\nBy subgroup (prec@5):")
    print(sub["prec@5"].to_string())

    # paired comparison vs B1 (same patients, same folds)
    paired = []
    for n in res:
        d = (res[n]["lift@5"] - res["B1 genomic rules"]["lift@5"]).dropna().to_numpy()
        rng = np.random.default_rng(0)
        b = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
        paired.append({"model": n, "Δprec@5 vs B1": d.mean(), "lo": np.percentile(b, 2.5), "hi": np.percentile(b, 97.5)})
    paired = pd.DataFrame(paired).set_index("model").round(3)
    print("\nPaired difference in prec@5 vs B1 [95% bootstrap CI]")
    print(paired.to_string())

    learned = ["M1 ridge: mutations", "M1 ridge: all", "M2 xgb: mutations", "M2 xgb: all", "M1 shuffled labels"]
    pdt = per_drug_table(pred, z, learned)
    print("\nPer-drug Spearman (median over drugs):")
    print(pdt[learned].median().round(3).to_string())
    print("\nKey drugs:")
    print(pdt.reindex([d for d in KEY_DRUGS if d in pdt.index]).round(3).to_string())

    OUT.mkdir(parents=True, exist_ok=True)
    pd.concat(res, names=["model"]).to_csv(OUT / f"models_{tag}_per_patient.csv")
    table.to_csv(OUT / f"models_{tag}_summary.csv")
    sub.to_csv(OUT / f"models_{tag}_subgroups.csv")
    paired.to_csv(OUT / f"models_{tag}_paired_vs_B1.csv")
    pdt.to_csv(OUT / f"models_{tag}_per_drug.csv")
    keep = ["M1 ridge: all", "M2 xgb: all", "M1 ridge: mutations"]
    pd.concat({n: pred[n] for n in keep}, names=["model"]).to_csv(OUT / f"models_{tag}_predictions.csv")
    z.to_csv(OUT / f"models_{tag}_observed_z.csv")
    print(f"\nWrote {OUT}/models_{tag}_*.csv")


if __name__ == "__main__":
    main()
