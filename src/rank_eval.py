"""Per-patient ranking metrics for drug recommendation.

A model outputs a SCORE per (patient, drug); LOWER = recommend first
(predicted more sensitive). Each patient is scored only over the drugs that were
actually tested on their cells, then metrics are averaged over patients with a
patient-level bootstrap CI.

Metrics (per patient):
    spearman      rank agreement of score with observed z (higher = better)
    spearman_raw  same against raw AUC: shows how much a ranking only tracks drug potency
    prec@5        fraction of the top-5 recommended drugs that are truly sensitive (z < -0.84)
    prevalence    fraction of that patient's tested drugs that are truly sensitive
                  = the expected prec@5 of a random ranking
    lift@5        prec@5 - prevalence (0 = no better than random)
    ndcg@10       graded: gain = max(0, -z), so strongly sensitive drugs count more
    regret        z of the #1 recommended drug minus the patient's best z (0 = perfect, higher = worse)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .targets import SENSITIVE_Z

METRICS = ["spearman", "spearman_raw", "prec@5", "prevalence", "lift@5", "ndcg@10", "regret"]


def _ndcg(order_gain: np.ndarray, ideal_gain: np.ndarray, k: int) -> float:
    disc = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = (order_gain[:k] * disc[: len(order_gain[:k])]).sum()
    idcg = (ideal_gain[:k] * disc[: len(ideal_gain[:k])]).sum()
    return float(dcg / idcg) if idcg > 0 else np.nan


def per_patient_metrics(
    score: pd.DataFrame,
    z: pd.DataFrame,
    raw: pd.DataFrame | None = None,
    k_prec: int = 5,
    k_ndcg: int = 10,
    threshold: float = SENSITIVE_Z,
    min_drugs: int = 10,
) -> pd.DataFrame:
    """One row of metrics per patient. Ties in `score` are broken by column order."""
    drugs = score.columns.intersection(z.columns)
    patients = score.index.intersection(z.index)
    rows = []
    for p in patients:
        s = score.loc[p, drugs]
        zz = z.loc[p, drugs]
        ok = s.notna() & zz.notna()
        if ok.sum() < min_drugs:
            continue
        s, zz = s[ok].to_numpy(float), zz[ok].to_numpy(float)
        order = np.argsort(s, kind="stable")
        sens = zz < threshold
        gain = np.clip(-zz, 0, None)
        row = {
            "patient": p,
            "n_drugs": int(ok.sum()),
            "spearman": stats.spearmanr(s, zz).statistic if np.ptp(s) > 0 else np.nan,
            "prec@5": sens[order[:k_prec]].mean(),
            "prevalence": sens.mean(),
            "ndcg@10": _ndcg(gain[order], np.sort(gain)[::-1], k_ndcg),
            "regret": zz[order[0]] - zz.min(),
        }
        row["lift@5"] = row["prec@5"] - row["prevalence"]
        if raw is not None:
            r = raw.loc[p, drugs][ok].to_numpy(float)
            row["spearman_raw"] = stats.spearmanr(s, r).statistic if np.ptp(s) > 0 else np.nan
        rows.append(row)
    return pd.DataFrame(rows).set_index("patient")


def summarise(per_patient: pd.DataFrame, n_boot: int = 1000, seed: int = 0) -> pd.DataFrame:
    """Mean of each metric with a 95% patient-bootstrap CI."""
    rng = np.random.default_rng(seed)
    cols = [c for c in METRICS if c in per_patient.columns]
    vals = per_patient[cols].to_numpy(float)
    n = len(vals)
    boots = np.array([np.nanmean(vals[rng.integers(0, n, n)], axis=0) for _ in range(n_boot)])
    lo, hi = np.nanpercentile(boots, [2.5, 97.5], axis=0)
    return pd.DataFrame({"mean": np.nanmean(vals, axis=0), "ci_lo": lo, "ci_hi": hi}, index=cols)


def compare(results: dict[str, pd.DataFrame], n_boot: int = 1000, seed: int = 0) -> pd.DataFrame:
    """Side-by-side 'mean [lo, hi]' table for several models' per-patient metrics."""
    out = {}
    for name, pp in results.items():
        s = summarise(pp, n_boot=n_boot, seed=seed)
        out[name] = [f"{m:.3f} [{l:.3f}, {h:.3f}]" for m, l, h in s.to_numpy()]
        idx = s.index
    table = pd.DataFrame(out, index=idx)
    table.loc["n_patients"] = [str(len(pp)) for pp in results.values()]
    return table
