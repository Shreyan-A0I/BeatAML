"""Partial screen: how much does screening a few anchor drugs add to Vania's RNA-based recommender?

    python -m analysis.partial_screen

Anchors are picked on Waves1+2 only (pivoted QR on the drug-standardised AUC of well-covered
drugs = the drugs that best span the panel). For each other drug, a ridge model uses Vania's
44 features plus the patient's observed z on the anchors. Scored once on Waves3+4, on the
non-anchor drugs only, so every arm ranks the same drugs.

Output: results/tables/partial_screen.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.linalg import qr
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

from analysis.common import ROOT

ALPHAS = np.logspace(-1, 4, 21)
KS = [0, 3, 5, 10, 20]


def pick_anchors(z_tr: pd.DataFrame, k: int, min_cover: float = 0.9) -> list[str]:
    """Greedy span of the panel: pivoted QR on well-covered drugs (missing -> 0, the drug mean)."""
    cand = z_tr.columns[z_tr.notna().mean() >= min_cover]
    _, _, piv = qr(z_tr[cand].fillna(0).to_numpy(), pivoting=True, mode="economic")
    return list(cand[piv[:k]])


def fit_predict(F_tr, F_te, z_tr, z_te, anchors, targets, use_features=True):
    """Per target drug: ridge on [features, anchor z]. Returns predicted z for test patients."""
    blocks_tr, blocks_te = [], []
    if use_features:
        blocks_tr.append(F_tr.to_numpy())
        blocks_te.append(F_te.to_numpy())
    if anchors:
        blocks_tr.append(z_tr[anchors].fillna(0).to_numpy())   # untested anchor -> drug mean
        blocks_te.append(z_te[anchors].fillna(0).to_numpy())
    A_tr, A_te = np.hstack(blocks_tr), np.hstack(blocks_te)
    sc = StandardScaler().fit(A_tr)
    A_tr, A_te = np.nan_to_num(sc.transform(A_tr)), np.nan_to_num(sc.transform(A_te))
    out = pd.DataFrame(np.nan, index=z_te.index, columns=targets)
    for d in targets:
        ok = z_tr[d].notna().to_numpy()
        if ok.sum() >= 30:
            out[d] = RidgeCV(alphas=ALPHAS).fit(A_tr[ok], z_tr[d].to_numpy()[ok]).predict(A_te)
    return out


def main():
    from src.models import _FeaturePrep, BLOCKS_ALL
    from src.rank_eval import per_patient_metrics
    from src.targets import DrugZScaler, load_recommender_data, select_drugs

    data = load_recommender_data()
    drugs = select_drugs(data.auc, data.train)
    sc = DrugZScaler().fit(data.auc.loc[data.train, drugs])
    z_tr, z_te = sc.transform(data.auc.loc[data.train, drugs]), sc.transform(data.auc.loc[data.test, drugs])
    feats = data.features()
    prep = _FeaturePrep(BLOCKS_ALL).fit(feats.loc[z_tr.index])
    F_tr, F_te = prep.transform(feats.loc[z_tr.index]), prep.transform(feats.loc[z_te.index])

    anchors20 = pick_anchors(z_tr, max(KS))
    print("anchor order:", anchors20)
    targets = [d for d in drugs if d not in anchors20]           # same drugs ranked in every arm
    rows, base = [], None
    for k in KS:
        for use_f, label in [(True, "RNA + mutations + clinical"), (False, "anchors only")]:
            if not use_f and k == 0:
                continue
            pred = fit_predict(F_tr, F_te, z_tr, z_te, anchors20[:k], targets, use_f)
            pp = per_patient_metrics(pred, z_te)
            if k == 0:
                base = pp
            d = pp["prec@5"] - base.loc[pp.index, "prec@5"]
            rng = np.random.default_rng(0)
            boot = [d.iloc[rng.integers(0, len(d), len(d))].mean() for _ in range(1000)]
            rows.append({"anchors": k, "features": label, "n_patients": len(pp), "prec@5": pp["prec@5"].mean(),
                         "chance": pp.prevalence.mean(), "spearman": pp.spearman.mean(),
                         "gain_vs_no_anchors": d.mean(), "lo": np.percentile(boot, 2.5), "hi": np.percentile(boot, 97.5)})
            print(rows[-1], flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "results/tables/partial_screen.csv", index=False)
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
