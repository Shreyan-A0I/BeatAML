"""Compare feature sets for per-drug AUC prediction under the Option A split.

Train on Waves1+2, test on Waves3+4, one ridge model per inhibitor.

    python -m src.evaluate

Feature sets:
    PC1         the six published van Galen scores (the baseline to beat)
    NMF10       ten NMF components learned from training expression
    PC1+NMF10   both together
    top2000     the 2,000 most variable training genes, raw (a reference point)

Reported per feature set: the median per-drug Spearman and R^2 on Waves3+4,
and how many drugs score above zero. Drugs are only scored when there are
enough specimens in both cohorts (see MIN_TRAIN, MIN_TEST).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

from .features import nmf_features, pc1_baseline, select_genes, split_by_cohort
from .load import COLS, build_modeling_frame

MIN_TRAIN = 50   # specimens with a drug AUC in Waves1+2 needed to fit a drug
MIN_TEST = 30    # specimens with a drug AUC in Waves3+4 needed to score a drug
K_NMF = 10
ALPHAS = np.logspace(-2, 4, 25)
FEATURE_ORDER = ["PC1", "NMF10", "PC1+NMF10", "top2000"]


def _fit_predict(Ftr: np.ndarray, ytr: np.ndarray, Fte: np.ndarray) -> np.ndarray:
    scaler = StandardScaler().fit(Ftr)
    model = RidgeCV(alphas=ALPHAS).fit(scaler.transform(Ftr), ytr)
    return np.asarray(model.predict(scaler.transform(Fte)), dtype=float)


def run() -> pd.DataFrame:
    f = build_modeling_frame(verbose=False)
    meta = f["meta"]
    rna_ids = meta[COLS["rna_id"]].tolist()

    # X and y are row-aligned to `meta` by construction in build_modeling_frame
    # (X by RNA-seq ID, y by base_samp). Re-key both to RNA-seq IDs so the PC1
    # lookup and the cohort masks line up.
    X = f["X"].set_axis(rna_ids, axis=0)
    y = f["y"].set_axis(rna_ids, axis=0)

    tr_mask, te_mask = split_by_cohort(meta)
    X_tr, X_te = X.loc[tr_mask], X.loc[te_mask]

    # Features that depend only on training data.
    genes = select_genes(X_tr, n_top=2000)
    nmf_tr, nmf_te, _ = nmf_features(X_tr, X_te, k=K_NMF)

    pc1 = pc1_baseline(rna_ids).to_numpy(dtype=float)
    pc1_tr, pc1_te = pc1[tr_mask], pc1[te_mask]
    if np.isnan(pc1_tr).any() or np.isnan(pc1_te).any():
        raise RuntimeError("PC1 baseline has missing rows for the modeling set")

    nmf_tr_v = nmf_tr.to_numpy(dtype=float)
    nmf_te_v = nmf_te.to_numpy(dtype=float)
    feature_sets: dict[str, tuple[np.ndarray, np.ndarray]] = {
        "PC1": (pc1_tr, pc1_te),
        "NMF10": (nmf_tr_v, nmf_te_v),
        "PC1+NMF10": (np.hstack([pc1_tr, nmf_tr_v]), np.hstack([pc1_te, nmf_te_v])),
        "top2000": (X_tr.loc[:, genes].to_numpy(dtype=float),
                    X_te.loc[:, genes].to_numpy(dtype=float)),
    }

    y_tr, y_te = y.loc[tr_mask], y.loc[te_mask]
    rows: list[dict[str, object]] = []
    for drug in y.columns:
        ytr_d = y_tr[drug].to_numpy(dtype=float)
        yte_d = y_te[drug].to_numpy(dtype=float)
        ok_tr = ~np.isnan(ytr_d)
        ok_te = ~np.isnan(yte_d)
        if ok_tr.sum() < MIN_TRAIN or ok_te.sum() < MIN_TEST:
            continue
        for name, (Ftr, Fte) in feature_sets.items():
            pred = _fit_predict(Ftr[ok_tr], ytr_d[ok_tr], Fte[ok_te])
            rho = float(pd.Series(pred).corr(pd.Series(yte_d[ok_te]), method="spearman"))
            rows.append({
                "drug": drug,
                "features": name,
                "n_train": int(ok_tr.sum()),
                "n_test": int(ok_te.sum()),
                "spearman": rho,
                "r2": float(r2_score(yte_d[ok_te], pred)),
            })
    return pd.DataFrame(rows)


def summarise(res: pd.DataFrame) -> pd.DataFrame:
    g = res.groupby("features", sort=False)
    out = pd.DataFrame({
        "drugs": g.size(),
        "median_spearman": g["spearman"].median(),
        "median_r2": g["r2"].median(),
        "drugs_r2_gt0": g["r2"].apply(lambda s: int((s > 0).sum())),
    })
    return out.loc[FEATURE_ORDER]


if __name__ == "__main__":
    res = run()
    print(f"scored drugs: {res['drug'].nunique()}  (MIN_TRAIN={MIN_TRAIN}, MIN_TEST={MIN_TEST})")
    print()
    print(summarise(res).round(3).to_string())
