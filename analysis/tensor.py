"""Plan 3.1: dose-level tensor factorization with drug-family guidance (Rubik-style).

    python -m analysis.tensor

Tensor: patients x drugs x 7 doses of raw normalized viability (replicates averaged).
Model: centred by drug x dose mean, then masked CP, fit by ALS on observed entries,
with a graph-Laplacian penalty pulling drugs of the same family together.
Tuning/evaluation: hold out whole patient-drug pairs (all doses) in 3 folds.
Metric that is comparable with the AUC imputers: per-drug Spearman between predicted
and observed response of held-out pairs (tensor: mean predicted viability; soft-impute:
predicted AUC), and viability RMSE.
Outputs: results/tables/tensor_cv.csv, data/interim/tensor_factors.pkl
"""
from __future__ import annotations

import itertools
import pickle

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from analysis.common import RESULTS, ROOT, load_frame
from analysis.part1_screen import imp_soft
from src.load import load_drug_families

CACHE = ROOT / "data/interim/dose_tensor.pkl"


def build_tensor():
    if CACHE.exists():
        return pickle.loads(CACHE.read_bytes())
    F = load_frame()
    meta, auc = F["meta"], F["auc"]
    r = pd.read_csv(ROOT / "data/raw/beataml_wv1to4_raw_inhibitor_v4_dbgap.txt", sep="\t", low_memory=False)
    r = r[r.passed_qc.astype(str).str.upper().eq("TRUE")]
    rna2pat = pd.Series(meta.index, index=meta.dbgap_rnaseq_sample)
    r = r[r.dbgap_rnaseq_sample.isin(rna2pat.index) & r.inhibitor.isin(auc.columns)]
    r["pat"] = r.dbgap_rnaseq_sample.map(rna2pat)
    g = r.groupby(["pat", "inhibitor", "well_concentration"]).normalized_viability.mean().reset_index()
    # dose index 0 = highest concentration; keep pairs with >= 7 doses, top 7 concentrations
    g = g.sort_values(["pat", "inhibitor", "well_concentration"], ascending=[True, True, False])
    g["k"] = g.groupby(["pat", "inhibitor"]).cumcount()
    n = g.groupby(["pat", "inhibitor"]).k.transform("max") + 1
    g = g[(n >= 7) & (g.k < 7)]
    pats, drugs = list(meta.index), list(auc.columns)
    T = np.full((len(pats), len(drugs), 7), np.nan)
    T[g.pat.map({p: i for i, p in enumerate(pats)}).to_numpy(), g.inhibitor.map({d: j for j, d in enumerate(drugs)}).to_numpy(),
      g.k.to_numpy()] = np.clip(g.normalized_viability.to_numpy(), 0, 150) / 100
    out = {"T": T, "pats": pats, "drugs": drugs}
    CACHE.write_bytes(pickle.dumps(out))
    return out


def family_laplacian(drugs):
    fam = load_drug_families()["drug_family"]
    fam = fam[fam.inhibitor.isin(drugs)]
    A = np.zeros((len(drugs), len(drugs)))
    idx = {d: i for i, d in enumerate(drugs)}
    for _, grp in fam.groupby("family"):
        ii = [idx[d] for d in grp.inhibitor.unique()]
        for a, b in itertools.combinations(ii, 2):
            A[a, b] = A[b, a] = 1
    return A


def cp_als(T, M, rank, lam_fam, A, mu=1.0, iters=60, seed=0):
    """Masked CP with ridge mu and family penalty lam_fam * sum_{ij in A} ||b_i - b_j||^2."""
    P, D, K = T.shape
    mean = np.nanmean(np.where(M, T, np.nan), axis=0)          # drug x dose
    mean = np.where(np.isnan(mean), np.nanmean(mean), mean)
    Y = np.where(M, T - mean, 0.0)
    rng = np.random.default_rng(seed)
    Ap, B, C = (0.1 * rng.standard_normal((n, rank)) for n in (P, D, K))
    deg = A.sum(1)
    I = np.eye(rank)
    for _ in range(iters):
        for p in range(P):                                     # patients
            m = M[p]
            if not m.any():
                Ap[p] = 0; continue
            dd, kk = np.nonzero(m)
            X = B[dd] * C[kk]
            Ap[p] = np.linalg.solve(X.T @ X + mu * I, X.T @ Y[p, dd, kk])
        for d in range(D):                                     # drugs, Gauss-Seidel with family pull
            m = M[:, d]
            pp, kk = np.nonzero(m)
            X = Ap[pp] * C[kk]
            nb = A[d] @ B
            B[d] = np.linalg.solve(X.T @ X + (mu + lam_fam * deg[d]) * I, X.T @ Y[pp, d, kk] + lam_fam * nb)
        for k in range(K):
            m = M[:, :, k]
            pp, dd = np.nonzero(m)
            X = Ap[pp] * B[dd]
            C[k] = np.linalg.solve(X.T @ X + mu * I, X.T @ Y[pp, dd, k])
    pred = mean + np.einsum("pr,dr,kr->pdk", Ap, B, C)
    return pred, Ap, B, C


def per_drug_spearman(pred, obs, held):
    rs = []
    for d in range(obs.shape[1]):
        m = held[:, d]
        if m.sum() >= 10:
            rs.append(spearmanr(pred[m, d], obs[m, d]).statistic)
    return float(np.nanmean(rs))


def main():
    TT = build_tensor()
    T, pats, drugs = TT["T"], TT["pats"], TT["drugs"]
    F = load_frame()
    AUC = F["auc"].loc[pats, drugs].to_numpy(float)
    pair_obs = ~np.isnan(T).any(2) & ~np.isnan(AUC)            # pairs observed in both representations
    print(f"tensor {T.shape}, observed cells {np.isfinite(T).mean():.1%}, pairs in both {pair_obs.sum()}")
    A = family_laplacian(drugs)
    print(f"family graph: {int(A.sum() / 2)} edges, {int((A.sum(1) > 0).sum())}/{len(drugs)} drugs with a family")

    rng = np.random.default_rng(0)
    pi, di = np.nonzero(pair_obs)
    fold = rng.integers(0, 10, len(pi))                         # use folds 0,1,2 (10% each)
    rows = []
    for f in range(3):
        held = np.zeros_like(pair_obs)
        held[pi[fold == f], di[fold == f]] = True
        Mtr = np.isfinite(T) & ~held[:, :, None]
        obs_auc = AUC
        # baselines
        mean_dd = np.nanmean(np.where(Mtr, T, np.nan), 0)
        base = np.broadcast_to(mean_dd, T.shape)
        rows.append({"fold": f, "model": "drug x dose mean", "rank": 0, "lam": 0,
                     "rmse_viab": float(np.sqrt(np.nanmean((base - T)[held] ** 2))),
                     "spearman_auc": np.nan})
        Mauc = np.isfinite(AUC) & ~held
        si = imp_soft(np.nan_to_num(AUC), Mauc, lam=5)
        rows.append({"fold": f, "model": "soft-impute (AUC)", "rank": 0, "lam": 5, "rmse_viab": np.nan,
                     "spearman_auc": per_drug_spearman(si, obs_auc, held)})
        for rank, lam in itertools.product([3, 6, 10, 15], [0.0, 3.0, 30.0]):
            pred, *_ = cp_als(T, Mtr, rank, lam, A, iters=40)
            rows.append({"fold": f, "model": "CP tensor", "rank": rank, "lam": lam,
                         "rmse_viab": float(np.sqrt(np.nanmean((pred - T)[held] ** 2))),
                         "spearman_auc": per_drug_spearman(pred.mean(2), obs_auc, held)})
            print(rows[-1], flush=True)
    cv = pd.DataFrame(rows)
    cv.to_csv(RESULTS / "tensor_cv.csv", index=False)
    s = cv.groupby(["model", "rank", "lam"])[["rmse_viab", "spearman_auc"]].mean().reset_index()
    print(s.round(4).to_string(index=False))

    best = s[s.model == "CP tensor"].sort_values("spearman_auc", ascending=False).iloc[0]
    print("best:", best.to_dict())
    pred, Ap, B, C = cp_als(T, np.isfinite(T), int(best["rank"]), float(best["lam"]), A, iters=80)
    fac = pd.DataFrame(Ap, index=pats, columns=[f"tf{r + 1}" for r in range(Ap.shape[1])])
    pickle.dump({"patient": fac, "drug": pd.DataFrame(B, index=drugs), "dose": C, "pred": pred,
                 "best": best.to_dict()}, open(ROOT / "data/interim/tensor_factors.pkl", "wb"))


if __name__ == "__main__":
    main()
