"""Part 1: imputation-screened mutation x cell-state x drug interaction discovery.

    python -m analysis.part1_screen                # primary split (seed 0), full report
    python -m analysis.part1_screen --seeds 0-9    # repeat the whole pipeline over splits

Design
  * Patients are split 60/40 into SCREEN and TEST, stratified by wave x ELN2017.
    Nothing from TEST is used by the imputers or the screen.
  * Step 1: imputers are fit on SCREEN observed entries. 20% of those entries are held
    out to choose the imputer; hyper-parameters are tuned on a further 15% of the
    remaining entries, so the holdout is never used for tuning.
  * Step 2: on the completed SCREEN matrix, fit  z(AUC) ~ mut + z(state) + mut:z(state)
    per drug, and rank triples by |interaction coefficient| (effect size, not p-value).
  * Step 3: test the top 100 on TEST using observed entries only: OLS interaction,
    bootstrap CI, BH-FDR over the 100.
  * Step 4: controls. (a) 100 random eligible triples not selected by the screen;
    (b) the same screen run on SCREEN observed entries only (no imputation).
    A triple "replicates" if TEST q < 0.10 and its sign matches the screen.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import train_test_split

from analysis.common import RESULTS, SUPP, known_interactions, load_frame

MIN_SCREEN_OBS = 40      # drug must have >= this many observed AUCs in SCREEN
MIN_CARRIERS = 20        # mutation must have >= this many carriers in SCREEN
MIN_TEST_GROUP = 8       # TEST observed carriers and non-carriers needed per triple
N_TOP = 100
N_BOOT = 1000
Q_REPLICATE = 0.10


# ---------------------------------------------------------------- split
def split(meta: pd.DataFrame, seed: int) -> tuple[pd.Index, pd.Index]:
    strata = meta["cohort"].astype(str) + "|" + meta["ELN2017"].fillna("none").astype(str)
    strata = strata.where(strata.map(strata.value_counts()) >= 5, meta["cohort"].astype(str))
    scr, tst = train_test_split(meta.index, test_size=0.4, stratify=strata, random_state=seed)
    return pd.Index(scr), pd.Index(tst)


# ---------------------------------------------------------------- imputers
def _standardise(Y: np.ndarray, M: np.ndarray):
    mu = np.nanmean(np.where(M, Y, np.nan), 0)
    sd = np.nanstd(np.where(M, Y, np.nan), 0)
    sd[sd == 0] = 1
    return (Y - mu) / sd, mu, sd


def imp_mean(Y, M, **_):
    mu = np.nanmean(np.where(M, Y, np.nan), 0)
    return np.where(M, Y, mu)


def imp_knn(Y, M, k=10, **_):
    return KNNImputer(n_neighbors=k, weights="distance").fit_transform(np.where(M, Y, np.nan))


def imp_mice(Y, M, **_):
    Z, mu, sd = _standardise(Y, M)
    est = IterativeImputer(estimator=BayesianRidge(), max_iter=15, n_nearest_features=40,
                           initial_strategy="mean", random_state=0, tol=1e-3)
    return est.fit_transform(np.where(M, Z, np.nan)) * sd + mu


def imp_soft(Y, M, lam=10.0, iters=200, **_):
    """Soft-impute (Mazumder et al. 2010) on per-drug standardised AUC."""
    Z, mu, sd = _standardise(Y, M)
    Z0 = np.where(M, Z, 0.0)
    W = np.zeros_like(Z0)
    for _ in range(iters):
        U, s, Vt = np.linalg.svd(np.where(M, Z0, W), full_matrices=False)
        Wn = (U * np.maximum(s - lam, 0)) @ Vt
        if np.linalg.norm(Wn - W) / max(np.linalg.norm(W), 1e-8) < 1e-5:
            W = Wn
            break
        W = Wn
    return np.where(M, Z, W) * sd + mu


def imp_rrr(Y, M, feats=None, r=5, wd=1e-3, epochs=1500, **_):
    """Reduced-rank regression AUC ~ features @ (A @ C), masked MSE, Adam."""
    Z, mu, sd = _standardise(Y, M)
    torch.manual_seed(0)
    Xf = torch.tensor(feats, dtype=torch.float32)
    Zt = torch.tensor(np.where(M, Z, 0.0), dtype=torch.float32)
    Mt = torch.tensor(M, dtype=torch.float32)
    p, d = Xf.shape[1], Zt.shape[1]
    A = torch.nn.Parameter(0.01 * torch.randn(p, r))
    C = torch.nn.Parameter(0.01 * torch.randn(r, d))
    b = torch.nn.Parameter(torch.zeros(d))
    opt = torch.optim.Adam([A, C, b], lr=1e-2)
    for _ in range(epochs):
        opt.zero_grad()
        pred = Xf @ A @ C + b
        loss = ((pred - Zt) ** 2 * Mt).sum() / Mt.sum() + wd * (A.pow(2).sum() + C.pow(2).sum())
        loss.backward()
        opt.step()
    W = (Xf @ A @ C + b).detach().numpy()
    return np.where(M, Z, W) * sd + mu


IMPUTERS = {
    "mean": (imp_mean, [{}]),
    "knn": (imp_knn, [{"k": k} for k in (5, 10, 20, 40)]),
    "mice": (imp_mice, [{}]),
    "soft_impute": (imp_soft, [{"lam": l} for l in (2, 5, 10, 20, 40)]),
    "rrr": (imp_rrr, [{"r": r} for r in range(1, 16)]),
}


def patient_features(F, scr: pd.Index, genes: list[str]) -> np.ndarray:
    """SCREEN-only patient features for RRR: states, eigengenes, mutations, 30 expression PCs."""
    X = F["X"].loc[scr]
    Xv = X.loc[:, X.var().sort_values().index[-2000:]].to_numpy(float)
    Xv = (Xv - Xv.mean(0)) / Xv.std(0)
    U, s, _ = np.linalg.svd(Xv, full_matrices=False)
    pcs = U[:, :30] * s[:30]
    parts = [F["states"].loc[scr].to_numpy(), F["mes"].loc[scr].to_numpy(), pcs,
             F["muts"].loc[scr, genes].fillna(0).to_numpy()]
    Z = np.hstack(parts).astype(float)
    return (Z - Z.mean(0)) / np.where(Z.std(0) == 0, 1, Z.std(0))


def _metrics(Y, P, H):
    y, p = Y[H], P[H]
    sse = ((y - p) ** 2).sum()
    out = {"rmse": float(np.sqrt(sse / H.sum())), "r2": float(1 - sse / ((y - y.mean()) ** 2).sum())}
    rs = []
    for j in range(Y.shape[1]):
        h = H[:, j]
        if h.sum() >= 5 and np.std(P[h, j]) > 0:
            rs.append(stats.pearsonr(Y[h, j], P[h, j])[0])
    out["median_drug_pearson"] = float(np.median(rs)) if rs else float("nan")
    out["n_drugs_scored"] = len(rs)
    return out


def choose_imputer(Y, M, feats, seed):
    rng = np.random.default_rng(seed)
    obs = np.argwhere(M)
    perm = rng.permutation(len(obs))
    n_hold, n_tune = int(0.2 * len(obs)), int(0.15 * len(obs))
    H = np.zeros_like(M); H[tuple(obs[perm[:n_hold]].T)] = True
    T = np.zeros_like(M); T[tuple(obs[perm[n_hold:n_hold + n_tune]].T)] = True
    rows, best = [], {}
    sse_mean = None
    for name, (fn, grid) in IMPUTERS.items():
        # tune on T (fit without H and T)
        scores = []
        for g in grid:
            P = fn(Y, M & ~H & ~T, feats=feats, **g)
            scores.append(((Y[T] - P[T]) ** 2).mean())
        g = grid[int(np.argmin(scores))]
        P = fn(Y, M & ~H, feats=feats, **g)          # refit on everything but the holdout
        m = _metrics(Y, P, H)
        sse = ((Y[H] - P[H]) ** 2).sum()
        if name == "mean":
            sse_mean = sse
            m["median_drug_pearson"] = float("nan")   # constant per drug: undefined
        m["r2_vs_drug_mean"] = float(1 - sse / sse_mean)
        rows.append({"imputer": name, "params": json.dumps(g), **m})
        best[name] = g
    res = pd.DataFrame(rows).set_index("imputer")
    winner = res.rmse.idxmin()
    return res, winner, best[winner]


# ---------------------------------------------------------------- interaction fits
def _zs(v):
    return (v - np.nanmean(v)) / np.nanstd(v)


def screen_effects(Y: np.ndarray, M: np.ndarray | None, S: np.ndarray, G: np.ndarray,
                   drugs, states, genes, min_obs=15) -> pd.DataFrame:
    """Interaction coefficient for every drug x state x gene.
    Y: patients x drugs. M: observed mask (None = complete). S: states. G: mutations (NaN ok)."""
    out = []
    for gi, g in enumerate(genes):
        gv = G[:, gi]
        for si, s in enumerate(states):
            sv = _zs(S[:, si])
            for di, d in enumerate(drugs):
                ok = ~np.isnan(gv) & (M[:, di] if M is not None else True)
                if ok.sum() < min_obs or gv[ok].sum() < 5:
                    continue
                y = _zs(Y[ok, di])
                X = np.column_stack([np.ones(ok.sum()), gv[ok], sv[ok], gv[ok] * sv[ok]])
                XtX_inv = np.linalg.pinv(X.T @ X)
                beta = XtX_inv @ X.T @ y
                r = y - X @ beta
                se = np.sqrt(r @ r / max(len(y) - 4, 1) * XtX_inv[3, 3])
                out.append((d, s, g, beta[3], se))
    return pd.DataFrame(out, columns=["drug", "state", "gene", "effect", "se"])


def test_triple(y, g, s, rng, n_boot=N_BOOT):
    """OLS z(y) ~ g + z(s) + g:z(s) on observed TEST rows. Returns effect, se, p, CI, n."""
    y, s = _zs(y), _zs(s)
    X = np.column_stack([np.ones(len(y)), g, s, g * s])
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    resid = y - X @ beta
    dof = len(y) - 4
    se = np.sqrt(resid @ resid / dof * XtX_inv[3, 3])
    p = 2 * stats.t.sf(abs(beta[3] / se), dof)
    lo = hi = np.nan
    if n_boot:
        idx = rng.integers(0, len(y), (n_boot, len(y)))
        Xb, yb = X[idx], y[idx]
        XtX = np.einsum("bni,bnj->bij", Xb, Xb) + 1e-8 * np.eye(4)
        Xty = np.einsum("bni,bn->bi", Xb, yb)
        bb = np.linalg.solve(XtX, Xty[..., None])[..., 3, 0]
        lo, hi = np.percentile(bb, [2.5, 97.5])
    return beta[3], se, p, lo, hi


def bh(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    q = p[o] * len(p) / (np.arange(len(p)) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q); out[o] = np.minimum(q, 1)
    return out


def known_flags() -> tuple[set, set]:
    k = known_interactions(0.10)
    exact = {(r.inhibitor, r.cell_type, r.gene) for r in k.itertuples()}
    fam = pd.read_excel(SUPP / "S6_family_cell_type.xlsx", header=1)
    fam = fam[fam.inter_qval < 0.10]
    members = pd.read_excel(ROOT_FAM, sheet_name="drug_family")
    famset = set()
    for r in fam.itertuples():
        for d in members.loc[members.family == r.family, "inhibitor"]:
            famset.add((d, r.cell_type, r.inter_gene.removeprefix("mut.")))
    return exact, famset


from analysis.common import ROOT  # noqa: E402
ROOT_FAM = ROOT / "data/raw/beataml_drug_families.xlsx"


# ---------------------------------------------------------------- pipeline
def run(seed: int, full: bool = True) -> dict:
    F = load_frame()
    meta, auc, S_all, G_all = F["meta"], F["auc"], F["states"], F["muts"]
    scr, tst = split(meta, seed)

    drugs = [d for d in auc.columns if auc.loc[scr, d].notna().sum() >= MIN_SCREEN_OBS]
    genes = [g for g in G_all.columns if G_all.loc[scr, g].sum() >= MIN_CARRIERS]
    states = list(S_all.columns)

    Y = auc.loc[scr, drugs].to_numpy(float)
    M = ~np.isnan(Y)
    feats = patient_features(F, scr, genes)

    # Step 1
    imp_table, winner, params = choose_imputer(Y, M, feats, seed)
    fn = IMPUTERS[winner][0]
    Yc = fn(Y, M, feats=feats, **params)

    # Step 2: completed-matrix screen and observed-only screen
    Sv, Gv = S_all.loc[scr].to_numpy(float), G_all.loc[scr, genes].to_numpy(float)
    eff_imp = screen_effects(Yc, None, Sv, Gv, drugs, states, genes)
    eff_obs = screen_effects(Y, M, Sv, Gv, drugs, states, genes)
    eff = eff_imp.merge(eff_obs, on=["drug", "state", "gene"], how="left", suffixes=("_imp", "_obs"))

    # Eligibility on TEST sample sizes only (no outcome values used)
    At, Gt, St = auc.loc[tst], G_all.loc[tst], S_all.loc[tst]

    def n_groups(r):
        ok = At[r.drug].notna() & Gt[r.gene].notna()
        c = int(Gt.loc[ok, r.gene].sum())
        return c, int(ok.sum()) - c
    ng = eff.apply(n_groups, axis=1, result_type="expand")
    eff["test_carriers"], eff["test_noncarriers"] = ng[0], ng[1]
    elig = eff[(eff.test_carriers >= MIN_TEST_GROUP) & (eff.test_noncarriers >= MIN_TEST_GROUP)].copy()

    rng = np.random.default_rng(1000 + seed)
    top_imp = elig.reindex(elig.effect_imp.abs().sort_values(ascending=False).index).head(N_TOP)
    top_obs = elig.dropna(subset=["effect_obs"])
    top_obs = top_obs.reindex(top_obs.effect_obs.abs().sort_values(ascending=False).index).head(N_TOP)
    rand = elig.drop(top_imp.index).sample(N_TOP, random_state=seed)
    # Empirical-Bayes ranking: shrink each imputed effect toward 0 by the reliability its
    # OBSERVED SCREEN data supports (se_obs, from the observed-only fit). Imputed cells add
    # no information about uncertainty, so the observed n is the honest denominator.
    e = elig.dropna(subset=["se_obs"]).copy()
    tau2 = max(e.effect_imp.var() - (e.se_obs ** 2).median(), 1e-4)
    e["effect_eb"] = e.effect_imp * tau2 / (tau2 + e.se_obs ** 2)
    top_eb = e.reindex(e.effect_eb.abs().sort_values(ascending=False).index).head(N_TOP)
    # Same shrinkage applied to the observed-only effects, so the imputation comparison is fair.
    tau2_o = max(e.effect_obs.var() - (e.se_obs ** 2).median(), 1e-4)
    e["effect_obs_eb"] = e.effect_obs * tau2_o / (tau2_o + e.se_obs ** 2)
    top_obs_eb = e.reindex(e.effect_obs_eb.abs().sort_values(ascending=False).index).head(N_TOP)
    print(f"  EB: var(effect_imp)={e.effect_imp.var():.4f} var(effect_obs)={e.effect_obs.var():.4f} "
          f"median se_obs^2={(e.se_obs ** 2).median():.4f} -> tau2_imp={tau2:.4f} tau2_obs={tau2_o:.4f}")

    exact, famset = known_flags()
    elig["known"] = [(d, s, g) in exact for d, s, g in zip(elig.drug, elig.state, elig.gene)]

    def test_arm(df, arm, screen_col):
        rows = []
        for r in df.itertuples():
            ok = (At[r.drug].notna() & Gt[r.gene].notna()).to_numpy()
            y = At[r.drug].to_numpy(float)[ok]
            g = Gt[r.gene].to_numpy(float)[ok]
            s = St[r.state].to_numpy(float)[ok]
            e, se, p, lo, hi = test_triple(y, g, s, rng, N_BOOT if full else 0)
            key = (r.drug, r.state, r.gene)
            rows.append({"arm": arm, "drug": r.drug, "state": r.state, "gene": r.gene,
                         "screen_effect": getattr(r, screen_col), "test_effect": e, "test_se": se,
                         "ci_lo": lo, "ci_hi": hi, "p": p, "n_test": int(ok.sum()),
                         "test_carriers": int(g.sum()),
                         "known": "paper-drug" if key in exact else ("paper-family" if key in famset else "new")})
        out = pd.DataFrame(rows)
        out["q"] = bh(out.p)
        out["sign_match"] = np.sign(out.test_effect) == np.sign(out.screen_effect)
        out["replicates"] = (out.q < Q_REPLICATE) & out.sign_match
        return out

    tests = pd.concat([test_arm(top_imp, "imputed_screen", "effect_imp"),
                       test_arm(top_eb, "imputed_screen_eb", "effect_eb"),
                       test_arm(top_obs, "observed_screen", "effect_obs"),
                       test_arm(top_obs_eb, "observed_screen_eb", "effect_obs_eb"),
                       test_arm(rand, "random", "effect_imp"),
                       test_arm(elig[elig.known], "paper_reported", "effect_imp")], ignore_index=True)

    summary = tests.groupby("arm").agg(
        n=("p", "size"), replicate=("replicates", "sum"), sign_match=("sign_match", "mean"),
        nominal_p05=("p", lambda p: int((p < 0.05).sum())),
        median_abs_test_effect=("test_effect", lambda x: float(np.median(np.abs(x)))),
        known_drug=("known", lambda k: int((k == "paper-drug").sum())),
        known_family=("known", lambda k: int((k == "paper-family").sum())),
        median_se=("test_se", "median"))
    summary["mde_80pct"] = 2.8 * summary.median_se   # |effect| detectable at 80% power, alpha .05
    corr = tests.groupby("arm").apply(lambda d: stats.spearmanr(d.screen_effect, d.test_effect)[0],
                                      include_groups=False)
    summary["rho_screen_vs_test"] = corr
    # How many paper-reported triples were eligible at all, and where did they rank?
    elig["rank_imp"] = elig.effect_imp.abs().rank(ascending=False)
    return {"seed": seed, "n_screen": len(scr), "n_test": len(tst), "n_drugs": len(drugs),
            "genes": genes, "imputers": imp_table, "winner": winner, "params": params,
            "n_eligible": len(elig), "tests": tests, "summary": summary,
            "known_eligible": elig[elig.known][["drug", "state", "gene", "effect_imp", "effect_obs", "rank_imp"]]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0")
    a = ap.parse_args()
    lo, _, hi = a.seeds.partition("-")
    seeds = range(int(lo), int(hi or lo) + 1)
    RESULTS.mkdir(parents=True, exist_ok=True)
    allsum = []
    for sd in seeds:
        R = run(sd, full=(sd == seeds[0]))
        tag = f"seed{sd}"
        R["imputers"].to_csv(RESULTS / f"part1_imputers_{tag}.csv")
        R["tests"].to_csv(RESULTS / f"part1_tests_{tag}.csv", index=False)
        R["known_eligible"].to_csv(RESULTS / f"part1_known_eligible_{tag}.csv", index=False)
        s = R["summary"].assign(seed=sd, winner=R["winner"])
        allsum.append(s)
        print(f"\n=== seed {sd}: SCREEN {R['n_screen']} / TEST {R['n_test']}, {R['n_drugs']} drugs, "
              f"genes {R['genes']}, eligible triples {R['n_eligible']}")
        print(R["imputers"].round(3).to_string())
        print(f"winner: {R['winner']} {R['params']}")
        print(s.round(3).to_string())
        print("paper-reported triples eligible:", len(R["known_eligible"]))
        print(R["known_eligible"].round(3).to_string())
    pd.concat(allsum).to_csv(RESULTS / f"part1_summary_seeds{a.seeds}.csv")


if __name__ == "__main__":
    main()
