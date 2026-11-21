"""Plan section 3 (headline): multi-view clustering, interaction discovery with
drug-family partial pooling, fixed-X knockoffs, the young PEAR1-high drug arm, GSEA.

    python -m analysis.plan_headline [cluster|interact|knockoff|pear1|gsea|all]

Cohort: n = 430 patients with RNA-seq and drug screen (one specimen each).
Outputs results/tables/plan_head_*.csv
"""
from __future__ import annotations

import sys
import warnings

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from scipy import stats
from scipy.linalg import eigh
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

from analysis.common import RESULTS, ROOT, load_frame
from analysis.part1_screen import MIN_CARRIERS, MIN_SCREEN_OBS, MIN_TEST_GROUP, N_TOP, bh, known_flags, \
    screen_effects, split, test_triple
from analysis.plan_common import drug_factors, scores
from src.load import load_drug_families

warnings.filterwarnings("ignore")


def frame():
    F = load_frame()
    meta = F["meta"]
    S = scores()
    sc = S["beat"].reindex(meta.dbgap_rnaseq_sample).set_axis(meta.index)
    return F, meta, sc


def alterations(F, meta, min_frac=0.04):
    G = F["muts"].copy()
    keep = [g for g in G.columns if G[g].mean() >= min_frac]
    G = G[keep]
    fus = meta.consensusAMLFusions.fillna("")
    for f in fus[fus.ne("")].str.split(";").explode().str.strip().value_counts().index:
        v = fus.str.contains(f, regex=False).astype(float)
        if v.mean() >= 0.03 and f:
            G[f"fus.{f}"] = v
    return G


def primary_family(drugs):
    fam = load_drug_families()["drug_family"]
    fam = fam[fam.inhibitor.isin(drugs)].drop_duplicates("inhibitor")
    return pd.Series(fam.family.to_numpy(), index=fam.inhibitor).reindex(drugs).fillna("none")


# ---------------------------------------------------------------- 1 multi-view clustering
def rbf(X):
    X = (X - X.mean(0)) / X.std(0).clip(1e-8)
    d2 = ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)
    return np.exp(-d2 / np.median(d2[d2 > 0]))


def center_norm(K):
    n = len(K)
    H = np.eye(n) - 1 / n
    K = H @ K @ H
    return K / np.trace(K)


def mkkm(Ks, k, iters=30):
    """Multiple-kernel k-means (Huang et al. 2012): min sum_v w_v^2 tr(K_v (I - HH')), sum w = 1."""
    w = np.ones(len(Ks)) / len(Ks)
    for _ in range(iters):
        K = sum(wi ** 2 * Ki for wi, Ki in zip(w, Ks))
        _, H = eigh(K, subset_by_index=[len(K) - k, len(K) - 1])
        cost = np.array([np.trace(Ki) - np.trace(H.T @ Ki @ H) for Ki in Ks])
        w_new = (1 / cost) / (1 / cost).sum()
        if np.abs(w_new - w).max() < 1e-6:
            break
        w = w_new
    Hn = H / np.linalg.norm(H, axis=1, keepdims=True).clip(1e-12)
    lab = KMeans(k, n_init=50, random_state=0).fit_predict(Hn)
    return lab, w


def run_cluster():
    F, meta, sc = frame()
    G = alterations(F, meta).fillna(0)
    fac, zc, ve = drug_factors()
    views = {"expression": sc[[c for c in sc.columns if c.startswith(("ct.", "me."))]].to_numpy(float),
             "mutations": G.to_numpy(float), "drug response": fac.to_numpy(float)}
    Ks = [center_norm(rbf(v)) for v in views.values()]
    rng = np.random.default_rng(0)
    stab = []
    for k in range(2, 7):
        full, w = mkkm(Ks, k)
        aris = []
        for _ in range(30):
            i = np.sort(rng.choice(len(meta), int(0.8 * len(meta)), replace=False))
            lab, _ = mkkm([K[np.ix_(i, i)] for K in Ks], k)
            aris.append(adjusted_rand_score(full[i], lab))
        stab.append({"k": k, "stability_ARI": np.mean(aris), **{f"w_{n}": x for n, x in zip(views, w)},
                     "sizes": np.bincount(full).tolist()})
        print(stab[-1], flush=True)
    stab = pd.DataFrame(stab)
    stab.to_csv(RESULTS / "plan_head_cluster_k.csv", index=False)
    k = int(stab.sort_values("stability_ARI", ascending=False).k.iloc[0])
    lab, w = mkkm(Ks, k)

    # survival beyond age + ELN; multi-view vs each single view
    T = pd.to_numeric(meta.overallSurvival, errors="coerce") / 365.25
    E = meta.vitalStatus.map({"Dead": 1, "Alive": 0})
    base = pd.DataFrame({"T": T, "E": E, "age": (meta.ageAtDiagnosis - 60) / 15,
                         "adv": meta.ELN2017.eq("Adverse").astype(int), "fav": meta.ELN2017.eq("Favorable").astype(int)},
                        index=meta.index)
    ok = base["T"].gt(0) & base.E.notna() & meta.ELN2017.notna() & base.age.notna()
    f0 = CoxPHFitter().fit(base[ok], "T", "E")
    surv = []
    labs = {"multi-view": lab}
    for i, v in enumerate(views):
        labs[f"single: {v}"] = KMeans(k, n_init=50, random_state=0).fit_predict(
            eigh(Ks[i], subset_by_index=[len(meta) - k, len(meta) - 1])[1])
    for name, l in labs.items():
        d = base[ok].join(pd.get_dummies(pd.Series(l, index=meta.index), prefix="c", drop_first=True).astype(int))
        f = CoxPHFitter().fit(d, "T", "E")
        lr = 2 * (f.log_likelihood_ - f0.log_likelihood_)
        surv.append({"clustering": name, "k": k, "LR_chi2_vs_age_ELN": lr, "df": k - 1, "p": stats.chi2.sf(lr, k - 1),
                     "C": f.concordance_index_, "C_base": f0.concordance_index_})
    surv = pd.DataFrame(surv)
    surv.to_csv(RESULTS / "plan_head_cluster_survival.csv", index=False)
    print(surv.round(4).to_string(index=False))

    # profile clusters
    young = meta.ageAtDiagnosis < 45
    pear_hi = sc.PEAR1 > sc.loc[young, "PEAR1"].quantile(2 / 3)
    prof = pd.DataFrame({"cluster": lab, "young_PEAR1_high": (young & pear_hi).astype(int), "young": young.astype(int),
                         "age": meta.ageAtDiagnosis, "adverse": meta.ELN2017.eq("Adverse").astype(int)}, index=meta.index)
    prof = prof.join(sc[["ct.HSC-like", "ct.Monocyte-like", "ct.GMP-like", "PEAR1", "LSC17"]]).join(G[[g for g in ["NPM1", "FLT3_ITD", "TP53", "RUNX1", "ASXL1", "DNMT3A", "NRAS", "KRAS", "TET2", "IDH2", "IDH1"] if g in G]])
    summary = prof.groupby("cluster").mean().round(2)
    summary["n"] = prof.groupby("cluster").size()
    summary.to_csv(RESULTS / "plan_head_cluster_profile.csv")
    print(summary.T.to_string())
    ct = pd.crosstab(prof.cluster, prof.young_PEAR1_high)
    print("young PEAR1-high by cluster:\n", ct.to_string(), "\nchi2 p =", stats.chi2_contingency(ct)[1])
    # drugs that separate clusters (observed AUC, Kruskal-Wallis, BH)
    auc = F["auc"]
    res = []
    for d in auc.columns:
        groups = [auc.loc[lab == c, d].dropna() for c in range(k)]
        if min(len(g) for g in groups) >= 10:
            h, p = stats.kruskal(*groups)
            res.append({"drug": d, "p": p, **{f"median_c{c}": g.median() for c, g in enumerate(groups)}})
    res = pd.DataFrame(res)
    res["q"] = bh(res.p)
    res.sort_values("p").to_csv(RESULTS / "plan_head_cluster_drugs.csv", index=False)
    print(f"{(res.q < 0.05).sum()}/{len(res)} drugs differ across clusters at q<0.05; top:\n",
          res.sort_values("p").head(12).round(4).to_string(index=False))
    pd.Series(lab, index=meta.index, name="cluster").to_csv(RESULTS / "plan_head_cluster_labels.csv")


# ---------------------------------------------------------------- 2 interactions with family pooling
def eb(eff, se, mu):
    tau2 = max(np.var(eff - mu) - np.median(se ** 2), 1e-4)
    shrink = tau2 / (tau2 + se ** 2)
    post = mu + shrink * (eff - mu)
    return post, np.sqrt(tau2 * se ** 2 / (tau2 + se ** 2)), tau2


def loo_group_mean(df, key):
    """Leave-one-out precision-weighted mean of effects within key groups (>= 3 members), else 0."""
    w = 1 / df.se ** 2
    g = df.assign(w=w, we=w * df.effect).groupby(key)
    sw, swe, n = g.w.transform("sum"), g.we.transform("sum"), g.w.transform("size")
    mu = (swe - w * df.effect) / (sw - w)
    return mu.where((n >= 3) & df[key[-1]].ne("none"), 0.0)


def run_interact(seeds=range(10)):
    F, meta, _ = frame()
    auc, S_all, G_all = F["auc"], F["states"], F["muts"]
    fam = primary_family(list(auc.columns))
    exact, famset = known_flags()
    rows, hits = [], []
    for seed in seeds:
        scr, tst = split(meta, seed)
        drugs = [d for d in auc.columns if auc.loc[scr, d].notna().sum() >= MIN_SCREEN_OBS]
        genes = [g for g in G_all.columns if G_all.loc[scr, g].sum() >= MIN_CARRIERS]
        states = list(S_all.columns)
        Y = auc.loc[scr, drugs].to_numpy(float)
        eff = screen_effects(Y, ~np.isnan(Y), S_all.loc[scr].to_numpy(float), G_all.loc[scr, genes].to_numpy(float),
                             drugs, states, genes)
        At, Gt, St = auc.loc[tst], G_all.loc[tst], S_all.loc[tst]
        okn = [(lambda ok: (int(Gt.loc[ok, r.gene].sum()), int(ok.sum()) - int(Gt.loc[ok, r.gene].sum())))(
            At[r.drug].notna() & Gt[r.gene].notna()) for r in eff.itertuples()]
        eff["tc"], eff["tn"] = zip(*okn)
        eff = eff[(eff.tc >= MIN_TEST_GROUP) & (eff.tn >= MIN_TEST_GROUP)].reset_index(drop=True)
        eff["family"] = eff.drug.map(fam)
        eff["post_global"], eff["sd_global"], t_g = eb(eff.effect, eff.se, 0.0)
        mu = loo_group_mean(eff, ["state", "gene", "family"])
        eff["post_family"], eff["sd_family"], t_f = eb(eff.effect, eff.se, mu)
        arms = {"raw effect": eff.effect.abs(), "EB global": eff.post_global.abs(),
                "EB family-pooled": eff.post_family.abs()}
        rng = np.random.default_rng(seed)
        picks = {a: eff.loc[s.sort_values(ascending=False).index[:N_TOP]] for a, s in arms.items()}
        picks["random"] = eff.sample(N_TOP, random_state=seed)
        for arm, top in picks.items():
            res = []
            for r in top.itertuples():
                ok = (At[r.drug].notna() & Gt[r.gene].notna()).to_numpy()
                t = test_triple(At[r.drug].to_numpy(float)[ok], Gt[r.gene].to_numpy(float)[ok],
                                St[r.state].to_numpy(float)[ok], rng, n_boot=0)
                res.append((t["effect"] if isinstance(t, dict) else t[0], t["p"] if isinstance(t, dict) else t[2]))
            te, tp = np.array(res).T
            q = bh(tp)
            sign = np.sign(te) == np.sign(top.effect.to_numpy())
            rep = sign & (q < 0.10)
            known = np.array([(r.drug, r.state, r.gene) in exact for r in top.itertuples()])
            known_f = np.array([(r.drug, r.state, r.gene) in famset for r in top.itertuples()])
            rows.append({"seed": seed, "arm": arm, "sign_match": sign.mean(), "replicate": int(rep.sum()),
                         "replicate_known": int((rep & known).sum()), "replicate_new": int((rep & ~known & ~known_f).sum()),
                         "tau2_global": t_g, "tau2_family": t_f})
            for r, ok_, k_, kf in zip(top.itertuples(), rep, known, known_f):
                if ok_:
                    hits.append({"seed": seed, "arm": arm, "drug": r.drug, "state": r.state, "gene": r.gene,
                                 "family": r.family, "known_exact": k_, "known_family": kf})
        print(pd.DataFrame(rows[-4:]).round(3).to_string(index=False), flush=True)
    t = pd.DataFrame(rows)
    t.to_csv(RESULTS / "plan_head_interact_arms.csv", index=False)
    h = pd.DataFrame(hits)
    h.to_csv(RESULTS / "plan_head_interact_hits.csv", index=False)
    s = t.groupby("arm")[["sign_match", "replicate", "replicate_known", "replicate_new"]].agg(["mean", "std"]).round(3)
    print(s.to_string())
    fp = t.pivot(index="seed", columns="arm", values="replicate")
    print("family vs global EB, paired Wilcoxon on replications:",
          stats.wilcoxon(fp["EB family-pooled"], fp["EB global"]).pvalue if (fp["EB family-pooled"] != fp["EB global"]).any() else "identical")
    if len(h):
        new = h[(h.arm == "EB family-pooled") & ~h.known_exact & ~h.known_family]
        print("new replicated triples (family-pooled arm), by number of splits:\n",
              new.groupby(["drug", "state", "gene"]).size().sort_values(ascending=False).head(15).to_string())


# ---------------------------------------------------------------- 3 fixed-X knockoffs
def fixed_x_knockoffs(X, rng):
    """Equicorrelated fixed-X knockoffs (Barber & Candes 2015). X centred, unit-norm columns, n >= 2p."""
    n, p = X.shape
    Sig = X.T @ X
    lam_min = np.linalg.eigvalsh(Sig).min()
    s = np.full(p, min(2 * lam_min, 1.0)) * 0.999
    Sinv = np.linalg.inv(Sig)
    A = 2 * np.diag(s) - np.diag(s) @ Sinv @ np.diag(s)
    C = np.linalg.cholesky(A + 1e-10 * np.eye(p)).T
    Q, _ = np.linalg.qr(np.c_[X, rng.standard_normal((n, p))])
    U = Q[:, p:2 * p]
    return X @ (np.eye(p) - Sinv @ np.diag(s)) + U @ C


def knockoff_threshold(W, q):
    for t in np.sort(np.abs(W[W != 0])):
        if (1 + (W <= -t).sum()) / max((W >= t).sum(), 1) <= q:
            return t
    return np.inf


def run_knockoff(q=0.10, reps=10):
    """Per-drug knockoff+ has a floor (it needs >= 1/q selections per drug), so with ~30
    features per drug it selects nothing. We therefore compute W for every drug x
    feature and apply ONE knockoff+ threshold across all pairs (a multi-task pooling;
    sign-flip symmetry holds per null pair, independence across drugs does not).
    Features are alterations only: adding the six (mutually correlated) cell-state scores
    forces knockoffs ~0.78-correlated with the originals and removes all power.
    Statistic: lasso coefficient difference (LassoCV on [X, X_knockoff])."""
    from sklearn.linear_model import LassoCV
    F, meta, sc = frame()
    auc = F["auc"]
    G = alterations(F, meta)
    G = G.loc[:, G.sum() >= 15]
    feats = G
    Ws, info = [], []
    for d in auc.columns:
        ok = auc[d].notna() & feats.notna().all(1)
        n = int(ok.sum())
        if n < 2 * feats.shape[1] + 20:
            continue
        X = feats[ok].to_numpy(float)
        if X.sum(0).min() < 5:                         # every alteration needs >= 5 carriers
            continue
        X = X - X.mean(0)
        X = X / np.linalg.norm(X, axis=0).clip(1e-12)
        y = auc.loc[ok, d].to_numpy(float)
        y = (y - y.mean()) / y.std()
        p = X.shape[1]
        W = []
        for r in range(reps):
            Xk = fixed_x_knockoffs(X, np.random.default_rng(r))
            b = LassoCV(cv=5, random_state=r).fit(np.c_[X, Xk], y).coef_
            W.append(np.abs(b[:p]) - np.abs(b[p:]))
        Ws.append(np.array(W))                       # reps x p
        bols = np.linalg.lstsq(np.c_[np.ones(n), X], y, rcond=None)[0][1:]
        info += [{"drug": d, "feature": f, "n": n, "direction": "sensitive" if bj < 0 else "resistant"}
                 for f, bj in zip(feats.columns, bols)]
    Wall = np.concatenate(Ws, axis=1)                # reps x (drugs*p)
    sel = np.zeros(Wall.shape[1])
    for r in range(reps):
        sel += Wall[r] >= knockoff_threshold(Wall[r], q)
    res = pd.DataFrame(info)
    res["select_freq"] = sel / reps
    res["is_alteration"] = res.feature.isin(G.columns)
    print(f"{res.drug.nunique()} drugs, {feats.shape[1]} alterations each")
    res = res[res.select_freq >= 0.5].sort_values(["is_alteration", "select_freq"], ascending=False)
    res.to_csv(RESULTS / "plan_head_knockoffs.csv", index=False)
    alt = res[res.is_alteration]
    print(f"selected (pooled knockoff+, q={q}, >=50% of {reps} draws): {len(res)} pairs, {len(alt)} alteration-drug")
    with pd.option_context("display.max_colwidth", 400):
        print(alt.groupby("feature").apply(lambda g: ", ".join(f"{d} ({x[0]})" for d, x in zip(g.drug, g.direction))).to_string())
    known = {"FLT3_ITD": ["quizartinib", "crenolanib", "sorafenib", "sunitinib", "gilteritinib", "midostaurin", "kw-2449",
                          "dovitinib", "foretinib", "cabozantinib", "linifanib", "tandutinib", "ponatinib"],
             "NRAS": ["selumetinib", "trametinib", "ci-1040", "pd0325901"], "KRAS": ["selumetinib", "trametinib", "ci-1040"]}
    flagged = alt.apply(lambda r: any(k in r.drug.lower() for k in known.get(r.feature, [])), axis=1)
    print(f"textbook pairs among selections: {int(flagged.sum())}; per alteration:")
    print(alt.feature.value_counts().to_string())


# ---------------------------------------------------------------- 4 young PEAR1-high drug arm
def run_pear1():
    F, meta, sc = frame()
    auc = F["auc"]
    fam = primary_family(list(auc.columns))
    age = meta.ageAtDiagnosis
    rows = []
    for band, m in [("age < 45", age < 45), ("age >= 45", age >= 45)]:
        for d in auc.columns:
            ok = m & auc[d].notna() & sc.PEAR1.notna()
            if ok.sum() < 25:
                continue
            y = auc.loc[ok, d]
            y = (y - y.mean()) / y.std()
            X = np.c_[np.ones(ok.sum()), sc.loc[ok, "PEAR1"], sc.loc[ok, "ct.HSC-like"]]
            XtX = np.linalg.pinv(X.T @ X)
            b = XtX @ X.T @ y.to_numpy()
            r = y.to_numpy() - X @ b
            se = np.sqrt(r @ r / (len(y) - 3) * XtX[1, 1])
            rows.append({"band": band, "drug": d, "family": fam[d], "n": int(ok.sum()), "effect": b[1], "se": se})
    t = pd.DataFrame(rows)
    out = []
    for band, g in t.groupby("band"):
        g = g.reset_index(drop=True)
        mu = loo_group_mean(g.assign(state="-", gene="-"), ["state", "gene", "family"])
        g["post"], g["post_sd"], tau2 = eb(g.effect, g.se, mu)
        g["z_post"] = g.post / g.post_sd
        g["p_raw"] = 2 * stats.norm.sf(np.abs(g.effect / g.se))
        g["q_raw"] = bh(g.p_raw)
        g["tau2"] = tau2
        out.append(g)
    t = pd.concat(out)
    w = t.pivot(index="drug", columns="band", values="post")
    t.to_csv(RESULTS / "plan_head_pear1_drugs.csv", index=False)
    young = t[t.band == "age < 45"].sort_values("z_post")
    print(f"young: {young.n.median():.0f} patients per drug (median); tau2 = {young.tau2.iloc[0]:.4f}")
    print("PEAR1-high young MORE sensitive (lower AUC), adj HSC-like:\n",
          young.head(12)[["drug", "family", "n", "effect", "se", "post", "z_post", "q_raw"]].round(3).to_string(index=False))
    print("PEAR1-high young MORE resistant:\n",
          young.tail(8)[["drug", "family", "n", "effect", "se", "post", "z_post", "q_raw"]].round(3).to_string(index=False))
    print("correlation of PEAR1 drug effects, young vs older:", round(w.corr().iloc[0, 1], 3))


# ---------------------------------------------------------------- 5 GSEA of PEAR1-correlated genes
def run_gsea(nperm=2000):
    from analysis.plan_common import beat_expression
    E, clin = beat_expression()
    E = E.loc[:, (E > 1).mean() > 0.2]
    young = (clin.ageAtDiagnosis < 45).to_numpy()
    sets = {}
    for line in open(ROOT / "data/raw/msigdb/h.all.symbols.gmt"):
        p = line.rstrip("\n").split("\t")
        sets[p[0].removeprefix("HALLMARK_")] = set(p[2:])
    out = []
    for grp, rows in [("all", np.ones(len(E), bool)), ("age < 45", young)]:
        Er = E[rows].rank()
        pr = Er["PEAR1"]
        r = Er.drop(columns="PEAR1").apply(lambda c: np.corrcoef(c, pr)[0, 1])
        r = r.sort_values(ascending=False)
        genes = r.index.to_numpy()
        w = np.abs(r.to_numpy())
        rng = np.random.default_rng(0)

        def es(mask):
            hit = np.where(mask, w, 0)
            ph = np.cumsum(hit) / hit.sum()
            pm = np.cumsum(~mask) / (~mask).sum()
            d = ph - pm
            return d[np.argmax(np.abs(d))]

        for name, s in sets.items():
            mask = np.isin(genes, list(s))
            if mask.sum() < 15:
                continue
            e = es(mask)
            null = np.array([es(rng.permutation(mask)) for _ in range(nperm)])
            same = null[np.sign(null) == np.sign(e)]
            nes = e / np.abs(same).mean()
            p = (1 + (np.abs(same) >= abs(e)).sum()) / (1 + len(same))
            out.append({"group": grp, "set": name, "size": int(mask.sum()), "ES": e, "NES": nes, "p": p})
    t = pd.DataFrame(out)
    t["q"] = t.groupby("group").p.transform(bh)
    t.to_csv(RESULTS / "plan_head_gsea_pear1.csv", index=False)
    for grp, g in t.groupby("group"):
        print(f"\n{grp}: top hallmarks correlated with PEAR1")
        print(g.sort_values("NES", ascending=False).head(8).round(3).to_string(index=False))
        print(g.sort_values("NES").head(5).round(3).to_string(index=False))


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    steps = {"cluster": run_cluster, "interact": run_interact, "knockoff": run_knockoff, "pear1": run_pear1,
             "gsea": run_gsea}
    for k, fn in steps.items():
        if what in (k, "all"):
            print(f"\n================ {k}")
            fn()
