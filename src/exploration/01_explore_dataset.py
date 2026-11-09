
# %% Setup 
import os, sys
from pathlib import Path

here = Path.cwd().resolve()
root = next((p for p in [here, *here.parents, here / "BeatAML"]
             if (p / "data" / "processed").is_dir()), None)
if root is None:
    raise RuntimeError(f"Can't find the BeatAML repo (a folder with data/processed) from {here}. Open the Project or BeatAML folder in VS Code.")
os.chdir(root)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
NAMES = ["expr", "auc", "pc1", "mutations", "clinical", "splits", "drugs", "genes"]
PROC = Path("data/processed")


def read_table(name):
    pq, cs = PROC / f"{name}.parquet", PROC / f"{name}.csv.gz"
    if pq.exists():
        try:
            return pd.read_parquet(pq)
        except ImportError:
            raise ImportError(f"Reading parquet needs pyarrow, and this Python ({sys.executable}) doesn't have it. "
                              "Pick the project .venv as the kernel, or run: pip install pyarrow") from None
    if cs.exists():
        return pd.read_csv(cs, index_col=0)
    raise FileNotFoundError(f"{name} not found in {PROC.resolve()} - the processed tables are missing.")


D = {n: read_table(n) for n in NAMES}
expr, auc, pc1, mut = D["expr"], D["auc"], D["pc1"], D["mutations"]
clin, drugs, genes = D["clinical"], D["drugs"], D["genes"]
train = clin.index[clin.split == "train"]
test = clin.index[clin.split == "test"]
keep = drugs.index[drugs.keep]
sym = genes["display_label"]
BLUE, ORANGE = "#4C78A8", "#F58518"
Path("results/figures").mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False})


def show(name):
    plt.tight_layout()
    plt.savefig(f"results/figures/{name}.png", dpi=150)
    plt.show()
    plt.close("all")


def spearman(a, b):
    return a.rank().corr(b.rank())


def find_drug(s):
    return next((c for c in auc.columns if s.lower() in c.lower()), None)


print(f"{len(clin)} patients: {len(train)} train, {len(test)} test | {auc.shape[1]} drugs ({len(keep)} kept) | {expr.shape[1]} genes")

# %% Who is in train vs test (age)
for name, ids, c in [("train", train, BLUE), ("test", test, ORANGE)]:
    plt.hist(clin.loc[ids, "age"].dropna(), bins=20, alpha=0.6, color=c, density=True, label=name)
plt.xlabel("age (years)"); plt.ylabel("density"); plt.title("Age"); plt.legend()
show("age")

# %% Tissue and stage
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, col in zip(axes, ["tissue", "stage"]):
    ct = pd.crosstab(clin[col].fillna("missing"), clin.split, normalize="columns")[["train", "test"]]
    ct.plot.bar(ax=ax, color=[BLUE, ORANGE])
    ax.set_title(col); ax.set_ylabel("share of split"); ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=30)
show("tissue_stage")

# %% ELN risk
ct = pd.crosstab(clin["eln2017"].fillna("missing"), clin.split, normalize="columns")[["train", "test"]]
ct.plot.bar(color=[BLUE, ORANGE])
plt.title("ELN 2017 risk"); plt.ylabel("share of split"); plt.xlabel("")
show("eln")

# %% Survival: Kaplan-Meier, train vs test
def km(t, e):
    s, xs, ss = 1.0, [0.0], [1.0]
    for u in np.unique(t[e == 1]):
        s *= 1 - ((t == u) & (e == 1)).sum() / (t >= u).sum()
        xs.append(u); ss.append(s)
    return np.array(xs), np.array(ss)

for name, ids, c in [("train", train, BLUE), ("test", test, ORANGE)]:
    d = clin.loc[ids]
    d = d[(d.os_days > 0) & d.is_dead.notna()]
    x, s = km(d.os_days.to_numpy(float), d.is_dead.to_numpy(int))
    plt.step(x / 365.25, s, where="post", color=c, label=f"{name} (n={len(d)})")
plt.xlabel("years"); plt.ylabel("survival probability"); plt.title("Overall survival"); plt.legend()
show("survival")

# %% Expression: all values
v = expr.to_numpy().ravel()
plt.hist(np.random.default_rng(0).choice(v, 500_000), bins=80, color="gray")
plt.xlabel("normalized expression (log scale, can be negative)"); plt.ylabel("count"); plt.title("Expression values")
show("expr_values")

# %% Expression: per-patient median by wave (batch check)
med = expr.median(axis=1)
waves = ["Waves1+2", "Waves3+4"]
plt.boxplot([med[clin.cohort == w] for w in waves])
plt.xticks([1, 2], waves); plt.ylabel("median expression"); plt.title("Per-patient median by wave")
show("expr_median_wave")

# %% Expression: mean vs variance per gene
plt.scatter(expr.mean(), expr.var(), s=2, alpha=0.25, color="gray")
plt.xlabel("mean"); plt.ylabel("variance"); plt.title("Genes: mean vs variance")
show("expr_mean_var")

# %% Expression: 20 most variable genes
top = expr.var().sort_values(ascending=False).head(20)[::-1]
plt.barh([sym.get(g, g) for g in top.index], top.values, color=BLUE)
plt.xlabel("variance"); plt.title("Most variable genes")
show("top_genes")

# %% Drugs: how many patients per drug (train vs test)
plt.scatter(drugs.n_train, drugs.n_test, c=np.where(drugs.keep, "green", "lightgray"), s=25)
plt.axvline(150, ls="--", c="k", lw=0.8); plt.axhline(50, ls="--", c="k", lw=0.8)
plt.xlabel("patients in train"); plt.ylabel("patients in test"); plt.title("Drugs (green = kept)")
show("drug_counts")

# %% Drugs: AUC distribution (low = sensitive)
a = auc[keep].to_numpy().ravel()
plt.hist(a[~np.isnan(a)], bins=60, color="gray")
plt.xlabel("AUC (low = sensitive)"); plt.ylabel("count"); plt.title("AUC, all kept drugs")
show("auc_hist")

# %% Drugs: AUC spread for the 25 most-measured drugs (train)
top_d = (drugs.loc[keep].n_train).sort_values(ascending=False).head(25).index
order = auc.loc[train, top_d].median().sort_values().index
plt.figure(figsize=(7, 8))
plt.boxplot([auc.loc[train, d].dropna() for d in order], vert=False, showfliers=False)
plt.yticks(range(1, len(order) + 1), order, fontsize=8); plt.xlabel("AUC"); plt.title("AUC by drug (train)")
show("auc_by_drug")

# %% Drugs: who was measured for what (dark = measured)
cols = drugs.loc[keep].sort_values("n_train", ascending=False).index
rows = list(train) + list(test)
plt.figure(figsize=(11, 6))
plt.imshow(auc.loc[rows, cols].notna(), aspect="auto", cmap="Greys", interpolation="nearest")
plt.axhline(len(train) - 0.5, color=ORANGE, lw=1.5)
plt.xlabel("kept drugs"); plt.ylabel("patients (train above orange line, test below)"); plt.title("Measured (dark) vs missing")
show("missingness")

# %% Drugs: drug-drug correlation (train)
A = auc.loc[train, keep]
C = A.corr(method="spearman", min_periods=40)
o = A.mean().sort_values().index
plt.figure(figsize=(7, 6))
plt.imshow(C.loc[o, o], cmap="RdBu_r", vmin=-1, vmax=1); plt.colorbar()
plt.xticks([]); plt.yticks([]); plt.title("Drug-drug Spearman correlation (ordered by mean AUC)")
show("drug_corr")

# %% Cell-type scores (the paper's baseline features)
fig, axes = plt.subplots(2, 3, figsize=(12, 6))
for ax, col in zip(axes.ravel(), pc1.columns):
    for ids, c in [(train, BLUE), (test, ORANGE)]:
        ax.hist(pc1.loc[ids, col].dropna(), bins=25, alpha=0.6, color=c, density=True)
    ax.set_title(col)
show("cell_scores")

# %% Sanity check: Venetoclax vs Monocyte-like (expect positive: monocytic = more resistant)
DRUG, SCORE = "venetoclax", "Monocyte-like"
d = find_drug(DRUG)
x, y = pc1.loc[train, SCORE], auc.loc[train, d]
ok = x.notna() & y.notna()
plt.scatter(x[ok], y[ok], s=10, alpha=0.5, color=BLUE)
plt.xlabel(f"{SCORE} score"); plt.ylabel(f"{d} AUC")
plt.title(f"Spearman rho = {spearman(x[ok], y[ok]):.2f}  (n={ok.sum()})")
show("sanity_scatter")

# %% Mutations: 25 most frequent genes
f_tr, f_te = mut.loc[train].mean(), mut.loc[test].mean()
g = (f_tr + f_te).sort_values(ascending=False).head(25).index[::-1]
y = np.arange(len(g))
plt.figure(figsize=(6, 7))
plt.barh(y - 0.2, f_tr[g], 0.4, color=BLUE, label="train"); plt.barh(y + 0.2, f_te[g], 0.4, color=ORANGE, label="test")
plt.yticks(y, g, fontsize=8); plt.xlabel("fraction of patients mutated"); plt.legend()
show("mutation_freq")

# %% One drug vs one gene (edit the two names)
DRUG, GENE = "venetoclax", "NPM1"
d = find_drug(DRUG)
s = mut.loc[train, GENE] if GENE in mut.columns else clin.loc[train, GENE]
y = auc.loc[train, d]
ok = s.notna() & y.notna()
wt, mu = y[ok & (s == 0)], y[ok & (s == 1)]
delta = (mu.mean() - wt.mean()) / wt.std()  # Glass's delta: negative = mutated more sensitive
plt.boxplot([wt, mu], showfliers=False)
plt.xticks([1, 2], [f"wild-type (n={len(wt)})", f"mutated (n={len(mu)})"]); plt.ylabel("AUC")
plt.title(f"{d} by {GENE}: Glass's delta = {delta:.2f}")
show("drug_vs_gene")

# %% Are the two waves comparable? mean AUC per drug
m_tr, m_te = auc.loc[train, keep].mean(), auc.loc[test, keep].mean()
plt.scatter(m_tr, m_te, s=18, color="gray")
lim = [min(m_tr.min(), m_te.min()), max(m_tr.max(), m_te.max())]
plt.plot(lim, lim, "r--", lw=0.8)
plt.xlabel("mean AUC, train"); plt.ylabel("mean AUC, test"); plt.title("Drug potency by wave (on the line = same)")
show("wave_check")

# %%
