"""Figures for literature/interactions_gaps_report.tex, from results/tables/part1_* and part2_*.

    python -m analysis.make_report_figures
"""
import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.common import RESULTS, ROOT

OUT = ROOT / "literature" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
BLUE, ORANGE, GREY, GREEN, RED = "#4C78A8", "#F58518", "#9D9D9D", "#54A24B", "#E45756"

# ---- Figure A: imputers and screen arms
imp = pd.concat([pd.read_csv(f) for f in glob.glob(str(RESULTS / "part1_imputers_seed*.csv"))])
s = pd.read_csv(RESULTS / "part1_summary_seeds0-9.csv")
fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.1), gridspec_kw={"width_ratios": [1, 1.25, 1.25]})
order = ["mean", "rrr", "knn", "mice", "soft_impute"]
lab = ["drug\nmean", "RRR", "kNN", "MICE", "soft-\nimpute"]
g = imp.groupby("imputer").rmse
ax[0].bar(lab, g.mean()[order], yerr=g.std()[order], color=[GREY, BLUE, BLUE, BLUE, ORANGE])
ax[0].set_ylabel("held-out RMSE (AUC units)")
ax[0].set_ylim(30, 52)
ax[0].set_title("Step 1: imputer choice (10 splits)")
arms = ["random", "observed_screen", "imputed_screen", "observed_screen_eb", "imputed_screen_eb", "paper_reported"]
alab = ["rand", "obs\nraw", "imp\nraw", "obs\nEB", "imp\nEB", "paper\n*"]
cols = [GREY, BLUE, ORANGE, BLUE, ORANGE, GREEN]
for k, metric, title, axis in [(1, "sign_match", "TEST sign agreement", "share of 100 triples"),
                               (2, "replicate", "Replicated (q<0.10, same sign)", "triples per split")]:
    data = [s[s.arm == a][metric].to_numpy() for a in arms]
    bp = ax[k].boxplot(data, tick_labels=alab, showfliers=False, widths=0.55, patch_artist=True)
    for patch, c in zip(bp["boxes"], cols):
        patch.set_facecolor(c); patch.set_alpha(0.35)
    for i, d in enumerate(data):
        ax[k].scatter(np.random.default_rng(i).normal(i + 1, 0.05, len(d)), d, s=8, color=cols[i], zorder=3)
    ax[k].set_title(title); ax[k].set_ylabel(axis)
    ax[k].tick_params(axis="x", labelsize=7.5)
ax[1].axhline(0.5, color="k", lw=0.6, ls=":")
fig.tight_layout()
fig.savefig(OUT / "p1_screen.png", dpi=200)

# ---- Figure B: ex vivo vs induction; PEAR1 by age band
ind = pd.read_csv(RESULTS / "part2_exvivo_induction.csv")
pe = pd.read_csv(RESULTS / "part2_pear1_adjustment.csv")
fig, ax = plt.subplots(1, 2, figsize=(10, 3.2), gridspec_kw={"width_ratios": [1, 1.35]})
ax[0].hist(ind.AUROC_CR, bins=20, color=GREY)
cy = ind[ind.inhibitor.str.startswith("Cytarabine")].AUROC_CR.iloc[0]
ax[0].axvline(cy, color=RED, lw=2)
ax[0].axvline(0.5, color="k", lw=0.6, ls=":")
ax[0].text(cy - 0.005, ax[0].get_ylim()[1] * 0.9, "cytarabine", color=RED, ha="right", fontsize=8)
ax[0].set_xlabel("AUROC of ex vivo AUC for induction CR")
ax[0].set_ylabel("drugs")
ax[0].set_title(f"{len(ind)} drugs, 286 induction patients")
sub = pe[pe.cohort == "de novo"]
bands = ["age < 45", "age 45-60", "age >= 60", "all ages"]
models = ["+ age + ELN", "+ age + ELN + HSC-like", "+ age + ELN + HSC-like + M3 (w/o PEAR1)"]
mlab = ["age + ELN", "+ HSC-like", "+ HSC-like + module 3"]
mc = [BLUE, ORANGE, RED]
for bi, b in enumerate(bands):
    for mi, m in enumerate(models):
        r = sub[(sub.ages == b) & (sub.model == m)].iloc[0]
        y = bi + (mi - 1) * 0.22
        ax[1].errorbar(r.HR_PEAR1_perSD, y, xerr=[[r.HR_PEAR1_perSD - r.lo], [r.hi - r.HR_PEAR1_perSD]],
                       fmt="o", color=mc[mi], ms=4, capsize=2, label=mlab[mi] if bi == 0 else None)
    ev = int(sub[(sub.ages == b)].events.iloc[0])
    ax[1].text(4.05, bi, f"{ev} deaths", va="center", fontsize=7.5)
ax[1].set_yticks(range(len(bands)), bands)
ax[1].axvline(1, color="k", lw=0.6)
ax[1].set_xscale("log")
ax[1].xaxis.set_minor_formatter(plt.NullFormatter())
ax[1].set_xticks([0.5, 1, 2, 4], ["0.5", "1", "2", "4"])
ax[1].set_xlim(0.3, 5.5)
ax[1].invert_yaxis()
ax[1].set_xlabel("PEAR1 hazard ratio per SD (de novo, 95% CI)")
ax[1].legend(fontsize=7, frameon=False, loc="lower right")
ax[1].set_title("Is PEAR1 independent?")
fig.tight_layout()
fig.savefig(OUT / "p2_exvivo_pear1.png", dpi=200)
print("written:", OUT / "p1_screen.png", OUT / "p2_exvivo_pear1.png")
