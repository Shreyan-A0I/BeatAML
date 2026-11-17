"""Rebuild the six van Galen cell-type scores the way Bottomly et al. 2022 did.

Method (Bottomly STAR Methods, "Cell-type scores"):
    1. Take the top 30 genes of each signature from van Galen et al. 2019,
       Table S3 (data/raw/mmc3.xlsx).
    2. Centre and scale the expression of those genes separately for bone
       marrow and for peripheral blood + leukapheresis specimens.
    3. Score each specimen as PC1 of the scaled gene block, signed so that the
       score follows the average expression of the gene set.

Fit on Waves1+2 only. Waves3+4 are projected with the Waves1+2 centring,
scaling and rotation, which is the same leak-free split as src/features.py.

Output: results/tables/replication_vangalen_top30_scores.csv (rows = specimens, one column
per cell type), plus a printed comparison with the published PC1 scores.
"""

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

# The scoring functions live in src/features.py (shared with the drug
# recommender); re-exported here so existing imports keep working.
from src.features import VG_N_TOP as N_TOP
from src.features import load_signatures, paper_score, pc1_baseline, split_by_cohort
from src.load import CELL_TYPES, COLS, build_modeling_frame


def main() -> None:
    """Compute the six replicated cell-type scores and write outputs.

    Inputs (no arguments; read from disk and project modules):
        - Modeling frame from src.load.build_modeling_frame: expression matrix X
          (specimens x genes, 4-tuple columns with the gene label in element 1),
          and metadata with the RNA ID and tissue columns.
        - Waves 1+2 / 3+4 split from src.features.split_by_cohort.
        - Gene signatures from data/raw/mmc3.xlsx via load_signatures().

    Outputs (returns None; side effects):
        - results/tables/replication_vangalen_top30_scores.csv: one row per
          specimen (indexed by RNA ID), one column per cell type in CELL_TYPES.
        - results/figures/replication_vangalen_scores.png: scatter plots of
          replicated vs published PC1 scores, one panel per cell type.
        - Printed to stdout: genes found/missing per signature, correlations
          with the published PC1 scores, and our score correlation matrix.
    """
    f = build_modeling_frame(verbose=False)
    meta = f["meta"]
    rna = meta[COLS["rna_id"]].tolist()
    X = f["X"].set_axis(rna, axis=0)
    tr, _ = split_by_cohort(meta)
    tissue_bm = meta[COLS["tissue"]].eq("Bone Marrow Aspirate").to_numpy()

    # Gene labels: the expression columns are 4-tuples, display_label is element 1.
    labels = pd.Index([c[1] for c in X.columns])
    dupes = labels.duplicated().sum()
    if dupes:
        print(f"note: {dupes} duplicated display labels; keeping the first occurrence")
    pos = {}
    for i, lab in enumerate(labels):
        pos.setdefault(lab, i)

    sigs = load_signatures()
    scores = {}
    print("Signature genes found in the expression matrix:")
    for ct in CELL_TYPES:
        found = [g for g in sigs[ct] if g in pos]
        missing = [g for g in sigs[ct] if g not in pos]
        print(f"  {ct:<16} {len(found)}/{N_TOP}  missing: {missing}")
        block = X.iloc[:, [pos[g] for g in found]].to_numpy(dtype=float)
        scores[ct] = paper_score(block, tissue_bm, tr)

    S = pd.DataFrame(scores, index=X.index)
    out = ROOT / "results" / "tables" / "replication_vangalen_top30_scores.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    S.to_csv(out)
    print(f"\nScores saved: {out}")

    # Compare with the published scores on the same specimens.
    pub = pc1_baseline(rna)
    print("\nCorrelation with published PC1 scores (training = Waves1+2, test = Waves3+4):")
    rows = []
    for ct in CELL_TYPES:
        r_all = np.corrcoef(S[ct], pub[ct])[0, 1]
        r_tr = np.corrcoef(S[ct][tr], pub[ct][tr])[0, 1]
        r_te = np.corrcoef(S[ct][~tr], pub[ct][~tr])[0, 1]
        rows.append((ct, r_all, r_tr, r_te))
    print(pd.DataFrame(rows, columns=["cell_type", "all", "Waves1+2", "Waves3+4"])
          .set_index("cell_type").round(3).to_string())

    # Score-to-score correlation matrices: replicated, published, and their difference.
    comp_corr = S[CELL_TYPES].corr()
    pub_corr = pub[CELL_TYPES].corr()
    diff_corr = comp_corr - pub_corr
    print("\nComputed Score Comparisons (replicated scores):")
    print(comp_corr.round(2).to_string())
    print("\nPublished score correlations:")
    print(pub_corr.round(2).to_string())
    print("\nDifference (replicated - published):")
    print(diff_corr.round(2).to_string())
    print(f"\nLargest absolute difference: {diff_corr.abs().to_numpy().max():.3f}")

    fig, axes = plt.subplots(2, 3, figsize=(11, 7))
    for ax, ct in zip(axes.flat, CELL_TYPES):
        ax.scatter(pub[ct][tr], S[ct][tr], s=10, alpha=0.6, label="Waves1+2")
        ax.scatter(pub[ct][~tr], S[ct][~tr], s=10, alpha=0.6, label="Waves3+4")
        r = np.corrcoef(S[ct], pub[ct])[0, 1]
        ax.set_title(f"{ct}  (r = {r:.3f})", fontsize=10)
        ax.set_xlabel("published PC1 score")
        ax.set_ylabel("replicated score")
        ax.grid(alpha=0.3)
    axes.flat[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Replication of van Galen top-30 cell-type scores (Bottomly method) "
                 f"on {len(S)} specimens", fontsize=12)
    fig.tight_layout()
    fig_out = ROOT / "results" / "figures" / "replication_vangalen_scores.png"
    fig.savefig(fig_out, dpi=150)
    plt.close(fig)
    print(f"Figure saved: {fig_out}")


if __name__ == "__main__":
    main()
