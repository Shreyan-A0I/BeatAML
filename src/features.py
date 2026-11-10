"""Latent feature builders for the Beat AML drug-response models.

Every builder is FIT ON THE TRAINING COHORT ONLY and then applied to the test
cohort. The default split is the paper's own: fit on Waves1+2, test on
Waves3+4 (see reports/schema_notes.md, Option A). Fitting on all specimens and
then cross-validating would leak test information into the features.

Builders:
    pc1_baseline()          published van Galen PC1 scores (6 cell types), no fitting
    select_genes()          top-variance genes, chosen on the training cohort
    nmf_features()          NMF on non-negative shifted expression, fit on train
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, cast

import numpy as np
import pandas as pd
from sklearn.decomposition import NMF

from .load import COLS, load_pc1_baseline


def split_by_cohort(
    meta: pd.DataFrame,
    train: str = "Waves1+2",
    test: str = "Waves3+4",
) -> tuple[np.ndarray, np.ndarray]:
    """Return (train_mask, test_mask) boolean arrays over the modeling rows."""
    cohort = meta[COLS["cohort"]].to_numpy()
    return np.asarray(cohort == train, dtype=bool), np.asarray(cohort == test, dtype=bool)


def pc1_baseline(specimen_ids: Iterable[str]) -> pd.DataFrame:
    """Published PC1 cell-type scores, aligned to `specimen_ids` (RNA-seq IDs).

    These were fit by the paper on Waves1+2 and projected onto Waves3+4, so they
    are only leak-free under the Option A split. Do not re-fit them inside CV.
    """
    pc1 = load_pc1_baseline()
    return pc1.reindex(list(specimen_ids))


def select_genes(X_train: pd.DataFrame, n_top: int = 2000) -> list[Any]:
    """Top `n_top` genes by variance, computed on the training rows only.

    Returns the column labels of `X_train` (for this dataset each label is a
    4-part tuple: stable_id, display_label, description, biotype).
    """
    var = np.asarray(X_train.var(axis=0), dtype=float)
    order = np.argsort(-var)[:n_top]
    return [X_train.columns[i] for i in order]


def nmf_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    k: int = 10,
    random_state: int = 0,
    n_top: int = 2000,
) -> tuple[pd.DataFrame, pd.DataFrame, NMF]:
    """Fit NMF on training specimens, transform both cohorts.

    Expression here is log-scaled and therefore has negative values, but NMF
    needs non-negative input. We shift every gene by its training-set minimum
    so the smallest training value is zero. The shift is a per-gene constant,
    so it does not change which genes are variable, only their zero point.
    Test values are shifted by the SAME training minimums, and clipped at zero
    so an unseen low value cannot make the input negative.
    """
    genes = select_genes(X_train, n_top=n_top)
    tr = X_train.loc[:, genes].to_numpy(dtype=float)
    te = X_test.loc[:, genes].to_numpy(dtype=float)

    shift = tr.min(axis=0)
    tr_nn = np.clip(tr - shift, 0.0, None)
    te_nn = np.clip(te - shift, 0.0, None)

    # sklearn's NMF signature has no annotations and infers n_components as str
    # from its "auto" default, so the int is cast for the type checker only.
    model = NMF(
        n_components=cast(Any, k),
        init="nndsvda",
        max_iter=1000,
        random_state=random_state,
    )
    W_tr = np.asarray(model.fit_transform(tr_nn), dtype=float)
    W_te = np.asarray(model.transform(te_nn), dtype=float)

    cols = [f"NMF{i + 1}" for i in range(k)]
    return (
        pd.DataFrame(W_tr, index=X_train.index, columns=pd.Index(cols)),
        pd.DataFrame(W_te, index=X_test.index, columns=pd.Index(cols)),
        model,
    )
