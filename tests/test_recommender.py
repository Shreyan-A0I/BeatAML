"""Tests for the drug recommender: metrics, feature replication, leakage, baselines.

    python -m pytest tests/ -q

Data-dependent tests need data/raw (see README). They load the n = 429
modeling cohort once per session (~10 s).
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import pandas as pd
import pytest

from src.features import PAPER_MODULES, alteration_table
from src.load import COLS, load_pc1_baseline
from src.models import GenomicRuleBaseline, PopulationBaseline
from src.rank_eval import per_patient_metrics
from src.run_baselines import cross_validate
from src.targets import DrugZScaler, load_recommender_data, select_drugs

# ---------------------------------------------------------------- metrics (no data)


def _toy(n_pat=4, n_drug=30, seed=0):
    rng = np.random.default_rng(seed)
    z = pd.DataFrame(rng.normal(size=(n_pat, n_drug)), columns=[f"d{i}" for i in range(n_drug)])
    return z


def test_perfect_ranking_scores_perfectly():
    z = _toy()
    m = per_patient_metrics(score=z, z=z)
    assert np.allclose(m["spearman"], 1.0)
    assert np.allclose(m["regret"], 0.0)
    assert np.allclose(m["ndcg@10"], 1.0)
    n_sens = (z < -0.84).sum(axis=1).to_numpy()
    assert np.allclose(m["prec@5"], np.minimum(n_sens, 5) / 5)


def test_reversed_ranking_scores_worst():
    z = _toy()
    m = per_patient_metrics(score=-z, z=z)
    assert np.allclose(m["spearman"], -1.0)
    assert (m["regret"] > 0).all()


def test_untested_drugs_are_ignored():
    z = _toy()
    z_missing = z.copy()
    z_missing.iloc[:, :5] = np.nan              # drugs not tested on anyone
    m = per_patient_metrics(score=z, z=z_missing)
    assert (m["n_drugs"] == z.shape[1] - 5).all()


# ---------------------------------------------------------------- data fixtures


@pytest.fixture(scope="session")
def data():
    return load_recommender_data()


@pytest.fixture(scope="session")
def feats(data):
    return data.features()


@pytest.fixture(scope="session")
def drugs(data):
    return select_drugs(data.auc, data.train)


# ---------------------------------------------------------------- cohort and features


def test_cohort_split_is_disjoint_and_complete(data):
    assert not (data.train & data.test).any()
    assert (data.train | data.test).all()
    # paper curve convention ("all"): BA2563 (dip-then-rise only) is included -> 267/163
    assert (data.train.sum(), data.test.sum()) == (267, 163)
    assert data.auc.index.equals(data.patients)


def test_celltype_scores_reproduce_published(data, feats):
    pub = load_pc1_baseline().reindex(data.meta[COLS["rna_id"]]).set_axis(data.patients)
    for ct in pub.columns:
        ours = feats[f"ct.{ct}"]
        assert np.corrcoef(ours, pub[ct])[0, 1] > 0.99999, ct
        assert np.abs(ours - pub[ct]).max() < 1e-6, ct


def test_paper_modules_used(feats):
    mods = [c for c in feats.columns if c.startswith("me.")]
    assert mods == [f"me.{m}" for m in PAPER_MODULES]


def test_reference_set_excludes_waves34(data):
    ref = set(data.ref_meta["dbgap_subject_id"])
    assert len(ref) == 372
    assert not ref & set(data.patients[data.test])
    assert set(data.patients[data.train]) <= ref        # all 266 modeling-train patients are in it


def test_features_of_training_patients_ignore_test_expression(data, feats):
    """Scramble every Waves3+4 expression value: training rows must not move."""
    rng = np.random.default_rng(1)
    X2 = data.X.copy()
    test_rows = np.flatnonzero(data.test)
    X2.iloc[test_rows] = X2.iloc[test_rows].to_numpy() * rng.uniform(0.5, 1.5, X2.iloc[test_rows].shape)
    f2 = data.features(X=X2)
    assert list(f2.columns) == list(feats.columns)          # same selected features
    tr = data.patients[data.train]
    pd.testing.assert_frame_equal(f2.loc[tr], feats.loc[tr])


def test_test_patient_features_are_independent_of_each_other(data, feats):
    """Perturb ONE test patient's expression: only that patient's row changes."""
    X2 = data.X.copy()
    i = int(np.flatnonzero(data.test)[0])
    X2.iloc[i] = X2.iloc[i] + 1.0
    f2 = data.features(X=X2)
    changed = ~np.isclose(f2.to_numpy(float), feats.to_numpy(float), equal_nan=True).all(axis=1)
    assert list(data.patients[changed]) == [data.patients[i]]


# ---------------------------------------------------------------- leakage in evaluation


def test_z_scaler_uses_training_auc_only(data, drugs):
    tr, te = data.patients[data.train], data.patients[data.test]
    s1 = DrugZScaler().fit(data.auc.loc[tr, drugs])
    auc2 = data.auc.copy()
    auc2.loc[te, drugs] = auc2.loc[te, drugs] * 3
    s2 = DrugZScaler().fit(auc2.loc[tr, drugs])
    pd.testing.assert_series_equal(s1.mean_, s2.mean_)
    pd.testing.assert_series_equal(s1.sd_, s2.sd_)


class _Spy(PopulationBaseline):
    """Records the patients every instance is fit and asked to predict on."""

    fitted: ClassVar[list[set]] = []
    predicted: ClassVar[list[set]] = []

    def fit(self, auc_train, features_train=None):
        _Spy.fitted.append(set(auc_train.index))
        return super().fit(auc_train, features_train)

    def predict(self, features, drugs):
        _Spy.predicted.append(set(features.index))
        return super().predict(features, drugs)


def test_cross_validation_never_touches_waves34(data, drugs):
    _Spy.fitted.clear(); _Spy.predicted.clear()
    alts = alteration_table(data.clinical, data.patients)
    res = cross_validate(data, alts, drugs, models={"spy": _Spy}, n_splits=5)
    holdout = set(data.patients[data.test])
    train = set(data.patients[data.train])
    assert len(_Spy.fitted) == 5
    for fit_ids, pred_ids in zip(_Spy.fitted, _Spy.predicted):
        assert not fit_ids & holdout                      # no Waves3+4 patient in any fit
        assert not pred_ids & holdout
        assert not fit_ids & pred_ids                     # fold-train and fold-test disjoint
    assert set(res["spy"].index) <= train
    assert res["spy"].index.is_unique                     # each patient scored once


# ---------------------------------------------------------------- baselines


def test_b0_gives_everyone_the_same_ranking(data, drugs):
    tr = data.patients[data.train]
    alts = alteration_table(data.clinical, data.patients)
    s = PopulationBaseline().fit(data.auc.loc[tr, drugs]).predict(alts.loc[tr[:5]], drugs)
    assert (s.nunique(axis=0) == 1).all()


def test_b1_puts_rule_drugs_first_for_flt3_itd(data, drugs):
    tr = data.patients[data.train]
    alts = alteration_table(data.clinical, data.patients)
    b1 = GenomicRuleBaseline().fit(data.auc.loc[tr, drugs])
    carrier = alts.index[alts["alt.FLT3_ITD"].fillna(0) > 0][0]
    plain = alts.index[alts.filter(like="alt.").fillna(0).sum(axis=1) == 0][0]
    s = b1.predict(alts.loc[[carrier, plain]], drugs)
    flt3_drugs = b1.rule_drugs_["FLT3_ITD"]
    top = s.loc[carrier].sort_values().index[: len(flt3_drugs)]
    assert set(top) == flt3_drugs
    # a patient with no alterations gets exactly B0's order
    b0 = PopulationBaseline().fit(data.auc.loc[tr, drugs]).predict(alts.loc[[plain]], drugs)
    assert list(s.loc[plain].sort_values().index) == list(b0.loc[plain].sort_values(kind="stable").index)


# ---------------------------------------------------------------- learned models


def _train_split(data, drugs, n_tr=180):
    from src.features import alteration_table as _alt
    tr = data.patients[data.train]
    return tr[:n_tr], tr[n_tr:], _alt


def test_learned_models_ignore_test_auc(data, feats, drugs):
    """M1/M2 predictions for held-out patients must not change if held-out AUC changes."""
    from src.models import PerDrugRidge, PooledXGB
    ptr, pte, _alt = _train_split(data, drugs)
    F = feats.join(_alt(data.clinical, data.patients))
    for make in (lambda: PerDrugRidge(), lambda: PooledXGB(n_estimators=50)):
        a = make().fit(data.auc.loc[ptr, drugs], F.loc[ptr]).predict(F.loc[pte], drugs)
        auc2 = data.auc.copy()
        auc2.loc[pte, drugs] = auc2.loc[pte, drugs] * 3
        b = make().fit(auc2.loc[ptr, drugs], F.loc[ptr]).predict(F.loc[pte], drugs)
        pd.testing.assert_frame_equal(a, b)


def test_m1_learns_and_shuffled_m1_does_not(data, feats, drugs):
    from src.models import PerDrugRidge
    ptr, pte, _alt = _train_split(data, drugs)
    z = DrugZScaler().fit(data.auc.loc[ptr, drugs]).transform(data.auc.loc[pte, drugs])
    real = PerDrugRidge().fit(data.auc.loc[ptr, drugs], feats.loc[ptr]).predict(feats.loc[pte], drugs)
    shuf = PerDrugRidge(shuffle_seed=0).fit(data.auc.loc[ptr, drugs], feats.loc[ptr]).predict(feats.loc[pte], drugs)
    m_real = per_patient_metrics(real, z)["spearman"].mean()
    m_shuf = per_patient_metrics(shuf, z)["spearman"].mean()
    assert m_real > 0.1
    assert abs(m_shuf) < 0.08
