"""The AUC curve policy in src.load.load_auc_long (team decision 2026-11-19).

    python -m pytest tests/test_auc_curves.py -q
"""

from __future__ import annotations

import pytest

from src.load import load_auc_long


@pytest.fixture(scope="module")
def raw():
    return load_auc_long(curves="all", converged_only=False)


def test_clean_drops_only_dip_then_rise(raw):
    clean = load_auc_long()                       # default policy
    assert (clean.curve_type.eq("decreasing") | clean.all_gt_50.astype(str).str.upper().eq("TRUE")).all()
    inc = clean[clean.curve_type.eq("increasing")]
    assert len(inc) > 0, "rising curves that never fell below 50% must be kept"
    assert inc.all_gt_50.astype(str).str.upper().eq("TRUE").all()


def test_policy_counts(raw):
    assert len(raw) == 63_395
    assert len(load_auc_long(curves="decreasing")) < len(load_auc_long()) < len(load_auc_long(curves="all"))


def test_legacy_keyword_still_works():
    assert len(load_auc_long(decreasing_only=False, converged_only=False)) == 63_395
    assert len(load_auc_long(decreasing_only=True)) == len(load_auc_long(curves="decreasing"))


def test_bad_policy_rejected():
    with pytest.raises(ValueError):
        load_auc_long(curves="rising")
