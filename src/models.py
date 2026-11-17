"""Drug recommenders. Every model shares one interface:

    model.fit(auc_train, features_train)   -> self
    model.predict(features, drugs)         -> DataFrame (patients x drugs) of SCORES,
                                              LOWER = recommend first

`auc_train` is the raw AUC matrix of the training patients (NaN = not tested).
`features` is a patient-indexed table from src.features (build_patient_features
blocks plus, for rule models, the `alt.*` columns from alteration_table()).

Baselines:
    RandomRanker          seeded random order: measures chance
    PopulationBaseline    B0: same list for everyone, most potent drug (lowest mean AUC) first
    GenomicRuleBaseline   B1: clinician-style lookup, alteration -> targeted drugs first,
                          then B0 order
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from .load import load_drug_families

# B1 rules: alteration (alt.* column without prefix) -> drug targets.
# "targets" are gene symbols looked up in the curated drug_gene sheet (the
# paper's high-confidence drug-target table); "drugs" names inhibitors directly
# for targets that sheet does not cover (the IDH inhibitors on the panel).
# Kept to well-established, guideline-level actionable lesions in AML.
DEFAULT_RULES: dict[str, dict[str, set[str]]] = {
    "FLT3_ITD": {"targets": {"FLT3"}},
    "FLT3":     {"targets": {"FLT3"}},              # TKD and other FLT3 mutations
    "KIT":      {"targets": {"KIT"}},
    "NRAS":     {"targets": {"MAP2K1", "MAP2K2"}},  # RAS pathway -> MEK inhibitors
    "KRAS":     {"targets": {"MAP2K1", "MAP2K2"}},
    "PTPN11":   {"targets": {"MAP2K1", "MAP2K2"}},
    "JAK2":     {"targets": {"JAK2"}},
    "BCR-ABL1": {"targets": {"ABL1"}},
    "IDH1":     {"drugs": {"AGI-5198"}},
    "IDH2":     {"drugs": {"AGI-6780"}},
}


def _broadcast(values: pd.Series, patients: pd.Index, drugs: Sequence[str]) -> pd.DataFrame:
    row = values.reindex(list(drugs)).to_numpy(float)
    return pd.DataFrame(np.tile(row, (len(patients), 1)), index=patients, columns=list(drugs))


class RandomRanker:
    """Random scores. Its expected prec@5 equals each patient's prevalence."""

    def __init__(self, seed: int = 0):
        self.seed = seed

    def fit(self, auc_train: pd.DataFrame, features_train: pd.DataFrame | None = None) -> RandomRanker:
        return self

    def predict(self, features: pd.DataFrame, drugs: Sequence[str]) -> pd.DataFrame:
        rng = np.random.default_rng(self.seed)
        return pd.DataFrame(rng.random((len(features), len(drugs))), index=features.index, columns=list(drugs))


class PopulationBaseline:
    """B0: rank every patient's drugs by the training mean raw AUC.

    This is "use the drugs that work best on average". It is the right bar for
    absolute AUC, but against the drug-standardised target it should be near
    chance, because every drug's training mean z is 0 by construction.
    """

    def fit(self, auc_train: pd.DataFrame, features_train: pd.DataFrame | None = None) -> PopulationBaseline:
        self.mean_auc_ = auc_train.mean()
        return self

    def predict(self, features: pd.DataFrame, drugs: Sequence[str]) -> pd.DataFrame:
        return _broadcast(self.mean_auc_, features.index, drugs)


class GenomicRuleBaseline:
    """B1: drugs targeting one of the patient's actionable alterations go first.

    Score = -(number of the patient's alterations the drug targets) + a
    tie-breaker in [0, 0.5) from the fallback model's order. A patient with no
    actionable alteration (or no DNA-seq) gets exactly the fallback ranking.
    """

    def __init__(self, rules: dict[str, dict[str, set[str]]] | None = None, fallback=None):
        self.rules = DEFAULT_RULES if rules is None else rules
        self.fallback = PopulationBaseline() if fallback is None else fallback

    def fit(self, auc_train: pd.DataFrame, features_train: pd.DataFrame | None = None) -> GenomicRuleBaseline:
        self.fallback.fit(auc_train, features_train)
        dg = load_drug_families()["drug_gene"]
        drugs = set(auc_train.columns)
        self.rule_drugs_: dict[str, set[str]] = {}
        for alt, rule in self.rules.items():
            hit = set(dg.loc[dg["Symbol"].isin(rule.get("targets", set())), "inhibitor"])
            hit |= set(rule.get("drugs", set()))
            self.rule_drugs_[alt] = hit & drugs
        return self

    def matched_drugs(self, features: pd.DataFrame) -> pd.DataFrame:
        """Patients x drugs count of matched rules (0 for no DNA-seq)."""
        drugs = sorted(set().union(*self.rule_drugs_.values()))
        hits = pd.DataFrame(0, index=features.index, columns=drugs, dtype=float)
        for alt, rdrugs in self.rule_drugs_.items():
            col = f"alt.{alt}"
            if col not in features.columns or not rdrugs:
                continue
            carrier = features[col].fillna(0).to_numpy(float) > 0
            hits.loc[carrier, sorted(rdrugs)] += 1
        return hits

    def predict(self, features: pd.DataFrame, drugs: Sequence[str]) -> pd.DataFrame:
        base = self.fallback.predict(features, drugs)
        tie = base.rank(axis=1, method="first").div(len(drugs) + 1) * 0.5
        hits = self.matched_drugs(features).reindex(columns=list(drugs), fill_value=0)
        return tie - hits
