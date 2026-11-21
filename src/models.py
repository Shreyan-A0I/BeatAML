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


# --------------------------------------------------------------------------- learned models
#
# Both learn the drug-standardised z (fit inside `fit` on the training AUC only)
# and return predicted z as the score, so lower = recommend first.

BLOCKS_ALL = ("ct", "me", "rna", "clin", "mut", "fus")
GENOMIC = ("mut", "fus")


class _FeaturePrep:
    """Select feature blocks by prefix and impute with training statistics.

    mut.*/fus.* are NaN for patients without DNA-seq: filled with 0 plus a
    `has_dna` indicator. Other NaNs (age, de novo) get the training median.
    """

    def __init__(self, blocks: Sequence[str]):
        self.blocks = tuple(blocks)

    def fit(self, feats: pd.DataFrame) -> _FeaturePrep:
        self.cols_ = [c for c in feats.columns if c.split(".", 1)[0] in self.blocks]
        self.genomic_ = [c for c in self.cols_ if c.split(".", 1)[0] in GENOMIC]
        self.median_ = feats[self.cols_].median()
        return self

    def transform(self, feats: pd.DataFrame) -> pd.DataFrame:
        F = feats.reindex(columns=self.cols_).astype(float)
        if self.genomic_:
            F["has_dna"] = F[self.genomic_].notna().any(axis=1).astype(float)
            F[self.genomic_] = F[self.genomic_].fillna(0.0)
        return F.fillna(self.median_)


class PerDrugRidge:
    """M1: one ridge regression per drug, predicting that drug's z."""

    def __init__(self, blocks: Sequence[str] = BLOCKS_ALL, alphas: np.ndarray | None = None,
                 min_n: int = 30, shuffle_seed: int | None = None):
        self.blocks = blocks
        self.alphas = np.logspace(-1, 4, 21) if alphas is None else alphas
        self.min_n = min_n
        self.shuffle_seed = shuffle_seed     # sanity check: permute patients' targets per drug

    def fit(self, auc_train: pd.DataFrame, features_train: pd.DataFrame) -> PerDrugRidge:
        from sklearn.linear_model import RidgeCV
        from sklearn.preprocessing import StandardScaler

        from .targets import DrugZScaler

        z = DrugZScaler().fit_transform(auc_train)
        self.prep_ = _FeaturePrep(self.blocks).fit(features_train.loc[auc_train.index])
        F = self.prep_.transform(features_train.loc[auc_train.index])
        self.scaler_ = StandardScaler().fit(F)
        Fs = pd.DataFrame(self.scaler_.transform(F), index=F.index).fillna(0.0)
        rng = np.random.default_rng(self.shuffle_seed)
        self.models_ = {}
        for d in z.columns:
            y = z[d].dropna()
            if len(y) < self.min_n:
                continue
            yv = rng.permutation(y.to_numpy()) if self.shuffle_seed is not None else y.to_numpy()
            self.models_[d] = RidgeCV(alphas=self.alphas).fit(Fs.loc[y.index].to_numpy(), yv)
        return self

    def predict(self, features: pd.DataFrame, drugs: Sequence[str]) -> pd.DataFrame:
        F = self.prep_.transform(features)
        Fs = np.nan_to_num(self.scaler_.transform(F))
        out = pd.DataFrame(0.0, index=features.index, columns=list(drugs))
        for d in drugs:
            if d in self.models_:
                out[d] = self.models_[d].predict(Fs)
        return out


class PooledXGB:
    """M2: one gradient-boosted model over all (patient, drug) pairs.

    Row = patient features + drug descriptors: drug id one-hot, kinase-family
    membership (families with >= 3 panel drugs), and `target_mutated` = the
    patient carries an alteration in one of the drug's curated target genes.
    Sharing across drugs lets trees learn mutation x maturation x drug-class
    interactions from ~20k training pairs instead of ~200 per drug.
    """

    def __init__(self, blocks: Sequence[str] = BLOCKS_ALL, n_estimators: int = 400, max_depth: int = 4,
                 learning_rate: float = 0.05, seed: int = 0):
        self.blocks = blocks
        self.params = dict(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate,
                           subsample=0.8, colsample_bytree=0.8, min_child_weight=20, reg_lambda=5.0,
                           random_state=seed, n_jobs=4, tree_method="hist")

    def _drug_table(self, drugs: Sequence[str]) -> pd.DataFrame:
        fam = load_drug_families()["drug_family"]
        fam = fam[fam["inhibitor"].isin(drugs)]
        big = fam["family"].value_counts()
        fam = fam[fam["family"].isin(big[big >= 3].index)]
        D = pd.crosstab(fam["inhibitor"], fam["family"]).clip(upper=1).reindex(list(drugs), fill_value=0)
        D.columns = [f"fam.{i}" for i in range(D.shape[1])]
        ids = pd.get_dummies(pd.Series(list(drugs), index=list(drugs)), prefix="drug").astype(float)
        return pd.concat([ids, D.astype(float)], axis=1)

    def _target_hits(self, features: pd.DataFrame, drugs: Sequence[str]) -> pd.DataFrame:
        """Patients x drugs: number of the drug's target genes the patient has altered."""
        dg = load_drug_families()["drug_gene"]
        out = pd.DataFrame(0.0, index=features.index, columns=list(drugs))
        alt_cols = {c[4:]: c for c in features.columns if c.startswith("alt.")}
        alt_cols.setdefault("FLT3", "alt.FLT3")
        for d in drugs:
            for g in dg.loc[dg["inhibitor"].eq(d), "Symbol"].unique():
                cols = [c for k, c in alt_cols.items() if (k == g or k.startswith(g + "_")) and c in features]
                for c in cols:
                    out[d] += features[c].fillna(0).to_numpy(float) > 0
        return out

    def _rows(self, features: pd.DataFrame, pairs: pd.DataFrame) -> np.ndarray:
        P = self.prep_.transform(features)
        hits = self._target_hits(features, self.drugs_)
        pidx = features.index.get_indexer(pairs["patient"])
        didx = pd.Index(self.drugs_).get_indexer(pairs["drug"])
        th = hits.to_numpy()[pidx, didx][:, None]
        return np.hstack([P.to_numpy()[pidx], self.dt_.to_numpy()[didx], th]).astype(np.float32)

    def fit(self, auc_train: pd.DataFrame, features_train: pd.DataFrame) -> PooledXGB:
        from xgboost import XGBRegressor

        from .targets import DrugZScaler

        self.drugs_ = list(auc_train.columns)
        self.dt_ = self._drug_table(self.drugs_)
        ft = features_train.loc[auc_train.index]
        self.prep_ = _FeaturePrep(self.blocks).fit(ft)
        z = DrugZScaler().fit_transform(auc_train)
        long = z.stack().rename("z").reset_index()
        long.columns = ["patient", "drug", "z"]
        X = self._rows(ft, long)
        self.feature_names_ = (list(self.prep_.transform(ft.iloc[:1]).columns) + list(self.dt_.columns)
                               + ["target_mutated"])
        self.model_ = XGBRegressor(**self.params).fit(X, long["z"].to_numpy())
        return self

    def predict(self, features: pd.DataFrame, drugs: Sequence[str]) -> pd.DataFrame:
        known = [d for d in drugs if d in self.drugs_]
        pairs = pd.MultiIndex.from_product([features.index, known], names=["patient", "drug"]).to_frame(index=False)
        pred = self.model_.predict(self._rows(features, pairs))
        out = pd.Series(pred, index=pd.MultiIndex.from_frame(pairs)).unstack("drug")
        return out.reindex(index=features.index, columns=list(drugs)).fillna(0.0)
