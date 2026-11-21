"""Targets for the per-patient drug recommender.

THE DESIGN DECISION
-------------------
Raw ex vivo AUC mixes three things:

    AUC[patient, drug] = drug potency          (some drugs kill every sample at the tested doses)
                       + patient's general sensitivity
                       + patient-specific, drug-specific response   <- what a recommender should find

Ranking drugs by raw AUC mostly re-discovers drug potency, which is the same for
every patient and partly an artefact of each drug's tested concentration range.
So the target is the DRUG-STANDARDISED AUC:

    z[patient, drug] = (AUC - mean_drug) / sd_drug        (mean/sd from TRAINING patients only)

z < 0 means "this patient's cells are more sensitive to this drug than a typical
AML sample". This is the same idea as "selective" drug sensitivity scores in
functional precision medicine.

THE "TRULY SENSITIVE" CUTOFF (a project choice, not from the paper)
-------------------------------------------------------------------
A patient counts as truly sensitive to a drug when z < SENSITIVE_Z = -0.84.
-0.84 is the 20th percentile of the standard normal, so the label reads "among
the most sensitive ~20% of patients for this drug". Why 20%:
  * strict enough that a positive is clearly better than typical, not noise
    around the mean;
  * loose enough that each patient has ~20 positives among ~116 drugs, so
    precision@5 can distinguish rankings at all;
  * observed share on Waves1+2 is ~18% (AUC is not exactly normal: ceiling
    effects pile curves up at high AUC), close to the nominal 20%.
Bottomly et al. use red/white/blue only as a colour scale and define no numeric
cutoff. Threshold-free metrics (Spearman, NDCG on graded z) are reported
alongside, so conclusions do not rest on this single value.

Consequence worth knowing: on the training data every drug's mean z is 0 by
construction, so "rank drugs by average potency" carries no information about z.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .features import build_patient_features, split_by_cohort
from .load import COLS, build_modeling_frame, load_auc_long

SENSITIVE_Z = -0.84          # 20th percentile of N(0,1): most sensitive ~20% of patients (see module docstring)
MIN_TRAIN = 60               # a drug needs this many tested Waves1+2 patients to be modelled
USE_STATUS = ("train/test", "combined")   # the paper's analysed drugs; "exploratory" excluded


@dataclass
class RecommenderData:
    """Everything the recommender needs, keyed by patient (dbgap_subject_id)."""

    meta: pd.DataFrame      # modeling rows (one per patient), indexed by patient
    X: pd.DataFrame         # specimens x genes expression, row-aligned to `meta`
    clinical: pd.DataFrame  # full cleaned clinical sheet (for DNA / fusion specimens)
    auc: pd.DataFrame       # patients x drugs raw AUC, NaN = not tested
    train: np.ndarray       # bool over meta rows: Waves1+2
    test: np.ndarray        # bool over meta rows: Waves3+4
    ref_X: pd.DataFrame     # all Waves1+2 RNA-seq specimens (372) x genes: fits PC1 features
    ref_meta: pd.DataFrame  # clinical rows row-aligned to ref_X

    @property
    def patients(self) -> pd.Index:
        return self.meta.index

    def features(self, X: pd.DataFrame | None = None, **kwargs) -> pd.DataFrame:
        """Paper features for the modeling patients (see build_patient_features).

        Expression features are fit on the 372 Waves1+2 RNA-seq specimens, which
        reproduces the published scores exactly and contains no Waves3+4 patient.
        Pass `X` to substitute the modeling expression (e.g. for leakage tests).
        """
        return build_patient_features(self.meta, self.X if X is None else X, self.clinical,
                                      self.train, ref_X=self.ref_X, ref_meta=self.ref_meta, **kwargs)


def load_recommender_data(statuses: Iterable[str] = USE_STATUS) -> RecommenderData:
    """Modeling cohort (n = 429) with a patient x drug raw AUC matrix.

    AUC rows keep `load_auc_long()`'s defaults (curves="all" = paper convention, converged) and
    the paper's drug statuses. Patients are those of build_modeling_frame():
    RNA-seq AND drug screen, one specimen of each per patient.
    """
    frame = build_modeling_frame(verbose=False)
    meta = frame["meta"]
    pid = COLS["patient_id"]
    assert meta[pid].is_unique, "expected one modeling specimen per patient"

    long = load_auc_long()
    long = long[long["status"].isin(list(statuses))]
    auc = long.pivot_table(index=COLS["drug_id"], columns=COLS["inhibitor"],
                           values=COLS["auc"], aggfunc="mean")
    auc = auc.reindex(meta[COLS["drug_id"]]).set_axis(meta[pid].to_numpy(), axis=0)
    auc.index.name = pid
    auc.columns.name = None

    train, test = split_by_cohort(meta)

    clin = frame["clinical"]
    ref_meta = clin[clin["manuscript_rnaseq"].eq("yes") & clin[COLS["cohort"]].eq("Waves1+2")]
    ref_meta = ref_meta.drop_duplicates(pid)
    ref_X = frame["expr"][ref_meta[COLS["rna_id"]].tolist()].T
    assert not np.isin(ref_meta[pid], meta.loc[test, pid]).any(), "Waves3+4 patient in the reference set"

    return RecommenderData(
        meta=meta.set_index(pid, drop=False),
        X=frame["X"],
        clinical=clin,
        auc=auc,
        train=train,
        test=test,
        ref_X=ref_X,
        ref_meta=ref_meta,
    )


def select_drugs(auc: pd.DataFrame, train: np.ndarray, min_train: int = MIN_TRAIN) -> list[str]:
    """Drugs tested on at least `min_train` training patients.

    Uses only how many patients were TESTED, never their AUC values, so it is
    safe to compute once on the whole training cohort.
    """
    n = auc.loc[train].notna().sum()
    return sorted(n[n >= min_train].index)


class DrugZScaler:
    """Per-drug standardisation fit on training patients only."""

    def fit(self, auc_train: pd.DataFrame) -> DrugZScaler:
        self.mean_ = auc_train.mean()
        self.sd_ = auc_train.std(ddof=1).replace(0, np.nan)
        return self

    def transform(self, auc: pd.DataFrame) -> pd.DataFrame:
        cols = auc.columns.intersection(self.mean_.index)
        return (auc[cols] - self.mean_[cols]) / self.sd_[cols]

    def fit_transform(self, auc_train: pd.DataFrame) -> pd.DataFrame:
        return self.fit(auc_train).transform(auc_train)


def sensitive(z: pd.DataFrame, threshold: float = SENSITIVE_Z) -> pd.DataFrame:
    """1.0 where truly sensitive, 0.0 where tested but not, NaN where not tested."""
    return (z < threshold).astype(float).where(z.notna())


def to_long(auc: pd.DataFrame, z: pd.DataFrame | None = None) -> pd.DataFrame:
    """Long format (patient, drug, auc[, z]) over tested pairs only, for pooled models."""
    out = auc.stack().rename("auc").reset_index()
    out.columns = [COLS["patient_id"], "drug", "auc"]
    if z is not None:
        zl = z.stack().rename("z").reset_index()
        zl.columns = [COLS["patient_id"], "drug", "z"]
        out = out.merge(zl, on=[COLS["patient_id"], "drug"], how="left")
    return out
