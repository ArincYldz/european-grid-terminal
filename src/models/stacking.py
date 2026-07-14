"""OUT-OF-FOLD prediction generation for model cascades (stacking).

The technique that PREVENTS leakage when feeding one model's (Step 3A
generation) output as a feature to another (Step 3B price). In interviews it
comes up as "leakage in stacking" or "target leakage via model cascade".
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from .generation_model import GenerationForecaster, _feature_columns

logger = logging.getLogger(__name__)


def oof_generation_predictions(
    feat: pd.DataFrame,
    target: str = "generation_mw",
    n_splits: int = 5,
    params: dict | None = None,
) -> pd.Series:
    """Produces temporal OOF (out-of-fold) generation predictions.

    Why OOF instead of in-sample?
      A model is overly optimistic on its own training data. If we train 3A on
      all data, predict on the SAME data, and feed that prediction to 3B, then
      3B gets used to a generation input far 'cleaner' than it will EVER see in
      production -> the price model collapses live. An OOF prediction is the
      prediction of a model trained WITHOUT seeing that row; it represents live
      conditions correctly.

    We use TimeSeriesSplit (not KFold) — even in a cascade the time direction
    must be preserved: no training on the future to predict the past.

    Note: in TimeSeriesSplit the rows BEFORE the first fold have no training
    data, so they get no OOF prediction (stay NaN). This is natural; we drop
    that leading part during 3B training.
    """
    df = feat.sort_values("target_time").reset_index(drop=True)
    feats = _feature_columns(df)
    oof = pd.Series(np.nan, index=df.index, name="predicted_generation_mw")

    tscv = TimeSeriesSplit(n_splits=n_splits)
    for fold, (tr, te) in enumerate(tscv.split(df), start=1):
        model = GenerationForecaster(params)
        model.fit(df.iloc[tr])
        oof.iloc[te] = model.predict(df.iloc[te])
        logger.info("OOF fold %d: %d rows predicted.", fold, len(te))

    n_valid = int(oof.notna().sum())
    logger.info("OOF generation prediction ready: %d/%d rows filled.", n_valid, len(oof))
    return oof
