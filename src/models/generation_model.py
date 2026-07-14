"""LightGBM-based generation forecast model + time-series-appropriate validation.

Interview axis: "Why LightGBM and why TimeSeriesSplit?"
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

logger = logging.getLogger(__name__)

# Columns that must NEVER be given to the model as input (timestamps, target,
# raw text). Accidentally turning these into features is also a leakage risk.
_NON_FEATURE_COLS = {
    "target_time", "available_at", "source", "generation_mw", "is_imputed",
}


def _feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in _NON_FEATURE_COLS and df[c].dtype != "object"]


def time_series_cv_score(
    df: pd.DataFrame,
    target: str = "generation_mw",
    n_splits: int = 5,
    params: dict | None = None,
) -> dict:
    """Forward-validation (walk-forward) CV with TimeSeriesSplit.

    Why TimeSeriesSplit, not KFold(shuffle=True)?
      Standard k-fold shuffles the data randomly; this means training on
      FUTURE rows and testing on PAST ones = leakage. Also, since consecutive
      hours are nearly identical (autocorrelation), shuffled CV mistakes
      memorization for "generalization" and grossly inflates the score.
      TimeSeriesSplit always trains on the PAST and tests on the FUTURE; it is
      the correct simulation of live use.
    """
    params = params or _default_params()
    df = df.sort_values("target_time").reset_index(drop=True)
    feats = _feature_columns(df)
    X, y = df[feats], df[target]

    tscv = TimeSeriesSplit(n_splits=n_splits)
    maes, rmses = [], []
    for fold, (tr, te) in enumerate(tscv.split(X), start=1):
        model = LGBMRegressor(**params)
        model.fit(X.iloc[tr], y.iloc[tr])
        pred = model.predict(X.iloc[te])
        mae = mean_absolute_error(y.iloc[te], pred)
        rmse = float(np.sqrt(mean_squared_error(y.iloc[te], pred)))
        maes.append(mae)
        rmses.append(rmse)
        logger.info("Fold %d — MAE %.1f MW | RMSE %.1f MW (n_test=%d)", fold, mae, rmse, len(te))

    return {
        "mae_mean": float(np.mean(maes)),
        "mae_std": float(np.std(maes)),
        "rmse_mean": float(np.mean(rmses)),
        "n_features": len(feats),
        "n_splits": n_splits,
    }


def _default_params() -> dict:
    return dict(
        n_estimators=600,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=30,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )


class GenerationForecaster:
    """Train -> predict -> report feature importance.

    Why LightGBM (the spec says XGBoost/LightGBM; we chose LightGBM)?
      - Fast and strong on tabular, mixed-scale weather features.
      - Handles missing values (NaN) NATIVELY — a perfect fit for our Step 1
        "don't fill, leave NaN" policy: it learns NaN as a separate branch, we
        inject no fabricated values.
      - Supports categorical/monotonic constraints and quantile loss (will be
        useful in Step 3B for P10/P90 price forecasting).
      - Generally faster training than XGBoost with comparable accuracy.
    """

    def __init__(self, params: dict | None = None):
        self.params = params or _default_params()
        self.model: LGBMRegressor | None = None
        self.features: list[str] = []

    def fit(self, df: pd.DataFrame, target: str = "generation_mw", valid_fraction: float = 0.2):
        df = df.sort_values("target_time").reset_index(drop=True)
        self.features = _feature_columns(df)

        # For early stopping we split off the validation set TEMPORALLY from
        # the last 20% — never randomly (leakage + autocorrelation).
        cut = int(len(df) * (1 - valid_fraction))
        tr, va = df.iloc[:cut], df.iloc[cut:]

        self.model = LGBMRegressor(**self.params)
        self.model.fit(
            tr[self.features], tr[target],
            eval_set=[(va[self.features], va[target])],
            eval_metric="l1",
            callbacks=[early_stopping(50, verbose=False), log_evaluation(0)],
        )
        logger.info(
            "Model trained. Best iteration: %s | number of features: %d",
            self.model.best_iteration_, len(self.features),
        )
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Call fit() first.")
        # Generation cannot be negative — enforce the physical constraint.
        return np.clip(self.model.predict(df[self.features]), 0, None)

    def feature_importance(self, top: int = 15) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError("Call fit() first.")
        imp = pd.DataFrame(
            {"feature": self.features, "gain": self.model.booster_.feature_importance("gain")}
        )
        return imp.sort_values("gain", ascending=False).head(top).reset_index(drop=True)
