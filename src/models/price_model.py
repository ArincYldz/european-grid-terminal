"""Price forecasting (Step 3B): quantile regression + calibrated negative-price risk.

Two separate outputs, two separate interview rationales:
  1. QuantileForecaster -> the price DISTRIBUTION (P10/P50/P90), not a point.
  2. NegativePriceClassifier -> a CALIBRATED negative-price probability (risk score).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss

logger = logging.getLogger(__name__)

_NON_FEATURE_COLS = {
    "target_time", "available_at", "source", "is_imputed",
    "price_eur_mwh", "generation_mw", "demand_mw",  # targets / future realized values
}


def price_feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in _NON_FEATURE_COLS and df[c].dtype != "object"]


def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, alpha: float) -> float:
    """Quantile (pinball) loss — the correct metric for a quantile forecast.

    MSE is symmetric; it penalizes under/over-prediction equally. Pinball is
    ASYMMETRIC: for alpha=0.9 it penalizes under-prediction heavily and
    over-prediction lightly; so the model places the 90th-percentile upper
    bound in the right spot. In risk management this penalty asymmetry is
    exactly what we want.
    """
    diff = y_true - y_pred
    return float(np.mean(np.maximum(alpha * diff, (alpha - 1) * diff)))


class QuantileForecaster:
    """Predicts the P10/P50/P90 bands of price (a distributional forecast).

    Why NOT a point forecast?
      The input to a trading decision is uncertainty. A point forecast says
      "price 45 EUR" and hides the negative-price risk (the tail). Saying
      P10 = -15, P50 = 42, P90 = 95 conveys "most likely 42, but 10% chance of
      negative". The Step 4 decision engine takes risk-adjusted positions with
      these bands.

    LightGBM learns each quantile as a SEPARATE model with
    objective='quantile', alpha=q.
    """

    def __init__(self, quantiles=(0.1, 0.5, 0.9), params: dict | None = None):
        self.quantiles = tuple(quantiles)
        self.params = params or dict(
            n_estimators=500, learning_rate=0.05, num_leaves=31,
            subsample=0.8, colsample_bytree=0.8, min_child_samples=40,
            random_state=42, n_jobs=-1, verbose=-1,
        )
        self.models: dict[float, LGBMRegressor] = {}
        self.features: list[str] = []

    def fit(self, df: pd.DataFrame, target: str = "price_eur_mwh"):
        df = df.sort_values("target_time").reset_index(drop=True)
        self.features = price_feature_columns(df)
        X, y = df[self.features], df[target]
        for q in self.quantiles:
            m = LGBMRegressor(objective="quantile", alpha=q, **self.params)
            m.fit(X, y)
            self.models[q] = m
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        preds = {f"p{int(q*100)}": self.models[q].predict(df[self.features]) for q in self.quantiles}
        out = pd.DataFrame(preds, index=df.index)
        # Prevent quantile CROSSING: independently trained models may produce
        # P10 > P90 (physically nonsensical). Sort per row to enforce
        # monotonicity (post-hoc, isotonic-like).
        sorted_vals = np.sort(out.to_numpy(), axis=1)
        return pd.DataFrame(sorted_vals, index=df.index, columns=out.columns)


class NegativePriceClassifier:
    """Predicts the negative-price probability (0-100%), CALIBRATED.

    Why a separate classifier + calibration?
      - Negative prices are RARE (imbalanced classes). Accuracy misleads:
        "always positive" is ~90% accurate but misses every negative event ->
        big loss. So we use class_weight='balanced' and measure with
        Brier/calibration, NOT accuracy.
      - The risk score must be a PROBABILITY, not a ranking. The Step 4
        decision engine will use a THRESHOLD like "store if probability > 70%";
        so when we say P=0.7, ~70% of those cases must really turn negative.
        Raw GBM scores are not calibrated; we fix them with isotonic regression.
    """

    def __init__(self, params: dict | None = None, calib_fraction: float = 0.2):
        self.params = params or dict(
            n_estimators=400, learning_rate=0.05, num_leaves=31,
            subsample=0.8, colsample_bytree=0.8, min_child_samples=40,
            class_weight="balanced", random_state=42, n_jobs=-1, verbose=-1,
        )
        self.calib_fraction = calib_fraction
        self.model: LGBMClassifier | None = None
        self.calibrator: IsotonicRegression | None = None
        self.features: list[str] = []

    def fit(self, df: pd.DataFrame, price_col: str = "price_eur_mwh"):
        df = df.sort_values("target_time").reset_index(drop=True)
        self.features = price_feature_columns(df)
        y = (df[price_col] < 0).astype(int)

        # Split off the calibration set TEMPORALLY from the last slice (no leakage).
        cut = int(len(df) * (1 - self.calib_fraction))
        Xtr, ytr = df[self.features].iloc[:cut], y.iloc[:cut]
        Xcal, ycal = df[self.features].iloc[cut:], y.iloc[cut:]

        self.model = LGBMClassifier(**self.params)
        self.model.fit(Xtr, ytr)

        # Isotonic calibration: maps the raw probability to the true frequency.
        raw_cal = self.model.predict_proba(Xcal)[:, 1]
        self.calibrator = IsotonicRegression(out_of_bounds="clip")
        self.calibrator.fit(raw_cal, ycal)

        base_rate = float(y.mean())
        logger.info("Negative-price base rate: %.1f%% (imbalanced class)", base_rate * 100)
        return self

    def predict_risk(self, df: pd.DataFrame) -> np.ndarray:
        """Calibrated negative-price probability [0, 1]."""
        raw = self.model.predict_proba(df[self.features])[:, 1]
        return self.calibrator.predict(raw)

    def evaluate(self, df: pd.DataFrame, price_col: str = "price_eur_mwh") -> dict:
        y = (df[price_col] < 0).astype(int).to_numpy()
        p = self.predict_risk(df)
        brier = brier_score_loss(y, p)
        # Calibration "goodness" summary: the true rate among our high-risk calls
        hi = p >= 0.5
        realized_hi = float(y[hi].mean()) if hi.any() else float("nan")
        return {
            "brier": float(brier),
            "base_rate": float(y.mean()),
            "n_high_risk": int(hi.sum()),
            "realized_rate_when_high": realized_hi,
        }
