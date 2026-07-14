"""Feature engineering for price forecasting (Step 3B).

The most critical feature: RESIDUAL LOAD = demand_forecast - generation_forecast.
This is the physical driver of price (merit order). Raw demand or raw
generation alone explain price weakly; their difference (residual load)
explains it strongly.

LEAKAGE WARNING (cascade/stacking):
  The generation input to this layer is Step 3A's FORECAST
  (predicted_generation), NOT realized generation. Likewise demand is a LOAD
  FORECAST. Both are available at decision time. Using the realized values
  creates train/serve skew (see build_price_feature_matrix docstring).
"""

from __future__ import annotations

import pandas as pd

from src.features.generation_features import add_calendar_features


def add_residual_load(
    df: pd.DataFrame,
    demand_col: str = "demand_forecast_mw",
    generation_col: str = "predicted_generation_mw",
) -> pd.DataFrame:
    """Residual load and its derivatives — the main driver of price."""
    df = df.copy()
    df["residual_load"] = df[demand_col] - df[generation_col]
    # Renewable penetration: generation/demand ratio (a negative-price signal)
    df["renewable_share"] = (df[generation_col] / df[demand_col]).clip(0, 2)
    return df


def add_ramp_features(df: pd.DataFrame, cols: list[str], time_col: str = "target_time") -> pd.DataFrame:
    """Ramp (rate-of-change) features — look ONLY to the past.

    Price spikes usually happen at sudden CHANGES (the sunset ramp, a wind
    drop). diff() = current - previous (past), so it is safe. Note: 'current'
    here is already a FORECAST value; the forecast series' own ramp is known
    at decision time, no leakage.
    """
    df = df.sort_values(time_col).copy()
    for col in cols:
        if col in df.columns:
            df[f"{col}_ramp1"] = df[col].diff(1)   # 1-step change
            df[f"{col}_ramp4"] = df[col].diff(4)   # 4-step (4h on hourly data) trend
    return df


def build_price_feature_matrix(
    df: pd.DataFrame,
    demand_col: str = "demand_forecast_mw",
    generation_col: str = "predicted_generation_mw",
    gas_col: str = "gas_price_eur_mwh",
) -> pd.DataFrame:
    """Feature matrix for the price model.

    The input df must contain these columns (ALL KNOWABLE at decision time):
      - demand_forecast_mw     : load FORECAST (ENTSO-E day-ahead load forecast)
      - predicted_generation_mw: Step 3A model FORECAST (NOT realized!)
      - gas_price_eur_mwh      : gas forward price (assumed intraday-constant)

    CASCADE LEAKAGE — the interview's scoring question:
      the predicted_generation_mw column must NOT be Step 3A's IN-SAMPLE
      prediction. 3A is overly optimistic on its own training data (overfit);
      if we feed that optimistic prediction to 3B, 3B gets used to an input of
      a quality it will never see in production. Solution: use 3A's
      OUT-OF-FOLD (OOF) predictions (see models.stacking.oof_generation_predictions).
    """
    df = add_residual_load(df, demand_col, generation_col)
    df = add_calendar_features(df)
    df = add_ramp_features(df, cols=["residual_load", generation_col, demand_col])
    return df
