"""Leakage-safe feature engineering for generation forecasting (Step 3A).

THIS IS THE HEART OF THE INTERVIEW. In time series, leakage almost ALWAYS
enters during feature engineering, from the wrong use of `shift`/`rolling`.
Three golden rules:

  RULE 1 — Input = FORECAST, target = REALIZED.
    The input to the generation-forecast model is the weather FORECAST (what
    we hold at decision time). The target is realized generation. In training
    we use past weather FORECAST runs; if we use realized weather we get
    train/serve skew (in production there is no realized weather yet).

  RULE 2 — Lags look ONLY to the past: shift(+k), never shift(-k).
    shift(+1) = "the value 1 hour ago" (past, safe).
    shift(-1) = "the value 1 hour later" (FUTURE, leakage!).

  RULE 3 — A rolling window does NOT include itself: shift first, then roll
    (or closed='left'). Otherwise, when computing the target at t, it folds
    the value at t into the average — a subtle but fatal leak.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# (Simplified) power-curve thresholds for a typical onshore wind turbine
CUT_IN_SPEED = 3.0     # m/s — below this the turbine does not spin, output = 0
RATED_SPEED = 12.0     # m/s — above this it saturates at rated power
CUT_OUT_SPEED = 25.0   # m/s — above this it stops for safety, output = 0


def add_power_curve_features(df: pd.DataFrame, wind_col: str = "wind_speed_100m") -> pd.DataFrame:
    """Turns raw wind speed into features matching the turbine's PHYSICS.

    Interview question: "Why not raw speed?"
    Answer: turbine power is proportional to the CUBE of wind speed (P ~ v^3),
    but this relationship is NOT valid everywhere — it is a piecewise curve:
      - v < cut-in (3 m/s):   output 0
      - cut-in..rated:        rises like ~v^3
      - rated..cut-out:       saturates at rated power (flat)
      - v > cut-out (25 m/s): safety shutdown, output suddenly 0
    Tree-based models (LightGBM) struggle to learn the cubic relation from raw
    v (the number of split thresholds grows). By handing over v^3 and the
    piecewise region flags we make the model's job easier (domain-knowledge
    injection). These lines use t's own weather forecast — NOT a lag, a
    same-instant transform — so they contain no leakage.
    """
    df = df.copy()
    v = df[wind_col]
    df[f"{wind_col}_cubed"] = v.pow(3)
    df["in_operating_range"] = ((v >= CUT_IN_SPEED) & (v < CUT_OUT_SPEED)).astype(int)
    df["above_rated"] = (v >= RATED_SPEED).astype(int)
    # Effective power-curve speed: v only within the operating range, else 0.
    eff = v.clip(lower=0)
    eff = eff.where((v >= CUT_IN_SPEED) & (v < CUT_OUT_SPEED), 0.0)
    df["effective_wind"] = eff.clip(upper=RATED_SPEED)  # saturate above rated
    return df


def add_calendar_features(df: pd.DataFrame, time_col: str = "target_time") -> pd.DataFrame:
    """Calendar/seasonality features — with CYCLICAL encoding.

    Interview question: "Why is the distance between hour 23 and hour 0 one,
    not 23?" If we feed raw 'hour' (0..23) the model thinks 23 and 0 are far
    apart; but hour 23 and hour 00 are neighbours. Solution: map them onto the
    unit circle with sin/cos — 23 and 0 become adjacent. Same for day-of-year
    (seasonal sun angle).
    """
    df = df.copy()
    t = df[time_col]
    hour = t.dt.hour + t.dt.minute / 60.0
    doy = t.dt.dayofyear

    df["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    df["is_daytime"] = ((hour >= 6) & (hour <= 20)).astype(int)  # rough mask for PV
    return df


def add_lag_features(
    df: pd.DataFrame,
    cols: list[str],
    lags: tuple[int, ...] = (1, 24),
) -> pd.DataFrame:
    """Makes past values into features. ONLY positive shift (past).

    Note: these lags are built from WEATHER FORECAST columns (the input side),
    not the target. Lagging the target is also possible, but then you must
    make sure the same lag is available at decision time in production (the
    publication delay!) — see entsoe_client.ACTUALS_PUBLICATION_LAG.
    """
    df = df.sort_values("target_time").copy()
    for col in cols:
        if col not in df.columns:
            continue
        for lag in lags:
            if lag <= 0:
                raise ValueError(
                    f"Lag must be positive (past). Got {lag} — shift(-k) = LEAKAGE."
                )
            df[f"{col}_lag{lag}"] = df[col].shift(lag)
    return df


def add_rolling_features(
    df: pd.DataFrame,
    cols: list[str],
    windows: tuple[int, ...] = (3, 24),
) -> pd.DataFrame:
    """Moving mean/std — the window does NOT include itself (shift(1) first).

    Critical subtlety: `shift(1).rolling(w)`. We shift to the previous row and
    then take the window; so when predicting the target at t, the value at t
    itself does NOT enter the average. `rolling(w)` alone includes t — that
    leaks information from the same instant as the target.
    """
    df = df.sort_values("target_time").copy()
    for col in cols:
        if col not in df.columns:
            continue
        shifted = df[col].shift(1)  # << shift into the past first
        for w in windows:
            df[f"{col}_rollmean{w}"] = shifted.rolling(w, min_periods=max(2, w // 2)).mean()
            df[f"{col}_rollstd{w}"] = shifted.rolling(w, min_periods=max(2, w // 2)).std()
    return df


def build_generation_feature_matrix(
    weather: pd.DataFrame,
    wind_col: str = "wind_speed_100m",
    solar_col: str = "shortwave_radiation",
) -> pd.DataFrame:
    """Builds the full feature matrix from raw weather forecast (pipeline).

    Order matters: same-instant transforms first (power curve, calendar), then
    the past-looking lag/rolling. The output is a model-ready feature matrix
    (it still carries target_time).
    """
    df = add_power_curve_features(weather, wind_col=wind_col)
    df = add_calendar_features(df)
    df = add_lag_features(df, cols=[wind_col, solar_col, "temperature_2m"], lags=(1, 24))
    df = add_rolling_features(df, cols=[wind_col, solar_col], windows=(3, 24))
    return df
