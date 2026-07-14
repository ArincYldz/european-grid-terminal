"""PHYSICS-FLAVOURED SYNTHETIC generation target (OFFLINE/TEST FALLBACK).

!!! NOTE — THIS IS NOT REAL DATA !!!
In the real project the target variable is the output of ENTSO-E
`fetch_actual_generation` (realized wind+solar MW). To run and test the
pipeline END-TO-END WITHOUT an ENTSO-E API key, we derive a noisy target from
the weather forecast via a physical power curve. When you add your key, the
real path in src/pipeline/data_assembly.py uses realized generation and this
module is only the fallback.

Why physics-flavoured anyway? So that the relationship the model learns (power
curve + solar radiation) is realistic, letting us validate the feature
engineering and leakage checks meaningfully.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.generation_features import (
    CUT_IN_SPEED,
    CUT_OUT_SPEED,
    RATED_SPEED,
)

# Northern Germany: wind-dominated, capacity large enough to OCCASIONALLY
# exceed regional demand (the physical precondition for negative prices).
WIND_CAPACITY_MW = 11000.0  # regional installed wind capacity (assumption)
SOLAR_CAPACITY_MW = 6000.0  # regional installed PV capacity (assumption)


def _power_curve(v: np.ndarray) -> np.ndarray:
    """Normalized (0..1) turbine power curve — piecewise physical model."""
    out = np.zeros_like(v, dtype=float)
    ramp = (v >= CUT_IN_SPEED) & (v < RATED_SPEED)
    out[ramp] = ((v[ramp] - CUT_IN_SPEED) / (RATED_SPEED - CUT_IN_SPEED)) ** 3
    rated = (v >= RATED_SPEED) & (v < CUT_OUT_SPEED)
    out[rated] = 1.0
    # v >= cut_out => 0 (safety shutdown), already 0
    return out


def synthetic_generation_target(
    weather: pd.DataFrame,
    wind_col: str = "wind_speed_100m",
    solar_col: str = "shortwave_radiation",
    seed: int = 42,
) -> pd.Series:
    """Produces a noisy, physical generation target (MW) from the weather forecast."""
    rng = np.random.default_rng(seed)
    v = weather[wind_col].to_numpy(dtype=float)
    ghi = weather[solar_col].to_numpy(dtype=float)

    wind_mw = _power_curve(v) * WIND_CAPACITY_MW
    # PV: ~linear with radiation; typical peak GHI ~900 W/m2 -> full capacity
    solar_mw = np.clip(ghi / 900.0, 0, 1) * SOLAR_CAPACITY_MW

    total = wind_mw + solar_mw
    # Multiplicative noise (measurement/forecast error) + small base noise
    noise = rng.normal(1.0, 0.08, size=total.shape)
    total = np.clip(total * noise, 0, None)
    return pd.Series(total, index=weather.index, name="generation_mw")
