"""Feature-engineering leakage tests (run without a DB or API).

These tests are the concrete proof for the interview question "how do you
PREVENT leakage at the code level?": the safeguards are protected by
automated tests.
"""

import numpy as np
import pandas as pd

from src.features.generation_features import (
    add_calendar_features,
    add_lag_features,
    add_rolling_features,
    build_generation_feature_matrix,
)


def _hourly(n: int = 100) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "target_time": idx,
            "wind_speed_100m": rng.uniform(0, 20, n),
            "shortwave_radiation": rng.uniform(0, 800, n),
            "temperature_2m": rng.uniform(-5, 30, n),
            "available_at": idx,  # for the test
            "source": "test",
        }
    )


def test_negative_lag_is_rejected():
    df = _hourly()
    try:
        add_lag_features(df, ["wind_speed_100m"], lags=(-1,))
        assert False, "shift(-k) leakage should have been rejected"
    except ValueError:
        pass


def test_lag_uses_only_past():
    df = _hourly()
    out = add_lag_features(df, ["wind_speed_100m"], lags=(1,))
    # lag1[t] must equal exactly wind[t-1] (the past)
    lag = out["wind_speed_100m_lag1"].to_numpy()
    orig = out["wind_speed_100m"].to_numpy()
    assert np.isnan(lag[0])  # the first row has no past
    assert np.allclose(lag[1:], orig[:-1])


def test_rolling_excludes_current_row():
    df = _hourly()
    out = add_rolling_features(df, ["wind_speed_100m"], windows=(3,))
    orig = out["wind_speed_100m"].to_numpy()
    roll = out["wind_speed_100m_rollmean3"].to_numpy()
    # For t=4 the window must be t-1,t-2,t-3 (excluding t ITSELF)
    expected_t4 = np.mean(orig[1:4])  # indices 1,2,3
    assert np.isclose(roll[4], expected_t4), (roll[4], expected_t4)
    # Proof: t's own value never enters its window
    assert not np.isclose(roll[4], np.mean(orig[2:5]))


def test_calendar_cyclical_wraps_around():
    df = _hourly(48)
    out = add_calendar_features(df)
    # Hour 23 and hour 0 must be neighbours in cyclical space:
    # the angle between their hour_sin/cos vectors must be small.
    h23 = out[out["target_time"].dt.hour == 23].iloc[0]
    h00 = out[out["target_time"].dt.hour == 0].iloc[0]
    dist = np.hypot(h23["hour_sin"] - h00["hour_sin"], h23["hour_cos"] - h00["hour_cos"])
    # The chord of a 1-hour separation (~0.26); proof against the raw |23-0|=23 disaster
    assert dist < 0.3, dist


def test_full_matrix_builds_and_has_no_future_columns():
    df = _hourly(200)
    out = build_generation_feature_matrix(df)
    # No feature name should imply a future/negative lag
    bad = [c for c in out.columns if "lag-" in c or "_lead" in c or "future" in c]
    assert not bad, f"Suspicious future features: {bad}"
    # power-curve and cyclical features must have been produced
    for col in ("wind_speed_100m_cubed", "effective_wind", "hour_sin", "doy_cos"):
        assert col in out.columns


if __name__ == "__main__":
    import sys

    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {fn.__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} tests passed.")
    sys.exit(1 if failed else 0)
