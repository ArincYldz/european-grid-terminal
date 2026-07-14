"""Price-model + cascade tests (no API; run with light training)."""

import numpy as np
import pandas as pd

from src.models.price_model import QuantileForecaster, pinball_loss
from src.models.stacking import oof_generation_predictions


def _price_frame(n: int = 600) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    rng = np.random.default_rng(0)
    residual = rng.normal(6000, 3000, n)
    price = 40 + 0.01 * (residual - 4000) + rng.normal(0, 5, n)
    return pd.DataFrame(
        {
            "target_time": idx,
            "residual_load": residual,
            "renewable_share": rng.uniform(0, 1, n),
            "gas_price_eur_mwh": rng.uniform(20, 60, n),
            "price_eur_mwh": price,
        }
    )


def test_pinball_median_equals_half_mae():
    y = np.array([10.0, 20.0, 30.0])
    p = np.array([12.0, 18.0, 33.0])
    # For alpha=0.5 pinball = 0.5 * mean absolute error
    expected = 0.5 * np.mean(np.abs(y - p))
    assert np.isclose(pinball_loss(y, p, 0.5), expected)


def test_pinball_asymmetry():
    # alpha=0.9: under-prediction must be penalized heavily
    y = np.array([100.0])
    under = pinball_loss(y, np.array([90.0]), 0.9)   # under by 10
    over = pinball_loss(y, np.array([110.0]), 0.9)   # over by 10
    assert under > over, (under, over)


def test_quantiles_do_not_cross():
    df = _price_frame()
    qf = QuantileForecaster(quantiles=(0.1, 0.5, 0.9)).fit(df)
    pred = qf.predict(df)
    # Monotonicity: for each row P10 <= P50 <= P90 (no crossing)
    assert (pred["p10"] <= pred["p50"] + 1e-9).all()
    assert (pred["p50"] <= pred["p90"] + 1e-9).all()


def test_oof_head_is_nan_and_aligned():
    # OOF: the rows before the first fold have no training data -> the head is NaN.
    idx = pd.date_range("2024-01-01", periods=600, freq="h", tz="UTC")
    rng = np.random.default_rng(1)
    df = pd.DataFrame(
        {
            "target_time": idx,
            "wind_speed_100m": rng.uniform(0, 20, 600),
            "shortwave_radiation": rng.uniform(0, 800, 600),
            "temperature_2m": rng.uniform(-5, 30, 600),
            "hour_sin": np.sin(np.arange(600)),
            "generation_mw": rng.uniform(0, 5000, 600),
        }
    )
    oof = oof_generation_predictions(df, n_splits=5)
    assert len(oof) == len(df)
    assert oof.iloc[:50].isna().all()   # the first block gets no prediction (correct behaviour)
    assert oof.iloc[-50:].notna().all()  # the last block must be filled


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
