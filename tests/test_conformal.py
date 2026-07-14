"""Conformal calibration tests (fast; light training on synthetic data)."""

import numpy as np
import pandas as pd

from src.models.conformal import (
    AdaptiveConformalForecaster,
    ConformalizedQuantileForecaster,
    empirical_coverage,
)


def _shifting_price_frame(n: int = 1200) -> pd.DataFrame:
    """A frame whose price variance GROWS over time (distribution shift),
    so the plain-CQR guarantee is stressed the way seasonality stresses it."""
    idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    rng = np.random.default_rng(0)
    resid = rng.normal(6000, 3000, n)
    trend_noise = rng.normal(0, 1 + 8 * np.arange(n) / n, n)  # variance rises with time
    price = 40 + 0.01 * (resid - 4000) + trend_noise
    return pd.DataFrame({
        "target_time": idx,
        "residual_load": resid,
        "renewable_share": rng.uniform(0, 1, n),
        "gas_price_eur_mwh": rng.uniform(20, 60, n),
        "price_eur_mwh": price,
    })


def test_cqr_correction_is_nonnegative():
    df = _shifting_price_frame()
    cut = int(len(df) * 0.8)
    cqr = ConformalizedQuantileForecaster().fit(df.iloc[:cut])
    assert cqr.correction_ >= 0.0


def test_aci_improves_coverage_over_base():
    df = _shifting_price_frame()
    cut = int(len(df) * 0.8)
    train, test = df.iloc[:cut], df.iloc[cut:]
    y = test["price_eur_mwh"].to_numpy()

    cqr = ConformalizedQuantileForecaster(lo=0.1, hi=0.9).fit(train)
    qb = cqr.base.predict(test)  # uncorrected base quantiles
    cov_base = empirical_coverage(y, qb["p10"].to_numpy(), qb["p90"].to_numpy())

    aci = AdaptiveConformalForecaster(lo=0.1, hi=0.9, gamma=0.05).fit(train)
    qa = aci.predict_adaptive(test)
    cov_aci = empirical_coverage(y, qa["p10"].to_numpy(), qa["p90"].to_numpy())

    # ACI must move coverage toward the 80% nominal, above the overconfident base.
    assert cov_aci >= cov_base
    assert cov_aci >= 0.72, cov_aci  # meaningfully closer to 0.80


def test_aci_widens_when_undercovering():
    # If we feed ACI a test set where reality always falls outside the base
    # band, alpha_t should drop and the band should widen over time.
    df = _shifting_price_frame(800)
    cut = int(len(df) * 0.8)
    aci = AdaptiveConformalForecaster(lo=0.1, hi=0.9, gamma=0.1).fit(df.iloc[:cut])
    test = df.iloc[cut:].copy()
    test["price_eur_mwh"] = test["price_eur_mwh"] + 500  # force massive under-coverage
    qa = aci.predict_adaptive(test)
    # alpha_t should have decreased from its starting target of 0.2
    assert qa["alpha_t"].iloc[-1] < qa["alpha_t"].iloc[0]


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