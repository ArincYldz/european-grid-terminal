"""Tests for the SHAP explainability layer.

The dashboard shows these numbers as the reason a forecast says what it says,
so the property that matters is ADDITIVITY: base + contributions must
reconstruct the model's own output. If that ever breaks, the explanation is
decoration rather than an explanation, and it would break silently — the bars
would still render, they would just be telling the user something untrue.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.explain import (
    aggregate_explanation,
    explain_rows,
    humanise,
    invalidators,
    similar_days,
)

FEATURES = ["residual_load", "predicted_generation_mw", "hour_sin", "price_lag24"]


def _fixture(n: int = 500, seed: int = 0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "residual_load": rng.normal(30000, 8000, n),
        "predicted_generation_mw": rng.normal(12000, 5000, n),
        "hour_sin": rng.uniform(-1, 1, n),
        "price_lag24": rng.normal(80, 30, n),
    })
    # Residual load is the dominant driver by construction.
    y = (0.004 * df.residual_load - 0.003 * df.predicted_generation_mw
         + 10 * df.hour_sin + rng.normal(0, 3, n))
    model = LGBMRegressor(n_estimators=120, verbose=-1, random_state=0).fit(df[FEATURES], y)
    return df, y, model


def test_shap_contributions_reconstruct_the_prediction():
    """base + sum(effects) == model output. This is what makes them Shapley values."""
    df, _, model = _fixture()
    rows = explain_rows(model, df.iloc[:25], FEATURES, top_n=len(FEATURES))
    truth = model.predict(df[FEATURES].iloc[:25])
    for i, r in enumerate(rows):
        assert abs(r["prediction_eur_mwh"] - truth[i]) < 0.15, (
            f"row {i}: shap says {r['prediction_eur_mwh']}, model says {truth[i]}")


def test_shap_finds_the_driver_that_actually_generates_the_target():
    """The strongest coefficient in the data must come out as the top feature."""
    df, _, model = _fixture()
    agg = aggregate_explanation(explain_rows(model, df.iloc[:120], FEATURES, top_n=4))
    assert agg["features"][0]["name"] == "residual_load", agg["features"]


def test_top_n_is_respected_and_ordered_by_absolute_effect():
    df, _, model = _fixture()
    rows = explain_rows(model, df.iloc[:5], FEATURES, top_n=2)
    for r in rows:
        assert len(r["features"]) == 2
        a, b = abs(r["features"][0]["effect_eur_mwh"]), abs(r["features"][1]["effect_eur_mwh"])
        assert a >= b, r["features"]


def test_aggregate_shares_sum_to_100():
    df, _, model = _fixture()
    agg = aggregate_explanation(explain_rows(model, df.iloc[:60], FEATURES, top_n=4))
    total = sum(f["share_pct"] for f in agg["features"])
    assert abs(total - 100.0) < 1.5, total


def test_aggregate_nets_out_a_driver_that_pushes_both_ways():
    """A feature that adds in some hours and subtracts in others must not look
    important in both directions — the signed mean is the honest summary."""
    per_row = [
        {"base_eur_mwh": 50, "prediction_eur_mwh": 60,
         "features": [{"name": "x", "label": "X", "effect_eur_mwh": 10.0, "share_pct": 100.0}]},
        {"base_eur_mwh": 50, "prediction_eur_mwh": 40,
         "features": [{"name": "x", "label": "X", "effect_eur_mwh": -10.0, "share_pct": 100.0}]},
    ]
    agg = aggregate_explanation(per_row)
    assert abs(agg["features"][0]["effect_eur_mwh"]) < 0.01, agg["features"]


def test_similar_days_returns_real_rows_closest_first():
    df, y, _ = _fixture()
    train = df.copy()
    train["target_time"] = pd.date_range("2026-01-01", periods=len(df), freq="h", tz="UTC")
    train["price_eur_mwh"] = y
    sims = similar_days(train, df.iloc[0], FEATURES, k=3)
    assert len(sims) == 3
    assert sims[0]["distance"] <= sims[1]["distance"] <= sims[2]["distance"]
    # The nearest neighbour of a training row is itself, at distance ~0.
    assert sims[0]["distance"] < 1e-6, sims[0]


def test_similar_days_degrades_quietly_without_usable_columns():
    assert similar_days(pd.DataFrame({"target_time": []}), pd.Series(dtype=float),
                        FEATURES, k=3) == []


def test_invalidators_flag_a_wide_band_and_a_negative_tail():
    out = invalidators(p10=-20, p50=40, p90=140, neg_risk=0.3,
                       top_features=[{"label": "Residual load", "share_pct": 60.0}])
    joined = " ".join(out).lower()
    assert "wide" in joined
    assert "negative" in joined
    assert "residual load" in joined
    # The unknown-unknowns caveat is always present.
    assert any("outage" in s.lower() for s in out)


def test_invalidators_stay_quiet_when_the_forecast_is_tight():
    out = invalidators(p10=48, p50=50, p90=52, neg_risk=0.0, top_features=[])
    assert not any("wide" in s.lower() for s in out)
    assert not any("negative prices" in s.lower() for s in out)


def test_humanise_falls_back_to_a_readable_name():
    assert humanise("residual_load") == "Residual load (demand minus renewables)"
    assert humanise("some_new_feature") == "Some new feature"


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as exc:
                fails += 1
                print(f"FAIL {name}: {exc}")
    print("--- all explain tests passed" if not fails else f"--- {fails} FAILED")
    sys.exit(1 if fails else 0)
