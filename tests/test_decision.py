"""Decision-engine + backtest tests (no API; run instantly)."""

import numpy as np
import pandas as pd

from src.backtest import run_backtest, run_naive_baseline
from src.decision import EconomicParams, decide, expected_price
from src.decision.economics import effective_curtail_floor
from src.decision.signals import SignalType


def test_expected_price_between_quantiles():
    e = expected_price(-10, 20, 60)
    assert -10 <= e <= 60
    # weighted mean = 0.3*-10 + 0.4*20 + 0.3*60 = 23
    assert np.isclose(e, 23.0)


def test_curtail_floor_is_negative_subsidy():
    p = EconomicParams(subsidy_eur_mwh=60)
    assert effective_curtail_floor(p) == -60


def test_deep_negative_triggers_curtail_when_storage_full():
    # Expected price far below -subsidy AND battery full (cannot store)
    # -> selling is loss-making, storing is impossible -> CURTAIL.
    # (If the battery were EMPTY the EV-optimal action would be STORE — keep the
    #  energy to sell later instead of wasting it; the policy captures this.)
    p = EconomicParams(subsidy_eur_mwh=20, expected_recovery_price=45,
                       storage_capacity_mwh=1000)
    sig = decide(generation_mwh=100, p10=-95, p50=-85, p90=-70, neg_risk=0.99,
                 hour=3, storage_charge_mwh=1000, params=p)  # battery full
    assert sig.kind == SignalType.CURTAIL
    assert sig.curtail_mwh == 100


def test_high_price_triggers_sell():
    p = EconomicParams(subsidy_eur_mwh=20, expected_recovery_price=45)
    sig = decide(generation_mwh=100, p10=40, p50=55, p90=70, neg_risk=0.0,
                 hour=19, storage_charge_mwh=0, params=p)
    assert sig.kind in (SignalType.SELL, SignalType.DISCHARGE)
    assert sig.sell_mwh == 100


def test_store_beats_sell_when_future_better():
    # Low current price, high expected recovery, low subsidy
    # -> storing must beat selling.
    p = EconomicParams(subsidy_eur_mwh=5, expected_recovery_price=80,
                       storage_capacity_mwh=1000, storage_power_mw=500)
    sig = decide(generation_mwh=100, p10=-5, p50=2, p90=8, neg_risk=0.5,
                 hour=2, storage_charge_mwh=0, params=p)
    assert sig.kind == SignalType.STORE
    assert sig.store_mwh > 0


def test_discharge_at_peak_when_battery_full():
    p = EconomicParams(subsidy_eur_mwh=20, expected_recovery_price=45)
    sig = decide(generation_mwh=50, p10=40, p50=50, p90=60, neg_risk=0.0,
                 hour=19, storage_charge_mwh=400, params=p)
    assert sig.discharge_mwh > 0


def _bt_frame(n=48):
    idx = pd.date_range("2024-06-01", periods=n, freq="h", tz="UTC")
    rng = np.random.default_rng(0)
    gen = rng.uniform(50, 200, n)
    return pd.DataFrame({
        "target_time": idx,
        "p10": rng.uniform(-20, 10, n), "p50": rng.uniform(10, 40, n),
        "p90": rng.uniform(40, 80, n), "neg_risk": rng.uniform(0, 0.3, n),
        "forecast_generation_mwh": gen,
        "actual_generation_mwh": gen * rng.normal(1.0, 0.05, n),
        "realized_price": rng.uniform(-30, 70, n),
    })


def test_backtest_applies_costs():
    df = _bt_frame()
    p = EconomicParams(subsidy_eur_mwh=20, expected_recovery_price=45)
    res = run_backtest(df, p)
    # Costs must be positive (transaction + imbalance applied)
    assert res.transaction_cost > 0
    assert res.imbalance_cost > 0
    # The ledger must carry one decision per row
    assert len(res.ledger) == len(df)


def test_strategy_not_worse_than_naive_on_average():
    # The EV-optimal policy must beat or tie the naive baseline (always sell)
    # at equal cost — because it curtails the deep-negative hours.
    df = _bt_frame(200)
    # Make some hours deep-negative so curtailment kicks in
    df.loc[:20, "realized_price"] = -80
    df.loc[:20, ["p10", "p50", "p90"]] = [-95, -85, -70]
    p = EconomicParams(subsidy_eur_mwh=20, expected_recovery_price=45)
    strat = run_backtest(df, p).total_pnl
    naive = run_naive_baseline(df, p).total_pnl
    assert strat >= naive - 1e-6, (strat, naive)


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
