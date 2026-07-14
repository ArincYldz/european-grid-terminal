"""Economic primitives for the decision engine.

Every threshold here is NOT a magic number, it derives from an economic
quantity. In an interview you must be able to answer "where does this 50 come
from?" with "from marginal cost + subsidy + storage opportunity cost".
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class EconomicParams:
    """Economic parameters of the asset and the market (all configurable)."""

    # Generation side
    srmc_eur_mwh: float = 1.0          # short-run marginal cost (renewables ~0)
    subsidy_eur_mwh: float = 60.0      # feed-in / CfD subsidy (per MWh produced)

    # Storage (battery) side
    storage_capacity_mwh: float = 2000.0   # reservoir size
    storage_power_mw: float = 500.0        # charge/discharge power (per 15min/hourly window)
    round_trip_efficiency: float = 0.86    # round-trip efficiency (charge*discharge)
    degradation_eur_mwh: float = 3.0       # cyclic degradation cost

    # Market frictions (CRITICAL in the backtest — forgetting these eliminates a candidate)
    transaction_cost_eur_mwh: float = 0.5  # spread / commission
    imbalance_penalty_eur_mwh: float = 45.0  # imbalance penalty (commitment vs delivery gap)

    # Peak hours (local time, typical evening demand peak)
    peak_hours: tuple[int, ...] = field(default=(8, 9, 10, 18, 19, 20))

    # Decision thresholds (derived; used by the functions below)
    charge_price_threshold: float = 15.0   # consider storing below this price
    expected_recovery_price: float = 70.0  # ~price at which stored energy is later sold


def expected_price(p10: float, p50: float, p90: float) -> float:
    """Approximate expected price (E[price]) from P10/P50/P90.

    Why not just p50? In a skewed distribution the median diverges from the
    mean; the negative-price tail skews the distribution left. A rough 3-point
    quadrature estimates the mean better than p50 alone. The weights
    (0.3, 0.4, 0.3) are a simple 3-point scheme; more quantiles -> better
    estimate (this is an approximation, deliberately simple).
    """
    return 0.3 * p10 + 0.4 * p50 + 0.3 * p90


def downside_risk(p10: float) -> float:
    """Downside risk: a 10% VaR proxy. p10 = "threshold of the worst 10%".

    Note: the true Expected Shortfall (the MEAN of the tail) is worse than p10;
    with only 3 quantiles we cannot compute exact ES, so we use p10 as a
    conservative downside indicator.
    """
    return p10


def effective_curtail_floor(p: EconomicParams) -> float:
    """The price floor for curtailment (stopping production).

    For subsidized renewables the sell-decision revenue is: price + subsidy. As
    long as this stays POSITIVE the producer keeps selling — even if the price
    is NEGATIVE! The producer only stops when price + subsidy < 0, i.e.
    price < -subsidy. This is the secret of why negative prices are so
    persistent: the subsidy shifts the selling incentive into negative
    territory.
    """
    return -p.subsidy_eur_mwh


def storage_opportunity_value(current_price: float, p: EconomicParams) -> float:
    """Net opportunity value (EUR) of storing 1 MWh NOW to sell it LATER.

    Value = efficiency * expected_future_price - current_price - degradation.
    If positive, storing beats selling now. The round-trip efficiency (eta<1)
    loss and degradation cost are subtracted — forgetting eta is a classic
    mistake (you lose ~14% of what you store in efficiency).
    """
    return (
        p.round_trip_efficiency * p.expected_recovery_price
        - current_price
        - p.degradation_eur_mwh
    )


def is_peak(hour: int, p: EconomicParams) -> bool:
    return hour in p.peak_hours


def calibrate_params_from_history(
    prices,
    mean_generation_mw: float,
    *,
    subsidy_eur_mwh: float = 0.0,
    recovery_quantile: float = 0.80,
    battery_power_fraction: float = 0.5,
    battery_hours: float = 4.0,
    **overrides,
) -> EconomicParams:
    """Derive economic parameters from TRAIN-set history — never from the test set.

    Why calibrate instead of hardcoding? Thresholds tuned for one price regime
    (e.g. synthetic prices with a median ~20) are wrong for another (real German
    2024 prices with a median ~79). Deriving them from the training data lets the
    same policy adapt to whatever market it is deployed in, WITHOUT peeking at the
    test set (that would be hindsight leakage).

      - expected_recovery_price = a high quantile of TRAIN prices (a realistic
        "sell-high" discharge target).
      - subsidy defaults to 0 (a MERCHANT asset): with no subsidy the curtail
        floor is 0, so the strategy stops production during real negative-price
        hours — the meaningful lever on real data.
      - battery is sized RELATIVE to the asset's own generation (power = a
        fraction of mean output, capacity = power x duration), so the arbitrage
        lever stays realistic at any asset scale.
    """
    recovery = float(np.quantile(np.asarray(prices, dtype=float), recovery_quantile))
    power = max(1.0, battery_power_fraction * float(mean_generation_mw))
    capacity = battery_hours * power
    return EconomicParams(
        subsidy_eur_mwh=subsidy_eur_mwh,
        expected_recovery_price=recovery,
        storage_power_mw=power,
        storage_capacity_mwh=capacity,
        **overrides,
    )
