"""PHYSICS-FLAVOURED SYNTHETIC demand + price (OFFLINE/TEST FALLBACK).

!!! THIS IS NOT REAL DATA !!!
In reality: demand = ENTSO-E load data, price = ENTSO-E day-ahead / EPEX
intraday. To run the pipeline end-to-end without an API key we generate price
from the FUNDAMENTAL LOGIC of the electricity market (merit order) so that the
negative-price event arises for the correct reason.

MERIT ORDER logic — the core idea to explain in an interview:
  What sets the price is not DEMAND but RESIDUAL LOAD:
        residual_load = demand - renewable_generation
  Plants are dispatched cheapest-first (merit order): first ~0-cost wind/solar,
  then nuclear/coal, and last the expensive gas. The price is the marginal cost
  of the LAST (most expensive) plant that meets the residual load.
    - Residual load HIGH -> expensive gas is on -> price high.
    - Residual load VERY LOW -> only "must-run" (non-stoppable, subsidized
      renewables + technical minimum) remains; these keep producing even if
      price goes NEGATIVE (shutdown cost + lost subsidy is more expensive)
      -> NEGATIVE PRICE.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def synthetic_demand(weather: pd.DataFrame, time_col: str = "target_time") -> pd.Series:
    """Typical double-peaked (morning + evening) daily + weekly demand profile (MW)."""
    t = weather[time_col]
    hour = t.dt.hour + t.dt.minute / 60.0
    dow = t.dt.dayofweek

    base = 12000.0
    # Morning (~08:00) and evening (~19:00) peaks
    morning = 2500 * np.exp(-((hour - 8) ** 2) / 6)
    evening = 3500 * np.exp(-((hour - 19) ** 2) / 8)
    night_dip = -2000 * np.exp(-((hour - 3) ** 2) / 8)
    weekend = np.where(dow >= 5, -1500.0, 0.0)  # demand drops on weekends
    # Winter heating / summer cooling — rough seasonal component
    doy = t.dt.dayofyear
    seasonal = 1500 * np.cos(2 * np.pi * (doy - 15) / 365.25)

    demand = base + morning + evening + night_dip + weekend + seasonal
    return pd.Series(demand, index=weather.index, name="demand_mw")


def synthetic_gas_price(weather: pd.DataFrame, time_col: str = "target_time", seed: int = 7) -> pd.Series:
    """Slowly-drifting natural gas price (EUR/MWh_th) — continuous across days."""
    t = weather[time_col]
    days = (t - t.min()).dt.total_seconds() / 86400.0
    rng = np.random.default_rng(seed)
    # Random walk + mean-reversion, daily
    n_days = int(days.max()) + 2
    walk = 30 + np.cumsum(rng.normal(0, 0.8, n_days))
    walk = np.clip(walk, 15, 70)
    gas = walk[days.astype(int)]
    return pd.Series(gas, index=weather.index, name="gas_price_eur_mwh")


def synthetic_price(
    demand: pd.Series,
    generation: pd.Series,
    gas_price: pd.Series,
    seed: int = 11,
) -> pd.Series:
    """Merit-order-based day-ahead price (EUR/MWh) — can go negative.

    residual_load = demand - renewables. Price is a monotonically increasing
    function of residual load; at low residual load it drops into NEGATIVE
    territory, at high residual load it rises via gas marginal cost and a
    scarcity premium.
    """
    rng = np.random.default_rng(seed)
    residual = (demand - generation).to_numpy()

    # Gas plant thermal efficiency ~0.5; electricity marginal cost ~gas*1.3.
    # This is the "normal-hours" floor where gas sets the price (~40-90).
    gas_base = gas_price.to_numpy() * 1.3

    # 1) RENEWABLE DEPRESSION: when residual load falls below the threshold
    #    (renewables start exceeding demand) price is pulled below the floor
    #    into negative territory. Crossover ~4500 MW -> negative-price rate ~6%.
    depression = -0.020 * np.clip(7000 - residual, 0, None)

    # 2) SCARCITY PREMIUM: at very high residual load expensive peaker plants
    #    are on and price spikes convexly.
    scarcity = 6e-7 * np.clip(residual - 12000, 0, None) ** 2

    price = gas_base + depression + scarcity
    # "Must-run" floor: shutdown cost + lost subsidy stop the price from
    # falling below a certain level (~-90 EUR/MWh).
    price = np.clip(price, -90, None)
    # Noise (demand forecast error, discreteness of bids)
    price += rng.normal(0, 6, size=price.shape)

    return pd.Series(price, index=demand.index, name="price_eur_mwh")
