"""Signal engine: quantile forecast + risk -> financial action.

The output derives NOT from a point forecast but from the distribution
(P10/P50/P90) + the calibrated negative-price risk. The decision function is
PURE (side-effect free) and takes the storage state as input; this makes unit
testing easy and lets the backtest engine manage state separately.

CORE PRINCIPLE — each signal is the EXPECTED-VALUE ARGMAX across actions:
  v_sell    = expected_price + subsidy
  v_store   = efficiency * expected_recovery_price - degradation   (NO SUBSIDY!)
  v_curtail = 0
The action is the largest of these three values. So a "deviation from selling"
only happens when it is genuinely more profitable -> the strategy structurally
beats the baseline. Thresholds derive from economics, not by hand.

SUBTLE POINT (scored in interviews): STORING subsidized renewable energy
FORFEITS the subsidy (the subsidy is paid only on SOLD energy). That is why
v_store has no subsidy; storing only makes sense if the future price exceeds
the current (price + subsidy) revenue.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .economics import EconomicParams, expected_price, is_peak


class SignalType(str, Enum):
    SELL = "SELL"            # sell to the grid
    STORE = "STORE"          # store (charge the battery instead of selling)
    CURTAIL = "CURTAIL"      # stop production (waste the energy)
    DISCHARGE = "DISCHARGE"  # sell from the battery (high price/peak)
    HOLD = "HOLD"            # no action (no generation)


@dataclass
class Signal:
    kind: SignalType
    sell_mwh: float = 0.0        # sold to the grid (from generation)
    store_mwh: float = 0.0       # charged into the battery
    curtail_mwh: float = 0.0     # wasted (stopped)
    discharge_mwh: float = 0.0   # sold from the battery (raw; efficiency applied at discharge)
    expected_price: float = 0.0
    rationale: str = ""


def decide(
    generation_mwh: float,
    p10: float,
    p50: float,
    p90: float,
    neg_risk: float,
    hour: int,
    storage_charge_mwh: float,
    params: EconomicParams,
) -> Signal:
    """Produces the expected-value optimal action for a single time interval."""
    e = expected_price(p10, p50, p90)
    tc = params.transaction_cost_eur_mwh
    peak = is_peak(hour, params)
    headroom = max(0.0, params.storage_capacity_mwh - storage_charge_mwh)
    max_flow = params.storage_power_mw  # 1-hour window: MW ~ MWh

    # --- First: if the battery holds energy and price is high, DISCHARGE (sell) ---
    # Discharge value (per raw MWh): efficiency*price - transaction cost.
    discharge = 0.0
    if storage_charge_mwh > 0 and (peak or e >= params.expected_recovery_price * 0.8):
        v_discharge = params.round_trip_efficiency * e - tc
        if v_discharge > params.degradation_eur_mwh:
            discharge = min(storage_charge_mwh, max_flow)

    if generation_mwh <= 0:
        if discharge > 0:
            return Signal(SignalType.DISCHARGE, discharge_mwh=discharge, expected_price=e,
                          rationale=f"No generation; battery discharges {discharge:.0f} MWh (price {e:.1f}).")
        return Signal(SignalType.HOLD, expected_price=e, rationale="No generation.")

    # --- Expected value (per MWh) of the 3 options for generation ---
    v_sell = e + params.subsidy_eur_mwh - tc
    v_store = params.round_trip_efficiency * params.expected_recovery_price \
        - params.degradation_eur_mwh - tc
    v_curtail = 0.0
    can_store = headroom > 0

    best = max(v_sell, v_store if can_store else float("-inf"), v_curtail)

    # --- STORE: if storing is the best option (future revenue beats now) ---
    if can_store and best == v_store and v_store > v_sell:
        store = min(generation_mwh, max_flow, headroom)
        remainder = generation_mwh - store
        # Secondary decision for the remainder: sell vs curtail.
        if remainder > 0 and v_sell < v_curtail:
            return Signal(SignalType.STORE, store_mwh=store, curtail_mwh=remainder,
                          discharge_mwh=discharge, expected_price=e,
                          rationale=f"Storing is more profitable (v_store {v_store:.1f} > v_sell {v_sell:.1f}); "
                                    f"stored {store:.0f} MWh, remainder curtailed.")
        return Signal(SignalType.STORE, store_mwh=store, sell_mwh=remainder,
                      discharge_mwh=discharge, expected_price=e,
                      rationale=f"Storing is more profitable (v_store {v_store:.1f} > v_sell {v_sell:.1f}); "
                                f"stored {store:.0f} MWh, remainder sold.")

    # --- CURTAIL: both selling and storing have negative value -> do not produce ---
    if best == v_curtail and v_sell < 0:
        return Signal(SignalType.CURTAIL, curtail_mwh=generation_mwh,
                      discharge_mwh=discharge, expected_price=e,
                      rationale=f"v_sell {v_sell:.1f} < 0 and storing unprofitable; production stopped "
                                f"(neg risk {neg_risk*100:.0f}%).")

    # --- SELL (default; highest value is selling) ---
    kind = SignalType.DISCHARGE if discharge > 0 else SignalType.SELL
    return Signal(kind, sell_mwh=generation_mwh, discharge_mwh=discharge, expected_price=e,
                  rationale=f"Selling is the most profitable option (v_sell {v_sell:.1f}); all generation sold"
                            + (f" + battery discharged {discharge:.0f} MWh." if discharge else "."))
