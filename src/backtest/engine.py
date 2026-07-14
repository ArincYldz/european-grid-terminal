"""Walk-forward backtest: settles signals against REALIZED prices with
REALISTIC costs.

What eliminates a candidate in a backtest is an optimistic P&L. Three fatal
omissions:
  1. LOOK-AHEAD: the decision is made using ONLY the forecasts known at
     decision time; the P&L is settled with the REALIZED price. Mixing the
     two = fake profit that peeks into the future.
  2. TRANSACTION COST (spread/commission): every trade erodes each MWh.
  3. IMBALANCE PENALTY: the commitment is made on the FORECAST, but delivery
     is the REALIZED generation; the difference is closed on the balancing
     market at a penalty price. Wind forecast error flows straight here.
  Plus the ROUND-TRIP EFFICIENCY (eta<1) loss on storage cannot be ignored.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from src.decision import EconomicParams, decide


@dataclass
class BacktestResult:
    total_pnl: float
    revenue: float
    subsidy: float
    transaction_cost: float
    imbalance_cost: float
    degradation_cost: float
    n_sell: int = 0
    n_store: int = 0
    n_curtail: int = 0
    n_discharge: int = 0
    curtailed_mwh: float = 0.0
    ledger: pd.DataFrame = field(default=None, repr=False)

    def summary(self) -> str:
        return (
            f"Total P&L : {self.total_pnl:,.0f} EUR\n"
            f"  revenue           : {self.revenue:,.0f}\n"
            f"  subsidy           : {self.subsidy:,.0f}\n"
            f"  transaction cost  : -{self.transaction_cost:,.0f}\n"
            f"  imbalance penalty : -{self.imbalance_cost:,.0f}\n"
            f"  degradation       : -{self.degradation_cost:,.0f}\n"
            f"Signals: SELL {self.n_sell} | STORE {self.n_store} | "
            f"CURTAIL {self.n_curtail} | DISCHARGE {self.n_discharge}\n"
            f"Curtailed energy: {self.curtailed_mwh:,.0f} MWh"
        )


def _subsidy_for(price: float, p: EconomicParams) -> float:
    """Subsidy rate. Simple model: fixed feed-in (old EEG).

    REFINEMENT (narrative): the 'negative-price rule' suspends the subsidy
    after 6 consecutive negative-price hours. A single-period simplification
    is to cut the subsidy when price<0. Kept fixed here by DEFAULT; a flag on
    `p` could switch it.
    """
    return p.subsidy_eur_mwh


def run_backtest(df: pd.DataFrame, params: EconomicParams | None = None) -> BacktestResult:
    """Smart-strategy backtest.

    Expected columns (all ready in df):
      target_time, p10, p50, p90, neg_risk,
      forecast_generation_mwh  (the FORECAST known at decision time),
      actual_generation_mwh    (the REALIZED delivery),
      realized_price           (settlement price, REALIZED)
    """
    p = params or EconomicParams()
    df = df.sort_values("target_time").reset_index(drop=True)

    charge = 0.0
    revenue = subsidy_tot = tc_tot = imb_tot = deg_tot = 0.0
    n = dict(SELL=0, STORE=0, CURTAIL=0, DISCHARGE=0, HOLD=0)
    curtailed = 0.0
    rows = []

    for r in df.itertuples(index=False):
        hour = pd.Timestamp(r.target_time).hour
        sig = decide(
            generation_mwh=r.forecast_generation_mwh,
            p10=r.p10, p50=r.p50, p90=r.p90, neg_risk=r.neg_risk,
            hour=hour, storage_charge_mwh=charge, params=p,
        )
        n[sig.kind.value] += 1
        pr = r.realized_price
        sub = _subsidy_for(pr, p)
        rev_i = sub_i = tc_i = imb_i = deg_i = 0.0  # per-interval accounting

        # --- SELL from generation: realized generation is delivered; the
        #     commitment was on the FORECAST, so the gap is an imbalance. ---
        sold_gen = 0.0
        if sig.sell_mwh > 0:
            sold_gen = r.actual_generation_mwh
            rev_i += sold_gen * pr
            sub_i += sold_gen * sub
            tc_i += sold_gen * p.transaction_cost_eur_mwh
            imb_i += abs(r.forecast_generation_mwh - r.actual_generation_mwh) * p.imbalance_penalty_eur_mwh

        # --- STORE: charge battery from own generation (forgoes subsidy) ---
        if sig.store_mwh > 0:
            stored = min(sig.store_mwh, r.actual_generation_mwh)
            charge += stored
            deg_i += stored * p.degradation_eur_mwh

        # --- CURTAIL: 0 revenue, 0 imbalance (planned curtailment, 0 delivery) ---
        if sig.curtail_mwh > 0:
            curtailed += min(sig.curtail_mwh, r.actual_generation_mwh)

        # --- DISCHARGE: sell from battery; efficiency applied at discharge ---
        if sig.discharge_mwh > 0:
            out = min(sig.discharge_mwh, charge)
            grid = p.round_trip_efficiency * out
            rev_i += grid * pr
            tc_i += grid * p.transaction_cost_eur_mwh
            deg_i += out * p.degradation_eur_mwh
            charge -= out

        revenue += rev_i; subsidy_tot += sub_i; tc_tot += tc_i; imb_tot += imb_i; deg_tot += deg_i
        pnl_i = rev_i + sub_i - tc_i - imb_i - deg_i
        rows.append((r.target_time, sig.kind.value, sig.expected_price, pr, sold_gen, charge, pnl_i))

    total = revenue + subsidy_tot - tc_tot - imb_tot - deg_tot
    ledger = pd.DataFrame(
        rows, columns=["target_time", "signal", "e_price", "realized", "sold_mwh", "charge_mwh", "pnl"]
    )
    return BacktestResult(
        total_pnl=total, revenue=revenue, subsidy=subsidy_tot,
        transaction_cost=tc_tot, imbalance_cost=imb_tot, degradation_cost=deg_tot,
        n_sell=n["SELL"], n_store=n["STORE"], n_curtail=n["CURTAIL"], n_discharge=n["DISCHARGE"],
        curtailed_mwh=curtailed, ledger=ledger,
    )


def run_naive_baseline(df: pd.DataFrame, params: EconomicParams | None = None) -> BacktestResult:
    """Naive baseline: ALWAYS sell all generation (no strategy).

    Same settlement + cost accounting -> a fair comparison. The smart
    strategy should beat this by curtailing deep-negative hours and selling
    from storage at peak.
    """
    p = params or EconomicParams()
    df = df.sort_values("target_time").reset_index(drop=True)

    revenue = subsidy_tot = tc_tot = imb_tot = 0.0
    rows = []
    for r in df.itertuples(index=False):
        pr = r.realized_price
        sold = r.actual_generation_mwh
        rev_i = sold * pr
        sub_i = sold * _subsidy_for(pr, p)
        tc_i = sold * p.transaction_cost_eur_mwh
        imb_i = abs(r.forecast_generation_mwh - r.actual_generation_mwh) * p.imbalance_penalty_eur_mwh
        revenue += rev_i; subsidy_tot += sub_i; tc_tot += tc_i; imb_tot += imb_i
        rows.append((r.target_time, "SELL", 0.0, pr, sold, 0.0, rev_i + sub_i - tc_i - imb_i))

    total = revenue + subsidy_tot - tc_tot - imb_tot
    ledger = pd.DataFrame(
        rows, columns=["target_time", "signal", "e_price", "realized", "sold_mwh", "charge_mwh", "pnl"]
    )
    return BacktestResult(
        total_pnl=total, revenue=revenue, subsidy=subsidy_tot,
        transaction_cost=tc_tot, imbalance_cost=imb_tot, degradation_cost=0.0,
        n_sell=len(df), ledger=ledger,
    )
