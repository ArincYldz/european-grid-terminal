"""Risk reporting: a single total-P&L number is never enough.

A senior desk reports RISK-ADJUSTED performance, not just how much money was
made. This module turns a per-interval P&L ledger into the metrics a risk
manager actually asks for:

  - Sharpe ratio     : return per unit of volatility. Rewards steady P&L,
                       punishes lumpy P&L. (Here it is a P&L-Sharpe: the
                       increments are cash P&L, not capital-normalized
                       returns, since the asset has no single capital base.)
  - Max drawdown     : the worst peak-to-trough fall of the equity curve.
                       Answers "how bad did it get before recovering?" — the
                       number that blows up funds, not the average.
  - CVaR / Expected  : the mean of the worst q% of interval P&Ls. Unlike VaR
    Shortfall          (a threshold), CVaR describes the SIZE of the tail
                       loss — what you actually lose on a bad hour.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Hourly settlement -> annualization factor for the Sharpe ratio.
HOURS_PER_YEAR = 24 * 365


@dataclass
class RiskMetrics:
    total_pnl: float
    mean_pnl: float
    volatility: float
    sharpe: float
    max_drawdown: float
    cvar_5pct: float
    n_periods: int

    def summary(self) -> str:
        return (
            f"Total P&L      : {self.total_pnl:,.0f} EUR\n"
            f"Mean P&L/hour  : {self.mean_pnl:,.0f} EUR\n"
            f"Volatility/hour: {self.volatility:,.0f} EUR\n"
            f"Sharpe (annual): {self.sharpe:.2f}\n"
            f"Max drawdown   : {self.max_drawdown:,.0f} EUR\n"
            f"CVaR 5% (hour) : {self.cvar_5pct:,.0f} EUR  (mean of worst 5% hours)\n"
            f"Periods        : {self.n_periods}"
        )


def compute_risk_metrics(pnl: pd.Series | np.ndarray, periods_per_year: int = HOURS_PER_YEAR) -> RiskMetrics:
    """Compute risk-adjusted metrics from a per-interval P&L series."""
    x = np.asarray(pnl, dtype=float)
    n = len(x)
    mean = float(np.mean(x)) if n else 0.0
    vol = float(np.std(x, ddof=1)) if n > 1 else 0.0

    # P&L-Sharpe: mean/vol per period, annualized by sqrt(periods_per_year).
    sharpe = (mean / vol) * np.sqrt(periods_per_year) if vol > 0 else 0.0

    # Max drawdown on the cumulative equity curve.
    equity = np.cumsum(x)
    running_peak = np.maximum.accumulate(equity) if n else np.array([0.0])
    drawdown = equity - running_peak            # <= 0
    max_dd = float(drawdown.min()) if n else 0.0

    # CVaR 5%: mean of the worst 5% of interval P&Ls (a loss; typically < 0).
    if n:
        k = max(1, int(np.ceil(0.05 * n)))
        worst = np.sort(x)[:k]
        cvar = float(np.mean(worst))
    else:
        cvar = 0.0

    return RiskMetrics(
        total_pnl=float(np.sum(x)), mean_pnl=mean, volatility=vol,
        sharpe=float(sharpe), max_drawdown=max_dd, cvar_5pct=cvar, n_periods=n,
    )


def plot_equity_curves(ledgers: dict[str, pd.DataFrame], out_path: str) -> str:
    """Plot cumulative P&L (equity) curves for one or more strategies.

    ledgers: {label -> ledger DataFrame with 'target_time' and 'pnl' cols}.
    Saves a PNG to out_path and returns the path.
    """
    import matplotlib
    matplotlib.use("Agg")  # headless: no display needed
    import matplotlib.pyplot as plt

    fig, (ax_eq, ax_dd) = plt.subplots(
        2, 1, figsize=(11, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )

    for label, led in ledgers.items():
        led = led.sort_values("target_time")
        t = pd.to_datetime(led["target_time"])
        equity = led["pnl"].cumsum() / 1e6  # million EUR
        ax_eq.plot(t, equity, label=label, linewidth=1.6)

        peak = equity.cummax()
        ax_dd.fill_between(t, (equity - peak), 0, alpha=0.4, label=label)

    ax_eq.set_title("Cumulative P&L (equity curve)")
    ax_eq.set_ylabel("Cumulative P&L (M EUR)")
    ax_eq.legend(loc="upper left")
    ax_eq.grid(True, alpha=0.3)

    ax_dd.set_title("Drawdown")
    ax_dd.set_ylabel("Drawdown (M EUR)")
    ax_dd.set_xlabel("Time")
    ax_dd.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path
