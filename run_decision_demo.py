"""Step 4 demo: forecasts -> signals -> cost-aware backtest -> risk report.

    python run_decision_demo.py

Full 3A->3B->4 chain: weather -> generation forecast (OOF) -> conformalized
price quantiles (ACI) + negative-price risk -> expected-value signals ->
walk-forward backtest. The smart strategy is measured against a naive
baseline with identical cost accounting, then reported with risk-adjusted
metrics (Sharpe / max drawdown / CVaR) and a P&L equity-curve plot.

Data source: real ENTSO-E + Open-Meteo if ENTSOE_API_KEY is set, otherwise a
clearly-labelled synthetic fallback (see src/pipeline/data_assembly.py).
"""

import logging
import os

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from src.backtest import compute_risk_metrics, plot_equity_curves, run_backtest, run_naive_baseline
from src.decision import calibrate_params_from_history
from src.features import build_generation_feature_matrix
from src.features.price_features import build_price_feature_matrix
from src.models import (
    AdaptiveConformalForecaster,
    NegativePriceClassifier,
    empirical_coverage,
    oof_generation_predictions,
)
from src.pipeline import DatasetConfig, assemble_dataset

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("demo4")

WEATHER_COLS = ["target_time", "temperature_2m", "wind_speed_100m",
                "wind_direction_100m", "shortwave_radiation", "cloud_cover"]
REPORT_DIR = "reports"

# Our asset is a single producer, modelled as a small share of the national
# renewable fleet (a large wind+solar portfolio). ~1% of Germany's ~21 GW mean
# renewable output -> ~200 MW mean, a realistic single-operator scale.
ASSET_FRACTION = 0.01


def build_decision_frame() -> pd.DataFrame:
    df = assemble_dataset(DatasetConfig())

    # 3A: generation features from WEATHER ONLY (never pass price/demand here
    # or they leak into the generation model as features).
    gen_feat = build_generation_feature_matrix(df[WEATHER_COLS].copy())
    gen_feat["generation_mw"] = df["generation_mw"].to_numpy()
    gen_feat = gen_feat.dropna(subset=["generation_mw"]).reset_index(drop=True)

    log.info("3A out-of-fold generation forecast (cascade leakage shield)...")
    gen_feat["predicted_generation_mw"] = oof_generation_predictions(gen_feat, n_splits=5)

    # 3B input frame: demand FORECAST + predicted generation (OOF) + gas.
    rng = np.random.default_rng(1)
    demand = df["demand_mw"].to_numpy()
    pf = pd.DataFrame({
        "target_time": df["target_time"].to_numpy(),
        "demand_forecast_mw": demand * rng.normal(1.0, 0.03, len(demand)),
        "predicted_generation_mw": gen_feat["predicted_generation_mw"].to_numpy(),
        "actual_generation_mw": df["generation_mw"].to_numpy(),
        "gas_price_eur_mwh": df["gas_price_eur_mwh"].to_numpy(),
        "price_eur_mwh": df["price_eur_mwh"].to_numpy(),
    })
    price_df = build_price_feature_matrix(pf).dropna().reset_index(drop=True)
    return price_df


def main() -> None:
    os.makedirs(REPORT_DIR, exist_ok=True)
    price_df = build_decision_frame()

    cut = int(len(price_df) * 0.8)
    train, test = price_df.iloc[:cut].copy(), price_df.iloc[cut:].copy()

    log.info("Fitting conformal quantiles (ACI) + negative-price risk...")
    aci = AdaptiveConformalForecaster(lo=0.1, hi=0.9, gamma=0.03).fit(train)
    q = aci.predict_adaptive(test)
    clf = NegativePriceClassifier().fit(train)

    y = test["price_eur_mwh"].to_numpy()
    cov = empirical_coverage(y, q["p10"].to_numpy(), q["p90"].to_numpy())
    print(f"\nConformal (ACI) P10-P90 coverage: {cov*100:.1f}% (target 80%)")

    # OUR asset is a single price-taking producer, NOT the whole national fleet.
    # The price MODEL above stays on unscaled national fundamentals (the market
    # price depends on national residual load); only OUR generation is scaled to
    # a single asset for the P&L. ASSET_FRACTION ~ share of the national fleet.
    decision_df = pd.DataFrame({
        "target_time": test["target_time"].to_numpy(),
        "p10": q["p10"].to_numpy(), "p50": q["p50"].to_numpy(), "p90": q["p90"].to_numpy(),
        "neg_risk": clf.predict_risk(test),
        "forecast_generation_mwh": test["predicted_generation_mw"].to_numpy() * ASSET_FRACTION,
        "actual_generation_mwh": test["actual_generation_mw"].to_numpy() * ASSET_FRACTION,
        "realized_price": y,
    })

    # Calibrate economic parameters from the TRAIN price history (no test peeking)
    # and size the battery relative to OUR asset's mean generation.
    mean_gen_asset = float(decision_df["actual_generation_mwh"].mean())
    params = calibrate_params_from_history(
        train["price_eur_mwh"].to_numpy(),
        mean_generation_mw=mean_gen_asset,
        subsidy_eur_mwh=0.0,            # merchant asset -> curtail floor at 0
        recovery_quantile=0.80,
    )
    print(f"\nCalibrated from train: recovery(sell-high)={params.expected_recovery_price:.0f} EUR/MWh, "
          f"battery {params.storage_power_mw:.0f} MW / {params.storage_capacity_mwh:.0f} MWh, "
          f"asset mean gen {mean_gen_asset:.0f} MW")

    strat = run_backtest(decision_df, params)
    naive = run_naive_baseline(decision_df, params)

    print("\n==================== NAIVE BASELINE (always sell) ===============")
    print(naive.summary())
    print("\n==================== SMART STRATEGY =============================")
    print(strat.summary())

    delta = strat.total_pnl - naive.total_pnl
    pct = delta / abs(naive.total_pnl) * 100 if naive.total_pnl else float("nan")
    print("\n==================== COMPARISON =================================")
    print(f"Strategy - Baseline = {delta:,.0f} EUR  ({pct:+.1f}%)")

    # ---- Risk-adjusted report ----
    rm_strat = compute_risk_metrics(strat.ledger["pnl"])
    rm_naive = compute_risk_metrics(naive.ledger["pnl"])
    print("\n==================== RISK REPORT — STRATEGY =====================")
    print(rm_strat.summary())
    print("\n==================== RISK REPORT — BASELINE =====================")
    print(rm_naive.summary())

    out = plot_equity_curves(
        {"Smart strategy": strat.ledger, "Naive baseline": naive.ledger},
        os.path.join(REPORT_DIR, "equity_curve.png"),
    )
    print(f"\nEquity-curve plot saved to: {out}")


if __name__ == "__main__":
    load_dotenv()
    main()
