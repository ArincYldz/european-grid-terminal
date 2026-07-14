"""Step 3B demo: cascade (3A->3B), quantile price forecast + negative-price risk.

    pip install -r requirements.txt
    python run_price_demo.py

Also shows how conformal calibration (ACI) closes the quantile coverage gap.
Data source: real ENTSO-E + Open-Meteo if ENTSOE_API_KEY is set, else a
clearly-labelled synthetic fallback (see src/pipeline/data_assembly.py).
"""

import logging

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from src.features import build_generation_feature_matrix
from src.features.price_features import build_price_feature_matrix
from src.models import (
    AdaptiveConformalForecaster,
    NegativePriceClassifier,
    QuantileForecaster,
    empirical_coverage,
    oof_generation_predictions,
    pinball_loss,
)
from src.pipeline import DatasetConfig, assemble_dataset

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("demo3b")

WEATHER_COLS = ["target_time", "temperature_2m", "wind_speed_100m",
                "wind_direction_100m", "shortwave_radiation", "cloud_cover"]


def main() -> None:
    df = assemble_dataset(DatasetConfig())

    # --- 3A features + target ---
    gen_feat = build_generation_feature_matrix(df[WEATHER_COLS].copy())
    gen_feat["generation_mw"] = df["generation_mw"].to_numpy()
    gen_feat = gen_feat.dropna(subset=["generation_mw"]).reset_index(drop=True)

    # --- CASCADE: 3A's OOF (leakage-safe) generation forecast ---
    log.info("Producing 3A out-of-fold generation forecast (cascade leakage shield)...")
    gen_feat["predicted_generation_mw"] = oof_generation_predictions(gen_feat, n_splits=5)

    # --- 3B input frame: demand FORECAST + OOF predicted generation + gas ---
    rng = np.random.default_rng(1)
    demand = df["demand_mw"].to_numpy()
    pf = pd.DataFrame({
        "target_time": df["target_time"].to_numpy(),
        "demand_forecast_mw": demand * rng.normal(1.0, 0.03, len(demand)),
        "predicted_generation_mw": gen_feat["predicted_generation_mw"].to_numpy(),
        "gas_price_eur_mwh": df["gas_price_eur_mwh"].to_numpy(),
        "price_eur_mwh": df["price_eur_mwh"].to_numpy(),
    })
    price_df = build_price_feature_matrix(pf).dropna().reset_index(drop=True)
    log.info("Price feature matrix: %d rows.", len(price_df))

    neg_rate = (price_df["price_eur_mwh"] < 0).mean()
    print(f"\nShare of negative-price hours: {neg_rate*100:.1f}%")

    # --- Temporal train/test split ---
    cut = int(len(price_df) * 0.8)
    train, test = price_df.iloc[:cut].copy(), price_df.iloc[cut:].copy()
    y = test["price_eur_mwh"].to_numpy()

    # --- Quantile price forecast ---
    qf = QuantileForecaster(quantiles=(0.1, 0.5, 0.9)).fit(train)
    q_pred = qf.predict(test)

    print("\n=== Quantile price forecast (test set) ===")
    for q, col in [(0.1, "p10"), (0.5, "p50"), (0.9, "p90")]:
        print(f"  Pinball@{int(q*100)}: {pinball_loss(y, q_pred[col].to_numpy(), q):.2f} EUR/MWh")
    covered = ((y >= q_pred["p10"]) & (y <= q_pred["p90"])).mean()
    print(f"  P10-P90 coverage: {covered*100:.1f}% (nominal ~80%)")

    # --- Conformal calibration (ACI) closes the coverage gap ---
    aci = AdaptiveConformalForecaster(lo=0.1, hi=0.9, gamma=0.03).fit(train)
    qa = aci.predict_adaptive(test)
    cov_aci = empirical_coverage(y, qa["p10"].to_numpy(), qa["p90"].to_numpy())
    print(f"  P10-P90 coverage after ACI: {cov_aci*100:.1f}% (target 80%)")

    # --- Negative-price risk (calibrated) ---
    clf = NegativePriceClassifier().fit(train)
    ev = clf.evaluate(test)
    print("\n=== Negative-price risk (test set) ===")
    print(f"  Base rate       : {ev['base_rate']*100:.1f}%")
    print(f"  Brier score     : {ev['brier']:.4f} (lower=better)")
    print(f"  Of the {ev['n_high_risk']} hours we called 'high risk' (>=50%), "
          f"realized negative rate: {ev['realized_rate_when_high']*100:.1f}%")


if __name__ == "__main__":
    load_dotenv()
    main()
