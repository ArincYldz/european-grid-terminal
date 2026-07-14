"""Step 1 demo: runs the data-ingestion layer end-to-end.

Usage:
    pip install -r requirements.txt
    python run_ingestion_demo.py

The ENTSO-E part needs ENTSOE_API_KEY in the .env file; without it that step
is skipped (Open-Meteo works without a key).
"""

import logging

import pandas as pd
from dotenv import load_dotenv

from src.ingestion import (
    ApiPermanentError,
    ApiTransientError,
    DataQualityError,
    detect_gaps,
    fetch_day_ahead_prices,
    fetch_forecast_weather,
    fetch_historical_weather,
    impute_physical_signal,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("demo")

# Northern Germany (near Bremen) — a region dense with wind farms
LAT, LON = 53.07, 8.80


def main() -> None:
    # --- 1) Historical weather (the target side for model training) ---
    hist = fetch_historical_weather(LAT, LON, "2025-06-01", "2025-06-30")
    print("\nERA5 historical data (first 3 rows):")
    print(hist.head(3).to_string(index=False))

    # --- 2) Live weather forecast (input for the intraday signal) ---
    fcst = fetch_forecast_weather(LAT, LON, forecast_days=2)
    print("\nWeather forecast snapshot (first 3 rows):")
    print(fcst.head(3).to_string(index=False))

    # --- 3) Quality layer: gap detection + cautious imputation ---
    value_cols = ["temperature_2m", "wind_speed_100m", "shortwave_radiation"]
    gridded = detect_gaps(hist, freq="1h")
    cleaned = impute_physical_signal(gridded, value_cols)
    print(f"\nNumber of imputed intervals: {int(cleaned['is_imputed'].sum())}")

    # --- 4) ENTSO-E day-ahead price (if an API key is present) ---
    try:
        start = pd.Timestamp("2025-06-01", tz="Europe/Berlin")
        end = pd.Timestamp("2025-06-08", tz="Europe/Berlin")
        prices = fetch_day_ahead_prices("DE_LU", start, end)
        print("\nDay-ahead prices (first 3 rows):")
        print(prices.head(3).to_string(index=False))
        neg = (prices["price_eur_mwh"] < 0).mean()
        print(f"Share of negative-price hours: {neg * 100:.1f}%")
    except ApiPermanentError as exc:
        log.warning("ENTSO-E step skipped: %s", exc)
    except (ApiTransientError, DataQualityError) as exc:
        log.error("ENTSO-E transient/data error: %s", exc)


if __name__ == "__main__":
    load_dotenv()
    main()
