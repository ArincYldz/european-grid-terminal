"""Step 2 demo: Ingestion -> Storage end-to-end.

First start TimescaleDB:
    docker compose up -d
Then:
    pip install -r requirements.txt
    python run_storage_demo.py

If the DB is not up the script skips in a controlled way (like the ENTSO-E
pattern in Step 1) — it never silently produces a wrong result.
"""

import logging

import pandas as pd
from dotenv import load_dotenv

from src.ingestion import fetch_forecast_weather, fetch_historical_weather
from src.storage import TimescaleStore, to_long_format
from src.storage.schema import series_meta_from_long

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("demo2")

LAT, LON = 53.07, 8.80
ENTITY = "bremen"
VALUE_COLS = ["temperature_2m", "wind_speed_100m", "shortwave_radiation", "cloud_cover"]
UNITS = {
    "temperature_2m": "C",
    "wind_speed_100m": "m/s",
    "shortwave_radiation": "W/m2",
    "cloud_cover": "%",
}


def main() -> None:
    try:
        store = TimescaleStore()
    except ValueError as exc:
        log.warning("Storage step skipped (no DATABASE_URL): %s", exc)
        return

    if not store.ping():
        log.warning("Could not connect to TimescaleDB. Did `docker compose up -d` run?")
        return

    # --- 1) Fetch and write a forecast snapshot (to demonstrate bitemporal) ---
    # In reality these are fetched at different times by cron; here we fetch
    # them back-to-back, writing a different available_at for the same target_time.
    for _ in range(1):
        fcst = fetch_forecast_weather(LAT, LON, forecast_days=2)
        long_fcst = to_long_format(fcst, ENTITY, VALUE_COLS)
        store.upsert_series_meta(series_meta_from_long(long_fcst, ENTITY, UNITS))
        store.upsert_measurements(long_fcst)

    # --- 2) Also write the historical realized data ---
    hist = fetch_historical_weather(LAT, LON, "2025-06-01", "2025-06-07")
    long_hist = to_long_format(hist, ENTITY, VALUE_COLS)
    store.upsert_series_meta(series_meta_from_long(long_hist, ENTITY, UNITS))
    n = store.upsert_measurements(long_hist)
    log.info("Total historical rows written: %d", n)

    # --- 3) Idempotence proof: rewrite the same data, row count must NOT grow ---
    before = _row_count(store)
    store.upsert_measurements(long_hist)
    after = _row_count(store)
    print(f"\nIdempotence test: {before} rows before, {after} rows after re-writing "
          f"-> {'PASS (no growth)' if before == after else 'FAIL (duplicates!)'}")

    # --- 4) Leakage-safe as-of read ---
    keys = [f"{ENTITY}.wind_speed_100m", f"{ENTITY}.temperature_2m"]
    decision = pd.Timestamp("2025-06-10", tz="UTC")  # historical data is now "known"
    asof = store.read_as_of(
        keys,
        decision_time=decision,
        start=pd.Timestamp("2025-06-01", tz="UTC"),
        end=pd.Timestamp("2025-06-02", tz="UTC"),
    )
    print(f"\nas-of read (known as of {decision.date()}), first 4 rows:")
    print(asof.head(4).to_string(index=False))


def _row_count(store: TimescaleStore) -> int:
    from sqlalchemy import text

    with store.engine.connect() as conn:
        return conn.execute(text("SELECT count(*) FROM measurements")).scalar_one()


if __name__ == "__main__":
    load_dotenv()
    main()
