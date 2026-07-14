"""ENTSO-E Transparency Platform collector (generation, load, day-ahead price).

We use the entsoe-py library (it handles the XML parsing) but wrap it in
our own exception classes and our two-timestamp (bitemporal) schema. The
layers above never know that "entsoe-py exists"; if the source changes
tomorrow, only this file changes (architectural answer: "anti-corruption
layer / adapter pattern").
"""

from __future__ import annotations

import logging
import os

import pandas as pd
import requests

from .exceptions import ApiPermanentError, ApiTransientError, DataQualityError

logger = logging.getLogger(__name__)

# ENTSO-E publishes actual generation/load data not in real time but with
# an operational delay (~1 hour for most items; post-settlement revisions
# also occur). Ignoring this delay in a backtest is a classic look-ahead
# bias.
ACTUALS_PUBLICATION_LAG = pd.Timedelta(hours=1)


def _get_client():
    """Validates the API key and builds an entsoe-py client."""
    api_key = os.environ.get("ENTSOE_API_KEY")
    if not api_key:
        raise ApiPermanentError(
            "ENTSOE_API_KEY is not set. Add your key to the .env file "
            "(see .env.example)."
        )
    from entsoe import EntsoePandasClient  # import here: don't load if no key
    return EntsoePandasClient(api_key=api_key)


def fetch_day_ahead_prices(
    country_code: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    """Day-ahead hourly prices (EUR/MWh).

    available_at logic: the DE-LU day-ahead auction closes at 12:00 CET on
    D-1, and results are published around 12:45 CET. So the price for ALL
    hours of delivery day D becomes known at the same instant, ~13:00 on
    D-1. Assuming a target_time-based "known at that hour" would be wrong.
    """
    client = _get_client()
    try:
        series = client.query_day_ahead_prices(country_code, start=start, end=end)
    except requests.HTTPError as exc:
        _reraise_http(exc)
    except (requests.ConnectionError, requests.Timeout) as exc:
        raise ApiTransientError(f"ENTSO-E access error: {exc}") from exc

    if series is None or series.empty:
        raise DataQualityError(f"ENTSO-E returned an empty price series: {country_code}")

    df = series.rename("price_eur_mwh").to_frame()
    df.index.name = "target_time"
    df = df.reset_index()
    df["target_time"] = df["target_time"].dt.tz_convert("UTC")

    # The price is published on the day before delivery, ~12:45 CET. For
    # simplicity we use 13:00 CET — rounding in the conservative (late)
    # direction is safe: knowing data LATER than reality costs nothing,
    # knowing it EARLIER (leakage) costs money.
    delivery_day = df["target_time"].dt.tz_convert("Europe/Berlin").dt.normalize()
    publish_local = delivery_day - pd.Timedelta(days=1) + pd.Timedelta(hours=13)
    df["available_at"] = publish_local.dt.tz_convert("UTC")
    df["source"] = "entsoe_day_ahead"
    logger.info("Day-ahead price: %d rows (%s)", len(df), country_code)
    return df


def fetch_actual_generation(
    country_code: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    """Actual generation (by source: wind, solar, gas...). MW.

    The TARGET variable of the generation-forecast model (Step A) comes
    from here.
    """
    client = _get_client()
    try:
        raw = client.query_generation(country_code, start=start, end=end, psr_type=None)
    except requests.HTTPError as exc:
        _reraise_http(exc)
    except (requests.ConnectionError, requests.Timeout) as exc:
        raise ApiTransientError(f"ENTSO-E access error: {exc}") from exc

    if raw is None or raw.empty:
        raise DataQualityError(f"ENTSO-E returned empty generation data: {country_code}")

    # entsoe-py sometimes returns a MultiIndex column of the form
    # (generation_type, Actual Aggregated/Consumption); flatten to one level.
    if isinstance(raw.columns, pd.MultiIndex):
        raw = raw.xs("Actual Aggregated", axis=1, level=1, drop_level=True)

    df = raw.copy()
    df.index.name = "target_time"
    df = df.reset_index()
    df["target_time"] = df["target_time"].dt.tz_convert("UTC")
    df["available_at"] = df["target_time"] + ACTUALS_PUBLICATION_LAG
    df["source"] = "entsoe_actual_generation"
    logger.info("Actual generation: %d rows (%s)", len(df), country_code)
    return df


def fetch_load(
    country_code: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    """Actual total load / demand (MW).

    Feeds the residual-load feature (demand - renewable generation), the
    physical driver of price. NOTE: for a leakage-safe live forecast you
    would use the ENTSO-E day-ahead LOAD FORECAST, not the realized load;
    the realized series here is for training the target side and backtest
    settlement. entsoe-py exposes `query_load_forecast` for the forecast.
    """
    client = _get_client()
    try:
        raw = client.query_load(country_code, start=start, end=end)
    except requests.HTTPError as exc:
        _reraise_http(exc)
    except (requests.ConnectionError, requests.Timeout) as exc:
        raise ApiTransientError(f"ENTSO-E access error: {exc}") from exc

    if raw is None or raw.empty:
        raise DataQualityError(f"ENTSO-E returned empty load data: {country_code}")

    # query_load returns a DataFrame with an 'Actual Load' column (or a Series
    # in older versions); normalize to a single 'demand_mw' column.
    if isinstance(raw, pd.Series):
        series = raw
    else:
        col = "Actual Load" if "Actual Load" in raw.columns else raw.columns[0]
        series = raw[col]

    df = series.rename("demand_mw").to_frame()
    df.index.name = "target_time"
    df = df.reset_index()
    df["target_time"] = df["target_time"].dt.tz_convert("UTC")
    df["available_at"] = df["target_time"] + ACTUALS_PUBLICATION_LAG
    df["source"] = "entsoe_actual_load"
    logger.info("Actual load: %d rows (%s)", len(df), country_code)
    return df


def _reraise_http(exc: requests.HTTPError) -> None:
    """Classifies an HTTP error as transient/permanent and re-raises."""
    status = exc.response.status_code if exc.response is not None else 0
    if status in (401, 403):
        raise ApiPermanentError(f"ENTSO-E auth error ({status}): check the API key.") from exc
    if status == 400:
        raise ApiPermanentError("ENTSO-E rejected the request (400): bad parameters.") from exc
    raise ApiTransientError(f"ENTSO-E HTTP {status}") from exc
