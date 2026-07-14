"""Open-Meteo collector (weather: historical + forecast).

CORE RULE AGAINST DATA LEAKAGE (the single most important idea in this file):
On every row we carry TWO timestamps:
  - target_time  : the instant the data BELONGS TO   (e.g. "wind at 14:00 tomorrow")
  - available_at : the instant the data IS IN OUR HANDS (e.g. "the 06:00 forecast run today")

When forecasting at decision time T, only rows with `available_at <= T` may
enter the feature set. Every join made without this filter is a potential
look-ahead bias.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .exceptions import ApiPermanentError, ApiTransientError, DataQualityError

logger = logging.getLogger(__name__)

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# Hourly variables needed for wind + solar generation forecasting
HOURLY_VARS = [
    "temperature_2m",
    "wind_speed_100m",        # closest level to turbine hub height
    "wind_direction_100m",
    "shortwave_radiation",    # main driver of PV generation (GHI)
    "cloud_cover",
]

# ERA5 reanalysis is published with a ~5 day delay. If we label this
# "historical actual" data with an available_at that ignores the delay, in a
# backtest we would act as if we knew data that had not yet been published.
ERA5_PUBLICATION_LAG = pd.Timedelta(days=5)

REQUEST_TIMEOUT = (5, 30)  # (connect, read) seconds — a request without a timeout is forbidden


def _build_session() -> requests.Session:
    """HTTP session with retry + exponential backoff.

    On transient errors (5xx, 429) it retries 3 times, with widening gaps
    (2s, 4s, 8s). It does NOT retry on 4xx errors because the request is
    malformed on our side; retrying it would be malformed too.
    """
    retry = Retry(
        total=3,
        backoff_factor=2,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def _get_json(url: str, params: dict) -> dict:
    """Single point of HTTP access: timeout, retry and error classification."""
    session = _build_session()
    try:
        response = session.get(url, params=params, timeout=REQUEST_TIMEOUT)
    except (requests.ConnectionError, requests.Timeout) as exc:
        raise ApiTransientError(f"Open-Meteo access error: {exc}") from exc

    if response.status_code >= 500 or response.status_code == 429:
        # We land here if the retry adapter gave up -> the layer above can
        # switch to a failover source or raise an alert.
        raise ApiTransientError(f"Open-Meteo {response.status_code}: {response.text[:200]}")
    if response.status_code >= 400:
        raise ApiPermanentError(f"Open-Meteo {response.status_code}: {response.text[:200]}")

    try:
        return response.json()
    except ValueError as exc:
        raise DataQualityError(f"Open-Meteo JSON could not be parsed: {exc}") from exc


def _to_frame(payload: dict) -> pd.DataFrame:
    """Validates the API response and turns it into a tidy DataFrame."""
    hourly = payload.get("hourly")
    if not hourly or "time" not in hourly:
        raise DataQualityError("No 'hourly' block in the response — the schema may have changed.")

    df = pd.DataFrame(hourly)
    df = df.rename(columns={"time": "target_time"})
    # We asked for timezone=UTC; still, we force tz-aware. In an electricity
    # market a naive timestamp = silently shifted data across DST transitions
    # (the 23/25-hour days).
    df["target_time"] = pd.to_datetime(df["target_time"], utc=True)

    missing = [v for v in HOURLY_VARS if v not in df.columns]
    if missing:
        raise DataQualityError(f"Expected columns are missing: {missing}")
    if df.empty:
        raise DataQualityError("API returned empty data.")
    return df


def fetch_historical_weather(
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Historical ACTUAL weather (ERA5 reanalysis).

    CAUTION — interview trap: this data is "what actually happened at that
    hour". It is used for work close to the TARGET side of the generation
    model (e.g. explaining realized generation); but it cannot be used as an
    INPUT to the price model, because in real life at decision time we hold
    a FORECAST, not the realized value. In training, inputs must also come
    from forecast data (see fetch_forecast_weather / Previous Runs API) —
    otherwise train/serve skew occurs.
    """
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(HOURLY_VARS),
        "timezone": "UTC",
        "wind_speed_unit": "ms",
    }
    payload = _get_json(ARCHIVE_URL, params)
    df = _to_frame(payload)

    # Reanalysis data is published ~5 days AFTER target_time.
    df["available_at"] = df["target_time"] + ERA5_PUBLICATION_LAG
    df["source"] = "open_meteo_era5"
    logger.info("ERA5: %d rows fetched (%s -> %s)", len(df), start_date, end_date)
    return df


def fetch_weather_window(
    latitude: float,
    longitude: float,
    past_days: int = 92,
    forecast_days: int = 2,
) -> pd.DataFrame:
    """Continuous hourly weather window: recent PAST + near FUTURE, one source.

    Why one endpoint for both? The forecast API's `past_days` returns the same
    model's output for recent history, so training weather and serving weather
    come from the SAME distribution (no archive-vs-forecast model skew), and
    there is no 5-day ERA5 publication gap right before "now" — exactly what a
    live forecasting job needs. Max past_days = 92.
    """
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "past_days": min(past_days, 92),
        "forecast_days": forecast_days,
        "hourly": ",".join(HOURLY_VARS),
        "timezone": "UTC",
        "wind_speed_unit": "ms",
    }
    payload = _get_json(FORECAST_URL, params)
    df = _to_frame(payload)
    df["available_at"] = datetime.now(timezone.utc)
    df["source"] = "open_meteo_window"
    logger.info("Weather window: %d rows (past %dd + next %dd)", len(df), past_days, forecast_days)
    return df


def fetch_forecast_weather(
    latitude: float,
    longitude: float,
    forecast_days: int = 3,
) -> pd.DataFrame:
    """Forward-looking weather FORECAST (input for the live intraday signal).

    available_at = now (the moment it was fetched). We write this snapshot to
    the DB as a SEPARATE record on every fetch (overwriting via upsert is
    FORBIDDEN): for the same target_time, the 06:00 and 12:00 runs give
    different forecasts; which one we knew depends on the decision time. If
    past forecast runs are needed, use the Open-Meteo "Previous Runs API"
    (previous-runs-api.open-meteo.com).
    """
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "forecast_days": forecast_days,
        "hourly": ",".join(HOURLY_VARS),
        "timezone": "UTC",
        "wind_speed_unit": "ms",
    }
    payload = _get_json(FORECAST_URL, params)
    df = _to_frame(payload)

    df["available_at"] = datetime.now(timezone.utc)
    df["source"] = "open_meteo_forecast"
    logger.info("Forecast: %d rows fetched (%d days)", len(df), forecast_days)
    return df
