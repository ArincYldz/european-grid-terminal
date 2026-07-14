"""Energy-Charts collector (Fraunhofer ISE) — REAL data, NO API key needed.

Why this source? ENTSO-E requires a (free but registration-gated) API key.
The Energy-Charts public API (api.energy-charts.info) exposes the same
fundamentals — generation by source, load, day-ahead price — for European
bidding zones with NO key. This lets the project run on REAL data immediately,
while ENTSO-E remains the primary source once a key is available.

Same anti-corruption-layer idea as entsoe_client: the layers above do not know
which source produced the data; only this file changes if the source changes.
"""

from __future__ import annotations

import logging
import time

import pandas as pd
import requests

from .exceptions import ApiPermanentError, ApiTransientError, DataQualityError

logger = logging.getLogger(__name__)

BASE_URL = "https://api.energy-charts.info"
REQUEST_TIMEOUT = (5, 60)  # (connect, read) — generation can be a large payload

# Renewable production types we sum into the generation target.
_RENEWABLE_TYPES = ("Wind onshore", "Wind offshore", "Solar")

# The API rate-limits aggressively (429 even at modest request rates), so we
# back off and retry before giving up. Empirically a few seconds is enough.
_MAX_RETRIES = 4
_BACKOFF_BASE_S = 4.0


def _get(path: str, params: dict) -> dict:
    last_err = "unknown"
    for attempt in range(_MAX_RETRIES):
        try:
            r = requests.get(BASE_URL + path, params=params, timeout=REQUEST_TIMEOUT)
        except (requests.ConnectionError, requests.Timeout) as exc:
            last_err = str(exc)
            time.sleep(_BACKOFF_BASE_S * (attempt + 1))
            continue
        if r.status_code == 429 or r.status_code >= 500:
            last_err = f"{r.status_code}: {r.text[:120]}"
            time.sleep(_BACKOFF_BASE_S * (attempt + 1))
            continue
        if r.status_code == 404:
            # Valid code but no content for this range — permanent for this query.
            raise ApiPermanentError(f"Energy-Charts 404 (no content): {path} {params}")
        if r.status_code >= 400:
            raise ApiPermanentError(f"Energy-Charts {r.status_code}: {r.text[:200]}")
        try:
            return r.json()
        except ValueError as exc:
            raise DataQualityError(f"Energy-Charts JSON could not be parsed: {exc}") from exc
    raise ApiTransientError(f"Energy-Charts exhausted retries: {last_err}")


def fetch_power(country: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Generation (renewable target) + load, resampled to HOURLY (MW).

    Returns columns: target_time, generation_mw (wind+solar), demand_mw (load).
    The raw feed is 15-minute; we resample to hourly means to match the
    weather grid and the price series.
    """
    d = _get("/public_power", {"country": country, "start": start_date, "end": end_date})
    if "unix_seconds" not in d or "production_types" not in d:
        raise DataQualityError("Energy-Charts public_power: unexpected schema.")

    ts = pd.to_datetime(d["unix_seconds"], unit="s", utc=True)
    prod = pd.DataFrame({p["name"]: p["data"] for p in d["production_types"]}, index=ts)
    prod.index.name = "target_time"

    have_renewables = [c for c in _RENEWABLE_TYPES if c in prod.columns]
    if not have_renewables:
        raise DataQualityError(f"No wind/solar columns in feed: {list(prod.columns)}")
    if "Load" not in prod.columns:
        raise DataQualityError("No 'Load' column in Energy-Charts feed.")

    wind_cols = [c for c in have_renewables if "Wind" in c]
    solar_cols = [c for c in have_renewables if "Solar" in c]

    out = pd.DataFrame(index=prod.index)
    out["generation_mw"] = prod[have_renewables].sum(axis=1, min_count=1)
    out["wind_mw"] = prod[wind_cols].sum(axis=1, min_count=1) if wind_cols else 0.0
    out["solar_mw"] = prod[solar_cols].sum(axis=1, min_count=1) if solar_cols else 0.0
    out["demand_mw"] = prod["Load"]

    hourly = out.resample("1h").mean().reset_index()
    logger.info("Energy-Charts power: %d hourly rows (%s, %s..%s)",
                len(hourly), country, start_date, end_date)
    return hourly


def fetch_price(bzn: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Day-ahead price, hourly (EUR/MWh). Columns: target_time, price_eur_mwh."""
    d = _get("/price", {"bzn": bzn, "start": start_date, "end": end_date})
    if "unix_seconds" not in d or "price" not in d:
        raise DataQualityError("Energy-Charts price: unexpected schema.")

    ts = pd.to_datetime(d["unix_seconds"], unit="s", utc=True)
    s = pd.Series(d["price"], index=ts, name="price_eur_mwh")
    s.index.name = "target_time"
    hourly = s.resample("1h").mean().reset_index()
    logger.info("Energy-Charts price: %d hourly rows (%s)", len(hourly), bzn)
    return hourly
