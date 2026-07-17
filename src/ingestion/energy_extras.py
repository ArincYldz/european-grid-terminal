"""Extra keyless data sources that widen the dashboard beyond price/generation.

Every endpoint here was verified live (2026-07-17) to work with NO API key.
Deliberately excluded, because they now gate access behind a key:
  - OpenChargeMap  -> 403 "You must specify an API key"
  - Ember          -> 403
  - Electricity Maps free tier -> 401, and only ONE zone, useless for 28.

Sources and their quirks:
  - Energy-Charts /co2eq, /cbpf, /installed_power  (Fraunhofer ISE, no key)
  - PVGIS /PVcalc  (EU JRC) — CLIMATOLOGY, so the result never changes:
    compute once, cache forever. No CORS, so it must be server-side.
  - OSM Overpass — has CORS but is slow (~35 s/country) and rate-limits hard,
    so callers should cache and tolerate failure.

Same anti-corruption-layer idea as the rest of src/ingestion: callers get
plain DataFrames/dicts and never learn which upstream produced them.
"""

from __future__ import annotations

import logging
import time

import pandas as pd
import requests

from .energy_charts import _get as _ec_get
from .exceptions import ApiPermanentError, ApiTransientError, DataQualityError

logger = logging.getLogger(__name__)

PVGIS_URL = "https://re.jrc.ec.europa.eu/api/v5_2/PVcalc"
OVERPASS_URL = "https://overpass-api.de/api/interpreter"

# Overpass is the flakiest dependency we have: be patient, then give up quietly.
_OVERPASS_TIMEOUT = (5, 180)
_OVERPASS_RETRIES = 3
_OVERPASS_BACKOFF_S = 20.0

# Overpass answers 406 to the requests default User-Agent. OSM's usage policy
# asks for an identifying agent anyway, so send a real one.
_OVERPASS_HEADERS = {
    "User-Agent": "EuropeanGridForecast/1.0 (open-source dashboard; OSM data via Overpass)",
}

# Cross-border payloads carry an aggregate row that is not a neighbour.
_CBPF_NON_COUNTRY = {"sum"}


# --------------------------------------------------------------------------
# Energy-Charts extras
# --------------------------------------------------------------------------
def fetch_carbon_intensity(country: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Grid carbon intensity, hourly (gCO2eq/kWh).

    Columns: target_time, co2eq_g_kwh. The raw feed is 15-minute; we resample
    to hourly to line up with the price and generation frames.
    """
    d = _ec_get("/co2eq", {"country": country, "start": start_date, "end": end_date})
    if "unix_seconds" not in d or "co2eq" not in d:
        raise DataQualityError("Energy-Charts co2eq: unexpected schema.")

    ts = pd.to_datetime(d["unix_seconds"], unit="s", utc=True)
    s = pd.Series(d["co2eq"], index=ts, name="co2eq_g_kwh", dtype="float64")
    s.index.name = "target_time"
    hourly = s.resample("1h").mean().reset_index()
    logger.info("Energy-Charts co2eq: %d hourly rows (%s)", len(hourly), country)
    return hourly


def fetch_cross_border_flows(country: str, start_date: str, end_date: str) -> list[dict]:
    """Mean net physical flow with each neighbour, in GW.

    Sign convention is the API's own (confirmed in its OpenAPI description):
    POSITIVE = import into `country`, NEGATIVE = export out of it. We keep that
    convention rather than inventing one, so the arrows cannot end up reversed.

    Returns [{"name", "net_gw", "direction"}], strongest flow first.
    """
    d = _ec_get("/cbpf", {"country": country, "start": start_date, "end": end_date})
    if "countries" not in d:
        raise DataQualityError("Energy-Charts cbpf: unexpected schema.")

    out = []
    for entry in d["countries"]:
        name = entry.get("name", "")
        if name.strip().lower() in _CBPF_NON_COUNTRY:
            continue
        vals = [v for v in entry.get("data", []) if v is not None]
        if not vals:
            continue
        net = sum(vals) / len(vals)
        out.append({
            "name": name,
            "net_gw": round(net, 3),
            "direction": "import" if net > 0 else "export",
        })
    out.sort(key=lambda r: abs(r["net_gw"]), reverse=True)
    logger.info("Energy-Charts cbpf: %d neighbours (%s)", len(out), country)
    return out


def fetch_installed_power(country: str) -> dict:
    """Installed capacity by production type, per year (GW).

    Returns {"years": [...], "types": {name: [gw per year]}}. The feed runs past
    today because it includes planned build-out; callers decide what to show.
    """
    d = _ec_get("/installed_power", {
        "country": country, "time_step": "yearly", "installation_decommission": "false",
    })
    if "time" not in d or "production_types" not in d:
        raise DataQualityError("Energy-Charts installed_power: unexpected schema.")

    types = {}
    for p in d["production_types"]:
        vals = p.get("data") or []
        # Drop types that are all-null for this country (feed is DE-shaped).
        if any(v is not None for v in vals):
            types[p["name"]] = vals
    logger.info("Energy-Charts installed_power: %d types (%s)", len(types), country)
    return {"years": d["time"], "types": types}


# --------------------------------------------------------------------------
# PVGIS — solar yield climatology (static: cache forever)
# --------------------------------------------------------------------------
def fetch_pv_yield(lat: float, lon: float, peakpower_kw: float = 1.0,
                   loss_pct: float = 14.0) -> dict | None:
    """Annual + monthly PV yield for one point, from PVGIS (EU JRC).

    Returns {"yearly_kwh_per_kwp", "monthly_kwh_per_kwp": [12]} or None when
    PVGIS has no radiation data for the point (i.e. it is at sea) — that is an
    expected outcome for a grid sweep, not an error worth raising.

    Uses PVGIS-SARAH2 satellite radiation, so this is measured climatology
    rather than a model guess. It does not change over time, which is exactly
    why the caller may cache it permanently.
    """
    params = {
        "lat": round(lat, 4), "lon": round(lon, 4),
        "peakpower": peakpower_kw, "loss": loss_pct,
        "outputformat": "json", "pvtechchoice": "crystSi",
        "mountingplace": "free", "optimalangles": 1,
    }
    try:
        r = requests.get(PVGIS_URL, params=params, timeout=(5, 60))
    except (requests.ConnectionError, requests.Timeout) as exc:
        raise ApiTransientError(f"PVGIS unreachable: {exc}") from exc

    if r.status_code == 400:
        return None  # off-grid / over sea
    if r.status_code == 429 or r.status_code >= 500:
        raise ApiTransientError(f"PVGIS {r.status_code}: {r.text[:120]}")
    if r.status_code >= 400:
        raise ApiPermanentError(f"PVGIS {r.status_code}: {r.text[:200]}")

    try:
        d = r.json()["outputs"]
    except (ValueError, KeyError) as exc:
        raise DataQualityError(f"PVGIS response not understood: {exc}") from exc

    yearly = d.get("totals", {}).get("fixed", {}).get("E_y")
    monthly = [m.get("E_m") for m in d.get("monthly", {}).get("fixed", [])]
    if yearly is None or len(monthly) != 12:
        return None
    return {
        "yearly_kwh_per_kwp": round(float(yearly) / peakpower_kw, 1),
        "monthly_kwh_per_kwp": [round(float(m) / peakpower_kw, 1) for m in monthly],
    }


# --------------------------------------------------------------------------
# OpenStreetMap Overpass — EV charging infrastructure
# --------------------------------------------------------------------------
def _overpass(query: str) -> dict:
    last_err = "unknown"
    for attempt in range(_OVERPASS_RETRIES):
        try:
            r = requests.post(OVERPASS_URL, data={"data": query},
                              headers=_OVERPASS_HEADERS, timeout=_OVERPASS_TIMEOUT)
        except (requests.ConnectionError, requests.Timeout) as exc:
            last_err = str(exc)
            time.sleep(_OVERPASS_BACKOFF_S * (attempt + 1))
            continue
        # 429 = too many slots, 504 = server busy. Both mean "come back later".
        if r.status_code in (429, 504) or r.status_code >= 500:
            last_err = f"HTTP {r.status_code}"
            time.sleep(_OVERPASS_BACKOFF_S * (attempt + 1))
            continue
        if r.status_code >= 400:
            raise ApiPermanentError(f"Overpass {r.status_code}: {r.text[:200]}")
        try:
            return r.json()
        except ValueError as exc:
            raise DataQualityError(f"Overpass JSON could not be parsed: {exc}") from exc
    raise ApiTransientError(f"Overpass exhausted retries: {last_err}")


def _count_from(payload: dict) -> int:
    for el in payload.get("elements", []):
        if el.get("type") == "count":
            return int(el.get("tags", {}).get("nodes", 0))
    raise DataQualityError("Overpass returned no count element.")


def fetch_ev_charger_count(iso2: str) -> dict:
    """Public EV charging points in a country, counted in OpenStreetMap.

    Returns {"total", "fast_dc"}. `fast_dc` counts stations tagged with a CCS
    (type2_combo) socket, the usual marker of a DC rapid charger.

    Two separate count queries rather than downloading the nodes: Germany alone
    has ~43k, which is far too much to ship to a browser and slow to parse.
    Counts are cheap and are all the map layer needs.
    """
    iso = iso2.upper()
    area = f'area["ISO3166-1"="{iso}"][admin_level=2]->.a;'
    total = _count_from(_overpass(
        f'[out:json][timeout:120];{area}node(area.a)["amenity"="charging_station"];out count;'))
    fast = _count_from(_overpass(
        f'[out:json][timeout:120];{area}'
        f'node(area.a)["amenity"="charging_station"]["socket:type2_combo"];out count;'))
    logger.info("Overpass EV (%s): %d total, %d fast DC", iso, total, fast)
    return {"total": total, "fast_dc": fast}
