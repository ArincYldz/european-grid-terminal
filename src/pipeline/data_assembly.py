"""Single source of truth for building the training/decision frame.

This centralizes logic that used to be duplicated across the demo scripts.
It assembles one aligned hourly DataFrame from the data sources, choosing
between the REAL data path (ENTSO-E + Open-Meteo) and a clearly-labelled
SYNTHETIC fallback.

Source priority (real data by default, no key required):
    1. ENTSO-E        — used when ENTSOE_API_KEY is set (official, key-gated).
    2. Energy-Charts  — REAL data with NO key (Fraunhofer ISE). The default
                        real source when no ENTSO-E key is present.
    3. Synthetic      — physics-flavoured stand-ins (src.models.synthetic*),
                        the last resort (offline / no internet / USE_SYNTHETIC=1).

In all real paths:
    - weather        : Open-Meteo ERA5 archive (free, no key)
    - generation_mw  : renewable generation (wind + solar)
    - demand_mw      : total load
    - price_eur_mwh  : day-ahead price
    - gas_price      : EXTERNAL commodity feed (not in either electricity API) —
                       still a documented placeholder even in real mode; see NOTE.

Switching: USE_SYNTHETIC=1 forces the synthetic fallback. Otherwise ENTSO-E is
used if a key exists, else Energy-Charts (real, keyless); if that source is
unreachable (no internet), it falls back to synthetic with a warning.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import pandas as pd

from src.ingestion import fetch_historical_weather

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DatasetConfig:
    latitude: float = 53.07
    longitude: float = 8.80
    country_code: str = "DE_LU"       # ENTSO-E bidding zone for the region
    ec_country: str = "de"            # Energy-Charts country code (ISO-2, lowercase)
    ec_bzn: str = "DE-LU"             # Energy-Charts bidding-zone code
    start_date: str = "2024-01-01"
    end_date: str = "2024-12-31"


def _force_synthetic() -> bool:
    return os.environ.get("USE_SYNTHETIC") == "1"


def _has_entsoe_key() -> bool:
    key = os.environ.get("ENTSOE_API_KEY")
    return bool(key) and "buraya" not in key and "your-" not in key and "put-your" not in key


def _aggregate_renewables(gen: pd.DataFrame) -> pd.Series:
    """Sum wind + solar columns from ENTSO-E generation into one MW series."""
    cols = [c for c in gen.columns if any(k in str(c) for k in ("Wind", "Solar"))]
    if not cols:
        raise ValueError(f"No wind/solar columns found in generation frame: {list(gen.columns)}")
    return gen[cols].sum(axis=1).rename("generation_mw")


def _assemble_real(cfg: DatasetConfig) -> pd.DataFrame:
    """Assemble the frame from live ENTSO-E + Open-Meteo sources.

    NOTE: cannot be exercised without an ENTSOE_API_KEY. The code path is
    complete and import-clean; verify it once you add your key to .env.
    """
    from src.ingestion import fetch_actual_generation, fetch_day_ahead_prices, fetch_load
    from src.models.synthetic_price import synthetic_gas_price  # gas: external feed placeholder

    start = pd.Timestamp(cfg.start_date, tz="UTC")
    end = pd.Timestamp(cfg.end_date, tz="UTC") + pd.Timedelta(days=1)

    weather = fetch_historical_weather(cfg.latitude, cfg.longitude, cfg.start_date, cfg.end_date)

    gen_raw = fetch_actual_generation(cfg.country_code, start, end)
    gen = gen_raw[["target_time"]].copy()
    gen["generation_mw"] = _aggregate_renewables(gen_raw.drop(columns=["target_time", "available_at", "source"]))

    load = fetch_load(cfg.country_code, start, end)[["target_time", "demand_mw"]]
    price = fetch_day_ahead_prices(cfg.country_code, start, end)[["target_time", "price_eur_mwh"]]

    df = (
        weather.drop(columns=["available_at", "source"], errors="ignore")
        .merge(gen, on="target_time", how="inner")
        .merge(load, on="target_time", how="inner")
        .merge(price, on="target_time", how="inner")
    )
    # Gas is a separate commodity (TTF); ENTSO-E does not provide it. Until a
    # real commodity feed (e.g. a Kaggle/API series) is wired in, use the
    # physics-flavoured placeholder so residual-load pricing still has a fuel
    # signal. Clearly a stand-in, even on the real path.
    df["gas_price_eur_mwh"] = synthetic_gas_price(df).to_numpy()
    logger.info("REAL dataset assembled: %d aligned hourly rows.", len(df))
    return df.reset_index(drop=True)


def _assemble_energy_charts(cfg: DatasetConfig) -> pd.DataFrame:
    """Assemble the frame from Energy-Charts (REAL, keyless) + Open-Meteo weather."""
    from src.ingestion.energy_charts import fetch_power, fetch_price
    from src.models.synthetic_price import synthetic_gas_price  # gas: external feed placeholder

    weather = fetch_historical_weather(cfg.latitude, cfg.longitude, cfg.start_date, cfg.end_date)
    power = fetch_power(cfg.ec_country, cfg.start_date, cfg.end_date)
    price = fetch_price(cfg.ec_bzn, cfg.start_date, cfg.end_date)

    df = (
        weather.drop(columns=["available_at", "source"], errors="ignore")
        .merge(power, on="target_time", how="inner")
        .merge(price, on="target_time", how="inner")
        .dropna(subset=["generation_mw", "demand_mw", "price_eur_mwh"])
        .reset_index(drop=True)
    )
    # Gas is a separate commodity (TTF), not in the electricity feeds — placeholder.
    df["gas_price_eur_mwh"] = synthetic_gas_price(df).to_numpy()
    logger.info("REAL (Energy-Charts) dataset assembled: %d aligned hourly rows.", len(df))
    return df


def _assemble_synthetic(cfg: DatasetConfig) -> pd.DataFrame:
    """Assemble the frame from physics-flavoured synthetic stand-ins."""
    from src.models.synthetic import synthetic_generation_target
    from src.models.synthetic_price import synthetic_demand, synthetic_gas_price, synthetic_price

    weather = fetch_historical_weather(cfg.latitude, cfg.longitude, cfg.start_date, cfg.end_date)
    df = weather.drop(columns=["available_at", "source"], errors="ignore").copy()

    df["generation_mw"] = synthetic_generation_target(weather).to_numpy()
    df["demand_mw"] = synthetic_demand(weather).to_numpy()
    df["gas_price_eur_mwh"] = synthetic_gas_price(weather).to_numpy()
    df["price_eur_mwh"] = synthetic_price(
        pd.Series(df["demand_mw"].to_numpy()),
        pd.Series(df["generation_mw"].to_numpy()),
        pd.Series(df["gas_price_eur_mwh"].to_numpy()),
    ).to_numpy()
    logger.warning("SYNTHETIC dataset assembled (%d rows). Targets are stand-ins, NOT real.", len(df))
    return df.reset_index(drop=True)


def assemble_dataset(cfg: DatasetConfig | None = None) -> pd.DataFrame:
    """Return one aligned hourly frame, using the best available data source.

    Priority: ENTSO-E (if key) -> Energy-Charts (real, keyless) -> synthetic.
    Output columns (all paths): target_time, weather vars, generation_mw,
    demand_mw, gas_price_eur_mwh, price_eur_mwh.
    """
    cfg = cfg or DatasetConfig()

    if _force_synthetic():
        logger.warning("USE_SYNTHETIC=1: using SYNTHETIC fallback.")
        return _assemble_synthetic(cfg)

    if _has_entsoe_key():
        return _assemble_real(cfg)

    # No key -> try REAL keyless source (Energy-Charts); fall back to synthetic
    # only if it is unreachable (e.g. no internet).
    try:
        return _assemble_energy_charts(cfg)
    except Exception as exc:  # noqa: BLE001 — any failure -> safe synthetic fallback
        logger.warning("Energy-Charts unavailable (%s); falling back to SYNTHETIC.", exc)
        return _assemble_synthetic(cfg)
