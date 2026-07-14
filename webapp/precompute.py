"""Per-country forecast precompute job for the web dashboard.

For EVERY candidate European country it:
  1. fetches real generation/load/price (Energy-Charts, keyless) and a
     continuous weather window (Open-Meteo: last ~90 days + next 2 days),
  2. trains the engine per country: two 3A LightGBM models (wind, solar),
     the 3B conformalized quantile price model (CQR) and the calibrated
     negative-price classifier,
  3. produces a next-24h forecast (generation, price band, risk, signals),
  4. writes webapp/site/data/{cc}.json + a countries.json index.

Countries that fail (no data / API outage) are SKIPPED with a log line —
discovery is dynamic, so "all fetchable countries" is literally what ships.

Serving-time notes (honest simplifications, documented for interviews):
  - CQR (fixed band widening) is used for the future interval, not ACI —
    ACI needs realized-coverage feedback which does not exist for the future.
  - Demand forecast for the next 24h is a seasonal-naive (same hour 7 days
    ago); a real deployment would use the TSO day-ahead load forecast.
  - 3B train features use in-sample 3A predictions (cheap) instead of OOF;
    the OOF discipline matters for BACKTESTS, while here the future rows are
    genuinely unseen. Documented skew, acceptable for a dashboard.

Run:  python webapp/precompute.py            (all countries, ~15-25 min)
      python webapp/precompute.py de fr nl   (subset)
"""

from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.decision import calibrate_params_from_history, decide
from src.features import build_generation_feature_matrix
from src.features.price_features import build_price_feature_matrix
from src.ingestion import fetch_weather_window
from src.ingestion.energy_charts import fetch_power, fetch_price
from src.models import GenerationForecaster, NegativePriceClassifier
from src.models.conformal import ConformalizedQuantileForecaster

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("precompute")

SITE_DATA = Path(__file__).parent / "site" / "data"
HISTORY_DAYS = 88          # Open-Meteo past_days cap is 92
FORECAST_HOURS = 24
ASSET_FRACTION = 0.01      # our asset = ~1% of the national renewable fleet
PAUSE_BETWEEN_COUNTRIES_S = 3.0

# Fast training profile for the dashboard job (6 models x ~30 countries).
FAST_GEN_PARAMS = dict(
    n_estimators=300, learning_rate=0.06, num_leaves=31, subsample=0.8,
    colsample_bytree=0.8, min_child_samples=20, random_state=42, n_jobs=-1, verbose=-1,
)
FAST_Q_PARAMS = dict(
    n_estimators=250, learning_rate=0.06, num_leaves=31, subsample=0.8,
    colsample_bytree=0.8, min_child_samples=30, random_state=42, n_jobs=-1, verbose=-1,
)


@dataclass(frozen=True)
class Country:
    code: str        # Energy-Charts country code
    bzn: str         # bidding zone for /price
    name_en: str     # world-atlas feature name (map matching)
    name_tr: str     # display name
    lat: float
    lon: float


# All plausible candidates; the job keeps whichever actually return data.
COUNTRIES = [
    Country("de", "DE-LU", "Germany", "Almanya", 51.2, 10.4),
    Country("at", "AT", "Austria", "Avusturya", 47.6, 14.1),
    Country("be", "BE", "Belgium", "Belçika", 50.6, 4.7),
    Country("bg", "BG", "Bulgaria", "Bulgaristan", 42.7, 25.5),
    Country("ch", "CH", "Switzerland", "İsviçre", 46.8, 8.2),
    Country("cz", "CZ", "Czechia", "Çekya", 49.8, 15.5),
    Country("dk", "DK1", "Denmark", "Danimarka", 56.0, 10.0),
    Country("ee", "EE", "Estonia", "Estonya", 58.7, 25.5),
    Country("es", "ES", "Spain", "İspanya", 40.3, -3.7),
    Country("fi", "FI", "Finland", "Finlandiya", 62.9, 26.0),
    Country("fr", "FR", "France", "Fransa", 46.6, 2.5),
    Country("gr", "GR", "Greece", "Yunanistan", 39.1, 22.0),
    Country("hr", "HR", "Croatia", "Hırvatistan", 45.5, 16.0),
    Country("hu", "HU", "Hungary", "Macaristan", 47.2, 19.4),
    Country("it", "IT-North", "Italy", "İtalya", 45.5, 9.2),
    Country("lt", "LT", "Lithuania", "Litvanya", 55.2, 23.9),
    Country("lu", "DE-LU", "Luxembourg", "Lüksemburg", 49.8, 6.1),
    Country("lv", "LV", "Latvia", "Letonya", 56.9, 24.6),
    Country("nl", "NL", "Netherlands", "Hollanda", 52.2, 5.3),
    Country("no", "NO2", "Norway", "Norveç", 58.9, 7.0),
    Country("pl", "PL", "Poland", "Polonya", 52.1, 19.4),
    Country("pt", "PT", "Portugal", "Portekiz", 39.6, -8.0),
    Country("ro", "RO", "Romania", "Romanya", 45.9, 25.0),
    Country("rs", "RS", "Serbia", "Sırbistan", 44.2, 20.9),
    Country("se", "SE3", "Sweden", "İsveç", 59.3, 15.0),
    Country("si", "SI", "Slovenia", "Slovenya", 46.1, 14.8),
    Country("sk", "SK", "Slovakia", "Slovakya", 48.7, 19.5),
    Country("me", "ME", "Montenegro", "Karadağ", 42.7, 19.3),
    Country("mk", "MK", "North Macedonia", "K. Makedonya", 41.6, 21.7),
    Country("ba", "BA", "Bosnia and Herz.", "Bosna-Hersek", 44.2, 17.8),
    Country("md", "MD", "Moldova", "Moldova", 47.2, 28.5),
    Country("cy", "CY", "Cyprus", "Kıbrıs", 35.1, 33.2),
]

WEATHER_COLS = ["target_time", "temperature_2m", "wind_speed_100m",
                "wind_direction_100m", "shortwave_radiation", "cloud_cover"]


def _train_generation_model(weather_hist: pd.DataFrame, target: pd.Series,
                            target_name: str) -> GenerationForecaster | None:
    """Train one 3A model (wind OR solar) on the history window."""
    feat = build_generation_feature_matrix(weather_hist[WEATHER_COLS].copy())
    feat["generation_mw"] = target.to_numpy()
    feat = feat.dropna(subset=["generation_mw"]).reset_index(drop=True)
    if len(feat) < 500 or feat["generation_mw"].max() <= 0:
        log.info("  %s: not enough signal, skipping model", target_name)
        return None
    model = GenerationForecaster(FAST_GEN_PARAMS).fit(feat)
    return model


def _predict_generation(model: GenerationForecaster | None,
                        weather_all: pd.DataFrame, future_idx: pd.Index) -> np.ndarray:
    if model is None:
        return np.zeros(len(future_idx))
    feat = build_generation_feature_matrix(weather_all[WEATHER_COLS].copy())
    feat = feat.set_index("target_time")
    fut = feat.loc[future_idx].reset_index()
    return model.predict(fut)


def process_country(c: Country) -> dict | None:
    """Full engine pass for one country. Returns the JSON payload or None."""
    now = pd.Timestamp.now("UTC").floor("h")
    start = (now - pd.Timedelta(days=HISTORY_DAYS)).strftime("%Y-%m-%d")
    end = (now + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    # ---- 1) Real market data + one continuous weather window ----
    power = fetch_power(c.code, start, end)
    price = fetch_price(c.bzn, start, end)
    weather = fetch_weather_window(c.lat, c.lon, past_days=HISTORY_DAYS, forecast_days=2)
    weather = weather.drop(columns=["available_at", "source"], errors="ignore")

    hist = (
        weather.merge(power, on="target_time", how="inner")
        .merge(price, on="target_time", how="inner")
        .dropna(subset=["generation_mw", "demand_mw", "price_eur_mwh"])
        .sort_values("target_time").reset_index(drop=True)
    )
    if len(hist) < 800:
        log.warning("  %s: only %d aligned rows — skipping", c.code, len(hist))
        return None

    last_known = hist["target_time"].max()
    future_hours = pd.date_range(last_known + pd.Timedelta(hours=1),
                                 periods=FORECAST_HOURS, freq="1h", tz="UTC")
    weather_idx = weather.set_index("target_time")
    if not future_hours.isin(weather_idx.index).all():
        log.warning("  %s: weather forecast does not cover horizon — skipping", c.code)
        return None

    # ---- 2) 3A: wind + solar models -> next-24h generation ----
    wind_model = _train_generation_model(hist, hist["wind_mw"], "wind")
    solar_model = _train_generation_model(hist, hist["solar_mw"], "solar")
    wind_fc = _predict_generation(wind_model, weather, future_hours)
    solar_fc = _predict_generation(solar_model, weather, future_hours)
    gen_fc = wind_fc + solar_fc

    # ---- 3) 3B features: history for training, forecasts for the future ----
    # In-sample 3A predictions as train features (see module docstring).
    hist_wind_pred = _predict_generation(wind_model, weather, hist["target_time"])
    hist_solar_pred = _predict_generation(solar_model, weather, hist["target_time"])
    train_pf = pd.DataFrame({
        "target_time": hist["target_time"].to_numpy(),
        "demand_forecast_mw": hist["demand_mw"].to_numpy(),
        "predicted_generation_mw": hist_wind_pred + hist_solar_pred,
        "price_eur_mwh": hist["price_eur_mwh"].to_numpy(),
    })
    train_df = build_price_feature_matrix(train_pf).dropna().reset_index(drop=True)

    # Seasonal-naive demand for the future: same hour one week earlier.
    dem = hist.set_index("target_time")["demand_mw"]
    naive_src = future_hours - pd.Timedelta(days=7)
    demand_fc = dem.reindex(naive_src).to_numpy()
    if np.isnan(demand_fc).any():                      # fallback: last 24h profile
        prof = dem.groupby(dem.index.hour).mean()
        demand_fc = np.where(np.isnan(demand_fc),
                             prof.reindex(future_hours.hour).to_numpy(), demand_fc)

    future_pf = pd.DataFrame({
        "target_time": future_hours,
        "demand_forecast_mw": demand_fc,
        "predicted_generation_mw": gen_fc,
        "price_eur_mwh": np.nan,
    })
    # Ramps need context: build features on train-tail + future, then slice.
    tail = train_pf.iloc[-48:]
    combo = pd.concat([tail, future_pf], ignore_index=True)
    combo_feat = build_price_feature_matrix(combo)
    future_df = combo_feat.iloc[len(tail):].reset_index(drop=True)

    # ---- 4) Price quantiles (CQR) + negative-price risk ----
    cqr = ConformalizedQuantileForecaster(lo=0.1, hi=0.9, params=FAST_Q_PARAMS).fit(train_df)
    q = cqr.predict(future_df)
    clf = NegativePriceClassifier(params=dict(FAST_Q_PARAMS, class_weight="balanced")).fit(train_df)
    neg_hist_rate = float((train_df["price_eur_mwh"] < 0).mean())
    if neg_hist_rate > 0:
        risk = clf.predict_risk(future_df)
    else:
        risk = np.zeros(len(future_df))   # classifier is degenerate with 0 positives

    # ---- 5) Signals for a single ~1%-of-fleet asset ----
    mean_gen_asset = float(hist["generation_mw"].mean()) * ASSET_FRACTION
    params = calibrate_params_from_history(
        train_df["price_eur_mwh"].to_numpy(), mean_generation_mw=mean_gen_asset,
        subsidy_eur_mwh=0.0, recovery_quantile=0.80,
    )
    signals, charge = [], 0.0
    for i, ts in enumerate(future_hours):
        sig = decide(
            generation_mwh=gen_fc[i] * ASSET_FRACTION,
            p10=float(q["p10"].iloc[i]), p50=float(q["p50"].iloc[i]),
            p90=float(q["p90"].iloc[i]), neg_risk=float(risk[i]),
            hour=int(ts.hour), storage_charge_mwh=charge, params=params,
        )
        # decide() already caps store by headroom and discharge by charge.
        charge = max(0.0, min(params.storage_capacity_mwh,
                              charge + sig.store_mwh - sig.discharge_mwh))
        signals.append(sig.kind.value)

    # ---- 6) JSON payload ----
    r1 = lambda a: [round(float(v), 1) for v in a]
    recent = hist.iloc[-48:]
    payload = {
        "code": c.code, "bzn": c.bzn, "name_en": c.name_en, "name_tr": c.name_tr,
        "updated_utc": now.isoformat(),
        "kpis": {
            "last_price_eur_mwh": round(float(hist["price_eur_mwh"].iloc[-1]), 1),
            "gen_forecast_gw": round(float(np.mean(gen_fc)) / 1000.0, 2),
            "max_neg_risk_pct": round(float(np.max(risk)) * 100.0, 1),
            "next_signal": signals[0],
            "neg_hours_history_pct": round(neg_hist_rate * 100.0, 1),
        },
        "forecast": {
            "hours": [t.isoformat() for t in future_hours],
            "wind_mw": r1(wind_fc), "solar_mw": r1(solar_fc),
            "p10": r1(q["p10"]), "p50": r1(q["p50"]), "p90": r1(q["p90"]),
            "neg_risk_pct": [round(float(v) * 100.0, 1) for v in risk],
            "signal": signals,
        },
        "recent": {
            "hours": [t.isoformat() for t in recent["target_time"]],
            "price_eur_mwh": r1(recent["price_eur_mwh"]),
            "generation_mw": r1(recent["generation_mw"]),
        },
    }
    return payload


def main(only: list[str] | None = None) -> None:
    SITE_DATA.mkdir(parents=True, exist_ok=True)
    targets = [c for c in COUNTRIES if not only or c.code in only]
    index, failed = [], []

    for c in targets:
        t0 = time.time()
        log.info("=== %s (%s) ===", c.name_en, c.code)
        try:
            payload = process_country(c)
        except Exception as exc:  # noqa: BLE001 — one country must not kill the job
            log.warning("  %s FAILED: %s", c.code, str(exc)[:200])
            failed.append(c.code)
            time.sleep(PAUSE_BETWEEN_COUNTRIES_S)
            continue
        if payload is None:
            failed.append(c.code)
            time.sleep(PAUSE_BETWEEN_COUNTRIES_S)
            continue

        (SITE_DATA / f"{c.code}.json").write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        index.append({"code": c.code, "name_en": c.name_en, "name_tr": c.name_tr,
                      "kpis": payload["kpis"]})
        log.info("  OK in %.0fs — price %.1f, gen %.2f GW, risk %.0f%%, signal %s",
                 time.time() - t0, payload["kpis"]["last_price_eur_mwh"],
                 payload["kpis"]["gen_forecast_gw"], payload["kpis"]["max_neg_risk_pct"],
                 payload["kpis"]["next_signal"])
        time.sleep(PAUSE_BETWEEN_COUNTRIES_S)

    (SITE_DATA / "countries.json").write_text(
        json.dumps({"updated_utc": pd.Timestamp.now("UTC").isoformat(),
                    "countries": index}, ensure_ascii=False), encoding="utf-8")
    log.info("DONE: %d countries written, %d skipped (%s)",
             len(index), len(failed), ",".join(failed) or "-")


if __name__ == "__main__":
    main(sys.argv[1:] or None)
