"""Per-country forecast precompute job for the web dashboard.

For EVERY country Energy-Charts serves it:
  1. fetches real generation/load/price + a continuous weather window,
  2. trains the engine per country: wind + solar + DEMAND LightGBM models,
     the CQR conformalized quantile price model and the calibrated
     negative-price classifier,
  3. produces a next-24h forecast (generation, price band, risk, signals
     with a short plain-English rationale),
  4. runs a 14-day HOLDOUT evaluation to report real skill (generation MAE,
     price-band coverage, strategy-vs-naive backtest edge + Sharpe) and a
     48-hour forecast-vs-actual REPLAY,
  5. writes webapp/site/data/{cc}.json + a countries.json index.

Countries that fail (no data / API outage) are SKIPPED with a log line.

Honest simplifications (documented for interviews):
  - CQR (fixed band widening), not ACI, for the future interval — ACI needs
    realized-coverage feedback the future does not have.
  - 3B train features use in-sample 3A predictions (not OOF) — the OOF
    discipline matters for BACKTESTS; future/holdout rows are genuinely unseen.
  - Holdout price features use REALIZED demand (isolates gen+price skill);
    the live future uses the demand MODEL's forecast.

Run:  python webapp/precompute.py            (all countries)
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

from src.backtest import compute_risk_metrics, run_backtest, run_naive_baseline
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
HISTORY_DAYS = 88
FORECAST_HOURS = 24
HOLDOUT_HOURS = 14 * 24        # 14 days held out for the skill metrics
REPLAY_HOURS = 48             # last 48h of the holdout shown as forecast-vs-actual
ASSET_FRACTION = 0.01
PAUSE_BETWEEN_COUNTRIES_S = 3.0

FAST_GEN_PARAMS = dict(
    n_estimators=300, learning_rate=0.06, num_leaves=31, subsample=0.8,
    colsample_bytree=0.8, min_child_samples=20, random_state=42, n_jobs=-1, verbose=-1,
)
FAST_Q_PARAMS = dict(
    n_estimators=250, learning_rate=0.06, num_leaves=31, subsample=0.8,
    colsample_bytree=0.8, min_child_samples=30, random_state=42, n_jobs=-1, verbose=-1,
)
WEATHER_COLS = ["target_time", "temperature_2m", "wind_speed_100m",
                "wind_direction_100m", "shortwave_radiation", "cloud_cover"]


@dataclass(frozen=True)
class Country:
    code: str
    bzn: str
    name_en: str
    lat: float
    lon: float


COUNTRIES = [
    Country("de", "DE-LU", "Germany", 51.2, 10.4),
    Country("at", "AT", "Austria", 47.6, 14.1),
    Country("be", "BE", "Belgium", 50.6, 4.7),
    Country("bg", "BG", "Bulgaria", 42.7, 25.5),
    Country("ch", "CH", "Switzerland", 46.8, 8.2),
    Country("cz", "CZ", "Czechia", 49.8, 15.5),
    Country("dk", "DK1", "Denmark", 56.0, 10.0),
    Country("ee", "EE", "Estonia", 58.7, 25.5),
    Country("es", "ES", "Spain", 40.3, -3.7),
    Country("fi", "FI", "Finland", 62.9, 26.0),
    Country("fr", "FR", "France", 46.6, 2.5),
    Country("gr", "GR", "Greece", 39.1, 22.0),
    Country("hr", "HR", "Croatia", 45.5, 16.0),
    Country("hu", "HU", "Hungary", 47.2, 19.4),
    Country("it", "IT-North", "Italy", 45.5, 9.2),
    Country("lt", "LT", "Lithuania", 55.2, 23.9),
    Country("lu", "DE-LU", "Luxembourg", 49.8, 6.1),
    Country("lv", "LV", "Latvia", 56.9, 24.6),
    Country("nl", "NL", "Netherlands", 52.2, 5.3),
    Country("no", "NO2", "Norway", 58.9, 7.0),
    Country("pl", "PL", "Poland", 52.1, 19.4),
    Country("pt", "PT", "Portugal", 39.6, -8.0),
    Country("ro", "RO", "Romania", 45.9, 25.0),
    Country("rs", "RS", "Serbia", 44.2, 20.9),
    Country("se", "SE3", "Sweden", 59.3, 15.0),
    Country("si", "SI", "Slovenia", 46.1, 14.8),
    Country("sk", "SK", "Slovakia", 48.7, 19.5),
    Country("me", "ME", "Montenegro", 42.7, 19.3),
]


def _train_target_model(weather_hist: pd.DataFrame, target: pd.Series) -> GenerationForecaster | None:
    """Train one LightGBM model (wind / solar / demand) on the history window."""
    feat = build_generation_feature_matrix(weather_hist[WEATHER_COLS].copy())
    feat["generation_mw"] = target.to_numpy()          # reuse the target slot
    feat = feat.dropna(subset=["generation_mw"]).reset_index(drop=True)
    if len(feat) < 400 or float(np.nanmax(feat["generation_mw"])) <= 0:
        return None
    return GenerationForecaster(FAST_GEN_PARAMS).fit(feat)


def _predict(model: GenerationForecaster | None, weather_all: pd.DataFrame,
             idx: pd.Index) -> np.ndarray:
    if model is None:
        return np.zeros(len(idx))
    feat = build_generation_feature_matrix(weather_all[WEATHER_COLS].copy()).set_index("target_time")
    return model.predict(feat.loc[idx].reset_index())


def _run_pipeline(train_hist: pd.DataFrame, weather: pd.DataFrame,
                  eval_index: pd.DatetimeIndex, eval_demand: np.ndarray) -> dict:
    """Train wind/solar + price models on train_hist, forecast for eval_index.

    Shared by the live-future run (train=all, demand=forecast) and the holdout
    run (train=history minus 14d, demand=realized).
    """
    wm = _train_target_model(train_hist, train_hist["wind_mw"])
    sm = _train_target_model(train_hist, train_hist["solar_mw"])
    wind = _predict(wm, weather, eval_index)
    solar = _predict(sm, weather, eval_index)
    gen = wind + solar

    tr_gen = _predict(wm, weather, train_hist["target_time"]) + \
        _predict(sm, weather, train_hist["target_time"])
    train_pf = pd.DataFrame({
        "target_time": train_hist["target_time"].to_numpy(),
        "demand_forecast_mw": train_hist["demand_mw"].to_numpy(),
        "predicted_generation_mw": tr_gen,
        "price_eur_mwh": train_hist["price_eur_mwh"].to_numpy(),
    })
    train_df = build_price_feature_matrix(train_pf).dropna().reset_index(drop=True)

    future_pf = pd.DataFrame({
        "target_time": eval_index,
        "demand_forecast_mw": eval_demand,
        "predicted_generation_mw": gen,
        "price_eur_mwh": np.nan,
    })
    tail = train_pf.iloc[-48:]
    combo = build_price_feature_matrix(pd.concat([tail, future_pf], ignore_index=True))
    eval_df = combo.iloc[len(tail):].reset_index(drop=True)

    cqr = ConformalizedQuantileForecaster(lo=0.1, hi=0.9, params=FAST_Q_PARAMS).fit(train_df)
    q = cqr.predict(eval_df)
    neg_rate = float((train_df["price_eur_mwh"] < 0).mean())
    if neg_rate > 0:
        clf = NegativePriceClassifier(params=dict(FAST_Q_PARAMS, class_weight="balanced")).fit(train_df)
        risk = clf.predict_risk(eval_df)
    else:
        risk = np.zeros(len(eval_df))
    return {"wind": wind, "solar": solar, "gen": gen,
            "p10": q["p10"].to_numpy(), "p50": q["p50"].to_numpy(),
            "p90": q["p90"].to_numpy(), "risk": risk,
            "train_prices": train_df["price_eur_mwh"].to_numpy()}


def _signals_for(gen: np.ndarray, p10, p50, p90, risk, hours, params) -> tuple[list, list]:
    """Run the decision policy over a horizon; return (signals, rationale)."""
    sigs, reasons, charge = [], [], 0.0
    for i, ts in enumerate(hours):
        sig = decide(generation_mwh=gen[i] * ASSET_FRACTION,
                     p10=float(p10[i]), p50=float(p50[i]), p90=float(p90[i]),
                     neg_risk=float(risk[i]), hour=int(ts.hour),
                     storage_charge_mwh=charge, params=params)
        charge = max(0.0, min(params.storage_capacity_mwh,
                              charge + sig.store_mwh - sig.discharge_mwh))
        sigs.append(sig.kind.value)
        reasons.append(_reason(sig.kind.value, float(p50[i]), float(risk[i])))
    return sigs, reasons


def _reason(kind: str, p50: float, risk: float) -> str:
    e = round(p50)
    if kind == "SELL":
        return f"Expected price {e} EUR/MWh — selling is the most profitable action."
    if kind == "STORE":
        return f"Price is low ({e} EUR/MWh) — store now, sell later at a higher price."
    if kind == "DISCHARGE":
        return f"High price ({e} EUR/MWh) — time to sell from the battery."
    if kind == "CURTAIL":
        return f"Price in negative territory ({e} EUR/MWh, risk {round(risk*100)}%) — curtail production."
    return "No generation — no action."


def _holdout(hist: pd.DataFrame, weather: pd.DataFrame) -> dict | None:
    """14-day out-of-sample skill metrics + a 48h forecast-vs-actual replay."""
    if len(hist) < HOLDOUT_HOURS + 600:
        return None
    htrain = hist.iloc[:-HOLDOUT_HOURS]
    htest = hist.iloc[-HOLDOUT_HOURS:].reset_index(drop=True)
    idx = pd.DatetimeIndex(htest["target_time"])

    r = _run_pipeline(htrain, weather, idx, htest["demand_mw"].to_numpy())
    gen_act = htest["generation_mw"].to_numpy()
    price_act = htest["price_eur_mwh"].to_numpy()

    denom = max(1.0, float(np.mean(gen_act)))
    gen_mae_pct = float(np.mean(np.abs(r["gen"] - gen_act)) / denom * 100.0)
    coverage = float(np.mean((price_act >= r["p10"]) & (price_act <= r["p90"])) * 100.0)

    # Mini backtest on the holdout window (strategy vs naive, identical costs).
    params = calibrate_params_from_history(
        r["train_prices"], mean_generation_mw=float(np.mean(gen_act)) * ASSET_FRACTION,
        subsidy_eur_mwh=0.0, recovery_quantile=0.80)
    dec = pd.DataFrame({
        "target_time": htest["target_time"].to_numpy(),
        "p10": r["p10"], "p50": r["p50"], "p90": r["p90"], "neg_risk": r["risk"],
        "forecast_generation_mwh": r["gen"] * ASSET_FRACTION,
        "actual_generation_mwh": gen_act * ASSET_FRACTION,
        "realized_price": price_act,
    })
    strat = run_backtest(dec, params)
    naive = run_naive_baseline(dec, params)
    edge = strat.total_pnl - naive.total_pnl
    edge_pct = edge / abs(naive.total_pnl) * 100.0 if naive.total_pnl else 0.0
    sharpe = compute_risk_metrics(strat.ledger["pnl"]).sharpe

    rp = htest.iloc[-REPLAY_HOURS:]
    s = slice(len(htest) - REPLAY_HOURS, len(htest))
    r1 = lambda a: [round(float(v), 1) for v in a]
    replay = {
        "hours": [t.isoformat() for t in rp["target_time"]],
        "gen_actual_mw": r1(rp["generation_mw"]),
        "gen_forecast_mw": r1(r["gen"][s]),
        "price_actual": r1(rp["price_eur_mwh"]),
        "price_p50": r1(r["p50"][s]),
        "price_p10": r1(r["p10"][s]),
        "price_p90": r1(r["p90"][s]),
    }
    return {
        "metrics": {
            "gen_mae_pct": round(gen_mae_pct, 1),
            "price_coverage_pct": round(coverage, 1),
            "backtest_edge_pct": round(edge_pct, 1),
            "backtest_sharpe": round(float(sharpe), 1),
            "holdout_days": HOLDOUT_HOURS // 24,
        },
        "replay": replay,
    }


def process_country(c: Country) -> dict | None:
    now = pd.Timestamp.now("UTC").floor("h")
    start = (now - pd.Timedelta(days=HISTORY_DAYS)).strftime("%Y-%m-%d")
    end = (now + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

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
    future = pd.date_range(last_known + pd.Timedelta(hours=1),
                           periods=FORECAST_HOURS, freq="1h", tz="UTC")
    if not future.isin(pd.DatetimeIndex(weather["target_time"])).all():
        log.warning("  %s: weather horizon incomplete — skipping", c.code)
        return None

    # ---- Demand forecast (#6): a real model, not seasonal-naive ----
    demand_model = _train_target_model(hist, hist["demand_mw"])
    demand_fc = _predict(demand_model, weather, future)
    if demand_model is None or np.all(demand_fc == 0):
        dem = hist.set_index("target_time")["demand_mw"]
        prof = dem.groupby(dem.index.hour).mean()
        demand_fc = prof.reindex(future.hour).to_numpy()

    # ---- Live 24h forecast ----
    fc = _run_pipeline(hist, weather, future, demand_fc)
    params = calibrate_params_from_history(
        fc["train_prices"], mean_generation_mw=float(hist["generation_mw"].mean()) * ASSET_FRACTION,
        subsidy_eur_mwh=0.0, recovery_quantile=0.80)
    signals, reasons = _signals_for(fc["gen"], fc["p10"], fc["p50"], fc["p90"],
                                    fc["risk"], future, params)

    # ---- Holdout skill metrics + replay (#1, #3) ----
    hold = _holdout(hist, weather)

    r1 = lambda a: [round(float(v), 1) for v in a]
    recent = hist.iloc[-48:]
    renew_share = float((hist["generation_mw"] / hist["demand_mw"]).clip(0, 3).mean()) * 100.0
    payload = {
        "code": c.code, "bzn": c.bzn, "name_en": c.name_en,
        "updated_utc": now.isoformat(),
        "kpis": {
            "last_price_eur_mwh": round(float(hist["price_eur_mwh"].iloc[-1]), 1),
            "gen_forecast_gw": round(float(np.mean(fc["gen"])) / 1000.0, 2),
            "max_neg_risk_pct": round(float(np.max(fc["risk"])) * 100.0, 1),
            "next_signal": signals[0],
            "neg_hours_history_pct": round(float((hist["price_eur_mwh"] < 0).mean()) * 100.0, 1),
            "renewable_share_pct": round(renew_share, 1),
        },
        "forecast": {
            "hours": [t.isoformat() for t in future],
            "wind_mw": r1(fc["wind"]), "solar_mw": r1(fc["solar"]),
            "p10": r1(fc["p10"]), "p50": r1(fc["p50"]), "p90": r1(fc["p90"]),
            "neg_risk_pct": [round(float(v) * 100.0, 1) for v in fc["risk"]],
            "signal": signals, "signal_reason": reasons,
        },
        "recent": {
            "hours": [t.isoformat() for t in recent["target_time"]],
            "price_eur_mwh": r1(recent["price_eur_mwh"]),
            "generation_mw": r1(recent["generation_mw"]),
        },
    }
    if hold:
        payload["metrics"] = hold["metrics"]
        payload["replay"] = hold["replay"]
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
        except Exception as exc:  # noqa: BLE001
            log.warning("  %s FAILED: %s", c.code, str(exc)[:200])
            failed.append(c.code); time.sleep(PAUSE_BETWEEN_COUNTRIES_S); continue
        if payload is None:
            failed.append(c.code); time.sleep(PAUSE_BETWEEN_COUNTRIES_S); continue

        (SITE_DATA / f"{c.code}.json").write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        idx = {"code": c.code, "name_en": c.name_en, "kpis": payload["kpis"]}
        if "metrics" in payload:
            idx["metrics"] = payload["metrics"]
        index.append(idx)
        m = payload.get("metrics", {})
        log.info("  OK in %.0fs — price %.0f, gen %.2f GW, risk %.0f%%, sig %s | "
                 "MAE %.0f%%, cov %.0f%%, edge %+.1f%%",
                 time.time() - t0, payload["kpis"]["last_price_eur_mwh"],
                 payload["kpis"]["gen_forecast_gw"], payload["kpis"]["max_neg_risk_pct"],
                 payload["kpis"]["next_signal"], m.get("gen_mae_pct", -1),
                 m.get("price_coverage_pct", -1), m.get("backtest_edge_pct", 0))
        time.sleep(PAUSE_BETWEEN_COUNTRIES_S)

    (SITE_DATA / "countries.json").write_text(
        json.dumps({"updated_utc": pd.Timestamp.now("UTC").isoformat(),
                    "countries": index}, ensure_ascii=False), encoding="utf-8")
    log.info("DONE: %d written, %d skipped (%s)", len(index), len(failed), ",".join(failed) or "-")


if __name__ == "__main__":
    main(sys.argv[1:] or None)
