# European Electricity Market — Intraday Price & Risk Forecasting Bot

An AI system that forecasts renewable generation and intraday electricity
prices, scores negative-price risk, and turns forecasts into STORE / SELL /
CURTAIL / ARBITRAGE signals — validated in a cost-aware backtest.

## Architecture
[Data Ingestion (APIs)] -> [Time-Series DB] -> [AI/Forecast Engine] -> [Decision/Signal Bot]

## Status — all four steps complete
- [x] **Step 1 — Data Ingestion**: Open-Meteo (historical + forecast), ENTSO-E
      (day-ahead price, actual generation, load), a data-quality layer, and a
      leakage-safe join (`merge_asof`).
- [x] **Step 2 — Time-series database (TimescaleDB)**: bitemporal long-format
      hypertable, compression + continuous aggregate, idempotent upsert
      (revision + idempotence), leakage-safe `read_as_of` (DISTINCT ON).
      `docker compose up -d` -> `python run_storage_demo.py`
- [x] **Step 3A — Generation forecast (LightGBM)**: leakage-safe feature
      engineering (power curve, cyclical calendar, past-looking lag/rolling),
      `TimeSeriesSplit` walk-forward CV, LightGBM + early stopping.
      `python run_generation_demo.py`
- [x] **Step 3B — Price + negative-price risk**: model cascade (3A->3B,
      leakage-safe via out-of-fold), residual-load feature, quantile
      regression (P10/P50/P90) + calibrated negative-price probability
      (isotonic). `python run_price_demo.py`
- [x] **Step 4 — Decision/signal engine + backtest**: expected-value argmax
      policy (SELL/STORE/CURTAIL/DISCHARGE), thresholds derived from economics
      (marginal cost, -subsidy floor, storage opportunity cost), walk-forward
      backtest with realistic costs (spread + imbalance penalty + round-trip
      efficiency). Parameters are calibrated from TRAIN history
      (`calibrate_params_from_history`) and the asset is scaled to a single
      producer, so the strategy is meaningful on real data (+~13% vs naive on
      real German 2024, driven by battery arbitrage + selective curtailment).
      `python run_decision_demo.py`

### Final additions
- [x] **Real-data integration**: `src/pipeline/data_assembly.py` assembles one
      aligned hourly frame using the best available source, in priority order:
      **ENTSO-E** (if `ENTSOE_API_KEY` is set) → **Energy-Charts** (Fraunhofer
      ISE, REAL data with NO key) → **synthetic** (offline last resort). So the
      project runs on REAL German electricity data by default, no key needed.
      Verified end-to-end on real 2024 data (8783 hourly rows).
- [x] **Risk reporting**: Sharpe ratio, max drawdown, and CVaR (`src/backtest/
      risk.py`), plus a P&L equity-curve + drawdown plot saved to
      `reports/equity_curve.png`.
- [x] **Conformal calibration**: split CQR and Adaptive Conformal Inference
      (ACI) close the quantile coverage gap (~69% -> ~80%). ACI is used because
      seasonal distribution shift breaks the exchangeability that plain CQR
      assumes.

## Setup
```bash
pip install -r requirements.txt
cp .env.example .env      # fill in ENTSOE_API_KEY (optional; synthetic fallback otherwise)
python run_decision_demo.py
```
Without an ENTSO-E key everything still runs on the synthetic fallback. With a
key, the same pipeline runs on real data. Start TimescaleDB (`docker compose up
-d`) only for the Step 2 storage demo.

## Core design rules (interview summary)
1. **Bitemporal schema**: every row carries `target_time` (the instant the data
   belongs to) and `available_at` (the instant it was learned). For decision
   time T, features join with `available_at <= T` only, via
   `pd.merge_asof(direction="backward")`.
2. **No silent filling**: bfill is forbidden in production (look-ahead), ffill
   is forbidden on price series. Short gaps are interpolated only on physical
   signals with an `is_imputed` flag; long gaps stay NaN -> the decision engine
   emits NO_SIGNAL.
3. **Typed errors**: transient (retry) / permanent (alert) / data-quality
   (stop the pipeline) are distinguished.
4. **Leakage defence is one thread through every layer**: bitemporal
   `available_at`, `shift`/`rolling` direction, out-of-fold cascade,
   `TimeSeriesSplit`, and backtest settlement all repeat the same principle.

## Web dashboard (ENTSO-E-style map)

`webapp/` turns the engine into a public dashboard: a fullscreen Europe map with
flags; click a country to zoom into it and open its forecasts, carbon, flows and
asset calculators alongside.

- `webapp/precompute.py` — runs the engine for EVERY country Energy-Charts
  serves (failures skipped) and writes static JSON to `webapp/site/data/`.
  Per country: real gen/load/price + weather window, three LightGBM models
  (wind / solar / DEMAND), CQR price quantiles, calibrated negative-price risk,
  hourly signals with a plain-English rationale, a 14-day HOLDOUT evaluation
  (generation MAE%, price-band coverage%, strategy-vs-naive edge% and Sharpe)
  and a 48h forecast-vs-actual replay. It also emits `flows.json`, one
  reconciled Europe-wide interconnector network.
- `webapp/build_solar_grid.py` — one-off. PVGIS yield lattice over Europe.
- `webapp/build_ev_cache.py` — weekly. EV charger counts from OpenStreetMap.
- `webapp/build_news_cache.py` — every 6 h. Headlines from Google News RSS.
- `webapp/site/` — fully static frontend, no build step.

### Explainable predictions
Every forecast can be opened up. `src/models/explain.py` runs **exact TreeSHAP**
via LightGBM's `pred_contrib=True` — real Shapley values, not a feature-importance
proxy, and not a sampled approximation. `tests/test_explain.py` asserts the
property that makes them meaningful: `base + Σcontributions == the model's own
output`. If that additivity ever broke, the bars would still render while
telling the user something untrue, so it is pinned in CI.

The "Explain prediction" panel shows the top 5 drivers with signed EUR/MWh
effects and shares, the conformal confidence interval with its measured
coverage, the most similar historical hours (nearest neighbours in the model's
own feature space, with what price actually did then), and what would
invalidate the forecast.

Local run:
```bash
python webapp/build_solar_grid.py    # once, ~20 min (climatology never changes)
python webapp/build_ev_cache.py      # weekly, ~25 min (Overpass is slow)
python webapp/precompute.py          # daily, ~10-20 min (rate-limited API)
python -m http.server 8942 --directory webapp/site
# open http://localhost:8942
```

### Data sources — all keyless, all verified live
| Source | Gives us | Notes |
|---|---|---|
| Energy-Charts (Fraunhofer ISE) | generation mix, day-ahead price, **carbon intensity**, **cross-border flows**, **installed capacity 2002–2030** | no key; CORS restricted, so server-side only |
| Open-Meteo | wind @100m, irradiance, temperature — forecast + archive | no key; **sends CORS**, so the wind calculator calls it live from the browser |
| PVGIS 5.2 (EU JRC) | solar yield climatology (SARAH2 satellite) | no key, no CORS → baked into a lattice once; it never changes |
| OpenStreetMap / Overpass | public EV charging points | ODbL; ~35 s/country and rate-limited → cached weekly, stale values kept on failure |
| Google News RSS | market headlines | keyless, no CORS; throttles after ~9 country queries → cached, paced, stale kept |

Rejected during the survey (they now gate access behind a key): OpenChargeMap
(403), Ember (403), Electricity Maps (401, and its free tier covers one zone).

### Three details the dashboard gets right
- **Capture rate.** A solar farm does not earn the average price — it generates
  when every other panel does, and that glut craters the midday price. We
  measure `mean(price × generation) / (mean(price) × mean(generation))` per
  country. German solar captured **~46%** over the last 88 days, so a revenue
  estimate built on the average price would be roughly double the truth.
- **Carbon forecast.** The feed publishes carbon history but its forecast field
  comes back empty, so we forecast it with the same weather-driven model class
  used for wind and solar. Carbon intensity is mostly a function of how much
  wind and sun displace fossil plants — the model independently learns that the
  cleanest hour is midday.
- **No count choropleths.** Every metric the map can shade by is intensive — a
  rate or a density. Shading by an absolute total would just rank countries by
  size, so charger counts are divided by the country's own area, measured off
  the drawn geometry. Only the polygons inside the map frame count: world-atlas
  ships whole sovereign states, and charging France for French Guiana inflates
  its area 17% (measured) and understates its density by the same margin.

### Known limits
- The upstream feed publishes a few hours behind real time, so the "Now" panel
  stamps each value with the hour it belongs to, and the EV tool prices against
  the next hour that has not started yet rather than the first forecast row.
- Solar yield is interpolated from a 2° lattice: ~10% error in mountains
  (Munich reads 1000 vs 1113 kWh/kWp direct from PVGIS) and on narrow coasts.
- Clock times render in the viewer's timezone, not the country's.
- Overpass 504s on a few countries per sweep; those keep their previous count
  and are retried next run.

### Deploy (Vercel)
1. Push this repo to GitHub.
2. In Vercel: New Project → import the repo → set **Root Directory** to
   `webapp/site` → Framework preset: **Other** (static). Deploy.
3. Optional freshness: `.github/workflows/refresh-data.yml` re-runs the
   precompute daily and commits the JSON; Vercel auto-redeploys on push.
   `.github/workflows/ci.yml` runs the full test suite on every push/PR.

## Tests
```bash
set PYTHONPATH=.        # Windows: $env:PYTHONPATH="."
python tests/test_feature_leakage.py
python tests/test_price_models.py
python tests/test_decision.py
python tests/test_conformal.py
python tests/test_storage_transform.py
python tests/test_extras.py       # flow reconciliation, capture rates
python tests/test_explain.py      # SHAP additivity, similar days, invalidators
```
