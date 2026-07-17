"""One-off builder for the PVGIS solar-yield grid behind the solar calculator.

Why a grid, and why a SEPARATE job from precompute.py?
PVGIS returns climatology (long-run satellite radiation averages). The answer
for a given point does not change from day to day, so re-fetching it nightly
would be ~780 pointless requests against a public EU service. This job runs
once, commits its output, and is only re-run if we change the grid.

The browser cannot call PVGIS itself (no CORS header — verified), so the values
have to be baked out here.

Output: webapp/site/data/solar_grid.json — a regular lat/lon lattice the
frontend interpolates over, so a user can click ANY point in Europe and get a
yield estimate without a server.

Sea points come back as null (PVGIS has no radiation there); the frontend
falls back to the nearest land node.

The job is RESUMABLE: it reloads its own output and only fetches missing
points, so a rate-limit or a dropped connection costs minutes, not the run.

Run:  python webapp/build_solar_grid.py
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ingestion.energy_extras import fetch_pv_yield
from src.ingestion.exceptions import ApiTransientError

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
log = logging.getLogger("solar_grid")
log.setLevel(logging.INFO)

OUT = Path(__file__).parent / "site" / "data" / "solar_grid.json"

# Bounds match the map's framing (Europe), with a small margin so the corners
# of the viewport still interpolate cleanly.
LON0, LON1, DLON = -13.0, 47.0, 2.0
LAT0, LAT1, DLAT = 34.0, 71.5, 1.5

PAUSE_S = 0.4          # be a good citizen; PVGIS answers in ~1.3 s anyway
SAVE_EVERY = 25        # checkpoint so a crash never loses much


def _axis(a0: float, a1: float, step: float) -> list[float]:
    n = int(round((a1 - a0) / step)) + 1
    return [round(a0 + i * step, 4) for i in range(n)]


def main() -> None:
    lons, lats = _axis(LON0, LON1, DLON), _axis(LAT0, LAT1, DLAT)
    total = len(lons) * len(lats)

    if OUT.exists():
        grid = json.loads(OUT.read_text(encoding="utf-8"))
        if grid.get("lons") != lons or grid.get("lats") != lats:
            log.warning("Grid definition changed — starting over.")
            grid = None
    else:
        grid = None

    if grid is None:
        grid = {
            "source": "PVGIS 5.2 (EU JRC), PVGIS-SARAH2 satellite radiation",
            "note": "Climatology: crystalline Si, optimal angles, 14% system loss.",
            "unit": "kWh per kWp",
            "lons": lons, "lats": lats,
            # Row-major [lat][lon]; null = no PVGIS data (sea / out of coverage).
            "yearly": [[None] * len(lons) for _ in lats],
            "monthly": [[None] * len(lons) for _ in lats],
        }

    done = sum(1 for row in grid["yearly"] for v in row if v is not None)
    todo = [(i, j) for i, _ in enumerate(lats) for j, _ in enumerate(lons)
            if grid["yearly"][i][j] is None]
    log.info("Grid %dx%d = %d points | %d done, %d to try",
             len(lons), len(lats), total, done, len(todo))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    sea = fetched = 0

    for n, (i, j) in enumerate(todo, 1):
        lat, lon = lats[i], lons[j]
        try:
            r = fetch_pv_yield(lat, lon)
        except ApiTransientError as exc:
            log.warning("  transient at (%.1f, %.1f): %s — backing off 30 s", lat, lon, exc)
            time.sleep(30)
            continue
        except Exception as exc:  # noqa: BLE001
            log.warning("  skip (%.1f, %.1f): %s", lat, lon, str(exc)[:100])
            continue

        if r is None:
            # Sea. Record 0 so the resume logic does not retry it forever;
            # the frontend treats 0 as "no land here".
            grid["yearly"][i][j] = 0
            grid["monthly"][i][j] = 0
            sea += 1
        else:
            grid["yearly"][i][j] = r["yearly_kwh_per_kwp"]
            grid["monthly"][i][j] = r["monthly_kwh_per_kwp"]
            fetched += 1

        if n % SAVE_EVERY == 0:
            OUT.write_text(json.dumps(grid), encoding="utf-8")
            log.info("  %d/%d (%d land, %d sea)", n, len(todo), fetched, sea)
        time.sleep(PAUSE_S)

    OUT.write_text(json.dumps(grid), encoding="utf-8")
    land = sum(1 for row in grid["yearly"] for v in row if v)
    log.info("DONE: %d land points, %d sea/empty, %.0f KB",
             land, total - land, OUT.stat().st_size / 1024)


if __name__ == "__main__":
    main()
