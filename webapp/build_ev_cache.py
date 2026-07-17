"""Builder for the EV-charging layer: station counts per country, from OSM.

Why this is a separate, cached job rather than part of the nightly precompute:
Overpass takes ~35 s per country and rate-limits hard, so a 28-country sweep is
~20 minutes of hammering a donated public service. Charging infrastructure also
moves on a scale of weeks, not hours. So we refresh at most weekly and keep the
previous answer whenever a country fails — a stale count is far better than a
blank panel.

Counts only, deliberately: Germany alone has ~43k charging nodes, which is both
too slow to fetch and too heavy to ship to a browser. The map needs the number,
not the pins. The genuinely useful part of this feature — WHEN to charge — comes
from our own price and carbon forecasts, not from OSM.

Output: webapp/site/data/ev_stations.json

Run:  python webapp/build_ev_cache.py            (only countries older than a week)
      python webapp/build_ev_cache.py --force    (refetch everything)
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ingestion.energy_extras import fetch_ev_charger_count

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("ev_cache")

OUT = Path(__file__).parent / "site" / "data" / "ev_stations.json"
MAX_AGE_DAYS = 7
PAUSE_S = 5.0

# ISO-3166 alpha-2 per dashboard country code. Luxembourg shares Germany's
# bidding zone but is its own country in OSM, so it gets its own query.
ISO2 = {
    "de": "DE", "at": "AT", "be": "BE", "bg": "BG", "ch": "CH", "cz": "CZ",
    "dk": "DK", "ee": "EE", "es": "ES", "fi": "FI", "fr": "FR", "gr": "GR",
    "hr": "HR", "hu": "HU", "it": "IT", "lt": "LT", "lu": "LU", "lv": "LV",
    "nl": "NL", "no": "NO", "pl": "PL", "pt": "PT", "ro": "RO", "rs": "RS",
    "se": "SE", "si": "SI", "sk": "SK", "me": "ME",
}


def main(force: bool = False) -> None:
    cache = {}
    if OUT.exists():
        cache = json.loads(OUT.read_text(encoding="utf-8")).get("countries", {})

    now = pd.Timestamp.now("UTC")
    fresh = stale = failed = 0

    for cc, iso in ISO2.items():
        prev = cache.get(cc)
        if prev and not force:
            age = (now - pd.Timestamp(prev["fetched_utc"])) / pd.Timedelta(days=1)
            if age < MAX_AGE_DAYS:
                fresh += 1
                continue

        try:
            r = fetch_ev_charger_count(iso)
            cache[cc] = {
                "total": r["total"],
                "fast_dc": r["fast_dc"],
                "fetched_utc": now.isoformat(),
            }
            log.info("  %s: %d chargers (%d fast DC)", cc, r["total"], r["fast_dc"])
        except Exception as exc:  # noqa: BLE001
            # Keep whatever we had. A week-old count is still a true count.
            if prev:
                log.warning("  %s failed (%s) — keeping cached value from %s",
                            cc, str(exc)[:60], prev["fetched_utc"][:10])
                stale += 1
            else:
                log.warning("  %s failed with no cached fallback: %s", cc, str(exc)[:80])
                failed += 1
        time.sleep(PAUSE_S)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "source": "OpenStreetMap via Overpass API (ODbL)",
        "note": "Counts of amenity=charging_station nodes; fast_dc = has a CCS socket.",
        "updated_utc": now.isoformat(),
        "countries": cache,
    }, ensure_ascii=False), encoding="utf-8")
    log.info("DONE: %d cached, %d still fresh, %d kept stale, %d unavailable",
             len(cache), fresh, stale, failed)


if __name__ == "__main__":
    main(force="--force" in sys.argv)
