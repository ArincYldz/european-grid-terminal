"""Builder for the market-news card: headlines per country, cached to disk.

Same reasoning as the EV cache. Google News throttles a burst of country
queries hard — measured: the first ~9 succeed, then every request is refused
for a while — and headlines do not change minute to minute. So this runs on its
own slow cadence with generous pauses, keeps the previous headlines whenever a
country fails, and precompute simply reads whatever is on disk.

Output: webapp/site/data/news.json

Run:  python webapp/build_news_cache.py            (only countries older than 6 h)
      python webapp/build_news_cache.py --force    (refetch everything)
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ingestion.energy_extras import fetch_energy_news

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("news_cache")

OUT = Path(__file__).parent / "site" / "data" / "news.json"
MAX_AGE_H = 6
PAUSE_S = 12.0     # deliberately slow; the throttle is the binding constraint


def main(force: bool = False) -> None:
    from webapp.precompute import COUNTRIES

    cache = {}
    if OUT.exists():
        try:
            cache = json.loads(OUT.read_text(encoding="utf-8")).get("countries", {})
        except (ValueError, OSError):
            cache = {}

    now = pd.Timestamp.now("UTC")
    fresh = stale = ok = 0

    for c in COUNTRIES:
        prev = cache.get(c.code)
        if prev and not force:
            age_h = (now - pd.Timestamp(prev["fetched_utc"])) / pd.Timedelta(hours=1)
            if age_h < MAX_AGE_H:
                fresh += 1
                continue
        try:
            items = fetch_energy_news(c.name_en, limit=6)
            if items:
                cache[c.code] = {"items": items, "fetched_utc": now.isoformat()}
                ok += 1
                log.info("  %s: %d headlines", c.code, len(items))
            elif prev:
                stale += 1
        except Exception as exc:  # noqa: BLE001
            if prev:
                log.warning("  %s failed (%s) — keeping %s", c.code, str(exc)[:50],
                            prev["fetched_utc"][:16])
                stale += 1
            else:
                log.warning("  %s failed, nothing cached: %s", c.code, str(exc)[:70])
        time.sleep(PAUSE_S)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "source": "Google News RSS",
        "note": "Headlines link to their publishers. The impact tag is a keyword match.",
        "updated_utc": now.isoformat(),
        "countries": cache,
    }, ensure_ascii=False), encoding="utf-8")
    log.info("DONE: %d cached total, %d refreshed, %d still fresh, %d kept stale",
             len(cache), ok, fresh, stale)


if __name__ == "__main__":
    main(force="--force" in sys.argv)
