"""One-off builder for the map's marker layers: landmarks and power plants.

Deliberately NOT on a schedule. Castles and power stations do not move, so this
runs once, its output is committed, and the nightly refresh never touches it.
Re-run it by hand only if you want to widen the tag filters or add a country.

Two things this job has to get right, both learned by measuring first:

1. **Micro-hydro would swamp the map.** Austria alone returns 400 `power=plant`
   objects and 369 are hydro, many of them 9 kW mill races. Plotting all of
   them says nothing about the grid. So capacities are parsed to MW and only
   the largest plants per source survive.

2. **Most `historic=memorial` objects are local plaques**, not landmarks — 156
   of Austria's 300 hits. They are dropped. What is left (castles, monuments,
   cathedrals, attractions) is additionally required to carry a `wikidata` tag,
   which is a decent proxy for "someone considered this notable".

Output: webapp/site/data/landmarks.json

Run:  python webapp/build_landmarks.py            (skips countries already done)
      python webapp/build_landmarks.py --force    (rebuild everything)
      python webapp/build_landmarks.py at de      (specific countries)
"""

from __future__ import annotations

import json
import logging
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ingestion.energy_extras import _overpass

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("landmarks")

OUT = Path(__file__).parent / "site" / "data" / "landmarks.json"
PAUSE_S = 8.0
MAX_LANDMARKS = 40          # per country — enough to feel populated, not a blanket
MAX_PLANTS_PER_SOURCE = 25  # per country per source, largest first

ISO2 = {
    "de": "DE", "at": "AT", "be": "BE", "bg": "BG", "ch": "CH", "cz": "CZ",
    "dk": "DK", "ee": "EE", "es": "ES", "fi": "FI", "fr": "FR", "gr": "GR",
    "hr": "HR", "hu": "HU", "it": "IT", "lt": "LT", "lu": "LU", "lv": "LV",
    "nl": "NL", "no": "NO", "pl": "PL", "pt": "PT", "ro": "RO", "rs": "RS",
    "se": "SE", "si": "SI", "sk": "SK", "me": "ME",
}

_UNIT_TO_MW = {"w": 1e-6, "kw": 1e-3, "mw": 1.0, "gw": 1e3}


def parse_mw(raw: str | None) -> float | None:
    """`plant:output:electricity` is free text: '6 MW', '142 kW', '11200 kW'.

    Returns megawatts, or None when the value is unparseable (`yes`, ranges,
    blanks). Unparseable is common enough that callers must handle it rather
    than assume zero — a plant with no capacity tag is not a 0 MW plant.
    """
    if not raw:
        return None
    m = re.match(r"\s*([\d.,]+)\s*([kMGw]*[Ww]?)", str(raw).strip())
    if not m:
        return None
    try:
        val = float(m.group(1).replace(",", "."))
    except ValueError:
        return None
    unit = (m.group(2) or "MW").lower()
    if not unit.endswith("w"):
        unit += "w"
    return val * _UNIT_TO_MW.get(unit, 1.0)


def _centre(elem: dict) -> tuple[float, float] | None:
    c = elem.get("center") or elem
    lat, lon = c.get("lat"), c.get("lon")
    return (round(float(lat), 4), round(float(lon), 4)) if lat and lon else None


def fetch_plants(iso: str) -> list[dict]:
    """Major solar / wind / hydro / nuclear stations, largest first."""
    area = f'area["ISO3166-1"="{iso}"][admin_level=2]->.a;'
    src = '["plant:source"~"^(solar|wind|hydro|nuclear)$"]'
    q = (f'[out:json][timeout:170];{area}('
         f'node(area.a)["power"="plant"]{src}["name"];'
         f'way(area.a)["power"="plant"]{src}["name"];'
         f'relation(area.a)["power"="plant"]{src}["name"];);out center 600;')
    els = _overpass(q).get("elements", [])

    buckets: dict[str, list[dict]] = {}
    for e in els:
        t = e.get("tags", {})
        pos = _centre(e)
        if not pos:
            continue
        source = t.get("plant:source")
        mw = parse_mw(t.get("plant:output:electricity"))
        buckets.setdefault(source, []).append({
            "name": t.get("name"), "lat": pos[0], "lon": pos[1],
            "mw": round(mw, 1) if mw is not None else None,
            "source": source,
        })

    out = []
    for source, rows in buckets.items():
        # Unknown capacity sorts last: we cannot claim it is big, but a named
        # nuclear station with no tag is still worth keeping over nothing.
        rows.sort(key=lambda r: (r["mw"] is None, -(r["mw"] or 0)))
        out.extend(rows[:MAX_PLANTS_PER_SOURCE])
    return out


def fetch_landmarks(iso: str) -> list[dict]:
    """Castles, monuments, cathedrals and attractions that carry a wikidata id."""
    area = f'area["ISO3166-1"="{iso}"][admin_level=2]->.a;'
    hist = '["historic"~"^(castle|monument)$"]["wikidata"]["name"]'
    attr = '["tourism"="attraction"]["wikidata"]["name"]'
    cath = '["building"~"^(cathedral|basilica)$"]["wikidata"]["name"]'
    q = (f'[out:json][timeout:170];{area}('
         f'node(area.a){hist};way(area.a){hist};'
         f'node(area.a){attr};way(area.a){attr};'
         f'way(area.a){cath};relation(area.a){cath};);out center 400;')
    els = _overpass(q).get("elements", [])

    rows = []
    for e in els:
        t = e.get("tags", {})
        pos = _centre(e)
        if not pos:
            continue
        if t.get("historic") in ("castle", "monument"):
            kind = t["historic"]
        elif t.get("building") in ("cathedral", "basilica"):
            kind = "cathedral"
        else:
            kind = "attraction"
        rows.append({
            "name": t.get("name"), "lat": pos[0], "lon": pos[1], "kind": kind,
            # A wikipedia article is a stronger notability signal than a bare
            # wikidata id, and a mapped outline means someone surveyed it.
            "_rank": (1 if t.get("wikipedia") else 0) + (1 if e.get("type") != "node" else 0),
        })
    rows.sort(key=lambda r: -r["_rank"])
    for r in rows:
        r.pop("_rank", None)
    return rows[:MAX_LANDMARKS]


def main(only: list[str] | None = None, force: bool = False) -> None:
    data = {}
    if OUT.exists() and not force:
        try:
            data = json.loads(OUT.read_text(encoding="utf-8")).get("countries", {})
        except (ValueError, OSError):
            data = {}

    targets = [c for c in ISO2 if not only or c in only]
    done = failed = 0

    for cc in targets:
        if cc in data and not force and not only:
            continue
        iso = ISO2[cc]
        entry = data.get(cc, {})
        try:
            entry["plants"] = fetch_plants(iso)
            log.info("  %s: %d plants", cc, len(entry["plants"]))
        except Exception as exc:  # noqa: BLE001
            log.warning("  %s plants failed: %s", cc, str(exc)[:70])
            entry.setdefault("plants", [])
        time.sleep(PAUSE_S)

        try:
            entry["landmarks"] = fetch_landmarks(iso)
            log.info("  %s: %d landmarks", cc, len(entry["landmarks"]))
        except Exception as exc:  # noqa: BLE001
            log.warning("  %s landmarks failed: %s", cc, str(exc)[:70])
            entry.setdefault("landmarks", [])
        time.sleep(PAUSE_S)

        if entry.get("plants") or entry.get("landmarks"):
            data[cc] = entry
            done += 1
        else:
            failed += 1

        # Checkpoint after every country: Overpass drops connections often
        # enough that losing a 40-minute sweep to one timeout is not acceptable.
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps({
            "source": "OpenStreetMap via Overpass API (ODbL)",
            "note": ("Static: plant and landmark positions do not change, so this "
                     "file is built once and committed, never refreshed on a cron."),
            "countries": data,
        }, ensure_ascii=False), encoding="utf-8")

    tot_p = sum(len(v.get("plants", [])) for v in data.values())
    tot_l = sum(len(v.get("landmarks", [])) for v in data.values())
    log.info("DONE: %d countries (%d new, %d empty) — %d plants, %d landmarks, %.0f KB",
             len(data), done, failed, tot_p, tot_l, OUT.stat().st_size / 1024)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    main(only=args or None, force="--force" in sys.argv)
