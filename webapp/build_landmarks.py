"""One-off builder for the map's decorative layer: two landmarks per country,
plus the energy sources that country actually runs.

Deliberately NOT on a schedule, and deliberately not scraped.

Why curated rather than queried: an earlier version pulled landmarks from
OpenStreetMap by tag. It returned 300 hits for Austria of which 156 were
`historic=memorial` (local plaques), ranked "Steinpyramide" above anything a
visitor would recognise, and timed out entirely on Germany. Ranking by OSM tags
does not mean fame. The map wants the Brandenburg Gate, so the list says
Brandenburg Gate.

Landmark coordinates are APPROXIMATE — city-accurate, which is all that can
matter when one screen pixel spans several kilometres. They position a small
decorative glyph, nothing is measured from them, and no forecast depends on
them.

The energy icons are the opposite: those come from real installed capacity in
each country's own payload (Energy-Charts), so a country only shows a nuclear
icon if it actually has nuclear capacity on the grid.

Output: webapp/site/data/landmarks.json

Run:  python webapp/build_landmarks.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("landmarks")

SITE_DATA = Path(__file__).parent / "site" / "data"
OUT = SITE_DATA / "landmarks.json"

# (name, glyph, lat, lon) — two per country, picked for recognisability.
# Glyphs are drawn in the frontend; see GLYPHS there for the full set.
LANDMARKS: dict[str, list[tuple]] = {
    "at": [("Schönbrunn Palace", "palace", 48.185, 16.312),
           ("St. Stephen's Cathedral", "cathedral", 48.209, 16.373)],
    "be": [("Atomium", "sphere", 50.895, 4.341),
           ("Grand-Place", "guildhall", 50.847, 4.352)],
    "bg": [("Alexander Nevsky Cathedral", "dome", 42.696, 23.333),
           ("Rila Monastery", "monastery", 42.133, 23.340)],
    "ch": [("Matterhorn", "mountain", 45.976, 7.658),
           ("Chapel Bridge, Lucerne", "bridge", 47.052, 8.307)],
    "cz": [("Prague Castle", "castle", 50.090, 14.400),
           ("Charles Bridge", "bridge", 50.086, 14.411)],
    "de": [("Brandenburg Gate", "gate", 52.516, 13.378),
           ("Cologne Cathedral", "cathedral", 50.941, 6.958)],
    "dk": [("The Little Mermaid", "statue", 55.693, 12.599),
           ("Nyhavn", "harbour", 55.680, 12.591)],
    "ee": [("Toompea Castle, Tallinn", "castle", 59.436, 24.740),
           ("Kadriorg Palace", "palace", 59.438, 24.791)],
    "es": [("Sagrada Família", "cathedral", 41.404, 2.174),
           ("Alhambra", "fortress", 37.176, -3.588)],
    "fi": [("Helsinki Cathedral", "dome", 60.170, 24.952),
           ("Suomenlinna", "fortress", 60.145, 24.988)],
    "fr": [("Eiffel Tower", "tower", 48.858, 2.294),
           ("Mont-Saint-Michel", "monastery", 48.636, -1.511)],
    "gr": [("Parthenon", "temple", 37.971, 23.727),
           ("Santorini", "dome", 36.393, 25.461)],
    "hr": [("Dubrovnik City Walls", "fortress", 42.641, 18.108),
           ("Diocletian's Palace", "palace", 43.508, 16.440)],
    "hu": [("Hungarian Parliament", "parliament", 47.507, 19.046),
           ("Tihany Abbey", "monastery", 46.914, 17.888)],
    "it": [("Colosseum", "colosseum", 41.890, 12.492),
           ("Leaning Tower of Pisa", "leaning", 43.723, 10.396)],
    "lt": [("Gediminas Tower", "tower", 54.687, 25.291),
           ("Trakai Island Castle", "castle", 54.652, 24.933)],
    "lu": [("Bock Casemates", "fortress", 49.611, 6.134),
           ("Vianden Castle", "castle", 49.935, 6.208)],
    "lv": [("House of the Blackheads", "guildhall", 56.947, 24.107),
           ("Rundāle Palace", "palace", 56.414, 24.024)],
    "me": [("Sveti Stefan", "harbour", 42.256, 18.892),
           ("Kotor Old Town", "fortress", 42.424, 18.771)],
    "nl": [("Kinderdijk Windmills", "windmill", 51.884, 4.640),
           ("Amsterdam Canals", "guildhall", 52.360, 4.885)],
    "no": [("Geirangerfjord", "mountain", 62.101, 7.005),
           ("Bryggen, Bergen", "harbour", 60.397, 5.324)],
    "pl": [("Wawel Castle", "castle", 50.054, 19.935),
           ("Warsaw Royal Castle", "palace", 52.248, 21.014)],
    "pt": [("Belém Tower", "tower", 38.692, -9.216),
           ("Pena Palace", "palace", 38.788, -9.391)],
    "ro": [("Bran Castle", "castle", 45.515, 25.367),
           ("Palace of the Parliament", "parliament", 44.428, 26.088)],
    "rs": [("Belgrade Fortress", "fortress", 44.823, 20.451),
           ("Church of Saint Sava", "dome", 44.798, 20.469)],
    "se": [("Stockholm Royal Palace", "palace", 59.327, 18.072),
           ("Visby City Wall", "fortress", 57.640, 18.296)],
    "si": [("Lake Bled Church", "monastery", 46.362, 14.088),
           ("Ljubljana Castle", "castle", 46.049, 14.508)],
    "sk": [("Bratislava Castle", "castle", 48.142, 17.100),
           ("St. Elisabeth Cathedral", "cathedral", 48.720, 21.258)],
}

# Installed-capacity type names (Energy-Charts) folded into the four icons.
#
# Two traps here, both found by checking Germany against reality:
#
#  - "Solar AC" (114.8 GW) and "Solar DC" (126.1 GW) are the SAME panels rated
#    at the inverter and at the module. Adding them gave 240.9 GW for a country
#    with roughly 100. So solar TAKES THE FIRST NAME PRESENT rather than summing,
#    preferring the AC/grid-facing figure that is comparable with wind and
#    nuclear ratings.
#  - Pumped storage is a store, not a source: it consumes about as much as it
#    returns. It is excluded, which leaves Germany on 5.3 GW of real hydro.
#
# Wind onshore + offshore ARE distinct assets, so those genuinely sum.
SOURCE_MAP = {
    "solar": (("Solar AC", "Solar DC", "Solar"), "first"),
    "wind": (("Wind onshore", "Wind offshore"), "sum"),
    "hydro": (("Hydro", "Hydro Run-of-River", "Hydro water reservoir"), "sum"),
    "nuclear": (("Nuclear",), "sum"),
}
MIN_GW = 0.15   # below this a source is a rounding error, not part of the mix


def energy_mix(code: str) -> list[dict]:
    """Which of solar / wind / hydro / nuclear this country actually runs.

    Reads the country's own payload, so the icons follow real installed
    capacity rather than an assumption. Uses the most recent year that has a
    number — the feed runs a few years ahead with planned build-out, and we
    want what exists, not what is scheduled.
    """
    path = SITE_DATA / f"{code}.json"
    if not path.exists():
        return []
    try:
        cap = json.loads(path.read_text(encoding="utf-8")).get("capacity")
    except (ValueError, OSError):
        return []
    if not cap:
        return []

    years, types = cap.get("years", []), cap.get("types", {})
    try:
        this_year = int(__import__("datetime").date.today().year)
    except Exception:  # noqa: BLE001
        this_year = 2026

    def latest(series):
        """Last reported value at or before the current year (the feed runs
        ahead with planned build-out; we want what exists)."""
        for i in range(len(years) - 1, -1, -1):
            try:
                if int(years[i]) > this_year:
                    continue
            except (TypeError, ValueError):
                continue
            v = series[i] if i < len(series) else None
            if v is not None:
                return float(v)
        return None

    out = []
    for icon, (names, how) in SOURCE_MAP.items():
        gw = 0.0
        for n in names:
            series = types.get(n)
            if not series:
                continue
            v = latest(series)
            if v is None:
                continue
            if how == "first":
                gw = v
                break
            gw += v
        if gw >= MIN_GW:
            out.append({"source": icon, "gw": round(gw, 1)})
    out.sort(key=lambda r: -r["gw"])
    return out


def main() -> None:
    countries = {}
    for code, marks in LANDMARKS.items():
        countries[code] = {
            "landmarks": [{"name": n, "glyph": g, "lat": la, "lon": lo}
                          for n, g, la, lo in marks],
            "energy": energy_mix(code),
        }
        mix = ", ".join(f"{e['source']} {e['gw']}GW" for e in countries[code]["energy"])
        log.info("  %s: %d landmarks | %s", code, len(marks), mix or "no capacity data")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "note": ("Landmark positions are approximate and decorative — city-accurate, "
                 "which is all that resolves at this map scale. Nothing is measured "
                 "from them. Energy icons come from real installed capacity "
                 "(Energy-Charts) in each country's own payload."),
        "countries": countries,
    }, ensure_ascii=False), encoding="utf-8")

    n_e = sum(1 for v in countries.values() if v["energy"])
    log.info("DONE: %d countries, %d with capacity data, %.0f KB",
             len(countries), n_e, OUT.stat().st_size / 1024)


if __name__ == "__main__":
    main()
