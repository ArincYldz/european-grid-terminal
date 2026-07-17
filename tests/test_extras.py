"""Tests for the extra data layers: flow reconciliation and capture rates.

Both encode a claim that is easy to get silently backwards:
  - a reversed flow arrow still renders, it is just a lie about physics;
  - a capture rate above 100% would flatter every revenue estimate on the site.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from webapp.precompute import _build_flow_network, _capture_rates


def test_flow_network_dedupes_the_two_sides_of_one_border():
    """Germany and France report the SAME link from opposite ends. One edge."""
    payloads = [
        {"code": "de", "flows": [{"name": "France", "net_gw": 1.97, "direction": "import"}]},
        {"code": "fr", "flows": [{"name": "Germany", "net_gw": -1.97, "direction": "export"}]},
    ]
    net = _build_flow_network(payloads)
    assert len(net["edges"]) == 1, f"expected one edge, got {net['edges']}"
    e = net["edges"][0]
    # Positive = import into the reporting country, so France exports to Germany.
    assert (e["src"], e["dst"]) == ("fr", "de"), e
    assert e["gw"] == 1.97


def test_flow_direction_follows_the_sign_not_the_reporter():
    """A single negative reading must point AWAY from the reporting country."""
    net = _build_flow_network([
        {"code": "de", "flows": [{"name": "Poland", "net_gw": -1.16}]},
    ])
    e = net["edges"][0]
    assert (e["src"], e["dst"]) == ("de", "pl"), e
    assert e["gw"] == 1.16


def test_flow_network_skips_unmapped_partners_and_tiny_flows():
    net = _build_flow_network([
        {"code": "de", "flows": [
            {"name": "Atlantis", "net_gw": 5.0},      # not a real partner
            {"name": "Belgium", "net_gw": 0.01},      # below the noise floor
            {"name": "Austria", "net_gw": -0.98},     # keep this one
        ]},
    ])
    assert [e["dst"] for e in net["edges"]] == ["at"]
    assert "at" in net["coords"] and "de" in net["coords"]


def test_flow_network_gives_every_drawn_country_coordinates():
    """The map cannot place an edge whose endpoint has no lat/lon."""
    net = _build_flow_network([
        {"code": "de", "flows": [{"name": "United Kingdom", "net_gw": 1.0},
                                 {"name": "Ukraine", "net_gw": -0.5}]},
    ])
    for e in net["edges"]:
        assert e["src"] in net["coords"], e
        assert e["dst"] in net["coords"], e


def _hist(price, solar, wind=None):
    n = len(price)
    return pd.DataFrame({
        "price_eur_mwh": np.array(price, dtype=float),
        "solar_mw": np.array(solar, dtype=float),
        "wind_mw": np.array(wind if wind is not None else [1.0] * n, dtype=float),
    })


def test_capture_rate_is_100pct_when_generation_is_flat():
    """Flat output earns exactly the average price — the definition's anchor."""
    r = _capture_rates(_hist(price=[10, 50, 90, 50], solar=[5, 5, 5, 5]))
    assert abs(r["solar_capture_pct"] - 100.0) < 1e-6, r


def test_capture_rate_drops_when_solar_generates_into_cheap_hours():
    """Cannibalisation: generating at the glut hours must score below 100%."""
    # Solar peaks exactly where the price collapses.
    r = _capture_rates(_hist(price=[100, 20, 20, 100], solar=[0, 10, 10, 0]))
    assert r["solar_capture_pct"] < 100.0, r
    assert r["solar_captured_eur_mwh"] == 20.0, r     # it only ever earns 20
    assert r["mean_price_eur_mwh"] == 60.0, r


def test_capture_rate_exceeds_100pct_when_generation_chases_scarcity():
    """The metric must stay honest in the other direction too, not clip at 100."""
    r = _capture_rates(_hist(price=[100, 20, 20, 100], solar=[10, 0, 0, 10]))
    assert r["solar_capture_pct"] > 100.0, r


def test_capture_rate_skips_a_technology_that_never_generates():
    r = _capture_rates(_hist(price=[50, 60], solar=[0, 0], wind=[3, 4]))
    assert "solar_capture_pct" not in r
    assert "wind_capture_pct" in r


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as exc:
                fails += 1
                print(f"FAIL {name}: {exc}")
    print("--- all extras tests passed" if not fails else f"--- {fails} FAILED")
    sys.exit(1 if fails else 0)
