import json
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import report  # noqa: E402

HERE = Path(__file__).parent
RESULTS = HERE / "out" / "full" / "results.json"
MACROS = HERE / "generated" / "macros.tex"


def test_number_formats():
    assert report.signed(0.0243) == "$+$0.024" and report.signed(-0.0126) == "$-$0.013"
    assert report.signed(-0.0002) == "$+$0.000"  # no minus sign on a zero
    assert report.bracket([0.9215, 0.92351]) == "[0.921, 0.924]"
    assert report.thousands(55879) == "55\\,879"


def test_macros_of_a_pair_hold_the_values_of_its_designs():
    side = lambda h, n: {"H": h, "H_band": [h - 0.01, h + 0.01], "flights": [n + 5, n], "pilots": [3, 2], "lags": [10, 100, 1000]}
    r = {"A_fixed": {"para": side(0.9, 100), "hang": side(0.88, 40), "T": 1000, "d_H": {"value": 0.02, "band": [0.0, 0.04]},
                     "d_alpha": {"value": [0.01, 0.04, 0.06]}}}
    m = report.Macros()
    m.pair("", r, ("para", "hang"), "All")
    text = "\n".join(m.lines)
    for expected in (r"\UniDHAllAfixed}{$+$0.020}", r"\UniDHBandAllAfixed}{[$+$0.000, $+$0.040]}", r"\UniTAllAfixed}{1000}",
                     r"\UniHFirstAllAfixed}{0.900}", r"\UniHSecondAllAfixed}{0.880}", r"\UniNFirstAllAfixed}{100}",
                     r"\UniPSecondAllAfixed}{2}", r"\UniDAlphaAllAfixedHundred}{$+$0.040}", r"\UniDAlphaAllAfixedThousand}{$+$0.060}", r"\UniHBandSecondAllAfixed}{[0.870, 0.890]}"):
        assert expected in text


@pytest.mark.skipif(not (RESULTS.exists() and MACROS.exists()), reason="no full run")
def test_generated_macros_match_the_results():
    res = json.loads(RESULTS.read_text())
    macros = dict(re.findall(r"\\newcommand\{\\Uni(\w+)\}\{(.*)\}", MACROS.read_text()))
    raw = res["strata"]["all"]
    for key, tok in report.DESIGN_TOKEN.items():
        assert macros[f"DHAll{tok}"] == report.signed(raw[key]["d_H"]["value"])
        assert macros[f"HFirstAll{tok}"] == f"{raw[key]['para']['H']:.3f}"
        assert macros[f"TAll{tok}"] == str(raw[key]["T"])
    assert macros["FunnelParaOpen"] == report.thousands(res["funnel"]["para"]["open"])
    assert macros["CellHangLowFlights"] == report.thousands(res["terrain"]["cells"]["hang"]["low mountains"]["flights"])
    assert macros["RegionParaAlpsFlights"] == report.thousands(res["terrain"]["regions"]["para"]["Alps"]["flights"])


@pytest.mark.skipif(not RESULTS.exists(), reason="no full run")
def test_results_are_consistent():
    res = json.loads(RESULTS.read_text())
    c = res["constants"]
    assert c["edges_m"] == [300.0, 800.0, 1500.0] and c["min_opening"] == 0.1 and c["tau_fixed"] == 1000
    cells = res["terrain"]["cells"]
    for d in ("para", "hang"):
        assert sum(cells[d][k]["flights"] for k in ("plains", "hills", "low mountains", "high mountains", "unknown")) == cells[d]["all"]["flights"]
        assert cells[d]["all"]["flights"] == res["funnel"][d]["open"]
        assert sum(r["flights"] for r in res["terrain"]["regions"][d].values()) <= cells[d]["all"]["flights"]
    for s, r in res["strata"].items():
        if "skipped" in r:
            continue
        a = r["A_fixed"]
        assert len(set(a["para"]["flights"])) == 1 and len(set(a["hang"]["flights"])) == 1  # the fixed population
        assert r["B"]["para"]["flights"][0] >= r["B"]["para"]["flights"][-1]
        lo, hi = a["d_H"]["band"]
        assert lo <= a["d_H"]["value"] <= hi
        assert a["d_H"]["value"] == pytest.approx(a["para"]["H"] - a["hang"]["H"])
