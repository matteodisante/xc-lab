import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent))
import analysis  # noqa: E402
import tamsd  # noqa: E402
import twoway  # noqa: E402
from synthetic import fbm_fixes  # noqa: E402

N_FLIGHTS, N_SAMPLES = 120, 400


def synthetic(h_para=0.8, h_hang=0.65, seed=0):
    """A flight table and pieces of fBm flights: para of exponent h_para, hang of h_hang, in two terrain classes
    (plains at 100 m, low mountains at 1000 m) and two regions' worth of takeoffs."""
    rng = np.random.default_rng(seed)
    rows, pieces = [], {"row": [], "duration": [], "sums": [], "origins": []}
    for d, h in (("para", h_para), ("hang", h_hang)):
        for i in range(N_FLIGHTS):
            mountain = i % 2 == 0
            rows.append({"discipline": d, "flight_id": f"{d}{i}", "pilot": f"{d} pilot {i % 20}", "date": f"2020-01-{1 + i % 5:02d}",
                         "lat0": 45.0 if mountain else 49.0, "lon0": 6.5 if mountain else 0.0, "flight_type": "Dist libre",
                         "ends_km": 20.0, "path_km": 100.0, "measured": True, "median_elev": 1000.0 if mountain else 100.0,
                         "n_points": 100, "n_on_dem": 100})
            for _, duration, sums, origins in tamsd.pieces(fbm_fixes(N_SAMPLES, h, rng), 0, 1e9):
                pieces["row"].append(len(rows) - 1)
                pieces["duration"].append(duration)
                pieces["sums"].append(sums)
                pieces["origins"].append(origins)
    table = pd.DataFrame(rows)
    table.loc[table.index[:3], ["ends_km"]] = 5.0  # three para flights that close the loop (opening 0.05): dropped
    return table, {k: np.array(v) for k, v in pieces.items()}


@pytest.fixture(scope="module")
def data():
    table, p = synthetic()
    F, q, (pilot_of_cell, block_of_cell) = analysis.selection(table, p)
    stacked = twoway.counts(pilot_of_cell, block_of_cell, np.random.default_rng(1), size=200)
    return table, F, q, stacked


def test_selection_drops_closed_flights_and_classifies_the_terrain(data):
    table, F, q, _ = data
    assert len(F) == len(table) - 3 and set(F["cls"]) == {"plains", "low mountains"}
    assert set(F["region"]) == {"Alps", "Channel Coast"}
    assert (F["longest"] == (N_SAMPLES - 1) * 10).all() and len(q["owner"]) == len(F) and q["owner"].max() == len(F) - 1


def test_available_flights_and_the_longest_lag(data):
    _, F, _, _ = data
    mask = analysis.population(F, "para", "all")
    n = analysis.available(F, mask)
    assert n[0] == mask.sum() and n[-1] == 0 and np.all(np.diff(n) <= 0)
    masks = {d: analysis.population(F, d, "all") for d in ("para", "hang")}
    T = analysis.longest_lag(F, masks)
    assert T == tamsd.LONG_LAGS[tamsd.LONG_LAGS <= (N_SAMPLES - 1) * 10 * tamsd.SUPPORT][-1]
    few = {"para": masks["para"], "hang": masks["hang"] & (F["pilot_code"] < 0).to_numpy()}
    assert analysis.longest_lag(F, few) is None  # no hang flight at all


def test_the_fixed_population_is_the_same_flights_at_every_lag(data):
    _, F, q, stacked = data
    c = analysis.curve(F, q, analysis.population(F, "para", "all"), 1000, stacked, True)
    assert len(set(c["n"])) == 1 and c["n"][0] == (F["discipline"] == "para").sum()
    c = analysis.curve(F, q, analysis.population(F, "para", "all"), 5000, stacked, False)
    assert c["n"][0] == (F["discipline"] == "para").sum() and c["n"][-1] < c["n"][0] and np.all(np.diff(c["n"]) <= 0)


def test_known_exponents_come_back_with_their_difference(data):
    _, F, q, stacked = data
    masks = {d: analysis.population(F, d, "all") for d in ("para", "hang")}
    r = analysis.pair(F, q, masks, stacked)
    a = r["A_fixed"]
    assert a["para"]["H"] == pytest.approx(0.8, abs=0.04) and a["hang"]["H"] == pytest.approx(0.65, abs=0.04)
    assert a["d_H"]["value"] == pytest.approx(a["para"]["H"] - a["hang"]["H"])
    lo, hi = a["d_H"]["band"]
    assert lo < 0.15 < hi and lo > 0.05
    assert a["para"]["alpha"][5] == pytest.approx(1.6, abs=0.12)
    assert r["B_to_1000"]["T"] == 1000 and r["B"]["para"]["flights"][0] == masks["para"].sum()


def test_two_populations_of_the_same_exponent_do_not_differ():
    table, p = synthetic(0.7, 0.7, seed=5)
    F, q, (pc, bc) = analysis.selection(table, p)
    stacked = twoway.counts(pc, bc, np.random.default_rng(2), size=200)
    r = analysis.pair(F, q, {d: analysis.population(F, d, "all") for d in ("para", "hang")}, stacked)
    lo, hi = r["A_fixed"]["d_H"]["band"]
    assert lo < 0 < hi


def test_stratified_cells_are_separated_and_a_small_cell_is_not_fitted(data):
    _, F, q, stacked = data
    plains = analysis.pair(F, q, {d: analysis.population(F, d, "plains") for d in ("para", "hang")}, stacked)
    assert plains["flights"]["hang"] == N_FLIGHTS // 2 and "A_fixed" in plains
    hills = analysis.pair(F, q, {d: analysis.population(F, d, "hills") for d in ("para", "hang")}, stacked)
    assert hills["skipped"] and "A_fixed" not in hills and hills["flights"] == {"para": 0, "hang": 0}


def test_region_pairs_use_the_region_masks(data):
    _, F, q, stacked = data
    alps, channel = (analysis.region_mask(F, "para", r) for r in ("Alps", "Channel Coast"))
    assert (alps & channel).sum() == 0 and alps.sum() == (analysis.population(F, "para", "low mountains")).sum()
    r = analysis.pair(F, q, {"Alps": alps, "Channel Coast": channel}, stacked)
    assert set(r["A_fixed"]) >= {"Alps", "Channel Coast", "d_H"} and r["flights"]["Alps"] == alps.sum()


def test_counts_and_funnel(data):
    table, F, _, _ = data
    c = analysis.counts(F)
    assert c["cells"]["hang"]["all"]["flights"] == N_FLIGHTS and c["cells"]["hang"]["plains"]["flights"] == N_FLIGHTS // 2
    assert c["regions"]["para"]["Alps"]["flights"] + c["regions"]["para"]["Channel Coast"]["flights"] == N_FLIGHTS - 3
    assert sum(c["histogram"]["n"]["para"]) == N_FLIGHTS - 3
    table = table.assign(eligible=True, declared=True)
    f = analysis.funnel(table)
    assert f["para"]["open"] == N_FLIGHTS - 3 and f["hang"]["open"] == N_FLIGHTS and f["pilots in both"] == 0
