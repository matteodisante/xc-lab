"""Every number of the calibration, tuning and results sections of group_identification.tex,
from the runs on SSD_DISANTE."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))  # assessment.py imports run, delays and contacts by name
_spec = importlib.util.spec_from_file_location("group_assessment", HERE / "assessment.py")
assessment = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(assessment)

pytestmark = pytest.mark.skipif(not assessment.PROFILE.exists(), reason="SSD_DISANTE is not mounted or not computed")
NARROW = {q: assessment.NARROW.format(q) for q in assessment.QS}


def band(name):
    return [round(x / 60, 1) for x in json.loads((assessment.RUNS / name / "calibration_band.json").read_text())
            ["percentiles_5_50_95"][::2]]


def shares(name):
    f = pd.read_parquet(assessment.RUNS / name / "flights.parquet").merge(
        pd.read_parquet(assessment.RUNS / name / "flight_summary.parquet"), on="flight")
    ok = (f.dt_native <= 10 + 1e-6) & (f.n_grid > 0)
    out = {}
    for d in ("para", "hang"):
        m = ok & (f.discipline == d)
        out[d] = (int(m.sum()), round(100 * (f.n_entries[m] > 0).mean(), 1))
    return out


def test_table_of_tau_star_and_groups():
    rows = {(assessment.WIDE, 0.05): (89.5, [79.5, 91.5], 8.9, 2.1, 6090),
            (NARROW[0.05], 0.05): (51.9, [49.1, 53.3], 13.7, 3.8, 10553),
            (NARROW[0.1], 0.1): (38.1, [37.0, 41.1], 21.7, 6.9, 17169),
            (NARROW[0.2], 0.2): (26.4, [25.2, 27.2], 31.7, 13.9, 26504)}
    for (name, q), (tau, interval, para, hang, episodes) in rows.items():
        c = assessment.calibration(name)
        assert round(c["tau_star"] / 60, 1) == tau == round(assessment.tau_star(c, q) / 60, 1)
        if interval is not None:
            assert band(name) == interval
        s = shares(name)
        assert (s["para"], s["hang"]) == ((154598, para), (6039, hang))
        assert len(pd.read_parquet(assessment.RUNS / name / "episodes.parquet")) == episodes


def test_chance_shares_of_figure_fdr():
    for name, expected in ((assessment.WIDE, [0.77, 0.59, 0.32]), (NARROW[0.05], [0.74, 0.47, 0.16])):
        c = assessment.calibration(name)
        tau, fdr = np.array(c["tau"]), np.array(c["fdr"])
        assert np.round(fdr[np.searchsorted(tau, [0, 600, 1800][:len(expected)])], 2).tolist() == expected


def test_delay_profile():
    p = json.loads(assessment.PROFILE.read_text())
    d, ratio = np.array(p["delay"]), np.array(p["ratio"])
    assert round(p["exposure_h"]) == 58910
    assert p["below"] == {"4": 105, "3": 195, "2": 500}
    assert round(ratio[d == 15][0], 1) == 5.6 and round(ratio[d == 900][0], 1) == 1.5
    assert round(ratio[d <= 120].min(), 1) == 3.7 and round(ratio[d <= 120].max(), 1) == 5.7
    # The rule rejected in the text: the delay holding 95% of the excess of real over chance.
    excess = np.maximum(np.array(p["real"]) - np.array(p["chance"]), 0) * np.where(d > 0, 2, 1)
    assert round(d[np.searchsorted(np.cumsum(excess) / excess.sum(), 0.95)] / 60) == 13
    apart = np.array(p["apart_real"])
    within = lambda km: apart[: int(km * 1000 / p["apart_bin_m"])].sum() / apart.sum()  # noqa: E731
    assert (round(within(2), 2), round(within(6), 2)) == (0.58, 0.94)


def test_delays_inside_the_intervals():
    intervals = pd.read_parquet(assessment.RUNS / assessment.WIDE / "intervals_raw.parquet")
    assert (len(intervals), int(intervals["ell_med"].isna().sum())) == (965770, 169686)
    classes = assessment.ell_classes(intervals)
    # Table "delay by duration": intervals, median |delay| (s), % within 2 min, % at the edge.
    rows = {(0, 10): (436444, 215, 31, 30), (10, 30): (246548, 165, 38, 14), (30, 90): (102358, 82.5, 64, 2.4)}
    for key, (n, median, within, edge) in rows.items():
        x = classes[key]
        assert (len(x), np.median(x), round(100 * (x <= 120).mean())) == (n, median, within)
        assert round(100 * (x >= 285).mean(), 1 if edge < 10 else None) == edge
    long = classes[90, np.inf]
    assert np.median(long) == 25 and round(100 * (long <= 120).mean()) == 90
    assert ((long >= 285).sum(), len(long)) == (7, 10734)
    assert assessment.calibration(assessment.WIDE)["tau_star"] / 60 == 89.5


def test_separation_bound_grid_and_costs():
    p = dict(assessment.run.STARTING, delta=120)
    assert round((p["eps_h"] + p["v_max"] * p["delta"]) / 1000, 1) == 3.8
    assert round(p["v_max"] * p["step"] / 2) == 69
    assert round((p["eps_h"] + p["v_max"] * (p["delta"] + 60)) / 1000, 1) == 5.5  # D_c
    assert 2 * (p["delta"] // p["step"]) + 1 == 49
    days = assessment.RUNS / NARROW[0.1] / "days"
    assert len([d for d in days.glob("*.pkl") if not d.name.startswith(".")]) == 6209


PLATEAU = {
    "start": (126.1, 0.034, 98, 210), "eps_h=250": (75.1, 0.058, 253, 537),
    "eps_h=1000": (126.8, 0.065, 162, 372), "eps_z=150": (126.1, 0.020, 61, 125),
    "eps_z=600": (126.1, 0.044, 122, 268), "delta=120": (42.6, 0.106, 600, 1170),
    "delta=600": (128.4, 0.062, 153, 365), "theta_max=30": (126.1, 0.034, 97, 208),
    "theta_max=90": (126.1, 0.034, 98, 210), "gap=60": (75.8, 0.087, 332, 732),
    "gap=300": (126.1, 0.055, 141, 318),
}


def test_plateau_table():
    rows = json.loads((assessment.run.DATA / "plateau" / "sample300.json").read_text())
    got = {k: (round(v["tau_star"] / 60, 1), round(v["f_G"], 3), v["episodes"], v["group_flights"])
           for k, v in rows.items()}
    assert got == PLATEAU
    assert {v["flights"] for v in rows.values()} == {7051}
    assert sum(v["tau_star"] == 7565 for v in rows.values()) == 6


def test_structure_of_the_groups():
    # Table "structure": number, median duration (min), % of items and of time with 2 and with
    # 5 or more participants, largest.
    rows = {(0.05, "episodes"): (10553, 74.6, 77, 67, 3.9, 8.1, 21),
            (0.1, "episodes"): (17169, 58.2, 74, 61, 5.7, 12.6, 53),
            (0.2, "episodes"): (26504, 44.5, 73, 56, 7.0, 17.2, 91),
            (0.05, "groups"): (20275, 52.7, 70, 81, 5.6, 2.3, 18),
            (0.1, "groups"): (40051, 30.4, 63, 77, 9.4, 3.5, 27),
            (0.2, "groups"): (75953, 17.1, 56, 74, 14.1, 5.0, 40)}
    for q in assessment.QS:
        runs = assessment.structure(NARROW[q])
        tau_star = assessment.calibration(NARROW[q])["tau_star"] / 60
        assert runs["episodes"][0].min() == tau_star
        for kind in ("episodes", "groups"):
            duration, n = runs[kind]
            sizes, by_count, by_time = assessment.size_shares(duration, n)
            got = (len(n), round(np.median(duration), 1), round(100 * by_count[sizes == 2][0]),
                   round(100 * by_time[sizes == 2][0]), round(100 * by_count[sizes >= 5].sum(), 1),
                   round(100 * by_time[sizes >= 5].sum(), 1), int(n.max()))
            assert got == rows[q, kind]
    duration = assessment.structure(NARROW[0.1])["groups"][0]
    assert round(np.percentile(duration, 10), 1) == 2.1
    assert round(np.median(assessment.structure(NARROW[0.1])["episodes"][0]) / 60, 2) == 0.97


def test_tau_star_by_class_of_shift():
    s = json.loads((assessment.RUNS / assessment.CHOSEN / "calibration_shifts.json").read_text())
    assert (s["q"], s["dates"], s["of_dates"]) == (0.1, 2737, 6209)
    # Table "shift classes": pairs, FDR at 10 and 30 min, tau* and its 5-95% band, min.
    rows = {"all": (241824, 0.48, 0.16, 38.2, [37.0, 41.5]), "short": (69444, 0.73, 0.31, 53.3, [49.7, 72.0]),
            "long": (119687, 0.26, 0.05, 23.0, [21.1, 24.4])}
    for c, (pairs, f10, f30, tau, interval) in rows.items():
        x = s["classes"][c]
        assert (x["pairs"], round(x["fdr_at"]["600"], 2), round(x["fdr_at"]["1800"], 2)) == (pairs, f10, f30)
        assert round(x["tau_star"] / 60, 1) == tau
        assert [round(v / 60, 1) for v in x["percentiles_5_50_95"][::2]] == interval
    short, long = s["classes"]["short"]["fdr_at"], s["classes"]["long"]["fdr_at"]
    assert (round(short["600"] / long["600"], 1), round(short["1800"] / long["1800"], 1)) == (2.8, 5.7)


def test_shifts_of_the_surrogates():
    r = assessment.shifts(assessment.CHOSEN)
    assert len(r) == 248741 and r.min() >= 245
    assert round(np.median(r) / 60) == 58
    assert (round(np.percentile(r, 5) / 60), round(np.percentile(r, 95) / 3600, 1)) == (8, 4.4)


def ratio_below(p, levels):
    """Real over chance per bin (the last bin, which holds the rest, left out), and the start of
    the first bin below each level."""
    g = np.array(p["real"])[:-1] / np.array(p["chance"])[:-1]
    return g, {x: int(np.argmax(g < x)) * p["bin"] for x in levels}


def test_separation_profile():
    p = json.loads(assessment.SEPARATION.read_text())
    assert (p["dates"], round(p["exposure_h"])) == (210, 58901)
    g, below = ratio_below(dict(p, bin=p["bin_m"]), (2, 1.5, 1.1, 1))
    assert round(g[0], 1) == 10.4 and below == {2: 500, 1.5: 900, 1.1: 2700, 1: 5700}
    d, real, chance = (np.array(p[k]) for k in ("distance_m", "real", "chance"))
    assert (round(real[d < 1000].sum(), 3), round(chance[d < 1000].sum(), 3)) == (0.066, 0.028)
    assert g[d[:-1] > 6000].mean() < 1


def test_tolerance_profiles():
    p = json.loads(assessment.TOLERANCES.read_text())
    assert (p["dates"], round(p["exposure_h"])) == (210, 58901)
    g, below = ratio_below(p["horizontal"], (2, 1.5))
    x = (np.arange(len(g)) + 0.5) * p["horizontal"]["bin"]
    assert round(g[0], 1) == 6.3 and below == {2: 250, 1.5: 500} and round(g[x > 2500].mean(), 2) == 1.13
    g, below = ratio_below(p["vertical"], (2, 1.5))
    x = (np.arange(len(g)) + 0.5) * p["vertical"]["bin"]
    assert round(g[0], 1) == 4.0 and below == {2: 150, 1.5: 275}
    assert round(g[(x > 300) & (x < 1000)].mean(), 2) == 1.51
    assert round(sum(p["gap"]["real"]) / sum(p["gap"]["chance"]), 2) == 1.46
    shares = assessment.gap_shares(p["gap"])
    assert round(np.abs(shares["real"] - shares["chance"]).max(), 2) == 0.02
    assert (round(shares["real"][-1], 2), round(shares["chance"][-1], 2)) == (0.08, 0.09)
