"""Every number of parameter_tuning.tex and of the section Tuning on the data of
group_identification.tex, from the runs on SSD_DISANTE."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))  # tuning.py imports run, delays and contacts by name
_spec = importlib.util.spec_from_file_location("group_tuning", HERE / "tuning.py")
tuning = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tuning)

pytestmark = pytest.mark.skipif(not tuning.PROFILE.exists(), reason="SSD_DISANTE is not mounted or not computed")
NARROW = {q: tuning.NARROW.format(q) for q in tuning.QS}


def band(name):
    return [round(x / 60, 1) for x in json.loads((tuning.RUNS / name / "calibration_band.json").read_text())
            ["percentiles_5_50_95"][::2]]


def shares(name):
    f = pd.read_parquet(tuning.RUNS / name / "flights.parquet").merge(
        pd.read_parquet(tuning.RUNS / name / "flight_summary.parquet"), on="flight")
    ok = (f.dt_native <= 10 + 1e-6) & (f.n_grid > 0)
    out = {}
    for d in ("para", "hang"):
        m = ok & (f.discipline == d)
        out[d] = (int(m.sum()), round(100 * (f.n_entries[m] > 0).mean(), 1))
    return out


def test_table_of_tau_star_and_groups():
    rows = {(tuning.WIDE, 0.05): (89.5, [79.5, 91.5], 8.9, 2.1, 6090),
            (NARROW[0.05], 0.05): (51.9, [49.1, 53.3], 13.7, 3.8, 10553),
            (NARROW[0.1], 0.1): (38.1, [37.0, 41.1], 21.7, 6.9, 17169),
            (NARROW[0.2], 0.2): (26.4, [25.2, 27.2], 31.7, 13.9, 26504)}
    for (name, q), (tau, interval, para, hang, episodes) in rows.items():
        c = tuning.calibration(name)
        assert round(c["tau_star"] / 60, 1) == tau == round(tuning.tau_star(c, q) / 60, 1)
        if interval is not None:
            assert band(name) == interval
        s = shares(name)
        assert (s["para"], s["hang"]) == ((154598, para), (6039, hang))
        assert len(pd.read_parquet(tuning.RUNS / name / "episodes.parquet")) == episodes


def test_chance_shares_of_figure_fdr():
    for name, expected in ((tuning.WIDE, [0.77, 0.59, 0.32]), (NARROW[0.05], [0.74, 0.47, 0.16])):
        c = tuning.calibration(name)
        tau, fdr = np.array(c["tau"]), np.array(c["fdr"])
        assert np.round(fdr[np.searchsorted(tau, [0, 600, 1800][:len(expected)])], 2).tolist() == expected


def test_delay_profile():
    p = json.loads(tuning.PROFILE.read_text())
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
    classes = tuning.ell_classes(pd.read_parquet(tuning.RUNS / tuning.WIDE / "intervals_raw.parquet"))
    short, long = classes[0, 10], classes[90, np.inf]
    assert round((short >= 285).mean(), 2) == 0.29
    assert np.median(long) == 25 and round((long <= 120).mean(), 1) == 0.9
    assert ((long >= 285).sum(), len(long)) == (6, 10734)


def test_separation_bound_grid_and_costs():
    p = dict(tuning.run.STARTING, delta=120)
    assert round((p["eps_h"] + p["v_max"] * p["delta"]) / 1000, 1) == 3.8
    assert round(p["v_max"] * p["step"] / 2) == 69
    assert round((p["eps_h"] + p["v_max"] * (p["delta"] + 60)) / 1000, 1) == 5.5  # D_c
    assert 2 * (p["delta"] // p["step"]) + 1 == 49
    days = tuning.RUNS / NARROW[0.1] / "days"
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
    rows = json.loads((tuning.run.DATA / "plateau" / "sample300.json").read_text())
    got = {k: (round(v["tau_star"] / 60, 1), round(v["f_G"], 3), v["episodes"], v["group_flights"])
           for k, v in rows.items()}
    assert got == PLATEAU
    assert {v["flights"] for v in rows.values()} == {7051}
    assert sum(v["tau_star"] == 7565 for v in rows.values()) == 6
