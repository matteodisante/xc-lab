"""The numbers of the README: the chosen run (delta = 2 min, q = 0.1) on SSD_DISANTE."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

OUT = Path("/Volumes/SSD_DISANTE/xc-lab/group-identification/runs/d120_eh500_ez300_th45_g120_q0.1")
pytestmark = pytest.mark.skipif(not (OUT / "calibration.json").exists(), reason="run run.py first")


@pytest.fixture(scope="module")
def out():
    return {name: pd.read_parquet(OUT / f"{name}.parquet") for name in
            ("flights", "flight_summary", "groups", "configurations", "episodes", "membership", "intervals_raw")}


def test_calibration_numbers_of_the_readme():
    c = json.loads((OUT / "calibration.json").read_text())
    assert (c["tau_star"], c["params"]["delta"], c["params"]["q"]) == (2285, 120, 0.1)
    tau, fdr = np.array(c["tau"]), np.array(c["fdr"])
    k = np.searchsorted(tau, c["tau_star"])
    assert fdr[k] <= c["params"]["q"] < fdr[k - 1]
    band = json.loads((OUT / "calibration_band.json").read_text())
    assert band["without_tau_star"] == 0 and band["percentiles_5_50_95"][::2] == [2220, 2465]


def test_flight_numbers_of_the_readme(out):
    f = out["flights"].merge(out["flight_summary"], on="flight")
    assert len(f) == 162485
    ok = (f.dt_native <= 10 + 1e-6) & (f.n_grid > 0)
    assert ok.sum() == 160637
    for d, (group, total, share) in {"para": (33501, 154598, 21.7), "hang": (418, 6039, 6.9)}.items():
        m = ok & (f.discipline == d)
        assert (m.sum(), (f.n_entries[m] > 0).sum()) == (total, group)
        assert round(100 * group / total, 1) == share


def test_group_numbers_of_the_readme(out):
    iv = out["intervals_raw"]
    assert len(iv) == 728866 and (iv.duration >= 2285).sum() == 36575
    assert len(out["groups"]) == 40051 and len(out["configurations"]) == 52856
    e = out["episodes"]
    assert len(e) == 17169 and e.n_flights.median() == 2 and e.n_flights.max() == 53
    assert round((e.t_end - e.t_start).median() / 3600, 2) == 0.97
    m = out["membership"]
    after = m[m.state != "L"]
    duration = after.t_end - after.t_start
    assert round(duration[after.state == "G"].sum() / duration.sum(), 3) == 0.123


def test_tables_are_consistent(out):
    m, groups = out["membership"], out["groups"]
    # Every G stretch is a group that contains the flight, over the group's time.
    g = m[m.state == "G"].merge(groups, on="group_id", suffixes=("", "_g"))
    assert (g.t_start == g.t_start_g).all() and (g.t_end == g.t_end_g).all()
    assert all(f in members for f, members in zip(g.flight, g.members))
    assert (groups.n == groups.members.map(len)).all() and (groups.n >= 2).all()
    # The configurations of a group keep its members, are connected and tile its time.
    c = out["configurations"].merge(groups, on="group_id", suffixes=("", "_g")).sort_values(["group_id", "t_start"])
    assert all(list(a) == list(b) for a, b in zip(c.members, c.members_g))
    assert (c.edges.map(len) >= c.n - 1).all()
    same = c.group_id.to_numpy()[1:] == c.group_id.to_numpy()[:-1]
    assert (c.t_start.to_numpy()[1:][same] == c.t_end.to_numpy()[:-1][same]).all()
    span = c.groupby("group_id").agg(a=("t_start", "min"), b=("t_end", "max"), a_g=("t_start_g", "first"),
                                     b_g=("t_end_g", "first"))
    assert (span.a == span.a_g).all() and (span.b == span.b_g).all()
    # The stretches of a flight tile its time from takeoff to landing.
    m = m.sort_values(["flight", "t_start"])
    same = m.flight.to_numpy()[1:] == m.flight.to_numpy()[:-1]
    assert (m.t_start.to_numpy()[1:][same] == m.t_end.to_numpy()[:-1][same]).all()
