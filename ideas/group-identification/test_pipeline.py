import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))  # run.py imports grid, contacts and sweep by name


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


contacts = _load("group_contacts", "contacts.py")
sweep = _load("group_sweep", "sweep.py")
models = _load("group_models_for_sweep", "models.py")
run = _load("group_run", "run.py")
distances = _load("group_distances", "distances.py")
P = dict(run.PARAMS)
XYZ = ["X", "Y", "Z"]


# Synthetic days: flat tracks (Z = 0) on the 5 s grid, so that distances are those of the plane.

def synthetic_day(seed, n=3, steps=500):
    rng = np.random.default_rng(seed)
    rows, table = [], []
    for f in range(n):
        start = 5 * int(rng.integers(0, 60))
        t = start + 5 * np.arange(steps)
        heading = np.cumsum(rng.normal(0, 0.15, steps)) + rng.uniform(0, 0.3)
        speed = rng.uniform(6, 12, steps)
        speed[rng.random(steps) < 0.01] = 40  # a few steps faster than v_max split the pieces
        x = np.cumsum(5 * speed * np.cos(heading)) + rng.normal(0, 300)
        y = np.cumsum(5 * speed * np.sin(heading)) + rng.normal(0, 300)
        z = 1500 + np.cumsum(rng.normal(0, 8, steps))
        state = np.repeat(rng.integers(0, 3, steps // 50 + 1), 50)[:steps]
        keep = rng.random(steps) > 0.05  # instants without a position
        g = pd.DataFrame({"f": f, "seg": (t > start + 1500).astype(int), "t": t, "X": x, "Y": y, "Z": 0.0, "z": z,
                          "ve": np.cos(heading), "vn": np.sin(heading), "state": state})[keep]
        rows.append(g)
        table.append({"departure": float(g.t.min()), "landing": float(g.t.max()), "X0": x[0], "Y0": y[0], "Z0": 0.0,
                      "n_grid": len(g), "takeoff": float(start)})
    return pd.DataFrame(table), pd.concat(rows, ignore_index=True)


def brute_force(table, grid, i, j, p):
    """Pair intervals and median delays from every pair of instants within delta."""
    gi, gj = grid[grid.f == i], grid[grid.f == j]
    ti, tj = gi.t.to_numpy(), gj.t.to_numpy()
    d = np.linalg.norm(gi[XYZ].to_numpy()[:, None] - gj[XYZ].to_numpy()[None], axis=2)
    dz = np.abs(gi.z.to_numpy()[:, None] - gj.z.to_numpy()[None])
    glide = (gi.state.to_numpy()[:, None] == 2) & (gj.state.to_numpy()[None] == 2)
    dot = gi.ve.to_numpy()[:, None] * gj.ve.to_numpy()[None] + gi.vn.to_numpy()[:, None] * gj.vn.to_numpy()[None]
    m = (np.abs(tj[None] - ti[:, None]) <= p["delta"]) & (d < p["eps_h"]) & (dz < p["eps_z"])
    m &= ~glide | (dot > np.cos(np.radians(p["theta_max"])))
    a = np.union1d(ti[m.any(axis=1)], tj[m.any(axis=0)])
    lo, hi = max(table.departure[i], table.departure[j]), min(table.landing[i], table.landing[j])
    a = a[(a >= lo) & (a <= hi)]
    if not a.size:
        return []
    best = np.where(m, d, np.inf).argmin(axis=1)
    lag = tj[best] - ti
    search = (gi.state.to_numpy() == 1) | (gj.state.to_numpy()[best] == 1)
    out = []
    for on, off in zip(*contacts.runs_of(a, 5, p["gap"])):
        ok = m.any(axis=1) & (ti >= on) & (ti < off) & ~search
        out.append((on, off, float(np.median(lag[ok])) if ok.any() else np.nan))
    return out


@pytest.mark.parametrize("seed", range(6))
def test_phases_1_to_3_find_every_correspondence(seed):
    table, grid = synthetic_day(seed)
    tracks = contacts.Tracks(grid, 5)
    found = contacts.real_intervals(table, grid, tracks, P)
    expected = [(i, j, *iv) for i in range(3) for j in range(i + 1, 3) for iv in brute_force(table, grid, i, j, P)]
    got = list(found[["a", "b", "t_on", "t_off", "ell_med"]].itertuples(index=False, name=None))
    assert len(got) == len(expected)
    for g, e in zip(sorted(got), sorted(expected)):
        assert g[:4] == e[:4]
        assert g[4] == pytest.approx(e[4], nan_ok=True)


def test_synthetic_days_have_contacts_and_fast_steps():
    # Without contacts the comparison above would be empty, without fast steps no piece would split.
    n_contacts, n_fast = 0, 0
    for s in range(6):
        table, grid = synthetic_day(s)
        n_contacts += len(contacts.real_intervals(table, grid, contacts.Tracks(grid, 5), P))
        for _, g in grid.groupby(["f", "seg"]):
            step = np.linalg.norm(np.diff(g[XYZ].to_numpy(), axis=0), axis=1) / np.diff(g.t.to_numpy())
            n_fast += (step > P["v_max"]).sum()
    assert n_contacts >= 6 and n_fast >= 6


def line_pair(delay=120, side=250.0, dz=0.0, angle=0.0, state=2):
    """i flies east at 10 m/s; j flies the same line `delay` s later, `side` m to the north,
    `dz` m higher, heading turned by `angle` degrees (its track stays the same line)."""
    t = 5 * np.arange(400)
    rows = []
    for f, (lag, y, h) in enumerate([(0, 0.0, 0.0), (delay, side, dz)]):
        a = np.radians(angle if f else 0.0)
        rows.append(pd.DataFrame({"f": f, "seg": 0, "t": t, "X": 10.0 * (t - lag), "Y": y, "Z": 0.0, "z": 1500 + h,
                                  "ve": np.cos(a), "vn": np.sin(a), "state": state}))
    table = pd.DataFrame({"departure": [0.0, 0.0], "landing": [1995.0, 1995.0], "X0": 0.0, "Y0": [0.0, side],
                          "Z0": 0.0, "n_grid": 400, "takeoff": 0.0})
    grid = pd.concat(rows, ignore_index=True)
    return table, grid, contacts.real_intervals(table, grid, contacts.Tracks(grid, 5), P)


def test_a_follower_on_the_same_line_is_one_contact_with_its_delay():
    _, _, found = line_pair()
    assert len(found) == 1
    assert found.ell_med.iat[0] == 120  # positive: b follows a
    assert found.t_on.iat[0] == 0 and found.t_off.iat[0] == 2000


def test_altitude_and_heading_in_glide_break_the_contact():
    assert line_pair(dz=350.0)[2].empty
    assert line_pair(angle=60.0)[2].empty
    assert len(line_pair(angle=60.0, state=0)[2]) == 1  # in a climb the heading is not compared
    assert len(line_pair(angle=30.0)[2]) == 1


def test_a_delay_beyond_delta_is_no_contact():
    # At the largest delay j is still 600 m behind along the line: farther than eps_h.
    assert line_pair(delay=P["delta"] + 60)[2].empty
    assert len(line_pair(delay=P["delta"])[2]) == 1


def test_the_shifted_copy_loses_the_real_contact():
    table, grid, found = line_pair(delay=60)
    assert len(found) == 1
    tracks = contacts.Tracks(grid, 5)
    tracks.shifted(1, P["r_min"], "copy")
    span = (table.departure[0], table.landing[0])
    instants = grid.t[grid.f == 0].to_numpy()
    assert contacts.pair_intervals(tracks, 0, "copy", instants, span, P) == []


def test_runs_join_gaps_shorter_than_g():
    # Runs [0, 15), [120, 130), [400, 405): the first gap lasts 105 s, the second 270 s.
    t = np.r_[0, 5, 10, 120, 125, 400]
    on, off = contacts.runs_of(t, 5, 120)
    assert on.tolist() == [0, 400] and off.tolist() == [130, 405]
    assert contacts.runs_of(t, 5, 106)[0].tolist() == [0, 400]
    assert contacts.runs_of(t, 5, 105)[0].tolist() == [0, 120, 400]


# Phase 4.

def as_models(intervals):
    flights = {f: (0, 0, 10**6) for iv in intervals for f in iv[:2]}
    return flights, [((a, b), on, off) for a, b, on, off in intervals]


def compare_with_models(intervals):
    configs, links = sweep.configurations(intervals)
    episode = sweep.episodes(len(configs), links)
    flights, pairs = as_models(intervals)
    ref, ref_episode, ref_blocks = models.sweep(flights, pairs)
    assert sorted((sorted(m), s, e) for m, s, e in configs) == sorted((sorted(c.members), c.start, c.end) for c in ref)
    mine = {frozenset((m, s) for k, (m, s, _) in enumerate(configs) if episode[k] == e) for e in set(episode)}
    theirs = {frozenset((c.members, c.start) for c in ref if ref_episode[c] == e) for e in set(ref_episode.values())}
    assert mine == theirs
    n = max(flights) + 1
    blocks = sweep.blocks(n, configs, episode)
    in_groups = {frozenset(np.flatnonzero(blocks == b)) for b in set(blocks)}
    assert {frozenset(b) for b in ref_blocks} <= in_groups | {frozenset([f]) for f in range(n)}


def test_sweep_matches_the_example_day_of_the_note():
    flights, pairs = models.example_day()
    intervals = [(a, b, on, off) for (a, b), on, off in pairs]
    compare_with_models(intervals)
    configs, links = sweep.configurations(intervals)
    assert sorted((sorted(m), s) for m, s, _ in configs) == [([1, 8, 22], 700), ([8, 14, 22], 810), ([8, 22], 775),
                                                             ([8, 22], 900)]
    assert len(set(sweep.episodes(len(configs), links))) == 2


@pytest.mark.parametrize("seed", range(20))
def test_sweep_matches_the_reference_on_random_days(seed):
    rng = np.random.default_rng(seed)
    intervals = []
    for a in range(7):
        for b in range(a + 1, 7):
            t = int(rng.integers(0, 50))
            for _ in range(int(rng.integers(0, 3))):
                on = t + int(rng.integers(1, 40))
                off = on + int(rng.integers(1, 40))
                intervals.append((a, b, on, off))
                t = off + 1
    compare_with_models(intervals)


def test_membership_matches_the_labels_of_the_note():
    flights, pairs = models.example_day()
    order = sorted(flights)
    index = {f: k for k, f in enumerate(order)}
    intervals = [(index[a], index[b], on, off) for (a, b), on, off in pairs]
    table = pd.DataFrame([{"takeoff": flights[f][0], "departure": flights[f][1], "landing": flights[f][2]} for f in order])
    configs, _ = sweep.configurations(intervals)
    mb = sweep.membership(table, configs)
    ref, _, _ = models.sweep(*models.example_day())
    for f in order:
        for t in range(flights[f][0], flights[f][2], 5):
            row = mb[(mb.flight == index[f]) & (mb.t_start <= t) & (mb.t_end > t)]
            assert len(row) == 1
            label = models.label(flights, ref, f, t)
            if isinstance(label, str):
                assert row.state.iat[0] == label
            else:
                m, _, _ = configs[row.conf.iat[0]]
                assert {order[k] for k in m} == set(label.members)


def test_tau_star_is_where_the_false_discovery_rate_falls_to_q():
    # 100 h of real exposure, 10 h of surrogates with intervals of 100, 200 and 300 s: chance
    # gives 10 intervals per surrogate interval; 50 real intervals of 1000 s.
    real = pd.DataFrame({"duration": np.full(50, 1000), "d_dec": 0.0})
    null = [(100 * 3600.0, 10 * 3600.0, np.array([100, 200, 300]))]
    c = run.calibrate(real, null, P)
    k = np.searchsorted(run.TAUS, [100, 200, 300, 305])
    assert np.allclose(np.array(c["fdr"])[k], [30 / 50, 20 / 50, 10 / 50, 0])
    assert c["tau_star"] == 305


def test_episode_distance_is_the_straight_line_of_a_group_stretch():
    rows = [(0, 100, 200, "L", -1, -1), (0, 200, 500, "S", -1, -1), (0, 500, 900, "G", 3, 7),
            (0, 900, 1300, "G", 4, 7), (0, 1300, 1500, "S", -1, -1), (1, 100, 1500, "S", -1, -1)]
    membership = pd.DataFrame(rows, columns=["flight", "t_start", "t_end", "state", "conf_id", "episode_id"])
    flights = pd.DataFrame({"flight": [0, 1], "utc0": [100.0, 100.0]})
    s = distances.group_stretches(membership, flights)
    assert list(s.itertuples(index=False, name=None)) == [(0, 7, 400.0, 1200.0)]
    t = np.arange(0, 1401, 1.0)
    fixes = pd.DataFrame({"t": t, "E": 6.0 * t, "N": 8.0 * t})  # 10 m/s
    assert distances.stretch_distances(fixes, s) == pytest.approx([8000.0])


# Phase 0 on the disk.

@pytest.mark.skipif(not run.grid.DISK.exists(), reason="SSD_DISANTE is not mounted")
def test_phase_0_on_one_date():
    archive = run.grid.archive()
    flights = archive[archive.date == "2016-07-14"]
    runs = {d: run.grid.phase_runs(d) for d in run.grid.DERIVED}
    table, grid = run.grid.day(flights, runs, P["step"], P["h_max"], P["r_dep"])
    assert len(table) == len(flights)
    assert (grid.t % 5 == 0).all()
    radius = np.linalg.norm(grid[XYZ].to_numpy(), axis=1)
    assert radius.min() > 6.356e6 and radius.max() < 6.379e6  # on the ellipsoid
    assert np.allclose(np.hypot(grid.ve, grid.vn), 1, atol=1e-6)
    for f, g in grid.groupby("f"):
        flight = table.iloc[f]
        assert g.t.min() >= flight.departure and g.t.max() <= flight.landing
        x0 = table.loc[f, ["X0", "Y0", "Z0"]].to_numpy(float)
        assert np.linalg.norm(g[XYZ].to_numpy()[0] - x0) > P["r_dep"] - 100
    assert set(grid.state) <= {0, 1, 2, 3}
