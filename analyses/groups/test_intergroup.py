import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import intergroup as ig  # noqa: E402
import tamsd  # noqa: E402
import twoway  # noqa: E402
from common.msd import tamsd_sums  # noqa: E402
from isolated import sphere  # noqa: E402

BASE = 29_000_000  # a minute of the index


def frame(rows, members=()):
    """A day frame from rows (flight, east, north of the position at each minute): `rows` maps a
    flight to (first minute, east array, north array); `members` lists groups as sets of flights, in
    group from the start to the end."""
    f, m, x = [], [], []
    for flight, (first, east, north) in (rows.items() if isinstance(rows, dict) else rows):
        k = len(east)
        f.append(np.full(k, flight))
        m.append(first + np.arange(k))
        x.append(sphere(45.0, 6.0, east, north))
    f, m, x = np.concatenate(f), np.concatenate(m), np.concatenate(x)
    mem = pd.DataFrame([(fl, BASE * 60.0, (BASE + 1000) * 60.0, "G", g) for g, s in enumerate(members) for fl in sorted(s)],
                       columns=["flight", "t_start", "t_end", "state", "group_id"])
    return ig.day_frame(f, m, x.astype(np.float32), ig.membership_table(mem))


def line(first, speed=250.0, n=40, east0=0.0, north0=0.0):
    """A glider flying east at `speed` m per minute from (east0, north0)."""
    return first, east0 + speed * np.arange(n), np.full(n, north0)


def at_minute(fr, flight, minute):
    return int(np.flatnonzero((fr["flight"] == flight) & (fr["minute"] == minute))[0])


def test_direction_of_motion_and_its_minimum():
    fr = frame({1: line(BASE), 2: (BASE, np.zeros(40), np.zeros(40))})
    i = at_minute(fr, 1, BASE + 20)
    n = fr["xyz"][i] / np.linalg.norm(fr["xyz"][i])
    assert abs(fr["u_own"][i] @ n) < 1e-9 and abs(np.linalg.norm(fr["u_own"][i]) - 1) < 1e-9
    east = np.cross([0, 0, 1], n)
    assert fr["u_own"][i] @ (east / np.linalg.norm(east)) > 0.999  # flying east
    assert np.isnan(fr["u_own"][at_minute(fr, 2, BASE + 20)]).all()  # does not move: no direction
    assert np.isnan(fr["u_own"][at_minute(fr, 1, BASE + 2)]).all()  # no position 5 minutes before


def test_ahead_behind_gaps_and_outsiders():
    rows = {1: line(BASE), 2: line(BASE, east0=3000.0), 3: line(BASE, east0=-2000.0), 4: line(BASE, east0=1000.0, north0=500.0),
            5: line(BASE, east0=30_000.0)}
    fr = frame(rows, members=[{1, 4}, {2}, {3}])  # 4 is in the group of 1; 2 and 3 are alone in groups of one
    row = np.array([at_minute(fr, 1, BASE + 20)])
    g = ig.gaps(fr, row)
    assert g["defined"][0]
    assert abs(g["d_a"][0] - 3000) < 1 and abs(g["d_b"][0] - 2000) < 1  # 4 is a member, 5 too far
    assert abs(g["d_ag"][0] - 3000) < 1 and np.isinf(g["d_as"][0])  # 2 is in another group
    assert abs(g["xt"][0]) < 1
    off = frame({1: line(BASE), 2: line(BASE + 21, east0=5200.0)})  # the other glider has no position at that minute
    assert np.isinf(ig.gaps(off, np.array([at_minute(off, 1, BASE + 20)]))["d_a"][0])


def test_cross_track_and_none_without_direction():
    fr = frame({1: line(BASE), 2: line(BASE, east0=4000.0, north0=3000.0)})
    g = ig.gaps(fr, np.array([at_minute(fr, 1, BASE + 20)]))
    assert abs(g["d_a"][0] - 5000) < 1 and abs(g["xt"][0] - 3000) < 5
    still = frame({1: (BASE, np.zeros(40), np.zeros(40)), 2: line(BASE, east0=1000.0)})
    g = ig.gaps(still, np.array([at_minute(still, 1, BASE + 20)]))
    assert not g["defined"][0] and np.isinf(g["d_a"][0])


def test_group_direction_length_width_and_leader_gap():
    rows = {1: line(BASE, east0=0.0), 2: line(BASE, east0=-400.0, north0=100.0), 3: line(BASE, east0=-1000.0, north0=-100.0),
            4: line(BASE, east0=3500.0, north0=200.0)}
    fr = frame(rows, members=[{1, 2, 3}])
    g = ig.group_geometry(fr, np.array([True]), BASE + 10, BASE + 30)
    assert set(g["n"]) == {3}
    assert np.allclose(g["length"], 1000, atol=2) and np.allclose(g["width"], 200, atol=2)
    assert np.allclose(g["d_a"], np.hypot(3500, 200), atol=2)  # from the leader (flight 1)
    assert np.allclose(g["xt"], 200, atol=2) and np.isinf(g["d_as"]).sum() == 0


def test_group_direction_is_the_centroid_displacement():
    # a member that wanders sideways does not change the direction of a group that moves east
    fr = frame({1: line(BASE), 2: line(BASE, north0=200.0), 3: (BASE, np.zeros(40), 400.0 * np.arange(40.0))},
               members=[{1, 2, 3}])
    i = at_minute(fr, 3, BASE + 20)
    n = fr["xyz"][i] / np.linalg.norm(fr["xyz"][i])
    east = np.cross([0, 0, 1], n)
    assert fr["u_grp"][i] @ (east / np.linalg.norm(east)) > 0.5  # mostly east, though flight 3 flies north
    assert fr["u_own"][i] @ (east / np.linalg.norm(east)) < 0.1


def test_piece_minutes_summary_and_classes():
    utc0 = np.array([0.0, 600.0])
    piece, minute = ig.piece_minutes(utc0, np.array([0, 1]), np.array([30.0, 0.0]), np.array([150.0, 60.0]))
    assert piece.tolist() == [0, 0, 0, 1, 1] and minute.tolist() == [1, 2, 3, 10, 11]  # 30 s .. 180 s; 600 s .. 660 s
    d_a = np.array([1000.0, 1000.0, np.inf, 9000.0, np.inf])
    d_b = np.array([np.inf, 500.0, np.inf, np.inf, 100.0])
    s = ig.piece_summary(piece, 2, d_a, d_b, np.array([True, True, True, True, False]))
    assert np.allclose(s["f_a_2000"], [2 / 3, 0]) and np.allclose(s["f_b_2000"], [1 / 3, 0])
    assert np.allclose(s["f_a_10000"], [2 / 3, 1]) and s["defined"].tolist() == [1.0, 0.5]
    cls = ig.neighbourhood(np.array([0.0, 0.7, 0.05, 0.3, 0.05]), np.array([0.05, 0.9, 0.6, 0.3, 0.2]))
    assert cls.tolist() == ["ALONE", "AHEAD", "BEHIND", "MIXED", "MIXED"]


def test_hloc_weights_equal_the_local_slope():
    rng = np.random.default_rng(0)
    msd = np.exp(np.cumsum(rng.normal(1.8, 0.1, len(tamsd.LAGS))))
    assert np.isclose(np.log10(msd) @ ig.hloc_weights(), tamsd.local_h(tamsd.LAGS, msd)[-1])


def walk(rng, sigma, n=400, rho=0.98, speed=10.0):
    """A glider on the 10-s grid with a heading around east that wanders with noise sigma."""
    theta = np.zeros(n)
    for t in range(1, n):
        theta[t] = rho * theta[t - 1] + sigma * rng.normal()
    return np.cumsum(speed * 10 * np.column_stack([np.cos(theta), np.sin(theta)]), axis=0)


def test_toy_gliders_ahead_raise_persistence_and_gliders_behind_do_not():
    rng = np.random.default_rng(3)
    n, sigma = 150, {"ALONE": 0.12, "AHEAD": 0.03, "BEHIND": 0.12}  # the heading wanders less with gliders ahead
    segments, sums, origins, flight, start, truth = [], [], [], [], [], []
    outsider = 10_000
    for k in range(n):
        for j, cond in enumerate(sigma):
            pos = walk(rng, sigma[cond])
            s, o = tamsd_sums(pos, tamsd.LAGS // 10)
            sums.append(s), origins.append(o), flight.append(k), start.append((BASE + 100 * j) * 60.0), truth.append(cond)
            at = pos[::6] + np.array([300_000.0 * j, 30_000.0 * k])
            segments.append((k, (BASE + 100 * j, at[:, 0], at[:, 1])))
            if cond != "ALONE":  # a glider 2 km ahead of the pilot, or behind it, flying along
                segments.append((outsider, (BASE + 100 * j, at[:, 0] + (2000.0 if cond == "AHEAD" else -2000.0), at[:, 1])))
                outsider += 1
    fr = frame(segments)
    pc = {"flight": np.array(flight), "start": np.array(start), "duration": np.full(len(flight), 3990), "state": np.full(len(flight), "solo"),
          "sums": np.array(sums), "origins": np.array(origins), "use": np.ones(len(flight), bool)}
    piece, minute = ig.piece_minutes(np.zeros(n), pc["flight"], pc["start"], pc["duration"])
    key = fr["flight"] * ig.KEY + fr["minute"]
    rows = np.searchsorted(key, pc["flight"][piece] * ig.KEY + minute)
    g = ig.gaps(fr, rows)
    pc.update(ig.piece_summary(piece, len(flight), g["d_a"], g["d_b"], g["defined"]))
    cls = ig.neighbourhood(pc["f_a_5000"], pc["f_b_5000"])
    assert (cls == np.array(truth)).mean() > 0.95  # the geometry recovers the neighbourhood
    code = np.arange(n)
    counts = twoway.counts(np.arange(n), np.arange(n), np.random.default_rng(1), size=200)
    out = ig.hloc_cells(pc, cls, code, np.arange(n), counts)["contrasts"]
    inter, crowd, info = out["inter-group"], out["crowd control"], out["information, solo"]
    for design in ("raw", "flight"):
        assert inter[design]["band"][0] > 0.02  # gliders ahead raise H_loc(1000 s): band above 0
        assert crowd[design]["band"][0] < 0 < crowd[design]["band"][1]  # gliders behind do not
        assert info[design]["band"][0] > 0.02
        assert inter[design]["value"] > 5 * abs(crowd[design]["value"])
