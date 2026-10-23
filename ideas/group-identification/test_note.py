import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

_spec = importlib.util.spec_from_file_location("group_models", Path(__file__).with_name("models.py"))
models = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(models)

# Starting values of the parameter table in the note; V_MAX is speed.max_kmh of xc-dynamics.
EPS_H, EPS_Z, DELTA, STEP, V_MAX, W_MAX = 500.0, 300.0, 300.0, 5.0, 100 / 3.6, 5.0
H_MAX, R_EARTH = 10.0, 6.371e6


# Pair contact (figure "contact").

def test_pair_has_a_delayed_contact_and_no_simultaneous_one():
    p = models.PAIR
    t, xi, xj = models.pair_tracks()
    M, dist = models.correspondences(t, xi, xj, p["delta"], p["eps_h"])
    assert np.diagonal(dist).min() > 900
    lags = (t[None, :] - t[:, None])[M]
    assert (lags > 0).all()
    best = [t[np.flatnonzero(M[k])][np.argmin(dist[k, M[k]])] - t[k] for k in range(len(t)) if M[k].any()]
    assert np.median(best) == pytest.approx(p["delay"], abs=p["step"])
    assert models.runs(M.any(axis=1), t, p["step"]) == [(90.0, 775.0)]
    assert models.runs(M.any(axis=0), t, p["step"]) == [(220.0, 885.0)]


def test_m_ij_and_m_ji_differ_by_at_most_delta_in_time():
    p = models.WAIT
    t, xi, xj = models.wait_tracks()
    M, _ = models.correspondences(t, xi, xj, p["delta"], p["eps_h"])
    m_ij, m_ji = M.any(axis=1), M.any(axis=0)
    assert models.runs(m_ij, t, p["step"]) == [(331.0, 601.0)]
    assert models.runs(m_ji, t, p["step"]) == [(451.0, 550.0)]
    assert {s: (m_ij[s], m_ji[s]) for s in (300, 380, 500, 560)} == {
        300: (False, False), 380: (True, False), 500: (True, True), 560: (True, False)}
    # the farthest instant of one from the other: m_ij reaches delta, m_ji lies inside m_ij
    far = [max(np.abs(t[v] - s).min() for s in t[u]) for u, v in ((m_ij, m_ji), (m_ji, m_ij))]
    assert far == [p["delta"], 0.0]
    q = models.FOLLOW
    t, xi, xj = models.follow_tracks()
    M, _ = models.correspondences(t, xi, xj, q["delta"], q["eps_h"])
    assert models.runs(M.any(axis=1), t, q["step"]) == [(0.0, 601.0)]
    assert models.runs(M.any(axis=0), t, q["step"]) == [(q["delay"], 601.0 + q["delay"])]  # the same run, delayed


# Altitude as a clock (figure "altitude_clock").

def test_altitude_selects_one_window_of_delays():
    p = models.THERMAL
    s = models.thermal_scan()
    lo, hi = (p["dz"] - p["eps_z"]) / p["climb"], (p["dz"] + p["eps_z"]) / p["climb"]
    assert (lo, hi) == (-50.0, 250.0)
    assert np.array_equal(s.lags[s.both], s.lags[(s.lags > lo) & (s.lags < hi)])


def test_horizontal_distance_alone_is_degenerate_in_the_thermal():
    s = models.thermal_scan()
    inside = s.lags >= models.THERMAL["t_in"]
    d = s.horizontal[inside]
    assert d.max() <= 2 * models.THERMAL["radius"] + 1e-9
    minima = (d[1:-1] < d[:-2]) & (d[1:-1] <= d[2:]) & (d[1:-1] < 5)
    assert minima.sum() == 21
    assert (s.ok_z & ~s.ok_h).any()


@pytest.mark.parametrize("dz, matched", [(880.0, True), (900.0, False)])
def test_same_thermal_matches_up_to_eps_z_plus_w_delta(dz, matched):
    s = models.thermal_scan({**models.THERMAL, "dz": dz})
    assert EPS_Z + 2.0 * DELTA == 900.0
    assert s.both.any() == matched


# Groups (figure "example_day").

def test_example_day_configurations_episodes_and_blocks():
    flights, pairs = models.example_day()
    configs, episode, blocks = models.sweep(flights, pairs)
    m = models.minutes
    expected = [({1, 8, 22}, "11:40", "12:55", 1), ({8, 22}, "12:55", "13:30", 1),
                ({8, 14, 22}, "13:30", "14:10", 1), ({8, 22}, "15:00", "15:40", 2)]
    got = [(set(c.members), c.start, c.end, episode[c]) for c in configs]
    assert got == [(s, m(a), m(b), e) for s, a, b, e in expected]
    assert blocks == [[1, 8, 14, 22], [5]]
    assert not any({1, 14} <= c.members for c in configs)


def test_each_flight_is_in_at_most_one_configuration_at_a_time():
    flights, pairs = models.example_day()
    configs, _, _ = models.sweep(flights, pairs)
    for t in range(models.minutes("11:00"), models.minutes("16:00")):
        for f in flights:
            assert sum(f in c.members and c.start <= t < c.end for c in configs) <= 1


# Numbers quoted in the note.

def test_filter_radius_and_delay_grid():
    assert EPS_H + V_MAX * (DELTA + 60) == pytest.approx(10500.0)
    L = math.floor(DELTA / STEP)
    assert 2 * L + 1 == 121


def test_delay_bound_and_resolution():
    assert 1500.0 / 5.0 == DELTA
    assert EPS_Z / 2.0 == 150.0 and EPS_Z / 4.0 == 75.0


def test_grid_errors():
    assert round(V_MAX * STEP / 2) == 69
    assert round(V_MAX * H_MAX / 2) == 139
    assert 8.0 * STEP / 2 == 20.0
    assert 2.0 * STEP / 2 == 5.0 and W_MAX * STEP / 2 == 12.5


def test_chord_is_shorter_than_the_distance_along_the_earth_by_about_a_millimetre_at_10_km():
    d = 10e3
    chord = 2 * R_EARTH * math.sin(d / (2 * R_EARTH))
    assert d - chord == pytest.approx(d**3 / (24 * R_EARTH**2), rel=1e-3)
    assert d - chord == pytest.approx(1e-3, abs=0.1e-3)


def test_east_north_directions_of_two_points_500_m_apart_differ_by_less_than_a_hundredth_of_a_degree():
    def basis(lat, lon):
        p, l = math.radians(lat), math.radians(lon)
        return (np.array([-math.sin(l), math.cos(l), 0.0]),
                np.array([-math.sin(p) * math.cos(l), -math.sin(p) * math.sin(l), math.cos(p)]))

    worst = 0.0
    for lat in (42.0, 46.0, 51.0):
        e0, n0 = basis(lat, 6.0)
        dlon = math.degrees(500 / (R_EARTH * math.cos(math.radians(lat))))
        dlat = math.degrees(500 / R_EARTH)
        for e1, n1 in (basis(lat, 6.0 + dlon), basis(lat + dlat, 6.0)):
            # The heading east at the second point, seen in the east-north plane of the first.
            angle = math.degrees(math.atan2(e1 @ n0, e1 @ e0))
            worst = max(worst, abs(angle))
    assert worst < 0.01


def test_float32_resolves_earth_centred_positions_to_half_a_metre():
    assert np.spacing(np.float32(R_EARTH)) == 0.5


def test_dataset_size_estimates():
    flights, hours = 1.86e5, 1.5
    rows = flights * hours * 3600 / STEP
    assert rows == pytest.approx(2.0e8, rel=0.01)
    assert 4 + 4 + 3 * 4 + 4 + 2 * 2 + 1 == 29
    assert rows * 30 / 1e9 == pytest.approx(6.0, rel=0.01)
    assert flights * hours * 60 == pytest.approx(1.7e7, rel=0.02)
    assert flights * hours * 3600 == pytest.approx(1.0e9, rel=0.01)


def test_cost_of_the_literal_computation():
    pairs, instants, delays = 200 * 199 // 2, 1.5 * 3600 / STEP, 2 * (120 // STEP) + 1
    assert (pairs, instants, delays) == (19900, 1080, 49)
    assert pairs * instants * delays == pytest.approx(1e9, rel=0.06)


# The KD-tree figure (figure "kdcells"): eight cells of two points, two reached by the circle.

def test_kd_tree_figure():
    pts = np.array([(0.7, 0.6), (1.4, 1.5), (2.8, 0.4), (3.3, 1.3), (0.8, 2.9), (1.1, 3.5), (2.4, 2.6),
                    (3.4, 3.4), (4.6, 0.7), (5.4, 1.4), (7.0, 0.5), (7.5, 1.2), (4.9, 3.2), (5.9, 2.3),
                    (7.1, 2.9), (7.6, 3.6)])
    cells = [(0, 2, 0, 2), (2, 4, 0, 2), (0, 1.6, 2, 4), (1.6, 4, 2, 4),
             (4, 6, 0, 1.8), (6, 8, 0, 1.8), (4, 6.5, 1.8, 4), (6.5, 8, 1.8, 4)]
    inside = [(pts[:, 0] >= x0) & (pts[:, 0] < x1) & (pts[:, 1] >= y0) & (pts[:, 1] < y1) for x0, x1, y0, y1 in cells]
    assert [int(m.sum()) for m in inside] == [2] * 8
    assert (pts[:, 0] < 4).sum() == 8
    q, r = (5.2, 2.6), 1.0
    reached = [c for c in cells if math.hypot(max(c[0] - q[0], 0, q[0] - c[1]), max(c[2] - q[1], 0, q[1] - c[3])) < r]
    assert reached == [(4, 6, 0, 1.8), (4, 6.5, 1.8, 4)]
