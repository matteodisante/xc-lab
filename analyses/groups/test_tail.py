import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import tail  # noqa: E402

sys.path.insert(0, str(Path(__file__).parents[1] / "phases"))
from durations import hill  # noqa: E402


def pareto(mu, n, xmin, seed):
    return xmin * np.random.default_rng(seed).random(n) ** (-1 / mu)


def test_hill_recovers_mu_on_pareto_samples():
    n, xmin = 20000, 300.0
    est = [hill(pareto(3, n, xmin, s), xmin) for s in range(10)]
    se = 3 / np.sqrt(n)  # the standard error of the Hill estimate
    assert abs(np.mean(est) - 3) < 3 * se / np.sqrt(10) + 1e-3
    assert all(abs(e - 3) < 4 * se for e in est)


def test_weighted_hill_with_integer_weights_is_the_hill_of_repeated_samples():
    x = pareto(3, 500, 300.0, 1)
    w = np.random.default_rng(2).integers(1, 5, len(x))
    for xmin in (300.0, 450.0, 1000.0):
        assert np.isclose(tail.weighted_hill(x, w, xmin), hill(np.repeat(x, w), xmin))
    assert np.isclose(tail.weighted_hill(x, np.ones(len(x)), 500.0), hill(x, 500.0))


def test_mu_curve_value_is_the_hill_estimate_and_the_band_holds_it():
    n, cells = 6000, 300
    x = pareto(3, n, 300.0, 3)
    cell = np.random.default_rng(4).integers(0, cells, n)
    counts = np.random.default_rng(5).multinomial(cells, np.full(cells, 1 / cells), size=3 * 200).astype(np.uint8)
    xmins = [300.0, 600.0, 1200.0]
    c = tail.mu_curve(x, cell, counts, xmins)
    for m, v, r in zip(xmins, c["value"], c["runs"]):
        assert np.isclose(v, hill(x, m)) and r == (x >= m).sum()
    lo, hi = np.array(c["band"])
    assert np.all(lo < c["value"]) and np.all(c["value"] < hi) and np.all(hi - lo < 1)


def test_descriptors_of_a_straight_glide_at_constant_speed_and_sink():
    t = np.arange(0.0, 401.0)
    e, n, z = 10.0 * t, np.zeros_like(t), 2000.0 - 1.0 * t
    d = tail.glide_descriptors(t, e, n, z, np.array([100]), np.array([400]))  # 300 s inside the track
    assert np.isclose(d["path (km)"][0], 3.0) and np.isclose(d["net displacement (km)"][0], 3.0)
    assert np.isclose(d["straightness"][0], 1.0)
    assert np.isclose(d["ground speed (km/h)"][0], 36.0)
    assert np.isclose(d["vertical speed (m/s)"][0], -1.0)
    assert np.isclose(d["height lost (m/km)"][0], 100.0)
    assert np.isclose(d["time rising (share)"][0], 0.0)


def test_descriptors_of_a_detour_and_a_rise():
    t = np.arange(0.0, 201.0)
    e = np.where(t <= 100, 10.0 * t, 1000.0 - 10.0 * (t - 100))  # out and back: net 0
    n, z = np.zeros_like(t), np.where(t <= 100, 0.0, 50.0)
    d = tail.glide_descriptors(t, e, n, z, np.array([0]), np.array([100]))
    assert np.isclose(d["time rising (share)"][0], 0.0)
    d = tail.glide_descriptors(t, e, n, z + 0.5 * t, np.array([0]), np.array([200]))
    assert np.isclose(d["straightness"][0], 0.0) and np.isclose(d["path (km)"][0], 2.0)
    assert d["time rising (share)"][0] > 0.99


def test_flight_glides_finds_the_runs_by_their_end_fixes():
    import pandas as pd
    t = np.arange(0.0, 301.0)
    fixes = pd.DataFrame({"segment_id": 0, "t": t, "E": 10.0 * t, "N": 0.0, "z": 1000.0 - t})
    runs = pd.DataFrame({"segment_id": [0, 0], "t_start": [0.0, 50.5], "t_end": [100.0, 200.0]}, index=[7, 9])
    d = tail.flight_glides(fixes, runs)
    assert list(d.index) == [7] and np.isclose(d["ground speed (km/h)"][7], 36.0)


def test_median_draws_is_the_median_of_the_repeated_runs():
    rng = np.random.default_rng(6)
    v, cell = rng.normal(size=301), rng.integers(0, 40, 301)
    counts = rng.integers(0, 4, size=(30, 40)).astype(np.uint8)
    got = tail.median_draws(v, cell, counts)
    for d, row in enumerate(counts):
        rep = np.repeat(v, row[cell])
        assert got[d] == np.sort(rep)[(len(rep) + 1) // 2 - 1]  # the lower median
    assert np.all(tail.median_draws(v, cell, np.ones((3, 40), np.uint8)) == np.sort(v)[150])
