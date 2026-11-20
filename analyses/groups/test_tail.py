import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import tail  # noqa: E402
import tail_ks  # noqa: E402
import tail_designs as td  # noqa: E402

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


def test_estimate_is_the_hill_estimate_with_its_iid_error_and_a_band():
    n, cells = 6000, 300
    x = pareto(3, n, 300.0, 11)
    cell = np.random.default_rng(12).integers(0, cells, n)
    counts = np.random.default_rng(13).multinomial(cells, np.full(cells, 1 / cells), size=3 * 200).astype(np.uint8)
    e, draws = td.estimate(x, cell, counts, 600.0)
    above = (x >= 600.0).sum()
    assert np.isclose(e["value"], hill(x, 600.0)) and e["n"] == above and len(draws) == 600
    assert np.isclose(e["se_iid"], e["value"] / np.sqrt(above))
    assert e["band"][0] < e["value"] < e["band"][1]
    assert np.isnan(td.estimate(x[:0], cell[:0], counts, 600.0)[0]["value"])


def test_combine_adds_values_draws_and_iid_errors_in_quadrature():
    rng = np.random.default_rng(14)
    a = ({"value": 4.0, "band": [3, 5], "n": 10, "se_iid": 0.3}, rng.normal(4, 0.2, 90))
    b = ({"value": 3.0, "band": [2, 4], "n": 10, "se_iid": 0.4}, rng.normal(3, 0.2, 90))
    e, d = td.combine([(1, a), (-0.5, b)])
    assert np.isclose(e["value"], 2.5) and np.allclose(d, a[1] - 0.5 * b[1])
    assert np.isclose(e["se_iid"], np.sqrt(0.3 ** 2 + 0.25 * 0.4 ** 2))


def test_ks_distance_is_small_from_the_start_of_the_power_law():
    rng = np.random.default_rng(15)
    x = np.r_[rng.uniform(50, 300, 4000), pareto(3, 6000, 300.0, 16)]
    k, best = td.ks_curve(x, np.array([150.0, 200.0, 300.0, 450.0]))
    d = dict(zip(k["xmin"], k["D"]))
    assert d[300.0] < 0.03 and d[150.0] > 5 * d[300.0] and d[200.0] > 3 * d[300.0]
    assert np.isclose(k["mu"][2], hill(x, 300.0)) and best in (300.0, 450.0)
    assert np.isnan(td.ks_curve(x[:10], np.array([300.0]))[0]["D"][0])


def test_weighted_median_is_the_lower_median_of_the_repeated_values():
    rng = np.random.default_rng(17)
    v, w = rng.normal(size=51), rng.integers(0, 4, size=(51, 6)).astype(float)
    got = td.weighted_median(v, w)
    for j in range(6):
        rep = np.sort(np.repeat(v, w[:, j].astype(int)))
        assert got[j] == rep[(len(rep) + 1) // 2 - 1]
    assert np.isnan(td.weighted_median(v, np.zeros((51, 2)))).all() and np.isnan(td.weighted_median(v[:0], np.zeros((0, 2)))).all()


def stretch_frames():
    """Flight 1: solo, group, solo, group, solo (state of the membership rows; times on a clock that starts at 0).
    Flight 2: solo only."""
    rows = [(1, "S", 0, 1000), (1, "G", 1000, 3000), (1, "S", 3000, 4000), (1, "G", 4000, 6000), (1, "S", 6000, 9000), (2, "S", 0, 5000)]
    membership = pd.DataFrame(rows, columns=["flight", "state", "t_start", "t_end"]).assign(episode_id=1, group_id=1, n=2)
    flights = pd.DataFrame({"flight": [1, 2], "dt_native": 1.0, "n_grid": 10, "utc0": 0.0})
    summary = pd.DataFrame({"flight": [1, 2], "n_entries": [2, 0]})
    return flights, membership, summary


def test_group_stretches_are_classed_by_their_place_in_the_flight():
    s = td.group_stretches(*stretch_frames())
    assert list(s["flight"]) == [1] * 5
    assert list(s["cls"]) == ["before", "in group", "between", "in group", "after"]
    assert list(s["length"]) == [1000, 2000, 1000, 2000, 3000]


def test_classify_puts_a_run_where_it_starts_and_ends():
    s = td.group_stretches(*stretch_frames())
    runs = pd.DataFrame({"flight": [1, 1, 1, 1, 1, 1, 1, 1, 2],
                         "t_start": [100, 1200, 3100, 4100, 6500, 900, 2900, 8900, 100],
                         "t_end": [900, 2800, 3900, 5900, 8000, 1100, 3100, 9500, 900]})
    c = td.classify(s, runs)
    assert list(c["cls"]) == ["before", "in group", "between", "in group", "after", "spanning", "spanning", "spanning", "outside"]
    assert list(c["start"][:8]) == ["before", "in group", "between", "in group", "after", "before", "in group", "after"]
    assert list(c["length"][:5]) == [1000, 2000, 1000, 2000, 3000] and np.isnan(c["length"][8])


def test_classify_forgives_a_small_overshoot_only_with_a_tolerance():
    s = td.group_stretches(*stretch_frames())
    runs = pd.DataFrame({"flight": [1, 1], "t_start": [995, 1200], "t_end": [2000, 3005]})
    assert list(td.classify(s, runs)["cls"]) == ["spanning", "spanning"]
    assert list(td.classify(s, runs, 10)["cls"]) == ["in group", "in group"]


def test_length_table_counts_the_stretches_long_enough():
    s = td.group_stretches(*stretch_frames())
    t = td.length_table(s, 1000.0)
    assert t["in group"]["stretches"] == 2 and t["in group"]["share >= xmin"] == 1.0
    assert t["in group"]["stretches >= 2 xmin"] == 2 and t["before"]["stretches >= 2 xmin"] == 0
    assert t["after"]["length quantiles (s)"]["50"] == 3000


def test_describe_gives_the_quartiles_and_a_median_band_of_the_long_glides():
    rows = pd.DataFrame({name: np.arange(1.0, 6.0) for name in td.DESCRIPTORS}).assign(last=[True, False, False, False, False])
    cell, counts = np.arange(5), np.random.default_rng(18).multinomial(5, np.full(5, 0.2), size=3 * 40).astype(np.uint8)
    out, med = td.describe(rows, cell, counts)
    assert out["runs"] == 5 and out["last run of the flight (share)"] == 0.2
    assert out["path (km)"]["median"] == 3.0 and out["path (km)"]["q25"] == 2.0 and med["path (km)"][1].shape == (120,)
    w = np.ones((5, 121))
    out_w, _ = td.describe(rows, None, None, w)
    assert out_w["path (km)"]["median"] == 3.0


def test_route_weights_give_the_sums_of_the_engine():
    from types import SimpleNamespace

    import isolated
    import twoway
    rng = np.random.default_rng(19)
    universe = np.arange(20)
    cell = rng.integers(0, 8, 20)
    stacked = rng.multinomial(8, np.full(8, 1 / 8), size=3 * 30).astype(np.uint8)
    ctx = SimpleNamespace(tracks={"flight": universe}, row=lambda f: np.searchsorted(universe, f), boot=twoway.weights(cell, stacked))
    sel = pd.DataFrame({"piece": [0, 0, 1, 2, 2, 2, 3, 3], "solo": [10, 11, 12, 10, 13, 14, 15, 16]})
    a, b = rng.random((4, 2)), rng.random((len(sel), 2))
    pieces, p = np.unique(sel["piece"], return_inverse=True)
    flights, s = np.unique(sel["solo"], return_inverse=True)
    group, control, _ = isolated.engine(p, s, a, b, ctx.row(pieces), ctx.row(flights), ctx.boot)
    wp, wm = td.route_weights(ctx, sel, np.arange(ctx.boot[1]))
    assert np.allclose(group, a.T @ wp) and np.allclose(control, b.T @ wm)
    assert np.allclose(wp[:, 0], 1) and np.allclose(wm[:, 0], 1 / np.array([2, 2, 1, 3, 3, 3, 2, 2]))


def spliced(n, xmin, mu, seed):
    """Whole seconds: a log-normal body below xmin and a power law (mu) from xmin up."""
    body = np.round(np.random.default_rng(seed).lognormal(4, 0.9, n))
    return np.r_[body[(body >= 1) & (body < xmin)], np.round(pareto(mu, n // 10, xmin, seed + 1))]


def test_ks_scan_gives_the_hill_exponent_at_its_threshold_and_finds_the_tail():
    x = spliced(60000, 300.0, 1.0, 1)
    r = tail_ks.thresholds({"a": (x, np.zeros(len(x), int))}, np.ones((1, 1)), 1)
    assert np.isclose(r["mu"]["a"][0], hill(x, r["xmin"][0])) and r["n"]["a"][0] == (x >= r["xmin"][0]).sum()
    assert 200 < r["xmin"][0] < 1200 and abs(r["mu"]["a"][0] - 1.0) < 0.1


def test_ks_scan_agrees_with_powerlaw_fit_on_the_exponent():
    import powerlaw

    x = spliced(30000, 150.0, 0.8, 2)
    f = powerlaw.Fit(x, verbose=0)
    r = tail_ks.thresholds({"a": (x, np.zeros(len(x), int))}, np.ones((1, 1)), 1)
    assert abs(r["mu"]["a"][0] - (f.power_law.alpha - 1)) < 0.05


def test_cell_weights_are_the_repetitions_of_the_runs():
    x, cell = spliced(20000, 300.0, 1.0, 3), None
    cell = np.random.default_rng(4).integers(0, 50, len(x))
    w = np.random.default_rng(5).integers(0, 4, (1, 50))
    edges = tail_ks.edges_of(x)
    cand = tail_ks.candidates(x, edges)
    again = np.repeat(x, w[0][cell])
    a = tail_ks.suffixes(w, tail_ks.histograms(x, cell, 50, edges))
    b = tail_ks.suffixes(np.ones((1, 1)), tail_ks.histograms(again, np.zeros(len(again), int), 1, edges))
    assert np.allclose(a[0], b[0]) and np.allclose(a[1], b[1])
    assert (tail_ks.best(*a, edges, cand) == tail_ks.best(*b, edges, cand)).all()


def test_common_threshold_is_the_largest_of_the_own_and_the_exponents_are_read_there():
    a, b = spliced(40000, 200.0, 1.0, 6), spliced(40000, 500.0, 1.0, 7)
    rng = np.random.default_rng(8)
    s = {"a": (a, rng.integers(0, 80, len(a))), "b": (b, rng.integers(0, 80, len(b)))}
    r = tail_ks.thresholds(s, rng.multinomial(80, np.full(80, 1 / 80), size=60), 80)
    assert (r["xmin"] == np.maximum(r["own"]["a"], r["own"]["b"])).all()
    assert np.isfinite(r["mu"]["a"]).all() and np.isfinite(r["mu"]["b"]).all()


def test_contextual_classifies_the_searches_by_their_neighbours():
    sys.argv += ["--declared-open", "--dt5"]
    import phase_terms as pt

    rows = [("transition", 0, 99), ("search", 100, 150), ("climb", 151, 300), ("search", 301, 330), ("climb", 331, 500),
            ("search", 501, 520), ("transition", 521, 600), ("search", 601, 650), ("transition", 651, 700), ("search", 905, 950)]
    r = pd.DataFrame(rows, columns=["phase", "t_start", "t_end"]).assign(fid=1, flight=1, seg=0, dt=1.0)
    r["phase"] = pd.Categorical(r["phase"], categories=["climb", "search", "transition"])
    r["s"] = r["t_end"] - r["t_start"] + 1
    out = pt.contextual(r)
    assert list(out["phase"]) == ["transition", "found", "climb", "exit", "transition", "failed", "transition", "unknown"]
    assert (out.loc[2, "t_start"], out.loc[2, "t_end"], out.loc[2, "recentres"]) == (151, 500, 1)
    assert out["s"].sum() == r["s"].sum()
