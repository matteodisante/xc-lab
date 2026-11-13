import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

# Imported by name, not from their files: the worker processes re-import them.
sys.path.insert(0, str(Path(__file__).parent))
import age  # noqa: E402
import blocks  # noqa: E402
import breathing  # noqa: E402
import budget  # noqa: E402
import decompose  # noqa: E402
import decomposition  # noqa: E402
import describe  # noqa: E402
import encounters  # noqa: E402
import exits  # noqa: E402
import glide  # noqa: E402
import headings  # noqa: E402
import longlags  # noqa: E402
import marking  # noqa: E402
import matched  # noqa: E402
import mechanisms  # noqa: E402
import network  # noqa: E402
import openflights  # noqa: E402
import partners  # noqa: E402
import isolated  # noqa: E402
import prepost  # noqa: E402
import performance  # noqa: E402
import phase_msd  # noqa: E402
import redundancy  # noqa: E402
import rotation  # noqa: E402
import size  # noqa: E402
import raw  # noqa: E402
import route  # noqa: E402
import survival  # noqa: E402
import tamsd  # noqa: E402
import twoway  # noqa: E402
import zones as zonelib  # noqa: E402
from common.msd import tamsd as lab_tamsd  # noqa: E402


def test_fft_msd_is_the_direct_average():
    pos = np.cumsum(np.random.default_rng(0).normal(size=(400, 2)), axis=0) + [3e5, -2e5]
    lags = np.array([1, 2, 7, 50, 399])
    direct = [((pos[d:] - pos[:-d]) ** 2).sum(axis=1).mean() for d in lags]
    assert lab_tamsd(pos, lags) == pytest.approx(direct, rel=1e-9)


def labels():
    """Three flights: 0 solo; 1 in a group in the middle of its flight; 2 local."""
    flights = pd.DataFrame({"flight": [0, 1, 2], "utc0": [1000.0, 2000.0, 3000.0], "dt_native": [1.0, 1.0, 1.0],
                            "n_grid": [10, 10, 0]})
    rows = [(0, 1000, 1100, "L", -1, -1), (0, 1100, 9000, "S", -1, -1),
            (1, 2000, 2100, "L", -1, -1), (1, 2100, 3000, "S", -1, -1), (1, 3000, 4000, "G", 5, 7),
            (1, 4000, 6000, "G", 6, 7), (1, 6000, 9000, "S", -1, -1), (2, 3000, 9000, "L", -1, -1)]
    membership = pd.DataFrame(rows, columns=["flight", "t_start", "t_end", "state", "group_id", "episode_id"])
    summary = pd.DataFrame({"flight": [0, 1, 2], "n_entries": [0, 1, 0], "block": [0, 1, 2]})
    return flights, membership, summary


def test_stretches_split_the_populations_on_the_flight_clock():
    s = tamsd.stretches(*labels())
    assert list(s.itertuples(index=False, name=None)) == [
        (0, "solo", -1, 100.0, 8000.0),
        (1, "group_solo", -1, 100.0, 1000.0),
        (1, "group", 7, 1000.0, 4000.0),  # two groups of one episode are one stretch
        (1, "group_solo", -1, 4000.0, 7000.0),
    ]


def test_stretches_by_size_end_where_the_group_changes():
    flights, membership, summary = labels()
    membership["n"] = [1, 1, 1, 1, 2, 5, 1, 1]  # flight 1: 2 pilots, then 5
    s = tamsd.stretches(flights, membership, summary, by_size=True)
    assert list(s.itertuples(index=False, name=None))[1:] == [
        (1, "group_solo", -1, 100.0, 1000.0),
        (1, "n2", 7, 1000.0, 2000.0),  # the two groups of one episode are two stretches
        (1, "n5-6", 7, 2000.0, 4000.0),
        (1, "group_solo", -1, 4000.0, 7000.0),
    ]
    assert list(tamsd.size_of(np.array([2, 3, 4, 5, 6, 7, 40]))) == ["n2", "n3", "n4", "n5-6", "n5-6", "n7+", "n7+"]


def test_select_adds_the_controls_and_drops_the_thin_populations():
    flight = np.r_[np.arange(40), np.arange(100, 120), np.arange(100, 115), np.arange(100, 106)]
    population = np.array(["solo"] * 40 + ["group_solo"] * 20 + ["n2"] * 15 + ["n3"] * 6)
    block = pd.Series(np.arange(200), index=np.arange(200))
    rows, labels = size.select(population, flight, block)
    assert sorted(set(labels)) == ["n2", "n2/alone", "solo"]  # n3 has 6 flights, the rest of group_solo goes
    assert sorted(flight[rows[labels == "n2/alone"]].tolist()) == list(range(100, 115))  # the n2 flights' solo time
    rows, labels = size.select(population, flight, block, flights=np.r_[np.arange(30), np.arange(100, 112)])
    assert sorted(set(labels)) == ["solo"]  # n2 has 12 flights there
    tied = pd.Series(np.arange(200) // 4, index=np.arange(200))  # four flights to a block: n2 has 4 blocks
    assert sorted(set(size.select(population, flight, tied)[1])) == ["solo"]


def test_choose_sitedays_by_sizes_with_enough_flights_at_most_two_per_takeoff_site():
    spec = {  # site-day: takeoff name, date, flights per size
        0: ("A", "d0", {"n2": 20, "n3": 20}), 1: ("A", "d1", {"n2": 20, "n3": 20, "n4": 20}),
        2: ("A", "d2", {"n2": 25, "n3": 25, "n4": 20}), 3: ("B", "d3", {"n2": 20, "n3": 20}),
        4: ("C", "d4", {"n2": 20})}
    flight, sized, ids, day, name, date = 0, [], [], [], [], []
    for s, (n, d, sizes) in spec.items():
        for key, count in sizes.items():
            for _ in range(count):
                sized.append(key), ids.append(flight), day.append(s), name.append(n), date.append(d)
                flight += 1
    flights = pd.DataFrame({"flight": ids, "date": date, "site_name": name}).drop_duplicates("flight")
    site = pd.Series(day, index=ids)
    pieces = {"flight": np.array(ids), "population": np.array(sized)}
    chosen = size.choose_sitedays(pieces, flights, site)
    assert [label for label, _ in chosen] == ["A, d2", "A, d1", "B, d3"]  # not A, d0: two of A are there; not C
    assert [len(f) for _, f in chosen] == [70, 60, 40]
    assert [label for label, _ in size.choose_sitedays(pieces, flights, site, ["A:d0", "C:d4"])] == ["A, d0", "C, d4"]
    with pytest.raises(SystemExit):
        size.choose_sitedays(pieces, flights, site, ["A:d9"])


def test_decompose_separates_the_group_effect_from_the_day():
    # Two strata (days): the good day raises the statistic by 10, being in a group by 2; group
    # flights fly mostly on the good day, so the pooled difference mixes the two.
    rows = []  # (population, stratum, value)
    for stratum, day, n_solo, n_group in ((0, 0.0, 90, 10), (1, 10.0, 10, 90)):
        rows += [("solo", stratum, day)] * n_solo + [("group_solo", stratum, day)] * n_group
        rows += [("group", stratum, day + 2.0)] * n_group
    pop, stratum, value = (np.array(c) for c in zip(*rows))
    units = decompose.Units(pop, stratum, np.arange(len(pop)), value[:, None], np.ones((len(pop), 1)))
    none = np.zeros((0, len(pop)), np.int32)
    raw = decompose.Units(pop, np.zeros(len(pop), int), units.block, units.values, units.den)
    assert decompose.effect(raw, "group", "solo", none)[0, 0] == pytest.approx(10.0)  # 2 + the day: 9.0 - (-1.0)
    assert decompose.effect(units, "group", "solo", none)[0, 0] == pytest.approx(2.0)
    assert decompose.effect(units, "group", "group_solo", none)[0, 0] == pytest.approx(2.0)
    assert decompose.effect(units, "group_solo", "solo", none)[0, 0] == pytest.approx(0.0)
    counts = np.ones((3, len(pop)), np.int32)  # a draw that takes every block once repeats the estimate
    assert decompose.effect(units, "group", "solo", counts)[:, 0] == pytest.approx([2.0] * 4)


def test_decompose_msd_units_average_log_msd_over_strata():
    # One stratum, two flights per population; G has twice the MSD of S at every lag: Delta log = log 2, Delta H = 0.
    p = {"flight": np.array([1, 2, 3, 4]), "population": np.array(["group", "group", "solo", "solo"]),
         "start": np.zeros(4), "duration": np.full(4, 2000.0),
         "sums": np.vstack([2 * tamsd.LAGS.astype(float) ** 1.8] * 2 + [tamsd.LAGS.astype(float) ** 1.8] * 2),
         "origins": np.ones((4, len(tamsd.LAGS)))}
    units = decompose.msd_units(p, None, None, pd.Series(np.arange(5), index=np.arange(5)))
    values = decompose.effect(units, "group", "solo", np.zeros((0, 4), np.int32))
    assert values[0] == pytest.approx(np.log(2.0) * np.ones(len(tamsd.LAGS)))
    stats = decompose.msd_statistics(values)
    assert stats["Delta H"][0] == pytest.approx(0.0, abs=1e-12) and stats["H_loc(1000 s)"][0] == pytest.approx(0.0, abs=1e-12)


def test_entries_exits_and_the_phase_at_them():
    spans = pd.DataFrame({"flight": [1, 1, 1, 2, 2, 3, 3],
                          "population": ["group_solo", "group", "group_solo", "group", "group_solo", "group_solo", "group"],
                          "a": [0, 100, 200, 0, 50, 0, 10], "b": [100, 200, 300, 50, 100, 10, 90]})
    ev = mechanisms.events(spans).sort_values(["flight", "t"])
    assert list(ev.itertuples(index=False, name=None)) == [
        (1, "entry", 100), (1, "exit", 200), (2, "at departure", 0), (2, "exit", 50), (3, "entry", 10),
        (3, "at landing", 90)]
    runs = pd.DataFrame({"flight": [1, 1], "t_start": [90.0, 151.0], "t_end": [150.0, 260.0], "phase": ["climb", "transition"]})
    points = pd.DataFrame({"flight": [1, 1, 2, 1], "t": [100.0, 200.0, 0.0, 150.5]})
    assert list(mechanisms.phase_at(points, runs)) == ["climb", "transition", "none", "none"]


def test_roles_of_the_two_pilots_from_the_signed_delay():
    groups = pd.DataFrame({"group_id": [0, 1], "episode_id": [0, 0], "t_start": [1000, 3000], "t_end": [2000, 4000],
                           "n": [2, 2], "members": [[5, 9], [5, 9]]})
    pairs = pd.DataFrame({"a": [5, 5], "b": [9, 9], "t_on": [900, 2900], "t_off": [2100, 4100], "ell_med": [30.0, 5.0]})
    flights = pd.DataFrame({"flight": [5, 9], "utc0": [0.0, 500.0]})
    r = mechanisms.roles(groups, pairs, flights).sort_values(["group_id", "flight"])
    assert list(r[["group_id", "flight", "population"]].itertuples(index=False, name=None)) == [
        (0, 5, "leader"), (0, 9, "follower"), (1, 5, "abreast"), (1, 9, "abreast")]  # 9 follows 5 by 30 s
    assert r.loc[r["flight"] == 9, "a"].tolist() == [500.0, 2500.0]  # on the clock of each flight


def test_hours_sample_the_stretches_in_local_time():
    utc0 = pd.Timestamp("2022-06-01 10:00", tz="UTC").timestamp()  # 12:00 in Paris (CEST)
    flights = pd.DataFrame({"flight": [1, 2, 3], "utc0": [utc0] * 3})
    spans = pd.DataFrame({"flight": [1, 2, 3], "population": ["group", "group_solo", "solo"], "a": [0.0, 3600.0, 0.0],
                          "b": [3600.0, 7200.0, 600.0]})
    h = mechanisms.hours(spans, flights)
    assert h["group"][12] == 1 and h["group_solo"][13] == 1 and h["solo"][12] == 1
    assert h["fraction in a group"][12] == 1 and h["fraction in a group"][13] == 0


def test_zone_cells_and_the_standardisation_of_solo_flights_to_the_zones_of_the_group():
    takeoff = pd.DataFrame({"lat0": [46.5, 46.5], "lon0": [2.5, 2.5]}, index=[1, 2])
    p = {"flight": np.array([1, 2, 2]), "mid_e": np.array([0.0, 10_000.0, 60_000.0]), "mid_n": np.array([0.0, 0.0, 0.0])}
    zone = zonelib.zone_of(p, takeoff)
    assert zone[0] == zone[1] and zone[2] != zone[0]  # 10 km east: the same 25-km cell; 60 km: another
    # Two zones. Solo flights: 9 in zone 0 (MSD 1), 1 in zone 1 (MSD 5). Group: 1 in zone 0, 9 in zone 1 (MSD 5).
    pop = np.array(["solo"] * 10 + ["group"] * 10)
    stratum = np.array([0] * 9 + [1] + [0] + [1] * 9)
    value = np.array([1.0] * 9 + [5.0] + [1.0] + [5.0] * 9)[:, None] * np.ones((1, 3))
    units = decompose.Units(pop, stratum, np.arange(20), value)
    mg, ms, own, share = zonelib.standardized(units, "group", "solo", np.zeros((0, 20), np.int32))
    assert mg[0] == pytest.approx([4.6] * 3) and share[0] == 1.0  # 0.1 * 1 + 0.9 * 5
    assert ms[0] == pytest.approx([4.6] * 3)  # the solo flights, reweighted to the zones of the group, give the same
    assert own[0] == pytest.approx([1.4] * 3)  # without reweighting they keep their own mix: 0.9 * 1 + 0.1 * 5
    pooled = decompose.Units(pop, np.zeros(20, int), np.arange(20), value)
    mg, ms, _, _ = zonelib.standardized(pooled, "group", "solo", np.zeros((0, 20), np.int32))
    assert mg[0, 0] == pytest.approx(4.6) and ms[0, 0] == pytest.approx(1.4)  # pooled, they differ: the zones
    lost = decompose.Units(pop, np.where(pop == "group", stratum + 2, stratum), np.arange(20), value)
    assert zonelib.standardized(lost, "group", "solo", np.zeros((0, 20), np.int32))[3][0] == 0.0  # no common zone


def test_windows_around_the_group_and_the_control_pairs():
    W = prepost.WINDOW_S
    spans = pd.DataFrame({
        "flight": [1, 1, 1, 2, 2, 3, 4],
        "population": ["group_solo", "group", "group_solo", "group", "group_solo", "solo", "group_solo"],
        "episode": [-1, 5, -1, 6, -1, -1, -1],
        "a": [0.0, 3000.0, 3000.0 + 2 * W, 0.0, 600.0, 0.0, 0.0],
        "b": [3000.0, 3000.0 + 2 * W, 5000.0 + 2 * W, 600.0, 600.0 + W, 6000.0, 900.0]})
    w = prepost.windows(spans, np.random.default_rng(0))
    ev = w[~w["population"].str.startswith("ctrl")].sort_values(["flight", "a"])
    assert list(ev[["flight", "population", "a", "b"]].itertuples(index=False, name=None)) == [
        (1, "pre", 3000.0 - W, 3000.0), (1, "start", 3000.0, 3000.0 + W), (1, "end", 3000.0 + W, 3000.0 + 2 * W),
        (1, "post", 3000.0 + 2 * W, 3000.0 + 3 * W),
        (2, "post", 600.0, 600.0 + W)]  # a group of 600 s is too short for start and end, its solo stretch after is long enough
    ctrl = w[w["population"].str.startswith("ctrl")]
    assert sorted(ctrl["population"]) == ["ctrlA1", "ctrlA2", "ctrlS1", "ctrlS2"]  # the solo time of flight 1 before the group; flight 3
    one = ctrl[ctrl["population"] == "ctrlS1"].iloc[0]
    two = ctrl[ctrl["population"] == "ctrlS2"].iloc[0]
    assert two["a"] == one["b"] and two["episode"] == one["episode"] and one["b"] - one["a"] == W
    terms = pd.DataFrame({"flight": [1, 1, 1], "population": "solo", "t_mid": [2000.0, 3100.0, 99999.0], "gain_m": 1.0})
    placed = prepost.place(terms, w)
    assert sorted(placed["population"]) == ["ctrlA2", "pre", "start"]  # a run can lie in a window and in a control window


def test_prepost_comparison_and_profile_on_a_synthetic_group_effect():
    # 40 events; the group windows have twice the MSD of the windows alone, at every lag.
    rows = []
    for e in range(40):
        for label, factor in (("pre", 1.0), ("start", 2.0), ("end", 2.0), ("post", 1.0)):
            rows.append((label, e, factor))
    pop, stratum, factor = (np.array(c) for c in zip(*rows))
    values = factor[:, None] * tamsd.LAGS[None, :].astype(float) ** 1.8
    units = decompose.Units(pop, stratum, np.arange(len(pop)), values)
    draws = np.ones((3, len(pop)), np.int32)
    out, _ = prepost.compare(units, draws, {"entry": ("start", "pre")})
    assert out["entry"]["events"] == 40
    assert out["entry"]["dH"] == pytest.approx(0.0, abs=1e-9)  # a constant factor does not change H
    prof = prepost.profile(units, draws)
    assert prof["events"] == 40 and prof["pre"]["H"] == pytest.approx(0.9, abs=1e-9)
    # a stretch of straighter motion in the group windows: H rises from pre to start
    values = values * np.where(np.isin(pop, ("start", "end"))[:, None], tamsd.LAGS[None, :].astype(float) ** 0.1, 1.0)
    out, _ = prepost.compare(decompose.Units(pop, stratum, np.arange(len(pop)), values), draws,
                             {"entry": ("start", "pre"), "exit": ("end", "post")})
    assert out["entry"]["dH"] == pytest.approx(0.05, abs=1e-9) and out["exit"]["dH"] == pytest.approx(0.05, abs=1e-9)


def straight(t_end, segments=((0, None),)):
    t = np.arange(0, t_end + 1, 1.0)
    seg = np.zeros(len(t), int)
    for s, start in segments:
        if start is not None:
            seg[t >= start] = s
    return pd.DataFrame({"segment_id": seg, "t": t, "E": 8.0 * t, "N": 0.0})


def test_pieces_of_a_straight_glide_have_msd_v2_tau2():
    spans = pd.DataFrame({"population": ["group"], "episode": [3], "a": [0.0], "b": [3000.0]})
    found = tamsd.pieces(straight(5000), spans)
    assert len(found) == 1
    pop, episode, start, duration, mid_e, mid_n, sums, origins = found[0]
    assert (pop, episode, start, duration) == ("group", 3, 0, 2990)
    assert (mid_e, mid_n) == (8.0 * 1500, 0.0)  # the grid point at the middle of the piece
    assert sums / origins == pytest.approx(64.0 * tamsd.LAGS.astype(float) ** 2, rel=1e-9)


def test_pieces_shorter_than_tau_max_over_support_or_across_segments_are_dropped():
    spans = pd.DataFrame({"population": ["solo"], "episode": [-1], "a": [0.0], "b": [2000.0]})
    assert len(tamsd.pieces(straight(5000), spans)) == 1
    # A segment boundary at 1000 s leaves two pieces shorter than 1250 s.
    assert tamsd.pieces(straight(5000, ((0, None), (1, 1000))), spans) == []
    short = spans.assign(b=1250.0)
    assert tamsd.pieces(straight(5000), short) == []
    assert len(tamsd.pieces(straight(5000), short.assign(b=1260.0))) == 1


def test_h_of_power_laws():
    lags = tamsd.LAGS.astype(float)
    assert tamsd.fit_h(lags, 3 * lags**1.4) == pytest.approx(0.7)
    broken = np.where(lags < 100, lags**2, 1e2 * lags)  # ballistic, then diffusive
    hl = tamsd.local_h(lags, broken)
    assert hl[lags <= 30] == pytest.approx(1.0)
    assert hl[lags >= 400] == pytest.approx(0.5)


def test_ensemble_weights_flights_equally_and_keeps_blocks_together():
    values = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    mean, draws = tamsd.ensemble(values, np.array([0, 0, 0]), np.random.default_rng(0))
    assert mean.tolist() == [3.0, 4.0]
    assert np.allclose(draws, [3.0, 4.0])  # one block: every draw is the whole sample
    _, draws = tamsd.ensemble(values, np.array([0, 1, 2]), np.random.default_rng(0))
    assert draws.shape == (tamsd.N_BOOT, 2) and draws[:, 0].std() > 0


def test_runs_are_cut_to_the_stretches_of_their_flight():
    runs = pd.DataFrame({"flight": [1, 1], "segment_id": [0, 0], "phase": ["climb", "transition"],
                         "t_start": [500.0, 900.0], "t_end": [899.0, 1500.0]})
    spans = pd.DataFrame({"flight": [1, 1], "population": ["group_solo", "group"], "episode": [-1, 7],
                          "a": [100.0, 1000.0], "b": [1000.0, 4000.0]})
    got = phase_msd.cut(runs, spans)
    assert list(got[["phase", "population", "t_start", "t_end"]].itertuples(index=False, name=None)) == [
        ("climb", "group_solo", 500.0, 899.0),
        ("transition", "group", 1000.0, 1500.0),
    ]  # the 100 fixes of the transition before 1000 s are too short


# The README numbers: flights, H with its band, H_loc at 100 s and 1000 s.
TAMSD = {
    ("para", "solo"): (119129, 0.899, [0.898, 0.899], 0.919, 0.837),
    ("para", "group"): (33435, 0.916, [0.915, 0.917], 0.927, 0.893),
    ("para", "group_solo"): (29611, 0.914, [0.913, 0.915], 0.932, 0.860),
    ("hang", "solo"): (5555, 0.881, [0.880, 0.882], 0.905, 0.817),
    ("hang", "group"): (418, 0.899, [0.894, 0.903], 0.914, 0.860),
    ("hang", "group_solo"): (373, 0.889, [0.885, 0.893], 0.912, 0.823),
}
# date, discipline, flights, largest group, duration (h), members in the MSD, H, band
EPISODES = [("2013-04-24", "para", 53, 16, 5.8, 49, 0.926, [0.920, 0.931]),
            ("2013-04-24", "hang", 53, 16, 5.8, 3, 0.861, [0.821, 0.890]),
            ("2015-04-11", "para", 52, 19, 5.1, 52, 0.912, [0.907, 0.917]),
            ("2016-03-19", "para", 49, 22, 5.2, 49, 0.879, [0.868, 0.888]),
            ("2022-03-26", "para", 47, 16, 3.8, 47, 0.883, [0.869, 0.896]),
            ("2018-04-21", "para", 47, 19, 6.5, 47, 0.923, [0.916, 0.929])]


@pytest.mark.skipif(not (tamsd.OUT / "tamsd.json").exists(), reason="run tamsd.py first")
def test_tamsd_numbers_of_the_readme():
    results = json.loads((tamsd.OUT / "tamsd.json").read_text())
    for (d, pop), (flights, h, band, at_100, at_1000) in TAMSD.items():
        r = results[d][pop]
        lags = np.array(r["lags"])
        assert r["flights"] == flights
        assert round(r["H"], 3) == h and np.round(r["H_band"], 3).tolist() == band
        assert [round(r["H_loc"][np.searchsorted(lags, x)], 3) for x in (100, 1000)] == [at_100, at_1000]


@pytest.mark.skipif(not (tamsd.OUT / "episodes.json").exists(), reason="run episodes.py first")
def test_episode_numbers_of_the_readme():
    curves = json.loads((tamsd.OUT / "episodes.json").read_text())
    got = [(c["date"], c["discipline"], c["n_flights"], c["n_max"], round(c["duration_s"] / 3600, 1), c["members"],
            round(c["H"], 3), np.round(c["H_band"], 3).tolist()) for c in curves]
    assert got == EPISODES


# The README numbers of the phases: flights with transitions and searches, slopes and bands.
PHASES = {
    ("para", "solo"): ((80186, 29243), (1.962, [1.961, 1.962]), (0.619, [0.585, 0.668])),
    ("para", "group"): ((23944, 5084), (1.977, [1.976, 1.978]), (0.606, [0.537, 0.685])),
    ("para", "group_solo"): ((22843, 9003), (1.977, [1.976, 1.978]), (0.715, [0.647, 0.787])),
    ("hang", "solo"): ((1535, 663), (1.963, [1.960, 1.966]), None),
    ("hang", "group"): ((121, 11), (1.977, [1.966, 1.990]), None),
    ("hang", "group_solo"): ((114, 44), (1.960, [1.952, 1.970]), None),
}


@pytest.mark.skipif(not (tamsd.OUT / "phase_slopes.json").exists(), reason="run phase_msd.py first")
def test_phase_numbers_of_the_readme():
    medians = json.loads((tamsd.OUT / "phase_msd.json").read_text())
    slopes = json.loads((tamsd.OUT / "phase_slopes.json").read_text())
    for (d, pop), (flights, transition, search) in PHASES.items():
        assert (medians[d][pop]["transition"]["flights"], medians[d][pop]["search"]["flights"]) == flights
        fitted = {phase: (round(slopes[d]["slope"][pop][phase], 3), np.round(slopes[d]["band"][pop][phase], 3).tolist())
                  for phase in ("transition", "search") if slopes[d]["band"][pop][phase] is not None}
        assert fitted["transition"] == transition
        if search is not None:
            assert fitted["search"] == search
    assert slopes["hang"]["slope"]["group"]["search"] is None  # no lag of the fit has 100 searches


def test_slopes_of_a_power_law():
    d = phase_msd.DELTAS[phase_msd.DELTAS <= phase_msd.MAX_PLOT].astype(float)
    assert phase_msd.slope(d, 2 * d**1.9) == pytest.approx(1.9)
    assert phase_msd.slope(d, np.where(d < 100, d**2, 100 * d**1.0), *phase_msd.SEARCH_FIT) == pytest.approx(1.0)


def test_takeoffs_within_3_km_on_one_day_are_one_site_and_blocks_merge():
    km = 1 / 111.2  # degrees of latitude
    lat = np.array([45.0, 45.0 + 2 * km, 45.0 + 4 * km, 45.0, 45.0 + 10 * km])
    days = np.array(["d1", "d1", "d1", "d2", "d1"])
    s = blocks.sites(lat, np.full(5, 6.0), days)
    assert s[0] == s[1] == s[2]  # 2 km apart, chained
    assert len({s[0], s[3], s[4]}) == 3  # another day; 6 km from the nearest
    b = blocks.merged(np.array([0, 0, 1, 2]), np.array([5, 6, 6, 7]))
    assert b[0] == b[1] == b[2] and b[3] != b[0]


def test_survival_counts_searches_before_a_climb_and_keeps_runs_whole():
    runs = pd.DataFrame({"flight_id": "a", "segment_id": 0, "phase": ["search", "climb", "search", "transition", "search"],
                         "t_start": [0.0, 10, 20, 30, 40]})
    assert survival.counted(runs)["phase"].tolist() == ["search", "climb", "transition"]
    runs = pd.DataFrame({"flight": 1, "phase": "climb", "t_start": [50.0, 500, 900], "t_end": [90.0, 899, 1500]})
    spans = pd.DataFrame({"flight": [1, 1], "population": ["group_solo", "group"], "episode": [-1, 7],
                          "a": [100.0, 1000.0], "b": [1000.0, 4000.0]})
    got = survival.assign(runs, spans)
    assert list(got[["population", "t_start", "t_end"]].itertuples(index=False, name=None)) == [
        ("group_solo", 500.0, 899.0),
        ("group", 900.0, 1500.0),  # its middle is in the group; its start is not cut
    ]  # the first run lies before every stretch


@pytest.mark.skipif(not (tamsd.OUT / "blocks.json").exists(), reason="run blocks.py first")
def test_block_numbers_of_the_readme():
    r = json.loads((tamsd.OUT / "blocks.json").read_text())
    assert [r[v]["blocks"] for v in ("episodes", "day and site", "day")] == [139684, 64128, 6209]
    widest = 0.0
    for v in ("day and site", "day"):
        for d in ("para", "hang"):
            for pop, x in r[v][d].items():
                e = r["episodes"][d][pop]
                for a, b in [(x["H_band"], e["H_band"]), *zip(x["H_loc_band"], e["H_loc_band"], strict=True)]:
                    widest = max(widest, (a[1] - a[0]) - (b[1] - b[0]))
    assert round(widest, 3) == 0.002
    assert np.round(r["day"]["para"]["solo"]["H_loc_band"][1], 3).tolist() == [0.836, 0.839]
    assert np.round(r["episodes"]["para"]["solo"]["H_loc_band"][1], 3).tolist() == [0.837, 0.838]


# phase, population: runs, flights, median (s), mu above the solo xmin and its band
SURVIVAL = {
    ("transition", "solo"): (1388316, 80241, 194, 4.54, [4.43, 4.65]),
    ("transition", "group"): (269867, 24097, 196, 3.52, [3.36, 3.72]),
    ("transition", "group_solo"): (392750, 23208, 213, 4.37, [4.18, 4.58]),
    ("search", "solo"): (420780, 72277, 51, 4.02, [3.88, 4.16]),
    ("search", "group"): (87614, 20681, 53, 4.10, [3.81, 4.45]),
    ("search", "group_solo"): (122354, 20163, 53, 4.04, [3.83, 4.30]),
    ("climb", "solo"): (1258867, 79545, 130, 8.85, [8.48, 9.26]),
    ("climb", "group"): (268561, 24034, 132, 9.54, [8.51, 10.76]),
    ("climb", "group_solo"): (362061, 22415, 130, 8.83, [8.02, 9.83]),
}


@pytest.mark.skipif(not (tamsd.OUT / "survival.json").exists(), reason="run survival.py first")
def test_survival_numbers_of_the_readme():
    fits = json.loads((tamsd.OUT / "survival.json").read_text())
    assert [round(fits["solo"][p]["xmin_common"]) for p in phase_msd.PHASES] == [1857, 495, 995]
    for (phase, pop), (runs, flights, median, mu, band) in SURVIVAL.items():
        f = fits[pop][phase]
        assert (f["n"], f["flights"], round(f["median"])) == (runs, flights, median)
        assert round(f["mu_common"], 2) == mu and np.round(f["mu_common_band"], 2).tolist() == band


def test_budget_terms_of_each_run():
    runs = pd.DataFrame({"flight_id": "a", "segment_id": 0, "t_start": [0.0, 10, 20, 30, 40],
                         "phase": ["search", "climb", "search", "transition", "climb"],
                         "n_fixes": [10, 100, 20, 300, 50]})
    c = budget.columns(runs)
    assert c["found_s"].tolist() == [10, 0, 0, 0, 0] and c["vain_s"].tolist() == [0, 0, 20, 0, 0]
    assert c["after_search"].tolist() == [0, 1, 0, 0, 0]  # the last climb follows a transition
    r = budget.ratios(c[["found", "searches", "climbs", "hours", "after_search", "s", "transition_s", "climb_s",
                         "found_s", "vain_s"]].sum())
    assert r["searches that end in a climb (%)"] == 50 and r["climbs that start after a search (%)"] == 50
    assert r["climbs per hour"] == pytest.approx(2 / (480 / 3600))


# The README table: value and band, per statistic and population (solo, group, group_solo).
BUDGET = {
    "time in transition (%)": [(58.4, [58.4, 58.5]), (57.6, [57.3, 57.9]), (60.8, [60.7, 61.0])],
    "time in climb (%)": [(30.8, [30.7, 30.9]), (33.0, [32.8, 33.2]), (29.1, [28.9, 29.2])],
    "time in a search that ends in a climb (%)": [(4.4, [4.4, 4.4]), (4.7, [4.6, 4.8]), (4.4, [4.3, 4.5])],
    "time in a search in vain (%)": [(6.3, [6.3, 6.4]), (4.7, [4.6, 4.8]), (5.7, [5.6, 5.8])],
    "climbs per hour": [(6.51, [6.50, 6.52]), (7.03, [6.99, 7.06]), (6.21, [6.18, 6.23])],
    "climbs that start after a search (%)": [(33.5, [33.4, 33.6]), (32.7, [32.3, 33.1]), (33.8, [33.5, 34.0])],
    "searches that end in a climb (%)": [(43.3, [43.2, 43.4]), (49.5, [49.2, 49.7]), (45.9, [45.7, 46.1])],
}


@pytest.mark.skipif(not (tamsd.OUT / "budget.json").exists(), reason="run budget.py first")
def test_budget_numbers_of_the_readme():
    r = json.loads((tamsd.OUT / "budget.json").read_text())
    assert [r[p]["flights"] for p in tamsd.POPULATIONS] == [80278, 24099, 23469]
    for name, rows in BUDGET.items():
        digits = 2 if name == "climbs per hour" else 1
        got = [(round(r[p][name]["value"], digits), np.round(r[p][name]["band"], digits).tolist()) for p in tamsd.POPULATIONS]
        assert got == rows, name


def test_elimination_drops_a_share_that_the_others_sum_to_and_a_copy():
    p1, p2, p3, q = np.random.default_rng(1).normal(size=(4, 400))
    units = pd.DataFrame({"p1": p1, "p2": p2, "p3": p3, "p4": -p1 - p2 - p3, "q": q})
    kept, _, dropped = redundancy.eliminate(units)  # the shares sum to a constant: the first goes,
    assert dropped[0] == ("p1", 1.0)  # then p4, which p2 and p3 still explain by two thirds
    assert kept == ["p2", "p3", "q"] and dropped[1][0] == "p4" and round(dropped[1][1], 1) == 0.7
    units["copy"] = q + 0.01 * np.random.default_rng(2).normal(size=400)
    kept, z, dropped = redundancy.eliminate(units)
    assert dropped[1][0] in ("q", "copy") and len({"q", "copy"} & set(kept)) == 1
    assert redundancy.r2(z, "p2", ["p3"]) < 0.01  # independent of another share


@pytest.mark.skipif(not (tamsd.OUT / "redundancy.json").exists(), reason="run redundancy.py first")
def test_redundancy_numbers_of_the_readme():
    r = json.loads((tamsd.OUT / "redundancy.json").read_text())
    assert r["units"] == 119 and r["populations"] == {"group": 55, "group_solo": 53, "solo": 11}
    assert sorted(r["kept"]) == sorted(budget.PLOTTED + performance.PLOTTED)
    assert [(d["statistic"], round(d["r2_at_the_time"], 2)) for d in r["dropped"]] == [
        ("time in transition (%)", 1.0), ("height gained per climb (m)", 0.98),
        ("climbs that start after a search (%)", 0.93), ("cross-country speed over 30 min (km/h)", 0.93),
        ("time in a search in vain (%)", 0.88), ("climbs per hour", 0.72),
        ("time in a search that ends in a climb (%)", 0.65)]
    assert {c: round(v, 2) for c, v in r["r2_of_the_dropped_on_the_kept"].items()} == {
        "time in transition (%)": 0.83, "time in a search that ends in a climb (%)": 0.65,
        "time in a search in vain (%)": 0.71, "climbs per hour": 0.71,
        "climbs that start after a search (%)": 0.77, "height gained per climb (m)": 0.91,
        "cross-country speed over 30 min (km/h)": 0.90}
    assert max(r["r2_of_the_kept_on_the_kept"].values()) < redundancy.LIMIT


SIZE_KEYS = ["solo", "n2", "n3", "n4", "n5-6", "n7+"]
# The README tables of size.py: per size, flights, H, H_loc(1000 s), H_loc(1000 s) of the same flights alone.
SIZE_TAMSD = {"solo": (119129, 0.899, 0.837, None), "n2": (26411, 0.917, 0.889, 0.861),
              "n3": (9480, 0.923, 0.906, 0.867), "n4": (4073, 0.924, 0.910, 0.865),
              "n5-6": (2686, 0.925, 0.915, 0.870), "n7+": (1142, 0.933, 0.944, 0.871)}
SIZE_RATIOS = {  # time in climb, searches that end in a climb | climb rate, glide speed, glide ratio
    "budget": {"solo": (30.8, 43.3), "n2": (32.7, 49.0), "n3": (33.6, 49.9), "n4": (33.6, 51.0),
               "n5-6": (33.1, 50.1), "n7+": (34.6, 51.7)},
    "performance": {"solo": (1.22, 30.1, 10.4), "n2": (1.27, 33.2, 12.5), "n3": (1.26, 33.5, 13.1),
                    "n4": (1.25, 33.5, 13.7), "n5-6": (1.23, 33.1, 14.1), "n7+": (1.20, 34.6, 15.9)}}
SIZE_ALONE = {"budget": [(27.8, 29.1), (45.8, 46.1)], "performance": [(1.33, 1.35), (32.5, 33.1), (11.3, 11.6)]}
SITEDAY_HLOC = {  # H_loc(1000 s) by size on each site-day
    "Saint Hilaire sud, 2022-03-26": {"solo": 0.750, "n2": 0.853, "n3": 0.894, "n4": 0.818, "n7+": 0.994},
    "Saint Hilaire sud, 2017-03-28": {"solo": 0.854, "n2": 0.931, "n3": 0.901, "n4": 0.956, "n7+": 0.873},
    "Méruz, 2019-05-01": {"n2": 0.926, "n3": 0.939, "n5-6": 0.938},
    "Méruz, 2016-04-28": {"n2": 0.894, "n3": 0.958, "n5-6": 0.957},
    "Chalvet Ouest, 2016-07-19": {"n2": 0.951, "n5-6": 0.892},
    "La Sambuy, 2020-07-25": {"n2": 0.941, "n3": 0.901}}


@pytest.mark.skipif(not (tamsd.OUT / "size_survival.json").exists(), reason="run size.py first")
def test_size_numbers_of_the_readme():
    def at_1000(r, key):
        return r[key][int(np.searchsorted(np.array(r["lags"]), 1000))]

    t = json.loads((tamsd.OUT / "size_tamsd.json").read_text())
    for size, (flights, h, h_loc, alone) in SIZE_TAMSD.items():
        r = t["para"][size]
        assert (r["flights"], round(r["H"], 3), round(at_1000(r, "H_loc"), 3)) == (flights, h, h_loc)
        if alone:
            assert round(at_1000(t["para"][f"{size}/alone"], "H_loc"), 3) == alone
    assert [round(x, 3) for x in (t["para"]["n7+"]["H_band"] + t["para"]["n7+"]["H_loc_band"][0][-1:]
            + t["para"]["n7+"]["H_loc_band"][1][-1:])] == [0.926, 0.940, 0.933, 0.953]
    hang = {s: (r["flights"], round(at_1000(r, "H_loc"), 3)) for s, r in t["hang"].items() if "/" not in s}
    assert hang == {"solo": (5555, 0.817), "n2": (385, 0.857), "n3": (60, 0.856)}
    for name, stats in (("budget", budget.PLOTTED), ("performance", performance.PLOTTED)):
        r = json.loads((tamsd.OUT / f"size_{name}.json").read_text())
        for size, values in SIZE_RATIOS[name].items():
            assert tuple(round(r[size][s]["value"], 1 if name == "budget" or s != "climb rate (m/s)" else 2)
                         for s in stats) == values, (name, size)
        for s, (low, high) in zip(stats, SIZE_ALONE[name]):
            digits = 2 if s == "climb rate (m/s)" else 1
            v = [r[f"{k}/alone"][s]["value"] for k in SIZE_KEYS[1:]]
            assert (round(min(v), digits), round(max(v), digits)) == (low, high), s
    p = json.loads((tamsd.OUT / "size_phase_msd.json").read_text())
    last = {k: phase_msd.local_exponent(*phase_msd.drawn(p[k]["transition"]))[-1] for k in SIZE_KEYS}
    assert round(last["solo"], 2) == 1.85 and phase_msd.drawn(p["solo"]["transition"])[0][-1] == 357
    assert all(1.90 <= round(last[k], 2) <= 1.95 for k in SIZE_KEYS[1:])
    s = json.loads((tamsd.OUT / "size_survival.json").read_text())
    got = {k: (round(s[k]["transition"]["mu_common"], 2), np.round(s[k]["transition"]["mu_common_band"], 2).tolist(),
               s[k]["transition"]["n_common"]) for k in SIZE_KEYS if s[k]["transition"]["n_common"] >= 50}
    assert got == {"solo": (4.54, [4.43, 4.65], 5111), "n2": (3.64, [3.44, 3.88], 795), "n3": (3.2, [2.85, 3.67], 206),
                   "n4": (3.38, [2.91, 3.92], 90), "n5-6": (3.81, [3.31, 4.61], 84)}
    assert s["n7+"]["transition"]["n_common"] == 30
    searches = {k: (round(s[k]["search"]["mu_common"], 2), np.round(s[k]["search"]["mu_common_band"], 2).tolist())
                for k in SIZE_KEYS if s[k]["search"]["n_common"] >= 50}
    assert searches == {"solo": (4.02, [3.88, 4.16]), "n2": (4.15, [3.77, 4.59]), "n3": (4.04, [3.4, 4.9])}
    climbs = {k: (round(s[k]["climb"]["mu_common"], 2), np.round(s[k]["climb"]["mu_common_band"], 2).tolist())
              for k in SIZE_KEYS if s[k]["climb"]["n_common"] >= 50}
    assert climbs == {"solo": (8.85, [8.5, 9.23]), "n2": (9.23, [8.03, 10.72])}


@pytest.mark.skipif(not (tamsd.OUT / "siteday_tamsd.json").exists(), reason="run size.py first")
def test_siteday_numbers_of_the_readme():
    days = json.loads((tamsd.OUT / "siteday_tamsd.json").read_text())
    assert {d: {s: round(r[s]["H_loc"][-1], 3) for s in r} for d, r in days.items()} == SITEDAY_HLOC
    chosen = json.loads((tamsd.OUT / "siteday_chosen.json").read_text())
    assert [(c["site_day"], c["flights"]) for c in chosen] == [
        ("Saint Hilaire sud, 2022-03-26", 108), ("Saint Hilaire sud, 2017-03-28", 64), ("Méruz, 2019-05-01", 73),
        ("Méruz, 2016-04-28", 44), ("Chalvet Ouest, 2016-07-19", 61), ("La Sambuy, 2020-07-25", 54)]
    rate = [r[p]["climb rate (m/s)"]["value"]
            for r in json.loads((tamsd.OUT / "siteday_performance.json").read_text()).values() for p in r]
    assert len(rate) == 26 and (round(min(rate), 2), round(max(rate), 2)) == (0.79, 2.11)
    kept = json.loads((tamsd.OUT / "redundancy.json").read_text())["kept_by_population"]
    assert kept["group"] == ["time in climb (%)", "time in a search in vain (%)", "searches that end in a climb (%)",
                             "glide speed (km/h)", "glide ratio"]
    assert kept["group_solo"] == ["time in climb (%)", "climbs per hour", "searches that end in a climb (%)",
                                  "glide speed (km/h)"]


@pytest.mark.skipif(not (tamsd.OUT / "zones.json").exists(), reason="run zones.py first")
def test_zone_numbers_of_the_report():
    z = json.loads((tamsd.OUT / "zones.json").read_text())
    k = int(np.searchsorted(np.array(z["none"]["S"]["lags"]), 1000))
    expected = {  # kept %, dH reweighted, dH as is, dH_loc(1000) reweighted, dH_loc(1000) as is, dH G - A
        "none": (100, 0.0172, 0.0172, 0.0550, 0.0550, 0.0021),
        "zone": (100, 0.0176, 0.0207, 0.0555, 0.0593, 0.0031),
        "zone x month": (99, 0.0167, 0.0217, 0.0544, 0.0637, 0.0035),
        "zone x hour": (99, 0.0166, 0.0219, 0.0520, 0.0639, 0.0053),
        "site-day x hour": (53, 0.0132, 0.0149, 0.0437, 0.0510, 0.0080)}
    for name, (kept, dh, dh_as_is, loc, loc_as_is, dh_a) in expected.items():
        r = z[name]
        assert round(100 * r["S"]["share of G kept"]) == kept, name
        assert (round(r["S"]["dH"], 4), round(r["S"]["dH_support"], 4)) == (dh, dh_as_is), name
        assert (round(r["S"]["dH_loc"][k], 4), round(r["S"]["dH_loc_support"][k], 4)) == (loc, loc_as_is), name
        assert round(r["A"]["dH"], 4) == dh_a, name
    assert [round(z["none"]["S"][c]["H"], 3) for c in ("G", "other")] == [0.916, 0.899]


def test_prepost_numbers_of_the_report():
    r = json.loads((tamsd.OUT / "prepost_3000.json").read_text())
    k = int(np.searchsorted(np.array(r["entry"]["lags"]), 1000))
    expected = {  # events, fitted dH, dH_loc(1000)
        "entry": (6524, 0.0101, 0.0269), "exit": (13643, 0.0052, 0.0307), "after minus before": (6320, 0.0037, -0.0033),
        "control, group flights alone": (16998, 0.0046, -0.0017), "control, solo flights": (69648, 0.0088, -0.0017)}
    for name, (events, dh, loc) in expected.items():
        assert r[name]["events"] == events, name
        assert (round(r[name]["dH"], 4), round(r[name]["dH_loc"][k], 4)) == (dh, loc), name
    p = r["profile"]
    assert p["events"] == 3334
    assert [round(p[s]["H_loc(1000)"], 4) for s in prepost.STAGES] == [0.8795, 0.9098, 0.9108, 0.8765]


def test_performance_reads_runs_at_their_ends_and_flies_30_min_straight():
    t = np.arange(0, 4001, 1.0)
    z = np.where(t < 600, 1000 + 2 * t, 2200 - (t - 600))  # climb at 2 m/s, then lose 1 m/s
    fixes = pd.DataFrame({"segment_id": 0, "t": t, "E": 10 * t, "N": 0.0, "z": z})
    runs = pd.DataFrame({"segment_id": 0, "t_start": [0.0, 600.0], "t_end": [599.0, 4000.0]})
    spans = pd.DataFrame({"flight": [1], "population": ["solo"], "episode": [-1], "a": [0.0], "b": [4001.0]})
    dz, chord, xc = performance.flight_terms(fixes, runs, spans)
    assert dz.tolist() == [1198.0, -3400.0] and chord[1] == 34000
    assert xc == [("solo", 18000.0 * 221, 221, 2000.0)]  # 401 grid points, 180 steps of 10 s; middle at 2000 s
    runs = pd.DataFrame({"flight": 1, "population": "solo", "phase": ["climb", "transition", "climb"],
                         "t_start": [0.0, 600, 4100], "t_end": [599.0, 4000, 4200]})
    xc = pd.DataFrame(xc, columns=["population", "xc_m", "xc_origins", "t_mid"]).assign(flight=1)
    rows = performance.table(runs, np.array([1198.0, -3400, np.nan]), np.array([5990.0, 34000, np.nan]), xc)
    r = budget.ratios(rows.sum(numeric_only=True), performance.STATS)  # the unreached climb is left out
    assert r["climb rate (m/s)"] == 2 and r["height gained per climb (m)"] == 1198
    assert r["glide speed (km/h)"] == pytest.approx(36) and r["glide ratio"] == 10
    assert r["cross-country speed over 30 min (km/h)"] == pytest.approx(36)


# The README table: value and band, per statistic and population (solo, group, group_solo).
PERFORMANCE = {
    "climb rate (m/s)": [(1.22, [1.22, 1.22]), (1.26, [1.25, 1.27]), (1.33, [1.32, 1.34])],
    "height gained per climb (m)": [(207, [206, 207]), (212, [210, 214]), (223, [221, 225])],
    "glide speed (km/h)": [(30.1, [30.1, 30.2]), (33.3, [33.1, 33.5]), (32.8, [32.6, 32.9])],
    "glide ratio": [(10.4, [10.4, 10.4]), (12.9, [12.8, 13.0]), (11.3, [11.2, 11.4])],
    "cross-country speed over 30 min (km/h)": [(17.2, [17.2, 17.3]), (20.0, [19.8, 20.3]), (20.0, [19.8, 20.1])],
}


@pytest.mark.skipif(not (tamsd.OUT / "performance.json").exists(), reason="run performance.py first")
def test_performance_numbers_of_the_readme():
    r = json.loads((tamsd.OUT / "performance.json").read_text())
    for name, rows in PERFORMANCE.items():
        k = performance.STATS[name][3]
        got = [(round(r[p][name]["value"], k), np.round(r[p][name]["band"], k).tolist()) for p in tamsd.POPULATIONS]
        assert got == rows, name
    terms = pd.read_parquet(tamsd.CACHE / "performance_terms.parquet")
    solo = terms[terms["population"] == "solo"]
    assert solo.loc[solo["xc_origins"] > 0, "flight"].nunique() == 112292
    assert solo.loc[solo["climbs"] + solo["transition_s"] > 0, "flight"].nunique() == 80242


def test_events_of_the_episode_figure_of_the_note():
    # c1 {1,2} + c2 {3,4} merge into c3, which splits into c4 {1,2} and c5 {3,4}; c4 dissolves in
    # flight, c5 when 4 lands; c6 {2,5} opens a new episode when 5 departs. Episode 1: c7 {6,7}
    # gains 8 (c8), which leaves in flight (c9), then c9 dissolves when 7 lands.
    rows = [(1, 0, 0, 100, [1, 2]), (2, 0, 0, 100, [3, 4]), (3, 0, 100, 200, [1, 2, 3, 4]),
            (4, 0, 200, 300, [1, 2]), (5, 0, 200, 280, [3, 4]), (6, 2, 400, 500, [2, 5]),
            (7, 1, 0, 60, [6, 7]), (8, 1, 60, 120, [6, 7, 8]), (9, 1, 120, 180, [6, 7])]
    groups = pd.DataFrame(rows, columns=["group_id", "episode_id", "t_start", "t_end", "members"])
    groups["members"] = groups["members"].map(np.array)
    flights = pd.DataFrame({"flight": range(1, 9), "departure": [-50, -50, -50, -50, 398, -50, -50, -50],
                            "landing": [900, 900, 900, 282, 900, 900, 178, 900]})
    ev = describe.events(groups, flights)
    assert ev["outcome"].tolist() == ["merge", "merge", "split", "leave", "landing", "leave", "join", "leave",
                                      "landing"]
    assert ev["dissolves"].tolist() == [False, False, False, True, True, True, False, False, True]
    assert ev["form"].tolist() == ["flight", "flight", "", "", "", "departure", "flight", "", ""]


def test_rates_are_counts_per_hour_and_pool_the_largest_sizes():
    table = pd.DataFrame({"n": [2, 2, 3, 9, 12], "hours": [1.0, 3.0, 2.0, 0.5, 0.5],
                          "outcome": ["leave", "join", "split", "merge", "merge"], "block": [0, 1, 2, 3, 3]})
    r = describe.rates(table, np.random.default_rng(0))
    assert r["hours"][:2] == [4.0, 2.0] and r["hours"][-1] == 1.0
    assert r["rate"]["leave"][0] == 0.25 and r["rate"]["join"][0] == 0.25 and r["rate"]["split"][1] == 0.5
    assert r["rate"]["merge"][-1] == 2.0


def test_exposure_is_cut_at_the_age_and_hour_edges():
    # a group of 30 min that starts at the age of 10 min and at 12:50: it crosses the age edges at 15 and 30 min
    # and the hour edge at 13:00 h
    mins = 60.0
    x = age.exposure(np.array([10 * mins]), np.array([30 * mins]), np.array([(12 * 60 + 50) * mins]),
                     np.array([0, 15, 30, 60, np.inf]) * mins, np.array([0, 13, 15, 24]) * 3600)
    assert x.shape == (1, 4, 3)
    assert (x[0] / mins).round(6).tolist() == [[5, 0, 0], [5, 10, 0], [0, 10, 0], [0, 0, 0]]
    assert x.sum() == 30 * mins


@pytest.mark.skipif(not (tamsd.OUT / "describe.json").exists(), reason="run describe.py first")
def test_describe_numbers_of_the_readme():
    r = json.loads((tamsd.OUT / "describe.json").read_text())
    count, hours = r["size_count"], r["size_hours"]
    assert (r["groups"], r["episodes"], round(100 * r["with_hang"], 1)) == (40051, 17169, 0.9)
    assert round(100 * count["2"] / sum(count.values())) == 63 and round(100 * hours["2"] / sum(hours.values())) == 77
    assert max(map(int, count)) == 27
    assert [round(100 * r[k], 1) for k in ("episode_single", "episode_with_split", "episode_with_merge")] == [74.3, 5.0, 4.4]
    assert (round(r["median_group_min"]), round(r["median_episode_min"]), round(100 * r["episode_2h"])) == (30, 58, 12)
    form, end = r["formations"], r["dissolutions"]
    assert sum(form.values()) == 18218 and round(100 * form["departure"] / 18218) == 34
    assert sum(end.values()) == 18376 and round(100 * end["leave"] / 18376) == 91 and round(100 * end["landing"] / 18376) == 9
    rate = r["rates"]["rate"]
    assert [round(rate[x][0], 2) for x in ("leave", "join")] == [0.96, 0.31]
    assert [round(rate[x][0], 3) for x in ("landing", "merge")] == [0.098, 0.085]
    assert (round(rate["split"][2], 2), round(rate["split"][-1], 1)) == (0.48, 2.7)
    assert (round(r["median_group_min_by_size"]["2"]), round(r["median_group_min_by_size"]["8"])) == (41, 4)


def test_chain_shapes_and_diameters():
    import chains
    assert chains.diameter(3, [(0, 1), (1, 2)]) == 2 and chains.diameter(3, [(0, 1), (1, 2), (0, 2)]) == 1
    assert chains.diameter(4, [(0, 1), (2, 3)]) == np.inf
    assert [chains.shape(3, 3), chains.shape(3, 2), chains.shape(4, 3), chains.shape(4, 4)] == \
        ["complete", "tree", "tree", "between"]
    group = pd.Series({"members": np.array([10, 11, 12]), "t_start": 0, "t_end": 100})
    edges = pd.DataFrame({"a": [10, 11, 10], "b": [11, 12, 12], "t_on": [-50, 0, 40], "t_off": [200, 100, 70]})
    assert chains.pieces_of(group, edges) == [(0, 40, 2, 2), (40, 70, 3, 1), (70, 100, 2, 2)]


def test_hierarchy_order_triangles_and_switches():
    import hierarchy
    assert hierarchy.order(3, [(0, 1), (1, 2)]) == (True, 1, 2)
    assert hierarchy.order(3, [(0, 1), (0, 2)]) == (True, 1, 1)
    assert hierarchy.order(3, [(0, 1), (1, 2), (2, 0)])[0] is False
    edges = [(0, 1, 30.0), (1, 2, 30.0), (0, 2, 30.0)]
    tris = hierarchy.triangle_list(edges)
    assert tris == [(0, 1, 2)]
    assert hierarchy.triangles(tris, hierarchy.directed(edges)) == (1, 0)
    assert hierarchy.triangles(tris, [(0, 1), (1, 2), (2, 0)]) == (0, 1)
    assert hierarchy.directed([(0, 1, 10.0), (0, 1, -20.0)]) == [(1, 0)]
    # 2 min ahead, 30 s behind (dropped), 2 min ahead, 2 min behind: one switch
    s = np.r_[np.ones(24), -np.ones(6), np.ones(24), -np.ones(24)].astype(int)
    assert hierarchy.runs(s, 5) == [(1, 48), (-1, 24)]


def test_cohesion_delay_statistics():
    import cohesion
    x = cohesion.dense(np.array([0, 5, 10, 20]), np.array([1, 2, 3, -1], np.int8),
                       np.array([False, False, True, False]), 0, 25, 5)
    assert np.allclose(x, [5, 10, np.nan, np.nan, -5], equal_nan=True)
    sums, counts, drift, dcount = cohesion.accumulate(np.array([0.0, 10.0, 20.0, 30.0]), np.zeros(4, int), 1, 5)
    assert sums[0, 0] == 300 and counts[0, 0] == 3 and sums[0, 1] == 800 and counts[0, 1] == 2
    s = np.array([5, 10, 30, 60, 120, 300, 600, 1200, 1800], float)
    sigma2, k, c = cohesion.fit(s, cohesion.ou(s, 900.0, 1 / 240, 50.0))
    assert np.allclose([sigma2, k, c], [900, 1 / 240, 50], rtol=1e-3)


@pytest.mark.skipif(not (tamsd.OUT / "chains.json").exists(), reason="run chains.py first")
def test_chain_numbers_of_the_readme():
    r = json.loads((tamsd.OUT / "chains.json").read_text())
    s, e = r["shares"], r["effect"]
    assert [round(100 * s["shape"][k][0]) for k in ("complete", "tree")] == [53, 47]
    assert (round(100 * s["shape"]["complete"][1]), round(100 * s["shape"]["complete"][-1], 1)) == (27, 2.5)
    assert (round(s["density"][0], 2), round(s["density"][-1], 2), round(100 * s["diameter"]["4"][-1])) == (0.84, 0.45, 61)
    assert s["not_connected"] == 0
    got = {k: (v["flights"], round(v["H_loc_1000"], 3), np.round(v["H_loc_1000_band"], 3).tolist()) for k, v in e.items()}
    assert got == {"n3 triangle": (5322, 0.906, [0.901, 0.910]), "n3 chain": (4882, 0.907, [0.902, 0.911]),
                   "n4+ complete": (1638, 0.915, [0.905, 0.923]), "n4+ tree": (1171, 0.907, [0.894, 0.920])}


@pytest.mark.skipif(not (tamsd.OUT / "partners.json").exists(), reason="run partners.py first")
def test_partner_numbers_of_the_readme():
    r = json.loads((tamsd.OUT / "partners.json").read_text())
    o, n = r["observed"], r["null"]
    assert (r["pair_days"], r["pilots"], r["pairs"], r["max_days"]) == (30550, 4525, 22513, 69)
    assert round(100 * o["repeat_share"], 1) == 37.6 and round(100 * n["repeat_share"]["mean"], 1) == 26.7
    assert np.round(100 * np.array(n["repeat_share"]["band"]), 1).tolist() == [26.5, 27.0]
    assert o["repeat_pairs"] == 3437 and round(n["repeat_pairs"]["mean"]) == 2677
    assert np.round(n["repeat_pairs"]["band"]).tolist() == [2644, 2721]
    assert round(100 * o["same_club_main"], 1) == 32.1 and round(100 * n["same_club"]["mean"], 1) == 24.2
    assert np.round(100 * np.array(n["same_club"]["band"]), 1).tolist() == [24.0, 24.4]


@pytest.mark.skipif(not (tamsd.OUT / "fission.json").exists(), reason="run fission.py first")
def test_fission_numbers_of_the_readme():
    r = json.loads((tamsd.OUT / "fission.json").read_text())
    band, pred = np.array(r["measured_band"]), np.array(r["measured"])
    t = np.array(r["T"][:len(pred)])
    assert (np.abs(pred / t - 1) < 0.04).all()
    inside = (band[0] <= t) & (t <= band[1])
    assert inside.tolist() == [True, False, True, True, True, True, True]
    assert (round(pred[1], 3), round(t[1], 3)) == (0.145, 0.148)
    assert (round(100 * r["multi_share"][0]), round(100 * r["multi_share"][-1])) == (6, 46)
    c = r["capita"]
    assert (round(c["a"], 3), np.round(c["a_band"], 3).tolist()) == (0.162, [0.158, 0.167])
    assert (round(c["gamma"], 2), np.round(c["gamma_band"], 2).tolist()) == (0.30, [0.28, 0.33])
    loss = np.array(r["rates"]["loss"]) / np.array(r["rates"]["n"])
    assert (round(loss[0], 2), round(loss[-1], 2), round(r["x"], 2)) == (0.53, 0.25, 0.25)
    assert (round(r["T"][-1] / r["log_series"][-1], -1), round(r["T"][-1] / r["per_capita"][-1])) == (290, 4)
    assert (round(r["bonabeau"]["alpha"], 1), round(r["bonabeau"]["n_c"])) == (3.4, 5)


@pytest.mark.skipif(not (tamsd.OUT / "cohesion.json").exists(), reason="run cohesion.py first")
def test_cohesion_numbers_of_the_readme():
    r = json.loads((tamsd.OUT / "cohesion.json").read_text())
    assert r["intervals"] == {"real": 34082, "chance": 134202, "chance_long": 980}
    f, d = r["fit"], r["drift"]

    def minutes(k, b):
        return round(1 / k / 60, 1), np.round(1 / np.array(b[::-1]) / 60, 1).tolist()

    assert minutes(f["real/all"]["k"], f["real/all"]["k_band"]) == (4.7, [4.6, 4.7])
    assert minutes(f["chance_long/all"]["k"], f["chance_long/all"]["k_band"]) == (4.8, [4.5, 5.1])
    assert (round(f["real/all"]["sigma"]), np.round(f["real/all"]["sigma_band"]).tolist()) == (55, [54, 55])
    assert (round(f["chance_long/all"]["sigma"]), np.round(f["chance_long/all"]["sigma_band"]).tolist()) == (79, [78, 80])
    assert (round(f["real/all"]["q"], 1), np.round(f["real/all"]["q_band"], 1).tolist()) == (21.4, [20.9, 22.0])
    assert (round(f["chance_long/all"]["q"]), np.round(f["chance_long/all"]["q_band"]).tolist()) == (43, [41, 46])
    assert (round(f["chance/all"]["q"]), np.round(f["chance/all"]["q_band"]).tolist()) == (62, [61, 63])
    assert [round(f[f"real/{n}"]["q"], 1) for n in ("2", "3", "4+")] == [21.2, 21.4, 21.8]
    assert minutes(d["real/all"]["k"], d["real/all"]["k_band"]) == (6.9, [6.7, 7.0])
    assert minutes(d["chance_long/all"]["k"], d["chance_long/all"]["k_band"]) == (7.5, [6.9, 8.1])
    s = r["sep"]
    km = {k: [round(v["median"][m // 5] / 1000, 1) for m in (30, 60, 120)] for k, v in s.items()}
    assert km["real_long"][1:] == [1.0, 2.9] and km["chance_long"][1:] == [2.4, 5.8]
    assert km["real"][0] == 3.7 and km["chance"][0] == 5.1


@pytest.mark.skipif(not (tamsd.OUT / "hierarchy.json").exists(), reason="run hierarchy.py first")
def test_hierarchy_numbers_of_the_readme():
    r = json.loads((tamsd.OUT / "hierarchy.json").read_text())
    o, s, w = r["order"], r["switches"], r["who_leads"]

    def pct(v, d=1):
        return round(100 * v["value"], d), np.round(100 * np.array(v["band"]), d).tolist()

    assert pct(o["cyclic"]) == (2.3, [1.8, 2.8]) and round(100 * o["cyclic_null"]["value"]) == 25
    assert round(100 * o["full_share"]) == 34 and round(100 * o["acyclic"]["value"], 1) == 98.7
    assert round(100 * o["acyclic_null"]["value"]) == 88
    assert pct(o["one_leader"]) == (80.9, [80.0, 82.0]) and pct(o["one_leader_null"]) == (77.2, [76.7, 77.8])
    assert [round(100 * o["by_size"][k]["one_leader"]) for k in ("3", "4", "5")] == [86, 68, 52]
    assert (round(s["real"]["per_hour"], 1), np.round(s["real"]["per_hour_band"], 1).tolist()) == (3.3, [3.3, 3.4])
    assert (round(s["chance"]["per_hour"], 1), np.round(s["chance"]["per_hour_band"], 1).tolist()) == (4.3, [4.1, 4.4])
    assert (round(100 * s["real"]["dominant_median"]), round(100 * s["chance"]["dominant_median"])) == (74, 69)
    assert (w["experience"]["pairs"], pct(w["experience"])) == (16294, (58.5, [57.8, 59.1]))
    assert (w["wing"]["pairs"], pct(w["wing"])) == (8475, (64.3, [63.4, 65.3]))


def test_route_covers_the_same_route_flown_the_same_way_and_not_a_far_one():
    x = np.arange(0, 20000, 100.0)
    piece = np.column_stack([x, np.zeros_like(x)])[::3]
    near = np.column_stack([x, np.full_like(x, 200.0)])  # the same route, 200 m to the side
    far = np.column_stack([x, np.full_like(x, 2000.0)])
    owner = np.r_[np.full(len(x), 7), np.full(len(x), 8)]
    share = route.coverage(piece, route.cKDTree(np.r_[near, far]), owner)
    assert share.to_dict() == {7: 1.0}
    assert route.coverage(piece, route.cKDTree(np.r_[near, far]), owner, exclude=[7]).empty
    assert route.segment(near, piece[0], piece[-1]) == (0, 3 * (len(piece) - 1))  # nearest to the last checked point
    assert route.segment(near[::-1], piece[0], piece[-1]) is None  # flown the other way


def test_route_sites_join_close_takeoffs_and_place_their_origins():
    lat = np.array([45.0, 45.0 + 300 / route.EARTH_M * 180 / np.pi, 46.0])
    lon = np.array([6.0, 6.0, 6.0])
    site, de, dn = route.site_frames(lat, lon)
    assert site[0] == site[1] != site[2]
    assert np.allclose(de, 0) and np.allclose(dn[:2], [-150, 150]) and dn[2] == 0


@pytest.mark.skipif(not (tamsd.OUT / "route_sample10000.json").exists(), reason="run route.py --sample 10000 first")
def test_route_numbers_of_the_readme():
    r = json.loads((tamsd.OUT / "route_sample10000.json").read_text())
    assert (r["group pieces"], r["alone flights"]) == (10000, 52371)
    a, s = r["any date"], r["same season and hour"]
    assert (a["pieces"], a["matches"], s["pieces"]) == (4933, 52912, 4183)
    rounded = lambda v: [round(v["value"], 3), [round(b, 3) for b in v["band"]]]
    assert rounded(a["H_loc(1000 s)"]) == [0.018, [0.014, 0.021]]
    assert rounded(s["H_loc(1000 s)"]) == [0.016, [0.012, 0.021]]
    assert rounded(a["baseline"]["H_loc(1000 s)"]) == [0.060, [0.055, 0.066]]
    explained = lambda e: round(100 * (1 - e["H_loc(1000 s)"]["value"] / e["baseline"]["H_loc(1000 s)"]["value"]))
    assert (explained(a), explained(s)) == (71, 74)
    variants = [json.loads((tamsd.OUT / f"route_sample2000{t}.json").read_text()) for t in
                ("", "_rho300_cover0.8", "_rho1000_cover0.8", "_rho500_cover0.7", "_rho500_cover0.9")]
    loc = [v[k]["H_loc(1000 s)"]["value"] for v in variants for k in ("any date", "same season and hour")]
    share = [explained(v[k]) for v in variants for k in ("any date", "same season and hour")]
    assert (round(min(loc), 3), round(max(loc), 3), min(share), max(share)) == (0.014, 0.022, 66, 76)


def test_route_share_is_taken_inside_each_draw_and_sizes_count_flights_and_pilots():
    gap, base = [0.02, 0.01, 0.03, 0.02], [0.06, 0.06, 0.03, 0.04]  # the point estimate first
    s = route.share_removed(gap, base)
    draws = 100 * (1 - np.array([0.01, 0.03, 0.02]) / np.array([0.06, 0.03, 0.04]))
    assert s["value"] == pytest.approx(100 * (1 - 0.02 / 0.06))
    assert s["band"] == pytest.approx(np.percentile(draws, [5, 95]).tolist())
    tracks = {"flight": np.array([10, 11, 12, 13])}
    kept = pd.DataFrame({"piece": [0, 1, 1], "solo": [2, 3, 3]})
    pilot = pd.Series([0, 0, 1, 2], index=[10, 11, 12, 13])
    assert route.sizes(kept, tracks, pilot) == {"flights": 2, "pilots": 1, "control flights": 2, "control pilots": 2}


@pytest.mark.skipif(not (tamsd.OUT / f"open_min{openflights.MIN_OPENING:g}" / "route.json").exists(), reason="run route.py --open first")
def test_route_open_share_agrees_with_its_gaps():
    r = json.loads((tamsd.OUT / f"open_min{openflights.MIN_OPENING:g}" / "route.json").read_text())
    e = r["any date"]
    gap, base = e["H_loc(1000 s)"]["value"], e["baseline"]["H_loc(1000 s)"]["value"]
    assert e["share removed (%)"]["value"] == pytest.approx(100 * (1 - gap / base))
    assert e["share removed (%)"]["band"][0] < e["share removed (%)"]["value"] < e["share removed (%)"]["band"][1]


def test_the_longest_common_lag_keeps_the_minimum_of_flights():
    import longlags
    lags = [1000, 1500, 2000]
    assert longlags.longest({"lags": lags, "flights": [500, longlags.MIN_FLIGHTS, longlags.MIN_FLIGHTS - 1]}) == 1500
    assert longlags.longest({"lags": lags, "flights": [3, 2, 1]}) == 0


def test_regions_keep_the_busiest_cells_by_flights_of_group_time():
    import regions
    p = {"population": np.array(["group"] * 5 + ["solo"] * 3), "flight": np.array([1, 2, 3, 4, 4, 9, 9, 9])}
    code = np.array([8, 8, 8, 7, 7, 7, 7, 7])
    kept = regions.choose(p, code)  # the solo flights do not count, and a flight counts once per cell
    assert list(kept.index) == [8, 7] and list(kept) == [3, 1]


def test_the_panels_of_longlags_take_the_nearest_and_the_common_longest_lag():
    import longlags
    available = {k: {"lags": [1000, 4150, 11450], "flights": [500, 300, n]} for k, n in zip(longlags.POOLED, (900, 150, 99))}
    assert longlags.panels(available) == {"fixed": longlags.FIXED_LAG, "longest": 4150}


def units_of_runs(climb_s, time_s, found, searches):
    """Per unit: the terms of a time share and of the searches that end in a climb, and the phase time."""
    rows = pd.DataFrame({"climb_s": climb_s, "s": time_s, "found": found, "searches": searches})
    unit = np.arange(len(rows))
    return rows, unit, pd.Series(time_s, index=unit, dtype=float)


def test_average_first_gives_the_mean_of_the_shares():
    stats = {k: budget.STATS[k] for k in ("time in climb (%)", "searches that end in a climb (%)")}
    rows, unit, time = units_of_runs([10.0, 40.0, 150.0], [100.0, 200.0, 300.0], [1, 3, 0], [2, 4, 0])
    r, q = budget.unit_rates(rows, unit, time, stats)
    value, draws = budget.mean_ratios(r, q, np.arange(3), np.ones((4, 3)), stats)
    assert value["time in climb (%)"] == pytest.approx(100 * np.mean([0.1, 0.2, 0.5]))
    assert draws[:, 0] == pytest.approx(value["time in climb (%)"])
    # the searches: sum (found_u / T_u) / sum (searches_u / T_u)
    assert value["searches that end in a climb (%)"] == pytest.approx(100 * (1 / 100 + 3 / 200) / (2 / 100 + 4 / 200))


def test_a_unit_with_one_search_weighs_as_its_searches_not_as_a_unit():
    stats = {"searches that end in a climb (%)": budget.STATS["searches that end in a climb (%)"]}
    # nine units with 10 searches, half found, and one unit with a single search that is found, all of 1000 s
    found, searches = [5.0] * 9 + [1.0], [10.0] * 9 + [1.0]
    rows, unit, time = units_of_runs([0.0] * 10, [1000.0] * 10, found, searches)
    r, q = budget.unit_rates(rows, unit, time, stats)
    value, _ = budget.mean_ratios(r, q, np.arange(10), np.ones((1, 10)), stats)
    mean_of_shares = 100 * np.mean(np.array(found) / np.array(searches))
    assert value.iloc[0] == pytest.approx(100 * 46 / 91)  # equal times: the ratio of sums
    assert abs(value.iloc[0] - 50) < abs(mean_of_shares - 50) / 5
    # the same units with time given by the number of searches: the share of each unit counts once
    unit_share = budget.unit_rates(rows, unit, pd.Series(searches, index=unit), stats)
    assert budget.mean_ratios(*unit_share, np.arange(10), np.ones((1, 10)), stats)[0].iloc[0] == pytest.approx(mean_of_shares)


def test_rates_per_unit_and_the_bootstrap_of_whole_blocks():
    stats = {"climbs per hour": budget.STATS["climbs per hour"], "climb rate (m/s)": ("gain_m", "climb_s", 1, 2)}
    rows = pd.DataFrame({"climbs": [2.0, 1.0, 0.0], "hours": [0.5, 0.5, 1.0], "gain_m": [100.0, 30.0, 0.0],
                         "climb_s": [200.0, 100.0, 0.0]})
    time = pd.Series([1800.0, 1800.0, 3600.0, 60.0], index=[7, 8, 9, 10])  # unit 10 has no row
    r, q = budget.unit_rates(rows, [7, 8, 9], time, stats)
    assert r["climbs per hour"].tolist() == pytest.approx([2 / 1800, 1 / 1800, 0, 0])
    assert q["climb rate (m/s)"].tolist() == pytest.approx([200 / 1800, 100 / 1800, 0, 0])
    # units 7 and 8 in block 0, 9 and 10 in block 1; draws: block 0 twice, block 1 twice, block 1 only
    value, draws = budget.mean_ratios(r, q, np.array([0, 0, 1, 1]), np.array([[2, 0], [1, 1], [0, 2]]), stats)
    assert value["climbs per hour"] == pytest.approx(2.0)  # the mean of 4, 2 and 0 climbs per hour
    assert draws[0, 0] == pytest.approx(3.0) and draws[1] == pytest.approx(value.to_numpy()) and draws[2, 0] == 0
    assert value["climb rate (m/s)"] == pytest.approx(130 / 300)


def test_survival_curve_with_equal_weights_is_the_plain_ccdf():
    x = np.array([3.0, 5.0, 5.0, 9.0, 40.0, 41.0, 100.0, 250.0])
    block = np.arange(len(x)) % 3
    grid = np.array([1.0, 5.0, 6.0, 41.0, 250.0, 1000.0])
    got = survival.survival_curve(x, block, np.ones((5, 3)), grid=grid)
    assert got["value"] == pytest.approx([(x > t).mean() for t in grid])
    assert np.array(got["band"]) == pytest.approx(np.array([got["value"]] * 2))


def test_survival_curve_with_weights_and_blocks():
    x, w = np.array([1.0, 2.0, 3.0]), np.array([1.0, 1.0, 2.0])
    grid = np.array([0.5, 1.5, 2.5, 3.0])
    counts = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]])
    got = survival.survival_curve(x, np.array([0, 1, 2]), counts, w, grid)
    assert got["value"] == pytest.approx([1, 0.75, 0.5, 0])
    # draws: only the run of x = 1; only x = 2; only x = 3; all: the band is the 5-95% over them at each point
    draws = np.array([[1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0], [1, 0.75, 0.5, 0]])
    assert got["band"] == pytest.approx(np.percentile(draws, [5, 95], axis=0))


def test_ensemble_with_given_counts_resamples_those_blocks():
    values = np.array([[1.0], [3.0], [10.0]])
    mean, draws = tamsd.ensemble(values, np.array([0, 0, 1]), None, np.array([[1, 0], [0, 2], [2, 1]]))
    assert mean[0] == pytest.approx(14 / 3)
    assert draws[:, 0] == pytest.approx([2.0, 10.0, (2 + 6 + 10) / 5])


def test_runs_belong_to_the_piece_that_holds_their_middle():
    import raw
    table = pd.DataFrame({"flight": [1, 1, 1, 2], "t_mid": [5.0, 15.0, 25.0, 5.0]})
    mask = raw.in_pieces(table, np.array([1, 1, 2]), np.array([0.0, 20.0, 100.0]), np.array([10.0, 30.0, 200.0]))
    assert mask.tolist() == [True, False, True, False]


def test_duration_shares_count_the_pieces_of_one_population_in_each_bin():
    import raw
    p = {"duration": np.array([1300.0, 1500.0, 2500.0, 9000.0, 1400.0]),
         "population": np.array(["group", "group", "group", "group", "solo"])}
    assert raw.duration_shares(p, "group", np.array([1250.0, 2000.0, 3000.0, 10000.0])).tolist() == [0.5, 0.25, 0.25]
    assert raw.duration_shares(p, "solo", np.array([1250.0, 2000.0])).tolist() == [1.0]


def synthetic_windows(n_events=30, g=0.04, d=0.015, with_post=True):
    """Windows pre, start, end, post of n events, one flight each: H of a window = base + g in the group + d x its
    position in time (pre 0, start 1, end 2 + stay, post 3 + stay), MSD = lag^(2H)."""
    lags = longlags.LONG_LAGS[longlags.LONG_LAGS <= 1840]
    rng = np.random.default_rng(1)
    rows = []
    for e in range(n_events):
        base, stay = rng.uniform(0.7, 0.8), rng.integers(0, 5)
        for label, inside, pos in (("pre", 0, 0), ("start", 1, 1), ("end", 1, 2 + stay), ("post", 0, 3 + stay)):
            if label == "post" and not with_post and e % 2:
                continue
            rows.append((label, e, base + g * inside + d * pos))
    pop, event, h = (np.array(c) for c in zip(*rows))
    msd = lags[None, :].astype(float) ** (2 * h[:, None])
    units = decompose.Units(pop, event, np.arange(len(pop)), msd, flight=np.arange(len(pop)))
    return units, lags, h


def test_drift_free_estimator_returns_the_effect_and_the_drift_exactly():
    g, dw = prepost.drift_free(np.array([0.055, 0.1]), np.array([0.025, 0.02]))
    assert g == pytest.approx([0.04, 0.06], abs=1e-15) and dw == pytest.approx([0.015, 0.04], abs=1e-15)


def test_the_msd_differences_on_four_window_events_give_g_and_the_drift():
    units, lags, _ = synthetic_windows()
    counts = np.random.default_rng(0).multinomial(len(units.population), np.full(len(units.population), 1 / len(units.population)), size=7)
    on = lambda labels: prepost.curves_on(units, counts, labels, lags)
    out = prepost.effects(on)
    for name, expected in (("D_in", 0.055), ("D_out", 0.025), ("g", 0.04), ("d w", 0.015)):
        values, events, windows, flights = out[name]
        assert events == 30 and windows == 30 and flights == 30
        k = list(lags).index(1000)
        assert values[0, k] == pytest.approx(expected, abs=1e-9) and values[0, -1] == pytest.approx(expected, abs=1e-9), name
    assert out["D_in"][1] == 30 and out["post - pre"][1] == 30


def test_events_without_a_window_leave_the_pairs_that_need_it():
    units, lags, _ = synthetic_windows(with_post=False)  # odd events have no post
    counts = np.ones((3, len(units.population)), np.int32)
    out = prepost.effects(lambda labels: prepost.curves_on(units, counts, labels, lags))
    assert out["D_in"][1] == 15 and out["D_out"][1] == 15 and out["g"][1] == 15 and out["post - pre"][1] == 15
    # the pre window on the 15 events with a post is the same as the post's partner: g is still exact there
    assert out["g"][0][0, list(lags).index(1000)] == pytest.approx(0.04, abs=1e-9)


def test_phase_differences_by_window_recover_the_effect_and_the_drift_with_the_new_estimator():
    stats = {"climbs per hour": budget.STATS["climbs per hour"]}
    rng = np.random.default_rng(2)
    meta, rate = [], []
    for e in range(20):
        base, stay = rng.uniform(1, 2), rng.integers(0, 4)
        for label, inside, pos in (("pre", 0, 0), ("start", 1, 1), ("end", 1, 2 + stay), ("post", 0, 3 + stay)):
            meta.append((e, label, e % 4)), rate.append(base + 0.5 * inside + 0.2 * pos)
    meta = pd.DataFrame(meta, columns=["flight", "population", "block"]).assign(episode=lambda m: m["flight"])
    # windows of different lengths: the rate per hour of each is the same whatever the time, so the mean over windows is exact
    r = pd.DataFrame({"climbs per hour": np.array(rate) / 3600})
    q = pd.DataFrame({"climbs per hour": np.full(len(r), 1 / 3600)})
    counts = np.random.default_rng(3).multinomial(4, np.full(4, 0.25), size=5)
    out = prepost.effects(lambda labels: prepost.phase_on(meta, r, q, counts, labels, stats))
    assert out["g"][0][0, 0] == pytest.approx(0.5, abs=1e-12) and out["d w"][0][0, 0] == pytest.approx(0.2, abs=1e-12)
    assert out["D_in"][0][0, 0] == pytest.approx(0.7) and out["post - pre"][2] == 20
    # a draw that keeps only the events of block 1 still gives g = 0.5 (the effect is the same in every event)
    only = np.array([[0, 5, 0, 0]])
    assert prepost.effects(lambda labels: prepost.phase_on(meta, r, q, only, labels, stats))["g"][0][0, 0] == pytest.approx(0.5, abs=1e-12)


def test_the_window_lags_stop_at_eight_tenths_of_the_window():
    assert prepost.window_lags(2400).max() == 1840 and prepost.window_lags(1250).max() == 1000


def test_prepost_open_numbers_of_the_report():
    path = tamsd.OUT / "open_min0.1" / "prepost_2400.json"
    if not path.exists():
        pytest.skip("prepost.py --open --window 2400 has not run")
    r = json.loads(path.read_text())
    d, k = r["differences"], "H_loc(1000 s)"
    assert r["lags"][-1] == 1840 and r["n_boot"] == 1000
    assert r["curves"]["stages"]["events"] == 1595 and d["g"]["events (MSD)"] == 1595
    assert (d["D_in"]["events (MSD)"], d["D_out"]["events (MSD)"], d["post - pre"]["events (MSD)"]) == (1595, 1595, 1818)
    assert [round(d[n][k]["value"], 3) for n in ("D_in", "D_out", "post - pre", "g", "d w")] == [0.025, 0.027, 0.002, 0.026, -0.001]
    assert d["g"][k]["value"] == pytest.approx((d["D_in"][k]["value"] + d["D_out"][k]["value"]) / 2, abs=1e-12)
    assert d["d w"][k]["value"] == pytest.approx((d["D_in"][k]["value"] - d["D_out"][k]["value"]) / 2, abs=1e-12)
    # H_loc(1000 s) has its window cut at 1000 s, as on dataset 1000 of the other points
    stages = r["curves"]["stages"]
    assert d["D_in"][k]["value"] == pytest.approx(stages["start"][k] - stages["pre"][k], abs=1e-12)


def _utc(*stamps):
    return pd.to_datetime(list(stamps), utc=True).as_unit("s").asi8.astype(float)


def test_matched_gaps_are_circular_and_in_local_time():
    # 23:30 and 00:30 Paris (winter, UTC+1) are one hour apart across midnight; summer is UTC+2
    a, b = _utc("2024-01-10 22:30", "2024-07-10 21:30"), _utc("2024-01-10 23:30", "2024-07-10 22:45")
    assert np.allclose(matched.time_of_day(a), [23.5, 23.5]) and np.allclose(matched.time_of_day(b), [0.5, 0.75])
    assert np.allclose(matched.hour_gap(a, b), [1.0, 1.25])
    assert np.allclose(matched.hour_gap(_utc("2024-03-01 08:00"), _utc("2024-03-01 11:30")), 3.5)
    m = matched.month_gap(_utc("2024-12-15", "2024-01-15", "2024-02-15", "2024-07-15"),
                          _utc("2024-01-15", "2024-12-15", "2024-12-15", "2024-01-15"))
    assert m.tolist() == [1, 1, 2, 6]


def test_matched_caliper_filter_and_the_main_caliper_rule():
    t = pd.DataFrame({"side": "G", "piece": [0, 0, 1, 2, 3], "solo": range(5), "cover": [0.9, 0.75, 0.85, 0.8, 0.8],
                      "dmonth": [0, 0, 2, 1, 1], "dhour": [0.5, 0.5, 1.5, 1.0, 1.01],
                      "class_piece": ["B", "B", "A", "C", None], "class_solo": ["B", "B", "A", "D", None],
                      "level_piece": ["beginners", "beginners", "beginners", "experienced", None],
                      "level_solo": ["beginners", "beginners", "beginners", "experienced", None],
                      "dur_piece": [2000, 2000, 4500, 4500, 4500], "dur_solo": [1300, 1300, 4500, 4500, 4500],
                      "isolated": [True, True, False, True, True]})
    pick = lambda *a, **k: matched.select(t, *a, **k)["solo"].tolist()
    assert pick("1000", "month1_hour1_class") == [0]  # cover 0.75, month 2, class C/D, hour 1.01 and no class leave
    assert pick("1000", "month1_hour1_level") == [0, 3]  # the level rescues C against D
    assert pick("1000", "month2_hour2_class", cover=0.7) == [0, 1, 2]
    assert pick("2760", "month2_hour2_class") == [2]  # the segment of 1300 s is too short; piece 2 is not isolated either
    assert pick("2760", "month2_hour2_level", isolation=True) == [3]
    assert pick("1000", "month1_hour1_class", cover=0.9) == [0]
    assert matched.tally(matched.select(t, "1000", "month2_hour2_level"))["all"] == {"pieces": 3, "median controls": 1.0}
    assert matched.tally(matched.select(t, "1000", "month2_hour2_level"))["beginners"]["pieces"] == 2
    pieces = {"a": {"beginners": 250, "experienced": 199}, "b": {"beginners": 200, "experienced": 200},
              "c": {"beginners": 900, "experienced": 900}}
    assert matched.choose_caliper(pieces) == "b"
    assert matched.choose_caliper({"a": pieces["a"]}) is None


def test_matched_isolation_of_a_segment_from_the_flags_of_its_points():
    flags = np.array([True, True, False, True, True])  # points at grid indices 0, 3, 6, 9, 12
    assert not matched.segment_isolated(flags, 0, 12, 3)
    assert matched.segment_isolated(flags, 7, 12, 3)  # points 9 and 12
    assert not matched.segment_isolated(flags, 4, 7, 3)  # point 6 only
    assert matched.segment_isolated(flags, 0, 4, 3)  # points 0 and 3


def _ones(rows, cols):
    return np.ones((len(rows), len(cols)))


def test_matched_survival_weighs_the_controls_by_one_over_the_controls_of_their_piece():
    grid = np.array([50.0, 500.0])
    # piece 0: runs 10, 100; its control: runs 10, 100. piece 1: runs 10; its two controls: runs 10 and runs 1000, 10
    piece = duration_counts_of([[10, 100], [10]], grid)
    control = np.vstack([duration_counts_of([[10, 100]], grid), duration_counts_of([[10], [1000, 10]], grid)])
    a, b = piece, control
    group, ctrl, _ = isolated.engine(np.array([0, 1, 1]), np.array([0, 1, 2]), a, b, np.array([0, 1]), np.array([0, 1, 2]), (_ones, 1))
    assert (group[:2, 0] / group[2, 0]).tolist() == [1 / 3, 0.0]  # 100 > 50 once of three runs
    # control side: piece 0 weighs 1 (runs 10, 100), piece 1 weighs 1/2 for each of its controls (runs 10; 1000, 10)
    total = 2 + 0.5 * 1 + 0.5 * 2
    assert ctrl[2, 0] == pytest.approx(total)
    assert ctrl[0, 0] / ctrl[2, 0] == pytest.approx((1 + 0.5) / total)  # runs above 50: 100 and 1000
    assert ctrl[1, 0] / ctrl[2, 0] == pytest.approx(0.5 / total)  # above 500: the 1000 only


def duration_counts_of(runs_by_unit, grid):
    unit = np.concatenate([np.full(len(r), u) for u, r in enumerate(runs_by_unit)])
    x = np.concatenate([np.asarray(r, float) for r in runs_by_unit])
    return matched.duration_counts(unit, x, len(runs_by_unit), grid)


def test_matched_duration_counts_with_equal_weights_are_the_plain_survival():
    x = np.array([5.0, 20.0, 20.0, 400.0, 90.0])
    grid = survival.GRID
    counts = matched.duration_counts(np.zeros(len(x), int), x, 1)[0]
    curve = survival.survival_curve(x, np.zeros(len(x), int), np.ones((2, 1)), grid=grid)
    assert (counts[:-1] / counts[-1]).tolist() == pytest.approx(curve["value"])


def test_matched_runs_go_to_the_window_that_holds_their_middle():
    ctx = object.__new__(matched.Context)
    runs = pd.DataFrame({"flight": [1, 1, 1, 2], "t_mid": [5.0, 15.0, 25.0, 15.0], "n_fixes": [3, 4, 5, 6]})
    ctx.runs = runs.assign(key=runs["flight"] * isolated.RANK + runs["t_mid"])
    unit, idx = matched.window_runs(ctx, np.array([1, 1, 2, 1]), np.array([0.0, 10.0, 0.0, 30.0]), np.array([20.0, 30.0, 10.0, 40.0]))
    assert unit.tolist() == [0, 0, 1, 1]  # window 2 (flight 2, [0, 10)) and window 3 hold no middle
    assert ctx.runs["n_fixes"].to_numpy()[idx].tolist() == [3, 4, 4, 5]


def test_matched_unit_rates_are_per_unit_shares_and_zero_without_phase_time():
    ctx = object.__new__(matched.Context)
    ctx.defs = raw.stat_defs()
    ctx.scale = np.array([next(s[n][2] for s in ctx.defs.values() if n in s) for n in matched.NAMES], float)
    rng = np.random.default_rng(3)
    base = {"flight": [1, 1, 1, 1], "t_mid": [5.0, 15.0, 25.0, 35.0]}
    b = pd.DataFrame(base | {c: rng.integers(1, 9, 4).astype(float) for st in ctx.defs["b"].values() for c in (*st[:2], "s")})
    p = pd.DataFrame(base | {c: rng.integers(1, 9, 4).astype(float) for st in ctx.defs["p"].values() for c in st[:2]})
    ctx.tables = {"b": b, "p": p}
    T, rq = ctx.rates(np.array([1, 1, 1]), np.array([0.0, 20.0, 100.0]), np.array([20.0, 40.0, 120.0]))
    assert T.tolist() == [b["s"][:2].sum(), b["s"][2:].sum(), 0.0]
    k = matched.NAMES.index("time in climb (%)")
    assert rq[0, k] == pytest.approx(b["climb_s"][:2].sum() / T[0])  # r = n / T
    assert rq[0, len(matched.NAMES) + k] == pytest.approx(1.0)  # q = d / T = s / T
    assert (rq[2] == 0).all()
    # the estimator on two units: scale * sum r / sum q, not the share of the sums
    r = rq[:2, k]
    assert 100 * r.sum() / rq[:2, len(matched.NAMES) + k].sum() == pytest.approx(100 * r.mean())


def test_matched_difference_of_deltas_is_taken_on_every_draw():
    draws = lambda g, c, gy, cy: {"m": {"group": (np.array(g, float), np.array(gy, float)),
                                        "control": (np.array(c, float), np.array(cy, float))}}
    out = matched.contrasts(draws([5, 4, 7], [1, 1, 1], [1, 2], [0, 1]), draws([3, 6, 2], [1, 1, 2], [4, 4], [1, 3]))
    assert out["G - S|G"]["m"][0].tolist() == [4, 3, 6] and out["A - S|A"]["m"][0].tolist() == [2, 5, 0]
    diff = out["(G - S|G) - (A - S|A)"]["m"]
    assert diff[0].tolist() == [2, -2, 6]
    assert diff[1].tolist() == [1 - 3, 1 - 1]  # years: (2 - 1) - (4 - 1) and (1 - 0) - (4 - 3)... per year
    # the band of the difference is that of its draws, not the difference of the bands
    x = np.r_[0.0, np.linspace(-1, 1, 102)]
    zero = {"m": {"group": (0 * x, None), "control": (0 * x, None)}}
    out = matched.contrasts({"m": {"group": (x, None), "control": zero["m"]["control"]}},
                            {"m": {"group": (-x, None), "control": zero["m"]["control"]}})
    lo, hi = matched.quoted(out["(G - S|G) - (A - S|A)"]["m"])["band"]
    assert (lo, hi) == pytest.approx(tuple(twoway.band(0.0, 2 * x[1:])))


def test_matched_long_series_stops_at_the_largest_lag_that_keeps_enough_pieces(monkeypatch):
    monkeypatch.setattr(matched, "MIN_PIECES", 2)
    t = pd.DataFrame({"side": "G", "piece": [0, 1, 2, 3], "solo": range(4), "cover": 0.9, "dmonth": 0, "dhour": 0.0,
                      "class_piece": "B", "class_solo": "B", "level_piece": "beginners", "level_solo": "beginners",
                      "dur_piece": [2000, 2000, 5000, 5000], "dur_solo": [5000, 1300, 5000, 5000], "isolated": True})
    line = {"caliper": "month1_hour1_class"}
    assert matched.long_lag(t, line, "all") == 2760  # pieces 2 and 3 last 5000 >= 2760 / 0.8
    t["dur_piece"] = [2000, 2000, 5000, 1500]
    assert matched.long_lag(t, line, "all") == 1500  # at 1500 s (pieces and segments of 1875 s) pieces 0 and 2 count
    t["dur_piece"] = 1000  # no piece reaches the 1250 s the lag of 1000 s needs
    assert matched.long_lag(t, line, "all") is None


def test_decomposition_shapley_identity_on_random_deltas():
    rng = np.random.default_rng(1)
    delta = {c: rng.normal(size=(7, 5)) for c in range(8)}
    phi = decomposition.shapley(delta)
    total = delta[7] + phi["R"] + phi["T"] + phi["W"]
    assert np.allclose(total, delta[0], rtol=0, atol=1e-12)
    delta[3][0, 0] = np.nan  # a NaN stays NaN on the two sides, the identity still passes
    delta[0][0, 0] = np.nan
    decomposition.shapley(delta)


def test_decomposition_shapley_of_additive_deltas_gives_the_parts_and_the_rest():
    a = {"R": 3.0, "T": -1.0, "W": 0.5}
    g = 0.7
    delta = {c: g + sum(v for k, (n, b) in zip("RTW", decomposition.FACTORS) for v in [a[n]] if not c & b) for c in range(8)}
    phi = decomposition.shapley(delta)
    assert {k: float(v) for k, v in phi.items()} == pytest.approx(a)
    assert delta[7] == pytest.approx(g)
    assert decomposition.shares(delta[0], {"R": phi["R"]})["R"] == pytest.approx(3.0 / (g + 2.5))
    assert decomposition.shares(1e-12, {"R": 1.0}) == {"R": None}


def _pool(starts, lengths, dates, labels):
    n = np.array(lengths)
    return decomposition.Pool(np.array(_utc(*starts)), n, np.array(dates), np.array(labels), np.arange(len(n)))


def test_decomposition_windows_sit_inside_their_piece_near_the_hour_and_are_as_long():
    # winter (UTC+1): the piece's middle is 23:40 local; pool pieces start at 09:00, 23:00 and 00:00 local
    mid = _utc("2024-01-10 22:40")[0]
    starts = [f"2024-01-{d:02d} {h}:00" for d in range(11, 31) for h in ("08", "22", "23")]  # local 09:00, 23:00, 00:00
    pool = _pool(starts, [600] * len(starts), [f"d{i}" for i in range(len(starts))], ["B"] * len(starts))
    n = 100
    rng = np.random.default_rng(0)
    rows, s = decomposition.windows_of(pool, n, "piece-day", "B", mid, decomposition.T | decomposition.W, rng, (1, 1), k=50)
    assert 0 < len(rows) <= 50 and len(set(pool.flight[rows])) == len(rows)
    assert (s >= 0).all() and (s + n <= pool.n[rows]).all()  # the window of n points lies inside the pool piece
    mids = pool.utc[rows] + (2 * s + n - 1) * tamsd.GRID_S / 2
    assert (matched.hour_gap(mids, np.full(len(mids), mid)) <= 1 + 1e-9).all()
    assert (matched.month_gap(mids, np.full(len(mids), mid)) <= 1).all()


def test_decomposition_windows_cross_midnight_and_obey_date_wing_and_length():
    mid = _utc("2024-01-10 23:15")[0]  # 00:15 local on the 11th
    pool = _pool(["2024-01-19 22:00", "2024-01-20 12:00", "2024-01-21 22:00", "2024-01-22 22:00"], [800, 800, 40, 800],
                 ["a", "b", "c", "a"], ["B", "B", "B", "A"])
    rows, s = decomposition.windows_of(pool, 100, "a", "B", mid, decomposition.T | decomposition.W, np.random.default_rng(3), (1, 1))
    assert rows.tolist() == []  # the only candidate on another date and long enough with the hour is piece 0, date a
    rows, s = decomposition.windows_of(pool, 100, "z", "B", mid, decomposition.T | decomposition.W, np.random.default_rng(3), (1, 1))
    assert rows.tolist() == [0]  # 1: noon never within an hour; 2: too short; 3: other wing
    assert (matched.hour_gap(pool.utc[rows] + (2 * s + 99) * tamsd.GRID_S / 2, np.full(1, mid)) <= 1 + 1e-9).all()
    rows, s = decomposition.windows_of(pool, 100, "z", "B", mid, 0, np.random.default_rng(3), (1, 1))
    assert sorted(rows.tolist()) == [0, 1, 3] and (s + 100 <= pool.n[rows]).all()  # no condition: any date, wing, hour


def test_decomposition_the_cells_with_route_are_nested_filters_of_the_match_table():
    # the last row: a class outside isolated.LEVELS ("Biplace", no level) matches itself and still has no wing
    t = pd.DataFrame({"side": "G", "piece": [0, 0, 1, 2, 3, 4], "solo": range(6), "cover": 0.9,
                      "dmonth": [0, 3, 1, 1, 0, 0], "dhour": [0.5, 0.5, 2.0, 1.0, 0.2, 0.1],
                      "class_piece": ["B", "B", "A", "C", "B", "Biplace"], "class_solo": ["B", "B", "A", "D", "B", "Biplace"],
                      "level_piece": ["beginners"] * 3 + ["experienced", "beginners", None],
                      "dur_piece": 2000, "dur_solo": [2000, 2000, 2000, 2000, 1000, 2000], "isolated": False})
    cal, win = (1, 1), t.iloc[:0]
    rows = lambda c: decomposition.cell_rows(t, win, c, "1000", cal, "class")["solo"].tolist()
    assert rows(decomposition.R) == [0, 1, 2, 3, 5]
    assert rows(decomposition.R | decomposition.T) == [0, 3, 5]
    assert rows(decomposition.R | decomposition.W) == [0, 1, 2]
    full = rows(decomposition.FULL)
    assert full == [0] == matched.select(t, "1000", "month1_hour1_class")["solo"].tolist()


def test_decomposition_every_cell_of_a_quantity_has_the_same_pieces(monkeypatch):
    # decompose() hands the same piece set to every cell: the rows come from pick(cell, set) restricted to the set
    calls = []
    monkeypatch.setattr(matched, "msd_series", lambda ctx, M, rows, T, years: calls.append(sorted(rows["piece"].unique())) or {"pieces": 0})
    monkeypatch.setattr(matched, "phase_series", lambda ctx, M, rows, years: {"pieces": 0})
    kept = {"1000": np.array([1, 2]), "2760": np.array([2]), "phase": np.array([1])}
    frame = pd.DataFrame({"piece": [0, 1, 1, 2, 3], "solo": range(5)})
    ctx = type("C", (), {"tracks": {"flight": np.arange(5)}})()
    decomposition.decompose(ctx, None, kept, lambda cell, name: frame[frame["piece"].isin(kept[name])])
    assert all(c in ([1, 2], [2]) for c in calls) and len(calls) == 16


# The numbers of the comparisons report: the macros of vs-solo/code/comparisons.py equal the JSON of the analyses.
OPEN_OUT = Path(__file__).parent / "out" / tamsd.LABELS.name / "open_min0.1"
_JSONS = ("raw", "matched_counts", "matched", "decomposition", "prepost_2400")


@pytest.mark.skipif(not all((OPEN_OUT / f"{n}.json").exists() for n in _JSONS), reason="run raw, matched, decomposition and prepost with --open first")
def test_comparisons_macros_equal_the_json(tmp_path, monkeypatch):
    sys.path.insert(0, str(Path(__file__).parent / "vs-solo" / "code"))
    import comparisons

    monkeypatch.setattr(comparisons, "GEN", tmp_path)
    j = {n: json.loads((OPEN_OUT / f"{n}.json").read_text()) for n in _JSONS}
    comparisons.macros.clear()
    comparisons.raw_tex(j["raw"])
    comparisons.counts_tex(j["matched_counts"])
    comparisons.matched_tex(j["matched"])
    comparisons.decomposition_tex(j["decomposition"])
    comparisons.prepost_tex(j["prepost_2400"], j["matched"])
    m, num, signed = comparisons.macros, comparisons.num, comparisons.signed
    g = j["raw"]["datasets"]["1000"]["populations"]["G"]
    assert m["RawFlightsOneG"] == num(g["flights"], 0)
    assert m["RawHKOneG"] == num(g["H_loc(1000 s)"]["value"], 3)
    assert m["RawHKOneGBand"] == f"[{num(g['H_loc(1000 s)']['band'][0], 3)}, {num(g['H_loc(1000 s)']['band'][1], 3)}]"
    s = j["raw"]["phase statistics"]["S"]
    assert m["RawStatClimbS"] == num(s["average first"]["time in climb (%)"]["value"], 1)
    assert m["RawSumSinkS"] == num(s["ratio of sums"]["height lost per km of glide (m/km)"]["value"], 0)
    d = j["matched"]["levels"]["all"]["deltas"]["G - S|G"]["H_loc(1000 s)"]
    assert m["MatDeltaHKG"] == signed(d["value"], 3) and m["MatDeltaHKGBand"].startswith("[")
    assert m["MatMinBeg"] == num(j["matched_counts"]["chosen caliper"]["G pieces on dataset 2760"]["beginners"], 0)
    dec = j["decomposition"]["levels"]["all"]["time in climb (%)"]
    assert m["DecPhiClimbR"] == signed(dec["phi"]["R"]["value"], 1)
    assert m["DecCtlClimbNone"] == num(dec["cells"]["none"]["control"], 1)
    p = j["prepost_2400"]["differences"]["g"]["H_loc(1000 s)"]
    assert m["PreGHK"] == signed(p["value"], 3)
    gap = p["value"] - d["value"]
    assert m["PreVsGapHK"] == signed(gap, 3)
    # the identity behind the decomposition holds in the stored values
    for q, x in j["decomposition"]["levels"]["all"].items():
        total = x["delta_F"]["value"] + sum(x["phi"][k]["value"] for k in "RTW")
        assert total == pytest.approx(x["delta_none"]["value"], abs=1e-9)
    assert 0 < float(m["RawGapCross"].replace("\\,", "")) <= float(m["RawGapSig"].replace("\\,", ""))
    st = j["prepost_2400"]["curves"]["stages"]
    g = 0.5 * ((np.array(st["start"]["H_loc"]) - np.array(st["pre"]["H_loc"])) + (np.array(st["end"]["H_loc"]) - np.array(st["post"]["H_loc"])))
    assert g[-1] > 0 and m["PreGCross"] == num(j["prepost_2400"]["lags"][int(np.flatnonzero(g <= 0)[-1]) + 1], 0)
    cut = comparisons.hloc_cut(j["raw"]["datasets"]["1000"]["populations"]["S"]["curve"], 1000, 1000)  # dataset 1000 stops at 1000 s
    assert cut == pytest.approx(j["raw"]["datasets"]["1000"]["populations"]["S"]["H_loc(1000 s)"]["value"], abs=1e-9)
    assert float(m["RawHKCutTwoS"]) == pytest.approx(comparisons.hloc_cut(j["raw"]["datasets"]["2760"]["populations"]["S"]["curve"], 1000, 1000), abs=5e-4)
    assert float(m["RawHMeanGap"]) <= 0.003 and float(m["RawHlocRangeLow"]) <= float(m["RawHlocRangeHigh"])
    # g against the length of the window
    runs = {w: json.loads((OPEN_OUT / f"prepost_{w}.json").read_text()) for w in comparisons.WINDOWS}
    comparisons.windows_tex(runs)
    for w, tag in comparisons.WINDOWS.items():
        g = runs[w]["differences"]["g"]
        assert m[f"PreWin{tag}HK"] == signed(g["H_loc(1000 s)"]["value"], 3)
        assert m[f"PreWin{tag}Events"] == num(g["events (MSD)"], 0)


def test_phase_median_interpolates_the_survival_in_log_duration():
    sys.path.insert(0, str(Path(__file__).parent / "vs-solo" / "code"))
    import comparisons

    s = {"grid": [100.0, 200.0, 400.0], "value": [0.9, 0.75, 0.25]}
    assert comparisons.phase_median(s) == pytest.approx(200 * 2 ** 0.5)  # halfway, in the log, between 200 and 400
    assert comparisons.phase_median({"grid": [100.0, 200.0], "value": [0.4, 0.1]}) == pytest.approx(100.0)


def test_twoway_variance_is_pilots_plus_blocks_minus_cells_at_least_the_larger_one_way():
    one = lambda sd: sd * np.array([-1.0, 1.0] * 50)  # variance sd ** 2
    assert twoway.variance(np.r_[one(2), one(3), one(1)]) == pytest.approx(4 + 9 - 1)
    assert twoway.variance(np.r_[one(2), one(1), one(3)]) == pytest.approx(4)  # 4 + 1 - 9 < 0: the larger one-way
    band = twoway.band(0.5, np.r_[one(2), one(3), one(1)])
    assert band == pytest.approx([0.5 - twoway.Z * np.sqrt(12), 0.5 + twoway.Z * np.sqrt(12)])


def test_twoway_counts_draw_pilots_blocks_and_cells_in_turn():
    cell, pilot, block = twoway.cell_codes(np.array([0, 0, 1, 1, 2]), np.array([0, 1, 1, 1, 0]))
    assert cell.tolist() == [0, 1, 2, 2, 3] and pilot.tolist() == [0, 0, 1, 2] and block.tolist() == [0, 1, 1, 0]
    c = twoway.counts(pilot, block, np.random.default_rng(0), size=40)
    assert c.shape == (120, 4) and c.dtype == np.uint8
    p, b, k = np.split(c, 3)
    assert (p[:, 0] == p[:, 1]).all() and (b[:, 1] == b[:, 2]).all() and (b[:, 0] == b[:, 3]).all()
    assert (p[:, [0, 2, 3]].sum(axis=1) == 3).all() and (b[:, [0, 1]].sum(axis=1) == 2).all() and (k.sum(axis=1) == 4).all()
    of, n = twoway.weights(cell, c)
    w = of(np.array([4, 0]), np.arange(n))
    assert n == 121 and (w[:, 0] == 1).all() and (w[:, 1:] == c[:, [3, 0]].T).all()


def test_rotation_positions_of_a_chain_are_the_known_passage_times():
    # 1 follows 0 by 30 s, 2 follows 1 by 70 s: passage times 0, 30, 100 s, centred
    x = rotation.positions(3, [(1, 0), (2, 1)], [[30.0, 70.0]])
    assert x[0] == pytest.approx(np.array([0.0, 30.0, 100.0]) - 130 / 3)
    assert x[0].argsort().tolist() == [0, 1, 2]


def test_rotation_positions_of_inconsistent_delays_are_the_least_squares_fit():
    # 1 follows 0 by 30 s, 2 follows 1 by 30 s, but 2 follows 0 by 0 s: x = (-10, 0, 10)
    x = rotation.positions(3, [(1, 0), (2, 1), (2, 0)], [[30.0, 30.0, 0.0]])
    assert x[0] == pytest.approx([-10.0, 0.0, 10.0])


def test_rotation_positions_use_the_finite_edges_and_leave_a_disconnected_instant_unordered():
    delays = [[30.0, 30.0, 0.0], [30.0, 70.0, np.nan], [30.0, np.nan, np.nan]]
    x = rotation.positions(3, [(1, 0), (2, 1), (2, 0)], delays)
    assert x[1] == pytest.approx(np.array([0.0, 30.0, 100.0]) - 130 / 3)  # a chain is exact
    assert np.isnan(x[2]).all()  # member 2 is joined to nobody


def test_rotation_spells_drop_the_short_runs_and_join_the_neighbours_of_one_pilot():
    step = 5
    run = lambda pilot, seconds: [pilot] * (seconds // step)
    heads = np.array(run(0, 100) + run(1, 30) + run(0, 100) + [-1] * 20 + run(2, 80) + run(2, 60))
    pilot, first, length = rotation.spells(heads, step)
    assert pilot.tolist() == [0, 2] and first.tolist() == [0, 66] and length.tolist() == [40, 28]


def test_rotation_kendall_of_a_stable_order_is_one_and_of_an_alternating_one_is_minus_one():
    sign = np.tile([1.0, 1.0, 1.0], (6, 1))
    ordered = np.ones(6, bool)
    assert rotation.kendall(sign, ordered, [1, 3]).tolist() == [[5.0, 5], [3.0, 3]]
    flip = np.array([[1.0], [-1.0], [1.0], [-1.0]])  # an order that alternates
    assert rotation.kendall(flip, np.ones(4, bool), [1])[0, 0] == -3.0


def test_rotation_climbing_marks_the_members_inside_their_climb_runs_on_the_group_grid():
    group = SimpleNamespace(members=(7, 8, 9), t_start=1000, t_end=1050)
    runs = {7: (np.array([1010.0, 1040.0]), np.array([1020.0, 1100.0])), 8: (np.array([900.0]), np.array([1012.0]))}
    c = rotation.climbing(group, 5, runs)  # instants 1000, 1005, ..., 1045; member 9 has no run
    assert c.shape == (10, 3)
    assert c[:, 0].nonzero()[0].tolist() == [2, 3, 8, 9]  # [1010, 1020) and [1040, 1050)
    assert c[:, 1].nonzero()[0].tolist() == [0, 1, 2]  # until 1012: the instant 1010 is in, 1015 out
    assert not c[:, 2].any()
    assert c.mean(axis=1)[2] == pytest.approx(2 / 3) and c.mean(axis=1)[5] == 0


def test_rotation_covered_is_the_union_of_the_clipped_intervals():
    on = rotation.covered(np.array([-3, 2, 4]), np.array([1, 5, 4]), 8)  # [-3, 1), [2, 5), and an empty one
    assert on.tolist() == [True, False, True, True, True, False, False, False]


def test_rotation_climb_events_give_the_start_end_and_member_in_time_order():
    group = SimpleNamespace(members=(7, 8), t_start=1000, t_end=1200)
    runs = {7: (np.array([900.0, 1100.0]), np.array([1050.0, 1163.0])), 8: (np.array([1020.0]), np.array([1080.0]))}
    e, end, who = rotation.climb_events(group, 5, runs)  # the run opened before the group is not a start
    assert e.tolist() == [4, 20] and end.tolist() == [16, 33] and who.tolist() == [1, 0]


def _moving_group(offsets, heights, heading, speed=10.0, steps=200, step=5):
    """Earth-centred positions (n, steps, 3) and altitudes (n, steps) of members at the fixed
    (along, across) offsets in m from a centroid that flies at `speed` m/s towards `heading` (rad
    from east, anticlockwise) from 46 N, 7 E."""
    lat, lon = np.radians(46.0), np.radians(7.0)
    east = np.array([-np.sin(lon), np.cos(lon), 0.0])
    north = np.array([-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat)])
    up = np.array([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])
    u = np.cos(heading) * east + np.sin(heading) * north
    w = -np.sin(heading) * east + np.cos(heading) * north
    c = 6.37e6 * up + speed * step * np.arange(steps)[:, None] * u
    xyz = np.stack([c + a * u + b * w for a, b in offsets])
    return xyz, np.array(heights, float)[:, None] * np.ones(steps)


def test_breathing_geometry_projects_the_offsets_along_and_across_the_route_of_the_centroid():
    xyz, z = _moving_group([(100, 0), (-50, 30), (-50, -30)], [10, 0, -10], heading=np.radians(40))
    h = breathing.ROUTE_S // 10
    l_par, l_perp, l_z = breathing.geometry(xyz, z, h)
    mid = slice(h, 200 - h)
    assert l_par[mid] == pytest.approx(np.sqrt(15000 / 3), rel=1e-2)  # (100^2 + 50^2 + 50^2) / 3
    assert l_perp[mid] == pytest.approx(np.sqrt(1800 / 3), rel=1e-2)
    assert l_z[mid] == pytest.approx(np.sqrt(200 / 3))
    assert np.isnan(l_par[:h]).all() and np.isnan(l_par[200 - h:]).all()  # the route window leaves the group


def test_breathing_geometry_is_nan_where_a_member_has_no_fix_and_the_length_follows_the_route_not_the_axes():
    xyz, z = _moving_group([(60, 0), (-60, 0)], [0, 0], heading=np.radians(-110))
    xyz[1, 100] = np.nan
    l_par, l_perp, _ = breathing.geometry(xyz, z, 60)
    assert np.isnan(l_par[100]) and np.isnan(l_perp[100]) and np.isfinite(l_par[99])
    assert l_par[70] == pytest.approx(60.0, rel=1e-2) and l_perp[70] == pytest.approx(0.0, abs=1.0)


def test_breathing_entries_need_a_glide_of_a_minute_and_stay_below_half_since():
    step = 5
    f = np.r_[np.zeros(20), [0.2] * 4, [0.5] * 6, [0.0] * 5, [0.5] * 3, [0.0] * 14, [1.0] * 3, [0.4] * 2, [0.5] * 3]
    # 100 s at 0, then 0.5 at 24; 25 s at 0 is too short to re-arm; 70 s at 0 again, then 1 at 52; the return to 0.5 is not an entry
    assert breathing.entries(f, step).tolist() == [24, 52]
    assert breathing.entries(np.full(30, 0.5), step).tolist() == []  # never at 0: no entry


def test_breathing_autocovariance_of_an_alternating_series_and_of_a_series_with_gaps():
    y = np.tile([1.0, -1.0], 10)
    a = breathing.autocovariance(y, 3)
    assert (a[:, 0] / a[:, 1]).tolist() == pytest.approx([1.0, -1.0, 1.0, -1.0])
    assert a[1, 1] == 19  # the pairs at lag 1
    y[4:6] = np.nan  # two gaps of opposite sign: the mean stays 0 and the pairs that touch them leave out
    b = breathing.autocovariance(y, 2)
    assert b[1, 1] == 16 and b[0, 1] == 18  # three pairs at lag 1 touch the gaps
    assert breathing.autocovariance(np.array([1.0, 2.0, 3.0]), 5).shape == (6, 2)  # lags longer than the series


def test_breathing_first_zero_of_a_damped_oscillation_and_of_a_series_that_stays_positive():
    lags = np.arange(40) * 0.5
    r = np.exp(-lags / 10) * np.cos(2 * np.pi * lags / 9)
    assert breathing.first_zero(r, lags) == 2.5  # a quarter of the period is 2.25
    assert np.isnan(breathing.first_zero(np.exp(-lags), lags))


def test_breathing_bootstrap_median_follows_the_block_weights_and_the_log_correlation_is_one_for_a_power_law():
    x, k = np.array([1.0, 2.0, 3.0, 10.0]), np.array([0, 1, 2, 3])
    m, d = breathing.boot_median(x, k, np.array([[1.0, 1, 1, 1], [0, 0, 0, 4], [3, 0, 0, 1]]))
    assert m == 2.5 and d.tolist() == [2.0, 10.0, 1.0]
    a = np.exp(np.random.default_rng(1).normal(size=50))
    r, dr = breathing.corr_log(a, 3 * a ** 2, np.arange(50) % 5, np.ones((4, 5)))
    assert r == pytest.approx(1.0) and dr == pytest.approx(np.ones(4))


def _retrace(delays, steps=200, step=5):
    """Earth-centred positions (n, T, 3) of members that retrace one curved path of varying speed,
    member i at the head's position of t - delays[i] s."""
    lat, lon = np.radians(46.0), np.radians(7.0)
    east = np.array([-np.sin(lon), np.cos(lon), 0.0])
    north = np.array([-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat)])
    up = np.array([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])
    path = lambda t: 6.37e6 * up + (8 * t + 2e-3 * t ** 2)[:, None] * east + (0.01 * t ** 2)[:, None] * north
    t = step * np.arange(steps)
    return np.stack([path(t - d) for d in delays])


def test_breathing_follow_reproduces_members_that_retrace_one_path_with_fixed_delays():
    xyz = _retrace([0, 20, 45])
    x = rotation.positions(3, [(1, 0), (2, 1)], np.tile([20.0, 25.0], (200, 1)))  # 1 follows 0 by 20 s, 2 follows 1 by 25 s
    pred = breathing.follow(xyz, x, 5)
    assert np.isnan(pred[1, :4]).all() and np.isfinite(pred[1, 4:]).all()  # t - 20 s precedes the group
    assert np.isnan(pred[2, :9]).all() and np.isfinite(pred[2, 9:]).all()
    assert pred[:, 9:] == pytest.approx(xyz[:, 9:])
    z = np.zeros((3, 200))
    l_par, l_perp, _, p_par, p_perp = breathing.geometry(xyz, z, 60, pred)
    ok = np.isfinite(p_par)
    assert ok.sum() == 80 and (np.flatnonzero(ok) >= 60).all()  # route window inside, and the delays inside
    assert p_par[ok] == pytest.approx(l_par[ok], rel=1e-9) and p_perp[ok] == pytest.approx(l_perp[ok], rel=1e-9)
    assert (l_par[ok] > 10).all()  # not a trivial zero


def test_breathing_follow_interpolates_a_fractional_delay_and_leaves_an_unordered_instant_nan():
    lat, lon = np.radians(46.0), np.radians(7.0)
    east = np.array([-np.sin(lon), np.cos(lon), 0.0])
    up = np.array([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])
    line = lambda t: 6.37e6 * up + 10.0 * t[:, None] * east
    t = 5.0 * np.arange(50)
    xyz = np.stack([line(t), line(t - 17.5)])
    x = rotation.positions(2, [(1, 0)], np.full((50, 1), 17.5))
    x[10] = np.nan
    pred = breathing.follow(xyz, x, 5)
    assert np.isnan(pred[:, 10]).all() and np.isnan(pred[1, :3]).all()
    keep = np.isfinite(pred[1].sum(axis=1))
    assert pred[1][keep] == pytest.approx(xyz[1][keep])  # a straight line is exact under linear interpolation
    assert pred[0][keep] == pytest.approx(xyz[0][keep])
    xyz2 = xyz.copy()
    xyz2[1] = line(t - 40.0)  # the follower falls back: the prediction keeps the 17.5 s delay
    pred2 = breathing.follow(xyz2, x, 5)
    assert np.linalg.norm(pred2[1, 20] - xyz2[1, 20]) == pytest.approx(10 * 22.5)


def _frame():
    """The east, north and up unit vectors at 46 N, 7 E."""
    lat, lon = np.radians(46.0), np.radians(7.0)
    east = np.array([-np.sin(lon), np.cos(lon), 0.0])
    north = np.array([-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat)])
    up = np.array([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])
    return east, north, up


def test_glide_glides_are_the_long_runs_of_no_climber_between_two_climbs():
    f = np.r_[[0.5] * 3, np.zeros(30), [1.0] * 4, np.zeros(10), [0.5] * 2, np.zeros(30)]
    # 150 s between two climbs; 50 s is too short; the last run reaches the end of the group
    assert glide.glides(f, 5) == [(3, 33)]
    assert glide.glides(np.r_[np.zeros(30), [0.5] * 3, np.zeros(30), [0.5]], 5) == [(33, 63)]  # the first run has no start
    assert glide.glides(np.zeros(60), 5) == []


def test_glide_cross_track_offset_is_positive_on_the_left_of_a_straight_head_and_zero_on_its_track():
    east, north, up = _frame()
    head = lambda t: 6.37e6 * up + 10.0 * t[:, None] * east
    t = 5.0 * np.arange(200)
    x = rotation.positions(2, [(1, 0)], np.full((200, 1), 20.0))  # member 1 follows member 0 by 20 s
    for offset in (40.0, -25.0, 0.0):
        xyz = np.stack([head(t), head(t - 20.0) + offset * north])  # north is the left of an eastward head
        d = glide.cross_track(xyz, x, 5)
        assert d[6:, 0] == pytest.approx(0.0) and d[6:, 1] == pytest.approx(offset, abs=1e-6)
        assert np.isnan(d[:6, 1]).all()  # the head's velocity at the passage needs the track VEL_S before it


def test_glide_cross_track_offset_on_a_turning_head_is_the_distance_to_its_track_not_to_a_fixed_route():
    east, north, up = _frame()
    rho, omega = 400.0, 0.025  # a left turn at 10 m/s
    head = lambda t: (6.37e6 * up + rho * np.sin(omega * t)[:, None] * east + rho * (1 - np.cos(omega * t))[:, None] * north)
    inward = lambda t: -np.sin(omega * t)[:, None] * east + np.cos(omega * t)[:, None] * north  # the left of the head
    t = 5.0 * np.arange(200)
    x = rotation.positions(2, [(1, 0)], np.full((200, 1), 20.0))
    on_track = glide.cross_track(np.stack([head(t), head(t - 20.0)]), x, 5)
    assert on_track[6:, 1] == pytest.approx(0.0, abs=0.1)  # 200 m behind along a curve: far from the chord, on the track
    for offset in (30.0, -30.0):
        xyz = np.stack([head(t), head(t - 20.0) + offset * inward(t - 20.0)])
        assert glide.cross_track(xyz, x, 5)[6:, 1] == pytest.approx(offset, abs=0.1)


def test_glide_variance_parts_split_the_changes_of_a_constructed_series():
    a = np.array([1.0, -1, 1, -1, 1, -1, 1, -1])
    b = np.array([1.0, 1, -1, -1, 1, 1, -1, -1])
    build = lambda *inc: np.r_[0.0, np.cumsum(inc[0])] + 0.3 * np.arange(len(inc[0]) + 1) * inc[1]  # drift: removed by the glide mean
    cases = ((a + b, b, [.5, .5, 0, 0, 0, .5, .5, 0]), (2 * a + b, b, [.2, .2, 0, .6, 0, .4, .2, .4]),
             (2 * a, a, [.25, .25, .5, 0, 1, .5, .5, 0]))  # the last: the two factors move together
    for y, c, expect in cases:
        m = glide.variance_parts(build(y, 1), build(a, 1), build(c, 1))
        assert m[0] == 8 and glide.shares(m) == pytest.approx(expect, nan_ok=True)
        assert glide.variance_parts(*(np.r_[0.0, np.cumsum(v)] for v in (y, a, c)))[1:] == pytest.approx(m[1:])
    log_v = np.r_[0.0, np.cumsum(b)]
    log_v[4] = np.nan  # the two changes that touch a missing value leave out
    assert glide.variance_parts(*(np.r_[0.0, np.cumsum(v)] for v in (a + b, a)), log_v)[0] == 6
    assert glide.variance_parts(np.zeros(2), np.zeros(2), np.zeros(2)).tolist() == [0] * 7  # one change: nothing to centre
    assert glide.variance_parts(*(np.r_[0.0, np.cumsum(v)] for v in (a + b, a, b)), lag=2)[0] == 7


def test_glide_one_series_reads_the_instants_of_a_glide_between_two_climbs():
    east, north, up = _frame()
    t = 5.0 * np.arange(400)
    xyz = np.stack([6.37e6 * up + 10.0 * t[:, None] * east, 6.37e6 * up + 10.0 * (t - 20.0)[:, None] * east + 40.0 * north])
    z = np.stack([np.full(400, 2000.0), np.full(400, 2060.0)])
    x = rotation.positions(2, [(1, 0)], np.full((400, 1), 20.0))
    f = np.r_[np.full(100, 0.5), np.zeros(200), np.full(100, 0.5)]
    [(inst, moments, seconds)] = glide.one_series(xyz, z, x, f, 5)
    assert seconds == 1000 and inst.shape == (34, 6)
    assert inst[:, 0] == pytest.approx(np.arange(34) * 0.03) and inst[:3, 1].tolist() == [0, 0.5, 1.0] and inst[-1, 2] == pytest.approx(10 / 60)
    assert inst[:, 3] == pytest.approx(20.0, rel=1e-3) and inst[:, 4] == pytest.approx(30.0)  # L_perp = 40 / 2, L_z = 60 / 2
    assert np.isnan(inst[0, 5]) and inst[1:, 5] == pytest.approx(1600.0, rel=1e-6)  # the first passage is before the glide
    assert moments.shape == (len(glide.DTS), 7) and moments[:, 0].min() > 0


def _zigzag(steps, d=0):
    """The Earth-centred track (T, 3) at 5 s of a head that goes east at 10 m/s and turns at
    random, delayed by d grid steps."""
    east, north, up = _frame()
    heading = np.cumsum(np.random.default_rng(1).normal(0, 0.35, steps + 50))
    v = 10.0 * np.column_stack([np.cos(heading), np.sin(heading)]) * 5.0
    p = np.vstack([np.zeros(2), np.cumsum(v, axis=0)])[25 - d:25 - d + steps]
    return 6.37e6 * up + p[:, :1] * east + p[:, 1:] * north


def _pair_glide(follower, delay_steps):
    """The pair-glide records of a head and a follower between two climbs, the first ahead."""
    t = 200
    xyz = np.stack([_zigzag(t), follower(t)])
    x = rotation.positions(2, [(1, 0)], np.full((t, 1), 5.0 * delay_steps))
    f = np.r_[np.full(20, 0.5), np.zeros(t - 40), np.full(20, 0.5)]
    return headings.one_series(xyz, np.full((2, t), 2000.0), x, f, 5)


def test_headings_a_follower_that_retraces_the_head_peaks_at_the_delay():
    [(l, n, S, tau)] = _pair_glide(lambda t: _zigzag(t, 12), 12)  # 60 s behind on the same track
    assert l == pytest.approx(60) and tau[0] == 60
    pooled = headings.correlation(S.astype(float))
    assert headings.peak(pooled, 5)[0][0] == pytest.approx(60, abs=0.1) and pooled[0].max() > 0.99


def test_headings_a_follower_that_copies_the_heading_at_the_same_time_peaks_at_zero():
    east, north, up = _frame()
    sideways = lambda t: _zigzag(t) + 500.0 * north  # same velocity as the head, 500 m to the side
    [(l, n, S, tau)] = _pair_glide(sideways, 12)  # the delays say that it is 60 s behind
    assert l == pytest.approx(60) and tau[0] == 0
    assert headings.peak(headings.correlation(S.astype(float)), 5)[0][0] == pytest.approx(0, abs=0.1)


def test_headings_the_pooled_correlation_from_the_sums_is_the_pearson_correlation_of_the_centred_series():
    rng = np.random.default_rng(3)
    series = [(rng.normal(size=(2, m)), rng.normal(size=(2, m))) for m in (40, 70, 55)]
    sums = [headings.lagged_sums(x, x * 0.5 + y, 3, 5)[1] for x, y in series]
    pooled = headings.correlation(np.sum(sums, axis=0))
    for lag in (-3, 0, 2):
        a, b = [], []
        for x, y in series:
            y = x * 0.5 + y
            lo, hi = max(0, -lag), min(x.shape[1], x.shape[1] - lag)
            a.append(x[:, lo:hi] - x[:, lo:hi].mean(axis=1, keepdims=True))
            b.append(y[:, lo + lag:hi + lag] - y[:, lo + lag:hi + lag].mean(axis=1, keepdims=True))
        for q in range(2):
            assert pooled[q, lag + 3] == pytest.approx(np.corrcoef(np.concatenate([u[q] for u in a]), np.concatenate([u[q] for u in b]))[0, 1])
    assert headings.lagged_sums(series[0][0][:, :20], series[0][1][:, :20], 3, 5)[0].tolist() == [0] * 7  # 20 instants are 100 s, an overlap of 120 s needs 24


def test_headings_wrapping_keeps_the_angle_in_minus_pi_exclusive_to_pi_inclusive():
    a = np.array([-np.pi, np.pi, 3 * np.pi, -3 * np.pi, 0.1, -0.1 - 2 * np.pi, 2 * np.pi])
    w = headings.wrap(a)
    assert w == pytest.approx([np.pi, np.pi, np.pi, np.pi, 0.1, -0.1, 0.0])
    assert (w > -np.pi).all() and (w <= np.pi).all()


def test_exits_thermals_lie_between_two_glides_of_a_minute_and_keep_short_gaps_inside():
    f = np.r_[np.zeros(12), [0.5] * 4, np.zeros(5), [1.0] * 4, np.zeros(12), [0.5] * 3, np.zeros(6), [0.5] * 2]
    # a gap of 25 s stays inside the thermal; the glide of 60 s is enough; the last stretch has no glide after it
    assert exits.thermals(f, 5) == [(0, 12, 25, 37)]
    assert exits.thermals(np.r_[[0.5] * 3, np.zeros(12), [0.5] * 3, np.zeros(12)], 5) == [(3, 15, 18, 30)]  # no glide before the first stretch
    assert exits.thermals(np.r_[np.zeros(12), [0.5] * 3, np.zeros(11), [0.5]], 5) == []  # 55 s is not a glide
    assert exits.thermals(np.zeros(60), 5) == [] and exits.thermals(np.ones(60), 5) == []


def test_exits_head_over_is_the_most_frequent_head_and_ties_go_to_the_row_nearest_the_thermal():
    x = np.array([[0, 5, 9], [0, 5, 9], [np.nan] * 3, [5, 0, 9], [5, 0, 9]])
    assert exits.head_over(x) == 0 and exits.head_over(x[::-1]) == 1  # 2 against 2: the first row decides
    assert exits.head_over(x[1:]) == 1 and exits.head_over(np.full((4, 3), np.nan)) == -1  # 2 against 1: the majority


def test_exits_exits_read_the_first_entry_and_the_last_exit_of_the_runs_that_overlap_the_thermal():
    t = 1000.0 + 5 * np.arange(40)
    z = 1000.0 + 2 * (t - 1000)
    run = (np.array([1020.0, 1100.0]), np.array([1050.0, 1150.0]))
    s, n, zs, zn, count = exits.exits(t, z, run, 3, 35)
    assert (s, n, zs, zn, count) == (1020, 1150, 1040, 1300, 2)  # altitude interpolated at the run ends
    assert exits.exits(t, z, run, 10, 20)[4] == 0  # [1050, 1100): the run that ends at 1050 is out, the one that starts at 1100 is not in yet
    assert exits.exits(t, z, run, 10, 21) == (1100, 1150, 1200, 1300, 1)  # it is in when 1100 is the last instant
    assert exits.exits(t, z, run, 30, 38)[4] == 0 and np.isnan(exits.exits(t, z, run, 30, 38)[1])  # the run ends at 1150, the instant 30
    z[28:32] = np.nan  # no fix around the end 1150 (instants 1140..1155)
    assert np.isnan(exits.exits(t, z, run, 3, 35)[3]) and exits.exits(t, z, run, 3, 35)[1] == 1150


def _climbs(runs, z, x_before, x_after):
    """A group of three flights 7, 8, 9 over 100 instants of 5 s from 1000, and what observe() would
    give: altitudes z (3, 100) and passage times that are x_before up to instant 20 and x_after from 52."""
    group = SimpleNamespace(members=(7, 8, 9), t_start=1000, t_end=1500, block=3)
    x = np.full((100, 3), np.nan)
    x[:20], x[52:] = x_before, x_after
    p = np.zeros((3, 100, 4))
    p[..., 3] = z
    return group, {m: r for m, r in zip(group.members, runs) if r is not None}, (p, x)


def test_exits_one_group_finds_the_thermal_its_leaders_and_the_exits_of_each_member(monkeypatch):
    t = 1000.0 + 5 * np.arange(100)
    z = np.array([2000.0 + (t - 1000), 1900.0 + (t - 1000), 2100.0 + (t - 1000)])
    runs = [(np.array([1100.0]), np.array([1200.0])), (np.array([1130.0]), np.array([1260.0])), (np.array([1150.0]), np.array([1220.0]))]
    group, run, seen = _climbs(runs, z, [0, 30, 60], [40, 0, 80])
    monkeypatch.setattr(exits, "observe", lambda *a: seen)
    [r] = exits.one_group(group, None, None, 5, run, None, None)
    assert r["block"] == 3 and r["n"] == 3 and r["seconds"] == 160 and (r["before"], r["after"]) == (0, 1)
    assert r["exit_t"].tolist() == [1200, 1260, 1220] and r["entry_t"].tolist() == [1100, 1130, 1150]
    assert r["exit_z"].tolist() == [2200, 2160, 2320] and r["entry_z"].tolist() == [2100, 2030, 2250] and r["runs"].tolist() == [1, 1, 1]
    rows = exits.pairs([r], [0])  # leader: member 7
    assert rows[:, 2].tolist() == [-40, 120] and rows[:, 3].tolist() == [60, 20] and rows[:, 4].tolist() == [100, 100]
    assert rows[:, 5].tolist() == [-70, 150] and rows[:, 6].tolist() == [30, -30] and rows[:, 7].tolist() == [1, 1] and rows[:, :2].tolist() == [[3, 3]] * 2
    assert exits.pairs([r], [1])[:, 2].tolist() == [40, 160]  # leader: member 8
    assert exits.pairs([r], [-1]).shape == (0, 8)
    assert exits.order([r]).tolist() == [[3, 3, 3, 0, 2, 0, 1, 1, 1]]  # first: 7, highest: 9


def test_exits_a_member_without_a_fix_at_its_exit_is_not_a_climber_and_a_leader_that_does_not_climb_gives_no_row(monkeypatch):
    t = 1000.0 + 5 * np.arange(100)
    z = np.array([2000.0 + (t - 1000), 1900.0 + (t - 1000), 2100.0 + (t - 1000)])
    z[1, 50:54] = np.nan
    runs = [(np.array([1100.0]), np.array([1200.0])), (np.array([1130.0]), np.array([1260.0])), None]  # member 9 has no climb run
    group, run, seen = _climbs(runs, z, [0, 30, 60], [40, 0, 80])
    monkeypatch.setattr(exits, "observe", lambda *a: seen)
    [r] = exits.one_group(group, None, None, 5, run, None, None)
    assert exits.climbers(r).tolist() == [0] and exits.pairs([r], [0]).shape == (0, 8)
    assert exits.pairs([r], [1]).shape == (0, 8) and exits.order([r]).shape == (0, 9)


def _thermal(exit_t, exit_z, before=0, after=0, block=0):
    exit_z = np.array(exit_z, float)
    return {"block": block, "n": len(exit_z), "seconds": 300, "before": before, "after": after, "exit_t": np.array(exit_t, float),
            "exit_z": exit_z, "entry_t": np.zeros(len(exit_z)), "entry_z": exit_z - 100, "runs": np.ones(len(exit_z))}


def test_exits_leader_stats_count_the_shares_with_the_tolerance_and_the_median_over_the_gain():
    dz = np.array([-50.0, -10, 0, 15, 30, 60])
    rows = np.column_stack([np.zeros(6), np.full(6, 2), dz, np.arange(6.0) - 2, [100, 100, 50, 200, 200, 200], dz, np.zeros(6), np.ones(6)])
    r = exits.leader_stats(rows, np.ones((5, 1)))
    assert r["rows"] == 6 and r["dz"]["value"] == 7.5 and r["dt"]["value"] == 0.5 and r["follower_later"]["value"] == 0.5
    s = {tol: {k: v["value"] for k, v in r["shares"][str(tol)].items()} for tol in exits.TOLS}
    assert s[20] == {"above": 2 / 6, "same": 3 / 6, "below": 1 / 6}  # 15 is within 20 m, 30 is not
    assert s[10] == {"above": 3 / 6, "same": 2 / 6, "below": 1 / 6}  # |-10| <= 10
    assert s[40] == {"above": 1 / 6, "same": 4 / 6, "below": 1 / 6}
    assert r["dz_over_gain"]["value"] == pytest.approx(0.075) and r["dz_over_gain"]["rows"] == 5  # the gain 50 m is below MIN_GAIN
    with np.errstate(all="ignore"):
        assert exits.leader_stats(np.zeros((0, 8)), np.ones((5, 1)))["rows"] == 0


def test_exits_a_leader_drawn_at_random_gives_a_distribution_of_dz_that_is_symmetric():
    rng = np.random.default_rng(1)
    ths = [_thermal([0, 60], rng.normal(2000, 100, 2)) for _ in range(5000)]
    lead = np.array([rng.choice(exits.climbers(r)) for r in ths])
    rows = exits.pairs(ths, lead)
    assert len(rows) == 5000 and abs(np.median(rows[:, 2])) < 10 and abs((rows[:, 2] > 20).mean() - (rows[:, 2] < -20).mean()) < 0.05
    both = np.r_[exits.pairs(ths, np.zeros(5000, int))[:, 2], exits.pairs(ths, np.ones(5000, int))[:, 2]]
    assert np.median(both) == pytest.approx(0, abs=1) and both.sum() == pytest.approx(0)  # the two orders of a pair are opposite


def test_exits_link_compares_the_highest_and_the_first_exit_with_the_head_after_and_with_chance():
    a = _thermal([1200, 1260, 1220], [2200, 2160, 2320], before=0, after=1)  # first 0, highest 2
    b = _thermal([1000, 1100], [2000, 2100], before=1, after=0)  # first 0, highest 1
    r = exits.link_stats(exits.order([a, b]), np.ones((5, 1)))["all"]
    assert r["thermals"] == 2 and r["with_head_after"] == 2
    assert r["first_is_head_after"]["share"]["value"] == 0.5 and r["highest_is_head_after"]["share"]["value"] == 0
    assert r["first_is_head_after"]["expected_members"] == pytest.approx((1 / 3 + 1 / 2) / 2) == pytest.approx(r["highest_is_head_after"]["expected_climbers"])
    assert r["leader_before"]["climbs"] == 2 and r["leader_before"]["first"]["value"] == 0.5 and r["leader_before"]["highest"]["value"] == 0.5
    assert r["leader_after"]["first"]["value"] == 0.5 and r["leader_after"]["highest"]["value"] == 0 and r["leader_before"]["expected"] == pytest.approx(5 / 12)
    assert exits.link_stats(exits.order([a, b]), np.ones((5, 1)))["2"]["thermals"] == 1


def test_exits_summarize_runs_on_thermals_and_the_figure_is_written(tmp_path):
    pair = lambda i: _thermal([0, 60 + 10 * i], [2000, 2000 + 20 * i - 100], before=0, after=i % 3, block=i % 4)  # the head after is member 2 in a third of them: no climber
    trio = lambda i: _thermal([0, 60, 90 + i], [2000, 2100, 2000 + 20 * i - 100], before=1, after=0, block=i % 4)
    ths = [pair(i) for i in range(40)] + [trio(i) for i in range(20)]
    r = exits.summarize(ths, 4, np.random.default_rng(0))
    assert r["thermals"]["all"]["total"] == 60 and r["thermals"]["3+"]["total"] == 20 and r["thermals"]["2"]["climbers_2plus"] == 40
    assert r["leaders"]["before"]["2"]["rows"] == 40 and r["leaders"]["before"]["3+"]["rows"] == 40  # 1 follower in a pair, 2 in a trio
    assert r["leaders"]["after"]["2"]["rows"] == 27  # 13 of the 40 pairs have a head after that does not climb
    assert r["link"]["all"]["thermals"] == 60 and len(r["hist"]["before"]) == len(exits.DZ_EDGES) - 1 and len(r["dz_by_dt"]["after"]) == len(exits.DT_EDGES) - 1
    exits.plot(r, tmp_path / "exits.pdf")
    assert (tmp_path / "exits.pdf").exists()


def test_network_pit_is_below_every_draw_uniform_under_ties_and_flat_under_the_null():
    rng = np.random.default_rng(0)
    assert (network.pit(np.zeros(100, int), np.ones((100, 50), int), rng) < 1 / 51).all()
    q = network.pit(np.full(20000, 3), np.full((20000, 50), 3), rng)
    assert q.min() >= 0 and q.max() <= 1 and abs(q.mean() - 0.5) < 0.01
    draws = rng.integers(0, 4, (20000, 99))  # k from the same law as the draws: k and the draws are exchangeable
    h = np.histogram(network.pit(rng.integers(0, 4, 20000), draws, rng), np.linspace(0, 1, 11))[0] / 20000
    assert np.abs(h * 10 - 1).max() < 0.1


def test_network_a_pilot_with_one_partner_is_loyal_against_a_null_that_spreads_them():
    days = pd.DataFrame({"date": np.repeat([f"d{i}" for i in range(10)], 2), "pool": 0,
                         "p": [0, 2] * 10, "q": [1, 3] * 10})
    # pilot 0 always meets 1; each day the null redraws a matching of 0, 1, 2, 3
    pilots = pd.Index([0, 1, 2, 3])
    k, s = network.distinct(days, pilots)
    assert (k[0], s[0]) == (1, 10)
    rng = np.random.default_rng(0)
    pools = partners.pools_of(days)
    draws = np.array([network.distinct(partners.null_draw(pools, rng), pilots)[0] for _ in range(200)]).T
    assert network.pit(k[:1], draws[:1], rng)[0] < 0.05
    assert (draws.min(0) >= 1).all() and draws[0].mean() > 2


def test_network_excess_split_sums_to_the_total_excess():
    real = np.array([30, 50, 20])
    draws = np.random.default_rng(1).poisson([20, 45, 25], (100, 3))
    e = network.excess(real, draws)
    assert np.isclose(sum(e["excess"]), real.sum() - draws.mean(0).sum())
    assert np.isclose(sum(e["share"]), 1)


def test_network_symmetrised_correlation_of_assortative_and_shuffled_pairs():
    rng = np.random.default_rng(2)
    a = rng.normal(size=2000)
    cls = rng.integers(0, 3, 2000)
    assert np.isclose(network.mixing(a, a, (cls, cls), 3)["corr"], 1)
    m = network.mixing(a, rng.permutation(a), (cls, rng.permutation(cls)), 3)
    assert abs(m["corr"]) < 0.1
    assert np.isclose(network.mixing(a, a, (cls, cls), 3)["same"], 1)


def test_partners_swaps_keep_each_pilots_partners_without_self_or_double_pairs_and_leave_a_clique():
    rng = np.random.default_rng(0)
    pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (3, 4), (4, 5), (5, 6), (2, 6)]
    out = partners.swaps(pairs, 200, rng)
    assert (np.bincount(np.ravel(out), minlength=7) == np.bincount(np.ravel(pairs), minlength=7)).all()
    assert all(a < b for a, b in out) and len(set(out)) == len(out)
    assert set(out) != set(pairs)
    clique = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    assert set(partners.swaps(clique, 100, rng)) == set(clique)


def _marking(t, x, flight, group=None, grouped=None):
    """Starts (group -1 = solo) or climbing fixes of some flights, at times t and positions x on a line."""
    t = np.atleast_1d(np.array(t, float))
    xyz = np.column_stack([np.atleast_1d(np.array(x, float)), np.zeros(len(t)), np.zeros(len(t))])
    flight = np.broadcast_to(flight, t.shape).copy()
    if group is not None:
        return SimpleNamespace(t=t, xyz=xyz, flight=flight, group=np.broadcast_to(group, t.shape).copy())
    return SimpleNamespace(t=t, xyz=xyz, flight=flight, grouped=np.broadcast_to(grouped, t.shape).copy())


NO_CODES = np.zeros(0, np.int64)


def test_marking_entry_pairs_give_distance_and_delay_of_every_pair_of_climbers():
    xyz = np.array([[0.0, 0, 0], [300, 400, 0], [0, 0, 1000]])
    d, w = marking.entry_pairs(np.array([100.0, 80.0, 130.0]), xyz)
    assert d.tolist() == [500, 1000, np.hypot(300, np.hypot(400, 1000))] and w.tolist() == [20, 30, 50]


def test_marking_a_start_next_to_a_marker_climbing_within_dt_is_marked_and_one_after_dt_is_not():
    start = _marking([1000], [0], 1, group=-1)
    inside = _marking([900, 940, 1000], [50, 50, 50], 2, grouped=False)  # climbing 100, 60 and 0 s before t
    assert marking.marked(start, inside, NO_CODES, 100, 120, [0])[0, :, 0].tolist() == [0, 1, 1]  # solo marker, any
    late = _marking([870], [50], 2, grouped=False)  # 130 s before t: just out of [t - 120, t]
    assert marking.marked(start, late, NO_CODES, 100, 120, [0]).sum() == 0
    far = _marking([960], [150], 2, grouped=False)  # in time, beyond R
    assert marking.marked(start, far, NO_CODES, 100, 120, [0]).sum() == 0
    after = _marking([1010], [50], 2, grouped=False)  # climbing later than t does not mark the start
    assert marking.marked(start, after, NO_CODES, 100, 120, [0]).sum() == 0


def test_marking_a_marker_of_the_group_or_the_pilot_itself_never_marks_and_the_marker_type_is_read_at_its_instant():
    start = _marking([1000, 1000], [0, 0], [1, 5], group=[7, -1])  # flight 1 in group 7 with flight 2; flight 5 solo
    points = _marking([950] * 3, [10] * 3, [1, 2, 3], grouped=[True, True, True])
    codes = np.array([7 * marking.CODE + 1, 7 * marking.CODE + 2], np.int64)
    out = marking.marked(start, points, codes, 100, 120, [0])
    assert out[0, :, 0].tolist() == [1, 0, 1]  # flight 3, of another group, marks; the pilot (1) and its mate (2) do not
    assert out[1, :, 0].tolist() == [1, 0, 1]  # the solo pilot 5 is marked by 1, 2 and 3 alike
    assert marking.marked(start, _marking([950], [10], 1, grouped=[True]), codes, 100, 120, [0])[0].sum() == 0
    mixed = _marking([950, 950], [10, 10], [3, 4], grouped=[True, False])
    assert marking.marked(_marking([1000], [0], 1, group=-1), mixed, NO_CODES, 100, 120, [0])[0, :, 0].tolist() == [1, 1, 1]


def test_marking_the_shift_moves_the_climbing_of_a_marker_out_of_the_window_and_averages_plus_and_minus():
    start = _marking([1000], [0], 1, group=-1)
    marker = _marking([940], [20], 2, grouped=False)  # 60 s before t, window dt = 120
    out = marking.marked(start, marker, NO_CODES, 100, 120, [0, 30, 100, 600])[0, 2]
    # +30 puts the fix at 970 (in), -30 at 910 (in); +100 at 1040 (out), -100 at 840 (out); 600 out both ways
    assert out.tolist() == [1, 1, 0, 0]
    recent = _marking([990], [20], 2, grouped=False)  # 10 s before t: the shift +30 moves it past t, -30 keeps it inside
    assert marking.marked(start, recent, NO_CODES, 100, 120, [30])[0, 2].tolist() == [0.5]


def test_marking_a_marker_shifted_outside_its_flight_marks_nothing():
    start = _marking([1000], [0], 1, group=-1)
    marker = _marking([0, 50, 100], [20] * 3, 2, grouped=False)  # its flight lasts from 0 to 100
    out = marking.marked(start, marker, NO_CODES, 100, 60, [0, 900, 2000, 7200])[0, 2]
    assert out[0] == 0  # not marked in reality: the marker flew long before
    assert out[1] == 0.5  # +900: the fixes move to 900..1000, those of 940..1000 are in the window, -900 is out
    assert out[2] == 0 and out[3] == 0  # shifted past t or before its flight: no fix left


def test_marking_one_date_counts_the_starts_by_kind_and_marks_them_with_the_markers_of_the_date():
    t = 1000 + 5 * np.arange(300)
    grid = pd.concat([pd.DataFrame({"f": f, "t": t, "X": x + 0.0 * t, "Y": 0.0, "Z": 0.0, "z": 1000.0, "ve": 0.0, "vn": 1.0, "state": 0})
                      for f, x in ((0, 0.0), (1, 30.0), (2, 5000.0))], ignore_index=True)
    tracks = marking.contacts.Tracks(grid, 5)
    row_of = {10: 0, 11: 1, 12: 2}
    runs = {10: (np.array([1500.0]), np.array([1600.0])), 11: (np.array([1450.0]), np.array([1550.0])), 12: (np.array([1450.0]), np.array([1550.0]))}
    day = pd.DataFrame({"group_id": [3], "t_start": [1000], "t_end": [2400], "members": [[10, 12]]})
    mine = pd.DataFrame({"flight": [10, 11, 12]})
    n, marks, lost = marking.one_date(day, mine, runs, tracks, row_of, 100, 120, [0], 5)
    assert n.tolist() == [2, 1] and lost.tolist() == [0, 0]  # 10 and 12 in a group at t - 120, 11 solo
    assert marks[0, 2, 0] == 1  # 10 is marked by 11 (30 m away, climbing since 50 s); 12 is 5 km away
    assert marks[1, 2, 0] == 0  # 11 starts at 1450: the others climb from 1500 and 1450 (12 is far)


def test_marking_half_shift_interpolates_where_the_share_falls_to_half_its_real_value():
    s = (0, 2, 5, 10, 20)
    share = np.array([[0.4, 0.4, 0.3, 0.1, 0.0], [0.4, 0.39, 0.38, 0.37, 0.36]])
    out = marking.half_shift(share, s)
    assert np.isclose(out[0], 5 + (0.75 - 0.5) / (0.75 - 0.25) * 5) and np.isnan(out[1])  # the second never reaches 0.2
    assert marking.half_shift(np.array([0.4, 0.2, 0.1, 0.0, 0.0]), (0, 2, 5, 10, 20)) == 2


def test_marking_summarize_takes_the_near_shift_from_dt_and_gives_the_now_ratio_and_the_half_shift():
    s = (0, 2, 5, 10)
    n = np.tile([[100.0, 100.0]], (20, 1))
    curve = np.array([0.4, 0.4, 0.2, 0.1])  # share of the starts marked against s
    marks = np.broadcast_to(n[:, :, None, None] * curve, (20, 2, 3, 4)).copy()
    out = marking.summarize(n, marks, np.zeros(2, int), 152, np.random.default_rng(0), s)
    assert out["S_near_min"] == 5  # 120 s < 152 s <= 300 s
    d = out["summary"]["group"]["any"]
    assert np.isclose(d["real"]["value"], 0.4) and np.isclose(d["near"]["value"], 0.2) and np.isclose(d["now"]["value"], 2)
    assert np.isclose(d["s_half"]["value"], 2 + (0.4 - 0.2) / (0.4 - 0.2) * 3) and out["summary"]["all"]["starts"] == 4000
    assert marking.summarize(n, marks, np.zeros(2, int), 120, np.random.default_rng(0), s)["S_near_min"] == 2  # s >= DT


def test_marking_the_contact_and_the_kind_of_the_pilot_are_read_at_t_minus_dt_and_early_starts_are_dropped():
    t = 1000 + 5 * np.arange(300)
    grid = pd.concat([pd.DataFrame({"f": f, "t": t, "X": x + 0.0 * t, "Y": 0.0, "Z": 0.0, "z": 1000.0, "ve": 0.0, "vn": 1.0, "state": 0})
                      for f, x in ((0, 0.0), (1, 30.0))], ignore_index=True)
    tracks = marking.contacts.Tracks(grid, 5)
    row_of = {10: 0, 11: 1}
    runs = {10: (np.array([1500.0, 1030.0]), np.array([1600.0, 1050.0])), 11: (np.array([1450.0]), np.array([1550.0]))}
    mine = pd.DataFrame({"flight": [10, 11]})

    def run(t_start):
        day = pd.DataFrame({"group_id": [3], "t_start": [t_start], "t_end": [2400], "members": [[10, 11]]})
        return marking.one_date(day, mine, runs, tracks, row_of, 100, 120, [0], 5)

    n, marks, lost = run(1450)  # they meet at 1450, after t - dt = 1380 for the start of 10 at 1500
    assert n.tolist() == [0, 2] and marks[1, 2, 0] == 1  # both solo at t - dt; 11, now the mate of 10, still marks its start (10 does not mark 11: it climbs later)
    n, marks, lost = run(1300)  # already together at 1380
    assert n.tolist() == [2, 0] and marks[0, 2, 0] == 0  # both start as group members and neither marks the other
    assert lost.tolist() == [0, 1]  # the start of 10 at 1030 is within 120 s of the first instant 1000


def encounter_points():
    """Close points of one pair (key 7) at 100-140 s and 300-320 s with a pair interval at [120, 135)."""
    t = np.r_[np.arange(100, 145, 5), np.arange(300, 325, 5)]
    d = np.r_[np.linspace(900, 100, 9), [500, 400, 300, 200, 100]]
    return np.full(len(t), 7), t, d, {7: np.array([[120, 135]])}


def test_encounters_a_pass_is_the_run_of_instants_within_the_radius_without_a_pair_interval():
    key, t, d, iv = encounter_points()
    p = encounters.passes(key, t, d, iv, 5, 10)
    assert p["start"].tolist() == [100, 135, 300] and p["end"].tolist() == [115, 140, 320]  # the interval cuts the first run
    assert p["d"].tolist() == [d[3], d[8], 100.0] and (t[p["i"]] == [115, 140, 320]).all()  # the closest point of each run
    assert encounters.passes(key, t, d, {}, 5, 10)["start"].tolist() == [100, 300]


def test_encounters_runs_of_a_pair_separated_by_less_than_the_gap_are_one_pass():
    key, t, d, iv = encounter_points()
    assert encounters.passes(key, t, d, iv, 5, 15)["start"].tolist() == [100, 135, 300]  # 15 s missing at the interval: not less than 15
    p = encounters.passes(key, t, d, iv, 5, 16)
    assert p["start"].tolist() == [100, 300] and p["end"].tolist() == [140, 320]
    assert encounters.passes(key, t, d, iv, 5, 156)["start"].tolist() == [100]  # 155 s missing before 300
    assert encounters.passes(key, t, d, iv, 5, 155)["start"].tolist() == [100, 300]
    two = encounters.passes(np.r_[key, key + 1], np.r_[t, t], np.r_[d, d], {}, 5, 1000)
    assert two["start"].tolist() == [100, 100]  # never across two pairs


def test_encounters_an_approach_starts_from_afar_with_both_flights_there_at_the_instant_before():
    f, t = np.array([0, 0, 1, 1]), np.array([95, 100, 95, 100])
    xyz = np.array([[0.0, 0, 0], [0, 0, 0], [3000, 0, 0], [100, 0, 0]])
    assert encounters.genuine(xyz, f, t, np.array([1]), np.array([3]), 2000, 5).tolist() == [True]
    xyz[2] = [100, 0, 0]  # already close at 95 s: two gliders that start together
    assert encounters.genuine(xyz, f, t, np.array([1]), np.array([3]), 2000, 5).tolist() == [False]
    f, t = np.array([0, 0, 1]), np.array([95, 100, 100])  # no position of the second at 95 s
    assert encounters.genuine(xyz[:3], f, t, np.array([1]), np.array([2]), 2000, 5).tolist() == [False]


def test_encounters_the_pair_passes_of_two_units_that_overlap_are_one_unit_pass_of_its_closest_pair():
    ua, ub = np.array([3, 3, 3, 4, -6]), np.array([-6, -6, -6, -6, 3])  # unit 3 (a group) and the solo -6, in either order
    start, end = np.array([100, 150, 400, 100, 255]), np.array([200, 260, 450, 120, 300])
    d = np.array([300.0, 100, 50, 10, 200])
    closest, s, e = encounters.merge_units(ua, ub, start, end, d)
    assert closest.tolist() == [1, 2, 3] and s.tolist() == [100, 400, 100] and e.tolist() == [300, 450, 120]
    assert encounters.merge_units(ua[:0], ub[:0], start[:0], end[:0], d[:0])[0].size == 0


def encounter_date(delta=120, intervals=None, groups=None):
    """One date, flights 10 and 11 side by side (100 m apart, 300 m of altitude) on the same heading:
    11 comes from 5000 m behind at 50 m/s until it is level with 10 at 1100 s; the pass is its
    time within 2000 m, from 1065 s."""
    t = np.arange(1000, 1305, 5)
    ya = 10.0 * (t - 1000)
    yb = ya - np.maximum(5000 - 50.0 * (t - 1000), 0)
    g = pd.concat([pd.DataFrame({"f": f, "t": t, "X": x, "Y": y, "Z": 0.0, "z": z}) for f, x, y, z in ((0, 0.0, ya, 1000.0), (1, 100.0, yb, 1300.0))])
    day = pd.DataFrame(groups or {"group_id": [], "t_start": [], "t_end": [], "members": []})
    p = {"step": 5, "gap": 120, "theta_max": 45.0, "delta": delta}
    return encounters.one_date(g.reset_index(drop=True), np.array([10, 11]), intervals or {}, day, 2000.0, p)


def test_encounters_joined_needs_an_interval_during_the_pass_or_within_delta_after_its_end():
    key = 10 * encounters.PAIR + 11
    cut = {key: np.array([[1150, 1400]])}  # the interval cuts the pass at 1145 and covers the rest of the tracks
    out, c = encounter_date(intervals=cut)
    assert out["type"].tolist() == [0] and out["d"].tolist() == [100.0] and np.allclose(out["dz"], 300)
    assert out["joined"].tolist() == [True] and c.tolist()[:4] == [1, 0, 0, 1]  # 1150 is within 120 s of 1145
    assert encounter_date(5, cut)[0]["joined"].tolist() == [True]  # 1150 = 1145 + 5
    assert encounter_date(4, cut)[0]["joined"].tolist() == [False]
    assert encounter_date()[0]["joined"].tolist() == [False]  # never joins
    out, c = encounter_date(intervals={key: np.array([[1150, 1250]])})  # a break shorter than the gap does not end the pass
    assert c.tolist()[:4] == [1, 0, 0, 1] and out["joined"].tolist() == [True]


def test_encounters_an_earlier_interval_between_the_units_makes_the_pass_no_first_approach():
    out, c = encounter_date(intervals={10 * encounters.PAIR + 11: np.array([[1000, 1040]])})
    assert len(out["d"]) == 0 and c[encounters.COUNTS.index("not_first_approach")] == 1


def test_encounters_the_units_are_the_groups_of_the_first_instant_and_any_member_pair_can_join():
    row = {"group_id": [3], "t_start": [900], "t_end": [2000], "members": [[10, 12]]}
    out, c = encounter_date(intervals={11 * encounters.PAIR + 12: np.array([[1200, 1300]])}, groups=row)
    assert out["type"].tolist() == [1] and out["joined"].tolist() == [True]  # 12, a member of 10's group, links to 11
    out, c = encounter_date(groups={**row, "members": [[10, 11]]})
    assert len(out["d"]) == 0 and c[encounters.COUNTS.index("same_unit")] == 1  # two members of one group


def test_encounters_the_heading_comes_from_the_neighbouring_instants_and_the_close_pairs_from_one_instant():
    f, t = np.repeat([0, 1], 3), np.tile([0, 5, 10], 2)
    xyz = np.array([[0, 0, 0], [10, 0, 0], [20, 0, 0], [0, 0, 0], [0, 10, 0], [0, 20, 0]], float)
    u = encounters.heading(xyz, f, t, np.array([1, 4, 0]), 5)
    assert u[:2].tolist() == [[20, 0, 0], [0, 20, 0]] and np.isnan(u[2]).all()  # none at the first instant of a flight
    assert np.allclose(encounters.angle(u[:1], u[1:2]), 90) and np.isnan(encounters.angle(u[:1], np.zeros((1, 3))))
    g = pd.DataFrame({"f": [0, 1, 2, 3], "t": [0, 0, 5, 5], "X": [0.0, 100, 50, 5000], "Y": 0.0, "Z": 0.0})
    a, b, dist = encounters.close_points(g, 200, 5)
    assert list(zip(a, b)) == [(0, 1), ] and np.allclose(dist, 100)  # row 2 is at another instant, row 3 is far


def test_encounters_the_standardisation_weighs_the_bins_with_the_solo_solo_passes():
    n = np.array([[10.0, 10, 10], [5, 15, 0], [1, 1, 1]])  # solo-solo, solo-group, group-group; the third bin has no solo-group
    j = np.array([[1.0, 3, 9], [1, 9, 0], [0, 0, 0]])
    assert np.isclose(encounters.standardised(n, j, 1), (10 * 0.2 + 10 * 0.6) / (10 * 0.1 + 10 * 0.3))  # 2, where the crude ratio is 2.5
    q = {"date": np.zeros(3, int), "type": np.zeros(3, int), "d": np.full(3, 100.0), "dz": np.array([0, 400, 0.0]),
         "joined": np.array([True, True, False]), "n": np.ones(3, int), "m": np.ones(3, int)}
    nk, jk = encounters.tally(q, 1, 800.0, 300.0, np.array([1]))  # 100 m is the second of 8 bins of 100 m
    assert nk[0, 0, :, 1, 0].tolist() == [2, 1] and jk[0, 0, :, 1, 0].tolist() == [1, 1]  # |dz| below / above eps_z
    nk, jk = np.zeros((20, 3, 2, encounters.N_BINS, 1)), np.zeros((20, 3, 2, encounters.N_BINS, 1))
    nk[...] = 100
    jk[:, 0], jk[:, 1], jk[:, 2] = 10, 20, 5
    out = encounters.summarize(nk, jk, np.array([1]), np.random.default_rng(0))["all"]
    assert out["passes"]["solo-solo"] == 20 * 2 * 8 * 100 and np.isclose(out["share"]["solo-group"]["value"], 0.2)
    assert np.isclose(out["ratio"]["solo-group"]["value"], 2) and np.isclose(out["ratio"]["group-group"]["band"][1], 0.5)


def test_encounters_the_expected_joins_are_the_independent_chances_of_the_member_pairs():
    ks = np.array([1, 2, 6])
    nk, jk = np.zeros((3, 2, encounters.N_BINS, 3)), np.zeros((3, 2, encounters.N_BINS, 3))
    nk[0, :, 0, 0], jk[0, :, 0, 0] = 100, 10  # solo-solo: P = 0.1 in the first bin, both |dz| classes
    nk[1, 0, 0, 1], jk[1, 0, 0, 1] = 50, 19  # solo-group of 2: 1 - 0.9^2 = 0.19 expected each, 9.5 in all
    nk[1, 1, 0, 2], jk[1, 1, 0, 2] = 10, 4  # solo-group of 6, |dz| above: 1 - 0.9^6 = 0.4686
    nk[1, 1, 3, 1], jk[1, 1, 3, 1] = 5, 5  # a bin without solo-solo passes: left out of both sums
    ratio, obs, exp = encounters.over_expected(nk, jk, ks, 1, (0, 1), True)
    assert np.isclose(exp, 50 * 0.19 + 10 * (1 - 0.9 ** 6)) and obs == 23 and np.isclose(ratio, 23 / exp)
    assert np.isclose(encounters.over_expected(nk, jk, ks, 1, (0,), False)[0], 2)  # 19 over 9.5 in the |dz| below class
    assert np.isclose(encounters.over_expected(nk, jk, ks, 1, (0, 1), True, ks == 2)[0], 2)  # the groups of 2
    assert np.isclose(encounters.over_expected(nk, jk, ks, 1, (0, 1), True, ks >= 3)[0], 4 / (10 * (1 - 0.9 ** 6)))
    assert np.isclose(encounters.over_expected(nk, jk, ks, 0, (0, 1), True)[0], 1)  # solo-solo against itself


# ---- mock events: the placebo of g

def _mock_world():
    """One real event (flight 1, group stretch 4000-9000 s, windows of 2400 s) and solo flights around it."""
    W = 2400
    events = pd.DataFrame({"episode": [5, 6], "flight": [1, 2], "t_in": [4000.0, 4000.0], "t_out": [9000.0, 9000.0]})
    solo = pd.DataFrame({  # the four windows need [1600, 11400] in one solo stretch
        "flight": [10, 11, 12, 13, 14, 15, 16, 17, 17, 18, 19],
        "a": [0.0, 2000.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 6500.0, 0.0, 1600.0],
        "b": [12000.0, 12000.0, 11000.0, 12000.0, 12000.0, 12000.0, 12000.0, 6000.0, 12000.0, 12000.0, 11400.0]})
    june = _utc("2020-06-15 08:00")[0]
    utc0 = {1: june, 2: june, 10: june, 11: june, 12: june, 13: june, 14: _utc("2020-12-15 08:00")[0], 15: june + 5 * 3600,
            16: _utc("2020-07-15 09:30")[0], 17: june, 18: june, 19: june}
    kind = {f: "B" for f in utc0} | {13: "D", 2: np.nan}
    info = pd.DataFrame({"utc0": pd.Series(utc0), "kind": pd.Series(kind), "site": pd.Series({f: int(f == 18) for f in utc0} | {1: 0})})
    return W, events, solo, info


def test_mock_events_need_the_four_windows_in_one_solo_stretch_and_the_calipers():
    W, events, solo, info = _mock_world()
    rng = np.random.default_rng(0)
    d = prepost.mock_draws(events, solo, info, W, "month2_hour2_class", rng, first_id=100)
    # 11 starts too late, 12 ends too early, 13 has another wing class, 14 is six months away, 15 five hours,
    # 17 has a gap between two stretches; 19 fits exactly; 16 is a month and 1.5 h away, within month 2 and hour 2
    assert sorted(d["flight"]) == [10, 16, 18, 19] and (d["real"] == 5).all() and (d["real_flight"] == 1).all()
    assert sorted(d["rank"]) == [0, 1, 2, 3] and d["id"].tolist() == [100, 101, 102, 103]
    assert (d["t_in"] == 4000).all() and (d["t_out"] == 9000).all()  # flight 2 has no wing level: no mock
    site = prepost.mock_draws(events, solo, info, W, "month2_hour2_class", rng, same_site=True)
    assert sorted(site["flight"]) == [10, 16, 19]  # 18 took off elsewhere
    tight = prepost.mock_draws(events, solo, info, W, "month1_hour1_class", rng)
    assert sorted(tight["flight"]) == [10, 18, 19]  # 16 is 1.5 h away: the hour caliper of 1 h rejects it
    assert sorted(prepost.mock_draws(events, solo, info, W, "month2_hour2_class", rng, pool=2)["flight"]) in ([10, 16], [10, 18], [10, 19], [16, 18], [16, 19], [18, 19])


def test_mock_draws_are_reproducible_and_stay_under_the_pool():
    W, events, solo, info = _mock_world()
    a = prepost.mock_draws(events, solo, info, W, "month2_hour2_class", np.random.default_rng(3), pool=3)
    b = prepost.mock_draws(events, solo, info, W, "month2_hour2_class", np.random.default_rng(3), pool=3)
    assert a.equals(b) and len(a) == 3


def test_mock_windows_sit_at_the_flight_clock_times_of_the_real_event():
    W = 2400
    draws = pd.DataFrame({"real": [5], "real_flight": [1], "flight": [10], "t_in": [4000.0], "t_out": [9000.0], "rank": [0], "id": [77]})
    w = prepost.mock_windows(draws, W)
    assert list(w[["flight", "population", "episode", "a", "b"]].itertuples(index=False, name=None)) == [
        (10, "pre", 77, 1600.0, 4000.0), (10, "start", 77, 4000.0, 6400.0), (10, "end", 77, 6600.0, 9000.0), (10, "post", 77, 9000.0, 11400.0)]
    # the real windows of the same event, as windows() places them
    spans = pd.DataFrame({"flight": [1, 1, 1], "population": ["group_solo", "group", "group_solo"], "episode": [-1, 5, -1],
                          "a": [0.0, 4000.0, 9000.0], "b": [4000.0, 9000.0, 12000.0]})
    real = prepost.windows(spans, np.random.default_rng(0), W)
    real = real[real["population"].isin(prepost.STAGES)]
    assert real.sort_values("a")[["a", "b"]].to_numpy().tolist() == w.sort_values("a")[["a", "b"]].to_numpy().tolist()
    assert prepost.event_times(real)[["flight", "t_in", "t_out"]].to_numpy().tolist() == [[1, 4000.0, 9000.0]]


def test_event_times_keep_the_events_with_the_four_windows():
    wins = pd.DataFrame({"flight": [1] * 4 + [2] * 3, "population": ["pre", "start", "end", "post", "start", "end", "post"],
                         "episode": [1] * 4 + [2] * 3, "a": [0.0, 10.0, 20.0, 30.0, 0.0, 5.0, 8.0], "b": [10.0, 20.0, 30.0, 40.0, 5.0, 8.0, 12.0]})
    e = prepost.event_times(wins)
    assert e[["episode", "flight", "t_in", "t_out"]].to_numpy().tolist() == [[1, 1, 10.0, 30.0]]


def test_valid_mocks_keep_the_first_of_each_real_event_that_have_four_windows():
    draws = pd.DataFrame({"real": [5, 5, 5, 5, 6], "id": [10, 11, 12, 13, 14], "rank": [0, 1, 2, 3, 0]})
    ok = lambda i, stages=prepost.STAGES: [(i, s) for s in stages]
    pieces = [x for i in (10, 12, 13, 14) for x in ok(i)] + ok(11, ("pre", "start"))  # 11 lacks end and post
    p = {"episode": np.array([x[0] for x in pieces]), "population": np.array([x[1] for x in pieces])}
    assert prepost.valid_mocks(draws, p, keep=2)["id"].tolist() == [10, 12, 14]


def test_events_of_a_group_weigh_one_together():
    event = np.array([0, 0, 1, 1, 2, 2])
    group = np.array([7, 7, 7, 7, 8, 8])
    assert prepost.shares(event, group, np.ones(6, bool)).tolist() == [0.5, 0.5, 0.5, 0.5, 1.0, 1.0]
    assert prepost.shares(event, group, event != 1).tolist() == [1.0, 1.0, 0.0, 0.0, 1.0, 1.0]


def test_mock_phase_statistics_weigh_the_mocks_of_one_real_event_as_one_event():
    stats = {"climbs per hour": budget.STATS["climbs per hour"]}
    rows = [(i, s, real, v) for i, real, v in ((10, 7, 1.0), (11, 7, 3.0), (12, 8, 5.0)) for s in prepost.STAGES]
    meta = pd.DataFrame(rows, columns=["episode", "population", "real", "value"]).assign(flight=lambda m: m["episode"], block=0)
    r, q = pd.DataFrame({"climbs per hour": meta["value"] / 3600}), pd.DataFrame({"climbs per hour": np.full(len(meta), 1 / 3600)})
    counts = np.ones((2, 1), np.int32)
    got, events = prepost.phase_on(meta, r, q, counts, prepost.STAGES, stats, group=meta["real"].to_numpy())
    assert events == 3 and got["pre"]["values"][0, 0] == pytest.approx((0.5 * 1 + 0.5 * 3 + 5) / 2)
    plain, _ = prepost.phase_on(meta, r, q, counts, prepost.STAGES, stats)
    assert plain["pre"]["values"][0, 0] == pytest.approx(3.0)
    only, events = prepost.phase_on(meta, r, q, counts, prepost.STAGES, stats, only=(meta["real"] == 8).to_numpy())
    assert events == 1 and only["pre"]["values"][0, 0] == pytest.approx(5.0)


def test_window_job_adds_the_distance_in_the_cross_country_time_to_the_sums_and_the_origins():
    t = np.arange(0, 3001, 10.0)
    fixes = pd.DataFrame({"segment_id": 0, "t": t, "E": 5.0 * t, "N": 0.0})  # 5 m/s east
    spans = pd.DataFrame({"population": ["post"], "episode": [3], "a": [100.0], "b": [2500.0]})
    lags = prepost.window_lags(2400)
    flight, (piece,) = prepost.window_job(9, fixes, spans, lags, 2390)
    k = performance.XC_S // 10
    assert flight == 9 and piece[:2] == ("post", 3) and piece[7][-1] == 240 - k and piece[6][-1] == pytest.approx(5.0 * performance.XC_S * (240 - k))
    assert len(piece[6]) == len(lags) + 1 and piece[6][list(lags).index(1000)] == pytest.approx(5000.0**2 * (240 - 100))
    brief = pd.DataFrame({"population": ["post"], "episode": [3], "a": [100.0], "b": [1100.0]})  # shorter than the displacement
    piece = prepost.window_job(9, fixes, brief, lags, 990)[1][0]
    assert piece[7][-1] == 0 and piece[6][-1] == 0


def _walk(rng, correlation_time):
    """A flight of 14 400 s on the 10-s grid: constant speed, heading with steps of variance 2 dt / the correlation
    time (a function of the flight clock)."""
    t = np.arange(0, 14401, 10.0)
    phi = np.cumsum(np.sqrt(20 / correlation_time(t)) * rng.standard_normal(len(t))) + rng.uniform(0, 2 * np.pi)
    return pd.DataFrame({"segment_id": 0, "t": t, "E": np.cumsum(100 * np.cos(phi)), "N": np.cumsum(100 * np.sin(phi))})


def _toy(flat, mid, group, seed, n_real=400, n_solo=300, size=40, mock_phase=True):
    """Real flights with a group stretch of two windows and solo flights, all on one profile of the persistence along
    the flight clock (correlation time flat + mid (1 - u^2), u = -1 at takeoff and +1 at landing); in the group
    stretch it is multiplied by `group`. Returns prepost.mock_results on the MSD and on a phase rate (+0.5 climbs per
    hour in the start and end windows when group != 1), the lags and the number of mock events."""
    W, lags, T = prepost.OPEN_WINDOW_S, prepost.window_lags(prepost.OPEN_WINDOW_S), 14400
    rng = np.random.default_rng(seed)
    u = lambda t: (t - T / 2) / (T / 2)
    profile = lambda t: flat + mid * (1 - u(t) ** 2)
    fixes, spans = {}, []
    for f in range(n_real):
        t_in = 3600 + rng.uniform(0, 1800)
        t_out = t_in + 2 * W
        fixes[f] = _walk(rng, lambda t: profile(t) * np.where((t >= t_in) & (t < t_out), group, 1.0))
        spans += [(f, "group_solo", -1, 0.0, t_in), (f, "group", f, t_in, t_out), (f, "group_solo", -1, t_out, float(T))]
    for f in range(n_real, n_real + n_solo):
        fixes[f] = _walk(rng, profile)
        spans.append((f, "solo", -1, 0.0, float(T)))
    spans = pd.DataFrame(spans, columns=["flight", "population", "episode", "a", "b"])
    n = n_real + n_solo
    info = pd.DataFrame({"utc0": _utc("2020-06-15 08:00")[0] + rng.uniform(0, 3600, n), "kind": "B", "site": 0}, index=np.arange(n))
    wins = prepost.windows(spans, rng, W)
    wins = wins[wins["population"].isin(prepost.STAGES)]
    events = prepost.event_times(wins)
    draws = prepost.mock_draws(events, spans[spans["population"] == "solo"], info, W, "month2_hour2_class", np.random.default_rng(2), first_id=1000)

    def measure(w):  # tamsd.measure on the synthetic fixes: the pieces of the windows `w`, MSD columns only
        rows = [(f, *x) for f, one in w.groupby("flight") for x in prepost.window_job(f, fixes[f], one, lags, W - tamsd.GRID_S)[1]]
        got = dict(zip(["flight", "population", "episode", "start", "duration", "mid_e", "mid_n", "sums", "origins"],
                       map(np.array, zip(*rows))))
        return got | {"sums": got["sums"][:, :-1], "origins": got["origins"][:, :-1]}

    cell, of_pilot, of_block = twoway.cell_codes(np.arange(n), np.arange(n))  # one resampling cell per flight
    block = pd.Series(cell, index=np.arange(n))
    counts = twoway.counts(of_pilot, of_block, np.random.default_rng(1), size=size)
    p = measure(wins)
    pm = measure(prepost.mock_windows(draws, W))
    kept = prepost.valid_mocks(draws, pm)
    units = decompose.msd_units(p, "episode", None, block)
    m_units, m_group = prepost.mock_units(pm, kept, block)
    # phase: one rate per window, higher by 0.5 in the two group windows when there is a group effect
    stats = {"climbs per hour": budget.STATS["climbs per hour"]}
    pr = np.random.default_rng(4)
    rate = lambda w, inside: pd.DataFrame({"climbs per hour": (pr.uniform(1, 2, len(w)) + 0.5 * inside * (group != 1.0)) / 3600})
    flat_q = lambda w: pd.DataFrame({"climbs per hour": np.full(len(w), 1 / 3600)})
    meta = wins.assign(block=block.reindex(wins["flight"]).to_numpy()).reset_index(drop=True)
    m_wins = prepost.mock_windows(kept, W)
    by_id = kept.set_index("id")
    m_meta = m_wins.assign(block=block.reindex(by_id["real_flight"].reindex(m_wins["episode"])).to_numpy(),
                           real=by_id["real"].reindex(m_wins["episode"]).to_numpy())
    real = SimpleNamespace(units=units, id=pd.factorize(p["episode"])[1][units.stratum], meta=meta,
                           r=rate(meta, meta["population"].isin(("start", "end"))), q=flat_q(meta))
    mock = SimpleNamespace(units=m_units, group=m_group, meta=m_meta, r=rate(m_meta, 0), q=flat_q(m_meta))
    if not mock_phase:  # no mock window has phase time
        mock = SimpleNamespace(units=m_units, group=m_group, meta=m_meta.iloc[:0], r=mock.r.iloc[:0], q=mock.q.iloc[:0])
    return prepost.mock_results(real, mock, stats, counts, lags), lags, len(kept)


def _quoted(entry, key="H_loc(1000 s)"):
    return entry[key]["value"], entry[key]["band"]


def test_a_profile_along_the_flight_gives_g_with_no_group_effect_and_the_mock_events_show_it():
    out, lags, mocks = _toy(flat=100.0, mid=3000.0, group=1.0, seed=0)
    c = out["counts"]
    assert c["real events with the four windows (MSD)"] == 400 and c["real events with a mock event (MSD)"] == 400
    assert c["mock events (MSD)"] == mocks == 2000 and c["solo flights (MSD)"] <= 300
    g, (lo, hi) = _quoted(out["real events with a mock"]["g"])
    gm, _ = _quoted(out["mock"]["g"])
    net, (nlo, nhi) = _quoted(out["real minus mock"]["g"])
    assert lo > 0.005 and g == pytest.approx(0.012, abs=0.004)  # an inverted U of H alone makes g clearly positive
    assert gm == pytest.approx(g, abs=0.004) and abs(net) < 0.004 and nlo < 0 < nhi  # the mock events give the same g
    # the difference is the difference of the draws, and the net effects are those of D_in and D_out
    assert net == pytest.approx(g - gm, abs=1e-12)
    for n in prepost.MOCK_EFFECTS:
        assert out["real minus mock"][n]["H_loc(1000 s)"]["value"] == pytest.approx(
            out["real events with a mock"][n]["H_loc(1000 s)"]["value"] - out["mock"][n]["H_loc(1000 s)"]["value"], abs=1e-12)
    d_in, d_out, dw = (out["mock"][k]["H_loc(1000 s)"]["value"] for k in ("D_in", "D_out", "d w"))
    assert gm == pytest.approx((d_in + d_out) / 2, abs=1e-12) and dw == pytest.approx((d_in - d_out) / 2, abs=1e-12)
    ph = out["real minus mock"]["g"]["climbs per hour"]  # a phase rate with no effect: the difference holds 0
    assert ph["band"][0] < 0 < ph["band"][1]


def test_a_group_effect_on_a_flat_profile_is_in_g_and_not_in_the_mock_events():
    out, lags, _ = _toy(flat=800.0, mid=0.0, group=4.0, seed=0)
    g, (lo, hi) = _quoted(out["real events with a mock"]["g"])
    gm, (mlo, mhi) = _quoted(out["mock"]["g"])
    net, (nlo, nhi) = _quoted(out["real minus mock"]["g"])
    assert lo > 0.05 and g == pytest.approx(0.1, abs=0.03)
    assert abs(gm) < 0.01 and mlo < 0.01 and net == pytest.approx(g - gm, abs=1e-12) and nlo > 0.05
    ph, mph = out["real events with a mock"]["g"]["climbs per hour"], out["mock"]["g"]["climbs per hour"]
    assert ph["value"] == pytest.approx(0.5, abs=0.15) and abs(mph["value"]) < 0.1  # the rate: real events carry the effect


# ---- cross-country speed in the three designs

XC_NAME = "cross-country speed over 30 min (km/h)"


def test_xc_of_a_track_is_the_distance_covered_in_the_displacement_time_summed_over_the_origins():
    k = performance.XC_S // 10
    pos = np.column_stack([5.0 * np.arange(0, 240) * 10, np.zeros(240)])  # 5 m/s
    assert matched.xc_of(pos) == (pytest.approx(5.0 * performance.XC_S * (240 - k)), 240 - k)
    assert matched.xc_of(pos[:k]) == (0.0, 0)  # too short for one origin


def test_raw_cross_country_speed_averages_first_over_the_flights_of_the_pieces_long_enough():
    scale = 3.6 / performance.XC_S
    b = pd.DataFrame({"flight": [1, 2, 3], "t_mid": [500.0, 500.0, 500.0], "s": [100.0, 200.0, 50.0]})
    x = pd.DataFrame({"flight": [1, 2], "t_mid": [1500.0, 1500.0], "xc_m": [450000.0, 720000.0], "xc_origins": [50.0, 40.0]})
    pieces = (np.array([1, 2]), np.array([0.0, 0.0]), np.array([3000.0, 3000.0]))  # flight 3 has no piece long enough
    out = raw.phase_statistics({"b": b, "x": x}, raw.XC_DEFS, pieces, pd.Series([0, 1, 2], index=[1, 2, 3]),
                               np.ones((3, 3), np.int32), (XC_NAME,))
    first, total = out["average first"][XC_NAME]["value"], out["ratio of sums"][XC_NAME]["value"]
    assert out["flights"] == 2 and first == pytest.approx(scale * (450000 / 100 + 720000 / 200) / (50 / 100 + 40 / 200))
    assert total == pytest.approx(scale * 1170000 / 90) and out["draws"].shape == (4, 1)
    assert set(out["average first"]) == {XC_NAME}


def test_matched_cross_country_speed_is_the_ratio_estimator_on_the_pieces_and_their_controls():
    ctx = object.__new__(matched.Context)
    ctx.tracks, ctx.universe, ctx.boot = {"flight": np.array([0, 1, 2, 3])}, np.arange(4), (_ones, 1)
    M = object.__new__(matched.Measured)
    M.piece, M.rows = np.array([0, 1]), np.array([4, 5])  # pieces 0, 1 (flights 0, 1) and one control each (tracks 2, 3)
    M.T_g, M.T_s = np.array([100.0, 200.0]), np.array([100.0, 100.0])
    xc_g, xc_s = np.array([[450000.0, 50], [720000.0, 40]]), np.array([[300000.0, 60], [0.0, 0.0]])
    M.xc_rq_g, M.xc_rq_s = xc_g / M.T_g[:, None], xc_s / M.T_s[:, None]
    sel = pd.DataFrame({"piece": [0, 1], "solo": [2, 3]}, index=[4, 5])
    out = matched.xc_series(ctx, M, sel, years=False)
    scale = 3.6 / performance.XC_S
    assert out["pieces"] == 2 and out["controls"] == 2
    assert out["boot"]["group"][0, 0] == pytest.approx(scale * (4500 + 3600) / (0.5 + 0.2))
    assert out["boot"]["control"][0, 0] == pytest.approx(scale * (3000 + 0) / (0.6 + 0))  # per-unit rates, the zero control counts 0 of both


def test_prepost_window_rates_add_the_cross_country_speed_of_each_window():
    defs = raw.stat_defs()
    rng = np.random.default_rng(3)
    base = {"flight": [1, 1], "t_mid": [5.0, 15.0]}
    b = pd.DataFrame(base | {c: rng.integers(1, 9, 2).astype(float) for st in defs["b"].values() for c in (*st[:2], "s")} | {"s": [2.0, 4.0]})
    p = pd.DataFrame(base | {c: rng.integers(1, 9, 2).astype(float) for st in defs["p"].values() for c in st[:2]})
    wins = pd.DataFrame({"flight": [1, 1], "population": ["pre", "start"], "episode": [0, 0], "a": [0.0, 10.0], "b": [10.0, 20.0]})
    xc = pd.DataFrame({"flight": [1], "population": ["start"], "episode": [0], "xc_m": [300.0], "xc_origins": [10.0]})
    block = pd.Series([0], index=[1])
    meta, r, q, stats = prepost.window_rates({"b": b, "p": p}, wins, block, defs, xc)
    assert list(stats) == [*raw.STATISTICS, XC_NAME] and list(r.columns) == list(stats)
    assert r[XC_NAME].tolist() == [0.0, 300.0 / 4] and q[XC_NAME].tolist() == [0.0, 10.0 / 4]  # a window with no cross-country piece gives 0
    _, r7, _, stats7 = prepost.window_rates({"b": b, "p": p}, wins, block, defs)
    assert list(stats7) == list(raw.STATISTICS) and (r7.to_numpy() == r[list(raw.STATISTICS)].to_numpy()).all()


def test_the_mock_and_cross_country_macros_come_from_the_keys_when_they_are_present():
    sys.path.insert(0, str(Path(__file__).parent / "vs-solo" / "code"))
    import comparisons

    out, _, _ = _toy(flat=100.0, mid=3000.0, group=1.0, seed=0, n_real=60, n_solo=40)
    entry = json.loads(json.dumps(out, default=int))
    comparisons.macros.clear()
    comparisons.mock_tex({"caliper": "month2_hour2_class", "mocks per event": 5, "pool per event": 10,
                          "variants": {"any site": entry, "same site": entry}})
    m, num, signed = comparisons.macros, comparisons.num, comparisons.signed
    k = "H_loc(1000 s)"
    assert m["PreMockGHK"] == signed(entry["mock"]["g"][k]["value"], 3) and m["PreMockSiteGHK"] == m["PreMockGHK"]
    assert m["PreNetGHK"] == signed(entry["real minus mock"]["g"][k]["value"], 3) and m["PreNetGHKBand"].startswith("[")
    assert m["PreMockSubsetGHK"] == signed(entry["real events with a mock"]["g"][k]["value"], 3)
    assert m["PreMockDwHK"] == signed(entry["mock"]["d w"][k]["value"], 3) and "PreMockDwHKBand" in m
    assert m["PreMockEvents"] == num(entry["counts"]["mock events (MSD)"], 0) and m["PreMockFlights"] == num(entry["counts"]["solo flights (MSD)"], 0)
    assert m["PreMockRealEvents"] == num(entry["counts"]["real events with a mock event (MSD)"], 0)
    assert not [x for x in m if x.endswith("Xc")]  # no cross-country statistic in these results: no macro
    entry["mock"]["g"][comparisons.XC] = entry["mock"]["g"][k]
    comparisons.mock_tex({"caliper": "month2_hour2_class", "mocks per event": 5, "pool per event": 10, "variants": {"any site": entry}})
    assert m["PreMockGXc"] == signed(entry["mock"]["g"][k]["value"], 1)



def test_mock_results_leave_out_the_phase_statistics_when_no_mock_window_has_phase_time():
    out, lags, _ = _toy(flat=100.0, mid=3000.0, group=1.0, seed=0, n_real=40, n_solo=30, mock_phase=False)
    c = out["counts"]
    assert c["mock events (MSD)"] > 0 and c["mock events (phase)"] == 0 and c["real events with a mock event (phase)"] == 0
    for key in ("mock", "real events with a mock", "real minus mock"):
        g = out[key]["g"]
        assert "H_loc(1000 s)" in g and "climbs per hour" not in g and g["events (phase)"] == 0


# The macros of the letter (letter/code/macros.py): alpha = 2 H_loc, read from the JSON.
def _letter_macros():
    import importlib.util

    spec = importlib.util.spec_from_file_location("letter_macros", Path(__file__).parent / "letter" / "code" / "macros.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_letter_alpha_macros_double_the_value_and_the_band_and_a_lag_off_the_curve_has_none():
    lm = _letter_macros()
    out = {}
    lm.put(out, "A", {"value": 0.46, "band": [0.45, 0.48]}, 3, 2)
    assert out == {"A": "0.920", "ABand": "[0.900, 0.960]"}
    lm.put(out, "B", {"value": -0.0105, "band": [-0.02, 0.005]}, 3, 2, True)
    assert out["B"] == "$-$0.021" and out["BBand"] == "[$-$0.040, $+$0.010]"
    curve = {"lags": [500, 1000], "value": [0.1, 0.2], "band": [[0.0, 0.1], [0.2, 0.3]]}
    assert lm.at_lag(curve, 1000) == {"value": 0.2, "band": [0.1, 0.3]} and lm.at_lag(curve, 1840) is None


def test_letter_stays_above_gives_the_lag_after_the_last_value_at_or_below_zero():
    lm = _letter_macros()
    assert lm.stays_above([10, 20, 30, 40], [0.1, -0.1, 0.2, 0.3]) == 30
    assert lm.stays_above([10, 20], [0.1, 0.2]) == 10 and lm.stays_above([10, 20], [0.1, -0.2]) is None


def test_letter_flight_macros_take_the_lags_the_window_measures_and_the_mock_when_it_is_there(tmp_path):
    lm = _letter_macros()
    q = lambda v: {"value": v, "band": [v - 0.01, v + 0.01]}
    effect = lambda v: {"events (MSD)": 10, "flights (MSD)": 9, "Delta H": q(v), "dH_loc": {"value": [v, 2 * v], "band": [[v - 0.01, 2 * v - 0.01], [v + 0.01, 2 * v + 0.01]]}}
    result = {"lags": [870, 1000], "differences": {"g": effect(0.01), "d w": effect(0.0), "post - pre": effect(0.02)}}
    (tmp_path / "prepost_1260.json").write_text(json.dumps(result))
    out = lm.flight_macros(tmp_path)
    assert out["LtFltShortGAlphaK"] == "$+$0.040" and out["LtFltShortGFit"] == "$+$0.010" and "LtFltShortGAlphaQ" not in out
    assert out["LtFltShortPostPreAlphaK"] == "$+$0.080" and out["LtFltShortEvents"] == "10"
    assert not [k for k in out if "Mock" in k or "Net" in k or k.endswith("Xc")]
    counts = {"mock events (MSD)": 50, "solo flights (MSD)": 40, "real events with a mock event (MSD)": 8}
    result["mock"] = {"variants": {"any site": {"counts": counts, "mock": {"g": effect(0.0)}, "real minus mock": {"g": effect(0.01)}}}}
    result["differences"]["g"][lm.XC] = q(1.4)
    (tmp_path / "prepost_1260.json").write_text(json.dumps(result))
    out = lm.flight_macros(tmp_path)
    assert out["LtFltShortNetGAlphaK"] == out["LtFltShortGAlphaK"] and out["LtFltShortMockEvents"] == "50"
    assert out["LtFltShortGXc"] == "$+$1.4" and out["LtFltShortMockReal"] == "8"


def test_matched_difference_curves_subtract_the_draws_of_the_two_sides_on_shared_lags():
    def side(delta, lags=(500, 1000)):
        return {"main": {"lags": np.array(lags), "boot": {"delta": {"H_loc": np.array(delta)}}}, "long": None}

    g = side([[0.03, 0.05], [0.02, 0.04], [0.04, 0.06], [0.03, 0.05]])  # the point estimate, then three draws
    a = side([[0.01, 0.01], [0.01, 0.00], [0.01, 0.02], [0.01, 0.01]])
    got = matched.difference_curves(g, a)
    assert list(got) == ["main"] and got["main"]["lags"] == [500, 1000]
    assert got["main"]["value"] == pytest.approx([0.02, 0.04])
    assert matched.difference_curves(g, side([[0.0, 0.0]], lags=(500, 2000))) == {}
