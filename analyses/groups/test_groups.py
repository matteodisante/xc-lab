import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Imported by name, not from their files: the worker processes re-import them.
sys.path.insert(0, str(Path(__file__).parent))
import blocks  # noqa: E402
import budget  # noqa: E402
import decompose  # noqa: E402
import mechanisms  # noqa: E402
import prepost  # noqa: E402
import performance  # noqa: E402
import phase_msd  # noqa: E402
import redundancy  # noqa: E402
import size  # noqa: E402
import survival  # noqa: E402
import tamsd  # noqa: E402
import zones as zonelib  # noqa: E402
from lab.msd import tamsd as lab_tamsd  # noqa: E402


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
    membership = pd.DataFrame(rows, columns=["flight", "t_start", "t_end", "state", "conf_id", "episode_id"])
    summary = pd.DataFrame({"flight": [0, 1, 2], "n_entries": [0, 1, 0], "block": [0, 1, 2]})
    return flights, membership, summary


def test_stretches_split_the_populations_on_the_flight_clock():
    s = tamsd.stretches(*labels())
    assert list(s.itertuples(index=False, name=None)) == [
        (0, "solo", -1, 100.0, 8000.0),
        (1, "group_solo", -1, 100.0, 1000.0),
        (1, "group", 7, 1000.0, 4000.0),  # two configurations of one episode are one stretch
        (1, "group_solo", -1, 4000.0, 7000.0),
    ]


def test_stretches_by_size_end_where_the_configuration_changes():
    flights, membership, summary = labels()
    membership["n"] = [1, 1, 1, 1, 2, 5, 1, 1]  # flight 1: 2 pilots, then 5
    s = tamsd.stretches(flights, membership, summary, by_size=True)
    assert list(s.itertuples(index=False, name=None))[1:] == [
        (1, "group_solo", -1, 100.0, 1000.0),
        (1, "n2", 7, 1000.0, 2000.0),  # the two configurations of one episode are two stretches
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
    configurations = pd.DataFrame({"conf_id": [0, 1], "episode_id": [0, 0], "t_start": [1000, 3000], "t_end": [2000, 4000],
                                   "n": [2, 2], "members": [[5, 9], [5, 9]]})
    pairs = pd.DataFrame({"a": [5, 5], "b": [9, 9], "t_on": [900, 2900], "t_off": [2100, 4100], "ell_med": [30.0, 5.0]})
    flights = pd.DataFrame({"flight": [5, 9], "utc0": [0.0, 500.0]})
    r = mechanisms.roles(configurations, pairs, flights).sort_values(["conf", "flight"])
    assert list(r[["conf", "flight", "population"]].itertuples(index=False, name=None)) == [
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
# date, discipline, flights, largest configuration, duration (h), members in the MSD, H, band
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
