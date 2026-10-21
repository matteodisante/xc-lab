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
import performance  # noqa: E402
import phase_msd  # noqa: E402
import survival  # noqa: E402
import tamsd  # noqa: E402
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
    pop, episode, duration, sums, origins = found[0]
    assert (pop, episode, duration) == ("group", 3, 2990)
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


def test_performance_reads_runs_at_their_ends_and_flies_30_min_straight():
    t = np.arange(0, 4001, 1.0)
    z = np.where(t < 600, 1000 + 2 * t, 2200 - (t - 600))  # climb at 2 m/s, then lose 1 m/s
    fixes = pd.DataFrame({"segment_id": 0, "t": t, "E": 10 * t, "N": 0.0, "z": z})
    runs = pd.DataFrame({"segment_id": 0, "t_start": [0.0, 600.0], "t_end": [599.0, 4000.0]})
    spans = pd.DataFrame({"flight": [1], "population": ["solo"], "episode": [-1], "a": [0.0], "b": [4001.0]})
    dz, chord, xc = performance.flight_terms(fixes, runs, spans)
    assert dz.tolist() == [1198.0, -3400.0] and chord[1] == 34000
    assert xc == [("solo", 18000.0 * 221, 221)]  # 401 grid points, 180 steps of 10 s
    runs = pd.DataFrame({"flight": 1, "population": "solo", "phase": ["climb", "transition", "climb"],
                         "t_start": [0.0, 600, 4100], "t_end": [599.0, 4000, 4200]})
    xc = pd.DataFrame(xc, columns=["population", "xc_m", "xc_origins"]).assign(flight=1)
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
