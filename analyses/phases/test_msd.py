import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Imported by name, not from its file: the worker processes of msd.compute re-import it.
sys.path.insert(0, str(Path(__file__).parent))
import msd  # noqa: E402


def test_fft_msd_is_the_direct_average():
    pos = np.cumsum(np.random.default_rng(0).normal(size=(400, 2)), axis=0) + [3e5, -2e5]
    lags = np.array([1, 2, 7, 50, 399])
    direct = [((pos[d:] - pos[:-d]) ** 2).sum(axis=1).mean() for d in lags]
    assert msd.msd(pos, lags) == pytest.approx(direct, rel=1e-9)


def test_runs_keep_his_selection():
    phases = [("search", 200), ("transition", 300), ("search", 250), ("climb", 400),
              ("search", 190), ("search", 300), ("transition", 500), ("climb", 100), ("search", 220)]
    runs = pd.DataFrame({"flight_id": "a", "segment_id": 0, "t_start": np.arange(len(phases)) * 1000.0,
                         "phase": [p for p, _ in phases], "n_fixes": [n for _, n in phases]})
    kept = msd.msd_runs(runs.sample(frac=1, random_state=0))
    # Searches before a transition go, also across a second search run; a final search stays;
    # runs of at most 180 fixes go.
    assert list(kept["t_start"]) == [1000, 2000, 3000, 6000, 8000]


def test_weighted_median_is_his():
    v = np.array([1.0, 2.0, 3.0])
    assert msd.weighted_median(v, np.array([1.0, 1.0, 3.0])) == 3
    assert msd.weighted_median(v, np.array([3.0, 1.0, 1.0])) == 1
    assert msd.weighted_median(v, np.array([1.0, 1.0, 2.0])) == 2


SAVED = msd.OUT / "msd.json"
saved = pytest.mark.skipif(not SAVED.exists(), reason="run msd.py first")
on_disk = pytest.mark.skipif(not msd.DISK.exists(), reason="SSD_DISANTE is not mounted")

# The README numbers: sqrt MSD(1 s) in m/s and the local exponent at 100 s, per phase;
# the smallest exponent of transitions over the plot and of climbs, at 17 s.
QUOTED = {
    "all": {
        "para": {"speed": (10.4, 8.6, 9.4), "at_100": (1.96, 0.70, 1.59), "transition_min": 1.89, "climb_min": 0.45},
        "hang": {"speed": (17.6, 13.5, 14.8), "at_100": (1.95, 0.69, 1.44), "transition_min": 1.85, "climb_min": 0.15},
    },
    "C_10000": {
        "para": {"speed": (10.8, 8.7, 9.5), "at_100": (1.97, 0.74, 1.57), "transition_min": 1.91, "climb_min": 0.47},
        "hang": {"speed": (18.6, 13.8, 15.2), "at_100": (1.96, 0.72, 1.47), "transition_min": 1.87, "climb_min": 0.15},
    },
}


@saved
@pytest.mark.parametrize("population", ["all", "C_10000"])
@pytest.mark.parametrize("discipline", ["para", "hang"])
def test_quoted_numbers(population, discipline):
    r, q = json.loads(SAVED.read_text())[population][discipline], QUOTED[population][discipline]
    for phase, speed, at_100 in zip(msd.PHASES, q["speed"], q["at_100"]):
        assert np.sqrt(r[phase]["msd"][0]) == pytest.approx(speed, abs=0.05)
        d, s = msd.slopes(r[phase])
        assert np.interp(np.log(100), np.log(d), s) == pytest.approx(at_100, abs=0.005)
    assert msd.slopes(r["transition"])[1].min() == pytest.approx(q["transition_min"], abs=0.005)
    d, s = msd.slopes(r["climb"])
    assert s.min() == pytest.approx(q["climb_min"], abs=0.005) and round(d[s.argmin()]) == 17


@saved
@on_disk
def test_saved_hang_result_is_reproduced():
    # The paraglider pass takes minutes; the hang one, seconds.
    fresh, stored = msd.compute("hang"), json.loads(SAVED.read_text())
    for pop in msd.POPULATIONS:
        for phase in msd.PHASES:
            assert fresh[pop][phase]["runs"] == stored[pop]["hang"][phase]["runs"]
            assert fresh[pop][phase]["msd"] == pytest.approx(stored[pop]["hang"][phase]["msd"], rel=1e-12)


def test_straight_glide_is_ballistic():
    t = np.arange(1000.0)
    fixes = pd.DataFrame({"flight_id": "a", "segment_id": 0, "t": t, "E": 10 * t, "N": 0.0})
    runs = pd.DataFrame({"flight_id": "a", "segment_id": 0, "t_start": [0.0, 100.0], "t_end": [99.0, 699.0],
                         "n_fixes": [100, 600], "phase": "transition"})
    out = msd.flight_msds(fixes, runs)
    assert np.isnan(out[0]).all()  # touches the unsmoothed start
    lags = msd.DELTAS < 600
    assert out[1, lags] == pytest.approx(100.0 * msd.DELTAS[lags] ** 2, rel=1e-9)
    assert np.isnan(out[1, ~lags]).all()
