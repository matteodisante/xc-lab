import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import powerlaw
import pyarrow.parquet as pq
import pytest

_spec = importlib.util.spec_from_file_location("durations", Path(__file__).with_name("durations.py"))
durations = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(durations)

on_disk = pytest.mark.skipif(not durations.DISK.exists(), reason="SSD_DISANTE is not mounted")


def _runs(*flights):
    rows = [(f, 0, float(k), p, n) for f, phases in flights for k, (p, n) in enumerate(phases)]
    return pd.DataFrame(rows, columns=["flight_id", "segment_id", "t_start", "phase", "n_fixes"])


def test_search_counts_only_before_a_climb():
    runs = _runs(
        ("a", [("search", 10), ("climb", 100), ("search", 20), ("transition", 300), ("search", 30)]),
        ("b", [("search", 40), ("climb", 200)]),
    )
    w = durations.waiting_times(runs.sample(frac=1, random_state=0))
    assert sorted(w["search"]) == [10, 40]
    assert sorted(w["climb"]) == [100, 200]
    assert list(w["transition"]) == [300]


def test_cohort_keeps_its_segments_after_the_search_rule():
    runs = _runs(("a", [("transition", 300), ("search", 10), ("climb", 100)]))
    runs.loc[2, "segment_id"] = 1  # the climb that confirms the search lies outside the cohort
    w = durations.waiting_times(runs, {("a", 0)})
    assert list(w["search"]) == [10] and list(w["transition"]) == [300] and len(w["climb"]) == 0


def test_trim_drops_the_extreme_values():
    x = np.r_[np.ones(10), np.arange(2.0, 100.0), np.full(10, 500.0)]
    t = durations.trim(x)
    assert t.min() == 2 and t.max() == 99


def test_fit_recovers_a_pareto_exponent():
    rng = np.random.default_rng(1)
    body = rng.uniform(1, 100, 3000)
    tail = 100 * rng.uniform(size=3000) ** (-1 / 2.5)
    f = durations.fit(np.r_[body, tail])
    assert f["mu"] == pytest.approx(2.5, abs=0.15)
    assert 80 < f["xmin"] < 130


def test_hill_is_the_powerlaw_estimate_at_fixed_xmin():
    x = np.random.default_rng(2).pareto(2.0, 2000) * 10 + 10
    f = powerlaw.Fit(x, xmin=20.0, verbose=0)
    assert durations.hill(x, 20.0) == pytest.approx(f.power_law.alpha - 1, rel=1e-9)


def test_log_resample_spans_the_survival():
    x, y = durations.ccdf(np.arange(1.0, 1001.0))
    rx, ry = durations.log_resample(x, y)
    assert len(rx) == 30 and rx[0] == pytest.approx(1) and rx[-1] == pytest.approx(999)
    assert ry[0] == pytest.approx(0.999)


# The README table: (mu, xmin in s, runs at or above xmin).
QUOTED = {
    "all": {
        "para": {"transition": (4.32, 1851, 8482), "search": (4.27, 648, 1772), "climb": (10.66, 1161, 707)},
        "hang": {"transition": (4.34, 1010, 823), "search": (3.29, 192, 714), "climb": (5.45, 608, 1087)},
    },
    "C_10000": {
        "para": {"transition": (4.14, 1648, 8119), "search": (3.84, 478, 2478), "climb": (9.08, 1032, 966)},
        "hang": {"transition": (3.38, 740, 1460), "search": (4.02, 243, 125), "climb": (5.92, 637, 529)},
    },
}


@on_disk
@pytest.mark.parametrize("discipline, members, labelled", [("para", 46273, 33450), ("hang", 2326, 690)])
def test_cohort_flights(discipline, members, labelled):
    segments = durations.cohort_segments(discipline)
    assert len({f for f, _ in segments}) == members
    runs = pd.read_parquet(durations.RUNS[discipline], columns=["flight_id", "segment_id"])
    assert runs.loc[durations.in_segments(runs, segments), "flight_id"].nunique() == labelled


@on_disk
@pytest.mark.parametrize("population", ["all", "C_10000"])
@pytest.mark.parametrize("discipline", ["para", "hang"])
def test_quoted_exponents(population, discipline):
    # mu is checked at the quoted xmin; the xmin search itself is rerun only for hang, since
    # on the 2 million paraglider runs it takes minutes.
    segments = durations.cohort_segments(discipline) if population == "C_10000" else None
    times = durations.waiting_times(pd.read_parquet(durations.RUNS[discipline]), segments)
    for phase, (mu, xmin, n_tail) in QUOTED[population][discipline].items():
        x = durations.trim(times[phase])
        assert durations.hill(x, xmin) == pytest.approx(mu, abs=0.005)
        assert (x >= xmin).sum() == n_tail
        if discipline == "hang":
            assert durations.fit(x)["xmin"] == xmin


@on_disk
def test_stored_runs_match_the_legacy_segmenter():
    # Hang flight 3336 lies wholly inside row group 1 of the cleaned fixes.
    from common.legacy import UNCLASSIFIED, segment

    root = durations.DISK / "hang_gliders/delta_cfd_igc/derived"
    columns = ["source", "flight_id", "segment_id", "t", "E", "N", "z"]
    fixes = pq.ParquetFile(root / "fixes.parquet").read_row_group(1, columns=columns).to_pandas()
    fixes = fixes[fixes["flight_id"] == "3336"]
    labelled = segment(fixes, "hang gliders").fixes
    labelled = labelled[labelled["phase"] != UNCLASSIFIED]
    mine = labelled.groupby("phase_run").agg(
        phase=("phase", "first"), t_start=("t", "first"), t_end=("t", "last"), n_fixes=("t", "size"))

    stored = pd.read_parquet(durations.RUNS["hang"], filters=[("flight_id", "==", "3336")])
    stored = stored.sort_values("t_start")[list(mine.columns)]
    assert len(stored) == 59
    pd.testing.assert_frame_equal(mine.reset_index(drop=True), stored.reset_index(drop=True),
                                  check_dtype=False)
