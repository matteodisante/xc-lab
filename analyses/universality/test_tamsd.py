import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent))
import tamsd  # noqa: E402
from synthetic import fbm_fixes  # noqa: E402

LAGS = tamsd.LAGS  # 10 s to 1000 s


def test_lags():
    assert LAGS[0] == 10 and LAGS[-1] == 1000 and tamsd.LONG_LAGS[0] == 10 and tamsd.LONG_LAGS[-1] > 20000
    assert np.all(np.diff(tamsd.LONG_LAGS) > 0) and np.all(tamsd.LONG_LAGS % 10 == 0)


def test_power_law_gives_h_and_alpha_exactly():
    m = 3.0 * LAGS**1.7  # MSD of H = 0.85
    assert tamsd.fit_h(LAGS, m) == pytest.approx(0.85)
    assert tamsd.local_h(LAGS, m) == pytest.approx(np.full(len(LAGS), 0.85))
    assert tamsd.alpha(LAGS, m) == pytest.approx(np.full(len(LAGS), 1.7))


def test_fit_and_local_h_match_a_polyfit_and_accept_draws():
    rng = np.random.default_rng(1)
    m = LAGS**1.6 * np.exp(rng.normal(0, 0.05, (4, len(LAGS))))
    for row in m:
        assert tamsd.fit_h(LAGS, row) == pytest.approx(np.polyfit(np.log10(LAGS), np.log10(row), 1)[0] / 2)
    assert tamsd.fit_h(LAGS, m).shape == (4,) and tamsd.local_h(LAGS, m).shape == (4, len(LAGS))
    near = np.abs(np.log10(LAGS) - np.log10(100)) <= 0.25 + 1e-9
    i = list(LAGS).index(100)
    assert tamsd.local_h(LAGS, m[0])[i] == pytest.approx(np.polyfit(np.log10(LAGS[near]), np.log10(m[0][near]), 1)[0] / 2)


def test_straight_flight_msd_is_distance_squared():
    t = np.arange(200) * 10.0
    fixes = pd.DataFrame({"segment_id": 0, "t": t, "E": 5.0 * t, "N": 0 * t})
    (start, duration, sums, origins), = tamsd.pieces(fixes, 0, 1e9)
    j = list(tamsd.LONG_LAGS).index(100)
    assert start == 0 and duration == 1990
    assert sums[j] / origins[j] == pytest.approx((5.0 * 100) ** 2)
    assert np.isnan(sums[tamsd.LONG_LAGS // 10 >= 200]).all() and (origins[tamsd.LONG_LAGS // 10 >= 200] == 0).all()


@pytest.mark.parametrize("h", [0.35, 0.7, 0.9])
def test_fbm_with_known_h_gives_h_back(h):
    rng = np.random.default_rng(7)
    n = 800
    flights = [tamsd.pieces(fbm_fixes(n, h, rng), 0, 1e9) for _ in range(30)]
    owner = np.arange(len(flights))
    msd = tamsd.msd_matrix(owner, np.array([p[0][1] for p in flights]), np.array([p[0][2] for p in flights]),
                           np.array([p[0][3] for p in flights]), len(flights), tamsd.LONG_LAGS / tamsd.SUPPORT)
    k = tamsd.LONG_LAGS <= 1000
    assert tamsd.fit_h(tamsd.LONG_LAGS[k], np.nanmean(msd[:, k], axis=0)) == pytest.approx(h, abs=0.03)


def test_grids_split_segments_clip_to_the_span_and_drop_short_pieces():
    t = np.r_[np.arange(0, 600, 10.0), np.arange(1000, 1100, 10.0), np.arange(2000, 2015, 5.0)]
    seg = np.r_[np.zeros(60), np.ones(10), np.full(3, 2)]
    fixes = pd.DataFrame({"segment_id": seg, "t": t, "E": t, "N": 0 * t})
    found = list(tamsd.grids(fixes, 0, 1e9))
    assert [(g[0], len(g[1])) for g in found] == [(0, 60), (1000, 10)]  # the 10 s piece is too short
    clipped = list(tamsd.grids(fixes, 100, 400))
    assert len(clipped) == 1 and clipped[0][0] == 100 and len(clipped[0][1]) == 30  # 100 .. 390 s
    assert np.allclose(clipped[0][1][:, 0], np.arange(100, 400, 10))


def test_fixed_population_keeps_the_same_flights_and_all_flights_lose_them():
    lags = np.array([10, 100, 300, 500, 1000, 1700])
    n = 4
    duration = np.array([100.0, 300.0, 600.0, 2000.0])  # one piece per flight
    sums, origins = np.ones((n, 6)), np.ones((n, 6)) * 5
    fixed = tamsd.msd_matrix(np.arange(n), duration, sums, origins, n, np.full(6, 250.0))
    assert (np.isfinite(fixed) == np.isfinite(fixed[:, :1])).all() and np.isfinite(fixed[:, 0]).sum() == 3
    allf = tamsd.msd_matrix(np.arange(n), duration, sums, origins, n, lags / tamsd.SUPPORT)
    counts = np.isfinite(allf).sum(axis=0)
    assert counts[0] == 4 and np.all(np.diff(counts) <= 0) and counts[-1] < 4
    assert counts.tolist() == [(duration >= x).sum() for x in lags / tamsd.SUPPORT]


def test_pieces_of_a_flight_pool_their_origins():
    sums = np.array([[10.0, 20.0], [30.0, 40.0]])
    origins = np.array([[2, 4], [3, 1]])
    m = tamsd.msd_matrix(np.array([0, 0]), np.array([500.0, 500.0]), sums, origins, 1, np.zeros(2))
    assert m[0] == pytest.approx([40 / 5, 60 / 5])
