"""TAMSD of straight lines (H = 1) and of a circle, lag grid, bootstrap weights."""

import numpy as np

import tamsd


def straight(n):
    return np.c_[np.arange(n) * tamsd.STEP, np.zeros(n)]


def test_straight_line_tamsd_is_s_squared():
    ks = np.array([1, 10, 100, 400])
    m = tamsd.tamsd(straight(2000), ks)
    assert np.allclose(m, (ks * tamsd.STEP) ** 2)
    assert np.isnan(tamsd.tamsd(straight(2000), np.array([501]))).all()  # beyond a quarter of the curve


def test_lags_are_log_spaced_up_to_a_quarter():
    ks = tamsd.lags(40000)
    assert ks[0] == 1 and ks[-1] == 10000 and 60 < len(ks) <= 80
    assert (np.diff(np.log(ks))[10:] > 0).all()


def test_straight_lines_have_h_one_everywhere():
    n = np.random.default_rng(1).integers(2000, 20000, 40)
    ks = tamsd.lags(n.max())
    rows = np.array([tamsd.tamsd(straight(k), ks) for k in n])
    tab, wins = tamsd.analyse(rows, ks, np.arange(len(n)), n_boot=50)
    assert np.nanmax(np.abs(tab["H"] - 1)) < 1e-6
    assert all(abs(w[3] - 1) < 1e-6 for w in wins[:3])
    assert tab["n"][0] == 40 and np.isnan(tab["mean"][tab["n"] < tamsd.MIN_CURVES]).all()


def test_lags_with_few_curves_are_dropped():
    ks = tamsd.lags(8000)
    rows = np.array([tamsd.tamsd(straight(n), ks) for n in [8000] * 5 + [800] * 20])
    mean, den = tamsd.weighted_means(rows, np.ones((1, 25), np.float32))
    assert np.isfinite(mean[0][ks <= 200]).all() and np.isnan(mean[0][ks > 200]).all()


def test_bootstrap_resamples_whole_groups():
    group = np.repeat(np.arange(10), 5)
    w = tamsd.bootstrap_weights(group, n_boot=20)
    assert w.shape == (20, 50)
    assert (w.reshape(20, 10, 5).std(axis=2) == 0).all()  # members of a group share their multiplicity
    assert (w.sum(axis=1) == 50).all()


def test_circle_has_bounded_tamsd():
    t = np.linspace(0, 2 * np.pi * 5, 2000)
    r = 500.0
    xy = np.c_[r * np.cos(t), r * np.sin(t)]
    m = tamsd.tamsd(xy, np.array([1, 100, 300]))
    assert m[2] < (2 * r) ** 2 and m[0] < m[1]
