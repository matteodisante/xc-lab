import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import flights as fl  # noqa: E402
import model as md  # noqa: E402
import pauses as pa  # noqa: E402

GLIDE, PAUSE = (4.0, 700.0, 0.5), (3.0, 300.0, 0.5)
V = 10.0


def line(length=100_000):
    return np.c_[np.arange(0, length, fl.STEP_M), np.zeros(int(length / fl.STEP_M))]


def h_of(rows, tau, lo, hi):
    h, _ = fl.window_table(rows, tau, np.arange(len(rows)), [(lo, hi)], 0, 0)
    return h[0, 0]


def test_ball_without_pauses_on_a_straight_line_has_h_one():
    rng = np.random.default_rng(0)
    rows = md.model_rows([line(60_000)] * 12, rng, V, GLIDE, None, 1)
    assert abs(h_of(rows, fl.LAGS_T.astype(float), 10, 3000) - 1) < 1e-4


def test_pauses_bend_the_msd_below_one_and_it_returns_to_ballistic_over_long_lags():
    rng = np.random.default_rng(1)
    rows = md.model_rows([line(300_000)] * 20, rng, V, GLIDE, PAUSE, 2)
    tau = fl.LAGS_T.astype(float)
    assert h_of(rows, tau, 10, 2000) < 0.97  # intermittency lowers H
    m = np.nanmean(rows, 0)
    alpha = np.gradient(np.log(m), np.log(tau)) / 2
    i_mid, i_long = np.searchsorted(tau, 300), np.searchsorted(tau, 5000)
    assert alpha[i_long] > alpha[i_mid]  # finite mean: the exponent recovers at long lags


def test_equilibrium_fraction_of_time_moving_and_first_phase():
    rng = np.random.default_rng(2)
    eg, ep = pa.lomax_cond_mean(*GLIDE), pa.lomax_cond_mean(*PAUSE)
    frac = np.mean([md.moving_series(rng, 1_000_000, V, GLIDE, PAUSE)[:3000].mean() for _ in range(300)])
    assert abs(frac - eg / (eg + ep)) < 0.02


def test_equilibrium_residual_has_mean_ET2_over_2ET():
    rng = np.random.default_rng(3)
    mu, tau0, x0 = 4.0, 700.0, 0.5
    x = np.array([md.equilibrium_residual(rng, mu, tau0, x0) for _ in range(60_000)])
    # E[T^2]/(2E[T]) for T = x0 + Lomax(mu, tau0 + x0)
    t = md.draw(rng, 600_000, mu, tau0, x0)
    assert abs(x.mean() - np.mean(t.astype(float) ** 2) / (2 * t.mean())) / x.mean() < 0.1


def test_the_ball_ends_at_the_end_of_the_path():
    rng = np.random.default_rng(4)
    p = line(20_000)
    xy = md.positions(p, rng, V, GLIDE, PAUSE)
    assert np.allclose(xy[-1], p[-1], atol=V) and xy[0, 0] < 50 + V * 400


def test_straight_null_without_pauses_has_h_one_and_with_pauses_matches_the_crest_ball_on_a_line():
    rng = np.random.default_rng(5)
    rows = md.model_rows([line(60_000)] * 12, rng, V, GLIDE, None, 1, kind="straight")
    assert abs(h_of(rows, fl.LAGS_T.astype(float), 10, 3000) - 1) < 1e-4
    a = md.model_rows([line(300_000)] * 20, np.random.default_rng(6), V, GLIDE, PAUSE, 2, kind="straight")
    b = md.model_rows([line(300_000)] * 20, np.random.default_rng(7), V, GLIDE, PAUSE, 2)
    assert abs(h_of(a, fl.LAGS_T.astype(float), 10, 2000) - h_of(b, fl.LAGS_T.astype(float), 10, 2000)) < 0.01


def test_random_null_with_exponential_durations_has_alpha_one_half_at_long_lags():
    rng = np.random.default_rng(8)
    glide, pause = (2980.0, 2e5, 0.5), (2980.0, 1.5e5, 0.5)  # near-exponential Lomax laws, means about 100 s
    rows = md.model_rows([line(5_000_000)] * 6, rng, V, glide, pause, 1, kind="random")
    tau = fl.LAGS_T.astype(float)
    m = np.nanmean(rows, 0)
    a = np.gradient(np.log(m), np.log(tau)) / 2
    k = (tau > 2000) & (tau < 8000)
    assert abs(a[k].mean() - 0.5) < 0.05
    assert abs(a[tau < 3].mean() - 1) < 0.05  # ballistic below one glide
