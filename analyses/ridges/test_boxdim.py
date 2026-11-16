"""Tests of boxdim.py on synthetic curves: line, smooth arc, Koch curve, fractional Brownian trace, pooling."""

import numpy as np

import boxdim as bd
import tamsd

OFFS = bd.offsets(16, 1)


def koch(level):
    pts = np.array([[0.0, 0.0], [1.0, 0.0]])
    rot = np.array([[0.5, -np.sqrt(3) / 2], [np.sqrt(3) / 2, 0.5]])
    for _ in range(level):
        a, b = pts[:-1], pts[1:]
        d = (b - a) / 3
        p1, p2 = a + d, a + 2 * d
        peak = p1 + d @ rot.T
        pts = np.stack([a, p1, peak, p2], 1).reshape(-1, 2)
        pts = np.vstack([pts, b[-1]])
    return pts


def fbm2d(n, h, seed):
    """Davies-Harte: two independent fractional Brownian paths of n steps, as an (n + 1, 2) curve."""
    k = np.arange(n + 1)
    g = 0.5 * (np.abs(k + 1) ** (2 * h) - 2 * np.abs(k) ** (2 * h) + np.abs(k - 1) ** (2 * h))
    lam = np.fft.fft(np.r_[g, g[-2:0:-1]]).real
    m = len(lam)
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(2):
        z = rng.standard_normal(m) + 1j * rng.standard_normal(m)
        out.append(np.cumsum(np.r_[0, np.fft.fft(np.sqrt(np.maximum(lam, 0) / m) * z).real[:n]]))
    return np.c_[out[0], out[1]]


def slopes(curve, lo, hi, per_decade=12):
    eps = bd.eps_grid(lo, hi, per_decade)
    n, d = bd.count_curves([curve], eps, OFFS)
    box = bd.dim_box(bd.window_slope(eps, n[0] - bd.END_BOX, lo, hi))
    return box, bd.dim_divider(bd.window_slope(eps, d[0], lo, hi))


def test_eps_grid():
    e = bd.eps_grid(200, 20000, 12)
    assert e[0] == 200 and np.isclose(e[-1], 20000) and len(e) == 25


def test_count_known_squares():
    x = np.linspace(0, 10, 1001)  # a segment of length 10 along x, boxes of side 1: 10 or 11 boxes
    line = np.c_[x, 0.5 * np.ones_like(x)]
    n, _ = bd.count_curves([line], np.array([1.0]), OFFS)
    assert 10 <= n[0, 0] <= 11


def test_line_is_one():
    x = np.arange(0, 50_000, 5.0)
    line = np.c_[x, 0.3 * x]
    box, div = slopes(line, 200, 5000)
    assert abs(box - 1) < 0.005 and abs(div - 1) < 0.005  # N - 1 is proportional to the length; N alone gives 0.98


def test_end_box_bias_of_raw_count():
    x = np.arange(0, 50_000, 5.0)
    eps = bd.eps_grid(200, 5000, 12)
    n, _ = bd.count_curves([np.c_[x, 0.3 * x]], eps, OFFS)
    d = bd.dim_box(bd.window_slope(eps, n[0], 200, 5000))
    assert 0.96 < d < 0.99


def test_smooth_curve_is_one():
    t = np.linspace(0, 3 * np.pi, 40_000)  # a wavy arc without self-intersections: sine of period 20 km, amplitude 5 km
    arc = np.c_[t * 6000, 5000 * np.sin(t)]
    box, div = slopes(arc, 200, 3000)
    assert abs(box - 1) < 0.03 and abs(div - 1) < 0.01


def test_koch_box_and_divider():
    k = koch(8)
    box, div = slopes(k, 3.0**-6, 3.0**-3, per_decade=12)
    target = np.log(4) / np.log(3)
    assert abs(box - target) < 0.03, box
    assert abs(div - target) < 0.03, div


def test_divider_of_square_side():
    sq = np.c_[np.linspace(0, 1, 2001), np.zeros(2001)]
    _, d = bd.count_curves([sq], np.array([0.1, 0.25]), OFFS)
    assert np.allclose(d[0], 1.0, atol=1e-6)


def test_fbm_trace_box_dimension():
    # trace of a planar fBm with H = 0.75: box dimension 1 / H = 4/3; the finite sample biases it low, tolerance 0.1
    h = 0.75
    vals = []
    for seed in range(6):
        c = fbm2d(2**17, h, seed)
        ext = np.ptp(c, axis=0).max()
        vals.append(slopes(c, ext / 200, ext / 12)[0])
    assert abs(np.mean(vals) - 1 / h) < 0.1, vals


def test_fbm_msd_exponent_matches_one_over_h():
    # the same curve: H from the TAMSD in time and D from boxes agree for fBm (time is not arc length, but fBm is the case of equality)
    h = 0.75
    c = fbm2d(2**15, h, 11)
    ks = tamsd.lags(len(c))
    m = tamsd.tamsd(c, ks)
    est = tamsd.window_h(ks.astype(float), m, 4, 2000)
    assert abs(est - h) < 0.05


def test_pooling_and_chord():
    # 40 straight lines of 30 km in 8 groups: D = 1, 1/H = 1 on the window, chord = arc length
    rng = np.random.default_rng(3)
    curves = []
    for _ in range(40):
        x = np.arange(0, 30_000, 5.0)
        a = rng.uniform(0, np.pi)
        curves.append(np.c_[np.cos(a) * x + rng.uniform(0, 1e5), np.sin(a) * x])
    eps = bd.eps_grid(200, 3000, 12)
    n, d = bd.count_curves(curves, eps, OFFS)
    length = np.full(40, 30_000.0)
    ks = tamsd.lags(len(curves[0]))
    rows = np.array([tamsd.tamsd(c, ks) for c in curves])
    res = bd.analyse_class(eps, n, d, length, rows, ks * 5.0, np.repeat(np.arange(8), 5), 16_000.0, n_boot=50)
    w = bd.window_table(res, 200.0, 3000.0)  # 16 km / 8 = 2 km caps the window
    assert abs(w["box"][0] - 1) < 0.01 and abs(w["div"][0] - 1) < 0.01 and abs(w["inv_h"][0] - 1) < 0.01
    assert w["eps_hi"] <= 16_000 / bd.EPS_OVER_LEN * (1 + 1e-9) and res["n"] == 40
    lo, hi = bd.interval(w["box"])
    assert lo <= w["box"][0] <= hi


def test_pick_class_and_eps_range():
    cls = {2000.0: {"n_groups": 500}, 8000.0: {"n_groups": 100}, 32000.0: {"n_groups": 30}, 64000.0: {"n_groups": 5}}
    assert bd.pick_class(cls, 900.0) == 8000.0  # 8000 / 8 = 1000 >= 900
    assert bd.pick_class(cls, 3000.0) == 32000.0
    assert bd.pick_class(cls, 7000.0) == 32000.0  # 64 km has too few groups: the largest class that has enough
    assert bd.pick_class({2000.0: {"n_groups": 3}}, 100.0) is None
    chord = lambda s: 0.8 * np.asarray(s)
    assert bd.eps_range(chord, 100.0, 1000.0) == (200.0, 800.0)  # lower end raised to EPS_MIN_M
    assert bd.eps_range(chord, 1000.0, np.inf) == (800.0, np.inf)


def test_local_table_of_lines():
    rng = np.random.default_rng(5)
    curves = [np.c_[np.arange(0, 40_000, 5.0), 0.2 * np.arange(0, 40_000, 5.0)] + rng.uniform(0, 1e5, 2) for _ in range(30)]
    eps = bd.eps_grid(200, 4000, 12)
    n, d = bd.count_curves(curves, eps, OFFS)
    ks = tamsd.lags(len(curves[0]))
    rows = np.array([tamsd.tamsd(c, ks) for c in curves])
    res = bd.analyse_class(eps, n, d, np.full(30, 40_000.0), rows, ks * 5.0, np.arange(30), 32_000.0, n_boot=20)
    box, div, inv = bd.local_table(res)
    ok = np.isfinite(box[0])
    assert ok.sum() >= 8
    assert np.allclose(box[0][ok], 1, atol=0.04)  # the local box slope of 30 lines scatters by about 0.02
    assert np.allclose(inv[0][ok], 1, atol=0.001) and np.allclose(div[0][ok], 1, atol=0.001)
