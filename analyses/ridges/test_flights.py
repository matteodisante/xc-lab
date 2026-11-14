"""Tests of flights.py, and the benchmark of paper/FLIGHTS.md section 8 on synthetic flights.

Synthetic flights follow a crest path at 35 km/h: Lomax glides (mu 3.93, tau0 700 s) and climbs (mu 3.88, tau0
200 s) in whole turns of 40 m radius and 25 s, optionally with AR(1) GPS error (3 m, 60 s) and 2 min on the ground
at both ends. They are written as IGC files and read back through flights.process. Without the crest paths of
DATA (paths_<region>.npz) the paths are smooth random curves.

    uv run python test_flights.py [region] [n_paths] [data_dir]    # benchmark table
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import flights as fl  # noqa: E402
import raw_igc as ri  # noqa: E402

V = 35 / 3.6
R, TURN_S = 40.0, 25
DATA = Path("/Volumes/SSD_DISANTE/xc-lab/ridges/data")


def lomax(rng, mu, tau0):
    return tau0 * (rng.random() ** (-1 / mu) - 1)


def random_path(rng, length_m=40_000, persistence_m=3000, origin=(940_000.0, 6_480_000.0)):
    """Smooth random curve every STEP_M: heading is a random walk, so it is straight below `persistence_m`."""
    n = int(length_m / fl.STEP_M)
    th = np.cumsum(rng.normal(0, np.sqrt(fl.STEP_M / persistence_m), n))
    step = fl.STEP_M * np.c_[np.cos(th), np.sin(th)]
    return np.asarray(origin) + np.vstack([[0, 0], np.cumsum(step, axis=0)])


def flight_positions(path, rng, circling=True, noise=False, ground=False):
    """1-Hz positions of a pilot who flies `path` at V, with Lomax glides and climbs in whole turns."""
    s_end = fl.STEP_M * (len(path) - 1)
    at = lambda s: np.c_[np.interp(s, fl.STEP_M * np.arange(len(path)), path[:, 0]),
                         np.interp(s, fl.STEP_M * np.arange(len(path)), path[:, 1])]
    if not circling:
        return at(np.arange(0, s_end, V))
    pos, s = [], 0.0
    while s < s_end:
        g = max(int(lomax(rng, 3.93, 700)), 1)
        pos.append(at(np.minimum(s + V * np.arange(1, g + 1), s_end)))
        s = min(s + V * g, s_end)
        if s >= s_end:
            break
        p = at(np.array([s, s + 1.0]))
        tan = (p[1] - p[0]) / np.hypot(*(p[1] - p[0]))
        c = p[0] + R * np.array([-tan[1], tan[0]])
        turns = int(np.ceil(lomax(rng, 3.88, 200) / TURN_S))
        a = 2 * np.pi * np.arange(1, turns * TURN_S + 1) / TURN_S
        # counter-clockwise circle that starts at p heading along the path
        pos.append(c + R * (np.cos(a)[:, None] * (p[0] - c) / R + np.sin(a)[:, None] * tan))
    return np.vstack(pos)


def with_errors(xy, rng, noise, ground, sigma=3.0, tau=60.0):
    if ground:
        xy = np.vstack([np.repeat(xy[:1], 120, 0), xy, np.repeat(xy[-1:], 120, 0)])
    if noise:
        phi = np.exp(-1 / tau)
        e = np.zeros_like(xy)
        e[0] = rng.normal(0, sigma, 2)
        w = rng.normal(0, sigma * np.sqrt(1 - phi**2), xy.shape)
        for i in range(1, len(xy)):
            e[i] = phi * e[i - 1] + w[i]
        xy = xy + e
    return xy


def write_igc(xy, file, t0=10 * 3600, step=1):
    """IGC file of Lambert-93 positions every `step` seconds."""
    from pyproj import Transformer

    lon, lat = Transformer.from_crs(2154, 4326, always_xy=True).transform(xy[:, 0], xy[:, 1])
    def ddm(x, w, hemi):
        d, m = divmod(int(round(abs(x) * 60000)), 60000)
        return f"{d:0{w}d}{m:05d}{hemi[int(x < 0)]}"

    lines = ["AXXX", "HFDTE010726"]
    for i in range(0, len(xy), step):
        t = t0 + i
        lines.append(f"B{t // 3600 % 24:02d}{t // 60 % 60:02d}{t % 60:02d}{ddm(lat[i], 2, 'NS')}{ddm(lon[i], 3, 'EW')}A0100001000")
    Path(file).write_text("\n".join(lines) + "\n")


def item(pos, name):
    return {"flight": name, "t": np.arange(len(pos), dtype=float), "xy": pos, "seg": np.zeros(len(pos), int)}


def run_flights(paths, rng, **kw):
    """flights.process of one synthetic flight per path; the mask covers all the paths (region `alps`)."""
    res = []
    for k, p in enumerate(paths):
        pos = flight_positions(p, rng, kw.get("circling", True))
        res.append(fl.process(item(with_errors(pos, rng, kw.get("noise", False), False), f"f{k}")))
    return res


def box_mask(paths, cell=1000.0):
    xy = np.vstack(paths)
    left, top = np.floor(xy[:, 0].min() / cell - 5) * cell, np.ceil(xy[:, 1].max() / cell + 5) * cell
    shape = (int((top - xy[:, 1].min()) / cell) + 10, int((xy[:, 0].max() - left) / cell) + 10)
    return {"alps": (np.ones(shape, bool), left, top, cell)}


def route_h(results, kind, window):
    rows = np.array([r["rows"][kind] for r in results if r["rows"] is not None])
    h, _ = fl.window_table(rows, fl.lag_m(kind), np.arange(len(rows)), [window], 0, 0)
    return h[0, 0]


def paths_h(paths, window):
    rows = np.array([fl.pooled([p], fl.LAGS_K) for p in paths])
    h, _ = fl.window_table(rows, fl.LAGS_M["route"], np.arange(len(rows)), [window], 0, 0)
    return h[0, 0]


def load_paths(region, n, data=DATA):
    """The first n crest paths of paths_<region>.npz (xy float32 offsets from origin, split by off), or None."""
    f = Path(data) / f"paths_{region}.npz"
    if not f.exists():
        return None
    z = np.load(f)
    xy = z["xy"].astype(float) + z["origin"]
    return [xy[a:b] for a, b in zip(z["off"][:-1], z["off"][1:])][:n]


def benchmark(paths, seed=0):
    rng, out = np.random.default_rng(seed), {}
    fl._init({"masks": box_mask(paths), "region_min": 0.8, "max_dt": 5, "min_opening": 0.0})
    for name, kw in (("no circling", dict(circling=False)), ("circling", {}), ("circling + noise", dict(noise=True))):
        res = run_flights(paths, rng, **kw)
        out[name] = {k: [route_h(res, k, w) for w in fl.WINDOWS_M[:3]] for k in fl.KINDS}
        out[name]["circling_frac"] = np.nanmean([r["info"].get("circling_frac", np.nan) for r in res])
        out[name]["status"] = [r["info"]["status"] for r in res]
    return out


# tests
def test_lambert93_matches_pyproj():
    from pyproj import Transformer

    lat, lon = np.array([45.0, 42.7, 46.1, 48.8]), np.array([6.0, 0.1, 7.0, 2.3])
    e, n = Transformer.from_crs(4326, 2154, always_xy=True).transform(lon, lat)
    ee, nn = fl.lambert93(lat, lon)
    assert np.abs(ee - e).max() < 0.01 and np.abs(nn - n).max() < 0.01


def test_lags_are_log_spaced_and_unique():
    assert 60 < len(fl.LAGS_K) <= fl.N_LAGS and np.all(np.diff(fl.LAGS_K) > 0) and fl.LAGS_K[0] == 1
    assert 60 < len(fl.LAGS_T) <= fl.N_LAGS and fl.LAGS_T[0] == 1


def test_tamsd_of_straight_line_is_s_squared_and_stops_at_a_quarter():
    x = np.c_[np.arange(400.0), 0 * np.arange(400.0)]
    tot, cnt = fl.tamsd(x, np.array([1, 10, 100, 101]))
    assert np.allclose(tot[:3] / cnt[:3], [1, 100, 10_000]) and cnt[3] == 0


def test_random_walk_has_h_one_half_and_straight_line_one():
    rng = np.random.default_rng(1)
    walks = [np.cumsum(rng.normal(size=(20_000, 2)), 0) for _ in range(30)]
    rows = np.array([fl.pooled([w], fl.LAGS_T) for w in walks])
    h, _ = fl.window_table(rows, fl.LAGS_T.astype(float), np.arange(30), [(10, 1000)], 0, 0)
    assert abs(h[0, 0] - 0.5) < 0.02
    line = np.array([fl.pooled([np.c_[np.arange(2000.0), np.zeros(2000)]], fl.LAGS_T)] * 12)
    h, _ = fl.window_table(line, fl.LAGS_T.astype(float), np.arange(12), [(10, 400)], 0, 0)
    assert abs(h[0, 0] - 1) < 1e-9


def test_bootstrap_interval_covers_the_value_and_counts_whole_groups():
    rng = np.random.default_rng(2)
    lag = fl.LAGS_T.astype(float)
    rows = np.array([lag**1.6 * rng.lognormal(0, 0.3) for _ in range(40)])  # H = 0.8 in every flight
    h, _ = fl.window_table(rows, lag, np.arange(40), [(5, 500)], 200, 0)
    lo, hi = fl.interval(h[1:, 0])
    assert abs(h[0, 0] - 0.8) < 1e-6 and lo <= 0.8 + 1e-6 and hi >= 0.8 - 1e-6
    lnm, _ = fl.resampled_means(rows, np.repeat(np.arange(4), 10), 10, np.random.default_rng(3))
    assert np.isfinite(lnm).all()  # every resample of 4 groups of 10 keeps at least 10 flights


def test_lags_with_fewer_than_ten_flights_are_dropped():
    lag = np.array([1.0, 2.0, 4.0])
    rows = np.full((12, 3), 1.0)
    rows[:5, 2] = np.nan
    lnm, _ = fl.resampled_means(rows, np.arange(12), 0, np.random.default_rng(0))
    assert np.isfinite(lnm[0, :2]).all() and np.isnan(lnm[0, 2])


def test_parser_dialects_midnight_and_v_fixes(tmp_path):
    a = ("AXXX\r\nHFDTEDATE:010726,01\r\nB2359584503000N00600000EA0100001000\r\n"
         "B0000004503000N00600500EV0100001000\r\nB0000014503000S00600500WA0100001000\r\n")
    f = tmp_path / "a.igc"
    f.write_text(a)
    t, lat, lon, valid = ri.read_igc(f)
    assert list(t) == [86398, 86400, 86401] and valid.tolist() == [True, False, True]
    assert np.allclose(lat, [45.05, 45.05, -45.05]) and np.allclose(lon, [6.0, 6.00833333, -6.00833333])
    f.write_text("AXXX\nHFDTE010726\nnothing\n")
    assert len(ri.read_igc(f)[0]) == 0


def test_circle_detection_finds_whole_runs_and_ignores_zigzag():
    rng = np.random.default_rng(4)
    g = np.c_[np.arange(300) * V, np.zeros(300)]  # glide
    a = 2 * np.pi * np.arange(1, 126) / TURN_S
    c = g[-1] + R * np.c_[np.sin(a), 1 - np.cos(a)]  # five turns
    z = c[-1] + np.c_[np.arange(1, 200) * V, 30 * np.sin(np.arange(1, 200) / 4)]  # zig-zag, no net turning
    xy = np.vstack([g, c, z]) + rng.normal(0, 1, (300 + 125 + 199, 2))
    a, b = fl._runs(fl.circling(xy))
    assert len(a) == 1 and abs((b - a)[0] - 125) < 12 and abs(a[0] - 300) < 8


def test_chord_replaces_circles_by_a_straight_move():
    a = 2 * np.pi * np.arange(50) / 25
    xy = np.c_[np.arange(50) * 5.0, 0 * a] + 40 * np.c_[np.sin(a), 1 - np.cos(a)]
    circ = np.zeros(50, bool)
    circ[10:40] = True
    r, keep = fl.chord(xy, circ)
    assert keep.sum() == 50 - 28 and np.allclose(r[10], xy[10]) and np.allclose(r[39], xy[39])
    assert np.allclose(np.diff(r[10:40], axis=0), (xy[39] - xy[10]) / 29)


def test_region_assignment_by_fraction_of_fixes():
    m = np.zeros((10, 10), bool)
    m[:, :5] = True  # west half of a 10 x 10 km square at left = 0, top = 10000
    masks = {"alps": (m, 0.0, 10_000.0, 1000.0)}
    xy = np.c_[np.linspace(100, 9900, 100), np.full(100, 5000.0)]
    frac, region = fl.region_of(masks, xy, 0.8)
    assert np.isclose(frac["alps"], 0.5) and np.isnan(frac["pyrenees"]) and region == ""
    assert fl.region_of(masks, xy[:45], 0.8)[1] == "alps"


def test_synthetic_flights_route_matches_paths_and_track_is_lower():
    rng = np.random.default_rng(6)
    paths = [random_path(rng, 30_000) for _ in range(12)]
    b = benchmark(paths, seed=7)
    for name, r in b.items():
        assert all(s == "ok" for s in r["status"]), r["status"]
    h0 = paths_h(paths, fl.WINDOWS_M[0])
    plain, circ, noisy = b["no circling"], b["circling"], b["circling + noise"]
    assert abs(plain["route"][0] - h0) < 0.01 and abs(circ["route"][0] - h0) < 0.01 and abs(noisy["route"][0] - h0) < 0.015
    assert circ["track"][0] < circ["route"][0] - 0.01 and circ["dyn"][0] < circ["dyn_route"][0]
    assert 0.1 < circ["circling_frac"] < 0.6


def test_raw_igc_ground_is_cut_and_the_route_is_the_same(tmp_path):
    rng = np.random.default_rng(9)
    p = random_path(rng, 30_000)
    pos = flight_positions(p, rng)
    write_igc(with_errors(pos, rng, True, True), tmp_path / "f.igc")
    raw = ri.raw_flight(tmp_path / "f.igc")
    assert len(pos) - 400 < len(raw["t"]) <= len(pos)  # the 2 min on the ground at each end are cut
    fl._init({"masks": box_mask([p]), "region_min": 0.8, "max_dt": 5, "min_opening": 0.0})
    r = fl.process({"flight": "raw", **raw})
    assert r["info"]["status"] == "ok" and 0.1 < r["info"]["circling_frac"] < 0.6


def test_run_and_finish_write_the_tables(tmp_path):
    rng = np.random.default_rng(8)
    paths = [random_path(rng, 30_000) for _ in range(12)]
    items = [{**item(flight_positions(p, rng), f"f{k}"), "discipline": "paragliders"} for k, p in enumerate(paths)]
    m = box_mask(paths)["alps"]
    np.savez(tmp_path / "mask_alps.npz", mask=m[0], left=m[1], top=m[2], cell=m[3])
    geo = np.array([fl.pooled([p], fl.LAGS_K) for p in paths])
    np.savez(tmp_path / "tamsd_rows_alps_paths.npz", s_m=fl.LAGS_M["route"], rows=geo, group=np.arange(len(paths)))
    out = tmp_path / "out"
    fl.finish(fl.run(items, fl.configure(tmp_path, min_opening=0.0), jobs=2), out, tmp_path, n_boot=50)
    import csv

    rows = list(csv.DictReader(open(out / "flights.csv")))
    assert len(rows) == 12 and {r["region"] for r in rows} == {"alps"} and {r["status"] for r in rows} == {"ok"}
    w = list(csv.DictReader(open(out / "flights_windows.csv")))
    assert len(w) == 4 * 4 and all(abs(float(r["dH"])) < 0.05 for r in w if r["kind"] == "route" and r["window_km"] != "10-inf")
    assert (out / "flights_local.csv").exists() and np.load(out / "flights_rows_dyn.npz")["discipline"][0] == "paragliders"


if __name__ == "__main__":
    region = sys.argv[1] if len(sys.argv) > 1 else "alps"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    paths = load_paths(region, n, sys.argv[3] if len(sys.argv) > 3 else DATA)
    if paths is None:
        print("no crest paths: smooth random curves")
        paths = [random_path(np.random.default_rng(i), 40_000) for i in range(n)]
    print(f"{len(paths)} paths; H of the paths: " + " / ".join(f"{paths_h(paths, w):.3f}" for w in fl.WINDOWS_M[:3]))
    for name, r in benchmark(paths).items():
        print(f"{name:28s} route {r['route'][0]:.3f} / {r['route'][1]:.3f} / {r['route'][2]:.3f}   "
              f"track {r['track'][0]:.3f} / {r['track'][1]:.3f}   dyn {r['dyn'][0]:.3f} / {r['dyn'][1]:.3f}   "
              f"dyn_route {r['dyn_route'][0]:.3f} / {r['dyn_route'][1]:.3f}   circling_frac {r['circling_frac']:.2f}   "
              f"ok {r['status'].count('ok')}/{len(r['status'])}")
