"""Does the crest pipeline need the 5 m DEM? The same pipeline on windows of the 25 m and of the 5 m DEM.

    uv run --with numba python dem_resolution.py --all --stage run      # crest curves of every window and run
    uv run --with numba python dem_resolution.py --all --stage report   # comparison, JSON, figures, report files

A window is a 25 x 25 km interior with a margin for the water routing. Every grid constant of the pipeline is a
length in metres (ridges.XSEC_M, curves.BRIDGE_M, curves.PRUNE_M) and goes back to cells at the cell size of the run; the thresholds in metres (valley area, heights, spline cut-off,
minimum length) do not change. Runs: A 25 m; B 5 m; C 5 m with a spline cut-off of 25 m; D as B with a larger margin.
Crests are kept in the interior only: curves of the window of at least MIN_LENGTH, cut at the interior, pieces of at
least MIN_LENGTH kept; no joins and no mountain-area rule (the interiors lie inside the mountain area).
"""

import argparse
import json
import resource
import time
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window
from scipy.spatial import cKDTree

import curves
import geo
import ridges
import run_flights
import tamsd

DEM5 = Path("/Volumes/SSD_DATA/dem/rge_alti_5m/rge_alti_5m.vrt")
ORIGIN5 = (310000.0, 6600000.0)  # Lambert-93 centre of cell (0, 0) of the 5 m mosaic; centres lie on multiples of 5 m
CACHE = geo.CACHE / "dem_resolution"
HERE = Path(__file__).parent
OUT = HERE / "out" / "dem_resolution"
REPORT = HERE / "crests"

# Interior boxes (x0, y0, x1, y1), Lambert-93 m, 25 x 25 km, where crest length and flight fixes are dense
WINDOWS = {
    "alps1": ("alps", (946000, 6513000, 971000, 6538000)),
    "alps2": ("alps", (957000, 6325000, 982000, 6350000)),
    "pyr1": ("pyrenees", (561000, 6189000, 586000, 6214000)),
    "pyr2": ("pyrenees", (432000, 6204000, 457000, 6229000)),
}
MASSIF = {"alps": ["alps1", "alps2"], "pyrenees": ["pyr1", "pyr2"]}
MARGIN_M = 5000.0  # water routing margin around the interior
# run: (cell m, spline cut-off m, margin m)
RUNS = {"A": (25.0, curves.SMOOTH_M, MARGIN_M), "B": (5.0, curves.SMOOTH_M, MARGIN_M),
        "C": (5.0, 25.0, MARGIN_M), "D": (5.0, curves.SMOOTH_M, 10000.0)}
# the grid constants of the 25 m pipeline, in metres
XSEC_M = ridges.XSEC_M
BRIDGE_M = curves.BRIDGE_M
PRUNE_M = curves.PRUNE_M
EDGE_M = curves.EDGE_M
BLOCK_M = 5000.0  # blocks of the paired bootstrap
KS = tamsd.lags(8000)  # lags of every TAMSD (up to a quarter of a 40 km curve)
H_WINDOWS = {"low": (100.0, 1000.0), "high": (1000.0, 5000.0), "grid": (25.0, 100.0)}  # m, 0.1-1 km, 1-5 km
KEEP_DH = 0.01  # criterion: keep 25 m if |H_B - H_A| (0.1-1 km) is below this or inside the interval of the difference
SAME_M = 1.0  # a B curve point is the same as a D one within this distance
ZOOM_M = 2000.0
RUN_COLOR = {"A": "#000000", "B": "#0072B2", "C": "#D55E00"}
RUN_LABEL = {"A": "25 m", "B": "5 m", "C": "5 m, cut-off 25 m"}


cells = ridges.cells


# --- DEM windows ---------------------------------------------------------------------------------------------

def _src5(box):
    x0, y0, x1, y1 = box
    c0, c1 = int(np.ceil((x0 - ORIGIN5[0]) / 5)), int(np.floor((x1 - ORIGIN5[0]) / 5))
    r0, r1 = int(np.ceil((ORIGIN5[1] - y1) / 5)), int(np.floor((ORIGIN5[1] - y0) / 5))
    return Window(c0, r0, c1 - c0 + 1, r1 - r0 + 1), (ORIGIN5[0] + 5 * c0, ORIGIN5[1] - 5 * r0)


def read5(box):
    """Elevations (float32, NaN = no data) of the 5 m cells with centres in box = (x0, y0, x1, y1), and the centre
    (x, y) of cell (0, 0). Rows go south."""
    win, org = _src5(box)
    with rasterio.open(DEM5) as src:
        z = src.read(1, window=win).astype(np.float32)
    z[z == -99999.0] = np.nan
    return z, org


def read_dem(cell, box):
    """Window of the DEM of cell size `cell` (25 or 5 m). The 25 m window has no data where the 5 m one has none,
    so that both runs see the same footprint (the 5 m mosaic stops at the border, the 25 m tiles go on)."""
    if cell == 5.0:
        return read5(box)
    z, org = geo.window(*box)
    z5, org5 = read5(box)
    h, w = z.shape
    ri = np.rint((org5[1] - (org[1] - 25.0 * np.arange(h))) / 5).astype(int)
    ci = np.rint(((org[0] + 25.0 * np.arange(w)) - org5[0]) / 5).astype(int)
    z[np.isnan(z5[np.ix_(np.clip(ri, 0, z5.shape[0] - 1), np.clip(ci, 0, z5.shape[1] - 1))])] = np.nan
    return z, org


# --- pipeline ------------------------------------------------------------------------------------------------

def inside(xy, box):
    x0, y0, x1, y1 = box
    return (xy[:, 0] >= x0) & (xy[:, 0] <= x1) & (xy[:, 1] >= y0) & (xy[:, 1] <= y1)


def clip(p, box, min_length=curves.MIN_LENGTH, step=curves.STEP):
    """Runs of the curve p inside the box that are at least min_length long."""
    ok = inside(p, box)
    d = np.diff(np.r_[False, ok, False].astype(int))
    return [p[a:b] for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)) if (b - a - 1) * step >= min_length]


def crest_curves(z, org, cell, smooth_m, box):
    """Crest curves of a DEM window (cell (0, 0) has its centre at org) cut at `box`, and the crest sides of the
    network inside the box (rows of 4 integers: end points in half metres, the smaller one first)."""
    lid, sz, tin, tout = ridges.drainage(z, cell=cell)
    acc, flat = ridges.crest_edges(z, lid, sz, tin, tout, xsec=cells(XSEC_M, cell))
    del lid, sz, tin, tout
    w = z.shape[1]
    sides = curves.bridge(acc, flat, cells(BRIDGE_M, cell))
    vid, inv = np.unique(sides.ravel(), return_inverse=True)
    inv = inv.reshape(-1, 2)
    xy = np.c_[org[0] - cell / 2 + cell * (vid % (w + 1)), org[1] + cell / 2 - cell * (vid // (w + 1))]
    ends = np.rint(2 * np.c_[xy[inv[:, 0]], xy[inv[:, 1]]]).astype(np.int64)
    ends = np.where((ends[:, :2] <= ends[:, 2:]).all(axis=1)[:, None], ends, ends[:, [2, 3, 0, 1]])
    ends = ends[inside(xy[inv[:, 0]], box) & inside(xy[inv[:, 1]], box)]
    v, off = curves.strokes(inv[:, 0], inv[:, 1], xy, prune_edges=cells(PRUNE_M, cell))
    out = []
    for a, b in zip(off[:-1], off[1:]):
        p = xy[v[a:b]]
        if np.hypot(*np.diff(p, axis=0).T).sum() < curves.MIN_LENGTH * 0.7:
            continue
        q = curves.smooth_resample(p, smooth=smooth_m)
        if (len(q) - 1) * curves.STEP >= curves.MIN_LENGTH:
            out += clip(q, box)
    return out, ends


def run_window(window, name):
    """One run on one window; saved in CACHE/<window>_<run>.npz with its peak memory."""
    t = time.time()
    region, box = WINDOWS[window]
    cell, smooth_m, margin = RUNS[name]
    big = (box[0] - margin, box[1] - margin, box[2] + margin, box[3] + margin)
    z, org = read_dem(cell, big)
    nan_all = float(np.isnan(z).mean())
    xs, ys = org[0] + cell * np.arange(z.shape[1]), org[1] - cell * np.arange(z.shape[0])
    sub = z[np.ix_((ys >= box[1]) & (ys <= box[3]), (xs >= box[0]) & (xs <= box[2]))]
    assert not np.isnan(sub).any(), f"{window}: no data inside the interior"
    cs, ends = crest_curves(z, org, cell, smooth_m, box)
    n = np.array([len(c) for c in cs])
    CACHE.mkdir(parents=True, exist_ok=True)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30
    np.savez_compressed(CACHE / f"{window}_{name}.npz", xy=np.concatenate(cs), off=np.r_[0, np.cumsum(n)], ends=ends,
                        peak_gb=peak, seconds=time.time() - t, shape=z.shape, nan_window=nan_all)
    print(f"{window} {name}: {z.shape} cells, {len(cs)} curves, {(n - 1).sum() * curves.STEP / 1e3:.0f} km, "
          f"peak {peak:.2f} GB, {time.time() - t:.0f} s", flush=True)


def load(window, name):
    f = np.load(CACHE / f"{window}_{name}.npz")
    return dict(curves=np.split(f["xy"], f["off"][1:-1]), ends=f["ends"], peak=float(f["peak_gb"]),
                seconds=float(f["seconds"]), nan_window=float(f["nan_window"]))


# --- comparison ----------------------------------------------------------------------------------------------

def nearest(a, b):
    """Distance from every point of a to the nearest point of b."""
    return cKDTree(b).query(a)[0]


def points(cs):
    return np.concatenate(cs)


def length_km(cs):
    return sum((len(c) - 1) * curves.STEP for c in cs) / 1e3


def dist_stats(a, b):
    d = nearest(points(a), points(b))
    return dict(mean=float(d.mean()), p95=float(np.percentile(d, 95)), median=float(np.median(d)), over_25m=float((d > 25).mean()))


def curve_rows(cs):
    return np.array([tamsd.tamsd(c, KS) for c in cs])


def blocks_of(cs, box, base):
    """Block of each curve (5 x 5 km cell of the interior holding its middle point), numbered from `base`."""
    nx = int(np.ceil((box[2] - box[0]) / BLOCK_M))
    mid = np.array([c[len(c) // 2] for c in cs])
    j = np.clip(((mid[:, 0] - box[0]) // BLOCK_M).astype(int), 0, nx - 1)
    i = np.clip(((box[3] - mid[:, 1]) // BLOCK_M).astype(int), 0, nx - 1)
    return base + i * nx + j


def fit_windows(s, m, ms=None):
    """H of the windows of H_WINDOWS from the mean TAMSD m (and, for draws, rows of ms)."""
    est = {k: tamsd.window_h(s, m, *w) for k, w in H_WINDOWS.items()}
    if ms is None:
        return est
    draws = {k: np.array([tamsd.window_h(s, x, *w) for x in ms]) for k, w in H_WINDOWS.items()}
    return est, draws


def interval(x):
    x = x[np.isfinite(x)]
    return [float(v) for v in np.percentile(x, [2.5, 97.5])] if len(x) > 20 else [None, None]


def analyse(rows):
    """alpha(s) and the H of the windows with a 95% bootstrap over curves (every curve is a group)."""
    s = KS * tamsd.STEP
    mean = tamsd.weighted_means(rows, np.ones((1, len(rows)), np.float32))[0][0]
    bm = tamsd.weighted_means(rows, tamsd.bootstrap_weights(np.arange(len(rows))))[0]
    est, draws = fit_windows(s, mean, bm)
    hb = np.array([tamsd.local_h(s, m) for m in bm])
    return dict(s_m=s.tolist(), mean=mean.tolist(), alpha=tamsd.local_h(s, mean).tolist(),
                alpha_lo=np.nanpercentile(hb, 2.5, axis=0).tolist(), alpha_hi=np.nanpercentile(hb, 97.5, axis=0).tolist(),
                H={k: dict(est=est[k], ci=interval(draws[k])) for k in H_WINDOWS})


def paired_difference(rows_x, blk_x, rows_y, blk_y, n_blocks):
    """H(y) - H(x) with the same block draws on both runs (blocks of BLOCK_M drawn with replacement)."""
    s = KS * tamsd.STEP
    rng = np.random.default_rng(tamsd.SEED)
    c = np.array([np.bincount(rng.integers(0, n_blocks, n_blocks), minlength=n_blocks) for _ in range(tamsd.N_BOOT)])
    mx = tamsd.weighted_means(rows_x, c[:, blk_x].astype(np.float32))[0]
    my = tamsd.weighted_means(rows_y, c[:, blk_y].astype(np.float32))[0]
    one = lambda r: tamsd.weighted_means(r, np.ones((1, len(r)), np.float32))[0][0]
    ex, ey = fit_windows(s, one(rows_x)), fit_windows(s, one(rows_y))
    dx, dy = fit_windows(s, one(rows_x), mx)[1], fit_windows(s, one(rows_y), my)[1]
    return {k: dict(est=ey[k] - ex[k], ci=interval(dy[k] - dx[k])) for k in H_WINDOWS}


def verdict(diff):
    d = diff["low"]
    lo, hi = d["ci"]
    return dict(small=bool(abs(d["est"]) < KEEP_DH), inside=bool(lo is not None and lo <= 0 <= hi),
                keep_25m=bool(abs(d["est"]) < KEEP_DH or (lo is not None and lo <= 0 <= hi)))


def exactness(b, d):
    """B against D (larger margin) in the interior: crest sides and curve points."""
    key = lambda e: np.ascontiguousarray(e).view([("", np.int64)] * 4).ravel()
    kb, kd = key(b["ends"]), key(d["ends"])
    pb, pd_ = points(b["curves"]), points(d["curves"])
    db, dd = nearest(pb, pd_), nearest(pd_, pb)
    return dict(sides_b=len(kb), sides_d=len(kd), sides_b_missing_in_d=float((~np.isin(kb, kd)).mean()),
                sides_d_missing_in_b=float((~np.isin(kd, kb)).mean()),
                curves_b=len(b["curves"]), curves_d=len(d["curves"]),
                length_b_km=length_km(b["curves"]), length_d_km=length_km(d["curves"]),
                b_points_off=float((db > SAME_M).mean()), d_points_off=float((dd > SAME_M).mean()),
                b_points_off_25m=float((db > 25).mean()), d_points_off_25m=float((dd > 25).mean()),
                max_dist_m=float(max(db.max(), dd.max())),
                identical=bool(len(kb) == len(kd) and np.isin(kb, kd).all() and (db <= SAME_M).all() and (dd <= SAME_M).all()))


def massif_curves(region, box):
    """Points of the curves of the full-massif 25 m run inside the box."""
    c = np.load(run_flights.newest_data() / f"curves_{region}.npz")
    p = c["xy"]
    return p[inside(p, box)]


def compare():
    res = dict(boxes={w: dict(region=r, box=b) for w, (r, b) in WINDOWS.items()}, margin_m=MARGIN_M,
               margin_d_m=RUNS["D"][2], runs={k: dict(cell=v[0], smooth=v[1], margin=v[2]) for k, v in RUNS.items()},
               xsec_m=XSEC_M, bridge_m=BRIDGE_M, prune_m=PRUNE_M, block_m=BLOCK_M, criterion=KEEP_DH,
               peak_gb={}, windows={}, massifs={})
    data = {(w, k): load(w, k) for w in WINDOWS for k in "ABCD" if (CACHE / f"{w}_{k}.npz").exists()}
    for (w, k), v in data.items():
        res["peak_gb"][f"{w}_{k}"] = round(v["peak"], 2)
    for w, (region, box) in WINDOWS.items():
        got = {k: data[w, k] for k in "ABCD" if (w, k) in data}
        r = dict(length_km={k: length_km(v["curves"]) for k, v in got.items()}, n={k: len(v["curves"]) for k, v in got.items()})
        if "A" in got and "B" in got:
            r["dist_AB"] = dist_stats(got["A"]["curves"], got["B"]["curves"])
            r["dist_BA"] = dist_stats(got["B"]["curves"], got["A"]["curves"])
        if "A" in got:
            m = massif_curves(region, box)
            d = nearest(points(got["A"]["curves"]), m)
            r["A_vs_massif_run"] = dict(mean=float(d.mean()), p95=float(np.percentile(d, 95)),
                                        within_25m=float((d <= 25).mean()), massif_km=len(m) * curves.STEP / 1e3)
        if "B" in got and "D" in got:
            r["B_vs_D"] = exactness(got["B"], got["D"])
        res["windows"][w] = r
    for region, ws in MASSIF.items():
        got = {k: [data[w, k] for w in ws] for k in "ABC" if all((w, k) in data for w in ws)}
        if set(got) != set("ABC"):
            continue
        cs = {k: [c for v in got[k] for c in v["curves"]] for k in got}
        blk = {k: np.concatenate([blocks_of(v["curves"], WINDOWS[w][1], 1000 * i) for i, (w, v) in enumerate(zip(ws, got[k]))])
               for k in got}
        ub = np.unique(np.concatenate(list(blk.values())))
        blk = {k: np.searchsorted(ub, b) for k, b in blk.items()}
        rows = {k: curve_rows(cs[k]) for k in got}
        r = dict(length_km={k: length_km(cs[k]) for k in got}, n={k: len(cs[k]) for k in got},
                 dist_AB=dist_stats(cs["A"], cs["B"]), dist_BA=dist_stats(cs["B"], cs["A"]),
                 tamsd={k: analyse(rows[k]) for k in got}, blocks=int(len(ub)))
        r["diff_BA"] = paired_difference(rows["A"], blk["A"], rows["B"], blk["B"], len(ub))
        r["diff_CA"] = paired_difference(rows["A"], blk["A"], rows["C"], blk["C"], len(ub))
        r["diff_CB"] = paired_difference(rows["B"], blk["B"], rows["C"], blk["C"], len(ub))
        r["verdict"] = verdict(r["diff_BA"])
        wd = {}
        for w in ws:
            rw = {k: curve_rows(data[w, k]["curves"]) for k in "ABC"}
            wd[w] = {k: analyse(rw[k])["H"] for k in "ABC"}
        r["by_window_H"] = wd
        res["massifs"][region] = r
    return res


# --- figures, macros, table ----------------------------------------------------------------------------------

def hillshade(z, cell, az=315.0, alt=45.0):
    gy, gx = np.gradient(z, cell)
    slope, asp = np.arctan(np.hypot(gx, gy)), np.arctan2(gy, -gx)
    a, h = np.radians(360 - az + 90), np.radians(alt)
    return np.sin(h) * np.cos(slope) + np.cos(h) * np.sin(slope) * np.cos(a - asp)


def zoom_box(cs, box):
    """ZOOM_M square of the interior with the most points of the 5 m curves."""
    p = points(cs)
    n = int((box[2] - box[0]) // ZOOM_M)
    j, i = ((p[:, 0] - box[0]) // ZOOM_M).astype(int), ((p[:, 1] - box[1]) // ZOOM_M).astype(int)
    ok = (j >= 0) & (j < n) & (i >= 0) & (i < n)
    h = np.zeros((n, n))
    np.add.at(h, (i[ok], j[ok]), 1)
    i0, j0 = np.unravel_index(np.argmax(h), h.shape)
    return (box[0] + j0 * ZOOM_M, box[1] + i0 * ZOOM_M, box[0] + (j0 + 1) * ZOOM_M, box[1] + (i0 + 1) * ZOOM_M)


def draw_map(ax, box, ca, cb, step=2, lw=0.5):
    """Hillshade of the 5 m DEM in box (every `step`-th cell) with the A and B curves."""
    z, org = read5(box)
    z = z[::step, ::step]
    x0, y1 = org
    ext = [x0 - 2.5, x0 - 2.5 + 5 * step * z.shape[1], y1 + 2.5 - 5 * step * z.shape[0], y1 + 2.5]
    ax.imshow(hillshade(np.nan_to_num(z), 5.0 * step), cmap="gray", extent=ext, vmin=0, vmax=1.2, interpolation="nearest")
    for cs, k, f in ((ca, "A", 1.0), (cb, "B", 0.5)):
        for c in cs:
            ax.plot(c[:, 0], c[:, 1], color=RUN_COLOR[k], lw=lw * f)
    ax.set_xlim(box[0], box[2])
    ax.set_ylim(box[1], box[3])
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def figure_window(w, data):
    import matplotlib.pyplot as plt
    import figstyle

    figstyle.apply()
    region, box = WINDOWS[w]
    ca, cb = data[w, "A"]["curves"], data[w, "B"]["curves"]
    zb = zoom_box(cb, box)
    fig, ax = plt.subplots(1, 2, figsize=(figstyle.TEXT_WIDTH_IN, 3.3))
    draw_map(ax[0], box, ca, cb)
    ax[0].plot([zb[0], zb[2], zb[2], zb[0], zb[0]], [zb[1], zb[1], zb[3], zb[3], zb[1]], color="#E69F00", lw=1)
    ax[0].set_title(f"{w}: 25 km, 10 m shading", loc="left")
    draw_map(ax[1], zb, ca, cb, step=1, lw=1)
    ax[1].set_title("zoom 2 x 2 km, 5 m shading", loc="left")
    ax[0].plot([], [], color=RUN_COLOR["A"], label="25 m DEM")
    ax[0].plot([], [], color=RUN_COLOR["B"], label="5 m DEM")
    ax[0].legend(loc="lower left", frameon=True, fontsize=7)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{w}.png", dpi=200)
    plt.close(fig)


def figure_alpha(res, data, path_pdf):
    """alpha(s) of A, B, C per massif and the zoom of one window as the third panel."""
    import matplotlib.pyplot as plt
    import figstyle

    figstyle.apply()
    fig, ax = plt.subplots(1, 3, figsize=(figstyle.TEXT_WIDTH_IN, 2.5), gridspec_kw=dict(width_ratios=[1, 1, 0.9], wspace=0.35))
    for a, (region, name) in zip(ax[:2], (("alps", "Alps"), ("pyrenees", "Pyrenees"))):
        t = res["massifs"][region]["tamsd"]
        for k in "ABC":
            s = np.array(t[k]["s_m"])
            a.plot(s, t[k]["alpha"], color=RUN_COLOR[k], lw=1.1, label=RUN_LABEL[k])
            a.fill_between(s, t[k]["alpha_lo"], t[k]["alpha_hi"], color=RUN_COLOR[k], alpha=0.2, lw=0)
        a.axvline(100, color=figstyle.GUIDE, lw=0.6)
        a.set_xscale("log")
        a.set_xlim(25, 3000)
        a.set_ylim(0.8, 1.05)
        a.set_xlabel("arc length $s$ (m)")
        a.set_title(name, loc="left", pad=14)
    ax[0].set_ylabel(r"$\alpha(s)$")
    ax[0].legend(loc="lower left", frameon=False)
    w = "alps1"
    box = WINDOWS[w][1]
    zb = zoom_box(data[w, "B"]["curves"], box)
    draw_map(ax[2], zb, data[w, "A"]["curves"], data[w, "B"]["curves"], step=1, lw=1)
    ax[2].set_title("Alps, 2 x 2 km", loc="left", pad=14)
    ax[2].plot([], [], color=RUN_COLOR["A"], label="25 m")
    ax[2].plot([], [], color=RUN_COLOR["B"], label="5 m")
    ax[2].legend(loc="lower left", frameon=True, fontsize=7)
    for a, c in zip(ax, "abc"):
        figstyle.letter(a, c)
    fig.savefig(path_pdf)
    plt.close(fig)


def tk(x):
    """Thousands with a thin space, for LaTeX."""
    return f"{x:,.0f}".replace(",", "\\,")


def fmt(x, n=3):
    return "n/a" if x is None or not np.isfinite(x) else f"{x:.{n}f}"


def hci(h, n=3):
    return f"{fmt(h['est'], n)} ({fmt(h['ci'][0], n)} to {fmt(h['ci'][1], n)})"


def dsign(h, n=3):
    return f"{h['est']:+.{n}f} ({h['ci'][0]:+.{n}f} to {h['ci'][1]:+.{n}f})"


def write_report(res):
    """crests/generated/dem_resolution_macros.tex and dem_resolution_table.tex."""
    names = {"alps": "Alps", "pyrenees": "Pyr"}
    mac = {"Boot": str(tamsd.N_BOOT), "Margin": f"{MARGIN_M / 1e3:g}", "MarginD": f"{RUNS['D'][2] / 1e3:g}", "Box": "25",
           "Block": f"{BLOCK_M / 1e3:g}", "Keep": f"{KEEP_DH:g}", "Windows": str(len(WINDOWS)),
           "Xsec": f"{XSEC_M:g}", "Bridge": f"{BRIDGE_M:g}", "Prune": f"{PRUNE_M:g}", "Smooth": f"{curves.SMOOTH_M:g}",
           "SmoothC": f"{RUNS['C'][1]:g}", "Valley": f"{ridges.VALLEY_KM2:g}"}
    rows = []
    for region, m in res["massifs"].items():
        R = names[region]
        for k in "ABC":
            mac[f"{R}Len{k}"] = tk(m["length_km"][k])
            mac[f"{R}N{k}"] = str(m["n"][k])
            for wk, wn in (("low", "Low"), ("high", "High"), ("grid", "Grid")):
                h = m["tamsd"][k]["H"][wk]
                mac[f"{R}H{wn}{k}"], mac[f"{R}H{wn}{k}Lo"], mac[f"{R}H{wn}{k}Hi"] = fmt(h["est"]), fmt(h["ci"][0]), fmt(h["ci"][1])
        for dk in ("BA", "CA", "CB"):
            for wk, wn in (("low", "Low"), ("high", "High")):
                h = m[f"diff_{dk}"][wk]
                mac[f"{R}DH{wn}{dk}"] = f"{h['est']:+.3f}".replace("-", "$-$")
                mac[f"{R}DH{wn}{dk}Lo"] = f"{h['ci'][0]:+.3f}".replace("-", "$-$")
                mac[f"{R}DH{wn}{dk}Hi"] = f"{h['ci'][1]:+.3f}".replace("-", "$-$")
        for a, b, key in (("AB", "AB", "dist_AB"), ("BA", "BA", "dist_BA")):
            mac[f"{R}Dist{a}Mean"] = f"{m[key]['mean']:.1f}"
            mac[f"{R}Dist{a}Pct"] = f"{m[key]['p95']:.1f}"
        mac[f"{R}Verdict"] = "keep the 25 m DEM" if m["verdict"]["keep_25m"] else "use the 5 m DEM"
        mac[f"{R}Small"] = "below" if m["verdict"]["small"] else "above"
        ex = [res["windows"][w]["B_vs_D"] for w in MASSIF[region] if "B_vs_D" in res["windows"][w]]
        if ex:
            mac[f"{R}ExactSides"] = f"{100 * (1 - max(e['sides_b_missing_in_d'] for e in ex)):.2f}"
            mac[f"{R}ExactPoints"] = f"{100 * (1 - max(e['b_points_off'] for e in ex)):.0f}"
            mac[f"{R}ExactPointsCell"] = f"{100 * (1 - max(e['b_points_off_25m'] for e in ex)):.1f}"
            mac[f"{R}ExactMax"] = f"{max(e['max_dist_m'] for e in ex):.1f}"
        am = [res["windows"][w]["A_vs_massif_run"] for w in MASSIF[region]]
        mac[f"{R}OverTwentyFive"] = f"{100 * np.mean([res['windows'][w]['dist_AB']['over_25m'] for w in MASSIF[region]]):.1f}"
        mac[f"{R}MassifMean"] = f"{np.mean([a['mean'] for a in am]):.1f}"
        mac[f"{R}MassifWithin"] = f"{100 * np.mean([a['within_25m'] for a in am]):.1f}"
        # table: three runs and the difference B - A
        rows.append(f"\\textit{{{ {"alps": "Alps", "pyrenees": "Pyrenees"}[region] }}} & & & & & \\\\")
        for k in "ABC":
            dist = f"{m['dist_AB']['mean']:.0f} ({m['dist_AB']['p95']:.0f}) / {m['dist_BA']['mean']:.0f} ({m['dist_BA']['p95']:.0f})" if k == "B" else ""
            rows.append(f"{k}, {RUN_LABEL[k]} & {m['n'][k]} & {tk(m['length_km'][k])} & {dist} & "
                        f"{hci(m['tamsd'][k]['H']['low'])} & {hci(m['tamsd'][k]['H']['high'])} \\\\")
        rows.append(f"B $-$ A & & & & {dsign(m['diff_BA']['low'])} & {dsign(m['diff_BA']['high'])} \\\\".replace("-", "$-$").replace("$$-$$", "$-$"))
        rows.append("\\midrule")
    rows.pop()
    (REPORT / "generated").mkdir(parents=True, exist_ok=True)
    (REPORT / "generated" / "dem_resolution_table.tex").write_text("% Generated by dem_resolution.py: do not edit.\n" + "\n".join(rows) + "\n")
    (REPORT / "generated" / "dem_resolution_macros.tex").write_text(
        "% Generated by dem_resolution.py: do not edit.\n" + "".join(f"\\newcommand{{\\Dr{k}}}{{{v}}}\n" for k, v in sorted(mac.items())))


def report():
    OUT.mkdir(parents=True, exist_ok=True)
    res = compare()
    (OUT / "dem_resolution.json").write_text(json.dumps(res, indent=1))
    data = {(w, k): load(w, k) for w in WINDOWS for k in "AB" if (CACHE / f"{w}_{k}.npz").exists()}
    for w in WINDOWS:
        if (w, "A") in data and (w, "B") in data:
            figure_window(w, data)
    if res["massifs"].keys() == MASSIF.keys():
        figure_alpha(res, data, REPORT / "figures" / "dem_resolution.pdf")
        write_report(res)
    for region, m in res["massifs"].items():
        print(region, "verdict", m["verdict"], "dH low", m["diff_BA"]["low"])


def write_parameters():
    CACHE.mkdir(parents=True, exist_ok=True)
    rows = "\n".join(f"| {w} | {r} | {b[0]} {b[1]} {b[2]} {b[3]} |" for w, (r, b) in WINDOWS.items())
    (CACHE / "PARAMETERS.md").write_text(f"""# DEM resolution test

Code: `xc-lab/analyses/ridges/dem_resolution.py`. Question: does the crest pipeline need the 5 m DEM (IGN RGE ALTI 5 m,
`{DEM5}`) or does the 25 m DEM (IGN BD ALTI) do? Same pipeline, same lengths in metres, on windows.

Interior boxes (Lambert-93 m, x0 y0 x1 y1, 25 x 25 km), chosen where the crest length and the flight fixes are dense
and the interior lies in the mountain area and in both DEMs:

| window | massif | box |
|---|---|---|
{rows}

Runs: A 25 m DEM; B 5 m DEM; C 5 m DEM with a spline cut-off of {RUNS['C'][1]:g} m; D as B with a {RUNS['D'][2] / 1e3:g} km margin
(A, B, C: {MARGIN_M / 1e3:g} km margin for the water routing). The 25 m window has no data where the 5 m one has none.

Grid constants in metres (cells at the cell size of the run): cross-section {XSEC_M:g} m, bridged flat stretches {BRIDGE_M:g} m,
pruned dead ends {PRUNE_M:g} m. Unchanged: valley {ridges.VALLEY_KM2:g} km2, cross-section drop {ridges.XSEC_DROP:g} m, height above
streams {ridges.HAND_MIN:g} m, strokes {curves.MAX_DEFLECTION:g} deg / {curves.DIRECTION_M:g} m, minimum length {curves.MIN_LENGTH:g} m,
spline cut-off {curves.SMOOTH_M:g} m, step {curves.STEP:g} m. Curves are cut at the interior; pieces under {curves.MIN_LENGTH:g} m dropped.

Files: `<window>_<run>.npz` (`xy`, `off`: curves in the interior; `ends`: crest sides of the interior in half metres;
`peak_gb`, `seconds`). Comparison: TAMSD lags {KS[0]}..{KS[-1]} samples of {tamsd.STEP:g} m, {tamsd.N_BOOT} bootstrap draws over
curves; the difference between runs uses blocks of {BLOCK_M / 1e3:g} km drawn with replacement, the same draws in both runs.
Criterion: keep 25 m if |H_B - H_A| on 0.1-1 km is below {KEEP_DH:g} or the 95% interval of the difference holds 0.
""")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--window", choices=list(WINDOWS))
    g.add_argument("--all", action="store_true")
    ap.add_argument("--run", choices=list("ABCD") + ["all"], default="all")
    ap.add_argument("--stage", choices=["run", "report"], default="run")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    if a.stage == "report":
        report()
    else:
        write_parameters()
        for w in (WINDOWS if a.all else [a.window]):
            for k in (list("ABCD") if a.run == "all" else [a.run]):
                if a.force or not (CACHE / f"{w}_{k}.npz").exists():
                    run_window(w, k)
