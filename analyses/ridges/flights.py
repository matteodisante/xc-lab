"""Hurst exponents of paragliding flights against the crest geometry (definitions of paper/FLIGHTS.md, section 3).

Input per flight: the cleaned fixes (time, Lambert-93 position, segment). Pieces of continuous track (a segment,
split at gaps) are interpolated to 1 s. Four kinds of motion: dyn (positions every 1 s, circles included),
dyn_route (each circle replaced by uniform motion along its chord), track (smoothing spline, resampled every
STEP_M of arc length) and route (the track with each circle replaced by its chord: the one to compare with
H_geo). The TAMSD of a flight is the mean over its pieces of the squared displacement at 80 log-spaced lags up to
a quarter of the piece; across flights, equal weight per flight, lags with MIN_PIECES flights. H is the slope of
ln mean TAMSD against ln lag over a window, divided by 2, with 1000 bootstrap resamples of whole flights.

cleaned.py streams the flights of the old repository; run_flights.py selects them and runs this module in chunks.
Writes in OUT (finish): flights.csv, flights_rows_<kind>.npz, flights_local.csv, flights_<region>.png and, when
DATA holds the crest files, flights_windows.csv (H, H_geo, dH per region, kind, window).
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_v] = "1"  # one thread per worker: the pool is the parallelism

import csv
import warnings
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline, splev, splprep
from scipy.ndimage import uniform_filter1d

V_KMH = 35.0  # speed of the ball: time lag = arc length / v
N_LAGS, MAX_FRAC, MIN_PIECES = 80, 0.25, 10
S_MAX_M, T_MAX_S, STEP_M = 100_000.0, 10_800.0, 5.0  # longest lag; arc-length step of track and route
WINDOWS_M = ((100.0, np.inf), (100.0, 1000.0), (1000.0, 10_000.0), (10_000.0, np.inf))
N_BOOT, SEED = 1000, 0
KINDS = ("dyn", "dyn_route", "track", "route")
ARC_KINDS = ("track", "route")
REGIONS = ("alps", "pyrenees")
REGION_MIN = 0.8  # fraction of fixes over a massif's mask
SMOOTH_RMS_M = 3 * np.sqrt(2)  # rms tolerance of the smoothing spline (3 m per axis)
TURN_DEG_S, TURN_AVG_S, MERGE_GAP_S, MIN_CIRCLE_S = 8.0, 20, 10, 20  # circle detection
MAX_DT_S, MIN_OPENING = 5.0, 0.1  # selection: median sampling interval; takeoff-landing over path
GAP_S, MIN_PIECE_S = 30.0, 300  # a piece ends at a longer gap or at the end of its segment
ARRAYS = ("t", "xy", "seg", "E", "N", "z", "frame")  # the keys of a flight item that are not columns of flights.csv

# Lambert-93: conic conformal, two parallels, GRS80
_A, _F = 6378137.0, 1 / 298.257222101
_E = np.sqrt(2 * _F - _F**2)


def _m(phi):
    return np.cos(phi) / np.sqrt(1 - (_E * np.sin(phi)) ** 2)


def _t(phi):
    s = _E * np.sin(phi)
    return np.tan(np.pi / 4 - phi / 2) / ((1 - s) / (1 + s)) ** (_E / 2)


_P1, _P2, _P0, _L0 = np.radians([44.0, 49.0, 46.5, 3.0])
_N = (np.log(_m(_P1)) - np.log(_m(_P2))) / (np.log(_t(_P1)) - np.log(_t(_P2)))
_FC = _m(_P1) / (_N * _t(_P1) ** _N)
_R0 = _A * _FC * _t(_P0) ** _N


def lambert93(lat, lon):
    """Easting and northing (m) of WGS84 degrees."""
    r = _A * _FC * _t(np.radians(lat)) ** _N
    th = _N * (np.radians(lon) - _L0)
    return 700_000.0 + r * np.sin(th), 6_600_000.0 + _R0 - r * np.cos(th)


# lags: log-spaced integers, in steps of STEP_M (arc kinds) or seconds (time kinds)
def _grid(top):
    return np.unique(np.round(np.geomspace(1, top, N_LAGS)).astype(int))


LAGS_K, LAGS_T = _grid(S_MAX_M / STEP_M), _grid(T_MAX_S)
LAGS_M = {k: LAGS_K * STEP_M for k in ARC_KINDS}
LAG_UNITS = {"dyn": LAGS_T, "dyn_route": LAGS_T, **{k: LAGS_K for k in ARC_KINDS}}


def lag_m(kind, v_kmh=V_KMH):
    """Lags of a kind in metres of arc length (time lags times v)."""
    return LAGS_K * STEP_M if kind in ARC_KINDS else LAGS_T * v_kmh / 3.6


def _runs(mask):
    """Starts and ends (exclusive) of the runs of True."""
    d = np.diff(np.r_[False, mask, False].astype(int))
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1)


def split_pieces(t, xy, seg):
    """Pieces of continuous track: split at a new segment and at gaps above GAP_S; short ones dropped."""
    cut = np.flatnonzero((np.diff(t) > GAP_S) | (np.diff(seg) != 0)) + 1
    out = [(a, b) for a, b in zip(np.r_[0, cut], np.r_[cut, len(t)]) if t[b - 1] - t[a] >= MIN_PIECE_S]
    return [(t[a:b], xy[a:b]) for a, b in out]


def regrid(t, xy):
    """Positions every 1 s (cubic spline through the fixes)."""
    g = np.arange(np.ceil(t[0]), np.floor(t[-1]) + 1)
    return g, CubicSpline(t, xy)(g)


# circles
def circling(xy):
    """Boolean mask of the 1-s positions in thermalling circles."""
    d = np.diff(xy, axis=0)
    h = np.arctan2(d[:, 1], d[:, 0])
    rate = np.degrees(np.angle(np.exp(1j * np.diff(h))))  # deg/s, signed: figures of eight cancel
    fast = np.r_[False, np.abs(uniform_filter1d(rate, TURN_AVG_S, mode="constant")) > TURN_DEG_S, False]
    a, b = _runs(~fast)
    for s, e in zip(a, b):  # merge the gaps shorter than MERGE_GAP_S between two runs
        if 0 < s and e < len(fast) and e - s < MERGE_GAP_S:
            fast[s:e] = True
    a, b = _runs(fast)
    for s, e in zip(a, b):
        if e - s <= MIN_CIRCLE_S:
            fast[s:e] = False
    return fast


def chord(xy, circ):
    """xy with each circling run replaced by uniform motion along its chord, and the mask of kept points."""
    xy, keep = xy.copy(), np.ones(len(xy), bool)
    for s, e in zip(*_runs(circ)):
        if e - s > 2:
            xy[s:e] = xy[s] + (xy[e - 1] - xy[s]) * np.linspace(0, 1, e - s)[:, None]
            keep[s + 1 : e - 1] = False
    return xy, keep


def resample(xy, step=STEP_M):
    """Points every `step` of arc length along the polyline."""
    s = np.r_[0, np.cumsum(np.hypot(*np.diff(xy, axis=0).T))]
    g = np.arange(0, s[-1], step)
    return np.c_[np.interp(g, s, xy[:, 0]), np.interp(g, s, xy[:, 1])]


def smooth(t, xy, g):
    """Smoothing spline of the fixes (rms residual SMOOTH_RMS_M) evaluated at the times g."""
    tck, _ = splprep([xy[:, 0], xy[:, 1]], u=t, s=len(t) * SMOOTH_RMS_M**2, k=3)
    return np.column_stack(splev(g, tck))


def sample_kinds(t, xy):
    """The four kinds of one piece as arrays of equally spaced samples (1 s, or STEP_M of arc), and the
    circling mask of the 1-s grid."""
    g, p = regrid(t, xy)
    circ = circling(p)
    s = smooth(t, xy, g)
    ps = chord(p, circ)[0]
    ss, keep = chord(s, circ)
    return {"dyn": p, "dyn_route": ps, "track": resample(s), "route": resample(ss[keep])}, circ


def tamsd(x, lags):
    """Sum of squared displacements and number of pairs at each lag (0 pairs beyond MAX_FRAC of the piece)."""
    n = len(x)
    tot, cnt = np.zeros(len(lags)), np.zeros(len(lags))
    for i, k in enumerate(lags):
        if k > MAX_FRAC * n:
            break
        d = x[k:] - x[:-k]
        tot[i], cnt[i] = np.einsum("ij,ij->", d, d), n - k
    return tot, cnt


def pooled(samples, lags):
    tc = [tamsd(x, lags) for x in samples]
    tot, cnt = sum(a for a, _ in tc), sum(b for _, b in tc)
    return np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan)


# regions
def load_masks(data):
    out = {}
    for r in REGIONS:
        f = Path(data) / f"mask_{r}.npz"
        if f.exists():
            z = np.load(f)
            out[r] = (z["mask"].astype(bool), float(z["left"]), float(z["top"]), float(z["cell"]))
    return out


def fraction_in(mask, xy):
    m, left, top, cell = mask
    j, i = np.floor((xy[:, 0] - left) / cell).astype(int), np.floor((top - xy[:, 1]) / cell).astype(int)
    ok = (i >= 0) & (i < m.shape[0]) & (j >= 0) & (j < m.shape[1])
    inside = np.zeros(len(xy), bool)
    inside[ok] = m[i[ok], j[ok]]
    return inside.mean()


def region_of(masks, xy, region_min):
    """Fractions of the fixes over each massif's mask, and the massif that holds at least region_min of them."""
    frac = {r: fraction_in(masks[r], xy) if r in masks else np.nan for r in REGIONS}
    best = max(REGIONS, key=lambda r: -1 if np.isnan(frac[r]) else frac[r])
    return frac, best if frac[best] >= region_min else ""


# one flight
def read_flight(item, cfg):
    """Pieces of a flight (item: t, seg and xy, or E, N, z and frame of cleaned.py), with the selection verdict
    and the numbers of flights.csv."""
    if "xy" not in item:
        import cleaned

        item = {**item, "xy": cleaned.lambert(item)}
    t, xy, seg = item["t"], item["xy"], item["seg"]
    same = seg[1:] == seg[:-1]
    step = np.hypot(*np.diff(xy, axis=0).T)[same]
    path_m = step.sum()
    st = {"status": "ok", "fixes": len(t), "dt_median_s": float(np.median(np.diff(t)[same])), "duration_s": float(t[-1] - t[0]),
          "track_km": path_m / 1000, "opening": np.hypot(*(xy[-1] - xy[0])) / path_m if path_m else np.nan}
    frac, st["region"] = region_of(cfg["masks"], xy, cfg["region_min"])
    st.update({f"frac_{r}": frac[r] for r in REGIONS})
    pieces = split_pieces(t, xy, seg)
    for bad, name in ((st["dt_median_s"] > cfg["max_dt"], "dt"), (not pieces, "short"),
                      (not st["opening"] >= cfg["min_opening"], "opening"), (not st["region"], "region")):
        if bad:
            return {**st, "status": name}, None
    return st, pieces


def process(item):
    """Analyse one flight; returns its row of flights.csv (`info`) and, if it passes the selection, its TAMSD
    rows per kind."""
    columns = {k: v for k, v in item.items() if k not in ARRAYS}
    try:
        st, pieces = read_flight(item, _CFG)
        res = {"info": {**columns, **st}, "rows": None}
        if pieces is None:
            return res
        kinds, circ_all, dur = [], 0, 0
        for t, xy in pieces:
            samples, circ = sample_kinds(t, xy)
            kinds.append(samples)
            circ_all += circ.sum()
            dur += len(circ)
        res["info"]["circling_frac"] = circ_all / dur
        res["rows"] = {k: pooled([s[k] for s in kinds if len(s[k]) > 4], LAG_UNITS[k]) for k in KINDS}
        return res
    except Exception as e:  # a bad flight must not stop a run of thousands
        return {"info": {**columns, "status": f"error: {e!r}"[:80]}, "rows": None}


_CFG = {"masks": {}, "region_min": REGION_MIN, "max_dt": MAX_DT_S, "min_opening": MIN_OPENING}


def _init(cfg):
    global _CFG
    _CFG = cfg


# bootstrap fits
def _weights(groups, n_boot, rng):
    """(n_boot, n_rows) counts of each row in resamples of whole groups, in chunks of 100."""
    _, g = np.unique(groups, return_inverse=True)
    m = g.max() + 1
    for s in range(0, n_boot, 100):
        c = rng.multinomial(m, np.full(m, 1 / m), size=min(100, n_boot - s))
        yield c[:, g].astype(float)


def resampled_means(rows, groups, n_boot, rng):
    """ln mean TAMSD for the data (row 0) and for n_boot resamples of the groups: NaN below MIN_PIECES rows."""
    v = np.isfinite(rows)
    x = np.where(v, rows, 0.0)
    v = v.astype(float)
    w = [np.ones((1, len(rows))), *_weights(groups, n_boot, rng)]
    out = []
    for c in w:
        cnt = c @ v
        with np.errstate(all="ignore"):
            out.append(np.where(cnt >= MIN_PIECES, np.log((c @ x) / cnt), np.nan))
    return np.vstack(out), np.vstack([c @ v for c in w])


def slope_h(lnm, lag, lo, hi):
    """H on a window: half the least-squares slope of ln mean against ln lag, per row of lnm; NaN under 2 lags."""
    x = np.log(lag)
    m = np.isfinite(lnm) & (lag >= lo) & (lag <= hi)
    n = m.sum(1)
    xm = (m * x).sum(1) / np.maximum(n, 1)
    ym = np.where(m, lnm, 0).sum(1) / np.maximum(n, 1)
    dx = np.where(m, x - xm[:, None], 0)
    with np.errstate(all="ignore"):
        h = (dx * np.where(m, lnm - ym[:, None], 0)).sum(1) / (dx**2).sum(1) / 2
    return np.where(n >= 2, h, np.nan)


def local_h(lnm, lag):
    return np.gradient(lnm, np.log(lag), axis=1) / 2


def interval(x, axis=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanpercentile(x, [2.5, 97.5], axis=axis)


def window_table(rows, lag, groups, windows, n_boot, seed):
    """H of each window: point value and the n_boot bootstrap replicates (n_boot + 1, n_windows)."""
    lnm, _ = resampled_means(rows, groups, n_boot, np.random.default_rng(seed))
    return np.stack([slope_h(lnm, lag, lo, hi) for lo, hi in windows], 1), lnm


def load_geo(data, region):
    """Lags (m), TAMSD rows and tree ids of the crest paths (tamsd_rows_<region>_paths.npz)."""
    z = np.load(Path(data) / f"tamsd_rows_{region}_paths.npz")
    pick = lambda names: next((z[n] for n in names if n in z.files), None)
    lag, rows, grp = pick(("s_m", "lags_m", "lag_m", "lags")), pick(("rows", "tamsd", "msd")), pick(("group", "groups", "tree", "trees"))
    if lag is None or rows is None:
        raise KeyError(f"tamsd_rows_{region}_paths.npz: no lag/rows arrays among {z.files}")
    return np.asarray(lag, float), np.asarray(rows, float), np.arange(len(rows)) if grp is None else np.asarray(grp)


def fmt_window(lo, hi):
    return f"{lo / 1000:g}-{'inf' if np.isinf(hi) else f'{hi / 1000:g}'}"


# the run
def write_csv(path, rows):
    cols = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, cols)
        w.writeheader()
        w.writerows(rows)


def analyse(results, data, out, v_kmh=V_KMH, n_boot=N_BOOT, seed=SEED):
    """Rows, windows and local H from the results of `process`; writes the files of OUT."""
    out = Path(out)
    ok = [r for r in results if r["rows"] is not None]
    for kind in KINDS:
        np.savez(out / f"flights_rows_{kind}.npz", lag_m=lag_m(kind, v_kmh), rows=np.array([r["rows"][kind] for r in ok]),
                 flight=np.array([r["info"].get("flight", "") for r in ok]), region=np.array([r["info"]["region"] for r in ok]),
                 discipline=np.array([r["info"].get("discipline", "") for r in ok]))
    windows, local = [], []
    for region in REGIONS:
        sel = [r for r in ok if r["info"]["region"] == region]
        geo = None
        if (Path(data) / f"tamsd_rows_{region}_paths.npz").exists():
            glag, grows, ggrp = load_geo(data, region)
            geo = (glag, grows, ggrp, *window_table(grows, glag, ggrp, WINDOWS_M, n_boot, seed + 1))
            for lag, lnm, kind in [(glag, geo[4], "geo")]:
                local += _local_rows(region, kind, lag, lnm, v_kmh, grows)
        if not sel:
            continue
        for kind in KINDS:
            rows = np.array([r["rows"][kind] for r in sel])
            lag = lag_m(kind, v_kmh)
            h, lnm = window_table(rows, lag, np.arange(len(rows)), WINDOWS_M, n_boot, seed)
            local += _local_rows(region, kind, lag, lnm, v_kmh, rows)
            for w, (lo, hi) in enumerate(WINDOWS_M):
                row = {"region": region, "kind": kind, "window_km": fmt_window(lo, hi), "n_flights": len(sel),
                       "H": h[0, w], "H_lo": interval(h[1:, w])[0], "H_hi": interval(h[1:, w])[1]}
                if geo is not None:
                    d = h[1:, w] - geo[3][1:, w]
                    row.update({"H_geo": geo[3][0, w], "Hgeo_lo": interval(geo[3][1:, w])[0], "Hgeo_hi": interval(geo[3][1:, w])[1],
                                "dH": h[0, w] - geo[3][0, w], "dH_lo": interval(d)[0], "dH_hi": interval(d)[1]})
                windows.append(row)
    write_csv(out / "flights_local.csv", local)
    plot_local(local, out)
    if windows:
        write_csv(out / "flights_windows.csv", windows)
    return windows, local


def plot_local(local, out):
    """flights_<region>.png: local H against arc length (lag times v) for the kinds and the crests."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for region in REGIONS:
        rows = [r for r in local if r["region"] == region]
        if not rows:
            continue
        fig, ax = plt.subplots(figsize=(5, 3.2))
        for kind in ("geo", *KINDS):
            k = [r for r in rows if r["kind"] == kind]
            if k:
                x = [r["lag_m"] for r in k]
                ax.plot(x, [r["H"] for r in k], label=kind, lw=1)
                ax.fill_between(x, [r["H_lo"] for r in k], [r["H_hi"] for r in k], alpha=0.15, lw=0)
        ax.set(xscale="log", xlabel="arc length s = v tau (m)", ylabel="local H", title=region, ylim=(0.4, 1.05))
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out / f"flights_{region}.png", dpi=130)
        plt.close(fig)


def _local_rows(region, kind, lag, lnm, v_kmh, rows):
    h = local_h(lnm, lag)
    lo, hi = interval(h[1:], axis=0)
    n = np.isfinite(rows).sum(0)
    return [{"region": region, "kind": kind, "lag_m": lag[i], "tau_s": lag[i] / (v_kmh / 3.6), "n": n[i],
             "mean_m2": np.exp(lnm[0, i]), "H": h[0, i], "H_lo": lo[i], "H_hi": hi[i]} for i in range(len(lag)) if np.isfinite(lnm[0, i])]


def configure(data, region_min=REGION_MIN, max_dt=MAX_DT_S, min_opening=MIN_OPENING):
    """The settings of `process`: the masks of DATA and the selection thresholds."""
    return {"masks": load_masks(data), "region_min": region_min, "max_dt": max_dt, "min_opening": min_opening}


def run(items, cfg, jobs=1):
    """`process` of every item (dicts: t, seg, xy or E, N, z, frame, and columns for flights.csv), in a pool."""
    if jobs > 1:
        from multiprocessing import Pool

        with Pool(jobs, _init, (cfg,)) as p:
            return list(p.imap(process, items, chunksize=4))
    _init(cfg)
    return [process(it) for it in items]


def finish(results, out, data, v_kmh=V_KMH, n_boot=N_BOOT):
    """Write flights.csv and the files of `analyse` into OUT."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "flights.csv", [r["info"] for r in results])
    return analyse(results, data, out, v_kmh, n_boot)
