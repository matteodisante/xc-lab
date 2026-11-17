"""Four crest runs of the whole massifs side by side: flat rule, footprint of the 5 m DEM, resolution.

    uv run --with numba python compare_dem.py data      # counts, H_geo, box dimension, zoom places -> out/compare_dem/compare_dem.json
    uv run --with numba python compare_dem.py report    # tables, macros and figures from the JSON and the caches

Runs (crest folders of geo.CACHE), each changing one thing from the one before:
    C1  BD ALTI 25 m, flood rule for the flats, all BD ALTI tiles of the departments, in-memory routing
    C2  as C1 with the flat rule of Barnes, Lehman and Mulla (2014)
    C3  as C2 on the footprint of the 5 m DEM (RGE ALTI read every fifth cell)
    C4  RGE ALTI 5 m, tiled routing, the same lengths in metres
C1 is the run of the report; C2 needs the files of `export_paths.py` and `run_dimension.py crests|network`.
Outputs: out/compare_dem/compare_dem.json, crests/generated/compare_dem_*.tex, crests/figures/compare_dem_*.pdf.
"""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

import boxdim as bd
import geo
import run_dimension as rd

HERE = Path(__file__).parent
OUT = HERE / "out" / "compare_dem"
GEN = HERE / "crests" / "generated"
FIG = HERE / "crests" / "figures"
REGIONS = ("alps", "pyrenees")
NAME = {"alps": "Alps", "pyrenees": "Pyrenees"}
KEY = {"alps": "Alps", "pyrenees": "Pyr"}
RUNS = ("C1", "C2", "C3", "C4")
RUN_KEY = {"C1": "One", "C2": "Two", "C3": "Three", "C4": "Four"}
PREFIX = {"C1": "bdalti25_", "C2": "bdalti25_", "C3": "rgealti5to25_", "C4": "rgealti5_"}
STEPS = {"Rule": ("C1", "C2"), "Foot": ("C2", "C3"), "Res": ("C3", "C4")}  # what changes from the first run to the second
H_WINDOWS = ("from 0.1 km", "0.1-1 km", "1-10 km", "beyond 10 km")
H_CURVE_WINDOWS = H_WINDOWS[:3]
D_WINDOWS = ("from 0.1 km", "0.1-1 km", "1-10 km")
WKEY = {"from 0.1 km": "From", "0.1-1 km": "Short", "1-10 km": "Mid", "beyond 10 km": "Long"}
WTEX = {"from 0.1 km": "from 0.1~km", "0.1-1 km": "0.1--1~km", "1-10 km": "1--10~km", "beyond 10 km": "beyond 10~km"}
MIN_ROWS = 100  # a lag is drawn if at least this many paths reach it (as in report_figures.py)
ZOOM_M = 3000.0
CELL_M = 500.0  # grid of the search for the zoom place
TOL_M = 25.0  # a point of one run farther than this from the other run is a point only that run has
MIN_POINTS = 800  # points (every 5 m) of each run a zoom square must hold
RUN_COLOR = {"C1": "#000000", "C2": "#009E73", "C3": "#0072B2", "C4": "#E69F00"}
RUN_LABEL = {"C1": "C1", "C2": "C2", "C3": "C3", "C4": "C4"}


def folder(run):
    """The crest folder of a run in geo.CACHE: C1 has the flood rule (no suffix), the others the flats-blm suffix."""
    found = [p for p in geo.CACHE.glob(PREFIX[run] + "*") if p.is_dir() and p.name.endswith("_flats-blm") == (run != "C1")]
    assert len(found) == 1, f"{run}: {found}"
    return found[0]


# --- numbers --------------------------------------------------------------------------------------------------

def counts(d, region):
    """Mountain area, curves, joins, trees and paths of a crest folder (lengths in km, areas in km2)."""
    c = np.load(d / f"curves_{region}.npz")
    t = np.load(d / f"tamsd_rows_{region}_paths.npz")
    n_joins = len(np.load(d / f"joins_{region}.npz")["a"])
    long = t["length"] >= 4 * 10_000.0  # a path reaches a lag of 10 km if it is at least four times as long
    return dict(area_km2=int(np.load(d / f"mask_{region}.npz")["mask"].sum()), length_km=float(c["length"].sum() / 1e3),
                curves=len(c["length"]), median_curve_km=float(np.median(c["length"]) / 1e3), joins=n_joins,
                trees=len(np.unique(t["tree"])), paths=len(t["length"]), median_path_km=float(np.median(t["length"]) / 1e3),
                longest_path_km=float(t["length"].max() / 1e3), paths_40km=int(long.sum()), trees_40km=len(np.unique(t["tree"][long])))


def hgeo(d, region):
    """{kind: {window: (H, lo, hi)}} of hgeo_windows.csv (NaN where the window is empty)."""
    w = pd.read_csv(d / "hgeo_windows.csv")
    w = w[w.region == region]
    return {k: {r.window: [r.H, r.ci_lo, r.ci_hi] for r in w[w.kind == k].itertuples()} for k in ("curves", "paths")}


def dimension(d, region):
    """Box and divider dimension of the crest paths on the windows of Section 4 and of the network on the boxes of Rak et al."""
    res = rd.crest_results(d, region, "paths")
    out = {}
    for name, c in rd.crest_windows(res).items():
        if c is None:
            continue
        w = c["w"]
        out[name] = dict(box=[float(w["box"][0]), *map(float, bd.interval(w["box"]))], div=[float(w["div"][0]), *map(float, bd.interval(w["div"]))],
                         inv_h=[float(w["inv_h"][0]), *map(float, bd.interval(w["inv_h"]))], lmin_km=c["lmin"] / 1e3, n=int(c["n"]), groups=int(c["groups"]),
                         eps_lo_m=float(c["e_lo"]), eps_hi_m=float(c["e_hi"]))
    net = rd.network_results(d, region)
    groups = {f"{k / 1e3:.0f}": int(v["n_groups"]) for k, v in res["classes"].items()}  # trees per class of length (km)
    return dict(windows=out, class_groups=groups, network=dict(small=float(net["small"]), large=float(net["large"]), cross_m=float(net["cross"])))


def points(d, region):
    """Points of the crest curves (Lambert-93, every 5 m) with the offsets of the curves."""
    c = np.load(d / f"curves_{region}.npz")
    return c["xy"], c["off"]


def zoom_place(p3, p4):
    """3 x 3 km square (x0, y0, x1, y1) with the most points that only one of the two runs has, among the squares that
    both fill with at least MIN_POINTS points; the squares lie on a grid of CELL_M."""
    only3 = cKDTree(p4).query(p3)[0] > TOL_M
    only4 = cKDTree(p3).query(p4)[0] > TOL_M
    x0, y0 = np.floor(np.minimum(p3.min(0), p4.min(0)) / CELL_M) * CELL_M
    x1, y1 = np.ceil(np.maximum(p3.max(0), p4.max(0)) / CELL_M) * CELL_M
    nx, ny = int((x1 - x0) / CELL_M) + 1, int((y1 - y0) / CELL_M) + 1

    def hist(p):
        i = ((p[:, 1] - y0) // CELL_M).astype(int)
        j = ((p[:, 0] - x0) // CELL_M).astype(int)
        h = np.zeros((ny, nx), np.int64)
        np.add.at(h, (i, j), 1)
        return h

    k = int(ZOOM_M / CELL_M)

    def win(h):
        c = np.pad(h.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
        return c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]

    n3, n4 = win(hist(p3)), win(hist(p4))
    o3, o4 = win(hist(p3[only3])), win(hist(p4[only4]))
    score = np.where((n3 >= MIN_POINTS) & (n4 >= MIN_POINTS), o3 + o4, -1)
    i, j = np.unravel_index(np.argmax(score), score.shape)
    box = (x0 + j * CELL_M, y0 + i * CELL_M, x0 + j * CELL_M + ZOOM_M, y0 + i * CELL_M + ZOOM_M)
    return dict(box=box, points_c3=int(n3[i, j]), points_c4=int(n4[i, j]), only_c3=int(o3[i, j]), only_c4=int(o4[i, j]),
                share_only_c3=float(only3.mean()), share_only_c4=float(only4.mean()))


def data():
    OUT.mkdir(parents=True, exist_ok=True)
    res = dict(runs={r: folder(r).name for r in RUNS}, counts={}, hgeo={}, dim={}, zoom={})
    for region in REGIONS:
        for r in RUNS:
            d = folder(r)
            res["counts"][f"{region}_{r}"] = counts(d, region)
            res["hgeo"][f"{region}_{r}"] = hgeo(d, region)
            res["dim"][f"{region}_{r}"] = dimension(d, region)
            print(region, r, "done", flush=True)
        p3, _ = points(folder("C3"), region)
        p4, _ = points(folder("C4"), region)
        res["zoom"][region] = zoom_place(p3, p4)
        print(region, "zoom", res["zoom"][region], flush=True)
    (OUT / "compare_dem.json").write_text(json.dumps(res, indent=1))


# --- tables and macros ------------------------------------------------------------------------------------------

def tk(x):
    return f"{x:,.0f}".replace(",", r"\,")


def num(x, n=3):
    t = f"{x:.{n}f}"
    return t.replace("-", r"\ensuremath{-}")


def delta(x):
    """|x| to three decimals, as '$<0.001$' where it rounds to zero."""
    return f"{abs(x):.3f}" if round(abs(x), 3) > 0 else r"$<0.001$"


def est(v, n=3):
    return "--" if v is None or not np.isfinite(v[0]) else num(v[0], n)


def ci(v, n=3):
    return "" if v is None or not np.isfinite(v[0]) else f"({num(v[1], n)}--{num(v[2], n)})"


def interval_rows(label, vals):
    """Two rows: the estimates of the four runs and under them their 95% intervals."""
    return [f"{label} & " + " & ".join(est(v) for v in vals) + r" \\",
            r" & " + " & ".join(r"\textcolor{ink2}{\scriptsize " + ci(v) + "}" if ci(v) else "" for v in vals) + r" \\"]


def counts_table(res):
    q = [("mountain area (km$^2$)", "area_km2", tk), ("crest curves, length (km)", "length_km", tk), ("curves", "curves", tk),
         ("median curve (km)", "median_curve_km", lambda x: f"{x:.2f}"), ("joins", "joins", tk), ("trees", "trees", tk), ("paths", "paths", tk),
         ("median path (km)", "median_path_km", lambda x: f"{x:.1f}"), ("longest path (km)", "longest_path_km", lambda x: f"{x:.0f}"),
         ("paths of 40 km or more", "paths_40km", tk), ("trees of these paths", "trees_40km", tk)]
    rows = []
    for label, key, f in q:
        rows.append(f"{label} & " + " & ".join(f(res["counts"][f"{r}_{k}"][key]) for r in REGIONS for k in RUNS) + r" \\")
    return "\n".join(rows)


def hgeo_table(res):
    rows = []
    for region in REGIONS:
        for kind, wins in (("paths", H_WINDOWS), ("curves", H_CURVE_WINDOWS)):
            rows += [r"\midrule" if rows else "", rf"\textit{{{NAME[region]}, {kind}}} & & & & \\"]
            for w in wins:
                rows += interval_rows(WTEX[w], [res["hgeo"][f"{region}_{k}"][kind].get(w) for k in RUNS])
    return "\n".join(x for x in rows if x)


def dim_table(res):
    rows = []
    for region in REGIONS:
        rows += [r"\midrule" if rows else "", rf"\textit{{{NAME[region]}, crest paths}} & & & & \\"]
        for key in ("box", "div"):
            for w in D_WINDOWS:
                vals = [res["dim"][f"{region}_{k}"]["windows"].get(w, {}).get(key) for k in RUNS]
                rows += interval_rows(rf"$D_{{\mathrm{{{key}}}}}$, {WTEX[w]}", vals)
        rows.append(rf"\textit{{{NAME[region]}, network}} & & & & \\")
        for key, label in (("small", "boxes 0.2--1.4~km"), ("large", "boxes 2.9--50~km")):
            rows.append(rf"$D_{{\mathrm{{box}}}}$, {label} & " + " & ".join(num(res["dim"][f"{region}_{k}"]["network"][key]) for k in RUNS) + r" \\")
    return "\n".join(x for x in rows if x)


def hdiff(res, region, a, b, w):
    """H of the paths in run b minus run a on window w (the difference of the values as the tables print them) (None if either is empty)."""
    va, vb = (res["hgeo"][f"{region}_{k}"]["paths"].get(w) for k in (a, b))
    return round(vb[0], 3) - round(va[0], 3) if va and vb and np.isfinite(va[0]) and np.isfinite(vb[0]) else None


def ddiff(res, region, a, b, w):
    """D_box of the crest paths in run b minus run a on window w, None if the two runs use another class or box sizes there."""
    va, vb = (res["dim"][f"{region}_{k}"]["windows"].get(w) for k in (a, b))
    if va is None or vb is None or va["lmin_km"] != vb["lmin_km"] or va["eps_hi_m"] != vb["eps_hi_m"]:
        return None
    return round(vb["box"][0], 3) - round(va["box"][0], 3)


def maxima(res):
    """Largest |change| per step of H of the paths on the windows below 10 km, on all four windows, and of D_box (windows of the same class);
    and whether every such interval holds the estimate of the other run and whether the two intervals overlap."""
    out = {}
    for step, (a, b) in STEPS.items():
        dh = [abs(hdiff(res, r, a, b, w)) for r in REGIONS for w in H_CURVE_WINDOWS]
        dh_all = dh + [abs(hdiff(res, r, a, b, "beyond 10 km")) for r in REGIONS]
        dd = [abs(x) for r in REGIONS for w in D_WINDOWS if (x := ddiff(res, r, a, b, w)) is not None]
        hold, overlap = [], []
        for r in REGIONS:
            for w in H_CURVE_WINDOWS:
                va, vb = (res["hgeo"][f"{r}_{k}"]["paths"][w] for k in (a, b))
                hold.append(va[1] <= vb[0] <= va[2] and vb[1] <= va[0] <= vb[2])
                overlap.append(max(va[1], vb[1]) <= min(va[2], vb[2]))
        out[step] = dict(max_dh=max(dh), max_dh_all=max(dh_all), max_dd=max(dd), intervals_hold_other=bool(all(hold)), intervals_overlap=bool(all(overlap)))
    return out


def separated(res):
    """(massif, window, run a, run b) of the pairs of runs whose 95% intervals of H of the paths do not overlap, on the windows below 10 km."""
    out = []
    for region in REGIONS:
        for w in H_CURVE_WINDOWS:
            for a, b in itertools.combinations(RUNS, 2):
                va, vb = (res["hgeo"][f"{region}_{k}"]["paths"][w] for k in (a, b))
                if max(va[1], vb[1]) > min(va[2], vb[2]):
                    out.append((region, w, a, b))
    return out


def macros(res):
    mac = {}
    for region in REGIONS:
        R = KEY[region]
        for r in RUNS:
            K = RUN_KEY[r]
            c = res["counts"][f"{region}_{r}"]
            mac[f"Cd{R}Area{K}"] = tk(c["area_km2"])
            mac[f"Cd{R}Len{K}"] = tk(c["length_km"])
            mac[f"Cd{R}Curves{K}"] = tk(c["curves"])
            mac[f"Cd{R}MedCurve{K}"] = f"{c['median_curve_km']:.2f}"
            mac[f"Cd{R}Joins{K}"] = tk(c["joins"])
            mac[f"Cd{R}Trees{K}"] = tk(c["trees"])
            mac[f"Cd{R}Paths{K}"] = tk(c["paths"])
            mac[f"Cd{R}MedPath{K}"] = f"{c['median_path_km']:.1f}"
            mac[f"Cd{R}LongPath{K}"] = f"{c['longest_path_km']:.0f}"
            mac[f"Cd{R}PathsForty{K}"] = tk(c["paths_40km"])
            mac[f"Cd{R}TreesForty{K}"] = tk(c["trees_40km"])
            h = res["hgeo"][f"{region}_{r}"]
            for w in H_WINDOWS:
                mac[f"Cd{R}HPaths{WKEY[w]}{K}"] = est(h["paths"].get(w))
            for w in H_CURVE_WINDOWS:
                mac[f"Cd{R}HCurves{WKEY[w]}{K}"] = est(h["curves"].get(w))
            dm = res["dim"][f"{region}_{r}"]
            for w in D_WINDOWS:
                mac[f"Cd{R}DBox{WKEY[w]}{K}"] = est(dm["windows"].get(w, {}).get("box"))
                mac[f"Cd{R}DDiv{WKEY[w]}{K}"] = est(dm["windows"].get(w, {}).get("div"))
            for w in D_WINDOWS:
                mac[f"Cd{R}DClass{WKEY[w]}{K}"] = f"{dm['windows'][w]['lmin_km']:.0f}"
                mac[f"Cd{R}DTrees{WKEY[w]}{K}"] = str(dm["windows"][w]["groups"])
            mac[f"Cd{R}GroupsThirtyTwo{K}"] = str(dm["class_groups"]["32"])
            mac[f"Cd{R}DNetSmall{K}"] = num(dm["network"]["small"])
            mac[f"Cd{R}DNetLarge{K}"] = num(dm["network"]["large"])
        for step, (a, b) in STEPS.items():
            ca, cb = res["counts"][f"{region}_{a}"], res["counts"][f"{region}_{b}"]
            for key, q in (("Area", "area_km2"), ("Len", "length_km"), ("Curves", "curves"), ("Paths", "paths")):
                mac[f"Cd{R}{key}Pct{step}"] = f"{100 * abs(cb[q] / ca[q] - 1):.1f}"
            mac[f"Cd{R}DensPct{step}"] = f"{100 * abs((cb['length_km'] / cb['area_km2']) / (ca['length_km'] / ca['area_km2']) - 1):.1f}"
            for w in H_WINDOWS:
                d = hdiff(res, region, a, b, w)
                if d is not None:
                    mac[f"Cd{R}DeltaH{WKEY[w]}{step}"] = delta(d)
            for w in D_WINDOWS:
                d = ddiff(res, region, a, b, w)
                if d is not None:
                    mac[f"Cd{R}DeltaD{WKEY[w]}{step}"] = delta(d)
        for k in ("Small", "Large"):
            for step, (a, b) in STEPS.items():
                mac[f"Cd{R}DNet{k}Delta{step}"] = delta(res["dim"][f"{region}_{b}"]["network"][k.lower()] - res["dim"][f"{region}_{a}"]["network"][k.lower()])
        z = res["zoom"][region]
        mac[f"Cd{R}OnlyThree"] = f"{100 * z['share_only_c3']:.0f}"
        mac[f"Cd{R}OnlyFour"] = f"{100 * z['share_only_c4']:.0f}"
        mac[f"Cd{R}SqOnlyThree"] = f"{100 * z['only_c3'] / z['points_c3']:.0f}"
        mac[f"Cd{R}SqOnlyFour"] = f"{100 * z['only_c4'] / z['points_c4']:.0f}"
    for step, m in maxima(res).items():
        mac[f"CdMaxDeltaH{step}"] = delta(m["max_dh"])
        mac[f"CdMaxDeltaHAll{step}"] = delta(m["max_dh_all"])
        mac[f"CdMaxDeltaD{step}"] = delta(m["max_dd"])
    for key, v in alpha_spread().items():
        mac[f"CdAlphaSpread{key}"] = delta(v)
    m = maxima(res)
    mac["CdMaxDeltaHAny"] = delta(max(v["max_dh"] for v in m.values()))
    mac["CdMaxDeltaDAny"] = delta(max(v["max_dd"] for v in m.values()))
    mac["CdPyrDeltaHShortTwoFour"] = delta(hdiff(res, "pyrenees", "C2", "C4", "0.1-1 km"))
    mac["CdMinPoints"] = str(MIN_POINTS)
    mac["CdTolM"] = f"{TOL_M:.0f}"
    mac["CdZoomKm"] = f"{ZOOM_M / 1e3:.0f}"
    mac["CdMinRows"] = str(MIN_ROWS)
    return mac


# --- figures ----------------------------------------------------------------------------------------------------

def alpha_spread():
    """Largest |alpha_k(s) - alpha_C1(s)| of the paths over the runs k = C2..C4 and both massifs, on s in 0.1-1, 1-5 and 5-10 km
    (alpha read from the lags that at least MIN_ROWS paths reach, interpolated in log s)."""
    out = {"Short": 0.0, "Mid": 0.0, "Long": 0.0}
    for region in REGIONS:
        t = {r: pd.read_csv(folder(r) / f"hgeo_{region}_paths.csv") for r in RUNS}
        t = {r: v[np.isfinite(v["mean"]) & (v["n"] >= MIN_ROWS)] for r, v in t.items()}
        for r in RUNS[1:]:
            for key, (lo, hi) in {"Short": (100, 1000), "Mid": (1000, 5000), "Long": (5000, 10000)}.items():
                s = np.geomspace(lo, hi, 200)
                a = np.interp(np.log(s), np.log(t["C1"].s_m), t["C1"].H)
                b = np.interp(np.log(s), np.log(t[r].s_m), t[r].H)
                out[key] = max(out[key], float(np.abs(a - b).max()))
    return out


def figure_alpha(res, path):
    import matplotlib.pyplot as plt

    import figstyle

    figstyle.apply()
    fig, ax = plt.subplots(1, 2, figsize=(figstyle.TEXT_WIDTH_IN, 2.9), gridspec_kw=dict(wspace=0.3))
    for a, region in zip(ax, REGIONS):
        for r in RUNS:
            t = pd.read_csv(folder(r) / f"hgeo_{region}_paths.csv")
            ok = np.isfinite(t["mean"]) & (t["n"] >= MIN_ROWS) & ((t.H_hi - t.H_lo) <= 0.1)
            a.plot(t.s_m[ok], t.H[ok], color=RUN_COLOR[r], lw=1.1, label=RUN_LABEL[r])
            a.fill_between(t.s_m[ok], t.H_lo[ok], t.H_hi[ok], color=RUN_COLOR[r], alpha=0.12, lw=0)
        a.axvline(100, color=figstyle.GUIDE, lw=0.6)
        a.set_xscale("log")
        a.set_xlim(5, 2e4)
        a.set_ylim(0.8, 1.02)
        a.axhline(1.0, color=figstyle.GUIDE, lw=0.6, zorder=0)
        a.set_xlabel("arc length $s$ (m)")
        a.set_title(NAME[region], loc="left", pad=6)
    ax[0].set_ylabel(r"local slope $\alpha(s)$")
    ax[0].legend(loc="lower left", frameon=False, ncol=2)
    for a, c in zip(ax, "ab"):
        figstyle.letter(a, c, y=1.1)
    fig.savefig(path)
    plt.close(fig)


def clipped(xy, off, box, pad=100.0):
    """The curves of (xy, off) that have a point in the box (padded)."""
    x0, y0, x1, y1 = box
    inside = (xy[:, 0] >= x0 - pad) & (xy[:, 0] <= x1 + pad) & (xy[:, 1] >= y0 - pad) & (xy[:, 1] <= y1 + pad)
    n = np.add.reduceat(inside.astype(np.int64), off[:-1])
    return [xy[a:b] for a, b, k in zip(off[:-1], off[1:], n) if k]


def figure_zoom(res, path):
    import matplotlib.pyplot as plt

    import dem_resolution as dr
    import figstyle

    figstyle.apply()
    fig, ax = plt.subplots(1, 2, figsize=(figstyle.TEXT_WIDTH_IN, 3.3), gridspec_kw=dict(wspace=0.08))
    for a, region in zip(ax, REGIONS):
        box = tuple(res["zoom"][region]["box"])
        z, org = dr.read5(box)
        x0, y1 = org
        ext = [x0 - 2.5, x0 - 2.5 + 5 * z.shape[1], y1 + 2.5 - 5 * z.shape[0], y1 + 2.5]
        a.imshow(dr.hillshade(np.nan_to_num(z), 5.0), cmap="gray", extent=ext, vmin=0, vmax=1.2, interpolation="nearest")
        for r, lw in (("C3", 1.8), ("C4", 0.9)):
            xy, off = points(folder(r), region)
            for c in clipped(xy, off, box):
                a.plot(c[:, 0], c[:, 1], color=RUN_COLOR[r], lw=lw)
        a.set_xlim(box[0], box[2])
        a.set_ylim(box[1], box[3])
        a.set_aspect("equal")
        a.set_xticks([])
        a.set_yticks([])
        a.set_title(NAME[region], loc="left", pad=6)
        a.plot([box[0] + 150, box[0] + 650], [box[1] + 150] * 2, color="white", lw=2)
        a.text(box[0] + 400, box[1] + 250, "500 m", color="white", ha="center", fontsize=7)
    ax[0].plot([], [], color=RUN_COLOR["C3"], lw=1.8, label="C3, 25 m grid")
    ax[0].plot([], [], color=RUN_COLOR["C4"], lw=0.9, label="C4, 5 m grid")
    fig.legend(loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.03))
    for a, c in zip(ax, "ab"):
        figstyle.letter(a, c, x=0.0, y=1.1)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def report():
    res = json.loads((OUT / "compare_dem.json").read_text())
    head = "% Generated by compare_dem.py: do not edit.\n"
    (GEN / "compare_dem_counts.tex").write_text(head + counts_table(res) + "\n")
    (GEN / "compare_dem_hgeo.tex").write_text(head + hgeo_table(res) + "\n")
    (GEN / "compare_dem_dimension.tex").write_text(head + dim_table(res) + "\n")
    (GEN / "compare_dem_macros.tex").write_text(head + "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in sorted(macros(res).items())))
    (OUT / "compare_dem_maxima.json").write_text(json.dumps(dict(steps=maxima(res), separated=separated(res)), indent=1))
    figure_alpha(res, FIG / "compare_dem_alpha.pdf")
    figure_zoom(res, FIG / "compare_dem_zoom.pdf")
    print("wrote tables, macros and figures", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("stage", choices=("data", "report"))
    a = ap.parse_args()
    data() if a.stage == "data" else report()
