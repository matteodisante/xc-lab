"""Box dimension of the crests and of the flight routes against 1/H (section crests/sections/dimension.tex).

    uv run --with numba python run_dimension.py crests [--sample N]     # box counts and dividers of every crest path
    uv run --with numba python run_dimension.py network                 # box counts of the union of the crest curves
    uv run --with numba python run_dimension.py flights [--sample N] [--jobs 4]   # routes and tracks of the para flights
    uv run --with numba --with pyproj python run_dimension.py report    # tables, figures and macros from the caches

Caches: the crest folder (dimension_<paths|curves>_<region>.npz, dimension_network_<region>.npz) and the flight folder
(dimension_flights.npz and chunks). Outputs: out/dimension/, crests/generated/dimension_*.tex, crests/figures/dimension_*.pdf.
Only `flights` imports flights.py, which sets one numba thread per process: run each stage in its own process.
"""

import argparse
import json
import resource
import time
from pathlib import Path

import numpy as np

import boxdim as bd
import tamsd

HERE = Path(__file__).parent
OUT = HERE / "out/dimension"
LAB = Path("/Volumes/SSD_DISANTE/xc-lab/ridges")
REGIONS = ("alps", "pyrenees")
EPS_NETWORK = (100.0, 50_000.0)  # m, box sizes of the network
MIN_LMIN_KM = min(bd.CLASS_MIN_KM)  # pieces and paths shorter than this are not counted
SAMPLE_SEED = 0


def crest_folder():
    import build

    return build.folder()


def peak_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30


def crest_curves(d, region, kind):
    """xy, off, length and group of the crest paths (kind "paths", the paths of H_geo) or of the curves before the joins."""
    if kind == "paths":
        z = np.load(d / f"paths_all_{region}.npz")
        xy, off, length, group = z["xy"], z["off"], z["length"], z["tree"]
    else:
        z = np.load(d / f"curves_{region}.npz")
        xy, off, length = z["xy"], z["off"], z["length"]
        group = np.arange(len(length))
    ref = np.load(d / f"tamsd_rows_{region}_{kind}.npz")
    assert np.allclose(length, ref["length"]) and (group == ref["group"]).all(), f"{kind} differ from the TAMSD rows"
    return xy, off.astype(np.int64), length, group


def crests(region, sample=0, kinds=("paths", "curves")):
    """N(eps) and L(eps) of every crest path (the paths of H_geo, same order as tamsd_rows_<region>_paths.npz) and of
    every curve (the same without the joins)."""
    d = crest_folder()
    for kind in kinds:
        t0 = time.time()
        xy, off, length, group = crest_curves(d, region, kind)
        idx = np.arange(len(length))
        if sample:
            idx = np.sort(np.random.default_rng(SAMPLE_SEED).choice(idx, sample, replace=False))
            pts = [xy[off[i]:off[i + 1]] for i in idx]
            off = np.r_[0, np.cumsum([len(p) for p in pts])].astype(np.int64)
            xy = np.concatenate(pts)
        eps, offs = bd.eps_grid(), bd.offsets()
        n = bd.count_many(xy, off, eps, offs)
        ell = bd.divider_many(xy, off, eps)
        f = (OUT / "sample" if sample else d) / f"dimension_{kind}_{region}.npz"
        f.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(f, eps=eps, n_box=n, n_div=ell, length=length[idx], group=group[idx], index=idx, n_offsets=len(offs))
        print(f"{region} {kind}: {len(idx)} curves, {time.time() - t0:.0f} s, peak {peak_gb():.2f} GB", flush=True)


def network(region):
    """N(eps) of the union of the crest curves of a region (one point set), eps from 100 m to 50 km."""
    t0 = time.time()
    d = crest_folder()
    c = np.load(d / f"curves_{region}.npz")
    xy = np.ascontiguousarray(c["xy"])
    eps = bd.eps_grid(*EPS_NETWORK)
    n = bd.count_many(xy, np.array([0, len(xy)], np.int64), eps, bd.offsets())[0]
    np.savez_compressed(d / f"dimension_network_{region}.npz", eps=eps, n_box=n, n_points=len(xy), length=c["length"].sum())
    print(f"{region} network: {len(xy)} points, {time.time() - t0:.0f} s, peak {peak_gb():.2f} GB", flush=True)


# flights
def flight_pieces(item):
    """Counts of the route and the track of every piece of one flight (None if the flight is not selected)."""
    import flights as fl

    st, pieces = fl.read_flight(item, _CFG)
    if pieces is None:
        return None
    out = []
    for t, xy in pieces:
        samples, _ = fl.sample_kinds(t, xy)
        for kind in ("route", "track"):
            x = samples[kind]
            ell = (len(x) - 1) * tamsd.STEP
            if len(x) > 4 and ell >= MIN_LMIN_KM * 1000:
                n, dv = bd.count_curves([x], _GRID["eps"], _GRID["offs"])
                out.append(dict(flight=st_id(item), kind=kind, region=st["region"], length=ell, n_box=n[0], n_div=dv[0],
                                tamsd=tamsd.tamsd(x, _GRID["ks"])))
    return out


def st_id(item):
    return item["flight"]


_CFG, _GRID = {}, {}


def _init(cfg, grid):
    global _CFG, _GRID
    _CFG, _GRID = cfg, grid


def flights(data, sample=0, jobs=4):
    import pickle
    from multiprocessing import Pool

    import cleaned as cl
    import flights as fl
    import run_flights as rf

    t0 = time.time()
    cfg = fl.configure(data)
    grid = dict(eps=bd.eps_grid(), offs=bd.offsets(), ks=fl.LAGS_K.astype(np.int64))
    disc = "paragliders"
    ids = rf.declared_open(disc)
    base = LAB / "flights_cleaned_v35_open0.1_dt5_reg0.8"
    out = (OUT / "sample") if sample else base
    out.mkdir(parents=True, exist_ok=True)
    (out / "dimension_chunks").mkdir(exist_ok=True)
    bd.count_curves([np.c_[np.arange(10.0), np.zeros(10)]], grid["eps"][:2], grid["offs"])  # compile before the pool
    res = []
    with Pool(jobs, _init, (cfg, grid)) as pool:
        if sample:
            pick = set(np.random.default_rng(SAMPLE_SEED).choice(sorted(ids & set(cl.frames(disc))), sample, replace=False))
            chunks = [("sample", cl.stream(disc, pick, 0, cl.n_row_groups(disc)))]
        else:
            n = cl.n_row_groups(disc)
            chunks = [(f"{disc}_{a:05d}", None) for a in range(0, n, rf.CHUNK)]
        for name, items in chunks:
            f = out / "dimension_chunks" / f"{name}.pkl"
            if f.exists() and not sample:
                res += pickle.loads(f.read_bytes())
                continue
            if items is None:
                a = int(name.split("_")[-1])
                items = cl.stream(disc, ids, a, a + rf.CHUNK)
            part = [p for r in pool.imap(flight_pieces, items, chunksize=4) if r for p in r]
            if not sample:
                rf.save_atomic(part, f)
            res += part
            print(f"{name}: {len(part)} pieces ({time.time() - t0:.0f} s)", flush=True)
    keys = ("flight", "kind", "region")
    np.savez_compressed(out / "dimension_flights.npz", eps=grid["eps"], s_m=grid["ks"] * tamsd.STEP, n_offsets=len(grid["offs"]),
                        length=np.array([r["length"] for r in res]), n_box=np.array([r["n_box"] for r in res]),
                        n_div=np.array([r["n_div"] for r in res]), tamsd=np.array([r["tamsd"] for r in res]),
                        **{k: np.array([r[k] for r in res]) for k in keys})
    print(f"{len(res)} pieces, {time.time() - t0:.0f} s, peak {peak_gb():.2f} GB (main process)", flush=True)


# report: tables, figures, macros
GEN = HERE / "crests/generated"
FIG = HERE / "crests/figures"
WINDOW_KEY = {"from 0.1 km": "From", "0.1-1 km": "Short", "1-10 km": "Mid", "beyond 10 km": "Long"}
WINDOW_TEX = {"from 0.1 km": "from 0.1~km", "0.1-1 km": "0.1--1~km", "1-10 km": "1--10~km"}
REGION_KEY = {"alps": "Alps", "pyrenees": "Pyr"}
REGION_NAME = {"alps": "Alps", "pyrenees": "Pyrenees"}
FIG_LMIN_M = 32_000.0  # class drawn in the scale-by-scale figure
FEHR = {"alps": 1.10}  # Fehr et al. 2009, main watershed line, divider method over 100 m to 200 km
RAK_SMALL_M, RAK_LARGE_M = (200.0, 1440.0), (2880.0, 50_000.0)  # Rak et al. 2019: 1-16 pixels (90 m to 1.4 km; here from 2x the smoothing cut-off) and 32 pixels up (here to 50 km)
JOIN_CLASSES_KM = (4, 8)  # classes on which the curves alone (no joins) are compared with the paths
NUM_WORD = {4: "Four", 8: "Eight"}
RAK = {"alps": (1.20, 1.84), "pyrenees": (1.25, 1.89)}
N_LONGEST = 10  # longest paths of a region for the single-path divider dimension
FLIGHT_DIR = LAB / "flights_cleaned_v35_open0.1_dt5_reg0.8"


def classes_of(eps, n_box, n_div, length, rows, s, group, end_box=bd.END_BOX):
    out = {}
    for km in bd.CLASS_MIN_KM:
        if (length >= km * 1000).sum() >= 3:
            out[km * 1000.0] = bd.analyse_class(eps, n_box, n_div, length, rows, s, group, km * 1000.0, end_box=end_box)
    return out


def crest_results(d, region, kind):
    z = np.load(d / f"dimension_{kind}_{region}.npz")
    t = np.load(d / f"tamsd_rows_{region}_{kind}.npz")
    args = (z["eps"], z["n_box"], z["n_div"], z["length"], t["rows"], t["s_m"], t["group"])
    mean = tamsd.weighted_means(t["rows"], np.ones((1, len(t["rows"])), np.float32))[0][0]
    return dict(classes=classes_of(*args), raw=classes_of(*args, end_box=0), chord=bd.s_to_chord(t["s_m"], mean), z=z, t=t)


def crest_windows(res):
    """For each window of tamsd.WINDOWS: (lmin, box sizes, table) from the chord of all the curves; None where no class reaches it."""
    out = {}
    for name, (lo, hi) in tamsd.WINDOWS.items():
        e_lo, e_hi = bd.eps_range(res["chord"], lo, hi)
        lmin = bd.pick_class(res["classes"], e_hi)
        out[name] = None
        if lmin is not None:
            e_hi = min(e_hi, lmin / bd.EPS_OVER_LEN)
            w = bd.window_table(res["classes"][lmin], e_lo, e_hi)
            if w is not None:
                raw = bd.window_table(res["raw"][lmin], w["eps_lo"], w["eps_hi"])
                out[name] = dict(lmin=lmin, e_lo=w["eps_lo"], e_hi=w["eps_hi"], w=w, raw=raw["box"][0],
                                 n=res["classes"][lmin]["n"], groups=res["classes"][lmin]["n_groups"])
    return out


def flight_results(region, kind):
    z = np.load(FLIGHT_DIR / "dimension_flights.npz")
    sel = (z["region"] == region) & (z["kind"] == kind)
    _, group = np.unique(z["flight"][sel], return_inverse=True)
    args = (z["eps"], z["n_box"][sel], z["n_div"][sel], z["length"][sel], z["tamsd"][sel], z["s_m"], group)
    return dict(classes=classes_of(*args), raw=classes_of(*args, end_box=0), n_flights=len(np.unique(group)))


def flight_windows(res, crest_win):
    """The windows of the crests (same class, same box sizes) on the flights."""
    out = {}
    for name, c in crest_win.items():
        out[name] = None
        if c is not None and c["lmin"] in res["classes"] and res["classes"][c["lmin"]]["n_groups"] >= bd.MIN_GROUPS:
            w = bd.window_table(res["classes"][c["lmin"]], c["e_lo"], c["e_hi"])
            if w is not None:
                raw = bd.window_table(res["raw"][c["lmin"]], c["e_lo"], c["e_hi"])
                out[name] = dict(lmin=c["lmin"], e_lo=w["eps_lo"], e_hi=w["eps_hi"], w=w, raw=raw["box"][0],
                                 n=res["classes"][c["lmin"]]["n"], groups=res["classes"][c["lmin"]]["n_groups"])
    return out


def std_inv_h(rows, s, group, lo, hi):
    """H and 1/H of all the curves on the standard window [lo, hi) of arc length, with bootstrap draws (row 0 = data)."""
    w = np.vstack([np.ones((1, len(rows)), np.float32), tamsd.bootstrap_weights(group, bd.N_BOOT, bd.SEED)])
    m, _ = tamsd.weighted_means(rows, w)
    h = np.array([tamsd.window_h(s, x, lo, hi) for x in m])
    return h, 1 / h


def fmt(x, n=3):
    """x to n decimals, with a minus sign that works in text and in maths and no negative zero."""
    t = f"{x:.{n}f}"
    return (t.lstrip("-") if float(t) == 0 else t).replace("-", r"\ensuremath{-}")


def cell(x):
    """Estimate (row 0) and 95% interval of draws x."""
    lo, hi = bd.interval(x)
    return f"{fmt(x[0])} ({fmt(lo)}, {fmt(hi)})"


def point(x):
    """Estimate (row 0) of draws x."""
    return fmt(x[0])


def km(m):
    return f"{m / 1000:.2f}" if m < 1000 else f"{m / 1000:.1f}"


def main_rows(groups, windows_keys=("from 0.1 km", "1-10 km")):
    """Compact table of D against 1/H: estimates, and the differences with their 95% intervals."""
    lines = []
    for gi, (label, wins) in enumerate(groups):
        lines += [r"\midrule" if gi else "", rf"\textit{{{label}}} & & & & & \\"]
        for name in windows_keys:
            w = wins[name]["w"]
            lines.append(" & ".join([WINDOW_TEX[name], point(w["box"]), point(w["div"]), point(w["inv_h"]),
                                     cell(w["box"] - w["inv_h"]), cell(w["div"] - w["inv_h"])]) + r" \\")
    return "\n".join(x for x in lines if x).removesuffix(r" \\")


def join_rows(joins):
    """Rows of the table paths against curves: joins[(region, class km, kind)] is a window table with its n."""
    lines = []
    for ri, region in enumerate(REGIONS):
        lines += [r"\midrule" if ri else "", rf"\textit{{{REGION_NAME[region]}}} & & & & & \\"]
        for km_class in JOIN_CLASSES_KM:
            for kind in ("paths", "curves"):
                w = joins[(region, km_class, kind)]
                lines.append(" & ".join([f"{kind}, $\\ge{km_class}$~km", f"{km(w['eps_lo'])}--{km(w['eps_hi'])}", f"{w['n']:,}".replace(",", r"\,"),
                                         cell(w["box"]), cell(w["div"]), cell(w["inv_h"])]) + r" \\")
    return "\n".join(x for x in lines if x).removesuffix(r" \\")


def cross_rows(crest, route, windows_keys=("from 0.1 km", "0.1-1 km", "1-10 km")):
    lines = []
    for ri, region in enumerate(REGIONS):
        lines += [r"\midrule" if ri else "", rf"\textit{{{REGION_NAME[region]}}} & & & & & \\"]
        for name in windows_keys:
            c, r = crest[region][name], route[region][name]
            if c is None or r is None:
                continue
            lines.append(" & ".join([WINDOW_TEX[name], point(r["w"]["inv_h"]), point(c["w"]["box"]), cell(c["w"]["box"] - r["w"]["inv_h"]),
                                     point(c["w"]["div"]), cell(c["w"]["div"] - r["w"]["inv_h"])]) + r" \\")
    return "\n".join(x for x in lines if x).removesuffix(r" \\")


def longest_paths(res, n=N_LONGEST):
    """Single-path divider and box dimensions of the n longest paths, fitted on eps from EPS_MIN_M to length / EPS_OVER_LEN."""
    z = res["z"]
    top = np.argsort(z["length"])[::-1][:n]
    eps = z["eps"]
    div = np.array([bd.dim_divider(bd.window_slope(eps, z["n_div"][i], bd.EPS_MIN_M, z["length"][i] / bd.EPS_OVER_LEN)) for i in top])
    box = np.array([bd.dim_box(bd.window_slope(eps, z["n_box"][i] - bd.END_BOX, bd.EPS_MIN_M, z["length"][i] / bd.EPS_OVER_LEN)) for i in top])
    return dict(div=div, box=box, lengths=z["length"][top], trees=z["group"][top])


def network_results(d, region):
    z = np.load(d / f"dimension_network_{region}.npz")
    eps, n = z["eps"], z["n_box"]
    local = bd.dim_box(bd.local_slope(eps, n))
    small = bd.dim_box(bd.window_slope(eps, n, *RAK_SMALL_M))
    large = bd.dim_box(bd.window_slope(eps, n, *RAK_LARGE_M))
    above = np.nonzero(local > 1.5)[0]
    cross = float(np.exp(np.interp(1.5, local[above[0] - 1:above[0] + 1], np.log(eps[above[0] - 1:above[0] + 1])))) if len(above) else np.nan
    return dict(eps=eps, n=n, local=local, small=small, large=large, cross=cross, length=float(z["length"]), points=int(z["n_points"]))


def band(x):
    return np.nanpercentile(x[1:], [2.5, 97.5], axis=0)


def local_tables(res, lmin):
    c = res["classes"][lmin]
    box, div, inv = bd.local_table(c)
    return c["eps"], box, div, inv


def figure_scales(crest, route):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker

    import figstyle

    figstyle.apply(9)
    fig, axs = plt.subplots(1, 2, figsize=(figstyle.TEXT_WIDTH_IN, 2.9), sharey=True)
    flight_color = figstyle.KIND_COLOR["route"]
    rows = []
    for ax, region, tag in zip(axs, REGIONS, "ab"):
        figstyle.letter(ax, tag)
        for res, color, who in ((crest[region], figstyle.REGION_COLOR[region], "crests"), (route[region], flight_color, "routes")):
            e, box, div, inv = local_tables(res, FIG_LMIN_M)
            ok = np.isfinite(box[0])
            lo, hi = band(box)
            ax.fill_between(e[ok] / 1000, lo[ok], hi[ok], color=color, alpha=0.2, lw=0)
            ax.plot(e[ok] / 1000, box[0][ok], color=color, lw=figstyle.LW)
            ax.plot(e[ok] / 1000, inv[0][ok], color=color, lw=figstyle.LW, ls=figstyle.FIT_DASH)
            if who == "crests":
                ax.plot(e[ok] / 1000, div[0][ok], color=color, lw=0.9, ls=":", marker="o", ms=2)
            for i in np.nonzero(ok)[0]:
                rows.append(dict(source=who, region=region, eps_m=e[i], D_box=box[0][i], D_box_lo=lo[i], D_box_hi=hi[i], D_div=div[0][i],
                                 inv_alpha=inv[0][i]))
        ax.axhline(1, color=figstyle.GUIDE, lw=0.6)
        ax.set(xscale="log", xlabel="box size $\\varepsilon$ (km)", title=REGION_NAME[region], ylim=(0.98, 1.16))
        ax.set_xticks([0.2, 0.5, 1, 2, 4])
        ax.set_xticklabels(["0.2", "0.5", "1", "2", "4"])
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    axs[0].set_ylabel("local $D(\\varepsilon)$ and $1/\\alpha$")
    handles = [plt.Line2D([], [], color="0.3", lw=figstyle.LW, label="box dimension $D(\\varepsilon)$"),
               plt.Line2D([], [], color="0.3", lw=figstyle.LW, ls=figstyle.FIT_DASH, label="$1/\\alpha(s)$, $s$ from the chord"),
               plt.Line2D([], [], color="0.3", lw=0.9, ls=":", marker="o", ms=2, label="divider dimension (crests)"),
               plt.Line2D([], [], color=figstyle.REGION_COLOR["alps"], lw=3, label="Alps crests"),
               plt.Line2D([], [], color=figstyle.REGION_COLOR["pyrenees"], lw=3, label="Pyrenees crests"),
               plt.Line2D([], [], color=flight_color, lw=3, label="flight routes")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.16))
    fig.tight_layout()
    fig.savefig(FIG / "dimension_scales.pdf")
    plt.close(fig)
    return rows


def figure_network(net):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import figstyle

    figstyle.apply(9)
    fig, axs = plt.subplots(1, 2, figsize=(figstyle.TEXT_WIDTH_IN, 2.9))
    for region in REGIONS:
        c, r = figstyle.REGION_COLOR[region], net[region]
        axs[0].plot(r["eps"] / 1000, r["n"], color=c, lw=figstyle.LW, label=REGION_NAME[region])
        axs[1].plot(r["eps"] / 1000, r["local"], color=c, lw=figstyle.LW, label=REGION_NAME[region])
        small, large = RAK[region]
        for (a, b), v in ((RAK_SMALL_M, small), (RAK_LARGE_M, large)):
            axs[1].plot([a / 1000, b / 1000], [v, v], color=c, lw=1.0, ls=":")
    axs[0].set(xscale="log", yscale="log", xlabel="box size $\\varepsilon$ (km)", ylabel="occupied boxes $N(\\varepsilon)$")
    axs[1].set(xscale="log", xlabel="box size $\\varepsilon$ (km)", ylabel="local $D(\\varepsilon)$", ylim=(0.9, 2.05))
    for ax in axs:
        ax.legend(frameon=False, loc="upper right" if ax is axs[0] else "lower right")
    figstyle.letter(axs[0], "a")
    figstyle.letter(axs[1], "b")
    axs[1].axhline(2, color=figstyle.GUIDE, lw=0.6)
    axs[1].axhline(1, color=figstyle.GUIDE, lw=0.6)
    fig.tight_layout()
    fig.savefig(FIG / "dimension_network.pdf")
    plt.close(fig)


def report():
    import flights as fl
    import pandas as pd

    t0 = time.time()
    d = crest_folder()
    mac = {}

    def put(name, value):
        mac[name] = value

    crest = {r: crest_results(d, r, "paths") for r in REGIONS}
    curve = {r: crest_results(d, r, "curves") for r in REGIONS}
    cwin = {r: crest_windows(crest[r]) for r in REGIONS}
    route = {r: flight_results(r, "route") for r in REGIONS}
    track = {r: flight_results(r, "track") for r in REGIONS}
    rwin = {r: flight_windows(route[r], cwin[r]) for r in REGIONS}
    twin = {r: flight_windows(track[r], cwin[r]) for r in REGIONS}
    print(f"analysis done ({time.time() - t0:.0f} s)", flush=True)

    # standard 1/H (all curves, windows of tamsd.py): crests and routes
    fz = np.load(FLIGHT_DIR / "flights_rows_route.npz")
    std = {}
    for region in REGIONS:
        t = crest[region]["t"]
        sel = (fz["region"] == region) & (fz["discipline"] == "paragliders")
        for who, (rows, s, grp) in (("crest", (t["rows"], t["s_m"], t["group"])), ("route", (fz["rows"][sel], fz["lag_m"], np.arange(sel.sum())))):
            for name, (lo, hi) in tamsd.WINDOWS.items():
                std[(who, region, name)] = std_inv_h(rows, s, grp, lo, hi)

    table_lines = []
    for who, wins in (("paths", cwin), ("route", rwin), ("track", twin)):
        for region in REGIONS:
            for name, c in wins[region].items():
                if c is None:
                    continue
                w = c["w"]
                row = dict(set=who, region=region, window=name, class_km=c["lmin"] / 1000, eps_lo_m=c["e_lo"], eps_hi_m=c["e_hi"], s_lo_m=w["s_lo"], s_hi_m=w["s_hi"],
                           n_curves=c["n"], n_groups=c["groups"], D_box_raw=c["raw"])
                for key in ("box", "div", "h", "inv_h"):
                    row[key] = w[key][0]
                    row[key + "_lo"], row[key + "_hi"] = bd.interval(w[key])
                for key in ("box", "div"):
                    diff = w[key] - w["inv_h"]
                    row["d" + key], (row["d" + key + "_lo"], row["d" + key + "_hi"]) = diff[0], bd.interval(diff)
                if who in ("paths", "route"):
                    sh, si = std[("crest" if who == "paths" else "route", region, name)]
                    row["H_std"], row["inv_H_std"] = sh[0], si[0]
                    row["inv_H_std_lo"], row["inv_H_std_hi"] = bd.interval(si)
                table_lines.append(row)
    pd.DataFrame(table_lines).to_csv(OUT / "dimension_windows.csv", index=False, float_format="%.5g")

    # tables (bodies of tabulars; the headers are in the section)
    GEN.mkdir(exist_ok=True)
    group_rows = [(f"{REGION_NAME[r]} crests", cwin[r]) for r in REGIONS] + [(f"{REGION_NAME[r]} routes", rwin[r]) for r in REGIONS]
    (GEN / "dimension_table_main.tex").write_text(main_rows(group_rows) + "\n")
    (GEN / "dimension_table_short.tex").write_text(main_rows(group_rows, ("0.1-1 km",)) + "\n")
    joins = {(r, km, kind): bd.window_table(res[r]["classes"][km * 1000.0], bd.EPS_MIN_M, km * 1000.0 / bd.EPS_OVER_LEN) | {"n": res[r]["classes"][km * 1000.0]["n"]}
             for km in JOIN_CLASSES_KM for kind, res in (("paths", crest), ("curves", curve)) for r in REGIONS}
    (GEN / "dimension_table_curves.tex").write_text(join_rows(joins) + "\n")
    for (r, k, kind), w in joins.items():
        p = f"DmJn{REGION_KEY[r]}{NUM_WORD[k]}{kind.capitalize()}"
        for key, tag in (("box", "Box"), ("div", "Div"), ("inv_h", "InvH")):
            lo, hi = bd.interval(w[key])
            put(p + tag, fmt(w[key][0]))
            put(p + tag + "Lo", fmt(lo))
            put(p + tag + "Hi", fmt(hi))
        put(p + "N", f"{w['n']:,}".replace(",", r"\,"))
        put(p + "EpsHiM", f"{w['eps_hi']:.0f}")
    (GEN / "dimension_table_cross.tex").write_text(cross_rows(cwin, rwin) + "\n")

    # macros
    for who, wins in (("Cr", cwin), ("Rt", rwin), ("Tk", twin)):
        for region in REGIONS:
            for name, c in wins[region].items():
                if c is None:
                    continue
                p, w = f"Dm{who}{REGION_KEY[region]}{WINDOW_KEY[name]}", c["w"]
                for key, tag in (("box", "Box"), ("div", "Div"), ("inv_h", "InvH"), ("h", "H")):
                    lo, hi = bd.interval(w[key])
                    put(p + tag, fmt(w[key][0]))
                    put(p + tag + "Lo", fmt(lo))
                    put(p + tag + "Hi", fmt(hi))
                for key, tag in (("box", "DiffBox"), ("div", "DiffDiv")):
                    lo, hi = bd.interval(w[key] - w["inv_h"])
                    put(p + tag, fmt((w[key] - w["inv_h"])[0]))
                    put(p + tag + "Lo", fmt(lo))
                    put(p + tag + "Hi", fmt(hi))
                put(p + "BoxRaw", fmt(c["raw"]))
                put(p + "EpsLo", f"{c['e_lo']:.0f}")
                put(p + "EpsHiKm", km(c["e_hi"]))
                put(p + "ClassKm", f"{c['lmin'] / 1000:.0f}")
                put(p + "N", f"{c['n']:,}".replace(",", r"\,"))
                put(p + "Groups", f"{c['groups']:,}".replace(",", r"\,"))
                if who in ("Cr", "Rt"):
                    sh, si = std[("crest" if who == "Cr" else "route", region, name)]
                    lo, hi = bd.interval(si)
                    put(p + "StdInvH", fmt(si[0]))
                    put(p + "StdInvHLo", fmt(lo))
                    put(p + "StdInvHHi", fmt(hi))
                    put(p + "StdH", fmt(sh[0]))
    for region in REGIONS:  # crest against route
        for name in ("from 0.1 km", "0.1-1 km", "1-10 km"):
            c, r = cwin[region][name], rwin[region][name]
            if c is None or r is None:
                continue
            for key, tag in (("box", "Box"), ("div", "Div")):
                x = c["w"][key] - r["w"]["inv_h"]
                lo, hi = bd.interval(x)
                p = f"DmX{REGION_KEY[region]}{WINDOW_KEY[name]}{tag}"
                put(p, fmt(x[0]))
                put(p + "Lo", fmt(lo))
                put(p + "Hi", fmt(hi))
    # single longest paths and the literature
    for region in REGIONS:
        lp = longest_paths(crest[region])
        k = REGION_KEY[region]
        put(f"DmLong{k}DivMean", fmt(lp["div"].mean()))
        put(f"DmLong{k}DivMin", fmt(lp["div"].min()))
        put(f"DmLong{k}DivMax", fmt(lp["div"].max()))
        put(f"DmLong{k}BoxMean", fmt(lp["box"].mean()))
        put(f"DmLong{k}LenMinKm", f"{lp['lengths'].min() / 1000:.0f}")
        put(f"DmLong{k}LenMaxKm", f"{lp['lengths'].max() / 1000:.0f}")
        put(f"DmLong{k}EpsHiKm", km(lp["lengths"].max() / bd.EPS_OVER_LEN))
        put(f"DmLong{k}Trees", str(len(np.unique(lp["trees"]))))
        c64 = crest[region]["classes"].get(64_000.0)
        if c64 is not None:
            w = bd.window_table(c64, bd.EPS_MIN_M, 64_000.0 / bd.EPS_OVER_LEN)
            if w is not None:
                lo, hi = bd.interval(w["div"])
                put(f"DmPool{k}Div", fmt(w["div"][0]))
                put(f"DmPool{k}DivLo", fmt(lo))
                put(f"DmPool{k}DivHi", fmt(hi))
                put(f"DmPool{k}EpsHiKm", km(w["eps_hi"]))
                put(f"DmPool{k}Groups", str(c64["n_groups"]))
                put(f"DmPool{k}N", str(c64["n"]))
    put("DmFehrAlps", fmt(FEHR["alps"], 2))
    net = {r: network_results(d, r) for r in REGIONS}
    for region in REGIONS:
        k, n = REGION_KEY[region], net[region]
        put(f"DmNet{k}Small", fmt(n["small"], 2))
        put(f"DmNet{k}Large", fmt(n["large"], 2))
        put(f"DmNet{k}Cross", km(n["cross"]))
        put(f"DmNet{k}LocLo", fmt(n["local"][int(np.argmin(abs(n["eps"] - bd.EPS_MIN_M)))], 2))
        put(f"DmNet{k}Max", fmt(np.nanmax(n["local"]), 2))
        put(f"DmRak{k}Small", fmt(RAK[region][0], 2))
        put(f"DmRak{k}Large", fmt(RAK[region][1], 2))
        put(f"DmNet{k}Points", f"{n['points']:,}".replace(",", r"\,"))
    put("DmRakSmallLoM", f"{RAK_SMALL_M[0]:.0f}")
    put("DmRakSmallPixLoM", "90")
    put("DmRakSmallHiKm", km(RAK_SMALL_M[1]))
    put("DmRakLargeLoKm", km(RAK_LARGE_M[0]))
    put("DmRakLargeHiKm", km(RAK_LARGE_M[1]))
    # constants
    for name, value in (("EpsMin", f"{bd.EPS_MIN_M:.0f}"), ("EpsMaxKm", km(bd.EPS_MAX_M)), ("PerDecade", bd.PER_DECADE), ("Offsets", bd.N_OFFSETS),
                        ("Ratio", bd.EPS_OVER_LEN), ("Boot", bd.N_BOOT), ("MinGroups", bd.MIN_GROUPS), ("Step", f"{tamsd.STEP:.0f}"),
                        ("Opening", fl.MIN_OPENING), ("MaxDt", f"{fl.MAX_DT_S:.0f}"), ("RegionMin", fl.REGION_MIN), ("MinPieceKm", MIN_LMIN_KM),
                        ("NetEpsMinM", f"{EPS_NETWORK[0]:.0f}"), ("NetEpsMaxKm", km(EPS_NETWORK[1])), ("NLongest", N_LONGEST)):
        put(f"Dm{name}", str(value))
    for region in REGIONS:
        put(f"DmFlights{REGION_KEY[region]}", f"{route[region]['n_flights']:,}".replace(",", r"\,"))
    local = figure_scales(crest, route)
    for region in REGIONS:
        for who, tag in (("crests", "Cr"), ("routes", "Rt")):
            r = [x for x in local if x["region"] == region and x["source"] == who]
            for pos, name in ((0, "Lo"), (-1, "Hi")):
                put(f"DmLoc{REGION_KEY[region]}{tag}Box{name}", fmt(r[pos]["D_box"], 2))
                put(f"DmLoc{REGION_KEY[region]}{tag}Inv{name}", fmt(r[pos]["inv_alpha"], 2))
    a = cwin["alps"]["from 0.1 km"]["w"]
    put("DmEstSpread", fmt(a["div"][0] - a["box"][0]))
    b = rwin["alps"]["from 0.1 km"]["w"]
    put("DmEstSpreadRoute", fmt(b["div"][0] - b["box"][0]))
    for r in REGIONS:
        for k, tag in ((4, "Four"), (8, "Eight")):
            put(f"DmJn{REGION_KEY[r]}{tag}DivGap", fmt(joins[(r, k, "paths")]["div"][0] - joins[(r, k, "curves")]["div"][0]))
    lines = ["% Generated by run_dimension.py: do not edit."] + [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(mac.items())]
    (GEN / "dimension_macros.tex").write_text("\n".join(lines) + "\n")

    # figures and local csv
    pd.DataFrame(local).to_csv(OUT / "dimension_local.csv", index=False, float_format="%.5g")
    figure_network(net)
    pd.DataFrame([dict(region=r, eps_m=e, N=n, D_local=dl) for r in REGIONS for e, n, dl in zip(net[r]["eps"], net[r]["n"], net[r]["local"])]).to_csv(
        OUT / "dimension_network.csv", index=False, float_format="%.5g")
    print(f"report done ({time.time() - t0:.0f} s), peak {peak_gb():.2f} GB", flush=True)




if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("stage", choices=("crests", "network", "flights", "report"))
    ap.add_argument("--region", choices=REGIONS, default=None)
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.stage in ("crests", "network"):
        for r in [a.region] if a.region else REGIONS:
            (crests if a.stage == "crests" else network)(r, *([a.sample] if a.stage == "crests" else []))
    elif a.stage == "flights":
        import run_flights  # noqa: F401  (sets one thread per process)

        flights(str(run_flights.newest_data()), a.sample, a.jobs)
    else:
        report()
