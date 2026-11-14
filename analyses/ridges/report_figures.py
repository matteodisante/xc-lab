"""Figures, macros and tables of the report crests/crest_geometry.tex, from the files of the cache folder.

    uv run --with numba --with pyproj python report_figures.py [--only map,divides,...]

Writes crests/figures/*.pdf (and joins_alps.png), crests/generated/macros.tex and hgeo_table.tex.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection

import build
import curves
import fig_joins
import geo
import joined
import mountains
import ridges
import tamsd

ROOT = Path(__file__).parent
FIG, GEN = ROOT / "crests" / "figures", ROOT / "crests" / "generated"
REGIONS = {"alps": "Alps", "pyrenees": "Pyrenees"}
COLOR = {"alps": "#2A78D6", "pyrenees": "#EB6834"}
INK, GREY = "#0B0B0B", "#9A9893"
LAGS = {100: "Hundred", 250: "TwoFifty", 500: "FiveHundred", 1000: "Km", 2000: "TwoKm", 5000: "FiveKm", 10000: "TenKm"}
WINDOW_KEY = {"from 0.1 km": "From", "0.1-1 km": "Short", "1-10 km": "Mid", "beyond 10 km": "Long"}
SITES = {"ecrins": (6.30, 44.97), "maurienne": (6.98, 45.27), "pyrenees": (-0.05, 42.75)}  # lon, lat
plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "legend.fontsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 100, "savefig.bbox": "tight"})


def lambert(lon, lat):
    from pyproj import Transformer
    return Transformer.from_crs(4326, 2154, always_xy=True).transform(lon, lat)


class Data:
    """Everything the figures read, loaded once per region."""

    def __init__(self, region):
        self.d = build.folder()
        self.region = region
        self.curves = dict(np.load(self.d / f"curves_{region}.npz"))
        self.mask = dict(np.load(self.d / f"mask_{region}.npz"))
        self.edges = dict(np.load(self.d / f"edges_{region}.npz"))
        self.hand = np.load(self.d / f"hand_{region}.npy", mmap_mode="r")
        self.rows = {k: dict(np.load(self.d / f"tamsd_rows_{region}_{k}.npz")) for k in ("curves", "paths")}
        self.h = {k: pd.read_csv(self.d / f"hgeo_{region}_{k}.csv") for k in ("curves", "paths")}
        self.paths = dict(np.load(self.d / f"paths_{region}.npz"))
        self.joins = dict(np.load(self.d / f"joins_{region}.npz"))

    def curve_list(self):
        c = self.curves
        return [c["xy"][a:b] for a, b in zip(c["off"][:-1], c["off"][1:])]

    def corner_xy(self, vid):
        w = int(self.edges["shape"][1])
        x0, y0 = float(self.edges["x0"]), float(self.edges["y0"])
        return np.c_[x0 - 12.5 + 25.0 * (vid % (w + 1)), y0 + 12.5 - 25.0 * (vid // (w + 1))]


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / name)
    plt.close(fig)
    print("wrote", name, flush=True)


# ---- map ------------------------------------------------------------------------------------------------

def mask_axes(ax, D, lw=0.25, joins=False):
    m = D.mask
    left, top, cell = float(m["left"]), float(m["top"]), float(m["cell"])
    mk = m["mask"]
    ext = (left, left + cell * mk.shape[1], top - cell * mk.shape[0], top)
    ax.imshow(np.ma.masked_where(~mk, np.ones(mk.shape)), cmap="Greys", vmin=0, vmax=4, extent=ext, alpha=0.5)
    ax.add_collection(LineCollection([p[::10] for p in D.curve_list()], colors=COLOR[D.region], linewidths=lw))
    if joins:
        ax.add_collection(LineCollection(np.stack([D.joins["a"], D.joins["b"]], axis=1), colors="red", linewidths=0.6))
    ax.set_xlim(ext[0], ext[1])
    ax.set_ylim(ext[2], ext[3])
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def fig_map(data):
    A, P = data["alps"], data["pyrenees"]
    fig, axs = plt.subplots(1, 2, figsize=(6.4, 3.6), gridspec_kw={"width_ratios": [1, 1.9]})
    for ax, D, t in zip(axs, (A, P), ("Alps", "Pyrenees")):
        mask_axes(ax, D, joins=True)
        ax.set_title(t, color=COLOR[D.region])
        ax.plot([], [], color=COLOR[D.region], lw=1, label="curves")
        ax.plot([], [], color="red", lw=1, label="joins")
    axs[0].legend(loc="lower left", frameon=False)
    for ax, side in zip(axs, (1, -1)):  # 50 km scale bar, in the empty corner
        x0, x1, y0, y1 = ax.axis()
        xs = x1 - 60000 if side == 1 else x0 + 10000
        ax.plot([xs, xs + 50000], [y0 + 8000] * 2, color=INK, lw=1.5)
        ax.text(xs + 25000, y0 + 14000, "50 km", ha="center", fontsize=7)
    fig.tight_layout()
    save(fig, "map.pdf")


# ---- divides --------------------------------------------------------------------------------------------

def window_for(site, half_x=2600.0, half_y=1800.0):
    x, y = lambert(*SITES[site])
    return x - half_x, y - half_y, x + half_x, y + half_y


def base_map(ax, D, box, streams=True):
    z, (zx, zy) = geo.window(*box)
    ny, nx = z.shape
    ext = (zx - 12.5, zx + 25 * nx - 12.5, zy - 25 * ny + 12.5, zy + 12.5)
    ax.imshow(fig_joins.hillshade(z), cmap="gray", extent=ext, vmin=0, vmax=1.4)
    if streams:
        x0, y0 = float(D.edges["x0"]), float(D.edges["y0"])
        rr = np.rint((y0 - (zy - 25 * np.arange(ny))) / 25).astype(int)
        cc = np.rint(((zx + 25 * np.arange(nx)) - x0) / 25).astype(int)
        h = np.asarray(D.hand[np.ix_(rr, cc)], float)
        ax.imshow(np.ma.masked_where(h >= 20, np.ones_like(h)), cmap="Blues", alpha=0.35, extent=ext, vmin=0, vmax=1.5)
        ax.imshow(np.ma.masked_where(h > 0, np.ones_like(h)), cmap="winter", alpha=0.9, extent=ext, vmin=0, vmax=1)
    ax.set_xlim(box[0], box[2])
    ax.set_ylim(box[1], box[3])
    ax.set_xticks([])
    ax.set_yticks([])
    return z, (zx, zy)


def sides_in(D, ids, box):
    xy = D.corner_xy(ids.ravel()).reshape(-1, 2, 2)
    ok = ((xy[:, :, 0] >= box[0]) & (xy[:, :, 0] <= box[2]) & (xy[:, :, 1] >= box[1]) & (xy[:, :, 1] <= box[3])).all(axis=1)
    return xy[ok]


def profile(z, zx, zy, ids, D):
    """Elevations on the axis through the two cells of a divide side, XSEC_CELLS + 2 cells each way."""
    (xa, ya), (xb, yb) = D.corner_xy(np.array(ids))
    if abs(xa - xb) < 1:  # vertical side: the cells lie left and right
        xc, yc, dx, dy = xa, (ya + yb) / 2, 1, 0
    else:
        xc, yc, dx, dy = (xa + xb) / 2, ya, 0, 1
    k = np.arange(-ridges.XSEC_CELLS - 1, ridges.XSEC_CELLS + 1)  # cells -n..-1 on one side, 0..n on the other
    xs, ys = xc + dx * (k + 0.5) * 25, yc - dy * (k + 0.5) * 25
    c, r = np.rint((xs - zx) / 25).astype(int), np.rint((zy - ys) / 25).astype(int)
    return (k + 0.5) * 25, z[r, c]


def fig_divides(data):
    D = data["alps"]
    box = window_for("ecrins")
    e = D.edges
    acc, flat = sides_in(D, e["acc"], box), sides_in(D, e["flat"], box)
    fig = plt.figure(figsize=(6.4, 3.3))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.5, 1], hspace=0.45, wspace=0.28)
    ax = fig.add_subplot(gs[:, 0])
    z, (zx, zy) = base_map(ax, D, box)
    ax.add_collection(LineCollection(flat, colors="#f1c40f", linewidths=0.6))
    ax.add_collection(LineCollection(acc, colors="red", linewidths=1.0))
    ax.plot([], [], color="red", lw=1, label="passes both tests")
    ax.plot([], [], color="#f1c40f", lw=1, label="fails the cross-section only")
    ax.plot([], [], color="#00e68a", lw=2, label="valley cells")
    ax.legend(loc="lower left", frameon=True, framealpha=0.85, fontsize=6)
    ax.plot([box[0] + 300, box[0] + 1300], [box[1] + 150] * 2, color="white", lw=1.5)
    ax.text(box[0] + 800, box[1] + 260, "1 km", color="white", ha="center", fontsize=7)
    ax.set_title("a")
    # two profiles: an accepted side and a side failing the cross-section
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    pick = lambda ids: ids[np.argmin(((D.corner_xy(ids[:, 0]) - [cx, cy]) ** 2).sum(axis=1))]
    ids_in = lambda arr: arr[(lambda xy: ((xy[:, 0] > box[0]) & (xy[:, 0] < box[2]) & (xy[:, 1] > box[1]) & (xy[:, 1] < box[3])))(D.corner_xy(arr[:, 0]))]
    for k, (ids, name) in enumerate(((ids_in(e["acc"]), "crest side"), (ids_in(e["flat"]), "side failing the cross-section"))):
        a = fig.add_subplot(gs[k, 1])
        if len(ids):
            s = pick(ids)
            if k == 1:  # a flat one: the largest cross-section deficit is not needed, the nearest suffices
                pass
            u, p = profile(z, zx, zy, s, D)
            top = max(p[ridges.XSEC_CELLS], p[ridges.XSEC_CELLS + 1])
            a.plot(u, p - top, "o-", color=INK, ms=3, lw=0.8)
            a.axhline(-ridges.XSEC_DROP, color="red", lw=0.8, ls="--")
            a.axvline(0, color=GREY, lw=0.6)
            for sgn in (-1, 1):
                a.axvline(sgn * (ridges.XSEC_CELLS * 25 + 12.5), color=GREY, lw=0.6, ls=":")
        a.set_title(name, fontsize=7)
        a.set_ylabel("$z-z_{\\mathrm{crest}}$ (m)")
        if k == 1:
            a.set_xlabel("distance across the side (m)")
    fig.text(0.62, 0.93, "b", fontweight="bold")
    save(fig, "divides.pdf")


# ---- curves ---------------------------------------------------------------------------------------------

def fig_curves(data):
    D = data["alps"]
    box = window_for("ecrins")
    e = D.edges
    acc = sides_in(D, e["acc"], box)
    br = curves.bridge(e["acc"], e["flat"])[len(e["acc"]):]
    bridged = sides_in(D, br, box)
    fig, axs = plt.subplots(1, 3, figsize=(6.6, 2.5), gridspec_kw={"width_ratios": [1.3, 1.3, 1]})
    for a in axs[:2]:
        a.set_facecolor("white")
        a.set_xlim(box[0], box[2])
        a.set_ylim(box[1], box[3])
        a.set_aspect("equal")
        a.set_xticks([])
        a.set_yticks([])
    axs[0].add_collection(LineCollection(acc, colors="red", linewidths=0.8))
    axs[0].add_collection(LineCollection(bridged, colors="#f1c40f", linewidths=0.8))
    axs[0].set_title("a  sides after the tests and the bridges", fontsize=7)
    cl = [p for p in D.curve_list() if (p[:, 0] > box[0] - 3000).any() and (p[:, 0] < box[2] + 3000).any() and (p[:, 1] > box[1] - 3000).any() and (p[:, 1] < box[3] + 3000).any()]
    cmap = plt.get_cmap("tab20")
    for i, p in enumerate(cl):
        axs[1].plot(p[:, 0], p[:, 1], color=cmap(i % 20), lw=1.0)
    axs[1].set_title("b  curves", fontsize=7)
    # zoom: sides and the spline points of one curve
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    best = min(cl, key=lambda p: ((p - [cx, cy]) ** 2).sum(axis=1).min())
    k = np.argmin(((best - [cx, cy]) ** 2).sum(axis=1))
    zb = (best[k, 0] - 75, best[k, 1] - 75, best[k, 0] + 75, best[k, 1] + 75)
    a = axs[2]
    a.add_collection(LineCollection(sides_in(D, e["acc"], zb), colors="red", linewidths=1.0))
    a.add_collection(LineCollection(sides_in(D, br, zb), colors="#f1c40f", linewidths=1.0))
    q = best[(best[:, 0] > zb[0]) & (best[:, 0] < zb[2]) & (best[:, 1] > zb[1]) & (best[:, 1] < zb[3])]
    a.plot(q[:, 0], q[:, 1], ".", color=INK, ms=4)
    a.set_xlim(zb[0], zb[2])
    a.set_ylim(zb[1], zb[3])
    a.set_aspect("equal")
    a.set_xticks([])
    a.set_yticks([])
    a.set_title("c  150 m: sides and points every 5 m", fontsize=7)
    for ax_ in axs:
        for s in ax_.spines.values():
            s.set_visible(True)
    save(fig, "curves.pdf")


# ---- joins ----------------------------------------------------------------------------------------------

def fig_joins_small(data):
    D = data["alps"]
    FIG.mkdir(parents=True, exist_ok=True)
    fig_joins.joins_figure("alps", FIG / "joins_alps.png", n=24, ncols=6,
                           loaded=(D.curve_list(), D.joins, D.hand, float(D.curves["x0"]), float(D.curves["y0"])))
    print("wrote joins_alps.png", flush=True)


# ---- trees and paths ------------------------------------------------------------------------------------

def fig_trees(data):
    D = data["alps"]
    J = D.joins
    cl = D.curve_list()
    p = J["big_path"]
    fig = plt.figure(figsize=(6.4, 2.9))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.5, 1, 1], wspace=0.5)
    ax = fig.add_subplot(gs[0])
    for c in J["big_curves"]:
        ax.plot(cl[c][:, 0], cl[c][:, 1], color=COLOR["alps"], lw=0.5)
    for a_, b_ in zip(J["a"][J["tree"] == J["big_tree"]], J["b"][J["tree"] == J["big_tree"]]):
        ax.plot([a_[0], b_[0]], [a_[1], b_[1]], color="red", lw=1.0)
    ax.plot(p[:, 0], p[:, 1], color=INK, lw=1.1)
    ax.plot(*p[0], marker="*", color="gold", mec=INK, ms=9, ls="")
    ax.plot(*p[-1], marker="o", color="white", mec=INK, ms=4, ls="")
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("a  the longest tree of the Alps", fontsize=7)
    ax.plot([], [], color=INK, lw=1.1, label="longest path")
    ax.plot([], [], color="red", lw=1, label="joins")
    ax.legend(loc="lower left", frameon=False, fontsize=6)
    # b: lengths of paths
    ax = fig.add_subplot(gs[1])
    for r in REGIONS:
        L = np.sort(data[r].rows["paths"]["length"]) / 1e3
        ax.plot(L, 1 - np.arange(len(L)) / len(L), color=COLOR[r], label=REGIONS[r])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("path length $L$ (km)")
    ax.set_ylabel("share of paths longer than $L$")
    ax.legend(frameon=False)
    ax.set_title("b", fontsize=7, loc="left")
    # c: trees by longest path
    ax = fig.add_subplot(gs[2])
    for r in REGIONS:
        rw = data[r].rows["paths"]
        g, L = rw["group"], rw["length"]
        top = np.array([L[g == t].max() for t in np.unique(g)]) / 1e3
        top = np.sort(top)
        ax.plot(top, len(top) - np.arange(len(top)), color=COLOR[r])
    ax.axvline(40, color=GREY, lw=0.6, ls=":")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("longest path of a tree $\\geq L$ (km)")
    ax.set_ylabel("number of trees")
    ax.set_title("c", fontsize=7, loc="left")
    save(fig, "trees.pdf")


# ---- TAMSD results --------------------------------------------------------------------------------------

def ghost(ax, lo=100.0):
    ax.axvspan(1, lo, color="#e6e5e1", lw=0)


def fig_tamsd(data):
    fig, axs = plt.subplots(1, 3, figsize=(6.8, 2.6), gridspec_kw={"width_ratios": [1, 1, 0.9], "wspace": 0.4})
    for r in REGIONS:
        for kind, ls in (("paths", "-"), ("curves", "--")):
            t = data[r].h[kind]
            ok = np.isfinite(t["mean"])
            axs[0].plot(t.s_m[ok], np.sqrt(t["mean"][ok]) / t.s_m[ok], ls, color=COLOR[r], lw=1.1)
            okh = ok & ((t.s_m <= 1000) if kind == "curves" else True)
            axs[1].plot(t.s_m[okh], t.H[okh], ls, color=COLOR[r], lw=1.1)
            if kind == "paths":
                axs[1].fill_between(t.s_m[ok], t.H_lo[ok], t.H_hi[ok], color=COLOR[r], alpha=0.25, lw=0)
    for a in axs[:2]:
        ghost(a)
        a.set_xscale("log")
        a.set_xlim(5, 5e4)
        a.set_xlabel("arc length $s$ (m)")
        sec = a.secondary_xaxis("top", functions=(lambda s: s / (tamsd.V_KMH / 3.6), lambda t: t * tamsd.V_KMH / 3.6))
        sec.set_xlabel("lag $\\tau$ at 35 km/h (s)")
        sec.set_xscale("log")
    axs[0].set_ylabel("$\\sqrt{\\overline{\\delta^2}}\\,/\\,s$")
    axs[0].set_ylim(0.3, 1.02)
    axs[1].set_ylabel("local $H(s)$")
    axs[1].set_ylim(0.6, 1.02)
    for r in REGIONS:
        axs[0].plot([], [], color=COLOR[r], label=REGIONS[r])
    axs[0].plot([], [], "-", color=INK, label="paths")
    axs[0].plot([], [], "--", color=INK, label="curves")
    axs[0].legend(frameon=False, loc="lower left")
    # windows
    w = pd.read_csv(data["alps"].d / "hgeo_windows.csv")
    xs = {"from 0.1 km": 0, "0.1-1 km": 1, "1-10 km": 2, "beyond 10 km": 3}
    for i, r in enumerate(REGIONS):
        for kind, fill, dx in (("paths", True, -0.12), ("curves", False, 0.12)):
            sub = w[(w.region == r) & (w.kind == kind)]
            x = np.array([xs[n] for n in sub.window]) + dx + (i - 0.5) * 0.2
            axs[2].errorbar(x, sub.H, yerr=[sub.H - sub.ci_lo, sub.ci_hi - sub.H], fmt="o", ms=4, lw=0.9, color=COLOR[r],
                            mfc=COLOR[r] if fill else "white", capsize=1.5)
    axs[2].set_xticks(range(4))
    axs[2].set_xticklabels(["from\n0.1 km", "0.1-\n1 km", "1-\n10 km", "beyond\n10 km"])
    axs[2].set_ylabel("$H_{\\mathrm{geo}}$ on the window")
    for a, t in zip(axs, "abc"):
        a.set_title(t, loc="left", fontsize=8, fontweight="bold")
    save(fig, "tamsd.pdf")


# ---- limits ---------------------------------------------------------------------------------------------

def fig_limits(data):
    D = data["alps"]
    ed = np.array([p[0] for p, e in zip(D.curve_list(), D.curves["edge"]) if e])
    g = np.floor(ed / 10000).astype(int)
    cells, cnt = np.unique(g, axis=0, return_counts=True)
    cx, cy = (cells[np.argmax(cnt)] + 0.5) * 10000
    box = (cx - 7000, cy - 5000, cx + 7000, cy + 5000)
    fig = plt.figure(figsize=(6.6, 2.7))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.4, 1, 1], wspace=0.4)
    ax = fig.add_subplot(gs[0])
    z, (zx, zy) = geo.window(*box)
    ny, nx = z.shape
    ext = (zx - 12.5, zx + 25 * nx - 12.5, zy - 25 * ny + 12.5, zy + 12.5)
    ax.imshow(np.where(np.isnan(z), 1.0, np.nan), cmap="Reds", vmin=0, vmax=2, extent=ext, alpha=0.5)
    ax.imshow(fig_joins.hillshade(np.nan_to_num(z)) , cmap="gray", extent=ext, vmin=0, vmax=1.4, alpha=np.where(np.isnan(z), 0, 1))
    c = D.curves
    for p, ed in zip(D.curve_list(), c["edge"]):
        if (p[:, 0] > box[0]).any() and (p[:, 0] < box[2]).any() and (p[:, 1] > box[1]).any() and (p[:, 1] < box[3]).any():
            ax.plot(p[:, 0], p[:, 1], color="#e8a317" if ed else COLOR["alps"], lw=1.0)
    ax.set_xlim(box[0], box[2])
    ax.set_ylim(box[1], box[3])
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("a  data edge (red: no data)", fontsize=7)
    ax.plot([], [], color="#e8a317", label="reaches the edge")
    ax.plot([], [], color=COLOR["alps"], label="other curves")
    ax.legend(loc="lower left", frameon=True, framealpha=0.85, fontsize=6)
    # b: local H near the grid scale
    ax = fig.add_subplot(gs[1])
    for r in REGIONS:
        for kind, ls in (("paths", "-"), ("curves", "--")):
            t = data[r].h[kind]
            ok = np.isfinite(t.H) & (t.s_m <= 1000)
            ax.plot(t.s_m[ok], t.H[ok], ls, color=COLOR[r], lw=1.1)
    ax.axvspan(0, 100, color="#e6e5e1", lw=0)
    ax.axvline(100, color=GREY, lw=0.6)
    ax.set_xlabel("arc length $s$ (m)")
    ax.set_ylabel("local $H(s)$")
    ax.set_title("b", fontsize=7, loc="left")
    # c: trees at the lag
    ax = fig.add_subplot(gs[2])
    for r in REGIONS:
        rw = data[r].rows["paths"]
        rows, g, s = rw["rows"], rw["group"], rw["s_m"]
        nt = np.array([len(np.unique(g[np.isfinite(rows[:, j])])) for j in range(rows.shape[1])])
        npth = np.isfinite(rows).sum(axis=0)
        ax.plot(s, nt, color=COLOR[r], lw=1.1)
        ax.plot(s, npth, ":", color=COLOR[r], lw=1.0)
    ax.axhline(tamsd.MIN_CURVES, color=GREY, lw=0.6)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("arc length $s$ (m)")
    ax.set_ylabel("trees (solid), paths (dotted)")
    ax.set_title("c", fontsize=7, loc="left")
    save(fig, "limits.pdf")


# ---- macros and tables ----------------------------------------------------------------------------------

def fmt_int(n):
    s = f"{int(round(n)):,}".replace(",", "\\,")
    return s


def low_cell(D, a, b):
    """1 if a join touches a cell below the height limit on a dense sampling every 0.25 m (an independent check)."""
    n = int(np.ceil(np.hypot(*(b - a)) / 0.25))
    p = a + np.linspace(0, 1, n + 1)[:, None] * (b - a)
    x0, y0 = float(D.edges["x0"]), float(D.edges["y0"])
    c, r = np.rint((p[:, 0] - x0) / 25).astype(int), np.rint((y0 - p[:, 1]) / 25).astype(int)
    return int(np.asarray(D.hand[r, c]).min() < ridges.HAND_MIN)


def macros(data):
    out = []

    def add(name, value):
        out.append(f"\\newcommand{{\\Cg{name}}}{{{value}}}")

    h3 = lambda v: f"{v:.3f}"
    w = pd.read_csv(data["alps"].d / "hgeo_windows.csv")
    peaks = json.loads((data["alps"].d / "peak_ram.json").read_text())
    for r, R in (("alps", "Alps"), ("pyrenees", "Pyr")):
        D = data[r]
        c, e = D.curves, D.edges
        L = c["length"]
        add(R + "Area", fmt_int(D.mask["mask"].sum()))
        add(R + "Cells", f"{float(e['shape'][0]) * float(e['shape'][1]) / 1e6:.0f}")
        add(R + "Curves", fmt_int(len(L)))
        add(R + "CurveKm", fmt_int(L.sum() / 1e3))
        add(R + "CurveMedianKm", f"{np.median(L) / 1e3:.2f}")
        add(R + "Sides", fmt_int(len(e["acc"])))
        add(R + "Flat", fmt_int(len(e["flat"])))
        add(R + "Bridged", fmt_int(len(curves.bridge(e["acc"], e["flat"])) - len(e["acc"])))
        add(R + "EdgeCurves", fmt_int(c["edge"].sum()))
        rw = D.rows["paths"]
        J = D.joins
        add(R + "Joins", fmt_int(len(J["length"])))
        add(R + "JoinMedianM", f"{np.median(J['length']):.0f}")
        add(R + "JoinsLow", str(sum(low_cell(D, a_, b_) for a_, b_ in zip(J["a"], J["b"]))))
        add(R + "Trees", fmt_int(len(np.unique(rw["group"]))))
        add(R + "Paths", fmt_int(len(rw["length"])))
        add(R + "PathMedianKm", f"{np.median(rw['length']) / 1e3:.0f}")
        add(R + "PathLongestKm", f"{rw['length'].max() / 1e3:.0f}")
        add(R + "PathLongKm", fmt_int((rw["length"] >= 40000).sum()))
        add(R + "EdgePaths", fmt_int(rw["edge"].sum()))
        add(R + "LongPaths", fmt_int(len(D.paths["tree"])))
        j = len(np.unique(rw["group"][np.isfinite(rw["rows"][:, np.argmin(abs(rw["s_m"] - 10000))])]))
        add(R + "TreesTenKm", str(j))
        for kind in ("paths", "curves"):
            K = kind.capitalize()
            for noedge in ("", "_noedge"):
                sub = w[(w.region == r) & (w.kind == kind + noedge)]
                for _, row in sub.iterrows():
                    key = R + K + ("Noedge" if noedge else "") + WINDOW_KEY[row.window]
                    add(key, h3(row.H) if np.isfinite(row.H) else "--")
                    add(key + "Lo", h3(row.ci_lo) if np.isfinite(row.ci_lo) else "--")
                    add(key + "Hi", h3(row.ci_hi) if np.isfinite(row.ci_hi) else "--")
            t = D.h[kind]
            for lag, word in LAGS.items():
                i = int(np.argmin(abs(t.s_m - lag)))
                if np.isfinite(t.H[i]):
                    add(f"{R}{K}H{word}", h3(t.H[i]))
                    add(f"{R}{K}H{word}Lo", h3(t.H_lo[i]))
                    add(f"{R}{K}H{word}Hi", h3(t.H_hi[i]))
                    add(f"{R}{K}Ratio{word}", f"{np.sqrt(t['mean'][i]) / t.s_m[i]:.2f}")
        for stage in ("drain", "curves", "hgeo"):
            add(R + "Ram" + stage.capitalize(), f"{peaks[f'{r}_{stage}']:.1f}")
    add("SecHundred", f"{100 / (tamsd.V_KMH / 3.6):.1f}")
    below = [data[r].h[k][(data[r].h[k].s_m < 100)].H.min() for r in REGIONS for k in ("paths", "curves")]
    add("GridHmin", f"{np.floor(min(below) * 100) / 100:.2f}")
    # parameters
    P = dict(Valley=ridges.VALLEY_KM2, ValleyCells=ridges.valley_cells(), XsecCells=ridges.XSEC_CELLS,
             XsecM=ridges.XSEC_CELLS * 25, XsecDrop=int(ridges.XSEC_DROP), HandMin=int(ridges.HAND_MIN),
             BridgeSides=curves.BRIDGE_EDGES, BridgeM=curves.BRIDGE_EDGES * 25, PruneSides=curves.PRUNE_EDGES,
             PruneM=curves.PRUNE_EDGES * 25, Deflection=int(curves.MAX_DEFLECTION), DirectionM=int(curves.DIRECTION_M),
             MinLengthKm=int(curves.MIN_LENGTH / 1000), Step=int(curves.STEP), Smooth=int(curves.SMOOTH_M),
             EdgeCells=curves.EDGE_CELLS, EdgeM=curves.EDGE_CELLS * 25, JoinM=int(joined.JOIN_M), Cone=int(joined.CONE_DEG),
             Snap=int(joined.SNAP_M), JoinComponents=joined.N_COMPONENTS, Runout=int(joined.RUNOUT_M),
             Lags=tamsd.N_LAGS, MinCurves=tamsd.MIN_CURVES, Boot=tamsd.N_BOOT, Speed=int(tamsd.V_KMH),
             LagFrac=int(tamsd.MAX_FRAC * 100), FitFrom=int(tamsd.FIT_FROM), MaskShare=int(build.MASK_SHARE * 100),
             MountainRadius=mountains.RADIUS, MountainRange=int(mountains.LER), BasinKm=mountains.BASIN_KM2,
             Workers=build.WORKERS, LongPathKm=10)
    for k, v in P.items():
        add("P" + k, v)
    try:
        r = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider", str(ROOT)],
                           capture_output=True, text=True, cwd=ROOT)
        names = [l for l in r.stdout.splitlines() if "::" in l]
        mine = ("ridges", "curves", "joined", "tamsd", "mountains", "geo")
        count = {f: sum(1 for n in names if f"test_{f}.py::" in n) for f in mine}
        add("NTests", sum(count.values()))
        for f in mine:
            add("NTests" + f.capitalize(), count[f])
    except Exception as ex:
        print("pytest collection failed", ex)
    GEN.mkdir(parents=True, exist_ok=True)
    (GEN / "macros.tex").write_text("\n".join(out) + "\n")
    # tables of H_geo per window; the last row has no line end, the report adds it after \\input
    names = {"from 0.1 km": "from 0.1~km", "0.1-1 km": "0.1--1~km", "1-10 km": "1--10~km", "beyond 10 km": "beyond 10~km"}

    def cell(r, k, win):
        q = w[(w.region == r) & (w.kind == k) & (w.window == win)]
        return "--" if not np.isfinite(q.H.iloc[0]) else f"{q.H.iloc[0]:.3f} ({q.ci_lo.iloc[0]:.3f}--{q.ci_hi.iloc[0]:.3f})"

    for fname, suffix in (("hgeo_table.tex", ""), ("hgeo_noedge_table.tex", "_noedge")):
        rows = [f"{names[win]} & " + " & ".join(cell(r, k + suffix, win) for r in REGIONS for k in ("paths", "curves"))
                for win in names]
        (GEN / fname).write_text(" \\\\\n".join(rows) + "\n")
    print("wrote macros and tables", flush=True)


STEPS = {"map": fig_map, "divides": fig_divides, "curves": fig_curves, "joins": fig_joins_small, "trees": fig_trees,
         "tamsd": fig_tamsd, "limits": fig_limits, "macros": macros}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=",".join(STEPS))
    a = ap.parse_args()
    data = {r: Data(r) for r in REGIONS}
    for k in a.only.split(","):
        STEPS[k](data)
