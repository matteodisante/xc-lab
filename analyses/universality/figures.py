"""The figures of universality.tex, all in one style: one colour per discipline (PARA, HANG), one line style per
design (A solid, B dash-dot), 8-pt serif text of the document (usetex, mathpazo), panels labelled (a), (b), ...,
the legend of a figure once, drawn at its print width (TEXT_IN). Data: results.json, takeoffs.npz, example_flight.npz."""

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

import regions
import terrain

BASEMAP = Path(__file__).resolve().parents[3] / "xc-viewer/src/xc_viewer/assets/basemap.json"
PARA, HANG = "#2a78d6", "#eb6834"  # blue and orange: told apart by every colour vision
COLORS = {"para": PARA, "hang": HANG}
CLASS_COLORS = {"plains": "#e8dd8e", "hills": "#a9c46c", "low mountains": "#a6794f", "high mountains": "#5b3a63",
                "unknown": "#c9c9c9"}
REGION_COLORS = {"Alps": "#5b3a63", "Pyrenees": "#a6794f", "Channel Coast": "#c9a800", "Champagne-Lorraine": "#6f9a2b"}
STYLE = {"A": "-", "B": (0, (5, 1.5, 1, 1.5))}  # fixed population solid, all flights dash-dot
DESIGNS = {"A_fixed": "A, $T=1000$ s", "B_to_1000": "B, lags to 1000 s", "A_longest": "A, longest $T$",
           "B_to_longest": "B, lags to longest $T$"}
MARKERS = {"A_fixed": ("o", True), "B_to_1000": ("o", False), "A_longest": ("s", True), "B_to_longest": ("s", False)}
INK, GREY, LIGHT = "#222222", "#888888", "#dddddd"
TEXT_IN = 6.3  # width of the text of universality.tex, inches

plt.rcParams.update({"text.usetex": True, "text.latex.preamble": r"\usepackage[sc]{mathpazo}\usepackage{xcolor}\definecolor{para}{HTML}{2A78D6}\definecolor{hang}{HTML}{EB6834}", "font.family": "serif",
                     "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
                     "legend.fontsize": 8, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
                     "lines.linewidth": 1.2, "pdf.fonttype": 42})
DASH = (0, (3, 2))


def save(fig, path):
    path = Path(path)
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"), dpi=200)
    plt.close(fig)


def clean(ax, grid=True):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if grid:
        ax.grid(color="#eeeeee", lw=0.5)
        ax.set_axisbelow(True)


def letter(ax, i, text="", right=None):
    """The panel letter and label on the left above the axes, and `right` (text) on the right."""
    ax.set_title(r"\textbf{(%s)}%s" % ("abcdefghij"[i], " " + text if text else ""), loc="left", fontsize=8, pad=3)
    if right:
        ax.set_title(right, loc="right", fontsize=8, pad=3)


def counts_label(ax, d, names):
    """Flights of the two disciplines, each in its colour, on the right above the axes."""
    last = None
    for k in reversed(names):
        text = f"{k} {d[k]['flights'][-1]:,}".replace(",", "\\,")
        if last is None:
            last = ax.text(1.0, 1.03, text, color=COLORS[k], transform=ax.transAxes, ha="right", va="bottom")
        else:
            last = ax.annotate(text, xy=(0, 0), xycoords=last, xytext=(-8, 0), textcoords="offset points", color=COLORS[k],
                               ha="right", va="bottom")


def style_of(key):
    return STYLE[key[0]]


# ---- alpha(tau) cells ---------------------------------------------------------------------------

def draw_alpha(ax, d, names, key, colors=COLORS):
    """alpha(tau) of the two populations of design `d`, the dashed 2H of their fits."""
    lags = np.array(d[names[0]]["lags"])
    ax.axhline(2, color=LIGHT, lw=0.8)
    ax.axhline(1, color=LIGHT, lw=0.8)
    for k in names:
        c = d[k]
        ax.fill_between(lags, *c["alpha_band"], color=colors[k], alpha=0.25, lw=0)
        ax.plot(lags, c["alpha"], color=colors[k], ls=style_of(key))
        ax.axhline(2 * c["H"], color=colors[k], lw=0.8, ls=DASH)
    ax.set_xscale("log")
    ax.set_xlim(lags[0] / 1.5, lags[-1] * 1.5)
    clean(ax, grid=False)


def draw_diff(ax, d, names, key):
    lags = np.array(d[names[0]]["lags"])
    ax.fill_between(lags, *d["d_alpha"]["band"], color=GREY, alpha=0.25, lw=0)
    ax.plot(lags, d["d_alpha"]["value"], color=INK, ls=style_of(key))
    ax.axhline(2 * d["d_H"]["value"], color=INK, lw=0.8, ls=DASH)
    ax.axhline(0, color=GREY, lw=0.8)
    ax.set_xscale("log")
    ax.set_xlim(lags[0] / 1.5, lags[-1] * 1.5)
    clean(ax, grid=False)


def draw_count(ax, d, names, key, colors=COLORS):
    for k in names:
        ax.plot(d[k]["lags"], d[k]["flights"], color=colors[k], ls=style_of(key))
    ax.set_xscale("log")
    ax.set_yscale("log")
    lags = np.array(d[names[0]]["lags"])
    ax.set_xlim(lags[0] / 1.5, lags[-1] * 1.5)
    clean(ax, grid=False)


def design_legend(fig, names, key):
    """The one legend of a design figure: disciplines, line style of the design, fit line, band."""
    h = [Line2D([], [], color=COLORS[k], label=k) for k in names if k in COLORS]
    h += [Line2D([], [], color=INK, ls=style_of(key), label="fixed population (A)" if key[0] == "A" else "all flights (B)"),
          Line2D([], [], color=GREY, ls=DASH, lw=0.8, label=r"fit, $2H$"), Patch(color=GREY, alpha=0.25, lw=0, label="90\\% band")]
    fig.legend(handles=h, loc="outside upper center", ncol=len(h), frameon=False, handlelength=1.6, columnspacing=1.0)


def design_figure(cells, names, path, key, ncols, counts=False):
    """cells: [(label, design dict)]: for each, alpha(tau), the difference of the first minus the second population
    and, with `counts`, the flights against the lag. The ranges are those of the whole figure."""
    rows = [1.6, 0.9] + ([0.7] if counts else [])
    nrows = math.ceil(len(cells) / ncols)
    fig = plt.figure(figsize=(TEXT_IN, (sum(rows) + 0.5) * nrows + 0.35), layout="constrained")
    grid = fig.add_gridspec(nrows, ncols)
    axes = []
    for i, (label, d) in enumerate(cells):
        sub = grid[i // ncols, i % ncols].subgridspec(len(rows), 1, height_ratios=rows, hspace=0.08)
        a = [fig.add_subplot(sub[j]) for j in range(len(rows))]
        draw_alpha(a[0], d, names, key)
        draw_diff(a[1], d, names, key)
        if counts:
            draw_count(a[2], d, names, key)
        for j in range(len(rows) - 1):
            a[j].sharex(a[-1])
            a[j].tick_params(labelbottom=False)
        a[-1].set_xlabel(r"$\tau$ [s]")
        letter(a[0], i, label)
        if ncols <= 2 and not counts and names[0] in COLORS:
            counts_label(a[0], d, names)
        if i % ncols == 0:
            for ax, text in zip(a, (r"$\alpha(\tau)$", r"$\Delta\alpha$", "flights")):
                ax.set_ylabel(text)
        axes.append(a)
    for r in range(len(rows)):
        for i, a in enumerate(axes):
            if i:
                a[r].sharey(axes[0][r])
            if i % ncols:
                a[r].tick_params(labelleft=False)
    v = np.concatenate([d[k]["alpha"] for _, d in cells for k in names])
    axes[0][0].set_ylim(v.min() - 0.05, v.max() + 0.05)
    v = np.concatenate([np.r_[d["d_alpha"]["value"], 2 * d["d_H"]["value"], 0] for _, d in cells])
    axes[0][1].set_ylim(v.min() - 0.05, v.max() + 0.05)
    if counts:
        v = np.concatenate([d[k]["flights"] for _, d in cells for k in names])
        axes[0][2].set_ylim(v.min() / 1.6, v.max() * 1.6)
    design_legend(fig, names, key)
    save(fig, path)


# ---- summary figures ----------------------------------------------------------------------------

def forest(rows, path, xlabel, width=TEXT_IN * 0.62):
    """Difference of H with its 90% band, one line per design (marker) and row (label, {design: d_H})."""
    fig, ax = plt.subplots(figsize=(width, 0.55 + 0.62 * len(rows)), layout="constrained")
    keys = list(DESIGNS)
    offsets = np.linspace(-0.27, 0.27, len(keys))
    for i, (label, ds) in enumerate(rows):
        for off, k in zip(offsets, keys):
            if k not in ds:
                continue
            y = -i - off
            ax.plot(ds[k]["band"], [y, y], color=INK, lw=1.0, solid_capstyle="butt")
            marker, filled = MARKERS[k]
            ax.plot(ds[k]["value"], y, marker, color=INK, ms=4, mfc=INK if filled else "white", mew=0.8)
    ax.axvline(0, color=GREY, lw=0.8)
    ax.set(yticks=-np.arange(len(rows)), yticklabels=[r[0] for r in rows], xlabel=xlabel)
    ax.set_ylim(-len(rows) + 0.5, 0.6)
    for i in range(1, len(rows)):
        ax.axhline(-i + 0.5, color=LIGHT, lw=0.6)
    h = [Line2D([], [], marker=MARKERS[k][0], ls="", color=INK, ms=4, mfc=INK if MARKERS[k][1] else "white", mew=0.8,
                label=DESIGNS[k]) for k in keys]
    clean(ax, grid=False)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    return fig, ax, h


def forest_figure(res, path):
    """The main summary: H para minus H hang, per terrain class and region, in the four designs."""
    rows = []
    sources = [(s, r) for s, r in res["strata"].items()] + list(res["regions"]["para_vs_hang"].items())
    for s, r in sources:
        if "skipped" not in r:
            rows.append((f"{s}\\\\ ({r['flights']['hang']:,} hang)".replace(",", "\\,"), {k: r[k]["d_H"] for k in DESIGNS if k in r}))
    fig, ax, h = forest(rows, path, r"$H_{\rm para}-H_{\rm hang}$")
    fig.legend(handles=h, loc="outside upper center", ncol=2, frameon=False)
    save(fig, path)


def region_figure(res, path):
    """(a) alpha(tau) of the paragliders of the four regions at T = 1000 s; (b) the difference of H between regions."""
    fig = plt.figure(figsize=(TEXT_IN, 3.6), layout="constrained")
    ga = fig.add_gridspec(1, 2, width_ratios=[1, 1.15])
    ax = fig.add_subplot(ga[0])
    pairs = res["regions"]["between"]["para"]
    ref = next(v for v in pairs.values() if "skipped" not in v)["A_fixed"]
    lags = np.array(next(c for c in ref.values() if isinstance(c, dict) and "lags" in c)["lags"])
    ax.axhline(2, color=LIGHT, lw=0.8)
    seen = {}
    for v in pairs.values():
        if "skipped" in v:
            continue
        for k in v["A_fixed"]:
            if k in REGION_COLORS and k not in seen:
                seen[k] = v["A_fixed"][k]
    for k, c in seen.items():
        ax.fill_between(lags, *c["alpha_band"], color=REGION_COLORS[k], alpha=0.25, lw=0)
        ax.plot(lags, c["alpha"], color=REGION_COLORS[k], label=f"{k} ({c['flights'][-1]:,})".replace(",", "\\,"))
        ax.axhline(2 * c["H"], color=REGION_COLORS[k], lw=0.8, ls=DASH)
    ax.set(xscale="log", xlabel=r"$\tau$ [s]", ylabel=r"$\alpha(\tau)$")
    ax.set_xlim(lags[0] / 1.5, lags[-1] * 1.5)
    ax.legend(frameon=False, loc="lower left", handlelength=1.4)
    clean(ax, grid=False)
    letter(ax, 0, "paragliders, $T=1000$ s")
    rows = [(n.replace(" | ", " $-$ "), {k: r[k]["d_H"] for k in DESIGNS if k in r}) for n, r in pairs.items() if "skipped" not in r]
    sub = fig.add_subplot(ga[1])
    keys, offsets = list(DESIGNS), np.linspace(-0.27, 0.27, 4)
    for i, (label, ds) in enumerate(rows):
        for off, k in zip(offsets, keys):
            if k in ds:
                y = -i - off
                sub.plot(ds[k]["band"], [y, y], color=INK, lw=1.0)
                sub.plot(ds[k]["value"], y, MARKERS[k][0], color=INK, ms=4, mfc=INK if MARKERS[k][1] else "white", mew=0.8)
    sub.axvline(0, color=GREY, lw=0.8)
    sub.set(yticks=-np.arange(len(rows)), yticklabels=[r[0] for r in rows], xlabel=r"$H$ first $-$ $H$ second (paragliders)")
    sub.set_ylim(-len(rows) + 0.5, 0.6)
    sub.tick_params(axis="y", length=0)
    clean(sub, grid=False)
    sub.spines["left"].set_visible(False)
    letter(sub, 1, "difference between regions")
    h = [Line2D([], [], marker=MARKERS[k][0], ls="", color=INK, ms=4, mfc=INK if MARKERS[k][1] else "white", mew=0.8,
                label=DESIGNS[k]) for k in keys]
    fig.legend(handles=h, loc="outside upper right", ncol=4, frameon=False, handletextpad=0.2, columnspacing=0.8)
    save(fig, path)


def fixed_vs_all_figure(res, path):
    """alpha(tau) of each discipline, all terrain, to the longest T: fixed population against all flights."""
    raw = res["strata"]["all"]
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_IN, 2.5), sharey=True, layout="constrained")
    for i, (ax, k) in enumerate(zip(axes, COLORS)):
        for key in ("A_longest", "B_to_longest"):
            c, lags = raw[key][k], np.array(raw[key][k]["lags"])
            ax.fill_between(lags, *c["alpha_band"], color=COLORS[k], alpha=0.25, lw=0)
            ax.plot(lags, c["alpha"], color=COLORS[k], ls=style_of(key))
            ax.axhline(2 * c["H"], color=COLORS[k], lw=0.8, ls=DASH if key[0] == "A" else (0, (1, 2)))
        ax.axhline(2, color=LIGHT, lw=0.8)
        ax.set(xscale="log", xlabel=r"$\tau$ [s]")
        ax.set_xlim(lags[0] / 1.5, lags[-1] * 1.5)
        clean(ax, grid=False)
        letter(ax, i, k, rf"$T={raw['A_longest']['T']}$ s")
    axes[0].set_ylabel(r"$\alpha(\tau)$")
    axes[0].set_ylim(1.6, 2.8)
    h = [Line2D([], [], color=INK, ls=STYLE["A"], label="fixed population (A)"),
         Line2D([], [], color=INK, ls=STYLE["B"], label="all flights (B)"),
         Line2D([], [], color=GREY, lw=0.8, ls=DASH, label=r"fit, $2H$, A"), Line2D([], [], color=GREY, lw=0.8, ls=(0, (1, 2)), label=r"fit, $2H$, B")]
    fig.legend(handles=h, loc="outside upper center", ncol=4, frameon=False)
    save(fig, path)


def msd_figure(res, path):
    """MSD of the two populations, all terrain, T = 1000 s, with the straight lines of the fits."""
    d = res["strata"]["all"]["A_fixed"]
    fig, ax = plt.subplots(figsize=(TEXT_IN * 0.55, 2.7), layout="constrained")
    for k in COLORS:
        c, lags = d[k], np.array(d[k]["lags"])
        m = np.array(c["msd"])
        ax.fill_between(lags, *c["msd_band"], color=COLORS[k], alpha=0.25, lw=0)
        ax.plot(lags, m, color=COLORS[k], label=rf"{k}, $H={c['H']:.3f}$")
        b = np.mean(np.log10(m)) - 2 * c["H"] * np.mean(np.log10(lags))
        ax.plot(lags, 10**b * lags ** (2 * c["H"]), color=COLORS[k], lw=0.8, ls=DASH)
    ax.set(xscale="log", yscale="log", xlabel=r"$\tau$ [s]", ylabel=r"MSD [m$^2$]")
    ax.plot([], [], color=GREY, lw=0.8, ls=DASH, label="power-law fit")
    ax.legend(frameon=False, loc="upper left")
    clean(ax)
    save(fig, path)


# ---- terrain ------------------------------------------------------------------------------------

def class_spans(ax, edges_m, top, alpha=0.3, labels=True, rotation=90):
    """Horizontal bands (vertical for rotation 90 on an elevation x axis) of the terrain classes."""
    bounds = [0, *edges_m, top]
    for (lo, hi), name in zip(zip(bounds[:-1], bounds[1:]), terrain.CLASSES):
        ax.axvspan(lo, hi, color=CLASS_COLORS[name], alpha=alpha, lw=0)
        if labels:
            ax.text((lo + min(hi, top)) / 2, 0.97, name, transform=ax.get_xaxis_transform(), rotation=rotation, ha="center", va="top")


def histogram_figure(res, path):
    t = res["terrain"]["histogram"]
    edges = np.array(t["edges"])
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_IN, 2.3), layout="constrained")
    for i, (ax, d) in enumerate(zip(axes, COLORS)):
        class_spans(ax, res["constants"]["edges_m"], 3200)
        ax.stairs(t["n"][d], edges, fill=True, color=COLORS[d], alpha=0.9, lw=0)
        for e in res["constants"]["edges_m"]:
            ax.axvline(e, color=INK, lw=0.8, ls=DASH)
        ax.set(xlabel="median ground elevation [m]", xlim=(0, 3200), ylim=(0, 1.45 * max(t["n"][d])))
        ax.set_ylabel("flights" if i == 0 else "")
        clean(ax, grid=False)
        letter(ax, i, f"{d}, {sum(t['n'][d]):,} flights".replace(",", "\\,"))
    save(fig, path)


def example_figure(res, example, path):
    """A real flight: the ground under its 10-s points, its recorded altitude, the median and the class edges."""
    e = np.load(example)
    t, ground, z, median = e["t"] / 3600, e["ground"], e["z"], float(e["median"])
    top = 3000
    fig, ax = plt.subplots(figsize=(TEXT_IN, 2.7), layout="constrained")
    bounds = [0, *res["constants"]["edges_m"], top]
    for (lo, hi), name in zip(zip(bounds[:-1], bounds[1:]), terrain.CLASSES):
        ax.axhspan(lo, hi, color=CLASS_COLORS[name], alpha=0.3, lw=0)
        ax.text(t[-1] * 1.01, (lo + hi) / 2, name, va="center", ha="left", color=INK)
    for edge in res["constants"]["edges_m"]:
        ax.axhline(edge, color=GREY, lw=0.6, ls=DASH)
    ax.fill_between(t, 0, np.where(np.isfinite(ground), ground, 0), color="#9a9893", lw=0, label="ground under the points")
    ax.plot(t, z, color=PARA, lw=0.9, label="recorded altitude")
    ax.plot(t, ground, ".", color=INK, ms=0.8, label=r"ground at a 10-s point")
    ax.axhline(median, color=INK, lw=1.2)
    ax.text(t[-1] * 0.99, median + 40, rf"median ground elevation $={median:.0f}$ m", va="bottom", ha="right", color=INK,
            bbox={"fc": "white", "ec": "none", "alpha": 0.85, "pad": 1.5})
    ax.set(xlim=(t[0], t[-1]), ylim=(0, top), xlabel="time since takeoff [h]", ylabel="elevation [m]")
    h, lab = ax.get_legend_handles_labels()
    fig.legend(h, lab, loc="outside upper center", ncol=3, frameon=False, handlelength=1.4, columnspacing=1.0)
    clean(ax, grid=False)
    save(fig, path)


def map_figure(res, takeoffs, path):
    """Takeoffs of the flights, coloured by terrain class, and the four region boxes; paragliders and hang gliders."""
    d = np.load(takeoffs)
    rings = json.loads(BASEMAP.read_text())["panels"]
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_IN, 3.9), sharex=True, sharey=True, layout="constrained")
    for i, (ax, disc) in enumerate(zip(axes, COLORS)):
        for ring in rings["world"]["rings"] + rings["france"]["rings"]:
            r = np.asarray(ring)
            ax.plot(r[:, 0], r[:, 1], color=GREY, lw=0.4)
        m = d["discipline"] == disc
        for name in ("unknown",) + terrain.CLASSES:
            q = m & (d["cls"] == name)
            ax.scatter(d["lon"][q], d["lat"][q], s=0.5 if disc == "para" else 3, color=CLASS_COLORS[name], lw=0, rasterized=True)
        for name, (w, e, s, n) in regions.BOXES.items():
            ax.add_patch(Rectangle((w, s), e - w, n - s, fill=False, ec=INK, lw=0.7, ls=DASH))
            ax.text((w + e) / 2, n + 0.1 if name in ("Alps", "Channel Coast") else s - 0.1, name,
                    ha="center", va="bottom" if name in ("Alps", "Channel Coast") else "top", fontsize=8,
                    bbox={"fc": "white", "ec": "none", "alpha": 0.8, "pad": 0.8})
        ax.set(xlim=(-5.5, 10.6), ylim=(41.0, 51.8), xlabel=r"longitude [$^\circ$E]")
        ax.set_aspect(1 / math.cos(math.radians(46.5)))
        clean(ax, grid=False)
        letter(ax, i, f"{disc}, {m.sum():,} takeoffs".replace(",", "\\,"))
    axes[0].set_ylabel(r"latitude [$^\circ$N]")
    h = [Line2D([], [], marker="o", ls="", color=CLASS_COLORS[n], ms=5, label=n) for n in terrain.CLASSES + ("unknown",)]
    fig.legend(handles=h, loc="outside lower center", ncol=5, frameon=False, handletextpad=0.1, columnspacing=1.0)
    save(fig, path)


# ---- schematics ---------------------------------------------------------------------------------

def design_sketch(path, support=0.8):
    """Which flights enter at which lag: (a) fixed population, (b) all flights. Durations are illustrative."""
    dur = np.array([350, 700, 1500, 2600, 4300, 7000, 11000])
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_IN, 2.5), sharex=True, sharey=True, layout="constrained")
    y = np.arange(len(dur))[::-1]
    label_x = 1.5e4
    ax = axes[0]
    ok = dur >= 1000 / support
    ax.barh(y, dur, color=np.where(ok, "#3d6f6b", "#cfcfcf"), height=0.6)
    ax.axvline(1000 / support, color=INK, lw=1.0, ls=DASH)
    ax.text(1000 / support, len(dur) - 0.3, r"$T/0.8$", ha="center", va="bottom", bbox={"fc": "white", "ec": "none", "pad": 0.8})
    for yy, o in zip(y, ok):
        ax.text(label_x, yy, "enters" if o else "excluded", va="center", ha="left", color=INK if o else GREY)
    letter(ax, 0, "fixed population, $T=1000$ s")
    ax = axes[1]
    taus = np.array([100, 300, 1000, 3000])
    thr = taus / support
    n_lags = (dur[:, None] >= thr[None]).sum(axis=1)
    shades = ["#cfcfcf", "#bcd3d0", "#8db5b0", "#5e9791", "#3d6f6b"]
    ax.barh(y, dur, color=[shades[n] for n in n_lags], height=0.6)
    for t_, x in zip(taus, thr):
        ax.axvline(x, color=INK, lw=0.8, ls=DASH)
        ax.text(x, len(dur) - 0.3, rf"${t_}$", ha="center", va="bottom", bbox={"fc": "white", "ec": "none", "pad": 0.8})
    for yy, n in zip(y, n_lags):
        ax.text(label_x, yy, f"{n} of 4 lags", va="center", ha="left", color=INK)
    letter(ax, 1, r"all flights, lines at $\tau/0.8$")
    for ax in axes:
        ax.set(xscale="log", xlim=(100, 8e4), yticks=[], xlabel="longest piece of the flight [s]")
        ax.set_xticks([1e3, 1e4])
        clean(ax, grid=False)
        ax.spines["left"].set_visible(False)
        ax.set_ylim(-0.6, len(dur) + 0.5)
    axes[0].set_ylabel("flights")
    save(fig, path)


def opening_sketch(path):
    """Opening = takeoff-to-landing distance over the path flown: a cross-country flight and one that returns."""
    s = np.linspace(0, 1, 400)
    cross = np.c_[12 * s, 2.5 * np.sin(2 * np.pi * s) + 0.6 * np.sin(14 * np.pi * s) + 3 * s]
    s2 = np.linspace(0, 1, 400)
    back = np.c_[2.8 * np.sin(2 * np.pi * s2) * 1.2 + 0.8 * s2, 2.2 * (1 - np.cos(2 * np.pi * s2)) + 0.25 * np.sin(16 * np.pi * s2) * s2 + 0.5 * s2]
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_IN, 2.2), layout="constrained")
    for i, (ax, p, title) in enumerate(zip(axes, (cross, back), ("flight kept", "flight dropped"))):
        path_km = np.hypot(*np.diff(p, axis=0).T).sum()
        chord = np.hypot(*(p[-1] - p[0]))
        ax.plot(*p.T, color=PARA, lw=1.2)
        ax.plot(*zip(p[0], p[-1]), color=INK, lw=1.0, ls=DASH)
        ax.plot(*p[0], "^", color=INK, ms=6)
        ax.plot(*p[-1], "s", color=INK, ms=5)
        ax.text(p[0, 0] - 0.3, p[0, 1] - 0.3, "takeoff", ha="right", va="top")
        ax.text(p[-1, 0] + 0.3, p[-1, 1] + (0.3 if i == 0 else -0.3), "landing", ha="left" if i else "center", va="bottom" if i == 0 else "top")
        ax.set_aspect("equal")
        ax.margins(0.12)
        ax.axis("off")
        letter(ax, i, rf"{title}: opening $=d/L={chord / path_km:.2f}$")
    save(fig, path)
