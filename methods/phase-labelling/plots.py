"""Figures of the note: uv run python methods/phase-labelling/plots.py from the root of xc-lab.
Needs out/cadence.csv from cadence.py and the flight disk."""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cadence  # noqa: E402
import phases  # noqa: E402

FIGURES = HERE / "figures"
EXAMPLE = 20356685  # a 3-h paraglider flight of the cadence sample, with search phases
# Phase colours checked for colour-vision deficiency, as in the paper: transition blue,
# search red-orange, climb green; every figure also names the phases.
PHASE = {0: "#1baf7a", 1: "#eb6834", 2: "#2a78d6"}
NAME = {0: "climb", 1: "search", 2: "transition"}
INK, INK2, MUTED, LIGHT = "#0b0b0b", "#52514e", "#9a9893", "#e6e5e1"
LIN, CUB = "#eb6834", "#2a78d6"


def setup(size=9):
    plt.rcParams.update({
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{mathpazo}\usepackage{amsmath}\usepackage{amssymb}",
        "font.family": "serif", "font.size": size, "axes.titlesize": size, "axes.labelsize": size,
        "xtick.labelsize": size - 1, "ytick.labelsize": size - 1, "legend.fontsize": size - 1,
        "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
        "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
        "figure.facecolor": "white", "savefig.facecolor": "white",
    })


def flight(fid, altitude=False):
    """The longest segment of a flight after the xc-dynamics pre-processing: times and the
    east-north-up positions of Vilpellet's frame, and with altitude=True the recorded GNSS
    altitude too, which the up coordinate falls below by d**2/2R away from the first fix."""
    cfg = yaml.safe_load((cadence.DYNAMICS / "configs/preprocessing.yaml").read_text())
    paths = pd.read_csv(cadence.ROOT / "catalog/catalog.csv", usecols=["flight_id", "local_path"]).set_index("flight_id").local_path
    fixes, _ = cadence.preprocessed(cadence.TRACKS / paths[fid], cfg)
    seg = fixes[fixes.segment == fixes.segment.value_counts().idxmax()]
    xyz = np.vstack(phases.enu(seg.lat.to_numpy(), seg.lon.to_numpy(), seg.gnss_alt.to_numpy()))
    t = np.round(seg.t.to_numpy(float))
    return (t, xyz, seg.gnss_alt.to_numpy()) if altitude else (t, xyz)


def strip(ax, x, states, y, h=0.7):
    """Coloured runs of a state sequence at height y, in the units of x."""
    dx = np.median(np.diff(x))
    edges = np.flatnonzero(np.diff(states) != 0) + 1
    for a, b in zip(np.r_[0, edges], np.r_[edges, len(states)]):
        if states[a] >= 0:
            ax.add_patch(Rectangle((x[a], y - h / 2), x[b - 1] - x[a] + dx, h, color=PHASE[states[a]], lw=0))


def phase_legend(ax, **kw):
    for s in (2, 1, 0):
        ax.plot([], [], color=PHASE[s], lw=4, label=NAME[s])
    ax.legend(frameon=False, handlelength=1.2, **kw)


def flight_figure(t, xyz, z, states, zoom):
    """(a) plan view of the whole piece, (b) the minutes of zoom, (c) the altitude in time."""
    def coloured(ax, x, y, s, lw):
        pts = np.column_stack([x, y]).reshape(-1, 1, 2)
        ax.add_collection(LineCollection(np.concatenate([pts[:-1], pts[1:]], axis=1),
                                         colors=[PHASE.get(v, LIGHT) for v in s[:-1]], lw=lw))
        ax.autoscale()

    fig = plt.figure(figsize=(6.3, 4.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.35], height_ratios=[1.25, 1], wspace=0.22, hspace=0.45)
    aa, az, ah = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])
    inside = (t >= zoom[0]) & (t <= zoom[1])
    coloured(aa, xyz[0] / 1e3, xyz[1] / 1e3, states, 0.8)
    aa.set_aspect("equal")
    e, n = xyz[0, inside] / 1e3, xyz[1, inside] / 1e3
    aa.add_patch(Rectangle((e.min() - 0.6, n.min() - 0.6), np.ptp(e) + 1.2, np.ptp(n) + 1.2, fill=False, ec=INK,
                           lw=0.8))
    aa.set_xlabel("east (km)", labelpad=1)
    aa.set_ylabel("north (km)", labelpad=1)
    aa.set_title("(a) plan view", loc="left", pad=3)

    coloured(az, xyz[0, inside], xyz[1, inside], states[inside], 1.3)
    az.set_aspect("equal")
    az.set_xticks([])
    az.set_yticks([])
    for side in ("left", "bottom"):
        az.spines[side].set_visible(False)
    x0, _ = az.get_xlim()
    y0, _ = az.get_ylim()
    az.plot([x0 + 20, x0 + 520], [y0 + 20] * 2, color=INK, lw=1)
    az.text(x0 + 270, y0 + 45, "500 m", ha="center", va="bottom", fontsize=7.5)
    first = np.flatnonzero(inside)[0]
    az.plot(xyz[0, first], xyz[1, first], "o", ms=4, mfc="white", mec=INK, mew=0.9)
    az.annotate("start", (xyz[0, first], xyz[1, first]), xytext=(-8, -12), textcoords="offset points", fontsize=7.5)
    az.set_title(rf"(b) the rectangle, {(zoom[1] - zoom[0]) / 60:.1f} minutes", loc="left", pad=3)

    hours = (t - t[0]) / 3600
    coloured(ah, hours, z - z[0], states, 0.8)
    ah.axvspan((zoom[0] - t[0]) / 3600, (zoom[1] - t[0]) / 3600, color=LIGHT, lw=0, zorder=0)
    ah.set_xlabel("time since the start of the piece (h)", labelpad=1)
    ah.set_ylabel("altitude above\nthe first fix (m)", labelpad=2)
    ah.set_title("(c) altitude; shaded: the minutes of (b)", loc="left", pad=3)
    phase_legend(ah, loc="upper right", ncols=3, bbox_to_anchor=(1.0, 1.28))
    fig.savefig(FIGURES / "flight_phases.pdf", bbox_inches="tight")
    plt.close(fig)


def features_figure(t, xyz, start, minutes=24):
    alpha = phases.ALPHA["para"]
    obs, turn, vz = phases.features(t, *xyz, alpha)
    final, path = phases.label(t, *xyz, alpha)
    sel = slice(start, start + 60 * minutes)
    tm = (t[sel] - t[sel][0]) / 60
    fig, axes = plt.subplots(4, 1, figsize=(6.3, 4.6), sharex=True,
                             gridspec_kw=dict(height_ratios=[1.3, 1.0, 0.9, 0.6], hspace=0.18))
    a0, a1, a2, a3 = axes
    a0.axhspan(-alpha, alpha, color=LIGHT, lw=0)
    a0.plot(tm, turn[sel], color=INK, lw=0.6)
    a0.set_ylabel("turn\n(rad per fix)", labelpad=2)
    a0.set_ylim(-0.6, 0.6)
    a0.text(0.3, alpha + 0.03, r"$\pm\alpha$", fontsize=8, color=INK2)
    a1.axhline(0, color=MUTED, lw=0.6)
    a1.plot(tm, vz[sel], color=INK, lw=0.8)
    a1.set_ylabel("$\\bar v_z$ over 30 s\n(m/s)", labelpad=2)
    names = ["straight", "rising", "one sign"]
    for r in range(3):
        y = 2 - r
        a2.add_patch(Rectangle((tm[0], y - 0.35), tm[-1] - tm[0], 0.7, color=LIGHT, lw=0))
        on = obs[sel, r]
        edges = np.flatnonzero(np.diff(on) != 0) + 1
        for a, b in zip(np.r_[0, edges], np.r_[edges, len(on)]):
            if on[a]:
                a2.add_patch(Rectangle((tm[a], y - 0.35), tm[b - 1] - tm[a] + 1 / 60, 0.7, color=INK2, lw=0))
        a2.text(-0.4, y, names[r], ha="right", va="center", fontsize=8)
    a2.set_ylim(-0.6, 2.6)
    a2.set_yticks([])
    a2.spines["left"].set_visible(False)
    strip(a3, tm, path[sel], 1)
    strip(a3, tm, final[sel], 0)
    a3.text(-0.4, 1, "Viterbi", ha="right", va="center", fontsize=8)
    a3.text(-0.4, 0, "after vote", ha="right", va="center", fontsize=8)
    a3.set_ylim(-0.6, 1.6)
    a3.set_yticks([])
    a3.spines["left"].set_visible(False)
    a3.set_xlim(tm[0], tm[-1])
    a3.set_xlabel("minutes", labelpad=1)
    phase_legend(a0, loc="upper right", ncols=3, bbox_to_anchor=(1.0, 1.32))
    fig.align_ylabels(axes)
    fig.savefig(FIGURES / "features.pdf", bbox_inches="tight")
    plt.close(fig)


def turn_density_figure(ids):
    alpha = phases.ALPHA["para"]
    values = {0: [], 1: [], 2: []}
    for fid in ids:
        t, xyz = flight(fid)
        for grid, vals in phases.regular_grid(t, xyz, 4.0):
            if len(grid) < 300:
                continue
            _, turn, _ = phases.features(grid, *vals, alpha)
            final, _ = phases.label(grid, *vals, alpha)
            for s in values:
                values[s].append(np.abs(turn[final == s]))
    bins = np.linspace(0, 0.8, 81)
    fig, ax = plt.subplots(figsize=(6.3, 2.5))
    for s in (2, 1, 0):
        v = np.concatenate(values[s])
        ax.stairs(np.histogram(v[np.isfinite(v)], bins=bins, density=True)[0], bins, color=PHASE[s], lw=1.2, label=NAME[s])
    ax.set_yscale("log")
    ax.axvline(alpha, color=INK, lw=0.8, ls="--")
    ax.text(alpha + 0.01, 30, rf"$\alpha={alpha:.3f}$ rad", fontsize=8)
    ax.set_xlabel(r"$|\theta|$, smoothed turn per fix (rad)", labelpad=1)
    ax.set_ylabel("density within\neach phase", labelpad=2)
    ax.set_xlim(0, 0.8)
    ax.legend(frameon=False, loc="upper right")
    fig.savefig(FIGURES / "turn_density.pdf", bbox_inches="tight")
    plt.close(fig)
    return {s: float(np.mean(np.concatenate(v)[np.isfinite(np.concatenate(v))] < alpha)) for s, v in values.items()}


def emission_figure():
    fig, ax = plt.subplots(figsize=(6.3, 1.9))
    order = [2, 1, 0]
    m = phases.EMISSION[order]
    ax.imshow(m, cmap="Greys", vmin=0, vmax=1.25, aspect="auto")
    for r in range(3):
        for c in range(8):
            v = m[r, c]
            txt = f"{v:.3f}" if v >= 0.0005 else r"$<10^{-3}$"
            ax.text(c, r, txt, ha="center", va="center", fontsize=8, color="white" if v > 0.55 else INK)
    ax.set_yticks(range(3), [NAME[s] for s in order])
    ax.set_xticks(range(8), [f"({c >> 2},{(c >> 1) & 1},{c & 1})" for c in range(8)])
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xlabel("symbol (straight, rising, one sign)", labelpad=3)
    fig.savefig(FIGURES / "emissions.pdf", bbox_inches="tight")
    plt.close(fig)


def interpolation_figure():
    rng = np.random.default_rng(3)
    t = np.arange(0, 121.0)
    period, r, wind = 26.0, 40.0, 4.0
    om = 2 * np.pi / period
    xy = np.vstack([r * np.cos(om * t) + wind * t, r * np.sin(om * t)]) + rng.normal(0, 1.0, (2, len(t)))
    z = 2.0 * t
    xyz = np.vstack([xy, z])
    keep = t % 5 == 0
    alpha = phases.ALPHA["para"]
    _, turn1, _ = phases.features(t, *xyz, alpha)
    out = {}
    for cub in (False, True):
        (grid, vals), = phases.regular_grid(t[keep], xyz[:, keep], 10.0, cub)
        _, turn, _ = phases.features(grid, *vals, alpha)
        out[cub] = (grid, vals, turn)
    fig, (ax, at) = plt.subplots(2, 1, figsize=(6.3, 4.0), gridspec_kw=dict(height_ratios=[1, 1.25], hspace=0.45))
    sl = t <= 80
    ax.plot(xy[0, sl], xy[1, sl], color=MUTED, lw=0.7, label="1 Hz fixes")
    g, v, _ = out[False]
    s2 = g <= 80
    ax.plot(v[0, s2], v[1, s2], color=LIN, lw=1.0, label="linear, from 5 s")
    g, v, _ = out[True]
    ax.plot(v[0, s2], v[1, s2], color=CUB, lw=1.0, label="cubic, from 5 s")
    k80 = keep & sl
    ax.plot(xy[0, k80], xy[1, k80], "o", ms=3.5, mfc="white", mec=INK, mew=0.8, label="kept fixes")
    ax.set_aspect("equal")
    ax.set_xlabel("east (m)", labelpad=1)
    ax.set_ylabel("north (m)", labelpad=1)
    ax.set_title("(a) a thermal drifting east with the wind, first 80 s", loc="left", pad=3)
    ax.legend(frameon=False, fontsize=7.5, loc="center left", bbox_to_anchor=(1.01, 0.5))
    at.axhspan(-alpha, alpha, color=LIGHT, lw=0)
    at.plot(t, turn1, color=MUTED, lw=1.0, label="1 Hz")
    at.plot(out[False][0], out[False][2], color=LIN, lw=1.0, label="linear")
    at.plot(out[True][0], out[True][2], color=CUB, lw=1.0, label="cubic")
    at.set_xlim(10, 110)
    at.set_ylim(-0.1, 0.7)
    at.set_xlabel("s", labelpad=1)
    at.set_ylabel("smoothed turn (rad per fix)", labelpad=2)
    at.set_title(r"(b) the turn the features see", loc="left", pad=3)
    at.text(12, alpha + 0.02, r"$\pm\alpha$", fontsize=8, color=INK2)
    at.legend(frameon=False, loc="upper right", ncols=3)
    fig.savefig(FIGURES / "interpolation.pdf", bbox_inches="tight")
    plt.close(fig)


def cadence_figure():
    d = pd.read_csv(HERE / "out/cadence.csv")
    cols = [f"c{a}{b}" for a in range(3) for b in range(3)]
    ks = [1, 2, 3, 4, 5, 10]
    per = d.groupby(["flight_id", "k", "cubic"])[cols].sum()
    m = per.to_numpy().reshape(-1, 3, 3)
    per = per.assign(agree=np.trace(m, axis1=1, axis2=2) / m.sum(axis=(1, 2)))
    tot = d.groupby(["k", "cubic"])[cols].sum()
    fig, axes = plt.subplots(1, 3, figsize=(6.3, 2.3), gridspec_kw=dict(wspace=0.4))
    for cub, color, name in ((False, LIN, "linear"), (True, CUB, "cubic")):
        agree, climb, search, lo, hi = [1.0], [1.0], [1.0], [1.0], [1.0]
        for k in ks[1:]:
            c = tot.loc[(k, cub)].to_numpy().reshape(3, 3)
            agree.append(np.trace(c) / c.sum())
            climb.append(c[0, 0] / c[0].sum())
            search.append(c[1, 1] / c[1].sum())
            q = per.xs((k, cub), level=("k", "cubic")).agree.quantile([0.1, 0.9])
            lo.append(q.iloc[0])
            hi.append(q.iloc[1])
        axes[0].fill_between(ks, lo, hi, color=color, alpha=0.15, lw=0)
        for ax, y in zip(axes, (agree, climb, search)):
            ax.plot(ks, y, "o-", color=color, ms=3.5, lw=1.2, label=name)
    for ax, title in zip(axes, ("(a) all fixes", "(b) native climb", "(c) native search")):
        ax.set_title(title, loc="left", pad=3)
        ax.set_xticks(ks)
        ax.set_xlabel("logging interval (s)", labelpad=1)
        ax.set_ylim(0.2, 1.02)
        ax.axvspan(5.5, 10.5, color=LIGHT, lw=0, zorder=0)
    axes[0].set_ylabel("share of fixes with\nthe native label", labelpad=2)
    axes[0].legend(frameon=False, loc="lower left")
    fig.savefig(FIGURES / "cadence.pdf", bbox_inches="tight")
    plt.close(fig)
    return tot


def cadence_example_figure(t, xyz, z, native):
    rows = [("1 Hz", native)]
    for cub, name in ((False, "5 s, linear"), (True, "5 s, cubic")):
        keep = (t % 5) == 0
        lab = cadence.labels_on_fixes(t[keep], xyz[:, keep], 5, cub, phases.ALPHA["para"])
        full = np.full(len(t), -1)
        full[keep] = lab
        # Each kept label covers its 5 s, so the strip reads like the native one.
        full = pd.Series(np.where(full >= 0, full, np.nan)).ffill(limit=4).fillna(-1).astype(int).to_numpy()
        rows.append((name, full))
    hours = (t - t[0]) / 3600
    fig, (ah, ast) = plt.subplots(2, 1, figsize=(6.3, 2.6), sharex=True, gridspec_kw=dict(height_ratios=[1.2, 1], hspace=0.1))
    ah.plot(hours, z - z[0], color=INK2, lw=0.7)
    ah.set_ylabel("altitude above\nthe first fix (m)", labelpad=2)
    for r, (name, lab) in enumerate(rows):
        strip(ast, hours, lab, 2 - r)
        ast.text(-0.02 * hours[-1], 2 - r, name, ha="right", va="center", fontsize=8)
    ast.set_ylim(-0.6, 2.6)
    ast.set_yticks([])
    ast.spines["left"].set_visible(False)
    ast.set_xlim(hours[0], hours[-1])
    ast.set_xlabel("hours", labelpad=1)
    phase_legend(ah, loc="upper right", ncols=3, bbox_to_anchor=(1.0, 1.3))
    fig.savefig(FIGURES / "cadence_example.pdf", bbox_inches="tight")
    plt.close(fig)


def save_frames(frames, name, size):
    """Write the animation embedded in the PDF: figures/<name>_frames.pdf, one page per frame,
    and figures/<name>_frames.tln, the timeline of the LaTeX package animate. frames is a list
    of (function drawing on a figure, seconds on screen); the last one is the static figure,
    shown by viewers that cannot play the animation."""
    fig = plt.figure(figsize=size)
    with PdfPages(FIGURES / f"{name}_frames.pdf") as pdf:
        for draw, _ in frames:
            draw(fig)
            pdf.savefig(fig)
    plt.close(fig)
    lines = [f":{1 / sec:.4g}:{n}" for n, (_, sec) in enumerate(frames)]
    (FIGURES / f"{name}_frames.tln").write_text("\n".join(lines) + "\n")


def synthetic_flight():
    """Eight minutes at 1 Hz built as a glide, a search and a climb, with 0.5 m of noise on
    every coordinate: times, east-north-up positions in metres."""
    rng = np.random.default_rng(7)
    t = np.arange(480.0)
    glide, search, climb = t < 160, (t >= 160) & (t < 300), t >= 300
    s = t - 160
    # Turn rate in rad/s: none in the glide; both ways in the search; left, one turn in 24 s, in the climb.
    rate = np.where(search, 0.32 * np.sign(np.sin(2 * np.pi * s / 17)) * (0.85 + 0.15 * np.cos(s / 11)), 0.0)
    rate = np.where(climb, 2 * np.pi / 24, rate)
    heading = np.radians(30) + np.cumsum(rate)
    speed = np.where(glide, 10.0, 8.0)
    drift = np.where(climb, 2.0 * (t - 300), 0.0)  # the thermal drifts east with a 2 m/s wind
    e = np.cumsum(speed * np.cos(heading)) + drift
    n = np.cumsum(speed * np.sin(heading))
    u = np.cumsum(np.where(glide, -1.2, np.where(search, -0.3 + 0.3 * np.sin(s / 9), 1.8)))
    return t, np.vstack([e, n, u]) + rng.normal(0, 0.5, (3, len(t)))


def window_answers(turn, vz_mean, k, alpha):
    """The three answers of the window of fix k, counted as the definitions in the note say."""
    w = turn[k - 15:k + 15]
    top = np.zeros(30, dtype=bool)
    top[np.argsort(-np.abs(w))[:15]] = True
    below = int(np.sum(np.abs(w) < alpha))
    left, right = int(np.sum(w[top] > 0)), int(np.sum(w[top] < 0))
    return w, top, below, left, right, vz_mean[k]


WINDOWS = {"glide": 80, "search": 230, "climb": 390}  # centre fixes of the windows of the figure


def window_figure():
    alpha = phases.ALPHA["para"]
    t, xyz = synthetic_flight()
    obs, turn, vz_mean = phases.features(t, *xyz, alpha)
    final, _ = phases.label(t, *xyz, alpha)

    def bars(ax, k, small=False):
        w, top, below, left, right, vz = window_answers(turn, vz_mean, k, alpha)
        answers = (int(below > 27), int(vz > 0), int(max(left, right) >= 14))
        assert answers == tuple(obs[k]), (k, answers, obs[k])  # the figure counts what the method counts
        x = np.arange(-15, 15)
        ax.axhspan(-alpha, alpha, color=LIGHT, lw=0)
        ax.axhline(0, color=MUTED, lw=0.5)
        ax.bar(x, w, width=0.75, color=np.where(top, INK, MUTED), lw=0)
        ax.set(xlim=(-16, 15), ylim=(-0.6, 0.6), xticks=[-15, 0, 14])
        ax.set_xlabel("fix in the window", labelpad=1)
        text = (rf"{below} of 30 below $\alpha$: $S={answers[0]}$" "\n"
                rf"$\bar v_z={vz:+.1f}$ m/s: $R={answers[1]}$" "\n"
                rf"15 largest: {left} left, {right} right: $O={answers[2]}$")
        ax.text(0.02, 0.97, text, transform=ax.transAxes, va="top", fontsize=7 if small else 7.5,
                linespacing=1.35, bbox=dict(facecolor="white", edgecolor="none", pad=0.5, alpha=0.85))

    def plan(ax, k, span=None):
        sel = slice(k - 15, k + 15)
        ctx = slice(max(0, k - 60), min(len(t), k + 60))
        ax.plot(xyz[0, ctx], xyz[1, ctx], color=MUTED, lw=0.6)
        ax.plot(xyz[0, sel], xyz[1, sel], color=PHASE[final[k]], lw=1.6)
        ax.plot(xyz[0, sel], xyz[1, sel], "o", ms=2.2, color=INK, mew=0)
        ax.plot(xyz[0, k], xyz[1, k], "o", ms=5, mfc="white", mec=INK, mew=1.0)
        e0, n0 = xyz[0, sel].mean(), xyz[1, sel].mean()
        half = span or 0.6 * max(np.ptp(xyz[0, sel]), np.ptp(xyz[1, sel]), 60)
        ax.set(xlim=(e0 - half, e0 + half), ylim=(n0 - half, n0 + half))
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for side in ("left", "bottom"):
            ax.spines[side].set_visible(False)
        bar = 50 if half < 120 else 100
        ax.plot([e0 + half * 0.9 - bar, e0 + half * 0.9], [n0 - half * 0.88] * 2, color=INK, lw=1)
        ax.text(e0 + half * 0.9 - bar / 2, n0 - half * 0.83, f"{bar} m", ha="center", va="bottom", fontsize=7)

    def static(fig):
        fig.clear()
        gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.05], left=0.08, right=0.99, top=0.93, bottom=0.1,
                              wspace=0.12, hspace=0.3)
        for c, (name, k) in enumerate(WINDOWS.items()):
            ap, ab = fig.add_subplot(gs[0, c]), fig.add_subplot(gs[1, c])
            plan(ap, k)
            ap.set_title(rf"({'abc'[c]}) {name}, window of fix {k}", loc="left", pad=2, fontsize=8.5)
            bars(ab, k, small=True)
            if c:
                ab.tick_params(labelleft=False)
            else:
                ab.set_ylabel("turn (rad per fix)", labelpad=1)

    def frame(fig, k):
        fig.clear()
        gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.5], height_ratios=[1.35, 1], left=0.13, right=0.99,
                              top=0.86, bottom=0.1, wspace=0.18, hspace=0.42)
        ap, ab, ast = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])
        plan(ap, k, span=160)
        ap.set_title("plan view: the window", loc="left", pad=2)
        bars(ab, k)
        ab.set_ylabel("turn (rad per fix)", labelpad=1)
        ab.set_title("its 30 turns, the 15 largest dark", loc="left", pad=2)
        rows = [("straight $S$", obs[:, 0]), ("rising $R$", obs[:, 1]), ("one sign $O$", obs[:, 2])]
        for r, (name, on) in enumerate(rows):
            y = 3 - r
            ast.add_patch(Rectangle((0, y - 0.35), len(t), 0.7, color=LIGHT, lw=0))
            edges = np.flatnonzero(np.diff(on[:k + 1]) != 0) + 1
            for a, b in zip(np.r_[0, edges], np.r_[edges, k + 1]):
                if on[a]:
                    ast.add_patch(Rectangle((a, y - 0.35), b - a, 0.7, color=INK2, lw=0))
            ast.text(-6, y, name, ha="right", va="center", fontsize=8)
        strip(ast, t[:k + 1], final[:k + 1], 0)
        ast.text(-6, 0, "label", ha="right", va="center", fontsize=8)
        ast.axvline(k, color=INK, lw=1)
        ast.axvspan(k - 15, k + 15, color=INK, alpha=0.08, lw=0)
        ast.set(xlim=(0, len(t)), ylim=(-0.6, 3.6), yticks=[])
        ast.spines["left"].set_visible(False)
        ast.set_xlabel("time (s)", labelpad=1)
        fig.text(0.09, 0.955, rf"fix {k}: symbol $(S,R,O)=({obs[k, 0]},{obs[k, 1]},{obs[k, 2]})$", fontsize=10,
                 va="center")
        handles = [plt.Line2D([], [], color=PHASE[s], lw=4) for s in (2, 1, 0)]
        fig.legend(handles, [NAME[s] for s in (2, 1, 0)], loc="center right", bbox_to_anchor=(0.99, 0.955),
                   ncols=3, frameon=False, handlelength=1.2)

    size = (6.3, 4.4)
    fig = plt.figure(figsize=size)
    static(fig)
    fig.savefig(FIGURES / "window.pdf")
    plt.close(fig)
    frames = [(lambda f, k=k: frame(f, k), 0.15) for k in range(15, len(t) - 15, 5)]
    save_frames(frames + [(static, 4.0)], "window", size)
    return {name: tuple(int(v) for v in obs[k]) for name, k in WINDOWS.items()}


if __name__ == "__main__":
    FIGURES.mkdir(exist_ok=True)
    setup()
    emission_figure()
    interpolation_figure()
    window_figure()
    cadence_figure()
    if not cadence.ROOT.exists():
        sys.exit("SSD_DATA is not mounted: the figures of real flights are left as they are")
    t, xyz, z = flight(EXAMPLE, altitude=True)
    native = cadence.labels_on_fixes(t, xyz, 1, False, phases.ALPHA["para"])
    (grid, vals), = [p for p in phases.regular_grid(t, xyz, 4.0) if len(p[0]) > 3000][:1]
    final, _ = phases.label(grid, *vals, phases.ALPHA["para"])
    runs = np.flatnonzero(np.diff(final) != 0) + 1
    search_start = [r for r in runs if final[r] == 1][0]
    start = max(0, search_start - 8 * 60)
    # The zoom of Figure 2: from 4.5 to 11 minutes of the 24 of Figure 8, a glide, a climb, a search.
    flight_figure(t, xyz, z, native, (grid[start] + 270, grid[start] + 660))
    cadence_example_figure(t, xyz, z, native)
    features_figure(grid, vals, start)
    sample = pd.read_csv(HERE / "out/cadence.csv").flight_id.unique()[:30]
    shares = turn_density_figure(sample)
    (HERE / "out/turn_shares.yaml").write_text(yaml.safe_dump({NAME[k]: v for k, v in shares.items()}))
