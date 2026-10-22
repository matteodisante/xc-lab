"""Figures and animations of the note: ``uv run python plots.py [--static]`` from this folder."""

import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import ListedColormap
from matplotlib.patches import Circle, Rectangle
from PIL import Image

import models

HERE = Path(__file__).resolve().parent
FIGURES, ANIMATIONS = HERE / "figures", HERE / "animations"

# Categorical colours checked for colour-vision deficiency; the third one always carries a label.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, LIGHT = "#0b0b0b", "#52514e", "#9a9893", "#e6e5e1"


def setup(size=9):
    plt.rcParams.update({
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{mathpazo}\usepackage{amsmath}\usepackage{amssymb}",
        "font.family": "serif", "font.size": size, "axes.titlesize": size, "axes.labelsize": size,
        "xtick.labelsize": size - 1, "ytick.labelsize": size - 1,
        "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.facecolor": "white", "savefig.facecolor": "white",
    })


def bars(ax, rows, x0, x1, step):
    for r, (name, mask, x, color) in enumerate(rows):
        y = len(rows) - 1 - r
        ax.add_patch(Rectangle((x0, y - 0.32), x1 - x0, 0.64, color=LIGHT, lw=0))
        for a, b in models.runs(mask, x, step):
            ax.add_patch(Rectangle((a, y - 0.32), b - a, 0.64, color=color, lw=0))
        ax.text(x0 - 0.025 * (x1 - x0), y, name, ha="right", va="center", fontsize=8.5)
    ax.set_xlim(x0, x1)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)


def save_gif(draw, frames, name, size, hold_end=2.5):
    """Render ``frames`` as (argument of draw, seconds on screen) into animations/<name>.gif."""
    fig = plt.figure(figsize=size, dpi=96)

    def render(frame):
        draw(fig, frame)
        fig.canvas.draw()
        return Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3])

    samples = [np.asarray(render(frames[n][0])) for n in (0, len(frames) // 2, -1)]
    palette = Image.fromarray(np.concatenate(samples)).quantize(colors=128, method=Image.Quantize.MEDIANCUT)
    images = [render(f).quantize(palette=palette, dither=Image.Dither.NONE) for f, _ in frames]
    plt.close(fig)
    durations = [int(1000 * sec) for _, sec in frames]
    durations[-1] += int(1000 * hold_end)
    raw = ANIMATIONS / f"{name}_raw.gif"
    images[0].save(raw, save_all=True, append_images=images[1:], duration=durations, loop=0)
    subprocess.run(["magick", str(raw), "-layers", "Optimize", str(ANIMATIONS / f"{name}.gif")], check=True)
    raw.unlink()


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


# Contact of a pair.

def contact(static):
    p = models.PAIR
    t, xi, xj = models.pair_tracks()
    M, dist = models.correspondences(t, xi, xj, p["delta"], p["eps_h"])
    km, delta = 1e-3, p["delta"]

    def draw(fig, k, full=False):
        fig.clear()
        gs = fig.add_gridspec(2, 2, width_ratios=[1.25, 1], height_ratios=[1.5, 1], left=0.02, right=0.985,
                              top=0.9, bottom=0.11, wspace=0.22, hspace=0.62)
        ax, ap, ab = fig.add_subplot(gs[:, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 1])
        tk, cx = t[k], xi[0, k] * km
        hits = np.flatnonzero(M[k])

        ax.plot(*(xi * km), color=BLUE, lw=1, alpha=0.25)
        ax.plot(*(xj * km), color=ORANGE, lw=1, alpha=0.25)
        ax.plot(*(xi[:, :k + 1] * km), color=BLUE, lw=1.6)
        ax.plot(*(xj[:, :k + 1] * km), color=ORANGE, lw=1.6)
        win = np.abs(t - tk) <= delta
        ax.plot(*(xj[:, win] * km), color=ORANGE, lw=7, alpha=0.22, solid_capstyle="butt")
        ax.add_patch(Circle(xi[:, k] * km, p["eps_h"] * km, fill=False, ls="--", lw=0.9, color=INK2))
        if hits.size:
            ax.scatter(*(xj[:, hits] * km), s=14, color=ORANGE, edgecolor=INK, lw=0.5, zorder=4)
            best = hits[np.argmin(dist[k, hits])]
            ax.plot([xi[0, k] * km, xj[0, best] * km], [xi[1, k] * km, xj[1, best] * km], color=INK, lw=0.8,
                    zorder=4)
            ax.annotate(rf"$\mathbf{{x}}_j(t+l)$, $l={t[best] - tk:+.0f}$ s", xj[:, best] * km, xytext=(22, -40),
                        textcoords="offset points", fontsize=8, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6))
        ax.scatter(*(xi[:, k:k + 1] * km), s=34, color=BLUE, edgecolor=INK, lw=0.6, zorder=5)
        ax.annotate(r"$\mathbf{x}_i(t)$", xi[:, k] * km, xytext=(0, 30), textcoords="offset points", fontsize=9,
                    color=BLUE, ha="center")
        ax.scatter(*(xj[:, k:k + 1] * km), s=30, facecolor="white", edgecolor=ORANGE, lw=1.4, zorder=5)
        ax.annotate(r"$\mathbf{x}_j(t)$: same instant", xj[:, k] * km, xytext=(0, -45), textcoords="offset points",
                    fontsize=8, color=ORANGE, ha="center", arrowprops=dict(arrowstyle="-", color=ORANGE, lw=0.6))
        ax.set_aspect("equal")
        ax.set_xlim(cx - 2.6, cx + 2.6)
        ax.set_ylim(-2.45, 2.45)
        ax.set_axis_off()
        ax.plot([cx + 1.4, cx + 2.4], [-2.25, -2.25], color=INK, lw=1.2)
        ax.text(cx + 1.9, -2.12, "1 km", ha="center", va="bottom", fontsize=8)
        ax.text(cx - 2.55, 2.4, r"\textbf{Plan view}", va="top", fontsize=9)
        ax.text(cx - 2.55, 2.05, r"circle: radius $\varepsilon_h = 500$ m" "\n" r"orange band: $j$ from $t-\delta$ to $t+\delta$",
                va="top", fontsize=7.5, color=INK2, linespacing=1.4)

        ap.fill_between([0, 1200], [-delta, 1200 - delta], [delta, 1200 + delta], color=LIGHT, lw=0)
        ap.plot([0, 1200], [0, 1200], color=MUTED, lw=0.7, ls="--")
        shown = M if full else np.where(np.arange(len(t))[:, None] <= k, M, False)
        a_idx, b_idx = np.nonzero(shown)
        ap.scatter(t[a_idx], t[b_idx], s=1.2, color=AQUA, marker="s", lw=0)
        ap.plot([tk, tk], [max(0, tk - delta), min(1200, tk + delta)], color=INK, lw=1.4)
        if hits.size:
            ap.scatter(np.full(hits.size, tk), t[hits], s=3, color=INK, marker="s", lw=0, zorder=4)
        if shown.any():
            ap.text(250, 820, r"$\mathcal{M}_{ij}$", fontsize=9, color=AQUA)
        ap.set(xlim=(0, 1200), ylim=(0, 1200), xticks=[0, 600, 1200], yticks=[0, 600, 1200])
        ap.set_aspect("equal")
        ap.set_xlabel(r"$t$ (s), clock of $i$", labelpad=1)
        ap.set_ylabel(r"$t'$ (s), clock of $j$", labelpad=1)
        ap.set_title(r"Plane $(t,t')$: the set $\mathcal{M}_{ij}$", loc="left", pad=4)
        ap.text(1180, 90, r"$|t'-t|\le\delta$", ha="right", fontsize=7.5, color=INK2)
        ap.text(700, 610, r"$t'=t$", fontsize=7.5, color=MUTED, rotation=45, ha="center")

        m_ij, m_ji = shown.any(axis=1), shown.any(axis=0)
        if not full:
            m_ij = m_ij & (t <= tk)
        bars(ab, [(r"$m_{ij}(t)$", m_ij, t, BLUE), (r"$m_{ji}(t)$", m_ji, t, ORANGE),
                  (r"$a_{ij}(t)$", m_ij | m_ji, t, AQUA)], 0, 1200, p["step"])
        ab.axvline(tk, color=INK, lw=1.2)
        ab.set_xticks([0, 300, 600, 900, 1200])
        ab.set_xlabel("time (s)", labelpad=1)
        ab.set_title(r"Contact: $a_{ij}=m_{ij}\vee m_{ji}$", loc="left", pad=4)
        head = rf"$t = {tk:.0f}$ s\quad "
        head += (rf"correspondences: $t' \in [{t[hits].min():.0f}, {t[hits].max():.0f}]$ s" if hits.size
                 else "no correspondence")
        fig.text(0.045, 0.955, head, fontsize=10, va="center")

    k_fig = int(np.searchsorted(t, 420.0))
    fig = plt.figure(figsize=(6.3, 3.5))
    draw(fig, k_fig, full=True)
    fig.savefig(FIGURES / "contact.pdf")
    plt.close(fig)
    frames = [(lambda f, k=k: draw(f, k), 0.125) for k in range(0, len(t), 4)]
    save_frames(frames + [(lambda f: draw(f, k_fig, full=True), 4.0)], "contact", (6.3, 3.5))
    if not static:
        save_gif(draw, [(k, 0.1) for k in range(0, len(t), 2)], "contact", size=(9.0, 5.0))


# Altitude as a clock.

def clock(static):
    p = models.THERMAL
    s = models.thermal_scan()
    lags, d, zj, zi, both = s.lags, s.horizontal, s.zj, s.zi, s.both
    delta, eps_z = p["delta"], p["eps_z"]
    k0 = int(np.argmin(np.abs(lags)))

    def draw(fig, l_now, full=False):
        fig.clear()
        gs = fig.add_gridspec(3, 2, width_ratios=[0.8, 1.25], height_ratios=[1, 1, 0.62], left=0.075, right=0.985,
                              top=0.89, bottom=0.085, wspace=0.28, hspace=0.62)
        aa, ab, ac, ad = (fig.add_subplot(gs[:, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 1]),
                          fig.add_subplot(gs[2, 1]))
        k = int(np.argmin(np.abs(lags - l_now)))
        seen = np.full(lags.size, True) if full else lags <= l_now

        aa.axhspan(zi - eps_z, zi + eps_z, color=LIGHT, lw=0)
        aa.axhline(zi, color=BLUE, lw=0.7, ls="--")
        aa.plot(s.xj[0], zj, color=ORANGE, lw=0.8)
        aa.scatter([s.xi[0]], [zi], s=40, color=BLUE, edgecolor=INK, lw=0.6, zorder=5)
        aa.annotate(r"$i$ at time $t$", (s.xi[0], zi), xytext=(14, 16), textcoords="offset points", fontsize=8,
                    color=BLUE, arrowprops=dict(arrowstyle="-", color=BLUE, lw=0.6))
        aa.scatter([s.xj[0, k0]], [zj[k0]], s=30, facecolor="white", edgecolor=ORANGE, lw=1.3, zorder=5)
        aa.annotate(r"$j$ at time $t$", (s.xj[0, k0], zj[k0]), xytext=(14, -22), textcoords="offset points",
                    fontsize=8, color=ORANGE, arrowprops=dict(arrowstyle="-", color=ORANGE, lw=0.6))
        if s.xj[0, k] < -88:
            aa.annotate(rf"$j$ is {abs(s.xj[0, k]):.0f} m away", xy=(-88, zj[k]), xytext=(-60, zj[k] + 60),
                        fontsize=8, color=ORANGE, arrowprops=dict(arrowstyle="->", color=ORANGE, lw=0.8))
        else:
            aa.scatter([s.xj[0, k]], [zj[k]], s=70, facecolor=ORANGE, edgecolor=AQUA if both[k] else MUTED, lw=2.2,
                       zorder=6)
        aa.annotate("", xy=(-70, zi), xytext=(-70, p["z_j0"]), arrowprops=dict(arrowstyle="<->", color=INK2, lw=0.7))
        aa.text(-66, (zi + p["z_j0"]) / 2, r"$\Delta z$", fontsize=8, color=INK2, va="center")
        aa.text(88, zi + eps_z - 12, r"$z_i(t)\pm\varepsilon_z$", fontsize=8, color=INK2, ha="right", va="top")
        aa.text(88, 890, r"radius 25 m, one turn every 20 s" "\n" r"$w = 2$ m/s", fontsize=7.5, color=INK2,
                ha="right", va="bottom", linespacing=1.4)
        aa.set(xlim=(-90, 90), ylim=(880, 1860))
        aa.set_xlabel("horizontal position (m)", labelpad=1)
        aa.set_ylabel("altitude (m)", labelpad=1)
        aa.set_title("Side view: one thermal", loc="left", pad=4)

        ab.plot(lags, d, color=INK, lw=0.5, alpha=0.18)
        ab.plot(lags[seen], d[seen], color=INK, lw=0.9)
        mins = np.flatnonzero((d[1:-1] < d[:-2]) & (d[1:-1] <= d[2:]) & (d[1:-1] < 5)) + 1
        ab.scatter(lags[mins], d[mins], s=9, color=INK2, zorder=4, lw=0)
        ab.axvline(l_now, color=INK, lw=1.1)
        ab.set(xlim=(-delta, delta), ylim=(0, 100))
        ab.set_ylabel("m", labelpad=1)
        ab.set_title(r"Horizontal distance $\|\mathbf{x}_i(t)-\mathbf{x}_j(t+l)\|$", loc="left", pad=4)
        box = dict(facecolor="white", edgecolor="none", pad=0.6)
        ab.text(-delta + 8, 96, r"$\varepsilon_h = 500$ m \textuparrow\ off scale: inside the thermal it always holds",
                fontsize=7.5, color=INK2, va="top", zorder=5, bbox=box)
        ab.text(-delta + 8, 78, r"inside the thermal: an equal minimum at every turn (20 s)", fontsize=7.5,
                color=INK2, va="top", zorder=5, bbox=box)

        ac.axhspan(zi - eps_z, zi + eps_z, color=LIGHT, lw=0)
        ac.axhline(zi, color=BLUE, lw=0.7, ls="--")
        ac.plot(lags, zj, color=ORANGE, lw=0.8, alpha=0.25)
        ac.plot(lags[seen], zj[seen], color=ORANGE, lw=1.4)
        ac.axvline(l_now, color=INK, lw=1.1)
        l_true = p["dz"] / p["climb"]
        ac.plot([l_true], [zi], marker="o", ms=4, color=INK)
        ac.annotate(r"$l = \Delta z / w$", (l_true, zi), xytext=(8, -14), textcoords="offset points", fontsize=8)
        ac.set(xlim=(-delta, delta), ylim=(880, 1860))
        ac.set_ylabel("m", labelpad=1)
        ac.set_title(r"Altitude $z_j(t+l)$ and band $z_i(t)\pm\varepsilon_z$", loc="left", pad=4)

        bars(ad, [("horizontal", s.ok_h & seen, lags, MUTED), ("vertical", s.ok_z & seen, lags, MUTED),
                  ("both", both & seen, lags, AQUA)], -delta, delta, 1.0)
        ad.axvline(l_now, color=INK, lw=1.1)
        ad.set_xlabel(r"delay $l$ (s)", labelpad=1)
        for axx in (ab, ac, ad):
            axx.set_xticks([-300, -150, 0, 150, 300])
        ab.tick_params(labelbottom=False)
        ac.tick_params(labelbottom=False)
        tick = r"\checkmark" if both[k] else r"$\times$"
        fig.text(0.075, 0.955, rf"$l = {lags[k]:+.0f}$ s\quad horizontal: {d[k]:.0f} m\quad "
                               rf"vertical: {abs(zi - zj[k]):.0f} m\quad correspondence: {tick}", fontsize=10, va="center")

    fig = plt.figure(figsize=(6.3, 3.9))
    draw(fig, p["dz"] / p["climb"], full=True)
    fig.savefig(FIGURES / "altitude_clock.pdf")
    plt.close(fig)
    frames = [(lambda f, l=l: draw(f, l), 0.125) for l in np.arange(-delta, delta + 1e-9, 10.0)]
    save_frames(frames + [(lambda f: draw(f, p["dz"] / p["climb"], full=True), 4.0)], "altitude_clock",
                (6.3, 3.9))
    if not static:
        save_gif(draw, [(l, 0.1) for l in np.arange(-delta, delta + 1e-9, 5.0)], "altitude-clock", size=(9.0, 5.4))


# A day of group flying.

POS = {1: (0.16, 0.56), 8: (0.48, 0.80), 22: (0.48, 0.30), 14: (0.82, 0.56), 5: (0.16, 0.10)}
ROWS = [1, 8, 22, 14, 5]
EPISODE_COLOUR = {1: BLUE, 2: ORANGE}
LOG = [("11:20", r"flight 1 departs (it leaves the $R_{\rm dep}$ disc)"),
       ("11:40", r"1--8 and 8--22 open: $c_1=\{1,8,22\}$ starts"),
       ("12:05", r"1--22 opens: same members, $c_1$ goes on"),
       ("12:30", r"1--22 closes: $c_1$ goes on"),
       ("12:55", r"1--8 closes: $c_1$ ends, $c_2=\{8,22\}$ starts; $c_1\to c_2$"),
       ("13:30", r"8--14 opens: $c_3=\{8,14,22\}$ starts; $c_2\to c_3$"),
       ("14:10", r"8--22 and 8--14 close: $c_3$ ends, so does E1"),
       ("15:00", r"8--22 opens again: $c_4=\{8,22\}$, a new episode E2"),
       ("15:40", r"8--22 closes: $c_4$ ends, so does E2")]


def hhmm(m):
    return f"{int(m) // 60}:{int(m) % 60:02d}"


def day(static):
    flights, pairs = models.example_day()
    configs, episode, _ = models.sweep(flights, pairs)
    name = {c: f"c{n + 1}" for n, c in enumerate(configs)}
    t0, t1 = models.minutes("11:00"), models.minutes("16:00")
    log = [(models.minutes(a), b) for a, b in LOG]

    def graph(ax, t, small=False):
        ax.set(xlim=(0, 1), ylim=(-0.04, 1.0))
        ax.set_axis_off()
        for (a, b), on, off in pairs:
            if on <= t < off:
                ax.plot(*zip(POS[a], POS[b]), color=INK, lw=1.4 if small else 2.0, zorder=1)
        for f, (x, y) in POS.items():
            st = models.label(flights, configs, f, t)
            if st is None:
                face, edge, ls, txt = "white", LIGHT, ":", LIGHT
            elif st == "L":
                face, edge, ls, txt = "white", MUTED, "--", MUTED
            elif st == "S":
                face, edge, ls, txt = "white", INK, "-", INK
            else:
                face, edge, ls, txt = EPISODE_COLOUR[episode[st]], INK, "-", "white"
            ax.scatter([x], [y], s=190 if small else 520, facecolor=face, edgecolor=edge, linewidths=0.9,
                       linestyle=ls, zorder=2)
            ax.text(x, y, str(f), ha="center", va="center", fontsize=7 if small else 9, color=txt, zorder=3)

    def timeline(ax, upto, marks=()):
        for r, f in enumerate(ROWS):
            y = len(ROWS) - 1 - r
            takeoff, dep, land = flights[f]
            end = min(land, upto)
            ax.text(t0 - 4, y, rf"flight {f}", ha="right", va="center", fontsize=8)
            if takeoff < end:
                ax.add_patch(Rectangle((takeoff, y - 0.3), min(dep, end) - takeoff, 0.6, facecolor="white",
                                       edgecolor=MUTED, hatch="////", lw=0.6))
            if dep < end:
                ax.add_patch(Rectangle((dep, y - 0.3), end - dep, 0.6, color=LIGHT, lw=0))
            for c in configs:
                if f in c.members and c.start < end:
                    b = min(c.end, end)
                    ax.add_patch(Rectangle((c.start + 0.6, y - 0.3), b - c.start - 1.2, 0.6,
                                           color=EPISODE_COLOUR[episode[c]], lw=0))
                    if b - c.start > 14:
                        ax.text((c.start + b) / 2, y, name[c], ha="center", va="center", fontsize=7.5, color="white")
        ye = -1.25
        ax.text(t0 - 4, ye, "episodes", ha="right", va="center", fontsize=8)
        for e in sorted(set(episode.values())):
            a = min(c.start for c in configs if episode[c] == e)
            b = max(c.end for c in configs if episode[c] == e)
            if a < upto:
                bb = min(b, upto)
                ax.plot([a, bb], [ye, ye], color=EPISODE_COLOUR[e], lw=2.2, solid_capstyle="butt")
                for x in [a] + ([b] if bb == b else []):
                    ax.plot([x, x], [ye - 0.18, ye + 0.18], color=EPISODE_COLOUR[e], lw=1.2)
                ax.text((a + bb) / 2, ye - 0.35, rf"E{e}", ha="center", va="top", fontsize=8)
        for tm in marks:
            ax.axvline(tm, color=INK2, lw=0.6, ls=":")
        ax.set(xlim=(t0 - 2, t1 + 2), ylim=(-2.1, len(ROWS) - 0.4), yticks=[])
        ax.spines["left"].set_visible(False)
        ticks = list(range(t0, t1 + 1, 60))
        ax.set_xticks(ticks)
        ax.set_xticklabels([hhmm(x) for x in ticks])

    def draw(fig, t):
        fig.clear()
        gs = fig.add_gridspec(2, 2, width_ratios=[0.8, 1.2], height_ratios=[1.05, 1], left=0.09, right=0.985,
                              top=0.9, bottom=0.07, wspace=0.08, hspace=0.25)
        ag, al, at = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])
        graph(ag, t)
        ag.set_title(r"Active contact graph $G(t)$", loc="left", pad=2)
        lines = []
        for m in models.active_components(flights, pairs, t):
            c = models.label(flights, configs, next(iter(m)), t)
            lines.append(rf"{name[c]} $=\{{{','.join(str(x) for x in sorted(m))}\}}$, episode E{episode[c]}")
        ag.text(0.0, -0.04, "\n".join(lines) if lines else "no group", fontsize=8.5, va="top")
        al.set_axis_off()
        al.set(xlim=(0, 1), ylim=(0, 1))
        al.set_title("Events", loc="left", pad=2)
        past = [x for x, _ in log if x <= t]
        y = 0.95
        for te, text in log:
            if te <= t:
                colour = INK if te == max(past) else MUTED
                al.text(0.0, y, hhmm(te), fontsize=8.5, va="top", color=colour)
                al.text(0.12, y, text, fontsize=8.5, va="top", color=colour)
                y -= 0.105
        timeline(at, t)
        at.axvline(t, color=INK, lw=1.2)
        fig.text(0.09, 0.955, rf"time {hhmm(t)}", fontsize=11, va="center")
        fig.text(0.985, 0.955, r"hatched: local phase L\quad grey: solo S\quad colour: episode", fontsize=8,
                 va="center", ha="right", color=INK2)

    snaps = [models.minutes(x) for x in ("11:50", "12:15", "13:10", "13:50", "15:20")]

    def summary(fig):
        fig.clear()
        gs = fig.add_gridspec(2, len(snaps), height_ratios=[0.8, 1.35], left=0.1, right=0.985, top=0.95,
                              bottom=0.07, hspace=0.12, wspace=0.05)
        for n, ts in enumerate(snaps):
            ax = fig.add_subplot(gs[0, n])
            graph(ax, ts, small=True)
            ax.set_title(hhmm(ts), pad=1, fontsize=8)
        timeline(fig.add_subplot(gs[1, :]), t1, marks=snaps)

    fig = plt.figure(figsize=(6.3, 3.6))
    summary(fig)
    fig.savefig(FIGURES / "example_day.pdf")
    plt.close(fig)
    events = {x for x, _ in log}
    frames = [(lambda f, t=t: draw(f, t), 1.5 if t in events else 0.15) for t in np.arange(t0, t1 + 0.01, 5.0)]
    save_frames(frames + [(summary, 4.0)], "example_day", (6.3, 3.6))
    if not static:
        frames = [(t, 1.6 if t in events else 0.1) for t in np.arange(t0, t1 + 0.01, 2.5)]
        save_gif(draw, frames, "example-day", size=(9.0, 5.4))


# m_ij and m_ji: a pilot who waits and a pilot who passes, and a follower.

def asymmetry():
    p, q = models.WAIT, models.FOLLOW
    t, xi, xj = models.wait_tracks()
    M, _ = models.correspondences(t, xi, xj, p["delta"], p["eps_h"])
    m_ij, m_ji = M.any(axis=1), M.any(axis=0)
    tk = 380.0
    k = int(np.searchsorted(t, tk))
    km, ring = 1e-3, np.linspace(0, 2 * np.pi, 100)
    x_j = xj[0] * km

    def plan(ax, title):
        ax.plot(x_j, 0 * x_j, color=ORANGE, lw=1, alpha=0.3)
        ax.plot(0.08 * np.cos(ring), 0.08 * np.sin(ring), color=BLUE, lw=1.4)
        ax.set_aspect("equal")
        ax.set(xlim=(-2.9, 1.1), ylim=(-0.78, 0.78), yticks=[])
        ax.spines["left"].set_visible(False)
        ax.set_title(title, loc="left", pad=3)

    def at(ax, s, label, dy, big=False):
        ax.scatter([x_j[int(s)]], [0], marker="D", s=30 if big else 16, color=ORANGE, edgecolor="white", lw=0.6, zorder=5)
        ax.text(x_j[int(s)], dy, label, fontsize=7.5, color=INK2, ha="center", va="bottom" if dy > 0 else "top")

    fig, (a, b) = plt.subplots(2, 1, figsize=(6.3, 4.3), gridspec_kw=dict(hspace=0.3))
    plan(a, r"(a) $i$ at 380 s: is $j$ on its path within $\pm\delta$?\quad $m_{ij}(380)=1$")
    a.plot([], [], "o", color=BLUE, ms=5, label=r"$i$: circles in $P$ until 600 s")
    a.plot([], [], "D", color=ORANGE, ms=4, label=r"$j$: straight line, in $P$ at 500 s")
    a.legend(loc="upper left", frameon=False, fontsize=7.5, handletextpad=0.3, borderaxespad=0.1)
    win = np.abs(t - tk) <= p["delta"]
    a.plot(x_j[win], 0 * x_j[win], color=ORANGE, lw=4, solid_capstyle="butt", zorder=3)
    a.add_patch(Circle((0, 0), p["eps_h"] * km, fill=False, ls="--", lw=0.9, color=BLUE))
    a.text(0.56, 0.3, r"$\varepsilon_h=500$ m" "\naround $i$", fontsize=7.5, color=INK2, va="center")
    for s, label, dy in ((260, r"$j$ at 260 s", 0.1), (380, r"$j$ at 380 s", 0.1), (450, r"$j$ at 450 s", -0.1)):
        at(a, s, label, dy)
    at(a, 500, r"$j$ at 500 s", -0.12, big=True)
    a.scatter([0], [0.08], s=26, color=BLUE, edgecolor="white", lw=0.6, zorder=6)
    a.text(-2.85, -0.3, r"thick: $j$ in the window $[260, 500]$ s", fontsize=7.5, color=INK2, va="top")
    a.set_xticklabels([])

    plan(b, r"(b) $j$ at 380 s: is $i$ on its path within $\pm\delta$?\quad $m_{ji}(380)=0$")
    b.plot(0.08 * np.cos(ring), 0.08 * np.sin(ring), color=BLUE, lw=4, zorder=3)
    b.text(0, 0.14, r"$i$ in $[260, 500]$ s:" "\nonly here", fontsize=7.5, color=INK2, ha="center", va="bottom")
    b.add_patch(Circle((x_j[k], 0), p["eps_h"] * km, fill=False, ls="--", lw=0.9, color=ORANGE))
    b.text(x_j[k] - 0.56, 0.3, r"$\varepsilon_h=500$ m" "\naround $j$", fontsize=7.5, color=INK2, ha="right", va="center")
    at(b, tk, r"$j$ at 380 s", 0.1, big=True)
    b.annotate("", xy=(0, -0.55), xytext=(x_j[k], -0.55), arrowprops=dict(arrowstyle="<->", color=INK, lw=0.8))
    b.text(x_j[k] / 2, -0.59, r"1.2 km $>\varepsilon_h$", fontsize=7.5, ha="center", va="top")
    b.set_xlabel("position along the route of $j$ (km)", labelpad=1)
    fig.savefig(FIGURES / "asymmetry_paths.pdf", bbox_inches="tight")
    plt.close(fig)

    fig = plt.figure(figsize=(6.3, 3.2))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.2], wspace=0.42, hspace=0.95, left=0.0, right=1.0)
    ax = fig.add_subplot(gs[:, 0])
    end = t[-1]
    ax.fill_between([0, end], [-p["delta"], end - p["delta"]], [p["delta"], end + p["delta"]], color=LIGHT, lw=0)
    ax.plot([0, end], [0, end], color=MUTED, lw=0.7, ls="--")
    ax.imshow(np.where(M.T, 1.0, np.nan), origin="lower", extent=(-0.5, end + 0.5, -0.5, end + 0.5), vmin=0, vmax=1,
              cmap=ListedColormap([AQUA]), interpolation="nearest", zorder=1.5)
    for lo, hi in models.runs(m_ij, t, p["step"]):
        ax.plot([lo, hi], [0, 0], color=BLUE, lw=4.5, solid_capstyle="butt", clip_on=False, zorder=6)
    for lo, hi in models.runs(m_ji, t, p["step"]):
        ax.plot([0, 0], [lo, hi], color=ORANGE, lw=4.5, solid_capstyle="butt", clip_on=False, zorder=6)
    hits = np.flatnonzero(M[k])
    ax.plot([tk, tk], [0, end], color=BLUE, lw=0.8, ls="--")
    ax.plot([tk, tk], [t[hits].min(), t[hits].max()], color=BLUE, lw=2.2, solid_capstyle="butt")
    ax.plot([0, end], [tk, tk], color=ORANGE, lw=0.8, ls="--")
    ax.plot([0, tk], [500, 500], color=INK2, lw=0.6, ls=":")
    ax.scatter([tk], [500], s=20, color=INK, zorder=7)
    ax.text(tk - 12, 515, r"pair $(380, 500)$", fontsize=7.5, ha="right", va="bottom")
    ax.text(612, 500, r"$\mathcal{M}_{ij}$", fontsize=9, ha="left", va="center", color=INK)
    ax.text(end - 15, 80, r"$|t'-t|\le\delta$", ha="right", fontsize=7.5, color=INK2)
    ax.set(xlim=(0, end), ylim=(0, end), xticks=[0, 300, 600], yticks=[0, 300, 600])
    ax.set_aspect("equal")
    ax.set_xlabel(r"$t$ (s), clock of $i$", labelpad=1)
    ax.set_ylabel(r"$t'$ (s), clock of $j$", labelpad=1)
    ax.set_title(r"(a) plane $(t,t')$ and the two shadows of $\mathcal{M}_{ij}$", loc="left", pad=4)

    tf, xif, xjf = models.follow_tracks()
    Mf, _ = models.correspondences(tf, xif, xjf, q["delta"], q["eps_h"])
    for axb, (tt, mij, mji), title in ((fig.add_subplot(gs[0, 1]), (t, m_ij, m_ji), r"(b) the waiting pilot"),
                                       (fig.add_subplot(gs[1, 1]), (tf, Mf.any(axis=1), Mf.any(axis=0)),
                                        r"(c) a follower, 90 s behind on the same route")):
        bars(axb, [(r"$m_{ij}(t)$", mij, tt, BLUE), (r"$m_{ji}(t)$", mji, tt, ORANGE),
                   (r"$a_{ij}(t)$", mij | mji, tt, AQUA)], 0, end, p["step"])
        axb.set_xticks([0, 200, 400, 600])
        axb.set_title(title, loc="left", pad=3)
    ax_b = fig.axes[1]
    ax_b.axvline(tk, color=INK, lw=1.0)
    fig.axes[2].set_xlabel("time (s)", labelpad=1)
    fig.savefig(FIGURES / "asymmetry_plane.pdf", bbox_inches="tight")
    plt.close(fig)


# Decimation, to scale.

def decimation():
    p = models.THERMAL
    omega, r = 2 * np.pi / p["period"], p["radius"]
    t1 = np.arange(0, 60.0 + 0.5, 1.0)
    t5 = np.arange(0, 60.0 + 0.5, 5.0)
    fig, (ax, az) = plt.subplots(1, 2, figsize=(6.3, 2.75), gridspec_kw=dict(width_ratios=[1, 1.15], wspace=0.35))
    ax.add_patch(Circle((0, 0), p["eps_h"], facecolor=LIGHT, edgecolor=INK2, lw=0.8, ls="--"))
    ax.plot(r * np.cos(omega * t1), r * np.sin(omega * t1), color=ORANGE, lw=0.8)
    ax.annotate("one whole thermal\n(radius 25 m)", (0, 0), xytext=(150, 330), fontsize=8, ha="left",
                arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6))
    ax.plot([-500, 0], [0, 0], color=INK2, lw=0.6)
    ax.text(-250, 25, r"$\varepsilon_h = 500$ m", fontsize=8, color=INK2, ha="center", va="bottom")
    ax.set_aspect("equal")
    ax.set(xlim=(-560, 560), ylim=(-680, 560))
    ax.set_axis_off()
    ax.set_title(r"(a) to scale: the tolerance $\varepsilon_h$", loc="left", pad=2)

    az.plot(r * np.cos(omega * t1), r * np.sin(omega * t1), color=ORANGE, lw=0.8)
    az.scatter(r * np.cos(omega * t1), r * np.sin(omega * t1), s=6, color=ORANGE, lw=0, label="1 Hz fixes")
    az.scatter(r * np.cos(omega * t5), r * np.sin(omega * t5), s=34, facecolor="white", edgecolor=INK, lw=1.0,
               zorder=4, label=r"grid, $\Delta = 5$ s")
    az.plot(r * np.cos(omega * t5[:5]), r * np.sin(omega * t5[:5]), color=INK, lw=0.7, ls=":")
    az.set_aspect("equal")
    az.set(xlim=(-45, 45), ylim=(-38, 38))
    az.set_xlabel("m", labelpad=1)
    az.set_ylabel("m", labelpad=1)
    az.legend(loc="upper right", frameon=False, fontsize=7.5, handletextpad=0.3, borderaxespad=0.1)
    az.set_title(r"(b) zoom: four grid points per turn", loc="left", pad=2)
    fig.savefig(FIGURES / "decimation.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    static = "--static" in sys.argv
    FIGURES.mkdir(exist_ok=True)
    ANIMATIONS.mkdir(exist_ok=True)
    setup()
    contact(static)
    clock(static)
    day(static)
    decimation()
    asymmetry()
