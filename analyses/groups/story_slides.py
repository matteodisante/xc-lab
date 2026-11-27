"""Figures of the slides of meetings/2026-11/group_story, drawn from the JSON of the analyses; nothing is recomputed.

Each figure is a story figure redrawn at the size of the slide with the helpers and colours of the modules that draw it
(contrasts.curve, contrasts.finish, contrasts.CLASS_COLORS, tamsd.COLORS, raw.GAP_COLORS, cohorts, prepost, phase_gaps,
carry): the same quantities and style, fewer panels, larger text. Written next to the JSON as `slide_<name>.pdf` and
`.png` at the size they have on the slide; the tables of the deck, cut from the generated tables of the report, go to the
`tikz/` folder of the deck:

    uv run python story_slides.py
"""

import json
import re
import sys
from pathlib import Path

for flag in ("--declared-open", "--dt5"):  # the run of the story: the modules read these from the command line
    if flag not in sys.argv:
        sys.argv.append(flag)

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

import carry as carry_module
import closure
import cohorts
import contrasts
import phase_gaps
import prepost
import raw
from tamsd import COLORS, save

DIR = contrasts.longlags.RESULTS
S, A, G = COLORS["solo"], COLORS["group_solo"], COLORS["group"]
GAP = raw.GAP_COLORS  # G - S orange, A - S aqua, G - A brown, as in the story
CLASSES, CLASS_COLORS, ALL = contrasts.CLASSES, contrasts.CLASS_COLORS, contrasts.ALL_COLOR
OPENING = ("#b9c7d6", "#6a88a8", "#1f3a5c")  # no threshold on the opening, above 0.1, above 0.5: light to dark steel
OPENING_TEXT = ("no threshold", "opening > 0.1", "opening > 0.5")
INK, GREY = "#222222", "#888888"
TAU = 1000  # s: the lag at which the story quotes the gaps
DECK = Path(__file__).resolve().parents[2] / "meetings/2026-11/group_story"
TABLES = Path(__file__).resolve().parent / "vs-solo/generated/comparisons"
WIDE = 6.1  # in: the width of a figure that takes the page
SIZES = {}  # path -> (width, height) in inches of what was saved, printed at the end

plt.rcParams.update({"font.size": 8.5, "axes.labelsize": 9, "axes.titlesize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
                     "legend.fontsize": 8, "pdf.fonttype": 42})


def write(fig, name, rect=None, pad=None):
    """tamsd.save, with room above the panels (`rect`) for a legend of the whole figure and optional `pad` around them."""
    path = DIR / f"slide_{name}.pdf"
    if rect or pad:
        fig.tight_layout(rect=rect, pad=pad if pad else 1.08)
        fig.savefig(path, bbox_inches="tight")
        fig.savefig(path.with_suffix(".png"), dpi=200, bbox_inches="tight")
    else:
        save(fig, path)
    plt.close(fig)
    img = plt.imread(path.with_suffix(".png"))
    SIZES[name] = (img.shape[1] / 200, img.shape[0] / 200)


def read(name):
    return json.loads((DIR / f"{name}.json").read_text())


def at(g, lag=TAU):
    """The {value, band} of a curve `g` (lags, value, band) at `lag`."""
    i = g["lags"].index(lag)
    return {"value": g["value"][i], "band": [g["band"][0][i], g["band"][1][i]]}


def percent(shares):
    """Whole percentages of `shares` (summing to 1) that add to 100: largest remainder."""
    raw_ = 100 * np.asarray(shares, dtype=float)
    out = np.floor(raw_).astype(int)
    out[np.argsort(out - raw_)[: 100 - out.sum()]] += 1
    return out


def bar(ax, x, y, q, vertical, color, ms=6, **kw):
    """One estimate {value, band} as a dot with its 90% band: at x with the value along y if `vertical`, else at y along x."""
    lo, hi = q["band"]
    err = [[q["value"] - lo], [hi - q["value"]]]
    fmt, lw, cap = kw.pop("fmt", "o"), kw.pop("lw", 1.6), kw.pop("capsize", 3)
    if vertical:
        ax.errorbar(x, q["value"], yerr=err, fmt=fmt, ms=ms, color=color, elinewidth=lw, capsize=cap, **kw)
    else:
        ax.errorbar(q["value"], y, xerr=err, fmt=fmt, ms=ms, color=color, elinewidth=lw, capsize=cap, **kw)


def spread(ys, gap):
    """Label heights close to `ys`, pushed apart to at least `gap`, in the order of `ys`."""
    order = np.argsort(ys)
    out = np.array(ys, dtype=float)
    for a, b in zip(order, order[1:]):
        out[b] = max(out[b], out[a] + gap)
    return out


# ---- the method ----

def nested():
    """Left: alpha(tau) of the nested cohorts of every piece (all flights, S+A+G together) and the pool (every piece that
    reaches each lag), clipped at the top of the axis. Right: the pieces that reach tau, per population: every piece up to
    the shortest tau_max, then those of each cohort."""
    c, a = read("cohorts"), read("cohorts")["all"]
    fig, (left, right) = plt.subplots(1, 2, figsize=(6.0, 2.0), gridspec_kw={"width_ratios": [1.25, 1]})
    pooled = a["pool"]
    lags, band = np.array(pooled["lags"]), np.array(pooled["band"])
    left.fill_between(lags, band[0], band[1], color=cohorts.POOL_COLOR, alpha=0.15, lw=0)
    left.plot(lags, pooled["value"], color=cohorts.POOL_COLOR, lw=1.6, label="pool")
    for k, (cohort, color) in enumerate(zip(a["alpha"], cohorts.shades())):
        lags, band = np.array(cohort["lags"]), np.array(cohort["band"])
        left.fill_between(lags, band[0], band[1], color=color, alpha=0.15, lw=0)
        left.plot(lags, cohort["alpha"], color=color, lw=1.3, label=f"to {cohort['tau_max']:,} s" if k in (0, len(cohorts.COHORTS) - 1) else None)
    left.set(xscale="log", xlabel=r"$\tau$ (s)", ylabel=r"$\alpha(\tau)$", ylim=(0.82, 1.1))
    left.legend(frameon=False, loc="upper left", fontsize=7)
    for pop, short in raw.POPS.items():
        n = [q["pieces"] for q in c["alpha"][pop]]
        right.plot([pooled["lags"][0], *cohorts.COHORTS], [n[0], *n], "o-", ms=3.5, lw=1.2, color=COLORS[pop], label=short, markevery=range(1, len(n) + 1))
    right.set(xscale="log", yscale="log", xlim=left.get_xlim(), xlabel=r"$\tau$ (s)", ylabel=r"pieces that reach $\tau$")
    right.legend(frameon=False, loc="lower left", fontsize=7)
    for ax in (left, right):
        ax.grid(color="#eeeeee", lw=0.5)
    write(fig, "cohorts")


def chain_pool():
    """alpha* (chained), the pool and every nested cohort, per population (chained_pool.pdf and cohorts.pdf together)."""
    s, c = read("perlag"), read("cohorts")
    fig, axes = plt.subplots(1, 3, figsize=(5.6, 1.8), sharey=True)
    for ax, (pop, short) in zip(axes, raw.POPS.items()):
        for k, (cohort, color) in enumerate(zip(c["alpha"][pop], cohorts.shades())):
            ax.plot(cohort["lags"], cohort["alpha"], color=color, lw=1.2 if k == len(c["alpha"][pop]) - 1 else 0.7)
        contrasts.curve(ax, s["pool"][pop], cohorts.POOL_COLOR, lw=1.1)
        contrasts.curve(ax, s["chained"][pop], COLORS[pop], lw=1.1)
        ax.legend(handles=[Line2D([], [], color=COLORS[pop], lw=1.4, label=r"chained $\alpha^*$")], frameon=False, loc="lower left",
                  fontsize=8, handlelength=1.2, borderaxespad=0.2)
        ax.set(xscale="log", xlabel=r"$\tau$ (s)")
        ax.set_title(short, fontsize=10)
        ax.xaxis.label.set_size(9)
        ax.tick_params(labelsize=8.5)
        ax.grid(color="#eeeeee", lw=0.5)
    axes[0].set(ylabel=r"$\alpha(\tau)$", ylim=(0.78, 1.05))
    axes[0].yaxis.label.set_size(9)
    shades, ends = cohorts.shades(), (cohorts.COHORTS[0], cohorts.COHORTS[-1])
    handles = [Line2D([], [], color=shades[i], lw=1.4, label=f"cohort to {ends[j]:,} s") for j, i in enumerate((0, -1))]
    handles += [Line2D([], [], color=cohorts.POOL_COLOR, lw=1.4, label="pool")]
    fig.legend(handles=handles, frameon=False, loc="upper center", ncol=3, fontsize=9, columnspacing=1.6, handlelength=1.6)
    write(fig, "chain_pool", rect=[0, 0, 1, 0.88], pad=0.5)


REFERENCE_COHORT = 2  # the cohort to 3600 s ("1 h"), the reference of cohort_msd


def cohort_msd():
    """cohorts_msd.pdf, the MSD row as a ratio: the MSD of every nested cohort and of the pool over that of the cohort to 1 h, on the
    lags they share, per population (the nested cohorts of longer flights lie above, the pool rises at long lags)."""
    c = read("cohorts")
    fig, axes = plt.subplots(1, 3, figsize=(5.6, 1.9), sharey=True)
    for ax, (pop, short) in zip(axes, raw.POPS.items()):
        ref = dict(zip(c["alpha"][pop][REFERENCE_COHORT]["lags"], c["alpha"][pop][REFERENCE_COHORT]["msd"]))
        for q, color, lw in [(c["pool"][pop], cohorts.POOL_COLOR, 1.5)] + [(x, col, 1.2) for x, col in zip(c["alpha"][pop], cohorts.shades())]:
            lags = [t for t in q["lags"] if t in ref]
            ax.plot(lags, [m / ref[t] for t, m in zip(q["lags"], q["msd"]) if t in ref], color=color, lw=lw)
        ax.axhline(1, color="k", lw=0.6)
        ax.set(xscale="log", xlabel=r"$\tau$ (s)")
        ax.set_title(short, fontsize=10)
        ax.tick_params(labelsize=8.5)
        ax.grid(color="#eeeeee", lw=0.5)
    axes[0].set_ylabel("MSD over MSD of the cohort to 1 h", fontsize=8)
    shades, ends = cohorts.shades(), (cohorts.COHORTS[0], cohorts.COHORTS[-1])
    handles = [Line2D([], [], color=shades[i], lw=1.4, label=f"cohort to {ends[j]:,} s") for j, i in enumerate((0, -1))]
    handles += [Line2D([], [], color=cohorts.POOL_COLOR, lw=1.4, label="pool")]
    fig.legend(handles=handles, frameon=False, loc="upper center", ncol=3, fontsize=9, columnspacing=1.6, handlelength=1.6)
    write(fig, "cohort_msd", rect=[0, 0, 1, 0.88], pad=0.5)


def cohort_gaps():
    """cohorts_gaps_msd.pdf, the bottom row without A - S: G - S and G - A of alpha of every nested cohort."""
    g = read("cohorts")["gaps"]
    fig, axes = plt.subplots(1, 2, figsize=(6.0, 2.0), sharey=True)
    for ax, name in zip(axes, ("G - S", "G - A")):
        for c, color in zip(g[name], cohorts.shades()):
            lags, band = np.array(c["lags"]), np.array(c["band"])
            ax.fill_between(lags, band[0], band[1], color=color, alpha=0.2, lw=0)
            ax.plot(lags, c["value"], color=color, lw=1.4, label=f"to {c['tau_max']:,} s")
        ax.axhline(0, color="k", lw=0.6)
        ax.set(xscale="log", xlabel=r"$\tau$ (s)", title=contrasts.title(name))
        ax.grid(color="#eeeeee", lw=0.5)
    axes[0].set(ylabel=r"$\Delta\alpha(\tau)$", ylim=cohorts.GAP_YLIM)
    axes[0].legend(frameon=False, loc="upper left", fontsize=7)
    write(fig, "cohort_gaps")


def weights():
    """weights.pdf, every flight: G - A of alpha* with one weight per piece (grey) and one per flight, and the three gaps with one
    weight per flight minus one per piece."""
    r = read("contrasts_all_perflight")["all"]
    fig, (left, right) = plt.subplots(1, 2, figsize=(6.0, 2.0))
    contrasts.curve(left, r["gaps per piece"]["G - A"], "#777777", "one weight per piece")
    contrasts.curve(left, r["gaps"]["G - A"], GAP["G - A"], "one weight per flight")
    left.set(ylabel=r"$\Delta\alpha^*$, G $-$ A", ylim=contrasts.ylim([r["gaps"]["G - A"], r["gaps per piece"]["G - A"]]))
    left.legend(frameon=False, loc="upper left", fontsize=7)
    for name in raw.GAPS:
        contrasts.curve(right, r["gaps minus per piece"][name], GAP[name], contrasts.title(name))
    right.set(ylabel=r"per flight $-$ per piece", ylim=contrasts.ylim([r["gaps minus per piece"][n] for n in raw.GAPS], (0, 0)))
    right.legend(frameon=False, loc="upper left", fontsize=7)
    for ax in (left, right):
        contrasts.finish(ax)
    write(fig, "weights")


# ---- is group time more persistent, the group or the flights ----

def persistent():
    """chained.pdf: alpha* of S, A, G (left) and G - S (right), solid up to cohorts.SELECTION_FREE and dashed beyond; the dotted
    cohort curves and the other gaps are in the story figure."""
    s = read("perlag")
    fig, (left, right) = plt.subplots(1, 2, figsize=(6.0, 2.0))
    for pop, short in raw.POPS.items():
        contrasts.curve(left, s["chained"][pop], COLORS[pop], short)
    left.set(ylabel=r"$\alpha^*(\tau)$", ylim=(0.78, cohorts.YLIM[1]))  # A goes just below cohorts.YLIM near 10^4 s
    left.legend(frameon=False, loc="upper right")
    contrasts.curve(right, s["chained gaps"]["G - S"], GAP["G - S"])
    right.set(ylabel=r"$\Delta\alpha^*(\tau)$", ylim=contrasts.ylim([s["chained gaps"]["G - S"]]))
    for ax in (left, right):
        contrasts.finish(ax)
    right.set_title(r"G $-$ S", color=GAP["G - S"], fontsize=10, fontweight="bold")
    left.set_title(r"$\alpha^*$ of S, A, G", fontsize=10)
    write(fig, "persistent")


def gaps():
    """identity.pdf: G - S, G - A and A - S of the chained alpha* on every flight."""
    s = read("perlag")
    fig, ax = plt.subplots(figsize=(3.6, 2.15))
    for name, g in s["chained gaps"].items():
        contrasts.curve(ax, g, GAP[name], contrasts.title(name))
    contrasts.finish(ax)
    ax.set(ylabel=r"$\Delta\alpha^*(\tau)$", ylim=contrasts.ylim(s["chained gaps"].values()))
    ax.legend(frameon=False, loc="upper left")
    write(fig, "gaps")


# ---- where ----

def ground_gaps():
    """ground_gaps.pdf, the G - A and A - S panels: one curve per ground class and on every flight (grey)."""
    r = read("contrasts_ground")
    names = ("G - A", "A - S")
    fig, axes = plt.subplots(1, 2, figsize=(6.0, 2.0), sharey=True)
    for ax, name in zip(axes, names):
        for k, color in (*CLASS_COLORS.items(), ("all", ALL)):
            contrasts.curve(ax, r[k]["gaps"][name], color, k)
        contrasts.finish(ax, contrasts.title(name))
    axes[0].set(ylabel=r"$\Delta\alpha^*(\tau)$", ylim=contrasts.ylim([r[k]["gaps"][n] for k in CLASSES for n in raw.GAPS]))
    axes[0].legend(frameon=False, loc="upper left", fontsize=7)
    write(fig, "ground_gaps")


def where():
    """Left: the share of the pieces of S, A, G at TAU in each ground class (the weights of simpson()). Right: alpha*(tau) of S per class
    (ground_alpha.pdf, the S panel)."""
    r = read("contrasts_ground")
    fig, (left, right) = plt.subplots(1, 2, figsize=(6.0, 2.0), gridspec_kw={"width_ratios": [1, 1.1]})
    pops = list(raw.POPS.items())[::-1]  # G on top
    for y, (pop, short) in enumerate(pops):
        n = np.array([r[k]["alpha"][pop]["pieces"][r[k]["alpha"][pop]["lags"].index(TAU)] for k in CLASSES], dtype=float)
        start = 0.0
        shares = n / n.sum()
        pct = percent(shares)
        for i, (k, share) in enumerate(zip(CLASSES, shares)):
            left.barh(y, share, left=start, color=CLASS_COLORS[k], height=0.62, label=k if y == 0 else None)
            if share > 0.08:
                left.text(start + share / 2, y, f"{pct[i]}%", ha="center", va="center", fontsize=7,
                          color="white" if k in CLASSES[1:] else INK)
            start += share
    left.set_yticks(range(len(pops)), [short for _, short in pops])
    left.set(xlim=(0, 1), xlabel=r"share of the pieces at $10^3$ s")
    left.xaxis.set_major_formatter(lambda x, _: f"{100 * x:.0f}%")
    left.spines[["top", "right"]].set_visible(False)
    left.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=4, fontsize=6.5, handlelength=1, columnspacing=0.8)
    for k in CLASSES:
        contrasts.curve(right, r[k]["alpha"]["solo"], CLASS_COLORS[k], k)
    contrasts.finish(right)
    right.set(ylabel=r"$\alpha^*(\tau)$ of S", ylim=(0.74, 1.0))  # the dashed mountain tails run below
    write(fig, "where")


def scale():
    """ground_relgaps.pdf, the G - A panel: the normalised gap (alpha*_G - alpha*_A) / (1 - alpha*_A) per ground class."""
    r = read("contrasts_ground")
    fig, ax = plt.subplots(figsize=(3.6, 2.15))
    for k in CLASSES:
        contrasts.curve(ax, r[k]["relative gaps"]["G - A"], CLASS_COLORS[k], k)
    contrasts.finish(ax)
    ax.set(ylabel=r"$(\alpha^*_G-\alpha^*_A)/(1-\alpha^*_A)$", ylim=contrasts.REL_YLIM)
    ax.legend(frameon=False, loc="upper left", fontsize=7)
    write(fig, "scale")


def cycle():
    """ground_relgaps.pdf, the G - A panel, with the glide and climb cycle of S, A, G in every class as a grey band on the lag
    axis and the peak of each curve as a dot (hollow: at the end of the solid range)."""
    r, p = read("contrasts_ground"), read("phases_refit")
    cyc = [60 * p["phases"][c]["statistics"][pop]["cycle of a glide and a climb (min)"]["value"] for c in CLASSES for pop in "SAG"]
    fig, ax = plt.subplots(figsize=(3.6, 2.15))
    ax.axvspan(min(cyc), max(cyc), color=GREY, alpha=0.35, lw=0)
    ax.text(min(cyc) / 1.1, 0.42, "cycle", color="#555555", fontsize=8, fontweight="bold", va="top", ha="right")
    for c in CLASSES:
        q = r[c]["relative gaps"]["G - A"]
        contrasts.curve(ax, q, CLASS_COLORS[c], c)
        end = max(t for t, ok in zip(q["lags"], q["solid"]) if ok)
        pk = p["peaks"][c]["G - A"]
        ax.plot(pk["lag"], pk["value"], "o", ms=6, mew=1.6, mec=CLASS_COLORS[c], mfc=CLASS_COLORS[c] if pk["lag"] < end else "white", zorder=5)
    contrasts.finish(ax)
    ax.set(ylabel=r"$(\alpha^*_G-\alpha^*_A)/(1-\alpha^*_A)$", ylim=contrasts.REL_YLIM, xlim=(10, 1e4))
    ax.legend(frameon=False, loc="upper left", fontsize=7)
    write(fig, "cycle")


# ---- which flights ----

def draw_cohort_gaps(ax, name, tau_max=3600):
    """cohorts.draw_gaps for one cohort: the gap `name` of the nested cohort to `tau_max`, at each threshold of the opening."""
    for tag, color, text in zip(("", "_min0.1", "_min0.5"), OPENING, OPENING_TEXT):
        c = [c for c in read(f"cohorts{tag}")["gaps"][name] if c["tau_max"] == tau_max][0]
        lags, band = np.array(c["lags"]), np.array(c["band"])
        ax.fill_between(lags, band[0], band[1], color=color, alpha=0.25, lw=0)
        ax.plot(lags, c["value"], color=color, lw=1.6, label=text)
    ax.axhline(0, color="k", lw=0.6)
    ax.set(xscale="log", xlabel=r"$\tau$ (s)", title=contrasts.title(name))
    ax.grid(color="#eeeeee", lw=0.5)


def opening():
    """cohorts_gaps.pdf and its variants with the opening above 0.1 and 0.5: G - S and G - A of the nested cohort to 3600 s."""
    fig, axes = plt.subplots(1, 2, figsize=(6.0, 2.0), sharey=True)
    for ax, name in zip(axes, ("G - S", "G - A")):
        draw_cohort_gaps(ax, name)
    axes[0].set(ylabel=r"$\Delta\alpha(\tau)$", ylim=cohorts.GAP_YLIM)
    axes[0].legend(frameon=False, loc="upper left")
    write(fig, "opening")


def opening_kind():
    """opening_kind.pdf without the threshold lines: the opening of the solo flights and of the flights with group time (the
    flights of opening.json; the per-flight openings are read from the flight table, since the JSON keeps only their summary)."""
    every = closure.flight_table()
    open_ = every[(every["declared"] == "open") & (every["dt_native"] <= contrasts.longlags.MAX_DT + 1e-6)]
    fig, ax = plt.subplots(figsize=(4.2, 2.9))
    for name, m, color in (("solo flights", ~open_["group"], S), ("group flights", open_["group"], G)):
        r = open_.loc[m, "ratio"]
        ax.hist(r.clip(0, 1), bins=np.linspace(0, 1, 41), density=True, histtype="step", lw=1.5, color=color,
                label=f"{name} ({len(r):,}), median {r.median():.2f}")
    ax.set(xlabel=r"opening $\mathcal{O}$", ylabel="density")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    write(fig, "opening_kind")


def class_row(width, height, ylabel):
    """A row of one panel per ground class on a shared lag axis."""
    fig, axes = plt.subplots(1, len(CLASSES), figsize=(width, height), sharex=True, sharey=True)
    axes[0].set_ylabel(ylabel)
    return fig, axes


def opening_ground():
    """ground_opening_rel.pdf, G - A row: the normalised G - A per ground class, with no threshold and above 0.1 and 0.5."""
    r = read("contrasts_ground_opening")
    fig, axes = class_row(WIDE, 2.0, r"$(\alpha^*_G-\alpha^*_A)/(1-\alpha^*_A)$")
    gs = [r[f"{c}|{contrasts.threshold(x)}"]["relative gaps"]["G - A"] for c in CLASSES for x in contrasts.THRESHOLDS]
    for ax, c in zip(axes, CLASSES):
        for x, color, text in zip(contrasts.THRESHOLDS, OPENING, OPENING_TEXT):
            contrasts.curve(ax, r[f"{c}|{contrasts.threshold(x)}"]["relative gaps"]["G - A"], color, text)
        contrasts.finish(ax, c)
    axes[0].set_ylim(contrasts.ylim(gs, (0, 0)))
    fig.legend(handles=axes[0].get_legend_handles_labels()[0], labels=OPENING_TEXT, frameon=False, loc="upper center", ncol=3,
               bbox_to_anchor=(0.5, 1.0))
    write(fig, "opening_ground", rect=(0, 0, 1, 0.9))


# ---- who, when ----

def pilots():
    """pilot.pdf, per class: A - S and A - S' (the solo flights of pilots who also fly in groups)."""
    r = read("contrasts_pilot")
    names = ("A - S", "A - S'")
    fig, axes = class_row(WIDE, 2.0, r"$\Delta\alpha^*(\tau)$")
    for ax, c in zip(axes, CLASSES):
        for name in names:
            contrasts.curve(ax, r[c]["gaps"][name], contrasts.PILOT_COLORS[name], contrasts.title(name))
        contrasts.finish(ax, c)
    axes[0].set_ylim(contrasts.ylim([r[k]["gaps"][n] for k in CLASSES for n in names]))
    axes[0].legend(frameon=False, loc="upper left", fontsize=7)
    write(fig, "pilots")


def steps(suffix, name):
    """prepost_ground.pdf and prepost_ground_long.pdf, panel of every flight: the exponent of the four windows (left) and the
    three steps with the drift between two consecutive solo windows of group flights (right); a dashed line joins the four
    exponents."""
    r = read("prepost_ground")["classes"]["all"]
    fig, (left, right) = plt.subplots(1, 2, figsize=(6.0, 2.0), gridspec_kw={"width_ratios": [1, 1.1]})
    left.plot(range(4), [r["alpha" + suffix][s]["value"] for s in prepost.STAGES], color=GREY, lw=1.0, ls="--", zorder=1)
    for i, stage in enumerate(prepost.STAGES):
        bar(left, i, None, r["alpha" + suffix][stage], True, prepost.LINES[stage][0], ms=7)
    left.set(xticks=range(4), xticklabels=prepost.STAGES, xlim=(-0.5, 3.5), ylabel=r"$\alpha$ of the window")
    keys = ("start - pre", "end - start", "post - end", "ctrlA2 - ctrlA1")
    for i, key in enumerate(keys):
        bar(right, i, None, r["differences" + suffix][key], True, prepost.GROUND_COLORS[key], ms=7)
    right.axhline(0, color="k", lw=0.6)
    right.set(xticks=range(4), xticklabels=["entry", "inside", "exit", "drift"], xlim=(-0.5, 3.5), ylabel=r"difference of $\alpha$")
    for ax in (left, right):
        ax.grid(color="#eeeeee", lw=0.5)
        ax.tick_params(axis="x", labelsize=8)
    write(fig, name)


def when_all():
    steps("", "when_all")


def when_long():
    steps("_long", "when_long")


def when_long_classes():
    """prepost_ground_long.pdf, the difference row: the three steps of the exponent over the long lags and the drift between two
    consecutive solo windows of group flights, per ground class and on every flight, with 90% bands."""
    r = read("prepost_ground")["classes"]
    keys = ("start - pre", "end - start", "post - end", "ctrlA2 - ctrlA1")
    names = ("entry", "inside", "exit", "drift")
    fig, ax = plt.subplots(figsize=(6.0, 2.0))
    for i, k in enumerate(KEYS):
        for j, key in enumerate(keys):
            bar(ax, i + (j - 1.5) * 0.17, None, r[k]["differences_long"][key], True, prepost.GROUND_COLORS[key], ms=5, lw=1.3, capsize=2)
    ax.axhline(0, color="k", lw=0.6)
    ax.set(xticks=range(len(KEYS)), xticklabels=SHORT[:-1] + ("every flight",), xlim=(-0.6, len(KEYS) - 0.4), ylabel=r"difference of $\alpha$")
    ax.grid(axis="y", color="#eeeeee", lw=0.5)
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=prepost.GROUND_COLORS[k], ms=5, label=n) for k, n in zip(keys, names)],
              frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.17), ncol=4, fontsize=8, handletextpad=0.1, columnspacing=1.2)
    write(fig, "when_long_classes")


def control_steps():
    """The three steps of the exponent over the long lags (prepost_ground.json, "mock", every flight): the real events that have a
    control and the control events (solo flights at the same flight-clock times), each with its 90% band."""
    m = read("prepost_ground")["mock"]["classes"]["all"]
    keys = ("start - pre", "end - start", "post - end")
    fig, ax = plt.subplots(figsize=(3.6, 2.2))
    for key, off, color in (("real", -0.12, INK), ("mock", 0.12, GREY)):
        for i, k in enumerate(keys):
            bar(ax, i + off, None, m[key][k], True, color, ms=7, label="real" if key == "real" and i == 0 else "control (solo flights)" if i == 0 else None)
    ax.axhline(0, color="k", lw=0.6)
    ax.set(xticks=range(3), xticklabels=["entry", "inside", "exit"], xlim=(-0.5, 2.5), ylabel=r"step of $\alpha$")
    ax.grid(color="#eeeeee", lw=0.5)
    ax.legend(frameon=False, loc="lower left", fontsize=7)
    write(fig, "control")


def carry():
    """carry.pdf, bottom row, every flight: A(d) - S of the exponent over the long lags per bin of the time d from the nearest
    group time; a grey dashed sketch (not data) shows the shape a carry-over would have."""
    c = read("carry")
    fig, ax0 = plt.subplots(figsize=(3.6, 2.1))
    axes = [ax0]
    for ax, k in zip(axes, ("all",)):
        r = c["classes"][k]
        for i, b in enumerate(r["bins"]):
            bar(ax, i, None, b["diff"], True, carry_module.BIN_COLOR, ms=7)
        # a sketch, not data: the shape a carry-over would have, high next to the group and fading with d to the far value
        n, far = len(r["bins"]), r["bins"][-1]["diff"]["value"]
        sketch = far + 0.012 * np.exp(-np.arange(n) / 0.9)
        ax.plot(range(n), sketch, color=GREY, lw=1.4, ls="--", label="if carry-over (sketch)")
        ax.legend(frameon=False, fontsize=7.5, loc="lower right")
        ax.axhline(0, color="k", lw=0.6)
        ax.set(xticks=range(len(r["bins"])), xticklabels=[t.replace(">=", "≥") for t in c["plain"]], xlim=(-0.5, len(r["bins"]) - 0.5),
               title="every flight" if k == "all" else k)
        ax.tick_params(axis="x", labelsize=7)
        ax.grid(color="#eeeeee", lw=0.5)
    axes[0].set_ylabel(r"$\alpha_{A(d)}-\alpha_S$")
    write(fig, "carry")


# ---- dynamics ----

KEYS = phase_gaps.CLASS_KEYS  # the four ground classes and every flight
SHORT = ("plains", "hills", "low mtn", "high mtn", "all")
KCOLORS = [CLASS_COLORS[c] for c in CLASSES] + [INK]
PHASES = (("glide", "time in glide (%)", "#7f9bb8"), ("search", "time in search (%)", "#9a7fc0"),
          ("climb", "time in climb (%)", "#2f4465"), ("exit", "time in exit (%)", "#b3aea4"))


def phase_data():
    return read("phases_refit")["phases"]


def class_axis(ax, ylabel):
    """Zero line, grid and the short class names under the dots of a panel whose x axis is the ground class."""
    ax.axhline(0, color="k", lw=0.6)
    ax.grid(axis="y", color="#eeeeee", lw=0.5)
    ax.set_xticks(range(len(KEYS)), SHORT, rotation=45, ha="right", rotation_mode="anchor", fontsize=7.5)
    ax.set_xlim(-0.6, len(KEYS) - 0.4)
    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.set_ylabel(ylabel)


def gap_panel(ax, ph, stat, title, ylabel):
    """G - A of the statistic `stat` per class and on every flight, with the 90% bands."""
    for i, (k, color) in enumerate(zip(KEYS, KCOLORS)):
        bar(ax, i, None, ph[k]["gaps"]["G - A"][stat], True, color, ms=6)
    ax.set_title(title, fontsize=8.5)
    class_axis(ax, ylabel)


def dyn_time():
    """Left: the share of the time in glide, search, climb and exit of S, A and G on every flight. Right: G - A (top) and A - S
    (bottom) of the three main shares per class and on every flight, with 90% bands."""
    ph = phase_data()
    fig = plt.figure(figsize=(WIDE, 2.5))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.5, 1], hspace=0.2, wspace=0.28)
    left = fig.add_subplot(gs[:, 0])
    for y, (short, color) in enumerate((("S", S), ("A", A), ("G", G))):
        x0 = 0
        for _, stat, c in PHASES:
            v = ph["all"]["statistics"][short][stat]["value"]
            left.barh(y, v, left=x0, height=0.66, color=c, edgecolor="white", lw=0.8)
            if v >= 15:
                left.text(x0 + v / 2, y, f"{v:.1f}", ha="center", va="center", fontsize=8, color="white" if c in ("#2f4465", "#7f9bb8", "#9a7fc0") else INK)
            elif v > 5:  # a narrow segment: its label above the bar
                left.text(x0 + v / 2, y - 0.37, f"{v:.1f}", ha="center", va="bottom", fontsize=7.5, color=INK)
            x0 += v
    left.set(xlim=(0, 100), ylim=(2.55, -0.55), yticks=[0, 1, 2], yticklabels=["S", "A", "G"], xlabel="share of the time (%)")
    for tick, color in zip(left.get_yticklabels(), (S, A, G)):
        tick.set(color=color, fontsize=11, fontweight="bold")
    left.tick_params(axis="y", length=0)
    left.spines[["top", "right", "left"]].set_visible(False)
    for row, gap in enumerate(("G - A", "A - S")):
        ax = fig.add_subplot(gs[row, 1])
        for j, (_, stat, c) in enumerate(PHASES[:3]):
            for i, k in enumerate(KEYS):
                bar(ax, i + (j - 1) * 0.26, None, ph[k]["gaps"][gap][stat], True, c, ms=4.5)
        class_axis(ax, gap.replace("-", "−") + " (pts)")
        if row == 0:
            ax.set_xticklabels([])
    fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in PHASES], labels=[n for n, _, _ in PHASES], frameon=False,
               loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.0), handlelength=1.2, columnspacing=1.2)
    write(fig, "dyn_time", rect=(0, 0, 1, 0.92))


FIND_ROWS = ((0, "Found more reliably?", None),
             (1.1, "searches ending in a climb", "searches that end in a climb (%)"),
             (2.9, "searches per hour", "searches per hour"),
             (4.7, "time in search", "time in search (%)"),
             (6.1, "Stronger?", None),
             (7.2, "climb rate", "climb rate (m/s)"),
             (9.0, "mean exit duration", "mean exit duration (s)"))
GAP_ROWS = (("G - A", "A", "o", -0.4), ("A - S", "S", "s", 0.4))  # gap, the population it is measured against, marker, sub-row offset


def marker_legend(fig, ms=4.5):
    """The legend of the markers of the two gaps (grey circle: G - A in % of A; grey square: A - S in % of S) above the classes'."""
    return [Line2D([], [], marker=m, ls="", color=GREY, ms=ms, label=f"{g.replace('-', '−')} (% of {base})") for g, base, m, _ in GAP_ROWS]


def dyn_find():
    """100 (G - A) / A of five statistics in two blocks (found more reliably: searches that end in a climb, searches per hour, time
    in search; stronger: climb rate, mean exit duration), with one dot per ground class and one, larger and black, on every flight,
    and 90% bands: the band of the gap divided by the A value, times 100."""
    ph = phase_data()
    offsets = np.linspace(-0.4, 0.4, len(KEYS))
    fig, ax = plt.subplots(figsize=(5.0, 4.9))
    for r, label, stat in FIND_ROWS:
        if stat is None:
            continue
        for k, color, off in zip(KEYS, KCOLORS, offsets):
            a = ph[k]["statistics"]["A"][stat]["value"]
            g = ph[k]["gaps"]["G - A"][stat]
            q = {"value": 100 * g["value"] / a, "band": [100 * g["band"][0] / a, 100 * g["band"][1] / a]}
            bar(ax, None, r + off, q, False, color, ms=6 if k == "all" else 3.5, lw=1.3, capsize=2, fmt="o",
                zorder=3 if k == "all" else 2)
    ax.set_ylim(9.8, -0.4)
    ax.set_yticks([r for r, _, _ in FIND_ROWS], [label for _, label, _ in FIND_ROWS], fontsize=9.5)
    for tick, (_, _, stat) in zip(ax.get_yticklabels(), FIND_ROWS):
        if stat is None:
            tick.set(fontweight="bold", color=INK)
    ax.tick_params(axis="y", length=0)
    ax.axvline(0, color="k", lw=0.8)
    ax.grid(axis="x", color="#eeeeee", lw=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", labelsize=9.5)
    ax.set_xlabel("G − A (% of A)", fontsize=9.5)
    ax.spines[["top", "right", "left"]].set_visible(False)
    fig.legend(handles=[Line2D([], [], marker="o", ls="", color=c, ms=6 if k == "all" else 4.5) for k, c in zip(KEYS, KCOLORS)],
               labels=SHORT[:-1] + ("every flight",), frameon=False, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.0),
               fontsize=8.5, handletextpad=0.05, columnspacing=0.6)
    write(fig, "dyn_find", rect=(0, 0, 1, 0.94))


DYN_BLOCKS = (("Glide", (("share of time", "time in glide (%)"), ("per hour", "glides per hour"), ("mean duration", "mean glide duration (s)"),
                         ("glide speed", "glide speed (km/h)"), ("height lost/km", "height lost per km of glide (m/km)"))),
              ("Search", (("share of time", "time in search (%)"), ("per hour", "searches per hour"), ("mean duration", "mean search duration (s)"),
                          ("ending in\na climb", "searches that end in a climb (%)"))),
              ("Climb", (("share of time", "time in climb (%)"), ("per hour", "climbs per hour"), ("mean duration", "mean climb duration (s)"),
                         ("climb rate", "climb rate (m/s)"))))


def percent_gap(ph, k, gap, base, stat):
    """The gap `gap` of the class `k` for `stat` as a percentage of the statistic of `base` (A for G - A, S for A - S), with its band."""
    a = ph[k]["statistics"][base][stat]["value"]
    g = ph[k]["gaps"][gap][stat]
    return {"value": 100 * g["value"] / a, "band": [100 * g["band"][0] / a, 100 * g["band"][1] / a]}


def dyn_phases(full=False):
    """The gaps of every statistic of the three phases, in % of the population they are measured against, one panel per phase
    with its name in bold (glide: share of time, per hour, mean duration, glide speed, height lost per km; search: share of time,
    per hour, mean duration, share ending in a climb; climb: share of time, per hour, mean duration, climb rate). 100 (G - A) / A
    per ground class and on every flight (circles, the larger black one: every flight), with 90% bands; with `full`
    (dyn_phases_full, the story's) every statistic has two lines, G - A above 100 (A - S) / S (squares), per class and on every
    flight. The panels share the x axis."""
    ph = phase_data()
    fig, axes = plt.subplots(1, 3, figsize=(WIDE, 3.3 if full else 2.3), sharex=True)
    for ax, (title, rows) in zip(axes, DYN_BLOCKS):
        for r, (_, stat) in enumerate(rows):
            if full:
                for gap, base, fmt, shift in GAP_ROWS:
                    for k, color, off in zip(KEYS, KCOLORS, np.linspace(-0.13, 0.13, len(KEYS))):
                        bar(ax, None, r + 0.75 * shift + off, percent_gap(ph, k, gap, base, stat), False, color, ms=4.6 if k == "all" else 2.7,
                            lw=1.0, capsize=1.3, fmt=fmt, zorder=3 if k == "all" else 2)
            else:
                for k, color, off in zip(KEYS, KCOLORS, np.linspace(-0.3, 0.3, len(KEYS))):
                    bar(ax, None, r + off, percent_gap(ph, k, "G - A", "A", stat), False, color, ms=5 if k == "all" else 3, lw=1.1, capsize=1.5,
                        zorder=3 if k == "all" else 2)
        ax.set_ylim(len(rows) - 0.4, -0.6)
        ax.set_yticks(range(len(rows)), [label for label, _ in rows], fontsize=8.5)
        ax.tick_params(axis="y", length=0)
        ax.tick_params(axis="x", labelsize=8.5)
        ax.axvline(0, color="k", lw=0.8)
        ax.grid(axis="x", color="#eeeeee", lw=0.5)
        ax.set_axisbelow(True)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.set_title(title, fontsize=10.5, fontweight="bold", color=INK, pad=3)
        ax.set_xlabel("gap (%)", fontsize=8.5)
    fig.legend(handles=[Line2D([], [], marker="o", ls="", color=c, ms=5 if k == "all" else 4) for k, c in zip(KEYS, KCOLORS)],
               labels=SHORT[:-1] + ("every flight",), frameon=False, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.0), fontsize=9,
               handletextpad=0.1, columnspacing=0.9)
    handles = marker_legend(fig, 4) if full else [Line2D([], [], marker="o", ls="", color=GREY, ms=4, label="G − A (% of A)")]
    fig.legend(handles=handles, frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.925 if full else 0.9), fontsize=9,
               handletextpad=0.1, columnspacing=1.2)
    write(fig, "dyn_phases_full" if full else "dyn_phases", rect=(0, 0, 1, 0.88 if full else 0.84))


def dyn_joins():
    """The ratio of the share of the events observed in each phase to the share expected at random times of the time alone
    (phase_joins.json), at the instant of entering group time (left) and of leaving it (right): one row per phase, one dot per ground
    class and one, larger and black, on every flight, with 90% bands; the line at 1 is chance."""
    j = read("phase_joins")
    offsets = np.linspace(-0.36, 0.36, len(KEYS))
    fig, axes = plt.subplots(1, 2, figsize=(WIDE, 1.9), sharex=True, sharey=True)
    for ax, (kind, title) in zip(axes, (("join", "enter a group"), ("leave", "leave a group"))):
        for r, phase in enumerate(j["phases"]):
            for k, color, off in zip(KEYS, KCOLORS, offsets):
                bar(ax, None, r + off, j[kind][k][phase]["ratio"], False, color, ms=5 if k == "all" else 3, lw=1.2, capsize=2,
                    zorder=3 if k == "all" else 2)
        ax.axvline(1, color="k", lw=0.8)
        ax.set_ylim(len(j["phases"]) - 0.4, -0.6)
        ax.set_yticks(range(len(j["phases"])), j["phases"], fontsize=8.5)
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="x", color="#eeeeee", lw=0.5)
        ax.set_axisbelow(True)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.set_title(title, fontsize=9.5, fontweight="bold", color=INK, pad=3)
        ax.set_xlabel("observed / chance", fontsize=8.5)
    fig.legend(handles=[Line2D([], [], marker="o", ls="", color=c, ms=5 if k == "all" else 3.5) for k, c in zip(KEYS, KCOLORS)],
               labels=SHORT[:-1] + ("every flight",), frameon=False, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.0), fontsize=8,
               handletextpad=0.1, columnspacing=1.0)
    write(fig, "dyn_joins", rect=(0, 0, 1, 0.86))


def dyn_glide():
    """G - A per class and on every flight of the glide speed and the height lost per km of glide, with 90% bands."""
    ph = phase_data()
    panels = (("glide speed (km/h)", "glide speed", "G − A (km/h)"),
              ("height lost per km of glide (m/km)", "height lost per km of glide", "G − A (m/km)"))
    fig, axes = plt.subplots(1, 2, figsize=(4.8, 2.0))
    for ax, (stat, title, ylabel) in zip(axes, panels):
        gap_panel(ax, ph, stat, title, ylabel)
    write(fig, "dyn_glide")


def dyn_perf():
    """phases_performance.pdf, three panels: G - A and A - S per ground class and on every flight of the height lost per km of glide,
    the searches that end in a climb and the climb rate, with 90% bands."""
    ph = phase_data()
    panels = (("height lost per km of glide (m/km)", "height lost\nper km (m/km)"), ("searches that end in a climb (%)", "searches ending\nin a climb (%)"),
              ("climb rate (m/s)", "climb rate (m/s)"))
    fig, axes = plt.subplots(1, 3, figsize=(WIDE, 2.2), sharey=True)
    for ax, (stat, title) in zip(axes, panels):
        for i, k in enumerate(KEYS):
            for gap, off in (("G - A", -0.17), ("A - S", 0.17)):
                bar(ax, None, i + off, ph[k]["gaps"][gap][stat], False, GAP[gap], ms=4.5, lw=1.3, capsize=2)
        ax.axvline(0, color="k", lw=0.6)
        ax.set_title(title, fontsize=8.5)
        ax.set_ylim(len(KEYS) - 0.4, -0.6)
        ax.grid(axis="x", color="#eeeeee", lw=0.5)
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.tick_params(labelsize=8)
    axes[0].set_yticks(range(len(KEYS)), SHORT[:-1] + ("every flight",))
    fig.legend(handles=[Line2D([], [], marker="o", ls="", color=GAP[g], ms=5, label=contrasts.title(g)) for g in ("G - A", "A - S")],
               frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.0), fontsize=9, handletextpad=0.1, columnspacing=1.4)
    write(fig, "dyn_perf", rect=(0, 0, 1, 0.88))


def tail_panels(name, draw, ylabel, xlabel, xscale="log"):
    """A row of three panels, glides, searches and climbs, on every flight; `draw(ax, phase, d)` fills one."""
    d = read("phases_refit")["tails"]["all"]
    fig, axes = plt.subplots(1, 3, figsize=(WIDE, 2.0), sharey=False)
    for ax, (phase, title) in zip(axes, (("transition", "glides"), ("search", "searches (any outcome)"), ("climb", "climbs"))):
        draw(ax, d[phase])
        ax.axvline(d[phase]["xmin"]["common"], color=GREY, lw=1.0, ls=":")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel(xlabel, fontsize=8.5)
        ax.tick_params(labelsize=8)
        ax.grid(color="#eeeeee", lw=0.5)
    axes[0].set_ylabel(ylabel, fontsize=9)
    fig.legend(handles=[Line2D([], [], color=COLORS[pop], lw=1.5, label=short) for pop, short in raw.POPS.items()] +
               [Line2D([], [], color=GREY, lw=1.0, ls=":", label=r"common $x_\mathrm{min}$")], frameon=False, loc="upper center", ncol=4,
               bbox_to_anchor=(0.5, 1.0), fontsize=8.5, handlelength=1.4, columnspacing=1.4)
    write(fig, name, rect=(0, 0, 1, 0.9))


def tails_survival():
    """phases_tails.pdf, the panel of every flight of each phase: the survival P(X>x) of S, A, G with its 90% band; the common
    threshold dotted."""
    def draw(ax, d):
        for pop, short in raw.POPS.items():
            s = d["survival"][short]
            g, v, b = np.array(s["grid"]), np.array(s["value"]), np.array(s["band"])
            ok = v > 0
            ax.fill_between(g[ok], np.maximum(b[0][ok], 1e-9), b[1][ok], color=COLORS[pop], alpha=0.25, lw=0)
            ax.plot(g[ok], v[ok], color=COLORS[pop], lw=1.4)
        ax.set(xscale="log", yscale="log", ylim=(1e-5, 1.2))
        ax.set_yticks([1, 1e-2, 1e-4])
        ax.minorticks_off()
    tail_panels("survival", draw, "$P(X>x)$", "duration $x$ (s)")


def tails_hill():
    """phases_hill.pdf, the panel of every flight of each phase: the Hill exponent mu(x_min) of S, A, G with its 90% band; the common
    threshold dotted."""
    def draw(ax, d):
        for pop, short in raw.POPS.items():
            h = d["hill"][short]
            ax.fill_between(h["xmin"], *h["band"], color=COLORS[pop], alpha=0.25, lw=0)
            ax.plot(h["xmin"], h["value"], color=COLORS[pop], lw=1.4)
        v = np.concatenate([d["hill"][short]["value"] for short in raw.POPS.values()])
        ax.set(xscale="log", ylim=(0.9 * v.min(), 1.1 * v.max()))
        ax.minorticks_off()
    tail_panels("hill", draw, r"$\hat\mu(x_\mathrm{min})$", r"$x_\mathrm{min}$ (s)")


def survival_panel(ax, phase, c, title, legend=False):
    """The survival of the durations of S, A, G with its 90% band and the common threshold (dotted)."""
    d = read("phases_refit")["tails"][c][phase]
    for pop, short in raw.POPS.items():
        s = d["survival"][short]
        g, v, b = np.array(s["grid"]), np.array(s["value"]), np.array(s["band"])
        ok = v > 0
        ax.fill_between(g[ok], np.maximum(b[0][ok], 1e-9), b[1][ok], color=COLORS[pop], alpha=0.3, lw=0)
        ax.plot(g[ok], v[ok], color=COLORS[pop], lw=1.5, label=short)
    ax.axvline(d["xmin"]["common"], color=GREY, lw=0.8, ls=":")
    ax.set(xscale="log", yscale="log", xlim=(1, 8e3), ylim=(1e-5, 1.2), xlabel="duration $x$ (s)", title=title)
    ax.set_xticks([1, 100, 1e4])
    ax.set_yticks([1, 1e-2, 1e-4])
    ax.minorticks_off()
    ax.grid(color="#eeeeee", lw=0.5)
    ax.set_ylabel("$P(X>x)$")
    if legend:
        ax.legend(frameon=False, loc="lower left", fontsize=7.5, handlelength=1.2)


def forest_axis(ax, labels):
    """The ground classes down the y axis of a horizontal forest, named only if `labels`."""
    ax.set_ylim(len(KEYS) - 0.4, -0.6)
    ax.set_yticks(range(len(KEYS)), SHORT if labels else [], fontsize=8)
    ax.grid(axis="x", color="#eeeeee", lw=0.5)
    ax.xaxis.set_major_locator(MaxNLocator(3, integer=True))


def mu_panel(ax, phase, title, labels=True):
    """The Hill exponent mu of S, A, G per class and on every flight, with 90% bands."""
    t = read("phases_refit")["tails"]
    for i, k in enumerate(KEYS):
        for off, pop, short in ((-0.26, "solo", "S"), (0, "group_solo", "A"), (0.26, "group", "G")):
            bar(ax, None, i + off, t[k][phase]["mu"][short], False, COLORS[pop], ms=4.5)
    forest_axis(ax, labels)
    ax.set(title=title, xlabel=r"$\mu$ (Hill exponent)")


def mu_gap_panel(ax, series, title):
    """mu_G - mu_A per class and on every flight, with 90% bands: one series per (phase, marker, offset) of `series`."""
    t = read("phases_refit")["tails"]
    for phase, fmt, off in series:
        for i, k in enumerate(KEYS):
            bar(ax, None, i + off, t[k][phase]["gaps"]["G - A"], False, GAP["G - A"], ms=5, fmt=fmt)
    ax.axvline(0, color="k", lw=0.6)
    forest_axis(ax, False)
    ax.set(title=title, xlabel=r"$\mu_G-\mu_A$")


def survival_curve(ax, g, v, band, n, color, lw, min_count=20, risk=None):
    """A survival curve with its 90% band, from its first to its last positive point. The part where fewer than `min_count`
    events remain above x is dashed: the count is n * P(X>x), n the number of events of the population, or `risk` (the number
    at risk at each point of `g`) when given; with `risk` the curve stops where nobody is left at risk."""
    ok = v > 0
    if risk is not None:
        ok &= risk > 0
    ax.fill_between(g[ok], np.maximum(band[0][ok], 1e-9), band[1][ok], color=color, alpha=0.3, lw=0)
    few = ok & ((v * n if risk is None else risk) < min_count)
    solid = ok & ~few
    ax.plot(g[solid], v[solid], color=color, lw=lw)
    if few.any():
        i = np.flatnonzero(few)[0]
        j = max(i - 1, 0)
        ax.plot(g[j:][ok[j:]], v[j:][ok[j:]], color=color, lw=lw, ls="--")
    return g[ok].max()


def few_legend(ax, loc, anchor=None):
    """A second legend of `ax` for the dashed part of a survival curve (the S, A, G legend of the panel stays)."""
    if ax.get_legend():
        ax.add_artist(ax.get_legend())
    ax.legend(handles=[Line2D([], [], color=GREY, lw=1.0, ls="--", label="fewer than 20 left")], loc=loc, bbox_to_anchor=anchor,
              fontsize=7, handlelength=1.0, handletextpad=0.3, borderaxespad=0.1, borderpad=0.2, facecolor="white", edgecolor="none",
              framealpha=0.9).set_zorder(5)


def dyn_tails_glides():
    """Glide durations on every flight. Left (2/3 of the width): mu_G - mu_A per ground class and on every flight, with 90%
    bands. Right: the survival P(X>x) of S, A, G with its 90% band on log axes, from a little before the common threshold
    x_min (dashed line). Each curve runs to its own last positive point, dashed where fewer than 20 durations remain above x
    (n P(X>x), n the number of durations of the population, from the JSON); the x axis ends at the largest last point."""
    t = read("phases_refit")["tails"]
    d = t["all"]["transition"]
    xmin = d["xmin"]["common"]
    fig = plt.figure(figsize=(6.4, 2.1))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 0.3, 1.35], wspace=0.5)
    left = fig.add_subplot(gs[0, :2])
    for i, k in enumerate(KEYS):
        bar(left, None, i, t[k]["transition"]["gaps"]["G - A"], False, "#7F9BB8", ms=5)
    left.axvline(0, color="k", lw=0.8)
    forest_axis(left, False)
    left.set_yticks(range(len(KEYS)), SHORT[:-1] + ("every flight",))
    left.xaxis.set_major_locator(MaxNLocator(5, integer=True))
    left.tick_params(labelsize=8)
    left.tick_params(axis="y", length=0)
    left.set_xlabel(r"$\mu_G-\mu_A$", fontsize=9)
    left.spines[["top", "right", "left"]].set_visible(False)
    right = fig.add_subplot(gs[0, 3])
    x0, xmax, lo, hi = 0.7 * xmin, 0, 1.0, 0.0
    for pop, k in (("solo", "S"), ("group_solo", "A"), ("group", "G")):
        s = {key: np.array(d["survival"][k][key]) for key in ("grid", "value", "band")}
        xmax = max(xmax, survival_curve(right, s["grid"], s["value"], s["band"], d["runs"][k], COLORS[pop], 1.4))
        sel = (s["grid"] >= x0) & (s["value"] > 0)
        lo, hi = min(lo, s["value"][sel].min()), max(hi, s["band"][1][sel].max())
    right.axvline(xmin, color=GREY, lw=1.2, ls=":")
    right.text(xmin, 1.0, r"$x_\mathrm{min}$", color=GREY, fontsize=8, ha="center", va="bottom", transform=right.get_xaxis_transform())
    right.set(xscale="log", yscale="log", xlim=(x0, xmax * 1.05), ylim=(lo / 3, hi * 1.5))
    right.minorticks_off()
    right.xaxis.set_major_locator(plt.FixedLocator([2000, 4000]))
    right.set_xticklabels(["2000", "4000"])
    right.grid(color="#eeeeee", lw=0.5)
    right.tick_params(labelsize=8, pad=1)
    right.set_xlabel("duration $x$ (s)", fontsize=8, labelpad=1)
    right.set_ylabel("$P(X>x)$", fontsize=8, labelpad=1)
    right.legend(handles=[Line2D([], [], color=c, lw=1.5, label=n) for c, n in ((S, "S"), (A, "A"), (G, "G"))], frameon=False,
                 loc="lower left", ncol=3, fontsize=7.5, handlelength=0.6, handletextpad=0.25, columnspacing=0.5, borderaxespad=0.1)
    few_legend(right, "upper right")
    write(fig, "dyn_tails_glides", rect=(0, 0, 1, 0.97))


def dyn_tails_sc():
    """Left: mu_G - mu_A per ground class and on every flight, with 90% bands, one row per class: of the censored Hill exponent
    of all the searches (circles; search_km.json, tail_censored.py) and of the durations of the climbs (squares). Right: the
    Kaplan-Meier survival P(T>t), T the time to find, of S, A, G of the searches (top; log y, linear x, threshold u of the
    censored Hill exponent dashed) and the survival P(X>x) of the climbs (bottom; common threshold x_min dashed), every flight,
    with 90% bands. Each curve runs to its own last positive point; the x axis ends at the largest one. The part with fewer
    than 20 events above x is drawn dashed: n P(X>x) durations for the climbs (n from the JSON), n P(T>t) searches for the
    searches (the number at risk, "at risk" of km tail in the JSON)."""
    t = read("phases_refit")["tails"]
    km = read("search_km")
    fig = plt.figure(figsize=(6.4, 2.4))
    gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 0.2, 1.7], wspace=0.5, hspace=0.45)  # a spacer column between the two parts
    ax = fig.add_subplot(gs[:, :2])
    for phase, fmt, off, color in (("search", "o", -0.17, "#9a7fc0"), ("climb", "s", 0.17, "#2f4465")):
        for i, k in enumerate(KEYS):
            gap = km["tail censored"][k]["gaps"]["G - A"] if phase == "search" else t[k][phase]["gaps"]["G - A"]
            bar(ax, None, i + off, gap, False, color, ms=5, fmt=fmt)
    ax.axvline(0, color="k", lw=0.8)
    forest_axis(ax, False)
    ax.set_yticks(range(len(KEYS)), SHORT[:-1] + ("every flight",))
    ax.xaxis.set_major_locator(MaxNLocator(5, integer=True))
    ax.tick_params(labelsize=8)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel(r"$\mu_G-\mu_A$", fontsize=9)
    ax.spines[["top", "right", "left"]].set_visible(False)
    for row, (phase, title) in enumerate((("search", "searches (KM)"), ("climb", "climbs"))):
        sx = fig.add_subplot(gs[row, 3])
        d = t["all"][phase]
        if phase == "search":
            curves = {k: (np.array(km["km tail"]["grid"]), *(np.array(km["km tail"]["survival"][k][key]) for key in ("value", "band"))) for k in "SAG"}
            n = km["km"]["all"]["searches"]
        else:
            curves = {k: tuple(np.array(d["survival"][k][key]) for key in ("grid", "value", "band")) for k in "SAG"}
            n = d["runs"]
        xmax = 0
        for pop, k in (("solo", "S"), ("group_solo", "A"), ("group", "G")):
            g, v, b = curves[k]
            risk = np.array(km["km tail"]["at risk"][k]) if phase == "search" else None
            xmax = max(xmax, survival_curve(sx, g, v, b, n[k], COLORS[pop], 1.2, risk=risk))
        xline = km["tail censored"]["all"]["u"] if phase == "search" else d["xmin"]["common"]
        sx.axvline(xline, color=GREY, lw=1.2, ls=":")
        sx.text(xline, 1.0, r"$x_\mathrm{min}$", color=GREY, fontsize=8, ha="center", va="bottom",
                transform=sx.get_xaxis_transform())
        sx.set(yscale="log", xlim=(0, 2250) if phase == "search" else (0, xmax * 1.03),  # searches: the KM tail up to 2250 s
               ylim=(3e-4, 1.5) if phase == "search" else (1e-6, 1.5))
        sx.text(0.97, 0.93, title, transform=sx.transAxes, fontsize=8, ha="right", va="top")
        sx.set_yticks([1, 1e-1, 1e-2, 1e-3] if phase == "search" else [1, 1e-3, 1e-6])
        sx.minorticks_off()
        sx.xaxis.set_major_locator(MaxNLocator(3, integer=True))
        sx.tick_params(labelsize=8, pad=1)
        sx.grid(color="#eeeeee", lw=0.5)
        sx.set_ylabel("$P(T>t)$" if phase == "search" else "$P(X>x)$", fontsize=8, labelpad=1)
        if row == 1:
            sx.set_xlabel("duration $x$ (s)", fontsize=8, labelpad=1)
        else:
            sx.legend(handles=[Line2D([], [], color=c, lw=1.5, label=n) for c, n in ((S, "S"), (A, "A"), (G, "G"))], frameon=False,
                      loc="lower left", ncol=3, fontsize=7.5, handlelength=0.6, handletextpad=0.25, columnspacing=0.5, borderaxespad=0.1)
        if row == 1:
            few_legend(sx, "upper right", (1, -0.4))
    handles = [Line2D([], [], marker=m, ls="", color=c, ms=5, label=n) for m, c, n in (("o", "#9a7fc0", "searches (given-up used while they last)"), ("s", "#2f4465", "climbs"))]
    ax.legend(handles=handles, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=2, fontsize=8, handletextpad=0.2, columnspacing=1.0)
    write(fig, "dyn_tails_sc", rect=(0, 0, 1, 0.93))


# ---- geometry ----

def route():
    """route_gaps.pdf, three panels: G - A on the route (band, solid) and in time (dotted) against the time lag, on every flight,
    in the plains and in the high mountains."""
    r = read("route_alpha")
    fig, axes = plt.subplots(1, 3, figsize=(WIDE, 2.0), sharey=True)
    gs = []
    for ax, k in zip(axes, ("all", "plains", "high mountains")):
        c, t, color = r["route"][k]["gaps"]["G - A"], r["route"][k]["time"]["G - A"], GAP["G - A"]
        x, keep = np.array(c["x"]), np.array(c["solid"])
        ax.fill_between(x, *c["band"], color=color, alpha=0.15, lw=0)
        ax.plot(x[keep], np.array(c["value"])[keep], color=color, lw=1.5, label="route")
        tx, tk = np.array(t["x"]), np.array(t["solid"])
        ax.plot(tx[tk], np.array(t["value"])[tk], color=color, lw=1.2, ls=":", label="time")
        contrasts.finish(ax, "every flight" if k == "all" else k)
        gs.append({"lags": c["lags"] if "lags" in c else c["x"], "value": c["value"], "band": c["band"]})
        gs.append({"lags": t["lags"] if "lags" in t else t["x"], "value": t["value"], "band": t["band"]})
    axes[0].set(ylabel=r"$\Delta\alpha^*$, G − A", ylim=contrasts.ylim(gs))
    axes[0].legend(frameon=False, loc="upper left", fontsize=7)
    write(fig, "route")


def route_classes():
    """route_classes.pdf, the story figure route_gaps.pdf at the size of the slide: per ground class and on every flight, G - A and
    A - S on the route (band on every lag; line solid on the lags with enough flights, dashed beyond) and in time (dotted)."""
    r = read("route_alpha")["route"]
    fig, axes = plt.subplots(1, 5, figsize=(6.4, 2.0), sharey=True)
    gs = []
    for ax, k in zip(axes, (*CLASSES, "all")):
        for name in ("G - A", "A - S"):
            c, t, color = r[k]["gaps"][name], r[k]["time"][name], GAP[name]
            x, v, keep = np.array(c["x"]), np.array(c["value"]), np.array(c["solid"])
            ax.fill_between(x, *c["band"], color=color, alpha=0.15, lw=0)
            tail = np.arange(len(keep)) >= keep.sum() - 1  # the solid part is a prefix
            ax.plot(x[keep], v[keep], color=color, lw=1.4)
            ax.plot(x[tail], v[tail], color=color, lw=1.4, ls="--")
            tx, tk = np.array(t["x"]), np.array(t["solid"])
            ax.plot(tx[tk], np.array(t["value"])[tk], color=color, lw=1.2, ls=":")
            gs += [{"lags": c["x"], "value": c["value"], "band": c["band"]}, {"lags": t["x"], "value": t["value"], "band": t["band"]}]
        ax.axhline(0, color="k", lw=0.6)
        ax.set(xscale="log")
        ax.set_title("every flight" if k == "all" else k, fontsize=9)
        ax.set_xlabel(r"$\tau$ (s)", fontsize=9)
        ax.tick_params(labelsize=8.5)
        ax.grid(color="#eeeeee", lw=0.5)
    axes[0].set_ylabel(r"$\Delta\alpha^*$", fontsize=9)
    axes[0].set_ylim(-0.03, 0.06)  # the range of the solid parts; the dashed tails run beyond it
    handles = [Line2D([], [], color=GAP[n], lw=1.4, ls=ls, label=f"{contrasts.title(n)} {w}")
               for n in ("G - A", "A - S") for w, ls in (("route", "-"), ("time", ":"))]
    fig.legend(handles=handles, frameon=False, loc="upper center", ncol=4, fontsize=9, columnspacing=1.6, handlelength=1.8)
    write(fig, "route_classes", rect=[0, 0, 1, 0.88], pad=0.5)


def route_check_slide():
    """route_check.pdf at the size of the slide: alpha* of S, A, G of the path with the circles kept, read in length (solid; dashed
    on the lags with few flights) and in time (dotted), and, on the right, their difference; 90% bands."""
    c = read("route_alpha")["check"]
    fig, (left, right) = plt.subplots(1, 2, figsize=(5.6, 1.75))
    ylim = [0.0, 0.0]
    for pop in raw.POPS:
        v, lags, keep = c[pop], np.array(c[pop]["lags"]), np.array(c[pop]["solid"])
        tail = np.arange(len(keep)) >= keep.sum() - 1  # the solid part is a prefix
        for ax, g in ((left, v["arc"]), (right, v["minus"])):
            value = np.array(g["value"])
            ax.fill_between(lags, *g["band"], color=COLORS[pop], alpha=0.15, lw=0)
            ax.plot(lags[keep], value[keep], color=COLORS[pop], lw=1.4)
            ax.plot(lags[tail], value[tail], color=COLORS[pop], lw=1.4, ls="--")
        left.plot(lags[keep], np.array(v["time"]["value"])[keep], color=COLORS[pop], lw=1.2, ls=":")
        band = np.array(v["minus"]["band"])[:, keep]
        ylim = [min(ylim[0], band[0].min()), max(ylim[1], band[1].max())]
    right.axhline(0, color="k", lw=0.6)
    right.set_ylim(1.1 * ylim[0], 1.1 * ylim[1])
    left.set(ylim=cohorts.YLIM)
    left.set_ylabel(r"$\alpha^*$ of the path", fontsize=9)
    right.set_ylabel(r"$\alpha^*_\ell-\alpha^*_\tau$", fontsize=9)
    for ax in (left, right):
        ax.set(xscale="log")
        ax.set_xlabel(r"$\tau$ (s)", fontsize=9)
        ax.tick_params(labelsize=8.5)
        ax.grid(color="#eeeeee", lw=0.5)
    handles = [Line2D([], [], color=COLORS[pop], lw=1.4, label=short) for pop, short in raw.POPS.items()]
    handles += [Line2D([], [], color="k", lw=1.2, ls=ls, label=name) for ls, name in (("-", "in length"), (":", "in time"))]
    fig.legend(handles=handles, frameon=False, loc="upper center", ncol=5, fontsize=9, columnspacing=1.6, handlelength=1.8)
    write(fig, "route_check", rect=[0, 0, 1, 0.9], pad=0.5)


def route_pops():
    """route_pops.pdf: alpha* of S, A, G on every flight, on the route (solid on the lags with enough flights, dashed beyond; 90% band)
    and in time (dotted), against tau (route lags mapped as tau = l / v, v the median route speed of S); on the right, route - time
    per population. The difference is that of the two plotted values at the time lags of the route (log interpolation, only inside
    the time range): the draws of the two curves are not stored, so it has no band."""
    r = read("route_alpha")
    fig, (left, right) = plt.subplots(1, 2, figsize=(5.6, 1.9))
    for pop in raw.POPS:
        c, t = r["route"]["all"]["alpha"][pop], r["check"][pop]["time"]
        x, v, keep = np.array(c["x"]), np.array(c["value"]), np.array(c["solid"])
        tail = np.arange(len(keep)) >= keep.sum() - 1  # the solid part is a prefix
        tl, tv, tk = np.array(r["check"][pop]["lags"]), np.array(t["value"]), np.array(r["check"][pop]["solid"])
        left.fill_between(x, *c["band"], color=COLORS[pop], alpha=0.15, lw=0)
        left.plot(x[keep], v[keep], color=COLORS[pop], lw=1.4)
        left.plot(x[tail], v[tail], color=COLORS[pop], lw=1.4, ls="--")
        left.plot(tl[tk], tv[tk], color=COLORS[pop], lw=1.2, ls=":")
        inside = keep & (x >= tl[0]) & (x <= tl[-1])
        right.plot(x[inside], v[inside] - np.interp(np.log(x[inside]), np.log(tl), tv), color=COLORS[pop], lw=1.4)
    right.axhline(0, color="k", lw=0.6)
    left.set(ylim=cohorts.YLIM)
    left.set_ylabel(r"$\alpha^*$", fontsize=9)
    right.set_ylabel(r"$\alpha^*_\mathrm{route}-\alpha^*_\mathrm{time}$", fontsize=9)
    for ax in (left, right):
        ax.set(xscale="log")
        ax.set_xlabel(r"$\tau$ (s)", fontsize=9)
        ax.tick_params(labelsize=8.5)
        ax.grid(color="#eeeeee", lw=0.5)
    handles = [Line2D([], [], color=COLORS[pop], lw=1.4, label=short) for pop, short in raw.POPS.items()]
    handles += [Line2D([], [], color="k", lw=1.2, ls=ls, label=name) for ls, name in (("-", "route"), (":", "time"))]
    fig.legend(handles=handles, frameon=False, loc="upper center", ncol=5, fontsize=9, columnspacing=1.6, handlelength=1.8)
    write(fig, "route_pops", rect=[0, 0, 1, 0.9], pad=0.5)


# ---- technical frames ----

def forest():
    """G - A of the chained alpha* at 1000 s: every flight and one weight per flight, the four ground classes, and the flights
    above two thresholds of the opening (the chains of perlag_min0.1 and perlag_min0.5)."""
    ground, perflight = read("contrasts_ground"), read("contrasts_all_perflight")["all"]
    groups = [
        [("every flight", at(read("perlag")["chained gaps"]["G - A"]), INK),
         ("one weight per flight", at(perflight["gaps"]["G - A"]), INK)],
        [(c, at(ground[c]["gaps"]["G - A"]), CLASS_COLORS[c]) for c in CLASSES],
        [(f"opening > {x}", at(read(f"perlag_min{x}")["chained gaps"]["G - A"]), OPENING[i + 1]) for i, x in enumerate((0.1, 0.5))],
    ]
    fig, ax = plt.subplots(figsize=(5.0, 2.7))
    y, ticks, names = 0, [], []
    for rows in groups:
        for name, q, color in rows:
            bar(ax, None, y, q, False, color, ms=6)
            ticks.append(y)
            names.append(name)
            y += 1
        y += 0.6
    ax.axvline(0, color="k", lw=0.6)
    ax.set(yticks=ticks, yticklabels=names, ylim=(y - 0.6, -0.7), xlabel=r"$\Delta\alpha^*(1000\ \mathrm{s})$, G − A")
    ax.grid(axis="x", color="#eeeeee", lw=0.5)
    write(fig, "forest")


ROWS = (("all", "every flight"), ("plains", "plains"), ("hills", "hills"), ("low mountains", "low mtn"),
        ("high mountains", "high mtn"))
TAILS = (("transition", "glides"), ("search", "searches\n(any outcome)"), ("climb", "climbs"))


def tails():
    """phases_tails.pdf, the Hill exponent mu of the durations of S, A, G above the common threshold of each phase, per ground class
    and on every flight, with 90% bands (a smaller mu is a heavier tail)."""
    t = read("phases_refit")["tails"]
    fig, axes = plt.subplots(1, 3, figsize=(WIDE, 3.0), sharey=True)
    ys = [k + 0.5 * (k > 0) for k in range(len(ROWS))]
    for ax, (phase, title) in zip(axes, TAILS):
        for k, (c, _) in enumerate(ROWS):
            for off, pop, short in ((-0.22, "solo", "S"), (0, "group_solo", "A"), (0.22, "group", "G")):
                bar(ax, None, ys[k] + off, t[c][phase]["mu"][short], False, COLORS[pop], ms=4.5, label=short if k == 0 else None)
        ax.set(title=title, xlabel=r"$\mu$")
        ax.locator_params(axis="x", nbins=4)
        ax.grid(axis="x", color="#eeeeee", lw=0.5)
    axes[0].set(yticks=ys, yticklabels=[n for _, n in ROWS], ylim=(ys[-1] + 0.5, -0.6), xlim=(2.6, 9.6))
    fig.legend(*axes[0].get_legend_handles_labels(), frameon=False, loc="upper right", ncol=3, bbox_to_anchor=(1.0, 1.0))
    write(fig, "tails", rect=(0, 0, 1, 0.93))


# ---- tables ----

def table_columns(text, keep):
    """The table `text` of the statistic and the gap in front, and of the class columns `keep` (counted from the third column)."""
    out = []
    for ln in text.splitlines():
        if ln.startswith(r"\begin{tabular}"):
            ln = r"\begin{tabular}{@{}l l " + "l " * len(keep) + "@{}}"
        elif ln.rstrip().endswith(r"\\") and " & " in ln:
            cells = ln.rstrip()[:-2].split(" & ")
            ln = " & ".join(cells[:2] + [cells[2 + k] for k in keep]) + r" \\"
        out.append(ln)
    return "\n".join(out) + "\n"


def signed(x, digits=3):
    return ("$+$" if x >= 0 else "$-$") + f"{abs(x):.{digits}f}"


def simpson():
    """simpson_table.tex and simpson_macros.tex: alpha*(TAU) of S and A per ground class, the shares of the pieces at TAU (the
    unit of the MSD), and the split of the difference of the weighted means into a within-class and a composition term."""
    r = read("contrasts_ground")
    pops = {"S": "solo", "A": "group_solo"}
    w = {P: np.array([r[c]["alpha"][pop]["pieces"][r[c]["alpha"][pop]["lags"].index(TAU)] for c in CLASSES], dtype=float)
         for P, pop in pops.items()}
    w = {P: n / n.sum() for P, n in w.items()}
    x = {P: np.array([at(r[c]["alpha"][pop])["value"] for c in CLASSES]) for P, pop in pops.items()}
    mean = {P: float(w[P] @ x[P]) for P in pops}
    within, comp = float(w["A"] @ (x["A"] - x["S"])), float((w["A"] - w["S"]) @ x["S"])
    assert abs(within + comp - (mean["A"] - mean["S"])) < 1e-12
    every = {P: at(r["all"]["alpha"][pop])["value"] for P, pop in pops.items()}
    pct = {P: percent(w[P]) for P in pops}
    rows = [f"{c} & {pct['S'][i]}\\,\\% & {pct['A'][i]}\\,\\% & {x['S'][i]:.4f} & {x['A'][i]:.4f} & {signed(x['A'][i] - x['S'][i], 4)} \\\\"
            for i, c in enumerate(CLASSES)]
    lines = [r"\begin{tabular}{@{}l r r r r r@{}}", r"\toprule",
             r"ground class & share $\Sl$ & share $\A$ & $\alpha^*_\Sl$ & $\alpha^*_\A$ & $\A-\Sl$ \\", r"\midrule", *rows, r"\midrule",
             f"weighted mean & & & {mean['S']:.4f} & {mean['A']:.4f} & {signed(mean['A'] - mean['S'], 4)} \\\\",
             f"every flight (measured) & & & {every['S']:.4f} & {every['A']:.4f} & {signed(every['A'] - every['S'], 4)} \\\\",
             r"\bottomrule", r"\end{tabular}"]
    DECK.joinpath("tikz").mkdir(exist_ok=True)
    (DECK / "tikz/simpson_table.tex").write_text("\n".join(lines) + "\n")
    mountain = {P: int(pct[P][2:].sum()) for P in pops}
    macros = {"SimpWithin": signed(within, 4), "SimpComp": signed(comp, 4), "SimpSum": signed(within + comp, 4),
              "SimpMeasured": signed(every["A"] - every["S"], 4), "SimpMountainS": str(mountain["S"]), "SimpMountainA": str(mountain["A"])}
    (DECK / "tikz/simpson_macros.tex").write_text("".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in macros.items()))


def memory_figure(name, curve):
    """legs.json: C(tau) = <cos of the heading change over the lag tau> of S, A and G in each ground class (curve(class, pop) gives
    value, band and tau_c), the level C = 1/e as a dotted line, tau_c of A as a dot on its curve, and the lag of the peak of the
    absolute G - A of the class (legs.json, "peak test") as a grey dashed line."""
    legs = read("legs")
    fig, axes = class_row(WIDE, 2.0, r"$C(\tau)$")
    lags = legs["lags_s"]
    for ax, c in zip(axes, CLASSES):
        for pop, color in (("solo", S), ("group_solo", A), ("group", G)):
            q = curve(legs, c, pop)
            ax.fill_between(lags, *q["band"], color=color, alpha=0.15, lw=0)
            ax.plot(lags, q["value"], color=color, lw=1.5)
        ax.axhline(np.exp(-1), color=INK, lw=0.8, ls=":")
        tc = curve(legs, c, "group_solo")["tau_c"]["value"]
        if tc is not None and np.isfinite(tc):
            ax.plot(tc, np.exp(-1), "o", ms=5, mfc="white", mec=A, mew=1.6, zorder=5)
        ax.axvline(legs["peak test"][c]["peak_lag_s"], color=GREY, lw=0.9, ls="--")
        ax.set(xscale="log", xlim=(lags[0], lags[-1]), xlabel=r"$\tau$ (s)", ylim=(0, 1))
        ax.set_title(c, fontsize=9)
        ax.grid(color="#eeeeee", lw=0.5)
    axes[0].text(12, np.exp(-1) + 0.02, "1/e", fontsize=7, va="bottom")
    handles = [Line2D([], [], color=color, lw=1.5) for color in (S, A, G)] + [
        Line2D([], [], ls="", marker="o", ms=5, mfc="white", mec=A, mew=1.6), Line2D([], [], color=GREY, lw=0.9, ls="--")]
    fig.legend(handles=handles, labels=["S", "A", "G", r"$\tau_c$ of A", r"peak of G $-$ A"], frameon=False, loc="upper center", ncol=5,
               bbox_to_anchor=(0.5, 1.0))
    write(fig, name, rect=(0, 0, 1, 0.9))


def route_memory():
    memory_figure("route_memory", lambda legs, c, pop: legs["strata"][c][pop]["time"])


def chained_line(ax, lags, value, band, color, lw=1.5, alpha=0.15):
    """The band, then the curve solid up to cohorts.SELECTION_FREE and dashed beyond (from the last solid point)."""
    lags, value = np.asarray(lags, float), np.asarray(value, float)
    ax.fill_between(lags, *np.array(band, float), color=color, alpha=alpha, lw=0)
    keep = lags <= cohorts.SELECTION_FREE
    tail = np.arange(len(lags)) >= keep.sum() - 1
    ax.plot(lags[keep], value[keep], color=color, lw=lw)
    ax.plot(lags[tail], value[tail], color=color, lw=lw, ls="--")


def gap_axis(ax, g):
    """A right y axis on `ax` with C*_G - C*_A (paired band, thinner line, lighter band) and its zero as a thin dotted line, all in
    the colour of G - A; returns the axis."""
    color = GAP["G - A"]
    right = ax.twinx()
    chained_line(right, g["lags"], g["value"], g["band"], color, lw=1.5, alpha=0.14)
    right.axhline(0, color=color, lw=0.8, ls=":")
    right.tick_params(axis="y", colors=color)
    return right


def gap_limits(gaps):
    """The limits of the right axis that hold the solid part of the curves `gaps` and their bands, padded."""
    v = np.concatenate([np.array(x, float)[np.array(g["lags"]) <= cohorts.SELECTION_FREE] for g in gaps for x in (g["value"], *g["band"])])
    pad = 0.05 * (v.max() - v.min())
    return v.min() - pad, v.max() + pad


def route_memory_chained():
    """legs.json, "chained": one row of panels, one per ground class. Left axis: C*(tau) of S, A, G (chain over every piece that
    reaches the lag; band: 90% over flights; solid up to cohorts.SELECTION_FREE, dashed beyond), the level 1/e, tau_c of A, and the
    lag of the peak of the absolute G - A of the class (grey dashed). Right axis, the same limits in every panel: C*_G - C*_A with
    its paired band."""
    legs = read("legs")
    chained, peaks = legs["chained"]["strata"], legs["peak test chained"]
    fig, axes = plt.subplots(1, len(CLASSES), figsize=(WIDE + 0.5, 2.1), sharex=True, sharey=True)
    x1 = max(chained[c][p]["lags"][-1] for c in CLASSES for p in ("solo", "group_solo", "group"))
    limits = gap_limits([chained[c]["G - A"] for c in CLASSES])
    for i, (ax, c) in enumerate(zip(axes, CLASSES)):
        for pop, color in (("solo", S), ("group_solo", A), ("group", G)):
            q = chained[c][pop]
            chained_line(ax, q["lags"], q["value"], q["band"], color)
        ax.axhline(np.exp(-1), color=INK, lw=0.8, ls=":")
        tc = chained[c]["group_solo"]["tau_c"]["value"]
        if tc is not None and np.isfinite(tc):
            ax.plot(tc, np.exp(-1), "o", ms=2.8, mfc="white", mec=A, mew=1.0, zorder=5)
        ax.axvline(peaks[c]["peak_lag_s"], color=GREY, lw=0.9, ls="--")
        ax.grid(color="#eeeeee", lw=0.5)
        ax.set_title(c, fontsize=10)
        ax.set(xscale="log", xlim=(10, x1), ylim=(0, 1))
        ax.set_xlabel(r"$\tau$ (s)", fontsize=10)
        ax.tick_params(labelsize=9)
        right = gap_axis(ax, chained[c]["G - A"])
        right.set_ylim(*limits)
        right.tick_params(axis="y", labelsize=9)
        if i < len(CLASSES) - 1:
            right.tick_params(axis="y", right=False, labelright=False)
        else:
            right.set_ylabel(r"$C^*_G-C^*_A$", color=GAP["G - A"], fontsize=10)
    axes[0].set_xticks([10, 100, 1000, 10000], ["10", "$10^2$", "$10^3$", "$10^4$"], fontsize=9)
    axes[0].minorticks_off()
    axes[0].set_ylabel(r"$C^*(\tau)$", fontsize=10)
    axes[0].text(12, np.exp(-1) + 0.02, "1/e", fontsize=9, va="bottom")
    handles = [Line2D([], [], color=color, lw=1.5) for color in (S, A, G)] + [
        Line2D([], [], color=GAP["G - A"], lw=1.5), Line2D([], [], ls="", marker="o", ms=2.8, mfc="white", mec=A, mew=1.0),
        Line2D([], [], color=GREY, lw=0.9, ls="--")]
    fig.legend(handles=handles, labels=["S", "A", "G", r"$C^*_G-C^*_A$", r"$\tau_c$ of A", r"peak of G $-$ A"], frameon=False,
               loc="upper center", ncol=6, fontsize=9.5, bbox_to_anchor=(0.5, 1.0))
    write(fig, "route_memory2", rect=(0, 0, 1, 0.9))


def zero_crossing(lags, value):
    """The lag where `value` first crosses zero from below, interpolated linearly in log lag between the two lags around the crossing."""
    lags, v = np.array(lags, float), np.array(value, float)
    i = next(i for i in range(1, len(v)) if v[i - 1] < 0 <= v[i])
    return float(np.exp(np.log(lags[i - 1]) + (0 - v[i - 1]) / (v[i] - v[i - 1]) * (np.log(lags[i]) - np.log(lags[i - 1]))))


def why_peak():
    """High mountains, two panels on a log lag axis up to the end of the solid part: C* of A and G of the route (top, 1/e dotted,
    tau_c of A as a hollow dot) and the absolute G - A of alpha* of the real path (bottom); the top panel has a right axis with C*_G - C*_A. Three zones on both panels: both
    remember (up to the lag where alpha*_G - alpha*_A of the bottom panel first crosses zero from below, interpolated linearly in
    log tau between the two lags around the crossing), A forgets and G holds (up to tau_c of A), both forget; the peak lag of G - A
    is the grey dashed line. Each panel has a legend of its own curves."""
    legs, c = read("legs"), "high mountains"
    chained, peak = legs["chained"]["strata"][c], legs["peak test chained"][c]
    gap_c = chained["G - A"]
    tc = chained["group_solo"]["tau_c"]["value"]
    x1 = cohorts.SELECTION_FREE
    fig, (top, bot) = plt.subplots(2, 1, figsize=(5.8, 2.9), sharex=True)
    for pop, color in (("group_solo", A), ("group", G)):
        q = chained[pop]
        chained_line(top, q["lags"], q["value"], q["band"], color)
    top.axhline(np.exp(-1), color=INK, lw=0.8, ls=":")
    top.plot(tc, np.exp(-1), "o", ms=5, mfc="white", mec=A, mew=1.4, zorder=5)
    top.text(11, np.exp(-1) + 0.02, "1/e", fontsize=8, va="bottom")
    right = gap_axis(top, gap_c)
    right.set_ylim(*gap_limits([gap_c]))
    right.set_ylabel(r"$C^*_G-C^*_A$", color=GAP["G - A"])
    g = read("contrasts_ground")[c]["gaps"]["G - A"]
    lag1 = zero_crossing(g["lags"], g["value"])
    contrasts.curve(bot, g, GAP["G - A"])
    bot.axhline(0, color=INK, lw=0.6)
    top.set(xscale="log", xlim=(10, x1), ylim=(0, 1), ylabel=r"route memory $C^*$")
    top.set_yticks([0, 0.5, 1])
    bot.set(xlabel=r"$\tau$ (s)", ylabel=r"$\alpha^*_G-\alpha^*_A$")
    k = cohorts.solid(g["lags"])
    v = np.concatenate([np.array(x, float)[k] for x in (g["value"], *g["band"])])
    pad = 0.08 * (v.max() - v.min())
    bot.set_ylim(v.min() - pad, v.max() + pad)
    zones = ((10, lag1, "both remember"), (lag1, tc, "A forgets, G holds"), (tc, x1, "both forget"))
    for i, (a, b, text) in enumerate(zones):
        for ax in (top, bot):
            ax.axvspan(a, b, color="#f0f0f0" if i % 2 == 0 else "white", lw=0, zorder=0)
            ax.axvline(peak["peak_lag_s"], color=GREY, lw=0.9, ls="--")
        top.text(np.sqrt(a * b), 1.03, text, fontsize=8.5, ha="center", va="bottom", transform=top.get_xaxis_transform())
    for ax in (top, bot):
        ax.grid(color="#eeeeee", lw=0.5, zorder=1)
    top.legend([Line2D([], [], color=A, lw=1.5), Line2D([], [], color=G, lw=1.5), Line2D([], [], color=GAP["G - A"], lw=1.5)],
               ["A", "G", r"$C^*_G-C^*_A$"], frameon=False, loc="lower left", ncol=3, fontsize=8, handlelength=1.5,
               borderaxespad=0.1, borderpad=0.1)
    bot.legend([Line2D([], [], color=GAP["G - A"], lw=1.5)], [r"$\alpha^*_G-\alpha^*_A$"], frameon=False, loc="upper left", fontsize=8,
               handlelength=1.5)
    write(fig, "why_peak")


def path_memory():
    memory_figure("path_memory", lambda legs, c, pop: legs["path"][c][pop])


def legs_tables():
    """legs_macros.tex and legs_table.tex: the memory of the route (lag and length at which C falls to 1/e, lag at 0.5) from legs.json;
    the table holds the two mountain classes, the plains and the hills (C does not fall to 1/e for any population) are asserted."""
    legs = read("legs")
    st, tmax = legs["strata"], int(legs["lags_s"][-1])
    first = lambda q: q["value"] is not None and np.isfinite(q["value"])
    sec = lambda q: f"{q['value']:.0f}" if first(q) else f"$>${tmax}"
    sband = lambda q: f"[{q['band'][0]:.0f}, {q['band'][1]:.0f}]"
    km = lambda q: f"{q['value'] / 1000:.1f}"
    kband = lambda q: f"[{q['band'][0] / 1000:.1f}, {q['band'][1] / 1000:.1f}]"
    macros = {"LegTmax": str(tmax)}
    for cls, tag in (("low mountains", "Low"), ("high mountains", "High")):
        for pop, P in (("solo", "S"), ("group_solo", "A"), ("group", "G")):
            time, space = st[cls][pop]["time"]["tau_c"], st[cls][pop]["space"]["s_c"]
            macros[f"LegTc{tag}{P}"] = f"{time['value']:.0f}"
            macros[f"LegTc{tag}{P}Band"] = sband(time)
            macros[f"LegSc{tag}{P}"] = km(space)
    for pop, P in (("solo", "S"), ("group_solo", "A"), ("group", "G")):
        macros[f"LegCPlains{P}"] = f"{st['plains'][pop]['time']['value'][-1]:.2f}"
    lags, peaks = np.array(legs["lags_s"], float), read("phases_refit")["peaks"]
    for c, tag in (("low mountains", "Low"), ("high mountains", "High")):  # where the memory of G exceeds that of A most
        gap = np.array(st[c]["group"]["time"]["value"], float) - np.array(st[c]["group_solo"]["time"]["value"], float)
        macros[f"LegGapLag{tag}"] = f"{lags[np.nanargmax(gap)]:.0f}"
        macros[f"LegGap{tag}"] = f"{np.nanmax(gap):.2f}"
        macros[f"LegPeakLag{tag}"] = str(peaks[c]["G - A"]["lag"])
    peak = read("legs")["peak test"]
    for c, tag in (("low mountains", "Low"), ("high mountains", "High"), ("all", "All")):
        macros[f"LegThalf{tag}A"] = f"{peak[c]['route']['A']['tau_half']['value']:.0f}"
        macros[f"LegRatio{tag}"] = f"{peak[c]['route']['ratio_peak_tau_half_A']:.1f}"
        macros[f"LegRatioTc{tag}"] = f"{peak[c]['route']['ratio_peak_tau_c_A']:.1f}"
    for c, tag in (("plains", "Plains"), ("hills", "Hills"), ("low mountains", "Low"), ("high mountains", "High")):  # chained C*
        q = legs["chained"]["strata"][c]["group_solo"]
        tc = q["tau_c"]["value"]
        macros[f"LegChainTc{tag}A"] = f"{tc:.0f}" if tc is not None and np.isfinite(tc) else f"$>${q['lags'][-1]:.0f}"
    pc = legs["peak test chained"]
    for c, tag in (("low mountains", "Low"), ("high mountains", "High")):
        macros[f"LegChainRatio{tag}"] = f"{pc[c]['peak_lag_s'] / pc[c]['A']['tau_c']['value']:.2f}"
    high = pc["high mountains"]  # LegChainGapLagHigh: the lag where C*_G - C*_A peaks, the lag of the peak of alpha*_G - alpha*_A
    assert high["gap_C"]["max_lag_s"] == high["peak_lag_s"]
    macros["LegChainGapLagHigh"] = f"{high['gap_C']['max_lag_s']:.0f}"
    macros["LegThalfHillsA"] = f"{peak['hills']['route']['A']['tau_half']['value']:.0f}"
    for c, tag in (("low mountains", "Low"), ("high mountains", "High")):
        for P in ("A", "G"):
            macros[f"LegPathTc{tag}{P}"] = f"{peak[c]['path'][P]['tau_c']['value']:.0f}"
    (DECK / "tikz/legs_macros.tex").write_text("".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in macros.items()))
    rows = []
    for cls in CLASSES[:2]:  # the plains and the hills are one line of the frame: C never falls to 1/e within the lags
        assert not any(first(st[cls][pop]["time"]["tau_c"]) for pop in ("solo", "group_solo", "group")), cls
    for cls in CLASSES[2:]:
        for i, (pop, P) in enumerate((("solo", r"$\Sl$"), ("group_solo", r"$\A$"), ("group", r"$\G$"))):
            time, space = st[cls][pop]["time"], st[cls][pop]["space"]
            c, s = time["tau_c"], space["s_c"]
            rows.append(f"{cls if i == 0 else ''} & {P} & {sec(time['tau_half'])} & "
                        + (f"{sec(c)} {sband(c)}" if first(c) else f"$>${tmax}") + " & "
                        + (f"{km(s)} {kband(s)}" if first(s) else f"$>${legs['cohort']['s_max_m'] / 1000:.0f}") + r" \\")
        rows.append(r"\addlinespace[2pt]")
    lines = [r"\begin{tabular}{@{}l l r r r@{}}", r"\toprule",
             r"ground class & & $\tau_{1/2}$ (s) & $\tau_c$ [90\,\% band] (s) & $s_c$ [90\,\% band] (km) \\", r"\midrule", *rows[:-1],
             r"\bottomrule", r"\end{tabular}"]
    (DECK / "tikz/legs_table.tex").write_text("\n".join(lines) + "\n")


PHASES_ALL_SPLIT = 8  # rows of the full table of the phase statistics in its first frame: time, glide, search, climb rate


def deck_tables():
    """The tables of the deck, from the generated tables of the report: the control events named as such, the three steps of every
    flight (real, control, real - control), the block of every flight of the full table of the phase statistics (without G - S, the sum of the other two gaps) in two
    frames of rows, and the main table of the phase statistics in two halves."""
    DECK.joinpath("tikz").mkdir(exist_ok=True)
    mock = (TABLES / "prepost_mock_table.tex").read_text()
    (DECK / "tikz/prepost_control_table.tex").write_text(re.sub(r"\{mock\}", "{control}", mock).replace("real $-$ mock", "real $-$ control"))
    row = next(ln for ln in mock.splitlines() if ln.startswith("all &")).rstrip()[:-2].split(" & ")
    lines = [r"\begin{tabular}{@{}l r r r@{}}", r"\toprule", r" & real & control & real $-$ control \\", r"\midrule"]
    for name, i in (("entry", 1), ("inside", 2), ("exit", 3)):
        lines.append(f"{name} & {row[i]} & {row[i + 3]} & {row[i + 6]} \\\\")
    (DECK / "tikz/prepost_control_all.tex").write_text("\n".join(lines + [r"\bottomrule", r"\end{tabular}"]) + "\n")
    lines = (TABLES / "phases_table.tex").read_text().splitlines()
    heads = [i for i, ln in enumerate(lines) if ln.startswith(r"\multicolumn")]
    last = max(i for i, ln in enumerate(lines) if "textit{all}" in ln)
    every = lines[:heads[0]] + lines[last:]
    top = [i for i, ln in enumerate(every) if ln.startswith(r"\multicolumn")][0] + 1  # the first row of statistics
    bottom = next(i for i, ln in enumerate(every) if ln.startswith(r"\bottomrule"))
    drop_gs = lambda ln: " & ".join(ln.rstrip()[:-2].split(" & ")[:6]) + r" \\"  # G - S = (G - A) + (A - S)
    head = [r"\begin{tabular}{@{}l r r r l l@{}}", *every[1:top - 1], every[top - 1].replace(r"\multicolumn{7}", r"\multicolumn{6}")]
    head = [drop_gs(ln) if ln.startswith("statistic") else ln for ln in head]
    rows = [drop_gs(ln) for ln in every[top:bottom]]
    for name, part in (("a", rows[:PHASES_ALL_SPLIT]), ("b", rows[PHASES_ALL_SPLIT:])):
        (DECK / f"tikz/phases_all_{name}.tex").write_text("\n".join(head + part + [r"\bottomrule", r"\end{tabular}"]) + "\n")
    main = (TABLES / "phases_main_table.tex").read_text()
    (DECK / "tikz/phases_main_a.tex").write_text(table_columns(main, (0, 1, 2)))
    (DECK / "tikz/phases_main_b.tex").write_text(table_columns(main, (3, 4)))


def km_tables():
    """tikz/km_macros.tex: the time to find a thermal (search_km.json, Kaplan-Meier with the failed searches censored),
    every flight: KM median of A and G and G - A, and the gaps of S_km(60 s), F_found(120 s), F_fail(120 s), with bands; and the
    gap mu_c,G - mu_c,A of the censored Hill exponent of the tail (tail_censored.py)."""
    k = read("search_km")["km"]["all"]
    mu = read("search_km")["tail censored"]["all"]["gaps"]["G - A"]
    s, g = k["summaries"], k["gaps"]["G - A"]
    sec = lambda q: f"{q['value']:.0f}"
    band = lambda q, d: f"[{signed(q['band'][0], d)}, {signed(q['band'][1], d)}]"
    m = {"KmMedA": sec(s["A"]["median time to find (s)"]), "KmMedG": sec(s["G"]["median time to find (s)"]),
         "KmMedGA": signed(g["median time to find (s)"]["value"], 0), "KmMedGABand": band(g["median time to find (s)"], 0),
         "KmSixtyGA": signed(g["S_km(60 s)"]["value"], 3), "KmSixtyGABand": band(g["S_km(60 s)"], 3),
         "KmFoundGA": signed(100 * g["F_found(120 s)"]["value"], 1), "KmFailGA": signed(100 * g["F_fail(120 s)"]["value"], 1),
         "KmMuGA": signed(mu["value"], 3), "KmMuGABand": band(mu, 3)}
    (DECK / "tikz/km_macros.tex").write_text("".join(f"\\newcommand{{\\{a}}}{{{b}}}\n" for a, b in m.items()))


if __name__ == "__main__":
    for f in (nested, chain_pool, cohort_msd, cohort_gaps, weights, persistent, gaps, ground_gaps, where, scale, cycle, opening, opening_kind, opening_ground, pilots,
              when_all, when_long, when_long_classes, control_steps, carry,
              dyn_time, dyn_phases, lambda: dyn_phases(True), dyn_joins, dyn_find, dyn_glide, dyn_perf, tails_survival, tails_hill, dyn_tails_glides, dyn_tails_sc, route,
              route_classes, route_check_slide, route_pops, route_memory, route_memory_chained, why_peak, path_memory, forest, tails):
        f()
    deck_tables()
    simpson()
    km_tables()
    legs_tables()
    for name, (w, h) in SIZES.items():
        print(f"slide_{name}: {w:.2f} x {h:.2f} in")
