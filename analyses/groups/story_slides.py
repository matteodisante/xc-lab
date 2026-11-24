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


def write(fig, name, rect=None):
    """tamsd.save, with room above the panels (`rect`) for a legend of the whole figure."""
    path = DIR / f"slide_{name}.pdf"
    if rect:
        fig.tight_layout(rect=rect)
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
    fmt = kw.pop("fmt", "o")
    if vertical:
        ax.errorbar(x, q["value"], yerr=err, fmt=fmt, ms=ms, color=color, elinewidth=1.6, capsize=3, **kw)
    else:
        ax.errorbar(q["value"], y, xerr=err, fmt=fmt, ms=ms, color=color, elinewidth=1.6, capsize=3, **kw)


def spread(ys, gap):
    """Label heights close to `ys`, pushed apart to at least `gap`, in the order of `ys`."""
    order = np.argsort(ys)
    out = np.array(ys, dtype=float)
    for a, b in zip(order, order[1:]):
        out[b] = max(out[b], out[a] + gap)
    return out


# ---- the method ----

def nested():
    """Left: alpha(tau) of the nested cohorts of S and the pool (every piece that reaches each lag), clipped at the top of the
    axis. Right: the pieces that reach tau, per population: every piece up to the shortest tau_max, then those of each cohort."""
    c = read("cohorts")
    fig, (left, right) = plt.subplots(1, 2, figsize=(6.0, 2.0), gridspec_kw={"width_ratios": [1.25, 1]})
    pooled = c["pool"]["solo"]
    lags, band = np.array(pooled["lags"]), np.array(pooled["band"])
    left.fill_between(lags, band[0], band[1], color=cohorts.POOL_COLOR, alpha=0.15, lw=0)
    left.plot(lags, pooled["value"], color=cohorts.POOL_COLOR, lw=1.6, label="pool")
    for k, (cohort, color) in enumerate(zip(c["alpha"]["solo"], cohorts.shades())):
        lags, band = np.array(cohort["lags"]), np.array(cohort["band"])
        left.fill_between(lags, band[0], band[1], color=color, alpha=0.15, lw=0)
        left.plot(lags, cohort["alpha"], color=color, lw=1.3, label=f"to {cohort['tau_max']:,} s" if k in (0, len(cohorts.COHORTS) - 1) else None)
    left.set(xscale="log", xlabel=r"$\tau$ (s)", ylabel=r"$\alpha(\tau)$ of S", ylim=(0.82, 1.1))
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
    fig, axes = plt.subplots(1, 3, figsize=(6.0, 2.0), sharey=True)
    for ax, (pop, short) in zip(axes, raw.POPS.items()):
        for k, (cohort, color) in enumerate(zip(c["alpha"][pop], cohorts.shades())):
            ends = k in (0, len(cohorts.COHORTS) - 1) and pop == "solo"
            ax.plot(cohort["lags"], cohort["alpha"], color=color, lw=1.8 if k == len(c["alpha"][pop]) - 1 else 0.9, label=f"cohort to {cohort['tau_max']:,} s" if ends else None)
        contrasts.curve(ax, s["pool"][pop], cohorts.POOL_COLOR, "pool" if pop == "solo" else None)
        contrasts.curve(ax, s["chained"][pop], COLORS[pop], r"chained $\alpha^*$" if pop == "solo" else None)
        ax.set(xscale="log", xlabel=r"$\tau$ (s)", title=short)
        ax.grid(color="#eeeeee", lw=0.5)
    axes[0].set(ylabel=r"$\alpha(\tau)$", ylim=(0.78, 1.05))
    axes[0].legend(frameon=False, loc="upper left", fontsize=6.5)
    write(fig, "chain_pool")


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
    """carry.pdf, bottom row, every flight: A(d) - S of the exponent over the long lags per bin of the distance d from the
    group time."""
    c = read("carry")
    fig, ax0 = plt.subplots(figsize=(3.6, 2.1))
    axes = [ax0]
    for ax, k in zip(axes, ("all",)):
        r = c["classes"][k]
        for i, b in enumerate(r["bins"]):
            bar(ax, i, None, b["diff"], True, carry_module.BIN_COLOR, ms=7)
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
    """Left: the share of the time in glide, search, climb and exit of A and G on every flight. Right: G - A of the three main
    shares per class and on every flight, with 90% bands."""
    ph = phase_data()
    fig, (left, right) = plt.subplots(1, 2, figsize=(WIDE, 2.0), gridspec_kw={"width_ratios": [1.5, 1]})
    for y, (pop, color) in enumerate((("group_solo", A), ("group", G))):
        x0 = 0
        for _, stat, c in PHASES:
            v = ph["all"]["statistics"][{"group_solo": "A", "group": "G"}[pop]][stat]["value"]
            left.barh(y, v, left=x0, height=0.62, color=c, edgecolor="white", lw=0.8)
            if v > 5:
                left.text(x0 + v / 2, y, f"{v:.1f}", ha="center", va="center", fontsize=8, color="white" if c in ("#2f4465", "#7f9bb8", "#9a7fc0") else INK)
            x0 += v
    left.set(xlim=(0, 100), ylim=(1.55, -0.55), yticks=[0, 1], yticklabels=["A", "G"], xlabel="share of the time (%)")
    for tick, color in zip(left.get_yticklabels(), (A, G)):
        tick.set(color=color, fontsize=11, fontweight="bold")
    left.tick_params(axis="y", length=0)
    left.spines[["top", "right", "left"]].set_visible(False)
    for j, (_, stat, c) in enumerate(PHASES[:3]):
        for i, k in enumerate(KEYS):
            bar(right, i + (j - 1) * 0.26, None, ph[k]["gaps"]["G - A"][stat], True, c, ms=4.5)
    class_axis(right, "G − A (pts)")
    fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in PHASES], labels=[n for n, _, _ in PHASES], frameon=False,
               loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.0), handlelength=1.2, columnspacing=1.2)
    write(fig, "dyn_time", rect=(0, 0, 1, 0.9))


def dyn_find():
    """G - A as a percentage of A, 100 (G - A) / A, of five statistics in two blocks (found more reliably: searches that end in
    a climb, searches per hour, time in search; stronger: climb rate, mean exit duration). One dot per ground class and one,
    larger and black, on every flight, with 90% bands: the band of G - A divided by the value of A, times 100."""
    ph = phase_data()
    rows = ((0, "Found more reliably?", None),
            (0.95, "searches ending in a climb", "searches that end in a climb (%)"),
            (2.4, "searches per hour", "searches per hour"),
            (3.85, "time in search", "time in search (%)"),
            (5.15, "Stronger?", None),
            (6.1, "climb rate", "climb rate (m/s)"),
            (7.55, "mean exit duration", "mean exit duration (s)"))
    offsets = np.linspace(-0.47, 0.47, len(KEYS))
    fig, ax = plt.subplots(figsize=(5.0, 2.8))
    for r, label, stat in rows:
        if stat is None:
            continue
        for k, color, off in zip(KEYS, KCOLORS, offsets):
            a = ph[k]["statistics"]["A"][stat]["value"]
            g = ph[k]["gaps"]["G - A"][stat]
            q = {"value": 100 * g["value"] / a, "band": [100 * g["band"][0] / a, 100 * g["band"][1] / a]}
            bar(ax, None, r + off, q, False, color, ms=6 if k == "all" else 3.5, zorder=3 if k == "all" else 2)
    ax.set_ylim(8.2, -0.4)
    ax.set_yticks([r for r, _, _ in rows], [label for _, label, _ in rows], fontsize=9.5)
    for tick, (_, _, stat) in zip(ax.get_yticklabels(), rows):
        if stat is None:
            tick.set(fontweight="bold", color=INK)
    ax.tick_params(axis="y", length=0)
    ax.axvline(0, color="k", lw=0.8)
    ax.grid(axis="x", color="#eeeeee", lw=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", labelsize=9.5)
    ax.set_xlabel("G − A (% of A)", fontsize=9.5)
    ax.spines[["top", "right", "left"]].set_visible(False)
    fig.legend(handles=[Line2D([], [], marker="o", ls="", color=c, ms=6 if k == "all" else 4.5)
                        for k, c in zip(KEYS, KCOLORS)], labels=SHORT[:-1] + ("every flight",), frameon=False,
               loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.0), fontsize=9, handletextpad=0.1, columnspacing=0.8)
    write(fig, "dyn_find", rect=(0, 0, 1, 0.92))


def dyn_glide():
    """G - A per class and on every flight of the glide speed and the height lost per km of glide, with 90% bands."""
    ph = phase_data()
    panels = (("glide speed (km/h)", "glide speed", "G − A (km/h)"),
              ("height lost per km of glide (m/km)", "height lost per km of glide", "G − A (m/km)"))
    fig, axes = plt.subplots(1, 2, figsize=(4.8, 2.0))
    for ax, (stat, title, ylabel) in zip(axes, panels):
        gap_panel(ax, ph, stat, title, ylabel)
    write(fig, "dyn_glide")


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


def dyn_tails_glides():
    """Glide durations on every flight, above the common threshold x_min. Left: the survival P(X>x) of S, A, G with its 90% band
    on log axes. Right: the ratio P_G(x)/P_A(x) of the values of the curves, up to the longest duration reached by A and G, with a
    line at 1; the JSON keeps the band of each curve and not the draws, so the ratio has no band."""
    d = read("phases_refit")["tails"]["all"]["transition"]
    xmin = d["xmin"]["common"]
    surv = {k: {key: np.array(v) for key, v in d["survival"][k].items()} for k in "SAG"}
    grid = surv["S"]["grid"]
    xmax = min(grid[s["value"] > 0].max() for s in surv.values())
    sel = (grid >= xmin) & (grid <= xmax)
    fig, (left, right) = plt.subplots(1, 2, figsize=(5.2, 2.3), gridspec_kw={"width_ratios": [1, 1]})
    for pop, k, c in (("solo", "S", S), ("group_solo", "A", A), ("group", "G", G)):
        s = surv[k]
        left.fill_between(grid[sel], s["band"][0][sel], s["band"][1][sel], color=c, alpha=0.3, lw=0)
        left.plot(grid[sel], s["value"][sel], color=c, lw=2, label=k)
    left.set(xscale="log", yscale="log", xlabel=r"duration $x$ (s), $x\geq x_\mathrm{min}$", ylabel="$P(X>x)$")
    left.set_xlim(xmin, xmax)
    left.minorticks_off()
    left.xaxis.set_major_locator(plt.FixedLocator([2000, 4000]))
    left.set_xticklabels(["2000", "4000"])
    left.grid(color="#eeeeee", lw=0.5)
    left.legend(frameon=False, loc="lower left", fontsize=10, handlelength=1.2)
    ratio = surv["G"]["value"][sel] / surv["A"]["value"][sel]
    right.plot(grid[sel], ratio, color=GAP["G - A"], lw=2)
    right.axhline(1, color="k", lw=0.8)
    right.set(xscale="log", xlabel=r"duration $x$ (s), $x\geq x_\mathrm{min}$", ylabel=r"$P_G/P_A$")
    right.set_xlim(xmin, xmax)
    right.minorticks_off()
    right.xaxis.set_major_locator(plt.FixedLocator([2000, 4000]))
    right.set_xticklabels(["2000", "4000"])
    right.yaxis.set_major_locator(MaxNLocator(4))
    right.grid(color="#eeeeee", lw=0.5)
    for ax in (left, right):
        ax.tick_params(labelsize=9)
        ax.xaxis.label.set_size(10)
        ax.yaxis.label.set_size(10)
    write(fig, "dyn_tails_glides")


def dyn_tails_sc():
    """mu_G - mu_A of the durations of the searches that end in a climb (circles) and of the climbs (squares) per ground class and
    on every flight, with 90% bands, one row per class."""
    t = read("phases_refit")["tails"]
    fig, ax = plt.subplots(figsize=(4.6, 2.2))
    for phase, fmt, off, color in (("search", "o", -0.17, "#9a7fc0"), ("climb", "s", 0.17, "#2f4465")):
        for i, k in enumerate(KEYS):
            bar(ax, None, i + off, t[k][phase]["gaps"]["G - A"], False, color, ms=6, fmt=fmt)
    ax.axvline(0, color="k", lw=0.8)
    forest_axis(ax, False)
    ax.set_yticks(range(len(KEYS)), SHORT[:-1] + ("every flight",))
    ax.xaxis.set_major_locator(MaxNLocator(5, integer=True))
    ax.tick_params(labelsize=9.5)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel(r"$\mu_G-\mu_A$", fontsize=10.5)
    ax.spines[["top", "right", "left"]].set_visible(False)
    fig.legend(handles=[Line2D([], [], marker=m, ls="", color=c, ms=6, label=n) for m, c, n in (("o", "#9a7fc0", "searches ending in a climb"), ("s", "#2f4465", "climbs"))],
               frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.0), fontsize=9, handletextpad=0.2, columnspacing=1.2)
    write(fig, "dyn_tails_sc", rect=(0, 0, 1, 0.9))


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
TAILS = (("transition", "glides"), ("search", "searches ending\nin a climb"), ("climb", "climbs"))


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


def why_peak():
    """High mountains, two panels on a log lag axis up to the end of the solid part: C* of A and G of the route (top, 1/e dotted,
    tau_c of A as a hollow dot) and the absolute G - A of alpha* of the real path (bottom); the top panel has a right axis with C*_G - C*_A. Three zones on both panels: both
    remember (up to the lag where the paired band of C*_G - C*_A first lies above zero), A forgets and G holds (up to tau_c of A),
    both forget; the peak lag of G - A is the grey dashed line."""
    legs, c = read("legs"), "high mountains"
    chained, peak = legs["chained"]["strata"][c], legs["peak test chained"][c]
    gap_c = chained["G - A"]
    lo = np.array(gap_c["band"][0], float)
    lag1 = float(np.array(gap_c["lags"])[lo > 0][0])
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
    bot.legend([Line2D([], [], color=A, lw=1.5), Line2D([], [], color=G, lw=1.5), Line2D([], [], color=GAP["G - A"], lw=1.5)],
               ["A", "G", r"$C^*_G-C^*_A$"], frameon=False, loc="upper left", ncol=3, fontsize=8, handlelength=1.5)
    write(fig, "why_peak")


def path_memory():
    memory_figure("path_memory", lambda legs, c, pop: legs["path"][c][pop])


def legs_tables():
    """legs_macros.tex and legs_table.tex: the memory of the route (lag and length at which C falls to 1/e, lag at 0.5) from legs.json."""
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
    macros["LegThalfHillsA"] = f"{peak['hills']['route']['A']['tau_half']['value']:.0f}"
    for c, tag in (("low mountains", "Low"), ("high mountains", "High")):
        for P in ("A", "G"):
            macros[f"LegPathTc{tag}{P}"] = f"{peak[c]['path'][P]['tau_c']['value']:.0f}"
    (DECK / "tikz/legs_macros.tex").write_text("".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in macros.items()))
    rows = []
    for cls in CLASSES:
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


def deck_tables():
    """The tables of the deck, from the generated tables of the report: the control events named as such, the three steps of every
    flight (real, control, real - control), the block of every flight of the full table of the phase statistics, and the main
    table of the phase statistics in two halves."""
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
    (DECK / "tikz/phases_all_table.tex").write_text("\n".join(lines[:heads[0]] + lines[last:]) + "\n")
    main = (TABLES / "phases_main_table.tex").read_text()
    (DECK / "tikz/phases_main_a.tex").write_text(table_columns(main, (0, 1, 2)))
    (DECK / "tikz/phases_main_b.tex").write_text(table_columns(main, (3, 4)))


if __name__ == "__main__":
    for f in (nested, chain_pool, persistent, gaps, ground_gaps, where, scale, cycle, opening, opening_kind, opening_ground, pilots, when_all, when_long, control_steps, carry,
              dyn_time, dyn_find, dyn_glide, dyn_tails_glides, dyn_tails_sc, route, route_memory, route_memory_chained, why_peak, path_memory, forest, tails):
        f()
    deck_tables()
    simpson()
    legs_tables()
    for name, (w, h) in SIZES.items():
        print(f"slide_{name}: {w:.2f} x {h:.2f} in")
