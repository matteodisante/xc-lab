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
    if vertical:
        ax.errorbar(x, q["value"], yerr=err, fmt="o", ms=ms, color=color, elinewidth=1.6, capsize=3, **kw)
    else:
        ax.errorbar(q["value"], y, xerr=err, fmt="o", ms=ms, color=color, elinewidth=1.6, capsize=3, **kw)


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
    three steps with the control of two solo windows of group flights (right)."""
    r = read("prepost_ground")["classes"]["all"]
    fig, (left, right) = plt.subplots(1, 2, figsize=(6.0, 2.0), gridspec_kw={"width_ratios": [1, 1.1]})
    for i, stage in enumerate(prepost.STAGES):
        bar(left, i, None, r["alpha" + suffix][stage], True, prepost.LINES[stage][0], ms=7)
    left.set(xticks=range(4), xticklabels=prepost.STAGES, xlim=(-0.5, 3.5), ylabel=r"$\alpha$ of the window")
    keys = ("start - pre", "end - start", "post - end", "ctrlA2 - ctrlA1")
    for i, key in enumerate(keys):
        bar(right, i, None, r["differences" + suffix][key], True, prepost.GROUND_COLORS[key], ms=7)
    right.axhline(0, color="k", lw=0.6)
    right.set(xticks=range(4), xticklabels=["entry", "inside", "exit", "control"], xlim=(-0.5, 3.5), ylabel=r"difference of $\alpha$")
    for ax in (left, right):
        ax.grid(color="#eeeeee", lw=0.5)
        ax.tick_params(axis="x", labelsize=8)
    write(fig, name)


def when_all():
    steps("", "when_all")


def when_long():
    steps("_long", "when_long")


def carry():
    """carry.pdf, bottom row: A(d) - S of the exponent over the long lags per bin of the distance d from the group time,
    on every flight and in the high mountains."""
    c = read("carry")
    fig, axes = plt.subplots(1, 2, figsize=(6.0, 2.0), sharey=True)
    for ax, k in zip(axes, ("all", "high mountains")):
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

PERF = (("height lost per km of glide (m/km)", "height lost per km\nof glide (m/km)"),
        ("searches that end in a climb (%)", "searches that end\nin a climb (%)"), ("climb rate (m/s)", "climb rate\n(m/s)"))


def performance():
    """phases_performance.pdf: G - A of the height lost per km, the searches that end in a climb and the climb rate, per class
    and on every flight, with 90% bands (forest of phase_gaps, the gap G - A only)."""
    ph = read("phases_refit")["phases"]
    ys = np.arange(len(phase_gaps.CLASS_KEYS))
    fig, axes = plt.subplots(1, 3, figsize=(WIDE, 2.0), sharey=True)
    for ax, (key, title) in zip(axes, PERF):
        for y, c in zip(ys, phase_gaps.CLASS_KEYS):
            bar(ax, None, y, ph[c]["gaps"]["G - A"][key], False, GAP["G - A"], ms=6)
        ax.axvline(0, color="k", lw=0.6)
        ax.grid(axis="x", color="#eeeeee", lw=0.5)
        ax.xaxis.set_major_locator(MaxNLocator(3))
        ax.set_title(title, fontsize=8.5)
    axes[0].set_yticks(ys, phase_gaps.CLASS_KEYS)
    axes[0].set_ylim(ys[-1] + 0.5, -0.5)
    write(fig, "performance")


def survival_row(panels, name):
    """phases_tails.pdf: the survival of the durations of S, A, G with its band, the common threshold (dotted) and the Hill
    exponent mu of each, one panel per (phase, class) of `panels`."""
    t = read("phases_refit")["tails"]
    fig, axes = plt.subplots(1, len(panels), figsize=(WIDE, 2.0), sharey=True)
    for ax, (phase, c, title) in zip(axes, panels):
        d = t[c][phase]
        for pop, short in raw.POPS.items():
            s, mu = d["survival"][short], d["mu"][short]
            g, v, b = np.array(s["grid"]), np.array(s["value"]), np.array(s["band"])
            ok = v > 0
            ax.fill_between(g[ok], np.maximum(b[0][ok], 1e-9), b[1][ok], color=COLORS[pop], alpha=0.2, lw=0)
            ax.plot(g[ok], v[ok], color=COLORS[pop], lw=1.5, label=f"{short}: {mu['value']:.2f}")
        ax.axvline(d["xmin"]["common"], color=GREY, lw=0.8, ls=":")
        ax.set(xscale="log", yscale="log", xlim=(1, 8e3), ylim=(1e-5, 1.2), xlabel=r"$\tau$ (s)", title=title)
        ax.set_xticks([1, 100, 1e4])
        ax.set_yticks([1, 1e-2, 1e-4])
        ax.minorticks_off()
        ax.grid(color="#eeeeee", lw=0.5)
        ax.legend(frameon=False, loc="lower left", fontsize=7, handlelength=1.2, title=r"$\mu$", title_fontsize=7, alignment="left")
    axes[0].set_ylabel("survival")
    write(fig, name)


def glides():
    survival_row([("transition", c, c) for c in CLASSES], "glides")


def searches_climbs():
    survival_row([(phase, c, f"{text}, {name}") for phase, text in (("search", "searches"), ("climb", "climbs"))
                  for c, name in (("all", "all"), ("high mountains", "high mtn"))], "searches_climbs")


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
    for f in (nested, chain_pool, persistent, gaps, ground_gaps, where, scale, cycle, opening, opening_kind, opening_ground, pilots, when_all, when_long, carry,
              performance, glides, searches_climbs, route, forest, tails):
        f()
    deck_tables()
    simpson()
    for name, (w, h) in SIZES.items():
        print(f"slide_{name}: {w:.2f} x {h:.2f} in")
