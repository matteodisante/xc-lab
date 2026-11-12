"""The figures of the section on the tails of the phase durations (comparisons/sections/tails.tex), from
tail.json, matched.json and tail_designs.json: out/<run>/open_min<m>/tails_*.pdf.

Colour as in figures.py: S blue, G orange, the same flights alone green (the solo time before and after the group
time: filled before, open after), the controls of A dark blue. A dot is an estimate, a bar its 90% band.

    uv run python vs-solo/code/tail_figures.py    # from analyses/groups, after tail_designs.py --open
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parents[2]))
from figures import COLOR, GREY, LIGHT, MUTED, WIDTH, forest  # noqa: E402
from openflights import MIN_OPENING  # noqa: E402
from tables import MAIN, OUTS  # noqa: E402

OPENED = OUTS / MAIN / f"open_min{MIN_OPENING:g}"
CTL = "#16427a"  # the controls of A
PAL = {"S": COLOR["S"], "A": COLOR["A"], "G": COLOR["G"], "S|G": COLOR["S"], "S|A": CTL,
       "before": COLOR["A"], "in group": COLOR["G"], "after": COLOR["A"], "spanning": GREY, "between": MUTED}
FILLED = {"after": False, "S|G": False, "S|A": False, "between": False}
ROUTE = ("G", "S|G", "A", "S|A")
FLIGHT = ("before", "in group", "after")
TITLE = {"transition": "transitions", "climb": "climbs", "search": "searches ending in a climb"}
NAME = {"S": "S", "A": "A", "G": "G", "S|G": "S|G", "S|A": "S|A", "before": "before", "in group": "in group", "after": "after",
        "spanning": "spanning", "between": "between"}
GAPS = (("raw", "G - S", "G − S"), ("raw", "A - S", "A − S"), ("route", "(G - S|G) - (A - S|A)", "(G − S|G) − (A − S|A)"),
        ("same flight", "in group - mean(before, after)", "in group − mean(before, after)"),
        ("same flight", "after - before", "after − before"))
XMIN_TICKS = [300, 500, 1000, 2000, 4000]


def save(fig, name):
    fig.savefig(OPENED / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OPENED / f"{name}.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def load():
    read = lambda n: json.loads((OPENED / f"{n}.json").read_text())
    return read("tail_designs"), read("tail"), read("matched")


def curve(ax, x, y, band, color, label, filled=True):
    ax.fill_between(x, np.array(band[0], float), np.array(band[1], float), color=color, alpha=0.18, lw=0)
    ax.plot(x, np.array(y, float), color=color, lw=1.3, ls="-" if filled else "--", label=label)


def survival(d, old, matched):
    """P(duration > t) of the transitions in the three designs, with the threshold of S."""
    xmin = d["xmin of S"]["transition"]
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 2.4), sharey=True, constrained_layout=True)
    sets = (("raw", {s: old["survival"][s] for s in ("S", "A", "G")}),
            ("same route", {"G": matched["survival"]["G"]["transition"]["G"], "S|G": matched["survival"]["G"]["transition"]["S|G"],
                            "A": matched["survival"]["A"]["transition"]["G"], "S|A": matched["survival"]["A"]["transition"]["S|G"]}),  # side A keeps the keys of the group side
            ("same flight", {c: d["same flight"]["transition"]["survival"][c] for c in FLIGHT}))
    for ax, (title, curves) in zip(axes, sets):
        for name, s in curves.items():
            t, v = np.array(s["grid"], float), np.array(s["value"], float)
            ok = v > 0
            band = np.array(s["band"], float)
            curve(ax, t[ok], v[ok], (np.maximum(band[0][ok], 1e-9), band[1][ok]), PAL[name], NAME[name], FILLED.get(name, True) and name != "S|A")
        ax.axvline(xmin, color=MUTED, lw=0.8, ls=":")
        ax.set(xscale="log", yscale="log", xlim=(100, 2e4), ylim=(1e-5, 1.5), xlabel=r"$\tau$ (s)")
        ax.set_title(title, loc="left")
        ax.legend(loc="lower left", fontsize=7)
        ax.grid(color=LIGHT, lw=0.5)
    axes[0].set_ylabel(r"$P(\tau_\mathrm{transition} > \tau)$")
    save(fig, "tails_survival")


def hill(d):
    """The Hill index against the threshold in the three designs, and the KS distance of the raw design."""
    fig, axes = plt.subplots(2, 2, figsize=(WIDTH, 4.6), constrained_layout=True)
    xmin = d["xmin of S"]["transition"]
    r = d["raw"]["transition"]["populations"]
    for s in ("S", "A", "G"):
        c = r[s]["hill"]
        curve(axes[0, 0], c["xmin"], c["value"], c["band"], PAL[s], s)
    for s in ROUTE:
        c = d["same route"]["transition"]["curves"][s]
        curve(axes[0, 1], c["xmin"], c["value"], c["band"], PAL[s], s, s not in ("S|G", "S|A"))
    for c in (*FLIGHT, "spanning"):
        h = d["same flight"]["transition"]["curves"][c]["hill"]
        curve(axes[1, 0], h["xmin"], h["value"], h["band"], PAL[c], c, c != "after")
    for s in ("S", "A", "G"):
        k = r[s]["ks"]
        axes[1, 1].plot(k["xmin"], np.array(k["D"], float), color=PAL[s], lw=1.3, label=s)
    titles = ("raw", "same route (weighted)", "same flight", "raw: Kolmogorov-Smirnov distance")
    for ax, title in zip(axes.flat, titles):
        ax.set(xscale="log", xlim=(300, 4000), xlabel=r"$x_\mathrm{min}$ (s)")
        ax.set_xticks(XMIN_TICKS, [str(x) for x in XMIN_TICKS])
        ax.minorticks_off()
        ax.axvline(xmin, color=MUTED, lw=0.8, ls=":")
        ax.set_title(title, loc="left")
        ax.legend(loc="best", fontsize=7)
        ax.grid(color=LIGHT, lw=0.5)
    for ax in axes.flat[:3]:
        ax.set(ylim=(0, 10), ylabel=r"Hill $\hat\mu(x_\mathrm{min})$")
    axes[1, 1].set(ylabel=r"$D(x_\mathrm{min})$", ylim=(0, 0.12))
    save(fig, "tails_hill")


def rows_mu(d, phase):
    """Rows (label, value, band, color, filled) of the populations and classes of the three designs, with the breaks."""
    raw, route, flight = d["raw"][phase], d["same route"][phase], d["same flight"][phase]["variants"]["whole in stretch"]["classes"]
    entries = [(f"raw {s}", raw["populations"][s]["at xmin of S"], s) for s in ("S", "A", "G")]
    entries += [(f"route {s}", route["populations"][s], s) for s in ROUTE]
    entries += [(f"flight {c}", flight[c], c) for c in (*FLIGHT, "spanning")]
    rows = []
    for label, e, key in entries:
        if e["value"] is None or None in e["band"]:
            continue
        n = f" ({e['n']:,.0f})" if e["n"] is not None else ""
        rows.append((label.replace("|", "|") + n, e["value"], tuple(e["band"]), PAL[key], FILLED.get(key, True)))
    return rows


def forest_mu(d):
    """mu of the transitions in every population and class, and the contrasts."""
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 3.6), constrained_layout=True, gridspec_kw={"width_ratios": [1.25, 1]})
    rows = rows_mu(d, "transition")
    forest(axes[0], rows, r"$\hat\mu$ at the threshold of S (runs above)", zero=False, breaks=(2, 6), ticks=4)
    cons = []
    for design, key, label in GAPS:
        e = {"raw": d["raw"]["transition"]["gaps"], "route": d["same route"]["transition"]["contrasts"],
             "same flight": d["same flight"]["transition"]["variants"]["whole in stretch"]["contrasts"]}[design][key]
        cons.append((label, e["value"], tuple(e["band"]), COLOR["R"], True))
    forest(axes[1], cons, "contrasts", breaks=(1, 2), ticks=4)
    save(fig, "tails_forest")


def forest_phases(d):
    """mu of the climbs and of the searches."""
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 3.6), constrained_layout=True)
    for ax, phase in zip(axes, ("climb", "search")):
        forest(ax, rows_mu(d, phase), f"{TITLE[phase]}: " + r"$\hat\mu$ (runs above)", zero=False, breaks=(2, 6), ticks=3)
        if ax is axes[1]:
            ax.set_yticklabels([])
    save(fig, "tails_phases")


def descriptors(d):
    """Medians of the descriptors of the long transitions, with their 90% bands."""
    keys = (("path (km)", "path (km)"), ("straightness", "straightness"), ("time rising (share)", "share of time rising"),
            ("vertical speed (m/s)", "vertical speed (m/s)"), ("height lost (m/km)", "height lost per km of net\ndisplacement (m/km)"),
            ("height lost, total (m)", "height lost in total (m)"))
    pops = [("raw", s, s) for s in ("S", "A", "G")] + [("same route", s, s) for s in ROUTE] + [("same flight", c, c) for c in (*FLIGHT, "spanning")]
    fig, axes = plt.subplots(2, 3, figsize=(WIDTH, 4.8), constrained_layout=True)
    for ax, (k, title) in zip(axes.flat, keys):
        rows = []
        for design, s, key in pops:
            e = d[design]["transition"]["descriptors"][s]
            rows.append((f"{design.replace('same ', '')} {NAME[s]} ({e['runs']})", e[k]["median"], tuple(e[k]["median band"]), PAL[key], FILLED.get(key, True)))
        forest(ax, rows, title, zero=False, breaks=(2, 6), ticks=3)
        ax.tick_params(axis="x", labelsize=7)
        if ax not in axes[:, 0]:
            ax.set_yticklabels([])
    save(fig, "tails_descriptors")


if __name__ == "__main__":
    d, old, matched = load()
    survival(d, old, matched)
    hill(d)
    forest_mu(d)
    forest_phases(d)
    descriptors(d)
