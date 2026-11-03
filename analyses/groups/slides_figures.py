"""Figures of the slides of meetings/2026-11/group_vs_solo that show a few panels or rows of the results of the
report, drawn from the JSON of the analyses; nothing is recomputed.

    uv run python slides_figures.py --open
"""

import json

import matplotlib.pyplot as plt
import curves
import decompose
import decomposition
import matched
from tamsd import COLORS, OUT, save

PHASE_STATS = ("time in climb (%)", "time in search (%)", "searches that end in a climb (%)", "climbs per hour",
               "climb rate (m/s)", "glide speed (km/h)", decompose.SINK)
FACTORS = ("H_loc(1840 s)", "time in climb (%)", "glide speed (km/h)", decompose.SINK)
AGREE = ("time in climb (%)", "climbs per hour", "time in search (%)", "climb rate (m/s)", "glide speed (km/h)", decompose.SINK)
MAIN = ("H_loc(1840 s)", "time in climb (%)", "glide speed (km/h)")


def read(name):
    return json.loads((OUT / f"{name}.json").read_text())


def quote(q):
    return {"value": float(q["value"]), "band": [float(x) for x in q["band"]]}


def raw_phases():
    """H_loc and the seven phase statistics of S, A, G, on a 3 x 3 grid sized for the slide: `slide_raw_phases`."""
    raw = read("raw")
    stats = raw["phase statistics"]
    hloc = {"$H_{\\mathrm{loc}}(1000\\,\\mathrm{s})$": ("1000", "H_loc(1000 s)"),
            "$H_{\\mathrm{loc}}(1840\\,\\mathrm{s})$": ("2760", "H_loc(1840 s)")}
    rows = [(label, COLORS[pop], {**{s: quote(stats[short]["average first"][s]) for s in PHASE_STATS},
                                  **{name: quote(raw["datasets"][ds]["populations"][short][key]) for name, (ds, key) in hloc.items()}})
            for label, pop, short in (("S", "solo", "S"), ("A", "group_solo", "A"), ("G", "group", "G"))]
    short = {"time in climb (%)": "climb time (%)", "time in search (%)": "search time (%)",
             "searches that end in a climb (%)": "searches → climb (%)", decompose.SINK: "height lost (m/km)"}
    rows = [(label, color, {short.get(k, k): v for k, v in quotes.items()}) for label, color, quotes in rows]
    panels = [*hloc, "climb time (%)", "search time (%)", "searches → climb (%)", "climbs per hour",
              "climb rate (m/s)", "glide speed (km/h)", "height lost (m/km)"]
    number = {name: f"{i}  {name}" for i, name in enumerate(panels, 1)}
    rows = [(label, color, {number[k]: v for k, v in quotes.items()}) for label, color, quotes in rows]
    matched.draw_forest(rows, OUT / "slide_raw_phases.pdf", None, stats=list(number.values()), per_row=3, width=4.4,
                        prefix="", zero=False, row_h=1.08)


def raw_delta():
    """H_loc of S, A, G with their flights in the legend, beside the differences, on the dataset of 1000 s."""
    d = read("raw")["datasets"]["1000"]
    left = [{**d["populations"][short]["curve"], "label": f"{short} ({d['populations'][short]['flights']:,} flights)", "color": COLORS[pop]}
            for pop, short in (("solo", "S"), ("group_solo", "A"), ("group", "G"))]
    gaps = read("raw")["gaps"]["1000"]["H_loc"]
    colors = {"G - S": COLORS["group"], "A - S": COLORS["group_solo"], "G - A": "#8c2f0f"}
    right = [{**gaps[name], "label": name, "color": colors[name]} for name in colors]
    g = gaps["G - S"]
    below = [i for i, v in enumerate(g["value"]) if v <= 0]
    turn = below[-1] + 1 if below and below[-1] + 1 < len(g["lags"]) else None
    curves.delta_figure(left, right, OUT / "slide_raw_delta",
                        vline=None if turn is None else (g["lags"][turn], f"G - S > 0 from {g['lags'][turn]:g} s"))


def factors():
    d = read("decomposition")["levels"]["all"]
    rows = [(label, color, {m: quote(d[m]["cells"][cell]["delta"]) for m in FACTORS})
            for label, color, cell in (("no factor fixed", "#555555", "none"), ("route fixed", "#16427a", "R"),
                                       ("route, season, hour, wing fixed", "#eb6834", "R+T+W"))]
    matched.draw_forest(rows, OUT / "slide_factors.pdf", "Open paragliders: group time against its controls",
                        stats=list(FACTORS), per_row=2, width=5.6)


def selection():
    d = read("matched")["levels"]["all"]["deltas"]
    rows = [(label, color, {m: quote(d[k][m]) for m in ("H_loc(1000 s)", "H_loc(1840 s)")})
            for label, color, k in (("G - S|G: the group", "#eb6834", "G - S|G"), ("A - S|A: the same pilots alone", "#1baf7a", "A - S|A"),
                                    ("difference", "#555555", "(G - S|G) - (A - S|A)"))]
    matched.draw_forest(rows, OUT / "slide_selection.pdf", "Same route, season, hour and wing",
                        stats=["H_loc(1000 s)", "H_loc(1840 s)"], per_row=2, width=5.4)


def agreement():
    route = read("matched")["levels"]["all"]["deltas"]["G - S|G"]
    flight = read("prepost_2400")["differences"]["g"]
    rows = [("same route: G - S|G", "#eb6834", {m: quote(route[m]) for m in AGREE}),
            ("same flight: g", "#8c2f0f", {m: quote(flight[m]) for m in AGREE})]
    matched.draw_forest(rows, OUT / "slide_agreement.pdf", "Group time against solo flight, two designs",
                        stats=list(AGREE), width=6.8)


def _finish(fig, ax, name):
    ax.grid(color="#eeeeee", lw=0.5)
    ax.tick_params(labelsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    save(fig, OUT / name)


def _bar(ax, y, q, color, **kw):
    ax.errorbar(q["value"], y, xerr=[[q["value"] - q["band"][0]], [q["band"][1] - q["value"]]], fmt="o", ms=5, color=color,
                elinewidth=1.8, capsize=2, **kw)


def fitted_h():
    """The fitted H (10 s to the end of the lags), one figure per design: of S, A, G on each dataset of the raw
    comparison; the differences of the same route; g of the same flight against the length of the window."""
    raw = read("raw")["datasets"]
    fig, ax = plt.subplots(figsize=(5.2, 2.6), constrained_layout=True)
    for k, (pop, short) in enumerate((("solo", "S"), ("group_solo", "A"), ("group", "G"))):
        for y, ds in enumerate(("1000", "2760", "longest")):
            h = raw[ds]["populations"][short]
            _bar(ax, y + 0.2 * (k - 1), {"value": h["H"], "band": h["H_band"]}, COLORS[pop], label=short if y == 0 else None)
    ax.set(yticks=range(3), yticklabels=[f"to {raw[ds]['tau_max']:,} s" for ds in ("1000", "2760", "longest")], ylim=(2.6, -0.6),
           xlabel="$H$, fitted from 10 s")
    ax.legend(frameon=False, fontsize=8)
    _finish(fig, ax, "slide_fitted_h_raw.pdf")
    deltas = read("matched")["levels"]["all"]["deltas"]
    fig, ax = plt.subplots(figsize=(5.2, 2.6), constrained_layout=True)
    for y, (k, m) in enumerate((k, m) for m in ("H (10 s to 1000 s)", "H (10 s to T)") for k in ("G - S|G", "A - S|A")):
        _bar(ax, y, quote(deltas[k][m]), "#eb6834" if k.startswith("G") else "#1baf7a")
    ax.axvline(0, color="#999999", lw=0.8, ls=":")
    ax.set(yticks=range(4), yticklabels=["G - S|G, to 1000 s", "A - S|A, to 1000 s", "G - S|G, to 2760 s", "A - S|A, to 2760 s"],
           ylim=(3.6, -0.6), xlabel="$\\Delta H$, fitted from 10 s")
    _finish(fig, ax, "slide_fitted_h_route.pdf")
    fig, ax = plt.subplots(figsize=(5.2, 2.6), constrained_layout=True)
    for y, w in enumerate((1260, 1800, 2400, 3000)):
        _bar(ax, y, quote(read(f"prepost_{w}")["differences"]["g"]["Delta H"]), "#8c2f0f")
    ax.axvline(0, color="#999999", lw=0.8, ls=":")
    ax.set(yticks=range(4), yticklabels=[f"window {w} s" for w in (1260, 1800, 2400, 3000)], ylim=(3.6, -0.6),
           xlabel="$\\Delta H$ of $g$, fitted from 10 s to 0.8 of the window")
    _finish(fig, ax, "slide_fitted_h_flight.pdf")


if __name__ == "__main__":
    result = read("decomposition")["levels"]["all"]
    decomposition.draw_decomposition(result, OUT / "decomposition_main.pdf", stats=list(MAIN), width=5.4)
    raw_phases()
    raw_delta()
    factors()
    selection()
    agreement()
    fitted_h()
