"""Figures of the slides of meetings/2026-11/group_vs_solo that show a few panels or rows of the results of the
report, drawn from the JSON of the analyses; nothing is recomputed.

    uv run python slides_figures.py --open
"""

import json

import matplotlib.pyplot as plt
import decompose
import decomposition
import matched
from tamsd import COLORS, OUT, save

STATS = ("time in climb (%)", "searches that end in a climb (%)", "climbs per hour", "glide speed (km/h)", decompose.SINK)
FACTORS = ("H_loc(1840 s)", "time in climb (%)", "glide speed (km/h)", decompose.SINK)
AGREE = ("time in climb (%)", "climbs per hour", "time in search (%)", "climb rate (m/s)", "glide speed (km/h)", decompose.SINK)
MAIN = ("H_loc(1840 s)", "time in climb (%)", "glide speed (km/h)")


def read(name):
    return json.loads((OUT / f"{name}.json").read_text())


def quote(q):
    return {"value": float(q["value"]), "band": [float(x) for x in q["band"]]}


def raw_phases():
    stats = read("raw")["phase statistics"]
    rows = [(label, COLORS[pop], {s: quote(stats[short]["average first"][s]) for s in STATS})
            for label, pop, short in (("S", "solo", "S"), ("A", "group_solo", "A"), ("G", "group", "G"))]
    matched.draw_forest(rows, OUT / "slide_raw_phases.pdf", "Open paragliders: the phase statistics", stats=list(STATS),
                        width=5.0, prefix="", zero=False)


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


def fitted_h():
    """The fitted H (10 s to the end of the lags) in the three designs: of S, A, G on each dataset of the raw
    comparison; the differences of the same route; g of the same flight against the length of the window."""
    fig, axes = plt.subplots(1, 3, figsize=(8.4, 2.9), constrained_layout=True)
    raw = read("raw")["datasets"]
    for k, (pop, short) in enumerate((("solo", "S"), ("group_solo", "A"), ("group", "G"))):
        for y, ds in enumerate(("1000", "2760", "longest")):
            h = raw[ds]["populations"][short]
            axes[0].errorbar(h["H"], y + 0.2 * (k - 1), xerr=[[h["H"] - h["H_band"][0]], [h["H_band"][1] - h["H"]]], fmt="o", ms=4,
                             color=COLORS[pop], elinewidth=1.6, capsize=2, label=short if y == 0 else None)
    axes[0].set(yticks=range(3), yticklabels=[f"to {raw[ds]['tau_max']:,} s" for ds in ("1000", "2760", "longest")], ylim=(2.6, -0.6),
                title="raw: H of S, A, G")
    axes[0].legend(frameon=False, fontsize=7)
    deltas = read("matched")["levels"]["all"]["deltas"]
    cases = [(k, m) for m in ("H (10 s to 1000 s)", "H (10 s to T)") for k in ("G - S|G", "A - S|A")]
    for y, (k, m) in enumerate(cases):
        q = quote(deltas[k][m])
        axes[1].errorbar(q["value"], y, xerr=[[q["value"] - q["band"][0]], [q["band"][1] - q["value"]]], fmt="o", ms=4,
                         color="#eb6834" if k.startswith("G") else "#1baf7a", elinewidth=1.6, capsize=2)
    axes[1].set(yticks=range(4), yticklabels=["G - S|G, to 1000 s", "A - S|A, to 1000 s", "G - S|G, to 2760 s", "A - S|A, to 2760 s"],
                ylim=(3.6, -0.6), title="same route: $\\Delta H$")
    for y, w in enumerate((1260, 1800, 2400, 3000)):
        q = quote(read(f"prepost_{w}")["differences"]["g"]["Delta H"])
        axes[2].errorbar(q["value"], y, xerr=[[q["value"] - q["band"][0]], [q["band"][1] - q["value"]]], fmt="o", ms=4,
                         color="#8c2f0f", elinewidth=1.6, capsize=2)
    axes[2].set(yticks=range(4), yticklabels=[f"window {w} s" for w in (1260, 1800, 2400, 3000)], ylim=(3.6, -0.6),
                title="same flight: $\\Delta H$ of $g$, to 0.8 W")
    for ax in axes[1:]:
        ax.axvline(0, color="#999999", lw=0.8, ls=":")
    for ax in axes:
        ax.grid(color="#eeeeee", lw=0.5)
        ax.tick_params(labelsize=7)
        ax.title.set_fontsize(8.5)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    save(fig, OUT / "slide_fitted_h.pdf")


if __name__ == "__main__":
    result = read("decomposition")["levels"]["all"]
    decomposition.draw_decomposition(result, OUT / "decomposition_main.pdf", stats=list(MAIN), width=5.4)
    raw_phases()
    factors()
    selection()
    agreement()
    fitted_h()
