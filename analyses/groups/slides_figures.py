"""Figures of the slides of meetings/2026-11/group_vs_solo that show a few panels or rows of the results of the
report, drawn from the JSON of the analyses; nothing is recomputed.

    uv run python slides_figures.py --open
"""

import json

import decompose
import decomposition
import matched
from tamsd import COLORS, OUT

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
    matched.draw_forest(rows, OUT / "slide_raw_phases.pdf", "Open paragliders: the phase statistics (5-95% bands)", stats=list(STATS),
                        width=5.0, prefix="", zero=False)


def factors():
    d = read("decomposition")["levels"]["all"]
    rows = [(label, color, {m: quote(d[m]["cells"][cell]["delta"]) for m in FACTORS})
            for label, color, cell in (("no factor fixed", "#555555", "none"), ("route fixed", "#16427a", "R"),
                                       ("route, season, hour, wing fixed", "#eb6834", "R+T+W"))]
    matched.draw_forest(rows, OUT / "slide_factors.pdf", "Open paragliders: group time against its controls (5-95% bands)",
                        stats=list(FACTORS), per_row=2, width=5.6)


def selection():
    d = read("matched")["levels"]["all"]["deltas"]
    rows = [(label, color, {m: quote(d[k][m]) for m in ("H_loc(1000 s)", "H_loc(1840 s)")})
            for label, color, k in (("G - S|G: the group", "#eb6834", "G - S|G"), ("A - S|A: the same pilots alone", "#1baf7a", "A - S|A"),
                                    ("difference", "#555555", "(G - S|G) - (A - S|A)"))]
    matched.draw_forest(rows, OUT / "slide_selection.pdf", "Same route, season, hour and wing (5-95% bands)",
                        stats=["H_loc(1000 s)", "H_loc(1840 s)"], per_row=2, width=5.4)


def agreement():
    route = read("matched")["levels"]["all"]["deltas"]["G - S|G"]
    flight = read("prepost_2400")["differences"]["g"]
    rows = [("same route: G - S|G", "#eb6834", {m: quote(route[m]) for m in AGREE}),
            ("same flight: g", "#8c2f0f", {m: quote(flight[m]) for m in AGREE})]
    matched.draw_forest(rows, OUT / "slide_agreement.pdf", "Group time against solo flight, two designs (5-95% bands)",
                        stats=list(AGREE), width=5.8)


if __name__ == "__main__":
    result = read("decomposition")["levels"]["all"]
    decomposition.draw_decomposition(result, OUT / "decomposition_main.pdf", stats=list(MAIN), width=5.4)
    raw_phases()
    factors()
    selection()
    agreement()
