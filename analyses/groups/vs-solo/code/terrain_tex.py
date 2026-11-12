"""The numbers, tables and figure of the section on the terrain (comparisons/sections/terrain.tex), from
terrain_compare.json and terrain.json: vs-solo/generated/comparisons/terrain_macros.tex, terrain_*_table.tex and
out/<run>/open_min<m>/terrain_forest.pdf. Every macro starts with Tn.

    uv run python vs-solo/code/terrain_tex.py    # from analyses/groups, after terrain.py --open --compare
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parents[2]))
import terrain  # noqa: E402  (its constants only)
from figures import GREY, forest  # noqa: E402
from openflights import MIN_OPENING  # noqa: E402
from tables import MAIN, OUTS, signed  # noqa: E402
from tables import num as _num  # noqa: E402

GEN = OUTS.parent / "vs-solo" / "generated" / "comparisons"
OPENED = OUTS / MAIN / f"open_min{MIN_OPENING:g}"
macros = {}
RAW = {"lowland": "Lowland", "low mountains": "Low", "high mountains": "High"}
BINARY = {"lowland": "Lowland", "mountain": "Mountain"}
COLORS = {"lowland": terrain.CLASS_COLORS[2], "low mountains": terrain.CLASS_COLORS[3], "high mountains": terrain.CLASS_COLORS[4],
          "mountain": "#7a5a8a"}
LABEL = {"lowland": "lowland", "low mountains": "low mountains", "high mountains": "high mountains", "mountain": "mountains"}


def num(x, digits=0):
    return ("$-$" if round(x, digits) < 0 else "") + _num(abs(x), digits)


def macro(name, value):
    macros["Tn" + name] = value


def quote(e, digits=3):
    lo, hi = e["band"]
    return signed(e["value"], digits), f"[{signed(lo, digits)}, {signed(hi, digits)}]"


def put(name, e, digits=3):
    value, band = quote(e, digits)
    macro(name, value)
    macro(name + "Band", band)


def parameters():
    p = terrain
    macro("Cell", num(p.CELL_M))
    macro("Block", num(p.BLOCK))
    macro("Source", num(p.SRC_M))
    macro("Gentle", f"{p.GENTLE:g}")
    macro("Window", num(p.WINDOW))
    macro("WindowKm", num(p.WINDOW * p.CELL_M / 1000, 1))
    macro("Coverage", f"{p.COVERAGE:g}")
    macro("PlainsRelief", num(p.PLAINS_RELIEF))
    macro("HillsRelief", num(p.HILLS_RELIEF))
    macro("LowRelief", num(p.LOW_MOUNTAIN_RELIEF))
    macro("PlainsGentle", f"{p.PLAINS_GENTLE:g}")
    macro("Majority", num(100 * p.MAJORITY))
    macro("GridStep", num(p.GRID_S))


def counts(c, t):
    """Macros of the pieces by class and population, and of the events."""
    pieces = t["pieces of dataset 1000"]
    for pop in "SAG":
        macro(f"Pieces{pop}", num(pieces[pop]["items"]))
        for k, tag in (("plains", "Plains"), ("hills", "Hills"), ("low mountains", "Low"), ("high mountains", "High"), ("unknown", "Unknown"), ("mixed", "Mixed")):
            macro(f"Pieces{tag}{pop}", num(pieces[pop]["classes"][k]["n"]))
        for k, tag in (("lowland", "Lowland"), ("mountain", "Mountain")):
            macro(f"Pieces{tag}{pop}", num(pieces[pop]["binary"][k]["n"]))
        pts = t["grid points"][pop]
        macro(f"Points{pop}", num(pts["points"] / 1e6, 1))
        macro(f"ShareLowland{pop}", num(pts["lowland"], 0))
        macro(f"ShareMountain{pop}", num(pts["mountain"], 0))
    ev = t["prepost events, windows of 2400 s"]
    macro("EventsAll", num(ev["events with four windows"]))
    for k, tag in (("lowland", "Lowland"), ("mountain", "Mountain"), ("unknown", "Unknown"), ("mixed", "Mixed")):
        macro(f"Events{tag}Classed", num(ev["binary"][k]["n"]))
    macro("MergedG", num(c["raw"]["plains merged into hills"]["pieces added to hills"]["G"]))
    macro("MergedFrom", num(c["raw"]["plains merged into hills"]["hills pieces before"]["G"]))
    macro("MergedTo", num(c["raw"]["plains merged into hills"]["hills pieces after"]["G"]))
    macro("FlightsClassed", num(t["open flights in the table"]))
    macro("MaskedShare", num(t["sea mask"]["share of the blocks with an elevation before the mask (%)"], 1))


def results(c):
    """Macros of the comparisons by class."""
    for k, tag in RAW.items():
        d = c["raw"]["binary"]["lowland"] if k == "lowland" else c["raw"]["four classes"][k]
        for pop in "SAG":
            for key, tag2 in (("pieces", "Pieces"), ("flights", "Flights"), ("pilots", "Pilots")):
                macro(f"Raw{tag}{tag2}{pop}", num(d["counts"][pop][key]))
        for g, gt in (("G - S", "GS"), ("A - S", "AS"), ("G - A", "GA")):
            put(f"Raw{tag}{gt}", d["H_loc(1000 s) gaps"][g], 3)
        for pop in "SAG":
            macro(f"Raw{tag}H{pop}", f"{d['H_loc(1000 s)'][pop]['value']:.3f}")
    d = c["raw"]["binary"]["mountain"]
    for pop in "SAG":
        for key, tag2 in (("pieces", "Pieces"), ("flights", "Flights"), ("pilots", "Pilots")):
            macro(f"RawMountain{tag2}{pop}", num(d["counts"][pop][key]))
    for g, gt in (("G - S", "GS"), ("A - S", "AS"), ("G - A", "GA")):
        put(f"RawMountain{gt}", d["H_loc(1000 s) gaps"][g], 3)
    for k, tag in BINARY.items():
        r, f = c["route"][k], c["flight"][k]
        for s in "GA":
            for key, tag2 in (("pieces", "Pieces"), ("controls", "Controls"), ("flights, %s side" % s, "Flights"), ("pilots, %s side" % s, "Pilots"),
                              ("pool flights", "Pool")):
                macro(f"Route{tag}{tag2}{s}", num(r["counts"][s][key]))
        for g, gt in (("G - S|G", "G"), ("A - S|A", "A"), ("(G - S|G) - (A - S|A)", "D")):
            put(f"Route{tag}{gt}", r["deltas"][g]["H_loc(1000 s)"], 3)
        macro(f"Flight{tag}Events", num(f["events"]))
        macro(f"Flight{tag}Flights", num(f["flights"]))
        macro(f"Flight{tag}Pilots", num(f["pilots"]))
        for eff, et in (("g", "G"), ("D_in", "In"), ("D_out", "Out"), ("d w", "Dw")):
            put(f"Flight{tag}{et}", f["effects"][eff]["H_loc(1000 s)"], 3)
    macro("RouteMatchedShareLowland", num(100 * c["route"]["lowland"]["counts"]["G"]["pieces"] / c["raw"]["binary"]["lowland"]["counts"]["G"]["pieces"], 1))
    macro("RouteMatchedShareMountain", num(100 * c["route"]["mountain"]["counts"]["G"]["pieces"] / c["raw"]["binary"]["mountain"]["counts"]["G"]["pieces"], 0))


def table(name, header, rows, spec):
    lines = [rf"\begin{{tabular}}{{{spec}}}", r"\toprule", header, r"\midrule"]
    for r in rows:
        lines.append(r"\addlinespace[2pt]\multicolumn{1}{@{}l}{\emph{" + r[1:] + r"}}\\" if isinstance(r, str) else " & ".join(r) + r" \\")
    (GEN / f"{name}.tex").write_text("\n".join(lines + [r"\bottomrule", r"\end{tabular}"]) + "\n")


def cells(e):
    lo, hi = e["band"]
    return [signed(e["value"], 3), f"[{signed(lo, 3)}, {signed(hi, 3)}]"]


def rows_of(c):
    """(label, units, flights, pilots, entry, color key) of the three designs."""
    rows = []
    for k in ("lowland", "low mountains", "high mountains", "mountain"):
        d = c["raw"]["binary"][k] if k in ("lowland", "mountain") else c["raw"]["four classes"][k]
        n = d["counts"]["G"]
        rows.append(("raw", k, n["pieces"], n["flights"], n["pilots"], d["H_loc(1000 s) gaps"]["G - S"]))
    for k in BINARY:
        d = c["route"][k]
        rows.append(("route", k, d["counts"]["G"]["pieces"], d["counts"]["G"]["flights, G side"], d["counts"]["G"]["pilots, G side"],
                     d["deltas"]["(G - S|G) - (A - S|A)"]["H_loc(1000 s)"]))
    for k in BINARY:
        d = c["flight"][k]
        rows.append(("flight", k, d["events"], d["flights"], d["pilots"], d["effects"]["g"]["H_loc(1000 s)"]))
    return rows


def tables(c, t):
    rows = ["#raw: pieces of G, G $-$ S"]
    design = {"raw": "raw: pieces of $\\G$, $\\G-\\Sl$", "route": "same route: matched pieces of $\\G$, $(\\G-\\SG)-(\\A-\\SAA)$",
              "flight": "same flight: events, $g$"}
    last = None
    for d, k, n, fl, pl, e in rows_of(c):
        if d != last:
            rows.append("#" + design[d])
            last = d
        rows.append([LABEL[k], num(n), num(fl), num(pl), *cells(e)])
    rows = rows[1:]
    table("terrain_table", r"class & units & flights & pilots & $\Delta\Hloc(1000~\mathrm{s})$ & 90\% band \\", rows, r"@{}l r r r r l@{}")
    p = t["pieces of dataset 1000"]
    head = r"class & $\Sl$ & $\A$ & $\G$ \\"
    body = [[name.replace("mountains", "mountains"), *[num(p[pop]["classes"][k]["n"]) for pop in "SAG"]]
            for k, name in (("plains", "plains"), ("hills", "hills"), ("low mountains", "low mountains"), ("high mountains", "high mountains"),
                            ("mixed", "mixed"), ("unknown", "unknown"))]
    body.append(["all pieces", *[num(p[pop]["items"]) for pop in "SAG"]])
    table("terrain_pieces_table", head, body, r"@{}l r r r@{}")


def figure(c):
    rows, breaks = [], []
    for d, k, n, fl, pl, e in rows_of(c):
        rows.append((f"{d} {LABEL[k]} ({n:,})", e["value"], tuple(e["band"]), COLORS[k], k != "mountain" or d != "raw"))
    forest(ax := plt.subplots(figsize=(6.6, 3.0), constrained_layout=True)[1], rows, r"$\Delta H_{\rm loc}(1000~{\rm s})$ by class and design",
           breaks=(3, 5), ticks=5)
    ax.set_xlim(-0.06, 0.1)
    fig = ax.figure
    fig.savefig(OPENED / "terrain_forest.pdf", bbox_inches="tight")
    fig.savefig(OPENED / "terrain_forest.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    GEN.mkdir(parents=True, exist_ok=True)
    c = json.loads((OPENED / "terrain_compare.json").read_text())
    t = json.loads((OPENED / "terrain.json").read_text())
    parameters()
    counts(c, t)
    results(c)
    tables(c, t)
    figure(c)
    (GEN / "terrain_macros.tex").write_text("\n".join(rf"\newcommand{{\{k}}}{{{v}}}" for k, v in sorted(macros.items())) + "\n")
    print(f"{len(macros)} macros")
