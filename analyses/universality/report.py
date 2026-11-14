"""The figures, tables and macros of universality.tex from out/<full | sample N>/results.json.

Figures go to out/<run>/, tables and macros to generated/ (the document reads them; never edited by hand).

    uv run python report.py [--sample N]    # after analysis.py
"""

import itertools
import json
import sys
from pathlib import Path

import figures
import tamsd
import regions
import terrain

HERE = Path(__file__).parent
STRATUM_TOKEN = {"all": "All", "plains": "Plains", "hills": "Hills", "low mountains": "Low", "high mountains": "High",
                 "Alps": "Alps", "Pyrenees": "Pyr", "Channel Coast": "Channel", "Champagne-Lorraine": "Champagne"}
DESIGN_TOKEN = {"A_fixed": "Afixed", "A_longest": "Alongest", "B_to_1000": "Bkilo", "B_to_longest": "Blongest"}
DESIGN_ROW = {"A_fixed": "A, $T=1000$ s", "A_longest": "A, longest $T$", "B_to_1000": "B, to 1000 s",
              "B_to_longest": "B, to longest $T$"}
D = ("para", "hang")


def signed(x):
    return f"{round(x, 3) + 0.0:+.3f}".replace("-", "$-$").replace("+", "$+$")


def bracket(b, fmt=lambda v: f"{v:.3f}"):
    return f"[{fmt(b[0])}, {fmt(b[1])}]"


def fitted(r):
    return "skipped" not in r


def designs(r):
    return [k for k in DESIGN_TOKEN if k in r]


def thousands(n):
    return f"{n:,}".replace(",", "\\,")


class Macros:
    def __init__(self):
        self.lines = []

    def add(self, name, value):
        self.lines.append(f"\\newcommand{{\\Uni{name}}}{{{value}}}")

    def pair(self, prefix, r, names, token):
        """Macros of one pair of populations over its designs: difference with its band, T, and for the first and
        second population H, its band, flights and pilots."""
        for k in designs(r):
            tok = DESIGN_TOKEN[k]
            self.add(f"{prefix}DH{token}{tok}", signed(r[k]["d_H"]["value"]))
            self.add(f"{prefix}DHBand{token}{tok}", "[" + ", ".join(signed(v) for v in r[k]["d_H"]["band"]) + "]")
            self.add(f"{prefix}T{token}{tok}", str(r[k]["T"]))
            lags = r[k][names[0]]["lags"]
            for name, lag in (("Hundred", 100), ("Thousand", 1000)):
                if lag in lags:
                    self.add(f"{prefix}DAlpha{token}{tok}{name}", signed(r[k]["d_alpha"]["value"][lags.index(lag)]))
            for tag, n in zip(("First", "Second"), names):
                c = r[k][n]
                self.add(f"{prefix}H{tag}{token}{tok}", f"{c['H']:.3f}")
                self.add(f"{prefix}HBand{tag}{token}{tok}", bracket(c["H_band"]))
                self.add(f"{prefix}N{tag}{token}{tok}", thousands(c["flights"][-1]))
                self.add(f"{prefix}P{tag}{token}{tok}", thousands(c["pilots"][-1]))


def table(path, header, rows, colspec, caption, label, note=None):
    body = "\n".join(" & ".join(r) + r" \\" if r != ["\\midrule"] else r[0] for r in rows)
    text = (f"\\begin{{table}}\n\\centering\n\\small\n\\renewcommand{{\\arraystretch}}{{1.15}}\\setlength{{\\tabcolsep}}{{4pt}}\n\\begin{{adjustbox}}{{max width=\\textwidth}}\n\\begin{{tabular}}{{{colspec}}}\n\\toprule\n{header} \\\\\n\\midrule\n"
            f"{body}\n\\bottomrule\n\\end{{tabular}}\n\\end{{adjustbox}}\n\\caption{{{caption}}}\\label{{{label}}}\n\\end{{table}}\n")
    path.write_text(text)


def stack(value, band):
    """A value over its band, in two lines of a table cell."""
    return rf"\shortstack{{{value}\\[-1pt]{{\scriptsize {band}}}}}"


def cell(c):
    return f"{thousands(c['flights'])} ({thousands(c['pilots'])})"


def counts_tables(res, out):
    t, cells = res["terrain"], res["terrain"]["cells"]
    keys = list(terrain.CLASSES) + ["unknown", "all"]
    table(out / "table_counts.tex", " & ".join(["Discipline"] + [k.capitalize() for k in keys]),
          [[d] + [cell(cells[d][k]) for k in keys] for d in D], "l" + "r" * len(keys),
          "Flights (pilots) per discipline and terrain class. A pilot who flew in two classes counts in both; in the "
          "last column, once.", "tab:counts")
    reg = res["terrain"]["regions"]
    min_cell = res["constants"]["min_cell"]
    table(out / "table_region_counts.tex", " & ".join(["Discipline"] + list(regions.BOXES)),
          [[d] + [cell(reg[d][r]) + ("" if reg[d][r]["flights"] >= min_cell else "$^\\dagger$") for r in regions.BOXES] for d in D],
          "l" + "r" * len(regions.BOXES),
          f"Flights (pilots) per discipline and region. $^\\dagger$ fewer than {min_cell} flights: not fitted.", "tab:regions-counts")
    f = res["funnel"]
    steps = [("run table", "all flights of the run table"), ("eligible", "logger interval $\\le$ 10 s, grid positions, a departure"),
             ("declared open", "eligible and declared open"), ("open", "declared open and opening $>$ 0.1"),
             ("known terrain", "open with at least half of the points on the DEM")]
    table(out / "table_funnel.tex", "Step & Para & Hang", [[desc, thousands(f["para"][k]), thousands(f["hang"][k])] for k, desc in steps]
          + [["pilots of the open flights", thousands(f["para"]["pilots"]), thousands(f["hang"]["pilots"])]], "lrr",
          f"Flights at each step of the selection. {f['pilots in both']} pilots have open flights in both disciplines.", "tab:funnel")


def design_row(label, r, k):
    d = r[k]
    first, second = (d[n] for n in list(d)[:2])
    both = lambda c: cell({"flights": c["flights"][-1], "pilots": c["pilots"][-1]})
    return [f"{label} ({d['T']} s)" if k.endswith("longest") else label, both(first), both(second),
            stack(f"{first['H']:.3f}", bracket(first["H_band"])), stack(f"{second['H']:.3f}", bracket(second["H_band"])),
            stack(signed(d["d_H"]["value"]), bracket(d["d_H"]["band"], signed))]


def h_table(path, entries, first, second, caption, label, extra=("Population",)):
    """entries: [(label, pair dict, design keys)]; a skipped pair gets a line of its own."""
    header = f"{extra[0]} & flights (pilots) {first} & flights (pilots) {second} & $H$ {first} & $H$ {second} & $\\Delta H$"
    rows = []
    for name, r, keys in entries:
        if not fitted(r):
            rows.append([name, "", "", "", "", "not fitted: " + ", ".join(f"{k} {v}" for k, v in r["flights"].items())])
            continue
        for k in keys:
            if k in r:
                rows.append(design_row(f"{name}, {DESIGN_ROW[k]}", r, k))
        rows.append(["\\midrule"])
    table(path, header, rows[:-1] if rows and rows[-1] == ["\\midrule"] else rows, "lrrccc", caption, label)


def make_figures(res, out):
    D_ = list(D)
    strata = {s: r for s, r in res["strata"].items() if s != "all" and fitted(r)}
    raw = res["strata"]["all"]
    figures.design_sketch(out / "design_sketch.pdf")
    figures.opening_sketch(out / "opening_sketch.pdf")
    figures.histogram_figure(res, out / "terrain_hist.pdf")
    if (out / "example_flight.npz").exists():
        figures.example_figure(res, out / "example_flight.npz", out / "terrain_rule.pdf")
    figures.map_figure(res, out / "takeoffs.npz", out / "map.pdf")
    figures.msd_figure(res, out / "msd_raw.pdf")
    figures.design_figure([(f"$T={raw[k]['T']}$ s", raw[k]) for k in ("A_fixed", "A_longest")], D_, out / "raw_A.pdf", "A_fixed", 2)
    figures.design_figure([("all terrain", raw["B"])], D_, out / "raw_B.pdf", "B", 1, counts=True)
    figures.design_figure([(f"{s}, $T={r['A_fixed']['T']}$ s", r["A_fixed"]) for s, r in strata.items()], D_, out / "strat_A_fixed.pdf",
                          "A_fixed", 2)
    figures.design_figure([(f"{s}, $T={r['A_longest']['T']}$ s", r["A_longest"]) for s, r in strata.items() if "A_longest" in r], D_,
                          out / "strat_A_longest.pdf", "A_longest", 2)
    figures.design_figure([(s, r["B"]) for s, r in strata.items()], D_, out / "strat_B.pdf", "B", 2, counts=True)
    pvh = res["regions"]["para_vs_hang"]
    cols = [(f"{r}, {t}", v[k]) for r, v in pvh.items() if fitted(v) for k, t in (("A_fixed", "A, 1000 s"), ("A_longest", "A, longest $T$"), ("B", "B"))]
    figures.design_figure(cols, D_, out / "regions_para_hang.pdf", "A_fixed", 3)
    figures.region_figure(res, out / "regions_between.pdf")
    figures.forest_figure(res, out / "forest.pdf")
    figures.fixed_vs_all_figure(res, out / "fixed_vs_all.pdf")


def main(run):
    res = json.loads((HERE / "out" / run / "results.json").read_text())
    out, gen = HERE / "out" / run, HERE / "generated"
    gen.mkdir(exist_ok=True)
    m = Macros()
    c = res["constants"]
    for name, v in (("MinOpening", c["min_opening"]), ("TauFixed", c["tau_fixed"]), ("Support", c["support"]),
                    ("NBoot", c["n_boot"]), ("SiteKm", f"{c['site_m'] / 1000:g}"), ("MinFlights", c["min_flights"]),
                    ("MinCell", c["min_cell"]), ("MinOnDem", f"{c['min_on_dem']:.0%}".replace("%", "\\%")),
                    ("Blocks", thousands(c["blocks"])), ("Cells", thousands(c["cells"]))):
        m.add(name, str(v))
    m.add("GridS", str(tamsd.GRID_S))
    m.add("HMax", str(tamsd.H_MAX))
    m.add("PieceFixed", f"{tamsd.TAU_FIXED / tamsd.SUPPORT:g}")
    m.add("Shortest", f"{tamsd.SHORTEST:g}")
    m.add("NLags", str(len(tamsd.LONG_LAGS)))
    m.add("MaxLag", thousands(int(tamsd.LONG_LAGS[-1])))
    m.add("FirstLag", str(tamsd.LONG_LAGS[0]))
    for name, e in zip(("EdgePlains", "EdgeHills", "EdgeLow"), c["edges_m"]):
        m.add(name, f"{e:g}")
    for d in D:
        for k, v in res["funnel"][d].items():
            m.add(f"Funnel{d.capitalize()}{k.title().replace(' ', '')}", thousands(v))
        m.add(f"Median{d.capitalize()}", f"{res['terrain']['cells'][d]['median']:.0f}")
        for k, v in res["terrain"]["cells"][d].items():
            if isinstance(v, dict):
                tok = STRATUM_TOKEN.get(k, k.capitalize())
                m.add(f"Cell{d.capitalize()}{tok}Flights", thousands(v["flights"]))
                m.add(f"Cell{d.capitalize()}{tok}Pilots", thousands(v["pilots"]))
        for r, v in res["terrain"]["regions"][d].items():
            m.add(f"Region{d.capitalize()}{STRATUM_TOKEN[r]}Flights", thousands(v["flights"]))
    m.add("PilotsBoth", str(res["funnel"]["pilots in both"]))
    for s, r in res["strata"].items():
        if fitted(r):
            m.pair("", r, D, STRATUM_TOKEN[s])
    for s, r in res["regions"]["para_vs_hang"].items():
        if fitted(r):
            m.pair("Reg", r, D, STRATUM_TOKEN[s])
    for d, pairs in res["regions"]["between"].items():
        for name, r in pairs.items():
            if fitted(r):
                a, b = (STRATUM_TOKEN[x] for x in name.split(" | "))
                m.pair(f"Between{d.capitalize()}", r, name.split(" | "), a + b)
    cells = [r[k]["d_H"] for r in list(res["strata"].values()) + list(res["regions"]["para_vs_hang"].values()) if fitted(r)
             for k in DESIGN_TOKEN if k in r]
    m.add("ForestN", str(len(cells)))
    m.add("ForestPos", str(sum(c["band"][0] > 0 for c in cells)))
    m.add("ForestMin", signed(min(c["value"] for c in cells)))
    m.add("ForestMax", signed(max(c["value"] for c in cells)))
    m.add("ForestAllPositive", str(sum(c["value"] > 0 for c in cells)))
    both = [r for r in list(res["strata"].values()) + list(res["regions"]["para_vs_hang"].values()) if fitted(r)]
    gap = lambda a, b: max(abs(r[a]["d_H"]["value"] - r[b]["d_H"]["value"]) for r in both if a in r and b in r)
    m.add("GapKilo", f"{gap('A_fixed', 'B_to_1000'):.3f}")
    m.add("GapLongest", f"{gap('A_longest', 'B_to_longest'):.3f}")
    drops = [r["A_longest"][d]["H"] - r["B_to_longest"][d]["H"] for r in both if "A_longest" in r for d in D]
    m.add("MaxDropB", f"{max(drops):.3f}")
    m.add("MinDropB", f"{round(min(drops), 3) + 0.0:.3f}")
    overlap = all(max(r[a]["d_H"]["band"][0], r[b]["d_H"]["band"][0]) <= min(r[a]["d_H"]["band"][1], r[b]["d_H"]["band"][1])
                  for r in both for a, b in (("A_fixed", "B_to_1000"), ("A_longest", "B_to_longest")) if a in r and b in r)
    assert overlap, "the bands of A and B no longer overlap: reword section fixed_vs_all"
    (gen / "macros.tex").write_text("% Written by report.py; do not edit.\n" + "\n".join(m.lines) + "\n")

    counts_tables(res, gen)
    # raw
    h_table(gen / "table_raw.tex", [("all terrain", res["strata"]["all"], list(DESIGN_TOKEN))], "para", "hang",
            "H of paragliders and hang gliders, all terrain: design A (fixed population, $T$ shown) and B (all flights); "
            "flights (pilots) at the longest lag of the design; bands 90\\%, difference para $-$ hang.", "tab:raw")
    strata = {s: r for s, r in res["strata"].items() if s != "all"}
    h_table(gen / "table_strat_A.tex", [(s, r, ["A_fixed", "A_longest"]) for s, r in strata.items()], "para", "hang",
            "As Table~\\ref{tab:raw}, design A per terrain class.", "tab:strat-A")
    h_table(gen / "table_strat_B.tex", [(s, r, ["B_to_1000", "B_to_longest"]) for s, r in strata.items()], "para", "hang",
            "As Table~\\ref{tab:raw}, design B per terrain class.", "tab:strat-B")
    pvh = res["regions"]["para_vs_hang"]
    h_table(gen / "table_regions.tex", [(r, v, list(DESIGN_TOKEN)) for r, v in pvh.items()], "para", "hang",
            "H of paragliders and hang gliders per region, as Table~\\ref{tab:raw}.", "tab:regions")
    rows, unfit = [], []
    for d in D:
        for name, r in res["regions"]["between"][d].items():
            if fitted(r):
                rows.append([d, name.replace(" | ", " -- ")] + [stack(signed(r[k]["d_H"]["value"]), bracket(r[k]["d_H"]["band"], signed))
                                                              if k in r else "" for k in DESIGN_TOKEN])
            else:
                unfit.append(f"{d}, {name.replace(' | ', ' -- ')}")
    table(gen / "table_between.tex", "Discipline & Regions & " + " & ".join(DESIGN_ROW[k] for k in DESIGN_TOKEN), rows, "llcccc",
          "Difference of $H$ between two regions, first minus second, within a discipline, with 90\\% bands. Not fitted (a cell "
          f"with fewer than {res['constants']['min_cell']} flights): all pairs of hang gliders ({len(unfit)}).", "tab:between")
    # fixed against all flights
    pops = [(s, r) for s, r in res["strata"].items()] + [(s, r) for s, r in pvh.items()]
    pops = [(s, r) for s, r in pops if fitted(r)]
    rows = [[s] + [stack(signed(r[k]["d_H"]["value"]), bracket(r[k]["d_H"]["band"], signed)) if k in r else "-" for k in DESIGN_TOKEN]
            for s, r in pops]
    table(gen / "table_fixed_vs_all.tex", "Population & " + " & ".join(DESIGN_ROW[k] for k in DESIGN_TOKEN), rows, "lcccc",
          "Difference of $H$, para $-$ hang, in the four designs: fixed population against all flights, over the same lags "
          "(to 1000 s, and to the longest $T$ of the population).", "tab:fixed-vs-all")
    rows = [[s, str(r["A_longest"]["T"])] + [f"{r[k][d]['H']:.3f}" for d in D for k in ("A_longest", "B_to_longest")]
            for s, r in pops if "A_longest" in r]
    table(gen / "table_fixed_vs_all_H.tex", "Population & $T$ (s) & para, A & para, B & hang, A & hang, B", rows, "lrrrrr",
          "$H$ of each discipline over the lags up to the longest $T$ of the population: fixed population (A) and all flights (B).",
          "tab:fixed-vs-all-H")

    make_figures(res, out)


if __name__ == "__main__":
    main(f"sample{sys.argv[sys.argv.index('--sample') + 1]}" if "--sample" in sys.argv else "full")
