"""The numbers and the tables of the section on the tails of the phase durations (comparisons/sections/tails.tex),
from tail_designs.json and tail.json: vs-solo/generated/comparisons/tail_macros.tex and tail_*_table.tex.

Every macro starts with Tl. Mu is the Hill index of P(X > x) ~ x^-mu above the threshold of S in the raw design;
a band is the two-way 90% band, "se" the error of independent runs, mu / sqrt(n).

    uv run python vs-solo/code/tails.py    # from analyses/groups, after tail_designs.py --open
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[2]))
from openflights import MIN_OPENING  # noqa: E402
from tables import MAIN, OUTS, signed  # noqa: E402
from tables import num as _num  # noqa: E402
from twoway import Z  # noqa: E402

GEN = OUTS.parent / "vs-solo" / "generated" / "comparisons"
OPENED = OUTS / MAIN / f"open_min{MIN_OPENING:g}"
macros = {}


def num(x, digits=0):
    """tables.num with a minus sign that prints in text mode."""
    return ("$-$" if round(x, digits) < 0 else "") + _num(abs(x), digits)

PHASES = {"transition": "", "climb": "Climb", "search": "Search"}
PHASE_TEXT = {"transition": "transitions", "climb": "climbs", "search": "searches ending in a climb"}
RAW = ("S", "A", "G")
ROUTE = {"G": "G", "S|G": "SG", "A": "A", "S|A": "SA"}
FLIGHT = {"before": "Before", "in group": "In", "after": "After", "between": "Between", "spanning": "Spanning"}
RAW_GAPS = {"G - S": "GS", "A - S": "AS", "G - A": "GA"}
ROUTE_GAPS = {"G - S|G": "G", "A - S|A": "A", "(G - S|G) - (A - S|A)": "D"}
FLIGHT_GAPS = {"in group - mean(before, after)": "In", "after - before": "AfterBefore"}
VARIANTS = {"whole in stretch": "", "spanning by its start": "Start", "stretches of 2 xmin": "Long"}
DESC = {"path (km)": ("Path", 1), "net displacement (km)": ("Net", 1), "straightness": ("Straight", 2),
        "ground speed (km/h)": ("Speed", 1), "vertical speed (m/s)": ("Vs", 2), "height lost (m/km)": ("Sink", 0),
        "time rising (share)": ("Rise", 2), "height lost, total (m)": ("Height", 0)}
DESC_TEXT = {"path (km)": "path (km)", "straightness": "straightness", "vertical speed (m/s)": "vertical speed (m/s)",
             "height lost (m/km)": "height lost per km (m/km)", "time rising (share)": "time rising (share)",
             "height lost, total (m)": "height lost (m)"}
GAP_TEX = {"G - S": r"$\G-\Sl$", "A - S": r"$\A-\Sl$", "G - A": r"$\G-\A$", "G - S|G": r"$\G-\SG$", "A - S|A": r"$\A-\SAA$",
           "(G - S|G) - (A - S|A)": r"$(\G-\SG)-(\A-\SAA)$", "in group - mean(before, after)": r"in group $-$ mean(before, after)",
           "after - before": r"after $-$ before"}
LABELS = {"S": r"$\Sl$", "A": r"$\A$", "G": r"$\G$", "S|G": r"$\SG$", "S|A": r"$\SAA$", "before": "before", "in group": "in group",
          "after": "after", "between": "between", "spanning": "spanning"}


def macro(name, value):
    macros["Tl" + name] = value


def band(e, digits=2, sign=False):
    f = signed if sign else num
    lo, hi = e["band"]
    return "---" if lo is None or hi is None else f"[{f(lo, digits)}, {f(hi, digits)}]"


def value(e, digits=2, sign=False):
    return (signed if sign else num)(e["value"], digits)


def sd_two_way(e):
    return np.nan if None in e["band"] else (e["band"][1] - e["band"][0]) / (2 * Z)


def put(name, e, digits=2, sign=False):
    """macros `name` (value), `name`Band, and, when the entry has them, `name`N and `name`Se (iid)."""
    if e is None or e["value"] is None:
        return
    macro(name, value(e, digits, sign))
    macro(name + "Band", band(e, digits, sign))
    if e.get("n") is not None:
        macro(name + "N", num(e["n"], 0))
    if e.get("se_iid") is not None:
        macro(name + "Se", num(e["se_iid"], 2))
        if np.isfinite(sd_two_way(e)):
            macro(name + "Sd", num(sd_two_way(e), 2))
            macro(name + "Infl", num(sd_two_way(e) / e["se_iid"], 1))


def cell(e, sign=False):
    """The table cells of an entry: mu, band, iid error, error of the band."""
    if e is None or e["value"] is None:
        return ["", "", "", ""]
    sd = sd_two_way(e)
    return [value(e, 2, sign), band(e, 2, sign), num(e["se_iid"], 2) if e["se_iid"] is not None else "", num(sd, 2) if np.isfinite(sd) else "---"]


def ns(e):
    return "" if e is None or e.get("n") is None else num(e["n"], 0)


def table(name, header, rows, spec):
    lines = [rf"\begin{{tabular}}{{{spec}}}", r"\toprule", header, r"\midrule"]
    for row in rows:
        lines.append(r"\addlinespace[2pt]\multicolumn{1}{@{}l}{\emph{" + row[1:] + r"}}\\" if isinstance(row, str) and row[0] == "#"
                     else (" & ".join(row) + r" \\") if not isinstance(row, str) else row)
    (GEN / f"{name}.tex").write_text("\n".join(lines + [r"\bottomrule", r"\end{tabular}"]) + "\n")


def main_table(d):
    """The key numbers: mu of the transitions at the threshold, in every population and class, with the contrasts."""
    raw, route, flight = d["raw"]["transition"], d["same route"]["transition"], d["same flight"]["transition"]["variants"]["whole in stretch"]
    rows = ["#raw: group time, the same flights alone, solo flights"]
    rows += [[LABELS[s], ns(raw["populations"][s]["at xmin of S"]), *cell(raw["populations"][s]["at xmin of S"])] for s in RAW]
    rows += [[GAP_TEX[g], "", *cell(raw["gaps"][g], True)] for g in RAW_GAPS]
    rows.append("#same route: pieces and their controls")
    rows += [[LABELS[s], ns(route["populations"][s]), *cell(route["populations"][s])] for s in ROUTE]
    rows += [[GAP_TEX[g], "", *cell(route["contrasts"][g], True)] for g in ROUTE_GAPS_ORDER]
    rows.append("#same flight: runs by their place in the flight")
    rows += [[LABELS[c], ns(flight["classes"][c]), *cell(flight["classes"][c])] for c in FLIGHT]
    rows += [[GAP_TEX[k], "", *cell(flight["contrasts"][k], True)] for k in FLIGHT_GAPS]
    table("tail_table", r"population or class & runs & $\hat\mu$ & 90\% band & s.e. iid & s.e. two-way \\",
          [r if not isinstance(r, str) else r for r in rows], r"@{}l r r l r r@{}")


ROUTE_GAPS_ORDER = ("G - S|G", "A - S|A", "(G - S|G) - (A - S|A)")


def variants_table(d):
    """The classes of the same-flight design in its three variants."""
    v = d["same flight"]["transition"]["variants"]
    rows = []
    for name, text in (("whole in stretch", "a run in the class when it lies whole in its stretch"),
                       ("spanning by its start", "a run that crosses a stretch goes to the class of the stretch where it starts"),
                       ("stretches of 2 xmin", "only the stretches of at least twice the threshold")):
        rows.append("#" + text)
        for c in ("before", "in group", "after", "between", "spanning"):
            e = v[name]["classes"][c]
            if e["value"] is not None:
                rows.append([LABELS[c], ns(e), *cell(e)])
        for k, text_k in (("in group - mean(before, after)", "in group $-$ mean(before, after)"), ("after - before", "after $-$ before")):
            rows.append([GAP_TEX[k], "", *cell(v[name]["contrasts"][k], True)])
    table("tail_variants_table", r"class & runs & $\hat\mu$ & 90\% band & s.e. iid & s.e. two-way \\", rows, r"@{}l r r l r r@{}")


def lengths_table(d):
    """The lengths of the stretches by class."""
    f = d["same flight"]
    rows = []
    for c in ("before", "in group", "after", "between"):
        t = f["stretch lengths"][c]
        q = t["length quantiles (s)"]
        rows.append([LABELS[c], num(t["stretches"], 0), *[num(q[k], 0) for k in ("10", "25", "50", "75", "90")],
                     num(100 * t["share >= xmin"], 0), num(100 * t["share >= 2 xmin"], 0),
                     num(f["transition"]["other"][c]["runs above"], 0)])
    table("tail_lengths_table", r"class & stretches & 10\% & 25\% & 50\% & 75\% & 90\% & $\ge x_{\min}$ (\%) & $\ge 2x_{\min}$ (\%) & runs above \\",
          rows, r"@{}l r r r r r r r r r@{}")


def descriptors_table(d):
    """The medians of the descriptors of the long glides, by design, population and class."""
    keys = ("path (km)", "straightness", "vertical speed (m/s)", "height lost (m/km)", "time rising (share)", "height lost, total (m)")
    head = "population or class & glides & " + " & ".join(DESC_TEXT[k] for k in keys) + r" & last run (\%) \\"
    blocks = (("raw", "raw", RAW), ("same route", "same route", tuple(ROUTE)), ("same flight", "same flight", tuple(FLIGHT)))
    rows = []
    for design, text, names in blocks:
        rows.append("#" + text)
        desc = d[design]["transition"]["descriptors"]
        for s in names:
            r = desc[s]
            rows.append([LABELS[s], num(r["runs"], 0), *[num(r[k]["median"], DESC[k][1]) for k in keys],
                         num(100 * r["last run of the flight (share)"], 0)])
    table("tail_descriptors_table", head, rows, r"@{}l r r r r r r r r@{}")


def phases_table(d):
    """mu of climbs and searches in every population and class."""
    rows = []
    for phase in ("climb", "search"):
        rows.append("#" + PHASE_TEXT[phase])
        raw, route, flight = d["raw"][phase], d["same route"][phase], d["same flight"][phase]["variants"]["whole in stretch"]
        for s in RAW:
            e = raw["populations"][s]["at xmin of S"]
            rows.append([f"raw {LABELS[s]}", ns(e), *cell(e)])
        for g in RAW_GAPS:
            rows.append(["raw " + GAP_TEX[g], "", *cell(raw["gaps"][g], True)])
        for s in ROUTE:
            rows.append([f"route {LABELS[s]}", ns(route["populations"][s]), *cell(route["populations"][s])])
        for c in ("before", "in group", "after"):
            rows.append([f"flight {LABELS[c]}", ns(flight["classes"][c]), *cell(flight["classes"][c])])
        for k in FLIGHT_GAPS:
            rows.append([GAP_TEX[k], "", *cell(flight["contrasts"][k], True)])
    table("tail_phases_table", r"population or class & runs & $\hat\mu$ & 90\% band & s.e. iid & s.e. two-way \\", rows, r"@{}l r r l r r@{}")


def threshold_macros(d, old):
    macro("Xmin", num(d["xmin of S"]["transition"], 0))
    macro("XminClimb", num(d["xmin of S"]["climb"], 0))
    macro("XminSearch", num(d["xmin of S"]["search"], 0))
    macro("Factor", num(d["stretch factor"], 0))
    macro("XminLong", num(d["stretch factor"] * d["xmin of S"]["transition"], 0))
    macro("NBoot", num(d["n_boot"], 0))
    macro("MinTail", "50")
    for s in RAW:
        macro(f"KsXmin{s}", num(d["raw"]["transition"]["populations"][s]["xmin by KS"], 0))
        r = d["raw"]["transition"]["populations"][s]
        macro(f"Runs{s}", num(r["runs"], 0))
    for c, tag in FLIGHT.items():
        macro(f"KsXminFlight{tag}", num(d["same flight"]["transition"]["curves"][c]["xmin by KS"], 0))
    ks = d["raw"]["transition"]["populations"]["S"]["ks"]
    i = int(np.nanargmin(np.array(ks["D"], float)))
    macro("KsDistanceS", num(ks["D"][i], 3))
    macro("KsGridMu", num(ks["mu"][i], 2))
    macro("CheckDifference", f"{d['check against tail.json']['largest difference']:.0e}")


def design_macros(d):
    for phase, tag in PHASES.items():
        raw, route, flight = d["raw"][phase], d["same route"][phase], d["same flight"][phase]
        for s in RAW:
            put(f"Raw{tag}Mu{s}", raw["populations"][s]["at xmin of S"])
        for g, t in RAW_GAPS.items():
            put(f"Raw{tag}Gap{t}", raw["gaps"][g], sign=True)
        for s, t in ROUTE.items():
            put(f"Route{tag}Mu{t}", route["populations"][s])
        for g, t in ROUTE_GAPS.items():
            put(f"Route{tag}MuCtl{t}", route["contrasts"][g], sign=True)
        for variant, vt in VARIANTS.items():
            if variant not in flight["variants"]:
                continue
            v = flight["variants"][variant]
            for c, t in FLIGHT.items():
                put(f"Flight{tag}{vt}Mu{t}", v["classes"][c])
            for g, t in FLIGHT_GAPS.items():
                put(f"Flight{tag}{vt}Delta{t}", v["contrasts"][g], sign=True)
    r = d["same route"]["transition"]
    for s, t in (("G", "G"), ("A", "A")):
        macro(f"RoutePieces{t}", num(r["pieces"][s], 0))
        macro(f"RouteControls{t}", num(r["controls"][s], 0))


def flight_macros(d):
    f = d["same flight"]
    macro("FlightFlights", num(f["flights"], 0))
    for c, t in FLIGHT.items():
        if c in f["stretch lengths"]:
            s = f["stretch lengths"][c]
            macro(f"Stretches{t}", num(s["stretches"], 0))
            macro(f"StretchMedian{t}", num(s["length quantiles (s)"]["50"], 0))
            macro(f"StretchShortShare{t}", num(100 * s["share >= xmin"], 0))
            macro(f"StretchLongShare{t}", num(100 * s["share >= 2 xmin"], 0))
            macro(f"StretchLongN{t}", num(s["stretches >= 2 xmin"], 0))
        o = f["transition"]["other"][c]
        macro(f"Above{t}", num(o["runs above"], 0))
        macro(f"AboveFlights{t}", num(o["flights above"], 0))
    macro("SpanningShare", num(100 * f["transition"]["other"]["spanning"]["runs above"] / sum(o["runs above"] for o in f["transition"]["other"].values()), 0))
    cross = f["transition"]["long by class and raw population"]
    macro("SpanningInRawG", num(cross["spanning"].get("G", 0), 0))
    macro("SpanningInRawA", num(cross["spanning"].get("A", 0), 0))
    macro("InGroupInRawG", num(cross["in group"].get("G", 0), 0))
    macro("OutsideInRawG", num(cross["outside"].get("G", 0), 0))
    macro("RawGAbove", num(d["raw"]["transition"]["populations"]["G"]["at xmin of S"]["n"], 0))
    tol = f["transition"]["tolerance"]
    macro("TolSpanningZero", num(tol["0 s"]["spanning"], 0))
    macro("TolSpanningTen", num(tol["10 s"]["spanning"], 0))
    macro("TolSpanningThirty", num(tol["30 s"]["spanning"], 0))
    macro("Outside", num(tol["0 s"]["outside"], 0))
    macro("RunsClassified", num(sum(v for k, v in f["classes of the runs"].items() if k != "outside"), 0))


def descriptor_macros(d):
    """The medians of the descriptors (value and band), and their gaps, of the three designs."""
    sets = {"Raw": (d["raw"]["transition"], RAW, RAW_GAPS, {s: s for s in RAW}),
            "Route": (d["same route"]["transition"], tuple(ROUTE), ROUTE_GAPS, ROUTE),
            "Flight": (d["same flight"]["transition"], tuple(FLIGHT), FLIGHT_GAPS, FLIGHT)}
    for design, (part, names, gaps, tags) in sets.items():
        for s in names:
            r = part["descriptors"][s]
            macro(f"Desc{design}Runs{tags[s]}", num(r["runs"], 0))
            macro(f"Desc{design}Last{tags[s]}", num(100 * r["last run of the flight (share)"], 0))
            for k, (t, digits) in DESC.items():
                e = r[k]
                macro(f"Desc{design}{t}{tags[s]}", num(e["median"], digits))
                macro(f"Desc{design}{t}{tags[s]}Band", f"[{num(e['median band'][0], digits)}, {num(e['median band'][1], digits)}]")
        for g, gt in gaps.items():
            for k, (t, digits) in DESC.items():
                e = part["descriptor gaps"].get(g, {}).get(k)
                if e:
                    macro(f"Desc{design}{t}Gap{gt}", signed(e["value"], digits))
                    macro(f"Desc{design}{t}Gap{gt}Band", f"[{signed(e['band'][0], digits)}, {signed(e['band'][1], digits)}]")


def hill_macros(d):
    """Hill index of S at a low and a high threshold of the grid, to say how much the comparison depends on it."""
    for s in RAW:
        c = d["raw"]["transition"]["populations"][s]["hill"]
        k = [i for i, x in enumerate(c["xmin"]) if x >= 1000][0]
        macro(f"HillAtThousand{s}", num(c["value"][k], 2))
        macro(f"HillAtThousand{s}N", num(c["runs"][k], 0))
        k = int(np.argmin(np.abs(np.array(c["xmin"]) - 2500)))
        macro(f"HillAtHigh{s}", num(c["value"][k], 2))
        macro(f"HillAtHigh{s}N", num(c["runs"][k], 0))
        macro("HillHigh", num(c["xmin"][k], 0))
    g = d["raw"]["transition"]["populations"]
    gaps = [g["G"]["hill"]["value"][i] - g["S"]["hill"]["value"][i] for i in range(len(g["S"]["hill"]["xmin"]))]
    macro("GapNegativeFrom", num(min(x for x, v in zip(g["S"]["hill"]["xmin"], gaps) if v < 0 and x >= 800), 0))
    xs = g["S"]["hill"]["xmin"]
    for x, tag in ((600, "SixHundred"), (1000, "Thousand"), (1700, "Threshold"), (2000, "TwoThousand")):
        k = int(np.argmin(np.abs(np.array(xs) - x)))
        macro(f"HillGap{tag}", signed(gaps[k], 2))
        macro(f"HillGap{tag}Xmin", num(xs[k], 0))


if __name__ == "__main__":
    GEN.mkdir(parents=True, exist_ok=True)
    d = json.loads((OPENED / "tail_designs.json").read_text())
    old = json.loads((OPENED / "tail.json").read_text())
    threshold_macros(d, old)
    design_macros(d)
    flight_macros(d)
    descriptor_macros(d)
    hill_macros(d)
    main_table(d)
    variants_table(d)
    lengths_table(d)
    descriptors_table(d)
    phases_table(d)
    (GEN / "tail_macros.tex").write_text("\n".join(rf"\newcommand{{\{k}}}{{{v}}}" for k, v in sorted(macros.items())) + "\n")
    print(f"{len(macros)} macros")
