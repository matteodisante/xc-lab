"""The tables and the numbers of comparisons/, from the JSON of the analyses: vs-solo/generated/comparisons/*.tex.

Every number quoted in the text of the report is a macro of generated/comparisons/macros.tex. The analyses run
on the declared open flights (out/<run>/open_declared) and on the open ones (out/<run>/open_min<m>); the macros
and tables of the story come from out/<run>/open_declared_dt5 (the flights with a logger interval up to 5 s).

    uv run python vs-solo/code/comparisons.py    # from analyses/groups, after opening.py and longlags.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[2]))
import closure  # noqa: E402
import cohorts  # noqa: E402
import longlags  # noqa: E402
import tail_ks  # noqa: E402
import tamsd  # noqa: E402
import twoway  # noqa: E402
from tables import MAIN, OUTS, macro, macros, num  # noqa: E402

GEN = OUTS.parent / "vs-solo" / "generated" / "comparisons"
DECLARED = OUTS / MAIN / "open_declared"  # the declared open flights, every logger interval
DT5 = OUTS / MAIN / "open_declared_dt5"  # the same up to longlags.MAX_DT s: the flights of the story
POPS = (("solo", "S"), ("group", "G"), ("group_solo", "A"))
GLOSS = {"Dist libre": "free distance", "Dist 1 pt": "distance, 1 turn point", "Dist 2 pts": "distance, 2 turn points",
         "Dist 3 pts": "distance, 3 turn points", "triangle FAI": "FAI triangle", "Quadrilatère": "quadrilateral",
         "Aller-Retour": "out and return", "Marche et Vol": "hike and fly", "Marche et Vol bouclé": "hike and fly, closed",
         "Marche et Vol FAI": "hike and fly, FAI triangle"}


def signed(x, digits):
    """x with its sign in math mode, kept when it rounds to 0 (a band edge of -0.000 is below 0); none for 0."""
    return ("$+$" if x > 0 else "$-$" if x < 0 else "") + num(abs(x), digits)


def read(folder, name):
    return json.loads((folder / f"{name}.json").read_text())


def write(name, lines):
    (GEN / f"{name}.tex").write_text("\n".join(lines) + "\n")


def sig2(x):
    """x to two significant figures: a lag read on a figure."""
    return num(float(f"{x:.2g}"), 0)


def from_lag(lags, ok):
    """The first lag from which `ok` holds at every later lag of `lags`; None when it fails at the last lag."""
    ok = np.asarray(ok, bool)
    i = int(np.flatnonzero(~ok)[-1]) + 1 if (~ok).any() else 0
    return lags[i] if i < len(lags) else None


def first_lag(lags, ok):
    """The first lag where `ok` holds; None when it never does."""
    return next((t for t, k in zip(lags, ok) if k), None)


def labels_tex(o):
    """The labels of the catalogue with their class and flights."""
    lines = [r"\begin{tabular}{@{}l l l r@{}}", r"\toprule", r"label & meaning & class & flights \\", r"\midrule"]
    for cls in ("open", "closed", "neither"):
        for row in (r for r in o["labels"] if r["class"] == cls):
            lines.append(f"{row['label']} & {GLOSS.get(row['label'], row['label'])} & {cls} & {num(row['flights'], 0)} \\\\")
    write("labels_table", lines + [r"\bottomrule", r"\end{tabular}"])
    total = {cls: sum(r["flights"] for r in o["labels"] if r["class"] == cls) for cls in ("open", "closed", "neither")}
    macro("FlightsAll", num(sum(total.values()), 0))
    for cls, tag in (("open", "Open"), ("closed", "Closed"), ("neither", "Neither")):
        macro(f"Flights{tag}", num(total[cls], 0))


def pct(b, key):
    """Share of the flights of b in key, from the counts: the "(%)" of opening.json is rounded already."""
    return 100 * b[key] / b["flights"]


def opening_tex(o):
    """Macros of opening.py and the table of the openings of the datasets of the declared open flights."""
    macro("MinOpening", f"{o['min_opening']:g}")
    macro("Stricter", f"{o['stricter']:g}")
    macro("MountainM", num(o["mountain_m"], 0))
    every = o["datasets"]["every declared open flight"]["all"]
    macro("Kept", num(every["above the threshold"], 0))
    macro("KeptShare", num(pct(every, "above the threshold"), 0))
    macro("StrictShare", num(pct(every, "above the stricter threshold"), 0))
    for name, tag in ((f"below {o['mountain_m']} m", "Low"), (f"{o['mountain_m']} m or more", "High")):
        b = o["by takeoff"][name]
        macro(f"Takeoff{tag}Flights", num(b["flights"], 0))
        macro(f"Takeoff{tag}Median", f"{b['quartiles'][1]:.2f}")
        macro(f"Takeoff{tag}Keep", num(pct(b, "above the threshold"), 0))
        macro(f"Takeoff{tag}KeepN", num(b["above the threshold"], 0))
        macro(f"Takeoff{tag}StrictN", num(b["above the stricter threshold"], 0))
        macro(f"Takeoff{tag}Strict", num(pct(b, "above the stricter threshold"), 0))
    macro("DurationS", num(o["duration_s"], 0))
    for name, tag in ((f"below {o['duration_s']} s", "Short"), (f"{o['duration_s']} s or more", "Long")):
        b = o["by duration"][name]
        macro(f"Duration{tag}Flights", num(b["flights"], 0))
        macro(f"Duration{tag}Median", f"{b['quartiles'][1]:.2f}")
        macro(f"Duration{tag}Keep", num(pct(b, "above the threshold"), 0))
        macro(f"Duration{tag}KeepN", num(b["above the threshold"], 0))
        macro(f"Duration{tag}StrictN", num(b["above the stricter threshold"], 0))
        macro(f"Duration{tag}Strict", num(pct(b, "above the stricter threshold"), 0))
    strata = dict(zip(o["by kind"]["solo flights"]["median by stratum"], ("Low", "High", "Short", "Long")))
    for kind, tag in (("solo flights", "Solo"), ("group flights", "Group")):
        b = o["by kind"][kind]
        macro(f"Kind{tag}Flights", num(b["flights"], 0))
        macro(f"Kind{tag}Median", f"{b['quartiles'][1]:.2f}")
        macro(f"Kind{tag}Keep", num(pct(b, "above the threshold"), 0))
        macro(f"Kind{tag}Strict", num(pct(b, "above the stricter threshold"), 0))
        macro(f"Kind{tag}Duration", num(b["median duration (s)"], 0))
        macro(f"Kind{tag}Mountain", num(b["mountain (%)"], 0))
        for name, stag in strata.items():
            macro(f"Kind{tag}Median{stag}", f"{b['median by stratum'][name]:.2f}")
    tags = dict(zip(o["datasets"], ("Every", "Fixed", "Longest")))
    lines = [r"\begin{tabular}{@{}l r r r r r r r r@{}}", r"\toprule",
             r" & \multicolumn{4}{c}{flights} & \multicolumn{4}{c}{median opening} \\",
             r"\cmidrule(lr){2-5}\cmidrule(l){6-9}", r"dataset & all & $\Sl$ & $\G$ & $\A$ & all & $\Sl$ & $\G$ & $\A$ \\", r"\midrule"]
    for name, d in o["datasets"].items():
        label = name if not name.startswith("lags up to ") else f"lags up to {num(int(name.split()[3]), 0)}~s"
        macro(f"Median{tags[name]}", f"{d['all']['quartiles'][1]:.2f}")
        macro(f"Flights{tags[name]}", num(d["all"]["flights"], 0))
        macro(f"Below{tags[name]}", num(100 - pct(d["all"], "above the threshold"), 0))
        cells = [num(d[t]["flights"], 0) if t in d else "" for _, t in POPS]
        medians = [f"{d[t]['quartiles'][1]:.2f}" if t in d else "" for _, t in POPS]
        for _, t in POPS:
            if t in d:
                macro(f"Median{tags[name]}{t}", f"{d[t]['quartiles'][1]:.2f}")
        lines.append(f"{label} & {num(d['all']['flights'], 0)} & " + " & ".join(cells) + f" & {d['all']['quartiles'][1]:.2f} & "
                     + " & ".join(medians) + r" \\")
    write("opening_table", lines + [r"\bottomrule", r"\end{tabular}"])


COHORT_TAU = 3600  # s: the cohort whose gaps the story quotes
GAP_LAGS = {10: "Ten", 100: "Hundred", 320: "ThreeTwenty", 1000: "Thousand", 2760: "Long"}


def cohort_gaps_tex(o):
    """GapThr{GS,AS,GA}Lag (and Band): the gaps of alpha of the cohort to COHORT_TAU s at GAP_LAGS, for the
    declared open flights (Thr = All) and the two thresholds on the opening (Min, Strict); GapThr{GS,AS,GA}From:
    the lag from which the lower edge of the band of the gap stays above 0 (two significant figures)."""
    macro("GapCohortTau", num(COHORT_TAU, 0))
    files = {"All": "cohorts", "Min": f"cohorts_min{o['min_opening']:g}", "Strict": f"cohorts_min{o['stricter']:g}"}
    for thr, name in files.items():
        if not (DT5 / f"{name}.json").exists():
            continue
        for gap, entries in read(DT5, name)["gaps"].items():
            c = next(e for e in entries if e["tau_max"] == COHORT_TAU)
            t = from_lag(c["lags"], np.array(c["band"][0]) > 0)
            if t is not None:
                macro(f"Gap{thr}{gap.replace(' - ', '')}From", sig2(t))
            for lag, tag in GAP_LAGS.items():
                k = c["lags"].index(lag)
                mac_value(f"Gap{thr}{gap.replace(' - ', '')}{tag}",
                          {"value": c["value"][k], "band": [c["band"][0][k], c["band"][1][k]]}, 3, True)


SPREAD_LAG = 1000  # s: up to this lag the cohorts are compared with the cohort to COHORT_TAU


def spread_tex(folder):
    """GapSpreadGA: the largest difference of G - A between a cohort and the cohort to COHORT_TAU s, at the lags
    up to SPREAD_LAG they share."""
    cs = {c["tau_max"]: c for c in read(folder, "cohorts")["gaps"]["G - A"]}
    ref = cs[COHORT_TAU]
    d = [abs(c["value"][c["lags"].index(t)] - ref["value"][ref["lags"].index(t)])
         for tau, c in cs.items() if tau != COHORT_TAU for t in c["lags"] if t in ref["lags"] and t <= SPREAD_LAG]
    macro("GapSpreadGA", num(max(d), 3))
    macro("GapSpreadLag", num(SPREAD_LAG, 0))
    pieces = [c["pieces"] for c in read(folder, "cohorts")["alpha"]["group"]]
    macro("CohortPiecesFirst", num(pieces[0], 0))
    macro("CohortPiecesLast", num(pieces[-1], 0))
    # alpha at SPREAD_LAG against the tau_max of the cohort: the step between consecutive cohorts
    steps = {}
    for pop, entries in read(folder, "cohorts")["alpha"].items():
        v = [c["alpha"][c["lags"].index(SPREAD_LAG)] for c in entries]
        steps[pop] = {c["tau_max"]: b - a for c, a, b in zip(entries[1:], v, v[1:])}
    after = [s[7200] for s in steps.values()]  # the cohort to 1 h to the one to 2 h
    before = [x for s in steps.values() for t, x in s.items() if t <= COHORT_TAU]
    macro("AlphaStepHourMin", num(min(after), 3))
    macro("AlphaStepHourMax", num(max(after), 3))
    macro("AlphaStepBeforeMax", num(max(before), 3))
    # the cohorts that reach beyond SPREAD_LAG, against each other: below and above SPREAD_LAG
    for gap, entries in read(folder, "cohorts")["gaps"].items():
        long = [c for c in entries if c["tau_max"] > SPREAD_LAG]
        for side, keep in (("Low", lambda t: t <= SPREAD_LAG), ("High", lambda t: t > SPREAD_LAG)):
            d = [abs(a["value"][a["lags"].index(t)] - b["value"][b["lags"].index(t)])
                 for i, a in enumerate(long) for b in long[i + 1:] for t in a["lags"] if t in b["lags"] and keep(t)]
            macro(f"GapSpread{side}{gap.replace(' - ', '')}", num(max(d), 3))


WINDOW_FULL = 5  # lags: a window of at least this many is the usual one; the ends of the grid hold fewer
POOL_LAGS = {1000: "Thousand", 2760: "Long", 6220: "VeryLong"}


def per_lag_tex():
    """The per-lag curve of cohorts.py on the declared open flights: where it is solid, its last lag, its gap
    from the cohort to SELECTION_FREE, the pooled alpha minus it (the composition term), and its gaps."""
    if not (DT5 / "perlag.json").exists():
        return
    s = read(DT5, "perlag")
    tags = dict(POPS)
    macro("PerLagSelFree", num(s["selection_free"], 0))
    macro("PieceMinS", num(s["min_piece_s"], 0))
    macro("HalfDecade", f"{s['half_decade']:g}")
    macro("Support", f"{s['support']:g}")
    macro("PerLagSolidEnd", num(max(t for t, k in zip(s["lags"], s["solid"]) if k), 0))
    for pop, t in tags.items():
        a = s["alpha"][pop]
        macro(f"PerLag{t}LastLag", num(a["lags"][-1], 0))
        macro(f"PerLag{t}LastPieces", num(a["pieces"][-1], 0))
        macro(f"PerLag{t}FirstPieces", num(a["pieces"][0], 0))
        macro(f"PerLag{t}MinusCohortMax", num(max(abs(v) for v in s["minus cohort"][pop]["value"]), 3))
        d = s["pool minus per-lag"][pop]
        for lag, tag in POOL_LAGS.items():
            if lag in d["lags"]:
                macro(f"Pool{t}Minus{tag}", signed(d["value"][d["lags"].index(lag)], 3))
        macro(f"Pool{t}Max", num(max(s["pool"][pop]["value"]), 2))
    for gap, g in s["gaps"].items():
        for lag, tag in GAP_LAGS.items():
            if lag in s["lags"]:
                mac_value(f"PerLag{gap.replace(' - ', '')}{tag}",
                          {"value": g["value"][s["lags"].index(lag)],
                           "band": [g["band"][0][s["lags"].index(lag)], g["band"][1][s["lags"].index(lag)]]}, 3, True)


def chained_tex():
    """The chained curve of cohorts.py on the declared open flights: where it is solid (the rule of the per-lag
    curve), its last lag and pieces, its gap from the cohort to SELECTION_FREE, the pool minus it, and its gaps."""
    if not (DT5 / "perlag.json").exists():
        return
    s = read(DT5, "perlag")
    macro("ChainSelFree", num(s["selection_free"], 0))
    macro("ChainSolidEnd", num(max(t for t, k in zip(s["lags"], s["solid"]) if k), 0))
    for pop, t in POPS:
        a = s["chained"][pop]
        macro(f"Chain{t}LastLag", num(a["lags"][-1], 0))
        macro(f"Chain{t}LastPieces", num(a["pieces"][-1], 0))
        macro(f"Chain{t}FirstPieces", num(a["pieces"][0], 0))
        macro(f"Chain{t}MinusCohortMax", num(max(abs(v) for v in s["chained minus cohort"][pop]["value"]), 3))
        d = s["pool minus chained"][pop]
        for lag, tag in POOL_LAGS.items():
            if lag in d["lags"]:
                macro(f"Pool{t}MinusChain{tag}", signed(d["value"][d["lags"].index(lag)], 3))
    for gap, g in s["chained gaps"].items():
        for lag, tag in GAP_LAGS.items():
            if lag in g["lags"]:
                i = g["lags"].index(lag)
                mac_value(f"Chain{gap.replace(' - ', '')}{tag}",
                          {"value": g["value"][i], "band": [g["band"][0][i], g["band"][1][i]]}, 3, True)


GROUND_TAGS = {"plains": "Plains", "hills": "Hills", "low mountains": "LowMtn", "high mountains": "HighMtn"}
GROUND_GAPS = {"G - S": "GS", "G - A": "GA", "A - S": "AS"}
GROUND_LAGS = {1000: "Thousand", 320: "ThreeTwenty"}


def gap_at(g, lag):
    """The {value, band} of a gap curve `g` at `lag`."""
    i = g["lags"].index(lag)
    return {"value": g["value"][i], "band": [g["band"][0][i], g["band"][1][i]]}


def dt_tex():
    """The filter of the story: the logger interval it keeps, and the flights of the cache it keeps and drops."""
    d = read(DT5, "longlags")["dt"]
    macro("DtMax", num(d["max_dt"], 0))
    macro("DtFlights", num(d["flights"], 0))
    macro("DtDroppedPercent", num(100 * (1 - d["flights"] / d["of"]), 1))


def ground_tex():
    """contrasts.py (design ground) on the declared open flights up to DtMax s: the constants of the ground class, the
    share of flights of unknown class and of class changed by the trim, the flights and the gaps of each class, the
    peaks of the normalised gaps, the lag from which G - A stays above 0 (Ground<class>GAPositive, two significant
    figures), the share of the flights in the mountains, and the table of the classes."""
    if not (DT5 / "contrasts_ground.json").exists():
        return
    s = read(DT5, "contrasts_ground")
    c = s["constants"]
    macro("GroundTrim", num(100 * c["trim"], 0))
    for k, e in zip(("Low", "Mid", "High"), c["edges_m"]):
        macro(f"GroundEdge{k}", num(e, 0))
    macro("GroundCell", num(c["cell_m"], 0))
    macro("GroundMinOnDem", num(100 * c["min_on_dem"], 0))
    macro("GroundUnknown", num(100 * s["share"]["unknown"], 1))
    macro("GroundChanged", num(100 * s["share"]["changed"], 1))
    for pop, t in POPS:
        total = sum(s[k]["flights"][pop] for k in GROUND_TAGS)
        macro(f"GroundLowland{t}", num(100 * sum(s[k]["flights"][pop] for k in ("plains", "hills")) / total, 0))
        macro(f"GroundMountain{t}", num(100 * sum(s[k]["flights"][pop] for k in ("low mountains", "high mountains")) / total, 0))
    lines = [r"\begin{tabular}{@{}l r r r r r r@{}}", r"\toprule",
             r" & \multicolumn{3}{c}{flights} & \multicolumn{3}{c}{gap at 1\,000~s} \\",
             r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
             r"ground & $\Sl$ & $\A$ & $\G$ & $\G-\Sl$ & $\G-\A$ & $\A-\Sl$ \\", r"\midrule"]
    for k, tag in GROUND_TAGS.items():
        flights = {t: s[k]["flights"][pop] for pop, t in POPS}
        for t in "SAG":
            macro(f"Ground{tag}Flights{t}", num(flights[t], 0))
        row = [k, *(num(flights[t], 0) for t in "SAG")]
        for gap, short in GROUND_GAPS.items():
            g = s[k]["gaps"][gap]
            for lag, lt in GROUND_LAGS.items():
                v = gap_at(g, lag)
                mac_value(f"Ground{tag}{short}{lt}", v, 3, True)
                if lag == 1000:
                    row.append(cell(v, 3, True))
        for gap, short in (("G - S", "GS"), ("G - A", "GA")):  # the peak of the normalised gap
            r = s[k]["relative gaps"][gap]
            i = max((t for t, ok in enumerate(r["solid"]) if ok), key=lambda t: r["value"][t])  # where selection-free
            macro(f"Ground{tag}Peak{short}", num(r["value"][i], 2))
            macro(f"Ground{tag}Peak{short}Lag", num(r["lags"][i], 0))
        g = s[k]["gaps"]["G - A"]  # the lag from which G - A stays resolved above 0, over the solid lags
        solid = [t for t, ok in zip(g["lags"], g["solid"]) if ok]
        t = from_lag(solid, np.array(g["band"][0][:len(solid)]) > 0)
        if t is not None:
            macro(f"Ground{tag}GAPositive", sig2(t))
        lines.append(" & ".join(row) + r" \\")
    write("ground_table", lines + [r"\bottomrule", r"\end{tabular}"])


WEIGHT_PIECES = {"solo": "S", "group_solo": "A", "group": "G"}


def weights_tex():
    """contrasts.py --perflight (design ground) on the declared open flights up to DtMax s: the gaps at 1000 s with one
    weight per flight (Weight<gap><class>Thousand) and their difference from the gaps per piece
    (WeightMinus<gap><class>Thousand, with Band), the largest difference over the solid lags of every gap and class
    (WeightMaxShift) and the mean pieces per flight of S, A, G at the first lag (WeightPiecesPerFlight<S|A|G>)."""
    if not (DT5 / "contrasts_ground_perflight.json").exists():
        return
    s = read(DT5, "contrasts_ground_perflight")
    shift = 0.0
    for k, cls in WHEN_TAGS.items():
        for gap, short in GROUND_GAPS.items():
            mac_value(f"Weight{short}{cls}Thousand", gap_at(s[k]["gaps"][gap], 1000), 3, True)
            mac_value(f"WeightMinus{short}{cls}Thousand", gap_at(s[k]["gaps minus per piece"][gap], 1000), 3, True)
            d = s[k]["gaps minus per piece"][gap]
            shift = max(shift, max(abs(v) for v, ok in zip(d["value"], d["solid"]) if ok))
    macro("WeightMaxShift", num(shift, 3))
    for pop, t in WEIGHT_PIECES.items():
        macro(f"WeightPiecesPerFlight{t}", num(s["all"]["pieces"][pop] / s["all"]["flights"][pop], 2))


def carry_tex():
    """carry.py on the declared open flights up to DtMax s: A(d) - S in the nearest and the farthest bin of the distance
    from the group time (Carry<class><Near|Far>, with Band), the bins (CarryNearLabel, CarryFarLabel, CarryBins) and the
    table carry_table: A(d) - S per class and bin, then the windows."""
    if not (DT5 / "carry.json").exists():
        return
    s = read(DT5, "carry")
    n = len(s["labels"])
    macro("CarryBins", num(n, 0))
    macro("CarryNearLabel", s["labels"][0])
    macro("CarryFarLabel", s["labels"][-1])
    lines = [r"\begin{tabular}{@{}l r " + "r" * n + "@{}}", r"\toprule", r"ground & $\Sl$ & " + " & ".join(s["labels"]) + r" \\", r"\midrule",
             rf"\multicolumn{{{n + 2}}}{{@{{}}l}}{{$\A(d)-\Sl$, exponent over {s['long_fit'][0]}--{s['long_fit'][1]}~s}} \\"]
    counts = [rf"\multicolumn{{{n + 2}}}{{@{{}}l}}{{windows}} \\"]
    for k, tag in WHEN_TAGS.items():
        r = s["classes"][k]
        for name, b in (("Near", r["bins"][0]), ("Far", r["bins"][-1])):
            mac_value(f"Carry{tag}{name}", b["diff"], 3, True)
            macro(f"CarryWindows{tag}{name}", num(b["windows"], 0))
        macro(f"CarryWindowsS{tag}", num(r["S"]["windows"], 0))
        lines.append(f"{k} & " + cell(r["S"], 3) + " & " + " & ".join(cell(b["diff"], 3, True) for b in r["bins"]) + r" \\")
        counts.append(f"{k} & {num(r['S']['windows'], 0)} & " + " & ".join(num(b["windows"], 0) for b in r["bins"]) + r" \\")
    write("carry_table", lines + [r"\midrule", *counts, r"\bottomrule", r"\end{tabular}"])


STAGES_TAG = ("pre", "start", "end", "post")
WHEN_TAGS = {**GROUND_TAGS, "all": "All"}


def route_tex():
    """route_alpha.py on the declared open flights up to DtMax s: the step and the speed of the route, the gaps on the
    route at the arc lengths that match 320 and 1000 s (Route<gap><class><scale>, with Band), the same gaps in time
    minus on the route (RouteMinus...), the largest difference between alpha* of the path and in time (RouteCheckMax) and of G - A
    (RouteCheckGAMax, over the solid lags, point estimates), and per class at the scale of 1000 s the share of G - A in time that
    the route keeps (RouteKeep<class>, %), the share that is the waiting (RouteWait<class>, %) and their extremes over the
    four classes (RouteKeepMin, RouteKeepMax)."""
    if not (DT5 / "route_alpha.json").exists():
        return
    r = read(DT5, "route_alpha")
    macro("RouteDs", num(r["ds_m"], 0))
    macro("RouteSpeed", num(r["speed_route_S"], 1))
    macro("RoutePathSpeed", num(r["speed_path_S"], 1))
    macro("RouteCheckMax", num(r["check"]["max_abs"], 3))
    g, a = r["check"]["group"], r["check"]["group_solo"]
    n = min(len(g["lags"]), len(a["lags"]))
    solid = np.array(g["solid"][:n]) & np.array(a["solid"][:n])
    shift = np.array(g["minus"]["value"][:n]) - np.array(a["minus"]["value"][:n])  # G - A in length minus G - A in time
    macro("RouteCheckGAMax", num(abs(shift[solid]).max(), 3))
    for tag in ("ThreeTwenty", "Thousand"):
        macro(f"RouteLag{tag}", num(r["route"]["all"]["at"][tag]["lag"], 0))
        macro(f"RouteTimeLag{tag}", num(r["route"]["all"]["at"][tag]["time_lag"], 0))
    for k, cls in WHEN_TAGS.items():
        for gap, short in GROUND_GAPS.items():
            for tag in ("ThreeTwenty", "Thousand"):
                at = r["route"][k]["at"][tag][gap]
                mac_value(f"Route{short}{cls}{tag}", at["route"], 3, True)
                mac_value(f"RouteMinus{short}{cls}{tag}", at["minus"], 3, True)
    keep = {}
    for k, cls in GROUND_TAGS.items():  # time = route + (time minus route)
        at = r["route"][k]["at"]["Thousand"]["G - A"]
        time = at["route"]["value"] + at["minus"]["value"]
        keep[k] = 100 * at["route"]["value"] / time
        macro(f"RouteKeep{cls}", num(keep[k], 0))
        macro(f"RouteWait{cls}", num(100 - keep[k], 0))
    macro("RouteKeepMin", num(min(keep.values()), 0))
    macro("RouteKeepMax", num(max(keep.values()), 0))


def prepost_ground_tex():
    """prepost.py --ground on the declared open flights up to DtMax s: the window, and per ground class and "all" the
    events, start - pre, end - start, post - end (the control of the group flights alone: ctrlA2 - ctrlA1) and the table."""
    if not (DT5 / "prepost_ground.json").exists():
        return
    s = read(DT5, "prepost_ground")
    macro("WhenWindow", num(s["window_s"], 0))
    lines = [r"\begin{tabular}{@{}l r r r r r@{}}", r"\toprule",
             r"ground & events & start $-$ pre & end $-$ start & post $-$ end & control \\", r"\midrule"]
    for k, tag in WHEN_TAGS.items():
        r = s["classes"][k]
        d = r["differences"]
        macro(f"When{tag}Events", num(r["events"], 0))
        mac_value(f"When{tag}StartMinusPre", d["start - pre"], 3, True)
        mac_value(f"When{tag}EndMinusStart", d["end - start"], 3, True)
        mac_value(f"When{tag}PostMinusEnd", d["post - end"], 3, True)
        mac_value(f"When{tag}Control", d["ctrlA2 - ctrlA1"], 3, True)
        lines.append(f"{k} & {num(r['events'], 0)} & " + " & ".join(cell(d[n], 3, True) for n in ("start - pre", "end - start", "post - end", "ctrlA2 - ctrlA1")) + r" \\")
    write("prepost_ground_table", lines + [r"\bottomrule", r"\end{tabular}"])
    prepost_long_tex(s)


def prepost_long_tex(s):
    """The exponent fitted over the long lags (prepost.py LONG_FIT) and the share of time in climb of each window, per
    class and "all": macros When{C}Long... and When{C}Climb{Pre,Start,End,Post}, and the table prepost_long_table."""
    lo, hi = s["long_fit"]
    macro("WhenLongLo", num(lo, 0))
    macro("WhenLongHi", num(hi, 0))
    steps = ("start - pre", "end - start", "post - end", "ctrlA2 - ctrlA1")
    lines = [r"\begin{tabular}{@{}l r r r r r r r r@{}}", r"\toprule",
             r" & \multicolumn{4}{c}{exponent over " + f"{lo}--{hi}" + r"~s} & \multicolumn{4}{c}{time in climb (\%)} \\",
             r"\cmidrule(lr){2-5}\cmidrule(l){6-9}",
             r"ground & start $-$ pre & end $-$ start & post $-$ end & control & pre & start & end & post \\", r"\midrule"]
    for k, tag in WHEN_TAGS.items():
        r = s["classes"][k]
        d = r["differences_long"]
        for name, short in zip(steps, ("StartMinusPre", "EndMinusStart", "PostMinusEnd", "Control")):
            mac_value(f"When{tag}Long{short}", d[name], 3, True)
        climb = {st: r["phases"]["share"][st]["climb"] for st in STAGES_TAG}
        for st, v in climb.items():
            mac_value(f"When{tag}Climb{st.capitalize()}", v, 1)
        lines.append(f"{k} & " + " & ".join(cell_d(d[n], 3) for n in steps) + " & "
                     + " & ".join(num(climb[st]["value"], 1) for st in STAGES_TAG) + r" \\")
    write("prepost_long_table", lines + [r"\bottomrule", r"\end{tabular}"])
    if "mock" in s:
        prepost_mock_tex(s)


def prepost_mock_tex(s):
    """The placebo of the three steps on the exponent over the long lags (prepost.py "mock"): macros WhenMock{C}{Step}
    (mock events), WhenRealMinusMock{C}{Step} (real events with a mock, minus their mocks) with Band, WhenMockEvents and
    WhenMockRealEvents (class "all"), and the table prepost_mock_table."""
    steps = {"start - pre": "StartMinusPre", "end - start": "EndMinusStart", "post - end": "PostMinusEnd"}
    kinds = {"real": "Real", "mock": "Mock", "real minus mock": "RealMinusMock"}
    macro("WhenMockEvents", num(s["mock"]["classes"]["all"]["mock events"], 0))
    macro("WhenMockRealEvents", num(s["mock"]["classes"]["all"]["real events with mocks"], 0))
    lines = [r"\begin{tabular}{@{}l r r r r r r r r r@{}}", r"\toprule",
             r" & \multicolumn{3}{c}{real} & \multicolumn{3}{c}{mock} & \multicolumn{3}{c}{real $-$ mock} \\",
             r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(l){8-10}",
             r"ground" + r" & start $-$ pre & end $-$ start & post $-$ end" * 3 + r" \\", r"\midrule"]
    for k, tag in WHEN_TAGS.items():
        r = s["mock"]["classes"][k]
        for kind, name in (("mock", "Mock"), ("real minus mock", "RealMinusMock")):
            for step, short in steps.items():
                mac_value(f"When{name}{tag}{short}", r[kind][step], 3, True)
        lines.append(f"{k} & " + " & ".join(cell_d(r[kind][step], 3) for kind in kinds for step in steps) + r" \\")
    write("prepost_mock_table", lines + [r"\bottomrule", r"\end{tabular}"])


THRESHOLD_KEYS = {"none": "none", "0.1": "$>0.1$", "0.5": "$>0.5$"}


def ground_opening_tex():
    """contrasts.py (design ground_opening): the lowland share among the flights with a known class, the table of the
    openings of S, A, G by class, and the table of G - A and A - S at 1000 s by class and opening threshold."""
    if not (DT5 / "contrasts_ground_opening.json").exists():
        return
    s = read(DT5, "contrasts_ground_opening")
    cl = s["classes"]
    for tag, key in (("All", "flights"), ("Strict", "flights_strict")):
        macro(f"OpenLowland{tag}", num(100 * sum(cl[k][key] for k in ("plains", "hills")) / cl["known"][key], 0))
    lines = [r"\begin{tabular}{@{}l r r r r r@{}}", r"\toprule",
             r" & \multicolumn{3}{c}{median $\mathcal O$} & \multicolumn{2}{c}{share of the flights (\%)} \\",
             r"\cmidrule(lr){2-4}\cmidrule(l){5-6}", r"ground & $\Sl$ & $\A$ & $\G$ & all & $\mathcal O>\Stricter$ \\", r"\midrule"]
    for k in GROUND_TAGS:
        row = [k, *(f"{cl[k]['median'][pop]:.2f}" for pop in ("solo", "group_solo", "group")),
               num(100 * cl[k]["flights"] / cl["known"]["flights"], 1), num(100 * cl[k]["flights_strict"] / cl["known"]["flights_strict"], 1)]
        lines.append(" & ".join(row) + r" \\")
    write("opening_class_table", lines + [r"\bottomrule", r"\end{tabular}"])
    lines = [r"\begin{tabular}{@{}l l l l l@{}}", r"\toprule", r" & & \multicolumn{3}{c}{opening threshold} \\",
             r"\cmidrule(l){3-5}", " ground & gap & " + " & ".join(THRESHOLD_KEYS.values()) + r" \\", r"\midrule"]
    for k, tag in GROUND_TAGS.items():
        for gap, text in (("G - A", r"$\G-\A$"), ("A - S", r"$\A-\Sl$")):
            cells = [cell(gap_at(s[f"{k}|{x}"]["gaps"][gap], 1000), 3, True) for x in THRESHOLD_KEYS]
            lines.append(f"{k if gap == 'G - A' else ''} & {text} & " + " & ".join(cells) + r" \\")
        if k != "high mountains":
            lines.append(r"\addlinespace")
    write("ground_opening_table", lines + [r"\bottomrule", r"\end{tabular}"])


def pilot_tex():
    """contrasts.py (design pilot): the share of the flights of S flown by pilots who also have group time, and the
    table of the flights of S and S' and of A - S and A - S' at 1000 s by class."""
    if not (DT5 / "contrasts_pilot.json").exists():
        return
    s = read(DT5, "contrasts_pilot")
    macro("PilotShareS", num(100 * s["all"]["S' share"], 0))
    lines = [r"\begin{tabular}{@{}l r r l l@{}}", r"\toprule", r"ground & $\Sl$ & $\Sl'$ & $\A-\Sl$ & $\A-\Sl'$ \\", r"\midrule"]
    for k, name in (*((k, k) for k in GROUND_TAGS), ("all", "all")):
        r = s[k]
        row = [name, num(r["flights"]["solo"], 0), num(r["flights"]["solo'"], 0)]
        row += [cell(gap_at(r["gaps"][g], 1000), 3, True) for g in ("A - S", "A - S'")]
        lines.append(" & ".join(row) + r" \\")
        if k == "high mountains":
            lines.append(r"\midrule")
    write("pilot_table", lines + [r"\bottomrule", r"\end{tabular}"])


PH_STATS = {"time in glide (%)": "ShareGlide", "time in search (%)": "ShareSearch", "time in climb (%)": "ShareClimb",
            "climbs per hour": "Cph", "glide speed (km/h)": "Speed", "height lost per km of glide (m/km)": "Sink",
            "searches that end in a climb (%)": "Found", "climb rate (m/s)": "Rate", "distance gained per climb (m)": "Chord",
            "height gained per climb (m)": "Gain", "drift speed of a climb (m/s)": "Drift", "time in exit (%)": "ShareExit",
            "searches per hour": "SearchPerHour", "exits per hour": "ExitPerHour", "mean exit duration (s)": "ExitMean",
            "median exit duration (s)": "ExitDur",
            "cycle of a glide and a climb (min)": "Cycle"}
PH_LABELS = {"time in glide (%)": r"time in glide (\%)", "time in search (%)": r"time in search (\%)", "time in climb (%)": r"time in climb (\%)",
             "climbs per hour": "climbs per hour", "glide speed (km/h)": "glide speed (km/h)",
             "height lost per km of glide (m/km)": "height lost per km of glide (m/km)",
             "searches that end in a climb (%)": r"searches ending in a climb (\%)", "climb rate (m/s)": "climb rate (m/s)",
             "distance gained per climb (m)": "distance gained per climb (m)",
             "height gained per climb (m)": "height gained per climb (m)", "drift speed of a climb (m/s)": "drift speed of a climb (m/s)",
             "time in exit (%)": r"time in exit (\%)", "searches per hour": "searches per hour", "exits per hour": "exits per hour",
             "mean exit duration (s)": "mean exit duration (s)", "median exit duration (s)": "median exit duration (s)", "cycle of a glide and a climb (min)": "glide and climb cycle (min)"}
# per hour and mean duration of each phase, and the refound searches (phase_gaps.PHYSICAL): macros, not in the full table
PH_PHYSICAL = {"glides per hour": "GlidePerHour", "mean glide duration (s)": "GlideMean", "mean search duration (s)": "SearchMean",
               "mean climb duration (s)": "ClimbMean", "refound searches per hour": "RefoundPerHour",
               "mean refound duration (s)": "RefoundMean", "found searches that follow a climb (%)": "RefoundShare"}
# the four kinds of search run as shares of their runs and of their time (phase_gaps.KINDS): macros and phases_kinds_table
PH_KINDS = {f"{k}: share of {w} (%)": f"Kind{k.title()}{w.title()}" for w in ("runs", "time") for k in ("found", "refound", "failed", "exit")}
PH_GAPS = {"G - A": "GA", "A - S": "AS", "G - S": "GS"}
PH_PHASES = {"climb": "Climb", "search": "Search", "transition": "Glide"}
PH_DIGITS = 3  # significant digits of the macros of the phases
PH_PREFIX = {"refit": "Ph", "vilpellet": "PhV"}  # the labels of the refit are the ones of the story


def sig(x, n=PH_DIGITS):
    """The decimals that give `x` n significant digits, from 0 to 4."""
    return 0 if x == 0 or not np.isfinite(x) else int(min(4, max(0, n - 1 - np.floor(np.log10(abs(x))))))


def sig_value(name, v, sign=False):
    """mac_value with the decimals of sig() of the value."""
    mac_value(name, v, sig(v["value"]), sign)


def sig_cell(v, sign=False):
    """cell() with the decimals of sig() of the value."""
    return cell(v, sig(v["value"]), sign)


def phases_macros(r, prefix, gaps=PH_GAPS):
    """The macros of one label set of phase_gaps.py: per class, `prefix`+statistic+population or gap+class (value and
    Band), the flights and hours, the thresholds and the exponents of the durations, the mean and the median of the durations
    (`prefix`Dur<Mean|Median><glide or climb><population or gap><class>, Band), the cycle and the peak lags."""
    for key, c in WHEN_TAGS.items():
        d = r["phases"][key]
        for pop in "SAG":
            macro(f"{prefix}Flights{pop}{c}", num(d["flights"][pop], 0))
            macro(f"{prefix}Hours{pop}{c}", num(d["hours"][pop], 0))
        if "contexts" in d:
            for k, tag in (("unknown", "SearchDropped"), ("found", "CtxFound"), ("refound", "CtxRefound"), ("failed", "CtxFailed"), ("exit", "CtxExit")):
                macro(f"{prefix}{tag}{c}", num(100 * d["contexts"]["all"]["share"][k], 1))
                if key == "all":
                    macro(f"{prefix}{tag}", num(100 * d["contexts"]["all"]["share"][k], 1))
        for stat, tag in {**PH_STATS, **PH_PHYSICAL, **PH_KINDS}.items():
            if stat not in d["gaps"]["G - A"]:
                continue
            for pop in ("S", "A", "G") if gaps is PH_GAPS else ():
                sig_value(f"{prefix}{tag}{pop}{c}", d["statistics"][pop][stat])
            for gap, g in gaps.items():  # the gaps of the kinds of search have the two decimals of phases_kinds_table
                v = d["gaps"][gap][stat]
                mac_value(f"{prefix}{tag}{g}{c}", v, 2 if stat in PH_KINDS else sig(v["value"]), True)
        for phase, ptag in PH_PHASES.items():
            t = r["tails"][key][phase]
            if t is None:
                continue
            macro(f"{prefix}Xmin{ptag}{c}", num(t["xmin"]["common"], 0))
            for pop in ("S", "A", "G") if gaps is PH_GAPS else ():
                sig_value(f"{prefix}Mu{ptag}{pop}{c}", t["mu"][pop])
                macro(f"{prefix}Above{ptag}{pop}{c}", num(t["runs_above"][pop], 0))
            for gap, g in gaps.items():
                sig_value(f"{prefix}Mu{ptag}{g}{c}", t["gaps"][gap], True)
            for name, x in ({} if phase == "search" else t["durations"]).items():  # the searches: Kaplan-Meier, duration_tex
                for pop in ("S", "A", "G") if gaps is PH_GAPS else ():
                    sig_value(f"{prefix}Dur{name.title()}{ptag}{pop}{c}", x["values"][pop])
                for gap, g in gaps.items():
                    sig_value(f"{prefix}Dur{name.title()}{ptag}{g}{c}", x["gaps"][gap], True)
        for gap, g in (("G - A", "GA"), ("G - S", "GS")):
            if r["peaks"] and gaps is PH_GAPS:
                macro(f"{prefix}Peak{g}Lag{c}", num(r["peaks"][key][gap]["lag"], 0))


def phases_main_table(r, kept):
    """The kept statistics, G - A and A - S (with their bands) per ground class."""
    lines = [r"\begin{tabular}{@{}l l " + "l " * len(WHEN_TAGS) + r"@{}}", r"\toprule",
             " statistic & gap & " + " & ".join(WHEN_TAGS) + r" \\", r"\midrule"]
    for stat in kept:
        for gap, text in (("G - A", r"$\G-\A$"), ("A - S", r"$\A-\Sl$")):
            cells = [sig_cell(r["phases"][k]["gaps"][gap][stat], True) for k in WHEN_TAGS]
            lines.append(f"{PH_LABELS[stat] if gap == 'G - A' else ''} & {text} & " + " & ".join(cells) + r" \\")
        lines.append(r"\addlinespace")
    write("phases_main_table", lines[:-1] + [r"\bottomrule", r"\end{tabular}"])


def phases_full_table(r):
    """Every statistic, per ground class: S, A, G, and the three gaps with their bands."""
    lines = [r"\begin{tabular}{@{}l r r r l l l@{}}", r"\toprule",
             r"statistic & $\Sl$ & $\A$ & $\G$ & $\G-\A$ & $\A-\Sl$ & $\G-\Sl$ \\", r"\midrule"]
    for key in WHEN_TAGS:
        d = r["phases"][key]
        n = d["flights"]
        lines.append(rf"\multicolumn{{7}}{{@{{}}l}}{{\textit{{{key}}}: {num(n['S'], 0)}, {num(n['A'], 0)}, {num(n['G'], 0)} flights}} \\")
        for stat in (k for k in PH_STATS if k in d["gaps"]["G - A"]):
            v = [num(d["statistics"][pop][stat]["value"], sig(d["statistics"][pop][stat]["value"])) for pop in "SAG"]
            g = [sig_cell(d["gaps"][gap][stat], True) for gap in PH_GAPS]
            lines.append(f"{PH_LABELS[stat]} & " + " & ".join(v + g) + r" \\")
        lines.append(r"\midrule")
    write("phases_table", lines[:-1] + [r"\bottomrule", r"\end{tabular}"])


def phases_tail_table(r):
    """The durations: per class and phase, the common threshold, mu of S, A, G above it and the gaps."""
    lines = [r"\begin{tabular}{@{}l l r l l l l l@{}}", r"\toprule",
             r"ground & phase & $x_\mathrm{min}$ (s) & $\Sl$ & $\A$ & $\G$ & $\G-\A$ & $\A-\Sl$ \\", r"\midrule"]
    for key in WHEN_TAGS:
        for phase, text in (("transition", "glide"), ("search", "search"), ("climb", "climb")):
            t = r["tails"][key][phase]
            if t is None:
                continue
            row = [key if phase == "transition" else "", text, num(t["xmin"]["common"], 0)]
            row += [sig_cell(t["mu"][pop]) for pop in "SAG"] + [sig_cell(t["gaps"][g], True) for g in ("G - A", "A - S")]
            lines.append(" & ".join(row) + r" \\")
        lines.append(r"\addlinespace")
    write("phases_tail_table", lines[:-1] + [r"\bottomrule", r"\end{tabular}"])


PH_UNITS = (("glide", "glides per hour", "mean glide duration (s)"), ("search", "searches per hour", "mean search duration (s)"),
            ("climb", "climbs per hour", "mean climb duration (s)"))


def phases_units_table(r):
    """The number per hour and the mean duration (s) of each phase on every flight: S, A, G and G - A with its band."""
    d = r["phases"]["all"]
    lines = [r"\begin{tabular}{@{}l r r r l r r r l@{}}", r"\toprule",
             r" & \multicolumn{4}{c}{per hour} & \multicolumn{4}{c}{mean duration (s)} \\", r"\cmidrule(lr){2-5}\cmidrule(l){6-9}",
             r"phase & $\Sl$ & $\A$ & $\G$ & $\G-\A$ & $\Sl$ & $\A$ & $\G$ & $\G-\A$ \\", r"\midrule"]
    for phase, hourly, mean in PH_UNITS:
        row = [phase]
        for stat in (hourly, mean):
            row += [num(d["statistics"][pop][stat]["value"], sig(d["statistics"][pop][stat]["value"])) for pop in "SAG"]
            row.append(sig_cell(d["gaps"]["G - A"][stat], True))
        lines.append(" & ".join(row) + r" \\")
    write("phases_units_table", lines + [r"\bottomrule", r"\end{tabular}"])
    mean = "mean climb duration (s)"  # PhClimbMeanClassBound: the whole percent above every |G - A| of the mean climb, in % of A, per class
    macro("PhClimbMeanClassBound", num(np.ceil(max(100 * abs(r["phases"][k]["gaps"]["G - A"][mean]["value"]
                                                          / r["phases"][k]["statistics"]["A"][mean]["value"]) for k in GROUND_TAGS)), 0))


def phases_kinds_table(r):
    """The four kinds of search run on every flight: found (after a glide), refound (after a climb), failed, exit, as a share of
    the runs and of the time of the four together, S, A, G and G - A with its band."""
    d = r["phases"]["all"]
    lines = [r"\begin{tabular}{@{}l r r r l r r r l@{}}", r"\toprule",
             r" & \multicolumn{4}{c}{share of the runs (\%)} & \multicolumn{4}{c}{share of the time (\%)} \\", r"\cmidrule(lr){2-5}\cmidrule(l){6-9}",
             r"kind & $\Sl$ & $\A$ & $\G$ & $\G-\A$ & $\Sl$ & $\A$ & $\G$ & $\G-\A$ \\", r"\midrule"]
    for kind, text in (("found", "found, after a glide"), ("refound", "found again, after a climb"), ("failed", "failed"),
                       ("exit", "exit, after a climb")):
        row = [text]
        for what in ("runs", "time"):
            stat = f"{kind}: share of {what} (%)"
            row += [num(d["statistics"][pop][stat]["value"], 1) for pop in "SAG"] + [cell(d["gaps"]["G - A"][stat], 2, True)]
        lines.append(" & ".join(row) + r" \\")
    write("phases_kinds_table", lines + [r"\bottomrule", r"\end{tabular}"])


DUR_PHASES = (("transition", "Glide"), ("search", "Search"), ("climb", "Climb"))
DUR_STATS = (("mean", "Mean"), ("median", "Median"))


def duration_summaries(r, km):
    """The mean and the median of the durations of every flight, {phase: {stat: {values: {S, A, G}, gaps: {G - A, A - S}}}}: the
    glides and the climbs from phases_refit.json, the searches from search_km.json (Kaplan-Meier median, restricted mean)."""
    k = km["km"]["all"]
    names = {"mean": "restricted mean time to find (s)", "median": "median time to find (s)"}
    out = {phase: r["tails"]["all"][phase]["durations"] for phase in ("transition", "climb")}
    out["search"] = {st: {"values": {pop: k["summaries"][pop][n] for pop in "SAG"},
                          "gaps": {gap: k["gaps"][gap][n] for gap in ("G - A", "A - S")}} for st, n in names.items()}
    return out, k["limit"]


def duration_tex(r, km):
    """PhDur<Mean|Median>Search<S|A|G|GA|AS>All (with Band; those of the glides and the climbs are phases_macros') and PhDurLimitAll,
    the mean and the median of the durations of every flight, and the table phases_duration_table: per phase, the mean and the
    median of S, A, G and G - A."""
    d, limit = duration_summaries(r, km)
    macro("PhDurLimitAll", num(limit, 0))
    lines = [r"\begin{tabular}{@{}l l r r r l@{}}", r"\toprule", r"phase & & $\Sl$ & $\A$ & $\G$ & $\G-\A$ \\", r"\midrule"]
    for phase, tag in DUR_PHASES:
        for stat, stag in DUR_STATS:
            x = d[phase][stat]
            if phase == "search":
                for pop in "SAG":
                    sig_value(f"PhDur{stag}{tag}{pop}All", x["values"][pop])
                for gap, g in (("G - A", "GA"), ("A - S", "AS")):
                    sig_value(f"PhDur{stag}{tag}{g}All", x["gaps"][gap], True)
            label = {"transition": "glide", "climb": "climb", "search": "search (KM)"}[phase] if stat == "mean" else ""
            lines.append(" & ".join([label, stat + (" (restricted)" if phase == "search" and stat == "mean" else "")] +
                                    [num(x["values"][pop]["value"], sig(x["values"][pop]["value"])) for pop in "SAG"] +
                                    [sig_cell(x["gaps"]["G - A"], True)]) + r" \\")
        lines.append(r"\addlinespace")
    write("phases_duration_table", lines[:-1] + [r"\bottomrule", r"\end{tabular}"])


JOIN_KINDS = (("join", "Enter"), ("leave", "Leave"))
JOIN_PHASES = (("glide", "Glide"), ("search", "Search"), ("climb", "Climb"))


def joins_tex(j):
    """The phase at which pilots enter and leave group time (phase_joins.py): Enter|Leave<Share|Base|Ratio><phase><class> (with
    Band; shares in %), the events (Enter|LeaveCount<class>, JoinFlights<class>, EnterAtDeparture, LeaveAtLanding, EventsNoPhase) and the table
    phases_joins_table (per ground class and every flight). A search is a run of any kind but a glide or a climb: the exits
    from a thermal and the runs with no neighbour count as searches."""
    for kind, tag in JOIN_KINDS:
        for key, c in WHEN_TAGS.items():
            macro(f"{tag}Count{c}", num(j["events by class"][key][kind], 0))
            for phase, ptag in JOIN_PHASES:
                x = j[kind][key][phase]
                for q, qtag in (("share", "Share"), ("baseline", "Base")):
                    mac_value(f"{tag}{qtag}{ptag}{c}", {"value": 100 * x[q]["value"], "band": [100 * b for b in x[q]["band"]]}, 1)
                mac_value(f"{tag}Ratio{ptag}{c}", x["ratio"], 2)
    for key, c in WHEN_TAGS.items():
        macro(f"JoinFlights{c}", num(j["flights"][key], 0))
    macro("EnterAtDeparture", num(j["events"]["join at departure"], 0))
    macro("LeaveAtLanding", num(j["events"]["leave at landing"], 0))
    macro("EventsNoPhase", num(j["events"]["without a phase"], 0))
    lines = [r"\begin{tabular}{@{}l l r r l r r l@{}}", r"\toprule",
             r" & & \multicolumn{3}{c}{enter a group} & \multicolumn{3}{c}{leave a group} \\", r"\cmidrule(lr){3-5}\cmidrule(l){6-8}",
             r"ground & phase & observed (\%) & chance (\%) & ratio & observed (\%) & chance (\%) & ratio \\", r"\midrule"]
    for key in WHEN_TAGS:
        for i, (phase, _) in enumerate(JOIN_PHASES):
            row = [key if i == 0 else "", phase]
            for kind, _ in JOIN_KINDS:
                x = j[kind][key][phase]
                row += [num(100 * x["share"]["value"], 1), num(100 * x["baseline"]["value"], 1), cell(x["ratio"], 2)]
            lines.append(" & ".join(row) + r" \\")
        lines.append(r"\addlinespace")
    write("phases_joins_table", lines[:-1] + [r"\bottomrule", r"\end{tabular}"])


def tail_numbers(r):
    """TailMu{Min,Max}: the smallest and largest tail exponent of S, A, G over the classes and phases; TailAbove{Min,Max}:
    the fewest and most durations of G above the common threshold; the constants of the threshold scan (tail_ks.py)."""
    tails = [t for k in WHEN_TAGS for t in r["tails"][k].values() if t]
    mu = [t["mu"][p]["value"] for t in tails for p in "SAG"]
    above = [t["runs_above"]["G"] for t in tails]
    macro("TailMuMin", num(min(mu), 1))
    macro("TailMuMax", num(max(mu), 1))
    macro("TailAboveMin", num(min(above), 0))
    macro("TailAboveMax", num(max(above), 0))
    macro("TailFineBelow", num(tail_ks.FINE_BELOW, 0))
    macro("TailBinPercent", f"{100 * tail_ks.BIN:g}")
    macro("TailCandidates", num(tail_ks.N_CANDIDATES, 0))
    macro("TailQuantileLo", f"{tail_ks.QUANTILES[0]:g}")
    macro("TailQuantileHi", f"{tail_ks.QUANTILES[1]:g}")
    macro("TailMinRuns", num(tail_ks.MIN_TAIL, 0))


def phases_cycle_table(r):
    """The cycle of a glide and a climb of S, A, G per class against the lag of the peak of the normalised gaps;
    PhCycleRatioMin/Max: the peak lags of G - A and G - S in cycles (lag over the mean cycle of S, A, G of the
    class), smallest and largest over the classes; PhCycleShortMin/Max: A minus G of the cycle (min), smallest and
    largest over the classes; PhCycleMin/Max: the shortest and the longest cycle of the table, over the rows and S, A, G; PhCycleShortAll: A minus G of the cycle
    on every flight."""
    cyc = "cycle of a glide and a climb (min)"
    every = [num(r["phases"][k]["statistics"][p][cyc]["value"], 1) for k in WHEN_TAGS for p in "SAG"]  # as printed in the table
    macro("PhCycleMin", min(every, key=float))
    macro("PhCycleMax", max(every, key=float))
    shorter_all = -r["phases"]["all"]["gaps"]["G - A"][cyc]["value"]
    macro("PhCycleShortAll", num(shorter_all, sig(shorter_all)))  # A minus G on every flight, a positive number
    lines = [r"\begin{tabular}{@{}l r r r r r@{}}", r"\toprule",
             r" & \multicolumn{3}{c}{cycle (min)} & \multicolumn{2}{c}{peak lag (s)} \\", r"\cmidrule(lr){2-4}\cmidrule(l){5-6}",
             r"ground & $\Sl$ & $\A$ & $\G$ & $\G-\A$ & $\G-\Sl$ \\", r"\midrule"]
    for key in WHEN_TAGS:
        row = [key] + [num(r["phases"][key]["statistics"][pop][cyc]["value"], 1) for pop in "SAG"]
        row += [num(r["peaks"][key][g]["lag"], 0) if r["peaks"] else "--" for g in ("G - A", "G - S")]
        lines.append(" & ".join(row) + r" \\")
    write("phases_cycle_table", lines + [r"\bottomrule", r"\end{tabular}"])
    if r["peaks"]:  # the peak lags in cycles of the class (the mean cycle of S, A, G), and A - G
        ratio = [r["peaks"][k][g]["lag"] / (60 * np.mean([r["phases"][k]["statistics"][p][cyc]["value"] for p in "SAG"]))
                 for k in WHEN_TAGS for g in ("G - A", "G - S")]
        macro("PhCycleRatioMin", num(min(ratio), 1))
        macro("PhCycleRatioMax", num(max(ratio), 1))
        shorter = [-r["phases"][k]["gaps"]["G - A"][cyc]["value"] for k in WHEN_TAGS]
        macro("PhCycleShortMin", num(min(shorter), 1))
        macro("PhCycleShortMax", num(max(shorter), 1))


def phases_tex():
    """The macros and the tables of phase_gaps.py on the declared open flights up to DtMax s: the macros of both label
    sets (the refit as Ph..., his labels as PhV..., the contexts of the searches and the gaps only), the tables of the refit, PhCtxMaxDiff (the largest
    difference, in points, between the shares of the four moments of a search of the two label sets)."""
    for label_set, prefix in PH_PREFIX.items():
        if (DT5 / f"phases_{label_set}.json").exists():
            r = read(DT5, f"phases_{label_set}")
            phases_macros(r, prefix, PH_GAPS if label_set == "refit" else {k: v for k, v in PH_GAPS.items() if k != "G - S"})
            if label_set == "refit":
                red = DT5 / "phases_redundancy_refit.json"
                kept = [s for s in PH_STATS if s != "cycle of a glide and a climb (min)"]
                kept = [s for s in kept if not red.exists() or s in read(DT5, "phases_redundancy_refit")["kept"]]
                phases_main_table(r, kept)
                phases_full_table(r)
                phases_tail_table(r)
                phases_cycle_table(r)
                phases_units_table(r)
                phases_kinds_table(r)
                if (DT5 / "search_km.json").exists():
                    duration_tex(r, read(DT5, "search_km"))
                if (DT5 / "phase_joins.json").exists():
                    joins_tex(read(DT5, "phase_joins"))
                tail_numbers(r)
                if (DT5 / "phases_vilpellet.json").exists():  # the four moments of a search: ours against his, in points
                    share = lambda x: x["phases"]["all"]["contexts"]["all"]["share"]
                    his = share(read(DT5, "phases_vilpellet"))
                    macro("PhCtxMaxDiff", num(100 * max(abs(share(r)[k] - his[k]) for k in ("found", "refound", "failed", "exit")), 1))


REFIT = OUTS.parents[2] / "methods" / "phase-labelling" / "out"  # the refit of the phase labels (methods/phase-labelling)
REFIT_CLASSES = {"plains": "Plains", "hills": "Hills", "low mountains": "LowMtn", "high mountains": "HighMtn"}


def refit_tex():
    """The macros Refit... of the refit of the phase labels (refit_*.json of methods/phase-labelling/out), among them
    RefitIterMax (the most iterations of a converged start) and RefitHangVsPara (the share of fixes where the hang
    glider fit and the paraglider fit agree, in %), and the agreement in sign of the gaps G - A and A - S between our
    labels and Vilpellet's (phases_*.json); skipped when a file is missing."""
    names = ("refit_params_para", "refit_params_hang", "refit_cadence", "refit_agreement")
    files = [REFIT / f"{n}.json" for n in names] + [DT5 / "phases_refit.json", DT5 / "phases_vilpellet.json"]
    if not all(f.exists() for f in files):
        return
    par, cad, agr = {"para": read(REFIT, names[0]), "hang": read(REFIT, names[1])}, read(REFIT, names[2]), read(REFIT, names[3])
    for d, tag in (("para", "Para"), ("hang", "Hang")):
        t = par[d]["training"]
        macro(f"RefitTrain{tag}", num(t["train_flights"], 0))
        macro(f"RefitHeld{tag}", num(t["held_out_flights"], 0))
        macro(f"RefitFlights{tag}", num(agr[d]["common_flights"], 0))
        macro(f"RefitAgree{tag}", num(100 * agr[d]["overall"]["same_label"], 1))
    macro("RefitPoolPara", num(par["para"]["training"]["pool"], 0))
    macro("RefitIterMax", num(max(x["iterations"] for d in par.values() for x in d["starts"] if x["converged"]), 0))
    macro("RefitHangVsPara", num(100 * par["hang"]["own_vs_para_refit_labels"]["same_label"], 3))
    macro("RefitStarts", num(len(par["para"]["starts"]), 0))
    ll, fixes = par["para"]["held_out_log_likelihood"], par["para"]["training"]["held_out_fixes"]
    for key, tag in (("his", "His"), ("refit", "Ours"), ("best_any_meaning", "Best")):
        macro(f"RefitLL{tag}ParaFix", signed(ll[key] / fixes, 3))
    for d, tag, gaps in (("para", "Para", ("2", "5")), ("hang", "Hang", ("5",))):
        for g, w in zip(gaps, ("Two", "Five")[-len(gaps):]):
            macro(f"RefitCadence{tag}{w}", num(100 * cad[d]["results"]["refit"][g]["same_label"], 1))
    for cls, tag in REFIT_CLASSES.items():
        macro(f"RefitAgree{tag}", num(100 * agr["para"]["per_class"][cls]["same_label"], 1))
    st = agr["para"]["statistics"]
    for key, tag in (("his", "His"), ("refit", "Ours")):
        macro(f"RefitSearchShare{tag}", num(100 * st[key]["time_share_pooled"]["search"], 1))
        macro(f"RefitSearchPerHour{tag}", num(st[key]["runs_per_hour_pooled"]["search"], 2))
    ours, his = read(DT5, "phases_refit"), read(DT5, "phases_vilpellet")
    pairs = [(ours["phases"][c]["gaps"][g][s]["value"], his["phases"][c]["gaps"][g][s]["value"])
             for c in WHEN_TAGS for g in ("G - A", "A - S") for s in ours["phases"][c]["gaps"][g]
             if s in his["phases"][c]["gaps"][g] and not s.startswith("cycle")]
    macro("PhSignAgree", num(sum((a > 0) == (b > 0) for a, b in pairs), 0))
    macro("PhSignTotal", num(len(pairs), 0))


def labels_check_tex():
    """The macros GroupCheck... of methods/group-identification/labels_check.py: the grouping of the chosen run with
    the refit phase labels against Vilpellet's on a sample of days; skipped when the JSON is missing."""
    f = REFIT.parent.parent / "group-identification" / "out" / "labels_check.json"
    if not f.exists():
        return
    c = json.loads(f.read_text())
    macro("GroupCheckDays", num(c["days"], 0))
    macro("GroupCheckFlightsChange", signed(c["flights_change_pct"], 1))
    macro("GroupCheckTimeChange", signed(c["time_change_pct"], 2))
    macro("GroupCheckShared", num(c["shared_of_union_pct"], 1))


def curves_tex(folder, prefix):
    """Macros of longlags.py on the flights of `folder`: the longest lag of each panel, H_loc there, the
    flights of each curve; and the table of the flights of the datasets."""
    long = read(folder, "longlags")
    macro(f"{prefix}MinFlights", num(long["min_flights"], 0))
    lines = [r"\begin{tabular}{@{}l r r r@{}}", r"\toprule", r"dataset & $\Sl$ & $\G$ & $\A$ \\", r"\midrule"]
    for name, panel in long["panels"].items():
        p = name.capitalize()
        macro(f"{prefix}Tau{p}", num(panel["tau_max"], 0))
        lines.append(f"lags up to {num(panel['tau_max'], 0)}~s & " + " & ".join(num(panel["curves"][k]["flights"], 0) for k, _ in POPS) + r" \\")
        for k, tag in POPS:
            c = panel["curves"][k]
            macro(f"{prefix}{p}Flights{tag}", num(c["flights"], 0))
            k1000 = c["lags"].index(1000)
            macro(f"{prefix}{p}HThousand{tag}", f"{c['H_loc'][k1000]:.3f}")
            macro(f"{prefix}{p}Fit{tag}", f"{c['H']:.3f}")
            macro(f"{prefix}{p}H{tag}", f"{c['H_loc'][-1]:.3f}")
            macro(f"{prefix}{p}H{tag}Band", f"[{c['H_loc_band'][0][-1]:.3f}, {c['H_loc_band'][1][-1]:.3f}]")
    write(f"{prefix.lower()}_datasets_table", lines + [r"\bottomrule", r"\end{tabular}"])


# ---- the four comparisons: raw, matched, decomposition, before and after ----
STAT7 = ("time in climb (%)", "time in search (%)", "searches that end in a climb (%)", "climbs per hour",
         "climb rate (m/s)", "glide speed (km/h)", "height lost per km of glide (m/km)")
HK, HQ, FITK, FITT = "H_loc(1000 s)", "H_loc(1840 s)", "H (10 s to 1000 s)", "H (10 s to T)"
QUANT = (HK, HQ, *STAT7)
TAG = dict(zip((HK, HQ, FITK, FITT, *STAT7), ("HK", "HQ", "FitK", "FitT", "Climb", "Search", "Found", "Cph", "Rate", "Glide", "Sink")))
LABEL = {HK: r"$\Hloc(1000~\mathrm{s})$", HQ: r"$\Hloc(1840~\mathrm{s})$", "time in climb (%)": r"time in climb (\%)",
         "time in search (%)": r"time in search (\%)", "searches that end in a climb (%)": r"searches ending in a climb (\%)",
         "climbs per hour": "climbs per hour", "climb rate (m/s)": "climb rate (m/s)", "glide speed (km/h)": "glide speed (km/h)",
         "height lost per km of glide (m/km)": "height lost per km of glide (m/km)"}
DIG = {HK: 3, HQ: 3, FITK: 3, FITT: 3, "time in climb (%)": 1, "time in search (%)": 1, "searches that end in a climb (%)": 1,
       "climbs per hour": 2, "climb rate (m/s)": 2, "glide speed (km/h)": 1, "height lost per km of glide (m/km)": 0}
DDIG = {**DIG, "climb rate (m/s)": 3, "climbs per hour": 2}  # digits of a difference
XC = "cross-country speed over 30 min (km/h)"  # not in STAT7: only the runs that measured it have it
TAG[XC], LABEL[XC], DIG[XC], DDIG[XC] = "Xc", XC, 1, 1
DS = {"1000": "One", "2760": "Two", "longest": "Long"}  # dataset tags of the macros: to 1000 s, to 2760 s, the longest lag
POP = ("S", "A", "G")
POPTEX = {"S": r"$\Sl$", "A": r"$\A$", "G": r"$\G$"}
PHASE = {"climb": "Climb", "search": "Search", "transition": "Glide"}
SIDES = {"G": "G", "S|G": "SG", "A": "A", "S|A": "SA"}
CELLS = {"none": "None", "R": "R", "T": "T", "R+T": "RT", "W": "W", "R+W": "RW", "T+W": "TW", "R+T+W": "F"}
CALIPER_TEXT = {"month1_hour1_class": r"month $\pm1$, hour $\pm1$, same class", "month1_hour1_level": r"month $\pm1$, hour $\pm1$, same level",
                "month2_hour2_class": r"month $\pm2$, hour $\pm2$, same class", "month2_hour2_level": r"month $\pm2$, hour $\pm2$, same level"}
CELL_TEXT = {"none": r"$\emptyset$", "R": "R", "T": "T", "R+T": "RT", "W": "W", "R+W": "RW", "T+W": "TW", "R+T+W": "RTW"}
LEVELS = ("beginners", "experienced", "all")
GAPTAG = {"G - S": "", "A - S": "AS", "G - A": "GA"}  # the raw gaps of raw.py and their macro suffixes


def with_xc(entry):
    """STAT7, and XC when `entry` (a dict by statistic) has it."""
    return [*STAT7, *([XC] if XC in entry else [])]


def piece(x):
    """A piece size in seconds: whole, or with one decimal."""
    return num(x, 0 if abs(x - round(x)) < 1e-9 else 1)


def bracket(b, d, sign=False):
    f = signed if sign else num
    return f"[{f(b[0], d)}, {f(b[1], d)}]"


def cell(v, d, sign=False):
    """`value [lo, hi]` of a {value, band} entry."""
    return (signed if sign else num)(v["value"], d) + " " + bracket(v["band"], d, sign)


def zero_in(v):
    return v["band"][0] <= 0 <= v["band"][1]


def mac_value(name, v, d, sign=False):
    """The macros NAME (value) and NAMEBand of a {value, band} entry."""
    macro(name, (signed if sign else num)(v["value"], d))
    macro(name + "Band", bracket(v["band"], d, sign))


def hloc_cut(c, lag, cut):
    """H_loc at `lag` of a curve `c` (lags, msd) with the window cut at `cut` s: the fit of tamsd.local_h on the
    lags up to `cut` only."""
    lags, msd = np.asarray(c["lags"], float), np.asarray(c["msd"], float)
    x, y = np.log10(lags[lags <= cut]), np.log10(msd[lags <= cut])
    near = np.abs(x - np.log10(lag)) <= 0.25 + 1e-9
    if near.sum() < 3:
        near = np.isin(np.arange(len(x)), np.argsort(np.abs(x - np.log10(lag)), kind="stable")[:3])
    return float(0.5 * np.polyfit(x[near], y[near], 1)[0])


def raw_tex(r):
    curves = {ds: {p: v["curve"] for p, v in r["datasets"][ds]["populations"].items()} for ds in ("1000", "2760", "longest")}
    span = lambda ds: [max(c["H_loc"]) - min(c["H_loc"]) for c in curves[ds].values()]  # how much H_loc varies along a curve
    for ds, tag in (("2760", "Two"), ("longest", "Long")):
        macro(f"RawHKCut{tag}S", num(hloc_cut(curves[ds]["S"], 1000, 1000), 3))
    macro("RawHlocRangeLow", num(min(span("1000") + span("2760")), 3))
    macro("RawHlocRangeHigh", num(max(span("1000") + span("2760")), 3))
    macro("RawHlocRangeLong", num(max(span("longest")), 3))
    macro("RawHMeanGap", num(max(abs(c["H"] - np.mean(c["H_loc"])) for d in curves.values() for c in d.values()), 3))
    for tag in ("S", "A", "G"):
        macro(f"RawGlideMedian{tag}", num(phase_median(r["survival"][tag]["transition"]), 0))
        macro(f"RawClimbMedian{tag}", num(phase_median(r["survival"][tag]["climb"]), 0))
    lines = [r"\begin{tabular}{@{}l l r r l l l@{}}", r"\toprule",
             r"dataset & & flights & pieces & fitted $H$ & $\Hloc(1000~\mathrm{s})$ & $\Hloc(1840~\mathrm{s})$ \\", r"\midrule"]
    for ds, d in r["datasets"].items():
        t = d["tau_max"]
        macro(f"RawTau{DS[ds]}", num(t, 0))
        macro(f"RawShortest{DS[ds]}", piece(t / 0.8))
        for k, p in enumerate(POP):
            v = d["populations"][p]
            for q, f in (("Flights", num(v["flights"], 0)), ("Pieces", num(v["pieces"], 0))):
                macro(f"Raw{q}{DS[ds]}{p}", f)
            macro(f"RawFit{DS[ds]}{p}", num(v["H"], 3))
            macro(f"RawFit{DS[ds]}{p}Band", bracket(v["H_band"], 3))
            mac_value(f"RawHK{DS[ds]}{p}", v[HK], 3)
            q = v[HQ]
            if q:
                mac_value(f"RawHQ{DS[ds]}{p}", q, 3)
            head = f"to {num(t, 0)}~s" if k == 0 else ""
            lines.append(f"{head} & {POPTEX[p]} & {num(v['flights'], 0)} & {num(v['pieces'], 0)} & {num(v['H'], 3)} {bracket(v['H_band'], 3)}"
                         f" & {cell(v[HK], 3)} & {cell(q, 3) if q else '--'} \\\\")
        lines.append(r"\midrule" if ds != "longest" else r"\bottomrule")
    write("raw_table", lines + [r"\end{tabular}"])
    for gap, tag in GAPTAG.items():  # the paired gaps, with the band of their own draws
        for ds, q in (("1000", HK), ("2760", HQ)):
            v, pops = r["gaps"][ds][q][gap], r["datasets"][ds]["populations"]
            assert abs(v["value"] - (pops[gap[0]][q]["value"] - pops[gap[-1]][q]["value"])) < 1e-12
            mac_value(f"RawGap{TAG[q]}{tag}", v, 3, True)
        curve = r["gaps"]["1000"]["H_loc"][gap]  # over the lags of dataset 1000: the first lag, and where G - S turns positive
        mac_value(f"RawGapFirst{tag}", {"value": curve["value"][0], "band": [curve["band"][0][0], curve["band"][1][0]]}, 3, True)
        if gap == "G - S":
            above = np.asarray(curve["value"]) > 0
            macro("RawGapCross", num(curve["lags"][int(np.flatnonzero(~above)[-1]) + 1], 0))
            sure = np.asarray(curve["band"][0]) > 0  # the lag from which the whole band is above 0
            macro("RawGapSig", num(curve["lags"][int(np.flatnonzero(~sure)[-1]) + 1], 0))
    macro("RawFirstLag", num(r["gaps"]["1000"]["H_loc"]["G - S"]["lags"][0], 0))
    ph = r["phase statistics"]
    lines = [r"\begin{tabular}{@{}l r r r@{}}", r"\toprule", r" & $\Sl$ & $\A$ & $\G$ \\", r"\midrule"]
    for p in POP:
        macro(f"RawPhaseFlights{p}", num(ph[p]["flights"], 0))
        macro(f"RawPhaseHours{p}", num(ph[p]["phase time (h)"], 0))
    lines.append(r"flights & " + " & ".join(num(ph[p]["flights"], 0) for p in POP) + r" \\")
    lines.append(r"phase time (h) & " + " & ".join(num(ph[p]["phase time (h)"], 0) for p in POP) + r" \\")
    for s in STAT7:
        lines.append(f"{LABEL[s]} & " + " & ".join(f"{num(ph[p]['average first'][s]['value'], DIG[s])} / {num(ph[p]['ratio of sums'][s]['value'], DIG[s])}" for p in POP) + r" \\")
        for p in POP:
            mac_value(f"RawStat{TAG[s]}{p}", ph[p]["average first"][s], DIG[s])
            macro(f"RawSum{TAG[s]}{p}", num(ph[p]["ratio of sums"][s]["value"], DIG[s]))
        for gap, tag in GAPTAG.items():
            mac_value(f"RawGap{TAG[s]}{tag}", r["phase gaps"][gap][s], DDIG[s], True)
    write("raw_ratio_table", lines + [r"\bottomrule", r"\end{tabular}"])
    for p in POP:
        for phase, tag in PHASE.items():
            macro(f"RawRuns{tag}{p}", num(r["survival"][p][phase]["runs"], 0))
    macro("NBoot", num(r["n_boot"], 0))
    if "xc" in r:  # on the pieces of at least performance.XC_S / SUPPORT s
        for p in POP:
            x = r["xc"][p]
            macro(f"RawXcFlights{p}", num(x["flights"], 0))
            mac_value(f"RawStatXc{p}", x["average first"][XC], DIG[XC])
            macro(f"RawSumXc{p}", num(x["ratio of sums"][XC]["value"], DIG[XC]))
        for gap, tag in GAPTAG.items():
            mac_value(f"RawGapXc{tag}", r["xc gaps"][gap], DDIG[XC], True)


def counts_tex(c):
    m = c["chosen caliper"]
    macro("MatMatches", num(c["matches"], 0))
    macro("MatPool", num(c["pool flights"], 0))
    macro("MatMin", num(c["min pieces"], 0))
    macro("MatRho", num(c["rho"], 0))
    macro("MatCover", f"{c['main cover']:g}")
    macro("MatMonth", num(m["month within"], 0))
    macro("MatHour", num(m["hour within"], 0))
    macro("MatChosenText", CALIPER_TEXT[m["name"]])
    macro("MatMinBeg", num(m["G pieces on dataset 2760"]["beginners"], 0))
    macro("MatMinExp", num(m["G pieces on dataset 2760"]["experienced"], 0))
    for p in ("G", "A"):
        macro(f"MatWithMatch{p}", num(c["pieces with a match"][p], 0))
    lines = [r"\begin{tabular}{@{}l r r r r r r r r@{}}", r"\toprule",
             r" & \multicolumn{3}{c}{$\G$, dataset to 1000~s} & \multicolumn{3}{c}{$\G$, dataset to 2760~s} & \multicolumn{2}{c}{$\A$, all} \\",
             r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(l){8-9}",
             r"caliper & beg. & exp. & all & beg. & exp. & all & 1000 & 2760 \\", r"\midrule"]
    q = c["query pieces"]
    lines.append("pieces without a condition & " + " & ".join(num(q["G"][ds][lv], 0) for ds in ("1000", "2760") for lv in LEVELS)
                 + f" & {num(q['A']['1000']['all'], 0)} & {num(q['A']['2760']['all'], 0)} \\\\")
    for name in c["calipers"]:
        g, a = c["pieces with a control"]["G"], c["pieces with a control"]["A"]
        mark = r"$\ast$ " if name == m["name"] else ""
        lines.append(f"{mark}{CALIPER_TEXT[name]} & " + " & ".join(num(g[ds][name][lv]["pieces"], 0) for ds in ("1000", "2760") for lv in LEVELS)
                     + f" & {num(a['1000'][name]['all']['pieces'], 0)} & {num(a['2760'][name]['all']['pieces'], 0)} \\\\")
    write("matched_counts_table", lines + [r"\bottomrule", r"\end{tabular}"])


def matched_tex(m):
    a = m["levels"]["all"]
    macro("MatLongT", num(a["long T"]["G"], 0))
    for lv in LEVELS:  # the long series: one lag for G and A (matched.common_long_lag)
        T = m["levels"][lv]["long T"]
        assert T["G"] == T["A"], T
        macro(f"MatLongT{lv.capitalize()}", num(T["G"], 0))
    assert m["levels"]["beginners"]["long T"]["G"] == 1840 < m["levels"]["experienced"]["long T"]["G"]  # the forest caption says so
    for p in ("G", "A"):
        for k, v in a["counts"][p].items():
            macro(f"MatN{p}" + "".join(w.capitalize() for w in k.replace(",", "").split()), num(v, 0))
    names = {"G": r"$\G$", "S|G": r"$\Sl\mid\G$", "A": r"$\A$", "S|A": r"$\Sl\mid\A$"}
    lines = [r"\begin{tabular}{@{}l r l l l@{}}", r"\toprule",
             r" & pieces & fitted $H$, 10 to 1000~s & $\Hloc(1000~\mathrm{s})$ & $\Hloc(1840~\mathrm{s})$ \\", r"\midrule"]
    for side, tag in SIDES.items():
        n = a["counts"]["G" if side in ("G", "S|G") else "A"]
        sd = a["sides"][side]
        for q in (HK, HQ, FITK, FITT, *with_xc(sd)):
            mac_value(f"MatSide{TAG[q]}{tag}", sd[q], DIG[q])
        pieces = n["pieces"] if side in ("G", "A") else n["controls"]
        macro(f"MatSidePieces{tag}", num(pieces, 0))
        lines.append(f"{names[side]} & {num(pieces, 0)} & {cell(sd[FITK], 3)} & {cell(sd[HK], 3)} & {cell(sd[HQ], 3)} \\\\")
        if side == "S|G":
            lines.append(r"\addlinespace")
    d = a["deltas"]
    dn = {"G - S|G": ("G", r"$\G-\Sl\mid\G$"), "A - S|A": ("A", r"$\A-\Sl\mid\A$"), "(G - S|G) - (A - S|A)": ("D", "difference")}
    lines.append(r"\midrule")
    for k, (tag, text) in dn.items():
        for q in (HK, HQ, FITK, FITT, *with_xc(d[k])):
            mac_value(f"MatDelta{TAG[q]}{tag}", d[k][q], DDIG[q], True)
            macro(f"MatDelta{TAG[q]}{tag}Years", bracket(d[k][q]["years"], DDIG[q], True))
        lines.append(f"{text} & & {cell(d[k][FITK], 3, True)} & {cell(d[k][HK], 3, True)} & {cell(d[k][HQ], 3, True)} \\\\")
    lines.append(r"\midrule")
    for k, (tag, text) in list(dn.items())[:2]:  # Delta H_loc at 1000 s on the long series: its window is not cut there
        c = a["delta curves"][k]["long"]
        i = c["lags"].index(1000)
        long = {"value": c["value"][i], "band": [c["band"][0][i], c["band"][1][i]]}
        mac_value(f"MatDeltaHKLong{tag}", long, 3, True)
        lines.append(f"{text}, dataset {a['long T']['G']} & & & {cell(long, 3, True)} & \\\\")
    write("matched_table", lines + [r"\bottomrule", r"\end{tabular}"])
    for lv in LEVELS:  # the three levels, for the forest plot's text
        for k, (tag, _) in dn.items():
            macro(f"MatLevel{lv.capitalize()}{tag}HK", signed(m["levels"][lv]["deltas"][k][HK]["value"], 3))
    for p in ("G", "A"):
        for ph, tag in PHASE.items():
            for key, side in zip(("G", "S|G"), ("G", "S|G") if p == "G" else ("A", "S|A")):
                macro(f"MatRuns{tag}{SIDES[side]}", num(m["survival"][p][ph][key]["runs"], 0))
    for k, (tag, _) in dn.items():
        macro(f"MatZero{tag}", num(sum(zero_in(d[k][q]) for q in QUANT), 0))
    macro("MatNQuant", num(len(QUANT), 0))
    for p in ("G", "A"):
        if a.get("xc counts", {}).get(p):
            for k, v in a["xc counts"][p].items():
                macro(f"MatXc{k.capitalize()}{p}", num(v, 0))
    rob = m["robustness"]
    macro("MatRobLines", num(len(rob), 0))
    same = rob["main"]["long T"]  # the lines of the main long series share its window; the others are cut shorter
    for q in (HK, HQ):
        v = [x["quotes"][q] for x in rob.values() if q in x["quotes"] and (q == HK or x["long T"] == same)]
        macro(f"MatRobLines{TAG[q]}", num(len(v), 0))
        macro(f"MatRobLow{TAG[q]}", signed(min(x["value"] for x in v), 3))
        macro(f"MatRobHigh{TAG[q]}", signed(max(x["value"] for x in v), 3))
        macro(f"MatRobZero{TAG[q]}", num(sum(zero_in(x) for x in v), 0))
    short = [x for x in rob.values() if HQ in x["quotes"] and x["long T"] != same]
    macro("MatRobShortLinesHQ", num(len(short), 0))
    macro("MatRobShortTLow", num(min(x["long T"] for x in short), 0))
    macro("MatRobShortTHigh", num(max(x["long T"] for x in short), 0))
    macro("MatRobShortLowHQ", signed(min(x["quotes"][HQ]["value"] for x in short), 3))
    macro("MatRobShortHighHQ", signed(max(x["quotes"][HQ]["value"] for x in short), 3))
    macro("MatRobPiecesLow", num(min(x["pieces"] for x in rob.values()), 0))
    macro("MatRobPiecesHigh", num(max(x["pieces"] for x in rob.values()), 0))
    macro("MatRobIsolationPieces", num(rob["isolation on"]["pieces"], 0))


def decomposition_tex(dec):
    macro("DecPiecesK", num(dec["pieces"]["1000"], 0))
    macro("DecPiecesQ", num(dec["pieces"]["2760"], 0))
    macro("DecPiecesPhase", num(dec["pieces"]["phase"], 0))
    macro("DecK", num(dec["K"], 0))
    a = dec["levels"]["all"]
    cells = list(CELLS)
    core = [r"\begin{tabular}{@{}l r r r r r r r r r@{}}", r"\toprule",
            r" & \multicolumn{1}{c}{$\G$} & \multicolumn{1}{c}{raw gap} & \multicolumn{2}{c}{$\Delta(\emptyset)$} & "
            r"\multicolumn{1}{c}{$\varphi_R$} & \multicolumn{1}{c}{$\varphi_T$} & \multicolumn{1}{c}{$\varphi_W$} & "
            r"\multicolumn{2}{c}{$\Delta(F)$} \\", r"\cmidrule(lr){4-5}\cmidrule(l){9-10}",
            r" & pieces & point 1 & controls & $\Delta$ & & & & controls & $\Delta$ \\", r"\midrule"]
    ctl = [r"\begin{tabular}{@{}l r " + "r " * 8 + r"@{}}", r"\toprule", r" & $\G$ & " + " & ".join(CELL_TEXT[c] for c in cells) + r" \\", r"\midrule"]
    dl = [r"\begin{tabular}{@{}l " + "r " * 8 + r"@{}}", r"\toprule", r" & " + " & ".join(CELL_TEXT[c] for c in cells) + r" \\", r"\midrule"]
    for q in QUANT:
        x, d = a[q], DDIG[q]
        t = TAG[q]
        macro(f"DecGroup{t}", num(x["cells"]["none"]["group"], DIG[q]))
        for c, tag in CELLS.items():
            v = x["cells"][c]
            mac_value(f"DecDelta{t}{tag}", v["delta"], d, True)
            macro(f"DecCtl{t}{tag}", num(v["control"], DIG[q]))
        for k in "RTW":
            mac_value(f"DecPhi{t}{k}", x["phi"][k], d, True)
        for k, v in x["shares"].items():
            macro(f"DecShare{t}{k}", num(100 * v, 0).replace("-", "$-$"))
        macro(f"DecNone{t}", signed(x["delta_none"]["value"], d))
        macro(f"DecNone{t}Band", bracket(x["delta_none"]["band"], d, True))
        none, F = x["cells"]["none"], x["cells"]["R+T+W"]
        core.append(f"{LABEL[q]} & {num(none['group'], DIG[q])} & \\RawGap{t} & {num(none['control'], DIG[q])} & {signed(none['delta']['value'], d)} & "
                    f"{signed(x['phi']['R']['value'], d)} & {signed(x['phi']['T']['value'], d)} & {signed(x['phi']['W']['value'], d)} & "
                    f"{num(F['control'], DIG[q])} & {signed(F['delta']['value'], d)} \\\\")
        ctl.append(f"{LABEL[q]} & {num(none['group'], DIG[q])} & " + " & ".join(num(x["cells"][c]["control"], DIG[q]) for c in cells) + r" \\")
        dl.append(f"{LABEL[q]} & " + " & ".join(cell_d(x["cells"][c]["delta"], d) for c in cells) + r" \\")
    for name, rows in (("decomposition_table", core), ("decomposition_controls_table", ctl), ("decomposition_deltas_table", dl)):
        write(name, rows + [r"\bottomrule", r"\end{tabular}"])
    c = a[HK]["cells"]
    for k in ("none", "R+T+W"):
        macro(f"DecControls{CELLS[k]}", num(c[k]["controls"], 0))
        macro(f"DecControlFlights{CELLS[k]}", num(c[k]["control flights"], 0))
    macro("DecControlsR", num(c["R"]["controls"], 0))
    ph = a["time in climb (%)"]["cells"]
    macro("DecPhaseControlsNone", num(ph["none"]["controls"], 0))
    macro("DecPhaseControlsF", num(ph["R+T+W"]["controls"], 0))


def cell_d(v, d):
    """Value and a dagger when the band keeps 0 out: a Delta of the 8 cells table."""
    return signed(v["value"], d) + ("" if zero_in(v) else r"$^\ast$")


WINDOWS = {1260: "A", 1800: "B", 2400: "C", 3000: "D"}  # window length (s): tag of its macros


def windows_tex(runs):
    """Macros of g at H_loc(1000 s) and in time in climb, and the events, for each window length of WINDOWS
    (`runs`: {window: prepost JSON})."""
    for w, tag in WINDOWS.items():
        g = runs[w]["differences"]["g"]
        macro(f"PreWin{tag}Len", num(w, 0))
        macro(f"PreWin{tag}Events", num(g["events (MSD)"], 0))
        mac_value(f"PreWin{tag}HK", g[HK], 3, True)
        mac_value(f"PreWin{tag}Climb", g["time in climb (%)"], 1, True)
        d = g["dH_loc"]
        top = int(np.argmax(d["value"]))  # the lag where g(tau) peaks
        macro(f"PreWin{tag}PeakLag", num(runs[w]["lags"][top], 0))
        mac_value(f"PreWin{tag}Peak", {"value": d["value"][top], "band": [d["band"][0][top], d["band"][1][top]]}, 3, True)
        macro(f"PreWin{tag}DIn", signed(runs[w]["differences"]["D_in"][HK]["value"], 3))
        macro(f"PreWin{tag}DOut", signed(runs[w]["differences"]["D_out"][HK]["value"], 3))
        if 1840 in runs[w]["lags"]:  # g at the lag 1840 s, where the window of H_loc is cut by the lags of this run
            i = runs[w]["lags"].index(1840)
            mac_value(f"PreWin{tag}HQ", {"value": d["value"][i], "band": [d["band"][0][i], d["band"][1][i]]}, 3, True)


def phase_median(s):
    """The duration (s) at which the survival of a phase crosses 1/2: linear in the log of the duration
    between the two grid points around the crossing."""
    v, t = np.asarray(s["value"]), np.log(s["grid"])
    i = int(np.argmax(v < 0.5))
    if i == 0:
        return float(np.exp(t[0]))
    return float(np.exp(t[i - 1] + (v[i - 1] - 0.5) / (v[i - 1] - v[i]) * (t[i] - t[i - 1])))


def prepost_tex(p, m):
    lag = p["lags"].index(1000)
    st = p["curves"]["stages"]  # the lag from which g = ((start - pre) + (end - post)) / 2 of H_loc(tau) stays above 0
    g = 0.5 * ((np.array(st["start"]["H_loc"]) - np.array(st["pre"]["H_loc"])) + (np.array(st["end"]["H_loc"]) - np.array(st["post"]["H_loc"])))
    macro("PreGCross", num(p["lags"][int(np.flatnonzero(g <= 0)[-1]) + 1], 0))
    for stage, c in p["curves"]["stages"].items():
        if stage == "events":
            continue
        macro(f"PreWindows{stage.capitalize()}", num(c["windows"], 0))
        macro(f"PreFit{stage.capitalize()}", num(c["H"], 3))
        macro(f"PreHK{stage.capitalize()}", num(c["H_loc(1000 s)"], 3))  # window cut at 1000 s, as points 1-3
    macro("PreEvents", num(p["curves"]["stages"]["events"], 0))
    macro("PreWindow", num(p["window_s"], 0))
    macro("PreMaxLag", num(max(p["lags"]), 0))
    macro("PreOrigins", num(p["window_s"] / 10 - max(p["lags"]) / 10, 0))
    macro("PreLags", num(len(p["lags"]), 0))
    d = p["differences"]
    CMP = {"D_in": "In", "D_out": "Out", "post - pre": "PP", "g": "G", "d w": "Dw", "ctrlA2 - ctrlA1": "CA", "ctrlS2 - ctrlS1": "CS"}
    for k, tag in CMP.items():
        for q in (HK, *with_xc(d[k])):
            mac_value(f"Pre{tag}{TAG[q]}", d[k][q], DDIG[q], True)
        macro(f"PreEventsMSD{tag}", num(d[k]["events (MSD)"], 0))
        macro(f"PreEventsPhase{tag}", num(d[k]["events (phase)"], 0))
        macro(f"PreFlightsPhase{tag}", num(d[k]["flights (phase)"], 0))
    macro("PreEventsMSDG", num(d["g"]["events (MSD)"], 0))
    g, d2 = d["g"], m["levels"]["all"]["deltas"]["G - S|G"]
    lines = [r"\begin{tabular}{@{}l l l r r@{}}", r"\toprule",
             r" & $g$, before and after & $\Delta(\G-\Sl\mid\G)$, same route & $g-\Delta$ & same sign \\", r"\midrule"]
    for q in QUANT:
        if q == HQ:
            continue
        t = TAG[q]
        diff = g[q]["value"] - d2[q]["value"]
        macro(f"PreVsGap{t}", signed(diff, DDIG[q]))
        macro(f"PreVsSign{t}", "yes" if g[q]["value"] * d2[q]["value"] > 0 else "no")
        macro(f"PreVsInside{t}", "yes" if d2[q]["band"][0] <= g[q]["value"] <= d2[q]["band"][1] else "no")
        lines.append(f"{LABEL[q]} & {cell(g[q], DDIG[q], True)} & {cell(d2[q], DDIG[q], True)} & {signed(diff, DDIG[q])} & {'yes' if g[q]['value'] * d2[q]['value'] > 0 else 'no'} \\\\")
    write("prepost_table", lines + [r"\bottomrule", r"\end{tabular}"])
    if "mock" in p:
        mock_tex(p["mock"])


def mock_tex(m):
    """Macros of the mock events of prepost.py: per variant ("Site" in the names of the same-site one), the effects of
    the mock events (Mock: g and d w), of the real events that have one (MockSubset) and their difference (Net)."""
    macro("PreMockK", num(m["mocks per event"], 0))
    macro("PreMockPool", num(m["pool per event"], 0))
    macro("PreMockCaliper", CALIPER_TEXT[m["caliper"]])
    for variant, v in m["variants"].items():
        s, c = ("Site" if variant == "same site" else ""), v["counts"]
        macro(f"PreMock{s}Events", num(c["mock events (MSD)"], 0))
        macro(f"PreMock{s}RealEvents", num(c["real events with a mock event (MSD)"], 0))
        macro(f"PreMock{s}RealAll", num(c["real events with the four windows (MSD)"], 0))
        macro(f"PreMock{s}Flights", num(c["solo flights (MSD)"], 0))
        macro(f"PreMock{s}EventsPhase", num(c["mock events (phase)"], 0))
        macro(f"PreMock{s}RealEventsPhase", num(c["real events with a mock event (phase)"], 0))
        for kind, key, effects in (("Mock", "mock", {"G": "g", "Dw": "d w"}), ("Net", "real minus mock", {"G": "g"}),
                                   ("MockSubset", "real events with a mock", {"G": "g"})):
            for tag, effect in effects.items():
                for q in (x for x in (HK, *STAT7, XC) if x in v[key][effect]):
                    mac_value(f"Pre{kind}{s}{tag}{TAG[q]}", v[key][effect][q], DDIG[q], True)


def summary_tex(r, m, p):
    d2, g = m["levels"]["all"]["deltas"], p["differences"]["g"]
    mark = lambda v: r"$^\circ$" if zero_in(v) else ""
    lines = [r"\begin{tabular}{@{}l l l l l@{}}", r"\toprule",
             r" & raw $\G-\Sl$ & matched $\Delta(\G-\Sl\mid\G)$ & selection $\Delta(\A-\Sl\mid\A)$ & before and after $g$ \\", r"\midrule"]
    for q in QUANT:
        raw = r["gaps"]["1000" if q == HK else "2760"][q]["G - S"] if q in (HK, HQ) else r["phase gaps"]["G - S"][q]
        row = [LABEL[q], cell(raw, DDIG[q], True) + mark(raw),
               cell(d2["G - S|G"][q], DDIG[q], True) + mark(d2["G - S|G"][q]),
               cell(d2["A - S|A"][q], DDIG[q], True) + mark(d2["A - S|A"][q]),
               (cell(g[q], DDIG[q], True) + mark(g[q])) if q in g else "--"]
        lines.append(" & ".join(row) + r" \\")
    macro("RawNonZeroCount", num(sum(not zero_in(r["phase gaps"]["G - S"][q]) for q in STAT7), 0))
    write("summary_table", lines + [r"\bottomrule", r"\end{tabular}"])


def perlag_toy_tex(t):
    """The toy of the pool and the per-lag cohort (perlag_scheme.py): its parameters and its two fits."""
    macro("ToyPlAlpha", num(t["alpha"], 1))
    macro("ToyPlBeta", num(t["beta"], 1))
    macro("ToyPlTzero", num(t["t0"], 0))
    macro("ToyPlTmax", num(t["t_max"], 0))
    macro("ToyPlTau", num(t["tau"], 0))
    macro("ToyPlPool", num(t["pool"], 3))
    macro("ToyPlPerlag", num(t["perlag"], 3))
    macro("ToyPlChained", num(t["chained"], 3))
    macro("ToyPlJump", num(t["jump"], 3))
    macro("ToyPlMean", num(t["mean"], 3))
    macro("ToyPlMeanMax", num(t["mean_max"], 3))
    macro("ToyPlCohorts", ", ".join(num(x, 0) for x in t["cohorts"]))
    macro("ToyPlTop", num(t["alpha"] + t["beta"] / 4, 3))


def toy_tex():
    """The numbers of the worked examples of the appendices, computed here (not results)."""
    n, d, t = np.array([1.0, 5.0, 2.0]), np.array([10.0, 20.0, 30.0]), np.array([100.0, 400.0, 500.0])  # n, d, T of three units
    for i in range(3):
        for name, v in (("N", n), ("D", d), ("T", t)):
            macro(f"Toy{name}{"abc"[i]}", num(v[i], 0))
    macro("ToyRatio", num(n.sum() / d.sum() * 100, 1))
    macro("ToyAvgFirst", num((n / t).sum() / (d / t).sum() * 100, 1))
    macro("ToyMeanShare", num(np.mean(n / d) * 100, 1))
    macro("ToySumN", num(n.sum(), 0))
    macro("ToySumD", num(d.sum(), 0))
    a = {"R": 0.30, "T": 0.05, "W": 0.10}
    g0 = 0.01
    macro("ToyShapleyF", num(g0, 2))
    macro("ToyShapleyNone", num(g0 + sum(a.values()), 2))
    for k, v in a.items():
        macro(f"ToyShapley{k}", num(v, 2))
    pos = np.array([0.0, 3.0, 4.0, 10.0])
    macro("ToyPos", ", ".join(num(x, 0) for x in pos))
    one, two = np.mean(np.diff(pos) ** 2), np.mean((pos[2:] - pos[:-2]) ** 2)
    macro("ToyMsdOne", num(one, 2))
    macro("ToyMsdTwo", num(two, 1))
    macro("ToyH", num(np.log2(two / one) / 2, 2))
    macro("ToyDin", num(3, 0))
    macro("ToyDout", num(1, 0))
    macro("ToyG", num((3 + 1) / 2, 0))
    macro("ToyDw", num((3 - 1) / 2, 0))
    macro("ToySurvDurations", "10, 30, 30, 100")
    macro("ToySurvAtThirty", num(1 / 4, 2))
    macro("ToySurvAtTwenty", num(3 / 4, 2))
    macro("HqLowest", num(1840 * 10 ** -0.25, 0))  # the lowest and the highest lag of the window of H_loc(1840 s)
    macro("HqHighest", num(1840 * 10 ** 0.25, 0))
    a, controls = (5.0, 7.0), ((2.0, 4.0), (6.0,))  # the reweighting example: two pieces, their controls
    for k, x in zip(("A", "B", "Ca", "Cb", "Cc"), (*a, *controls[0], *controls[1])):
        macro(f"ToyRw{k}", num(x, 0))
    macro("ToyRwGroup", num(np.mean(a), 1))
    macro("ToyRwControl", num(np.mean([np.mean(c) for c in controls]), 1))
    macro("ToyRwPooled", num(np.mean([x for c in controls for x in c]), 1))
    psi, pilot, block = np.array([1.0, 1.0, -2.0]), np.array([1, 1, 2]), np.array([0, 1, 1])  # the two-way example
    by = lambda code: float(sum(psi[code == c].sum() ** 2 for c in np.unique(code)))
    v = {"P": by(pilot), "B": by(block), "C": by(pilot * 10 + block)}
    pairs = sum(psi[k] * psi[l] for k in range(3) for l in range(3) if pilot[k] == pilot[l] or block[k] == block[l])
    assert v["P"] + v["B"] - v["C"] == pairs
    for k, x in v.items():
        macro(f"ToyCgm{k}", num(x, 0))
    macro("ToyCgmV", num(pairs, 0))
    macro("ToyCgmPig", num(v["P"] + v["B"] + v["C"], 0))


def coverage_tex(cov):
    """The table of coverage.py and the ranges the appendix quotes."""
    designs, new = ("raw", "matched", "prepost"), "two-way, normal"
    old = {d: next(k for k in next(iter(cov[d]["scenarios"].values())) if k not in ("true sd", new)) for d in designs}
    lines = [r"\begin{tabular}{@{}l l l l l l l@{}}", r"\toprule",
             r" & \multicolumn{2}{c}{point 1} & \multicolumn{2}{c}{point 2} & \multicolumn{2}{c}{point 4} \\",
             r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(l){6-7}",
             r"variances & two-way & before & two-way & before & two-way & before \\", r"\midrule"]
    show = lambda r: f"{100 * r['coverage of 90%']:.0f} ({r['sd / true sd']:.2f})"
    for name, var in cov["scenarios"].items():
        cells = [show(cov[d]["scenarios"][name][k]) for d in designs for k in (new, old[d])]
        lines.append(f"{name} ({', '.join(f'{x:g}' for x in var)}) & " + " & ".join(cells) + r" \\")
    write("coverage_table", lines + [r"\bottomrule", r"\end{tabular}"])
    pick = lambda d, k, f: [r[k][f] for r in cov[d]["scenarios"].values()]
    newc = [x for d in designs for x in pick(d, new, "coverage of 90%")]
    news = [x for d in designs for x in pick(d, new, "sd / true sd")]
    macro("CovNewMin", num(100 * min(newc), 0))
    macro("CovNewMax", num(100 * max(newc), 0))
    macro("CovNewSdMin", num(min(news), 2))
    macro("CovNewSdMax", num(max(news), 2))
    macro("CovOldRawMin", num(100 * min(pick("raw", old["raw"], "coverage of 90%")), 0))
    macro("CovOldMatchedMin", num(100 * min(pick("matched", old["matched"], "coverage of 90%")), 0))
    macro("CovOldMatchedSdMin", num(min(pick("matched", old["matched"], "sd / true sd")), 1))
    macro("CovOldMatchedSdMax", num(max(pick("matched", old["matched"], "sd / true sd")), 1))
    macro("CovReuse", num(cov["matched"]["sum K^2 / sum K"], 1))
    macro("CovReplicates", num(cov["replicates"], 0))
    macro("CovDraws", num(cov["draws per family"], 0))


def definitions_tex():
    """The numbers of the definitions, from the constants of tamsd, longlags, cohorts and twoway: the grid of lags
    (GridS, LagRatio, LagDecades, LagSteps, LagFactor, LagsPerDecade, LagMin, LagMax, LagCount, LagList), the lags of the
    window of alpha (WindowLagsMin/Max: its fewest and most lags among the windows of at least 5 lags), the pieces
    (PieceMinMinutes, SupportOriginsPercent, PlusFactor = 10^HalfDecade / Support, PlusLagHours, PlusPieceHours: the
    pieces that give alpha at the longest cohort lag) and the factor of the bands (BandZ)."""
    lags = np.asarray(longlags.LONG_LAGS)
    decades = np.log10(tamsd.TAU_MAX)
    steps = round(np.log(tamsd.TAU_MAX) / np.log(longlags.STEP))
    macro("GridS", num(tamsd.GRID_S, 0))
    macro("LagRatio", f"{longlags.STEP:.3f}")
    macro("LagDecades", num(decades, 0))
    macro("LagSteps", num(steps, 0))
    macro("LagFactor", num(tamsd.TAU_MAX, 0))
    macro("LagsPerDecade", num(steps / decades, 0))
    macro("LagMin", num(lags[0], 0))
    macro("LagMax", num(lags[-1], 0))
    macro("LagCount", num(len(lags), 0))
    short = lags < tamsd.TAU_MAX  # the list breaks where the lags reach TAU_MAX
    macro("LagList", ", ".join(num(t, 0) for t in lags[short]) + r",\\ " + ", ".join(num(t, 0) for t in lags[~short]))
    size = [len(cohorts.window(t)) for t in lags]
    macro("WindowLagsMin", num(min(n for n in size if n >= WINDOW_FULL), 0))
    macro("WindowLagsMax", num(max(size), 0))
    macro("PieceMinMinutes", num(tamsd.TAU_MAX / tamsd.SUPPORT / 60, 0))
    macro("SupportOriginsPercent", num(100 * (1 - tamsd.SUPPORT), 0))
    plus = 10 ** tamsd.HALF_DECADE / tamsd.SUPPORT
    hours = cohorts.COHORTS[-1] / 3600
    macro("PlusFactor", num(plus, 1))
    macro("PlusLagHours", num(hours, 0))
    macro("PlusPieceHours", num(plus * hours, 1))
    macro("BandZ", f"{twoway.Z:.3f}")


PER_LAG_EXAMPLE = 2250  # s: the lag of the example of the per-lag cohort


def perlag_window_tex():
    """PerLagExample and PerLagCoincideEnd (the longest lag whose window ends within TAU_MAX: pool and per-lag cohort
    coincide up to it), and the table perlag_window_table: for the grid lags around PerLagExample, the longest grid lag
    of their window and the shortest piece that gives the MSD at PerLagExample there; then the pool."""
    lags = np.asarray(longlags.LONG_LAGS)
    top = lambda t: int(lags[cohorts.window(t)].max())
    macro("PerLagExample", num(PER_LAG_EXAMPLE, 0))
    macro("PerLagCoincideEnd", num(max(t for t in lags if top(t) <= tamsd.TAU_MAX), 0))
    k = int(np.flatnonzero(lags == PER_LAG_EXAMPLE)[0])
    lines = [r"\begin{tabular}{rrr}", r"\toprule",
             rf"window of $\tau$ & longest grid lag $\le\tau^+$ & pieces of $\mathrm{{MSD}}_\tau({num(PER_LAG_EXAMPLE, 0)}~\mathrm{{s}})$ \\",
             r"\midrule"]
    for t in lags[k - 1:k + 2]:
        lines.append(rf"{num(t, 0)}~s & {num(top(t), 0)}~s & $T_p\ge{num(top(t) / tamsd.SUPPORT, 0)}$~s \\")
    lines += [r"\midrule", rf"pool, any window & & $T_p\ge{num(PER_LAG_EXAMPLE / tamsd.SUPPORT, 0)}$~s \\", r"\bottomrule", r"\end{tabular}"]
    write("perlag_window_table", lines)


SAME_TOL = 1e-3  # a difference of alpha in the third decimal
RUNS_TOL = 1e-6  # the lower edge of a band above 0 beyond rounding


def pool_tex(c, s):
    """The lags read on the figures of the pool, from the JSON they plot (cohorts.json `c`, perlag.json `s`), in two
    significant figures. Against the nested cohorts: PoolSame (the first lag where the pool alpha differs from the
    cohort to TAU_MAX by SAME_TOL, in the first population to do it), PoolLeave (the lag from which the pool stays outside the 90% band
    of the cohort to SELECTION_FREE, in the first population to do it), PoolBelowEnd (the last lag where the pool of S
    lies below that band), PoolAbove{A,G} (the lag from which the pool stays above it), PoolSteep (the lag from which the
    pool lies above every cohort that reaches the lag, in the three populations). Against the chain: PoolMsdCross (the lag
    beyond TAU_MAX from which the pool MSD of S is at or above the chained one), PoolRunsAway{S,A,G} (the first lag where
    the lower edge of the band of pool minus alpha* is above 0, RUNS_TOL) and PerLagPoolRunsAway{S,A,G} (the same for
    pool minus per-lag alpha)."""
    same, leave, steep = [], [], []
    for pop, tag in POPS:
        cs = {e["tau_max"]: e for e in c["alpha"][pop]}
        pool = dict(zip(c["pool"][pop]["lags"], c["pool"][pop]["value"]))
        base = dict(zip(cs[tamsd.TAU_MAX]["lags"], cs[tamsd.TAU_MAX]["alpha"]))
        same.append(first_lag(list(base), [abs(pool[t] - base[t]) >= SAME_TOL for t in base]))
        ref = cs[cohorts.SELECTION_FREE]
        lo, hi = ref["band"]
        ts = [t for t in ref["lags"] if t in pool]
        i = [ref["lags"].index(t) for t in ts]
        below, above = [pool[t] < lo[j] for t, j in zip(ts, i)], [pool[t] > hi[j] for t, j in zip(ts, i)]
        leave.append(from_lag(ts, np.array(below) | np.array(above)))
        if tag == "S":
            macro("PoolBelowEnd", sig2(max(t for t, b in zip(ts, below) if b)))
        else:
            macro(f"PoolAbove{tag}", sig2(from_lag(ts, above)))
        reach = {t: [dict(zip(e["lags"], e["alpha"]))[t] for e in cs.values() if t in e["lags"]] for t in pool}
        ts = [t for t in pool if reach[t]]
        steep.append(from_lag(ts, [pool[t] > max(reach[t]) for t in ts]))
    macro("PoolSame", sig2(min(t for t in same if t is not None)))
    macro("PoolLeave", sig2(min(t for t in leave if t is not None)))
    macro("PoolSteep", sig2(max(steep)))
    pm = dict(zip(c["pool"]["solo"]["lags"], c["pool"]["solo"]["msd"]))
    ch = dict(zip(s["chained"]["solo"]["lags"], s["chained"]["solo"]["msd"]))
    ts = [t for t in ch if t in pm and t > tamsd.TAU_MAX]
    macro("PoolMsdCross", sig2(from_lag(ts, [pm[t] >= ch[t] for t in ts])))
    for pop, tag in POPS:
        for key, name in (("pool minus chained", "PoolRunsAway"), ("pool minus per-lag", "PerLagPoolRunsAway")):
            d = s[key][pop]
            macro(f"{name}{tag}", sig2(first_lag(d["lags"], np.array(d["band"][0]) > RUNS_TOL)))


def chain_gaps_tex(s):
    """The lags read on the figure of the gaps of alpha* (perlag.json `s`), in two significant figures, over the solid
    lags: ChainGAGrow (the lag from which the lower edge of the band of G - A stays above 0), ChainGSGrow (the lag from
    which the lower edge of the band of G - S stays above the upper edge at the lag GAP_LAGS[100]), ChainASNegative (the
    first lag where the upper edge of the band of A - S is below 0), ChainGAZero (where G - A changes sign, interpolated in log
    lag) and ChainGSSmall (the largest |G - S| up to 100 s)."""
    end = max(t for t, k in zip(s["lags"], s["solid"]) if k)
    g = {name: d for name, d in s["chained gaps"].items()}
    lags = [t for t in g["G - A"]["lags"] if t <= end]
    lower = lambda name: np.array(g[name]["band"][0][:len(lags)])
    macro("ChainGAGrow", sig2(from_lag(lags, lower("G - A") > 0)))
    top = g["G - S"]["band"][1][g["G - S"]["lags"].index(100)]
    macro("ChainGSGrow", sig2(from_lag(lags, lower("G - S") > top)))
    macro("ChainASNegative", sig2(first_lag(g["A - S"]["lags"], np.array(g["A - S"]["band"][1]) < 0)))
    ga = g["G - A"]
    sign = next(i for i, v in enumerate(ga["value"]) if v > 0)  # G - A changes sign between the lags sign - 1 and sign
    x0, x1, v0, v1 = *np.log(ga["lags"][sign - 1:sign + 1]), *ga["value"][sign - 1:sign + 1]
    macro("ChainGAZero", sig2(float(np.exp(x0 + (x1 - x0) * (0 - v0) / (v1 - v0)))))
    below = [abs(v) for t, v in zip(g["G - S"]["lags"], g["G - S"]["value"]) if t <= 100]
    macro("ChainGSSmall", num(max(below), 3))


def opening_above_one_tex():
    """OpeningAboveOne: the solo flights of the story (declared open, logger interval up to longlags.MAX_DT s) with an
    opening above 1, a path shorter than the chord; skipped when the flights table is missing."""
    if not (closure.LABELS / "flights.parquet").exists():
        return
    f = closure.flight_table()
    f = f[(f["declared"] == "open") & (f["dt_native"] <= longlags.MAX_DT + 1e-6)]
    macro("OpeningAboveOne", num(int(((f["ratio"] > 1) & ~f["group"]).sum()), 0))


if __name__ == "__main__":
    GEN.mkdir(parents=True, exist_ok=True)
    o = read(DT5, "opening")
    macro("tauStar", num(read(OUTS / MAIN, "facts")["tau_star_s"] / 60, 1))  # the grouping, whichever the flights
    labels_tex(o)
    opening_tex(o)
    cohort_gaps_tex(o)
    per_lag_tex()
    chained_tex()
    dt_tex()
    ground_tex()
    route_tex()
    prepost_ground_tex()
    weights_tex()
    carry_tex()
    ground_opening_tex()
    pilot_tex()
    phases_tex()
    refit_tex()
    labels_check_tex()
    curves_tex(DT5, "Decl")
    opened = OUTS / MAIN / f"open_min{o['min_opening']:g}"
    if (opened / "longlags.json").exists():
        curves_tex(opened, "Open")
    toy_tex()
    if (OUTS / MAIN / "perlag_scheme.json").exists():
        perlag_toy_tex(read(OUTS / MAIN, "perlag_scheme"))
    files = {n: opened / f"{n}.json" for n in ("raw", "matched_counts", "matched", "decomposition", "prepost_2400")}
    if all(f.exists() for f in files.values()):
        j = {n: json.loads(f.read_text()) for n, f in files.items()}
        raw_tex(j["raw"])
        counts_tex(j["matched_counts"])
        matched_tex(j["matched"])
        decomposition_tex(j["decomposition"])
        prepost_tex(j["prepost_2400"], j["matched"])
        runs = {w: json.loads((opened / f"prepost_{w}.json").read_text()) for w in WINDOWS if (opened / f"prepost_{w}.json").exists()}
        if len(runs) == len(WINDOWS):
            windows_tex(runs)
        summary_tex(j["raw"], j["matched"], j["prepost_2400"])
    definitions_tex()
    perlag_window_tex()
    opening_above_one_tex()
    if (DT5 / "cohorts.json").exists():
        spread_tex(DT5)
        if (DT5 / "perlag.json").exists():
            pool_tex(read(DT5, "cohorts"), read(DT5, "perlag"))
            chain_gaps_tex(read(DT5, "perlag"))
    if (opened / "coverage.json").exists():
        coverage_tex(read(opened, "coverage"))
    (GEN / "macros.tex").write_text("\n".join(rf"\newcommand{{\{k}}}{{{v}}}" for k, v in sorted(macros.items())) + "\n")
    print(f"{len(macros)} macros")
