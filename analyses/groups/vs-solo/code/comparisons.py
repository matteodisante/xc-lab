"""The tables and the numbers of comparisons/, from the JSON of the analyses: vs-solo/generated/comparisons/*.tex.

Every number quoted in the text of the report is a macro of generated/comparisons/macros.tex. The analyses run
on the declared open flights (out/<run>/open_declared) and on the open ones (out/<run>/open_min<m>).

    uv run python vs-solo/code/comparisons.py    # from analyses/groups, after opening.py and longlags.py
"""

import json

import numpy as np

from tables import MAIN, OUTS, macro, macros, num

GEN = OUTS.parent / "vs-solo" / "generated" / "comparisons"
DECLARED = OUTS / MAIN / "open_declared"
POPS = (("solo", "S"), ("group", "G"), ("group_solo", "A"))
GLOSS = {"Dist libre": "free distance", "Dist 1 pt": "distance, 1 turn point", "Dist 2 pts": "distance, 2 turn points",
         "Dist 3 pts": "distance, 3 turn points", "triangle FAI": "FAI triangle", "Quadrilatère": "quadrilateral",
         "Aller-Retour": "out and return", "Marche et Vol": "hike and fly", "Marche et Vol bouclé": "hike and fly, closed",
         "Marche et Vol FAI": "hike and fly, FAI triangle"}


def signed(x, digits):
    """x with its sign in math mode; no sign when it rounds to 0."""
    r = round(x, digits)
    return ("$+$" if r > 0 else "$-$" if r < 0 else "") + num(abs(x), digits)


def read(folder, name):
    return json.loads((folder / f"{name}.json").read_text())


def write(name, lines):
    (GEN / f"{name}.tex").write_text("\n".join(lines) + "\n")


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


def opening_tex(o):
    """Macros of opening.py and the table of the openings of the datasets of the declared open flights."""
    macro("MinOpening", f"{o['min_opening']:g}")
    macro("Stricter", f"{o['stricter']:g}")
    macro("MountainM", num(o["mountain_m"], 0))
    every = o["datasets"]["every declared open flight"]["all"]
    macro("Kept", num(every["above the threshold"], 0))
    macro("KeptShare", num(every["above the threshold (%)"], 0))
    macro("StrictShare", num(every["above the stricter threshold (%)"], 0))
    for name, tag in ((f"below {o['mountain_m']} m", "Low"), (f"{o['mountain_m']} m or more", "High")):
        b = o["by takeoff"][name]
        macro(f"Takeoff{tag}Flights", num(b["flights"], 0))
        macro(f"Takeoff{tag}Median", f"{b['quartiles'][1]:.2f}")
        macro(f"Takeoff{tag}Keep", num(b["above the threshold (%)"], 0))
        macro(f"Takeoff{tag}Strict", num(b["above the stricter threshold (%)"], 0))
    tags = dict(zip(o["datasets"], ("Every", "Fixed", "Longest")))
    lines = [r"\begin{tabular}{@{}l r r r r r r r r@{}}", r"\toprule",
             r" & \multicolumn{4}{c}{flights} & \multicolumn{4}{c}{median opening} \\",
             r"\cmidrule(lr){2-5}\cmidrule(l){6-9}", r"dataset & all & $\Sl$ & $\G$ & $\A$ & all & $\Sl$ & $\G$ & $\A$ \\", r"\midrule"]
    for name, d in o["datasets"].items():
        label = name if not name.startswith("lags up to ") else f"lags up to {num(int(name.split()[3]), 0)}~s"
        macro(f"Median{tags[name]}", f"{d['all']['quartiles'][1]:.2f}")
        macro(f"Flights{tags[name]}", num(d["all"]["flights"], 0))
        macro(f"Below{tags[name]}", num(100 - d["all"]["above the threshold (%)"], 0))
        cells = [num(d[t]["flights"], 0) if t in d else "" for _, t in POPS]
        medians = [f"{d[t]['quartiles'][1]:.2f}" if t in d else "" for _, t in POPS]
        for _, t in POPS:
            if t in d:
                macro(f"Median{tags[name]}{t}", f"{d[t]['quartiles'][1]:.2f}")
        lines.append(f"{label} & {num(d['all']['flights'], 0)} & " + " & ".join(cells) + f" & {d['all']['quartiles'][1]:.2f} & "
                     + " & ".join(medians) + r" \\")
    write("opening_table", lines + [r"\bottomrule", r"\end{tabular}"])


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
DS = {"1000": "One", "2250": "Two", "longest": "Long"}  # dataset tags of the macros: to 1000 s, to 2250 s, the longest lag
POP = ("S", "A", "G")
POPTEX = {"S": r"$\Sl$", "A": r"$\A$", "G": r"$\G$"}
PHASE = {"climb": "Climb", "search": "Search", "transition": "Glide"}
SIDES = {"G": "G", "S|G": "SG", "A": "A", "S|A": "SA"}
CELLS = {"none": "None", "R": "R", "T": "T", "R+T": "RT", "W": "W", "R+W": "RW", "T+W": "TW", "R+T+W": "F"}
CALIPER_TEXT = {"month1_hour1_class": r"month $\pm1$, hour $\pm1$, same class", "month1_hour1_level": r"month $\pm1$, hour $\pm1$, same level",
                "month2_hour2_class": r"month $\pm2$, hour $\pm2$, same class", "month2_hour2_level": r"month $\pm2$, hour $\pm2$, same level"}
CELL_TEXT = {"none": r"$\emptyset$", "R": "R", "T": "T", "R+T": "RT", "W": "W", "R+W": "RW", "T+W": "TW", "R+T+W": "RTW"}
LEVELS = ("beginners", "experienced", "all")


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


def raw_tex(r):
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
    macro("RawGapHK", signed(r["datasets"]["1000"]["populations"]["G"][HK]["value"] - r["datasets"]["1000"]["populations"]["S"][HK]["value"], 3))
    macro("RawGapHQ", signed(r["datasets"]["2250"]["populations"]["G"][HQ]["value"] - r["datasets"]["2250"]["populations"]["S"][HQ]["value"], 3))
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
        g, s_ = ph["G"]["average first"][s]["value"], ph["S"]["average first"][s]["value"]
        macro(f"RawGap{TAG[s]}", signed(g - s_, DDIG[s]))
    write("raw_ratio_table", lines + [r"\bottomrule", r"\end{tabular}"])
    for p in POP:
        for phase, tag in PHASE.items():
            macro(f"RawRuns{tag}{p}", num(r["survival"][p][phase]["runs"], 0))
    macro("NBoot", num(r["n_boot"], 0))


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
    macro("MatMinBeg", num(m["G pieces on dataset 2250"]["beginners"], 0))
    macro("MatMinExp", num(m["G pieces on dataset 2250"]["experienced"], 0))
    for p in ("G", "A"):
        macro(f"MatWithMatch{p}", num(c["pieces with a match"][p], 0))
    lines = [r"\begin{tabular}{@{}l r r r r r r r r@{}}", r"\toprule",
             r" & \multicolumn{3}{c}{$\G$, dataset to 1000~s} & \multicolumn{3}{c}{$\G$, dataset to 2250~s} & \multicolumn{2}{c}{$\A$, all} \\",
             r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(l){8-9}",
             r"caliper & beg. & exp. & all & beg. & exp. & all & 1000 & 2250 \\", r"\midrule"]
    q = c["query pieces"]
    lines.append("pieces without a condition & " + " & ".join(num(q["G"][ds][lv], 0) for ds in ("1000", "2250") for lv in LEVELS)
                 + f" & {num(q['A']['1000']['all'], 0)} & {num(q['A']['2250']['all'], 0)} \\\\")
    for name in c["calipers"]:
        g, a = c["pieces with a control"]["G"], c["pieces with a control"]["A"]
        mark = r"$\ast$ " if name == m["name"] else ""
        lines.append(f"{mark}{CALIPER_TEXT[name]} & " + " & ".join(num(g[ds][name][lv]["pieces"], 0) for ds in ("1000", "2250") for lv in LEVELS)
                     + f" & {num(a['1000'][name]['all']['pieces'], 0)} & {num(a['2250'][name]['all']['pieces'], 0)} \\\\")
    write("matched_counts_table", lines + [r"\bottomrule", r"\end{tabular}"])


def matched_tex(m):
    a = m["levels"]["all"]
    macro("MatLongT", num(a["long T"]["G"], 0))
    for p in ("G", "A"):
        for k, v in a["counts"][p].items():
            macro(f"MatN{p}" + "".join(w.capitalize() for w in k.replace(",", "").split()), num(v, 0))
    names = {"G": r"$\G$", "S|G": r"$\Sl\mid\G$", "A": r"$\A$", "S|A": r"$\Sl\mid\A$"}
    lines = [r"\begin{tabular}{@{}l r l l l@{}}", r"\toprule",
             r" & pieces & fitted $H$, 10 to 1000~s & $\Hloc(1000~\mathrm{s})$ & $\Hloc(1840~\mathrm{s})$ \\", r"\midrule"]
    for side, tag in SIDES.items():
        n = a["counts"]["G" if side in ("G", "S|G") else "A"]
        sd = a["sides"][side]
        for q in (HK, HQ, FITK, FITT, *STAT7):
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
        for q in (HK, HQ, FITK, FITT, *STAT7):
            mac_value(f"MatDelta{TAG[q]}{tag}", d[k][q], DDIG[q], True)
            macro(f"MatDelta{TAG[q]}{tag}Years", bracket(d[k][q]["years"], DDIG[q], True))
        lines.append(f"{text} & & {cell(d[k][FITK], 3, True)} & {cell(d[k][HK], 3, True)} & {cell(d[k][HQ], 3, True)} \\\\")
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
    rob = m["robustness"]
    macro("MatRobLines", num(len(rob), 0))
    for q in (HK, HQ):
        v = [x["quotes"][q] for x in rob.values()]
        macro(f"MatRobLow{TAG[q]}", signed(min(x["value"] for x in v), 3))
        macro(f"MatRobHigh{TAG[q]}", signed(max(x["value"] for x in v), 3))
        macro(f"MatRobZero{TAG[q]}", num(sum(zero_in(x) for x in v), 0))
    macro("MatRobPiecesLow", num(min(x["pieces"] for x in rob.values()), 0))
    macro("MatRobPiecesHigh", num(max(x["pieces"] for x in rob.values()), 0))
    macro("MatRobIsolationPieces", num(rob["isolation on"]["pieces"], 0))


def decomposition_tex(dec):
    macro("DecPiecesK", num(dec["pieces"]["1000"], 0))
    macro("DecPiecesQ", num(dec["pieces"]["2250"], 0))
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


def prepost_tex(p, m):
    lag = p["lags"].index(1000)
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
        for q in (HK, *STAT7):
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
    assert macros["PreVsInsideHK"] == "yes", "g must lie in the band of point 2 at H_loc(1000 s)"
    same = {q for q in QUANT if q != HQ and g[q]["value"] * d2[q]["value"] > 0}
    assert same == {HK, "time in climb (%)", "searches that end in a climb (%)", "climbs per hour"}, same  # the text names these


def summary_tex(r, m, p):
    d2, g = m["levels"]["all"]["deltas"], p["differences"]["g"]
    mark = lambda v: r"$^\circ$" if zero_in(v) else ""
    lines = [r"\begin{tabular}{@{}l r l l l@{}}", r"\toprule",
             r" & raw $\G-\Sl$ & matched $\Delta(\G-\Sl\mid\G)$ & selection $\Delta(\A-\Sl\mid\A)$ & before and after $g$ \\", r"\midrule"]
    for q in QUANT:
        pops = r["datasets"]["1000" if q == HK else "2250"]["populations"] if q in (HK, HQ) else None
        if pops:
            gap = pops["G"][q]["value"] - pops["S"][q]["value"]
            sep = pops["G"][q]["band"][0] > pops["S"][q]["band"][1] or pops["G"][q]["band"][1] < pops["S"][q]["band"][0]
        else:
            ph = r["phase statistics"]
            gap = ph["G"]["average first"][q]["value"] - ph["S"]["average first"][q]["value"]
            a, b = ph["G"]["average first"][q]["band"], ph["S"]["average first"][q]["band"]
            sep = a[0] > b[1] or a[1] < b[0]
        macro(f"SumSep{TAG[q]}", "yes" if sep else "no")
        row = [LABEL[q], signed(gap, DDIG[q]) + ("" if sep else r"$^\circ$"),
               cell(d2["G - S|G"][q], DDIG[q], True) + mark(d2["G - S|G"][q]),
               cell(d2["A - S|A"][q], DDIG[q], True) + mark(d2["A - S|A"][q]),
               (cell(g[q], DDIG[q], True) + mark(g[q])) if q in g else "--"]
        lines.append(" & ".join(row) + r" \\")
    macro("RawSepCount", num(sum(macros[f"SumSep{TAG[q]}"] == "yes" for q in STAT7), 0))
    write("summary_table", lines + [r"\bottomrule", r"\end{tabular}"])


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
    macro("HqLowest", num(1840 * 10 ** -0.25, 0))  # the lowest lag of the window of H_loc(1840 s)


if __name__ == "__main__":
    GEN.mkdir(parents=True, exist_ok=True)
    o = read(DECLARED, "opening")
    macro("tauStar", num(read(OUTS / MAIN, "facts")["tau_star_s"] / 60, 1))  # the grouping, whichever the flights
    labels_tex(o)
    opening_tex(o)
    curves_tex(DECLARED, "Decl")
    opened = OUTS / MAIN / f"open_min{o['min_opening']:g}"
    if (opened / "longlags.json").exists():
        curves_tex(opened, "Open")
    toy_tex()
    files = {n: opened / f"{n}.json" for n in ("raw", "matched_counts", "matched", "decomposition", "prepost_2400")}
    if all(f.exists() for f in files.values()):
        j = {n: json.loads(f.read_text()) for n, f in files.items()}
        raw_tex(j["raw"])
        counts_tex(j["matched_counts"])
        matched_tex(j["matched"])
        decomposition_tex(j["decomposition"])
        prepost_tex(j["prepost_2400"], j["matched"])
        summary_tex(j["raw"], j["matched"], j["prepost_2400"])
    (GEN / "macros.tex").write_text("\n".join(rf"\newcommand{{\{k}}}{{{v}}}" for k, v in sorted(macros.items())) + "\n")
    print(f"{len(macros)} macros")
