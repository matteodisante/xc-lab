"""The tables and the numbers of comparisons/, from the JSON of the analyses: vs-solo/generated/comparisons/*.tex.

Every number quoted in the text of the report is a macro of generated/comparisons/macros.tex. The analyses run
on the declared open flights (out/<run>/open_declared) and on the open ones (out/<run>/open_min<m>).

    uv run python vs-solo/code/comparisons.py    # from analyses/groups, after opening.py and longlags.py
"""

import json

from tables import MAIN, OUTS, macro, macros, num

GEN = OUTS.parent / "vs-solo" / "generated" / "comparisons"
DECLARED = OUTS / MAIN / "open_declared"
POPS = (("solo", "S"), ("group", "G"), ("group_solo", "A"))
GLOSS = {"Dist libre": "free distance", "Dist 1 pt": "distance, 1 turn point", "Dist 2 pts": "distance, 2 turn points",
         "Dist 3 pts": "distance, 3 turn points", "triangle FAI": "FAI triangle", "Quadrilatère": "quadrilateral",
         "Aller-Retour": "out and return", "Marche et Vol": "hike and fly", "Marche et Vol bouclé": "hike and fly, closed",
         "Marche et Vol FAI": "hike and fly, FAI triangle"}


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
    tags = dict(zip(o["datasets"], ("Every", "Thousand", "Fixed", "Longest")))
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
    main, long = read(folder, "tamsd")["para"], read(folder, "longlags")
    macro(f"{prefix}MinFlights", num(long["min_flights"], 0))
    lines = [r"\begin{tabular}{@{}l r r r@{}}", r"\toprule", r"dataset & $\Sl$ & $\G$ & $\A$ \\", r"\midrule",
             "lags up to 1000~s & " + " & ".join(num(main[k]["flights"], 0) for k, _ in POPS) + r" \\"]
    for k, tag in POPS:
        macro(f"{prefix}Flights{tag}", num(main[k]["flights"], 0))
        macro(f"{prefix}HThousand{tag}", f"{main[k]['H_loc'][-1]:.3f}")
    for name, panel in long["panels"].items():
        p = name.capitalize()
        macro(f"{prefix}Tau{p}", num(panel["tau_max"], 0))
        lines.append(f"lags up to {num(panel['tau_max'], 0)}~s & " + " & ".join(num(panel["curves"][k]["flights"], 0) for k, _ in POPS) + r" \\")
        for k, tag in POPS:
            c = panel["curves"][k]
            macro(f"{prefix}{p}Flights{tag}", num(c["flights"], 0))
            macro(f"{prefix}{p}H{tag}", f"{c['H_loc'][-1]:.3f}")
            macro(f"{prefix}{p}H{tag}Band", f"[{c['H_loc_band'][0][-1]:.3f}, {c['H_loc_band'][1][-1]:.3f}]")
    write(f"{prefix.lower()}_datasets_table", lines + [r"\bottomrule", r"\end{tabular}"])


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
    (GEN / "macros.tex").write_text("\n".join(rf"\newcommand{{\{k}}}{{{v}}}" for k, v in sorted(macros.items())) + "\n")
    print(f"{len(macros)} macros")
