"""The macros, the table and the figures of the section on the group and the gliders around it, from
out/<run>/open_min<m>/intergroup.json: vs-solo/generated/comparisons/intergroup_macros.tex (names
start with Ig), intergroup_table.tex and out/<run>/open_min<m>/intergroup_*.pdf.

    uv run python vs-solo/code/intergroup_tables.py [_sample200]    # from analyses/groups, after intergroup.py --open
"""

import json
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parents[2]))
from figures import COLOR, GREY, LIGHT, MUTED, WIDTH, forest  # noqa: E402
from tables import MAIN, OUTS, num, signed  # noqa: E402

OPEN = OUTS / MAIN / "open_min0.1"
GEN = OUTS.parent / "vs-solo" / "generated" / "comparisons"
RADII = {"1000": "One", "2000": "Two", "5000": "Five", "10000": "Ten"}
MAIN_R = "5000"
STATE = {"solo": "Solo", "group": "Group"}
CLS = {"ALONE": "Alone", "AHEAD": "Ahead", "BEHIND": "Behind", "MIXED": "Mixed"}
CON = {"intra-group": "Intra", "inter-group": "Inter", "crowd control": "Crowd", "information, solo": "InfoSolo",
       "information, group": "InfoGroup", "both": "Both"}
CON_TEXT = {"intra-group": r"intra-group: (group, alone) $-$ (solo, alone)",
            "inter-group": r"inter-group: (solo, ahead) $-$ (solo, alone)",
            "crowd control": r"crowd control: (solo, behind) $-$ (solo, alone)",
            "information, solo": r"information, solo: (solo, ahead) $-$ (solo, behind)",
            "information, group": r"information, group: (group, ahead) $-$ (group, behind)",
            "both": r"both: (group, ahead) $-$ (solo, alone)"}
CON_SHORT = {"intra-group": "intra-group", "inter-group": "inter-group", "crowd control": "crowd control", "information, solo": "information, solo",
             "information, group": "information, group", "both": "both"}
PHASE = {"time in climb (%)": ("Climb", 1), "searches that end in a climb (%)": ("Found", 1), "glide speed (km/h)": ("Glide", 1),
         "height lost per km of glide (m/km)": ("Sink", 0)}
PHASE_LABEL = {"time in climb (%)": "time in climb\n(points)", "searches that end in a climb (%)": "searches ending\nin a climb (points)",
               "glide speed (km/h)": "glide speed\n(km/h)", "height lost per km of glide (m/km)": "height lost\nper km (m/km)"}
MIN_EVENTS = 100  # events a bin needs to enter the "meeting" of the two curves
macros = {}


def band(b, d, sign=False):
    f = signed if sign else num
    return f"[{f(b[0], d)}, {f(b[1], d)}]"


def put(name, q, d, sign=False):
    """The macros NAME and NAMEBand of a {value, band} quote; "n/a" when None."""
    macros[name], macros[name + "Band"] = ("n/a", "n/a") if q is None else ((signed if sign else num)(q["value"], d), band(q["band"], d, sign))


def step_a(a):
    macros["IgGroups"], macros["IgGroupMinutes"] = num(a["groups"], 0), num(a["group minutes"], 0)
    for key, tag in (("length", "Len"), ("width", "Wid"), ("gap ahead", "Gap"), ("gap ahead, other group", "GapOther"), ("gap ahead, solo", "GapSolo"),
                     ("cross-track", "Xt")):
        for q, t in (("q10", "Low"), ("median", "Med"), ("q90", "High")):
            v = a[key][q]
            macros[f"Ig{tag}{t}"] = "n/a" if v is None else num(v / 1000 if tag not in ("Len", "Wid", "Xt") else v, 1 if tag not in ("Len", "Wid", "Xt") else 0)
        macros[f"Ig{tag}None"] = num(100 * a[key]["none"], 0)
    for name, tag in (("group", "Group"), ("G pieces", "GPieces"), ("A", "A"), ("S", "S")):
        s = a[f"share, {name} minutes"] if name != "group" else a["share, group minutes"]
        for r, t in RADII.items():
            macros[f"IgShare{tag}{t}"] = num(100 * s[r], 0)
    for r, t in RADII.items():
        macros[f"IgShareOther{t}"] = num(100 * a["share, group minutes, other group"][r], 0)
        macros[f"IgShareSolo{t}"] = num(100 * a["share, group minutes, solo"][r], 0)
    for name, tag in (("G pieces", "GPieces"), ("A", "A"), ("S", "S")):
        macros[f"IgMinutes{tag}"] = num(a[f"share, {name} minutes"]["minutes"], 0)


def step_b(b):
    for r, t in RADII.items():
        d = b[r]
        for state, S in STATE.items():
            for c, C in CLS.items():
                cell = d["cells"][state][c]
                for k, K in (("pieces", "Pieces"), ("flights", "Flights"), ("pilots", "Pilots")):
                    macros[f"IgCell{t}{S}{C}{K}"] = num(cell[k], 0)
                put(f"IgCell{t}{S}{C}H", cell["H_loc"], 3)
        for name, N in CON.items():
            res = d["contrasts"][name]
            put(f"IgCon{t}{N}Raw", res.get("raw"), 3, True)
            put(f"IgCon{t}{N}Flight", res.get("flight"), 3, True)
            macros[f"IgCon{t}{N}Flights"] = num(res.get("flights both", 0), 0)
            for stat, (T, digits) in PHASE.items():
                put(f"IgPh{t}{N}{T}Raw", res.get("phase raw", {}).get(stat), digits, True)
                q = res.get("phase flight", {}).get(stat)
                put(f"IgPh{t}{N}{T}Flight", q, digits, True)
                macros[f"IgPh{t}{N}{T}Flights"] = num(q["flights"], 0) if q else "0"
        for c, C in CLS.items():
            for state, S in STATE.items():
                macros[f"IgClass{t}{S}{C}"] = num(d["classes"][c][state], 0)


def meeting(curve):
    """The lower edge (km) of the first bin from which the difference ahead - behind has a band that includes 0
    in every later bin (the bin "none" apart) with at least MIN_EVENTS events on both sides, or None when it does not."""
    diff, a, b = curve["a-b"], curve["a"], curve["b"]
    ok = [i for i in range(len(diff) - 1) if diff[i] and a[i]["events"] >= MIN_EVENTS and b[i]["events"] >= MIN_EVENTS]
    first = None
    for i in reversed(ok):
        if diff[i]["band"][0] <= 0 <= diff[i]["band"][1]:
            first = i
        else:
            break
    return first


def step_c(c, edges):
    for stat, T in (("search found", "Search"), ("glide: chord over time", "Speed"), ("glide: loss over chord", "Sink")):
        for state, S in STATE.items():
            curve = {k: c["events"][stat][f"{state} {k}"] if k != "a-b" else c["events"][stat][f"{state} a-b"] for k in ("a", "b", "a-b")}
            i = meeting(curve)
            macros[f"IgMeet{T}{S}"] = "none" if i is None else f"{edges[i] / 1000:g}"
            macros[f"IgMeet{T}{S}Defined"] = "yes" if i is not None else "no"
            macros[f"IgEvents{T}{S}"] = num(sum(x["events"] for x in curve["a"]), 0)
            for side, s in (("a", "A"), ("b", "B")):
                for k, x in enumerate(curve[side]):
                    scale = {"Search": 100, "Speed": 3.6, "Sink": 1000}[T]
                    q = x["estimate"]
                    macros[f"Ig{T}{S}{s}{chr(65 + k)}"] = "n/a" if q is None else num(scale * q["value"], 1 if T != "Sink" else 0)
                    macros[f"Ig{T}{S}{s}{chr(65 + k)}Events"] = num(x["events"], 0)
            for k, q in enumerate(curve["a-b"]):
                scale = {"Search": 100, "Speed": 3.6, "Sink": 1000}[T]
                macros[f"Ig{T}{S}Diff{chr(65 + k)}"] = "n/a" if q is None else signed(scale * q["value"], 1 if T != "Sink" else 0)
                macros[f"Ig{T}{S}Diff{chr(65 + k)}Band"] = "n/a" if q is None else band([scale * x for x in q["band"]], 1 if T != "Sink" else 0, True)
    for state, S in STATE.items():
        for side, s in (("a", "A"), ("b", "B")):
            for k, x in enumerate(c["hloc"][f"{state} {side}"]):
                put(f"IgHloc{S}{s}{chr(65 + k)}", x["H_loc"], 3)
                macros[f"IgHloc{S}{s}{chr(65 + k)}Pieces"] = num(x["pieces"], 0)
        for k, q in enumerate(c["hloc"][f"{state} a-b"]):
            put(f"IgHloc{S}Diff{chr(65 + k)}", q, 3, True)


def table(b):
    """The cells at the main radius with their counts, and the five contrasts with their bands."""
    t = RADII[MAIN_R]
    lines = [r"\begin{tabular}{@{}l l r r r l@{}}", r"\toprule", r"state & class & pieces & flights & pilots & $\Hk$ \\", r"\midrule"]
    for state, S in STATE.items():
        for c, C in CLS.items():
            lines.append(f"{state} & {c.lower()} & \\IgCell{t}{S}{C}Pieces & \\IgCell{t}{S}{C}Flights & \\IgCell{t}{S}{C}Pilots & "
                         f"\\IgCell{t}{S}{C}H{{}} \\IgCell{t}{S}{C}HBand \\\\")
    lines += [r"\midrule", r"\multicolumn{6}{@{}l}{$\Delta\Hk$: raw \quad within the flight (flights with both cells)} \\"]
    for name, N in CON.items():
        lines.append(f"\\multicolumn{{2}}{{@{{}}l}}{{{CON_TEXT[name]}}} & & \\multicolumn{{2}}{{l}}{{\\IgCon{t}{N}Raw{{}} \\IgCon{t}{N}RawBand}} & "
                     f"\\IgCon{t}{N}Flight{{}} \\IgCon{t}{N}FlightBand{{}} (\\IgCon{t}{N}Flights) \\\\")
    (GEN / "intergroup_table.tex").write_text("\n".join(lines + [r"\bottomrule", r"\end{tabular}"]) + "\n")


def radii_table():
    """Delta H_loc(1000 s) of the contrasts at each radius of the classes, raw and within the flight."""
    lines = [r"\begin{tabular}{@{}l l l l l@{}}", r"\toprule", r"contrast & line & $R=\IgRadLow$~km & $R=\IgRMain$~km & $R=\IgRadHigh$~km \\", r"\midrule"]
    for name, N in CON.items():
        for k, (key, label) in enumerate((("Raw", "raw"), ("Flight", "within the flight"))):
            cells = [f"\\IgCon{t}{N}{key}{{}} \\IgCon{t}{N}{key}Band" for t in ("Two", "Five", "Ten")]
            lines.append(f"{CON_SHORT[name] if k == 0 else ''} & {label} & " + " & ".join(cells) + r" \\")
        lines.append(r"\addlinespace")
    (GEN / "intergroup_radii_table.tex").write_text("\n".join(lines[:-1] + [r"\bottomrule", r"\end{tabular}"]) + "\n")


def phase_table():
    """The four phase statistics of the contrasts at the main radius, raw and within the flight."""
    t = RADII[MAIN_R]
    names = {"Climb": "time in climb (points)", "Found": "searches ending in a climb (points)", "Glide": "glide speed (km/h)", "Sink": "height lost per km (m/km)"}
    lines = [r"\begin{tabular}{@{}l l l l l@{}}", r"\toprule", r"contrast & statistic & raw & within the flight & flights \\", r"\midrule"]
    for name, N in CON.items():
        for k, (T, label) in enumerate(names.items()):
            lines.append(f"{CON_SHORT[name] if k == 0 else ''} & {label} & \\IgPh{t}{N}{T}Raw{{}} \\IgPh{t}{N}{T}RawBand & "
                         f"\\IgPh{t}{N}{T}Flight{{}} \\IgPh{t}{N}{T}FlightBand & \\IgPh{t}{N}{T}Flights \\\\")
        lines.append(r"\addlinespace")
    (GEN / "intergroup_phase_table.tex").write_text("\n".join(lines[:-1] + [r"\bottomrule", r"\end{tabular}"]) + "\n")


def save(fig, name):
    fig.savefig(OPEN / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OPEN / f"{name}.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def figure_geometry(a):
    """Left: the 10-90% range and the median of the length and width of the groups and of the gaps, on one
    distance axis (log). Right: the share of time with an outsider ahead within R."""
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(WIDTH, 2.6), constrained_layout=True, gridspec_kw={"width_ratios": [1.5, 1]})
    rows = [("length of the group", "length", COLOR["G"]), ("width of the group", "width", COLOR["G"]),
            ("gap ahead, any outsider", "gap ahead", COLOR["S"]), ("gap ahead, another group", "gap ahead, other group", GREY),
            ("gap ahead, solo glider", "gap ahead, solo", COLOR["A"])]
    for y, (label, key, color) in zip(range(len(rows))[::-1], rows):
        q = a[key]
        if q["median"] is None:
            continue
        ax.plot([q["q10"], q["q90"]], [y, y], color=color, lw=1.8)
        ax.plot(q["median"], y, "o", color=color, ms=5)
        ax.annotate(f"{100 * q['none']:.0f}% none" if q["none"] else "", (q["q90"], y), xytext=(4, 0), textcoords="offset points", va="center",
                    fontsize=6.5, color=MUTED)
    ax.set_yticks(range(len(rows))[::-1], [r[0] for r in rows])
    ax.set(xscale="log", xlabel="distance (m)")
    ax.grid(axis="x", color=LIGHT, lw=0.6)
    for name, label, color, ls in (("share, group minutes", "group, from the leader", COLOR["G"], "-"), ("share, G pieces minutes", r"$\mathrm{G}$, from the pilot", COLOR["G"], "--"),
                                   ("share, A minutes", r"$\mathrm{A}$", COLOR["A"], "-"), ("share, S minutes", r"$\mathrm{S}$", COLOR["S"], "-")):
        bx.plot([int(r) / 1000 for r in RADII], [100 * a[name][r] for r in RADII], marker="o", ms=3.5, color=color, ls=ls, label=label)
    bx.set(xscale="log", xlabel="radius $R$ (km)", ylabel="% of time, outsider ahead within $R$", xticks=[1, 2, 5, 10],
           xticklabels=["1", "2", "5", "10"])
    bx.legend(fontsize=6.5)
    bx.grid(color=LIGHT, lw=0.6)
    save(fig, "intergroup_geometry")


def figure_effects(b):
    """Forest plots of the contrasts of Step B at the main radius, raw and within the flight."""
    d = b[MAIN_R]["contrasts"]
    stats = [("H_loc(1000 s)", r"$\Delta H_{\rm loc}(1000~{\rm s})$", None), *((s, PHASE_LABEL[s], s) for s in PHASE)]
    fig, axes = plt.subplots(1, len(stats), figsize=(WIDTH + 1.2, 3.1), constrained_layout=True)
    for ax, (key, title, phase) in zip(axes, stats):
        rows = []
        for name in CON:
            for design, label, color, filled in (("raw", "", COLOR["S"], True), ("flight", "", COLOR["G"], False)):
                res = d[name]
                q = res.get(design) if phase is None else res.get("phase raw" if design == "raw" else "phase flight", {}).get(phase)
                if q is None:
                    continue
                rows.append((name if design == "raw" else "", q["value"], tuple(q["band"]), color, filled))
        forest(ax, rows, title, breaks=tuple(range(1, len(rows), 2))[:-1], ticks=3)
        ax.tick_params(axis="x", labelsize=6.5)
        if ax is not axes[0]:
            ax.set_yticklabels([])
        ax.set_title(title, fontsize=7, loc="left")
    save(fig, "intergroup_effects")


def binlabels(edges):
    """Labels of the bins between `edges` (m), in km, and of the last bin, "none" within the last edge."""
    e = [x / 1000 for x in edges]
    return [f"<{e[1]:g}"] + [f"{e[i]:g}-{e[i + 1]:g}" for i in range(1, len(e) - 1)] + [f">{e[-1]:g}"]


def figure_range(c, edges):
    """Step C: the statistic by the gap ahead (full) and behind (dashed), solo and group, for the searches ending in a climb,
    the glide speed, the height lost per km, and H_loc(1000 s) by the median gap of the piece."""
    fig, axes = plt.subplots(1, 4, figsize=(WIDTH + 1.2, 2.5), constrained_layout=True)
    spec = [("search found", "searches ending\nin a climb (%)", 100), ("glide: chord over time", "glide speed (km/h)", 3.6),
            ("glide: loss over chord", "height lost per km (m/km)", 1000)]
    labels = binlabels(edges)
    for ax, (stat, ylabel, scale) in zip(axes, spec):
        for state, color in (("solo", COLOR["S"]), ("group", COLOR["G"])):
            for side, ls, filled in (("a", "-", True), ("b", "--", False)):
                pts = [(i, x["estimate"]) for i, x in enumerate(c["events"][stat][f"{state} {side}"]) if x["estimate"] and x["events"] >= MIN_EVENTS]
                xs = [i for i, _ in pts]
                ax.errorbar(np.array(xs) + (0.06 if state == "group" else -0.06), [scale * q["value"] for _, q in pts],
                            yerr=[[scale * (q["value"] - q["band"][0]) for _, q in pts], [scale * (q["band"][1] - q["value"]) for _, q in pts]],
                            color=color, ls=ls, marker="o", ms=3.5, mfc=color if filled else "white", lw=1, capsize=1.5, elinewidth=0.8)
        ax.set(xlabel="gap (km)", ylabel=ylabel, xticks=range(len(labels)), xticklabels=labels)
        ax.tick_params(axis="x", labelsize=6, rotation=45)
        ax.grid(color=LIGHT, lw=0.5)
    ax = axes[3]
    hl = binlabels(c["params_h"])
    for state, color in (("solo", COLOR["S"]), ("group", COLOR["G"])):
        for side, ls, filled in (("a", "-", True), ("b", "--", False)):
            pts = [(i, x["H_loc"]) for i, x in enumerate(c["hloc"][f"{state} {side}"]) if x["H_loc"]]
            ax.errorbar(np.array([i for i, _ in pts]) + (0.06 if state == "group" else -0.06), [q["value"] for _, q in pts],
                        yerr=[[q["value"] - q["band"][0] for _, q in pts], [q["band"][1] - q["value"] for _, q in pts]], color=color, ls=ls,
                        marker="o", ms=3.5, mfc=color if filled else "white", lw=1, capsize=1.5, elinewidth=0.8)
    ax.set(xlabel="median gap of the piece (km)", ylabel=r"$H_{\rm loc}(1000~{\rm s})$", xticks=range(len(hl)), xticklabels=hl)
    ax.tick_params(axis="x", labelsize=6, rotation=45)
    ax.grid(color=LIGHT, lw=0.5)
    save(fig, "intergroup_range")


if __name__ == "__main__":
    r = json.loads((OPEN / f"intergroup{sys.argv[1] if len(sys.argv) > 1 else ''}.json").read_text())
    GEN.mkdir(parents=True, exist_ok=True)
    edges = r["params"]["BINS"]
    r["step C"]["params_h"] = r["params"]["H_BINS"]
    p = r["params"]
    macros.update({"IgTDir": num(p["T_DIR_S"] / 60, 0), "IgMinDisp": num(p["MIN_DISP_M"], 0), "IgRMax": num(p["R_MAX"] / 1000, 0),
                   "IgFLow": f"{p['F_LOW']:g}", "IgFHigh": f"{p['F_HIGH']:g}", "IgMinDefined": num(100 * p["MIN_DEFINED"], 0),
                   "IgRMain": num(p["R_MAIN"] / 1000, 0), "IgRadii": ", ".join(f"{x / 1000:g}" for x in p["RADII"][:-1]) + f" and {p['RADII'][-1] / 1000:g}",
                   "IgRadLow": f"{p['RADII'][1] / 1000:g}", "IgRadHigh": f"{p['RADII'][-1] / 1000:g}", "IgMinEvents": num(MIN_EVENTS, 0),
                   "IgBins": ", ".join(f"{x / 1000:g}" for x in p["BINS"][1:-1]) + f" and {p['BINS'][-1] / 1000:g}",
                   "IgHBins": ", ".join(f"{x / 1000:g}" for x in p["H_BINS"][1:-1]) + f" and {p['H_BINS'][-1] / 1000:g}",
                   **{f"IgEdge{n}": f"{x / 1000:g}" for n, x in zip(("One", "Two", "Three", "Four", "Five", "Six"), p["BINS"][1:])},
                   **{f"IgHEdge{n}": f"{x / 1000:g}" for n, x in zip(("One", "Two", "Three", "Four"), p["H_BINS"][1:])},
                   **{f"IgRad{n}": f"{x / 1000:g}" for n, x in zip(("One", "Two", "Five", "Ten"), p["RADII"])},
                   "IgBinLow": f"{p['BINS'][1] / 1000:g}", "IgBinLowTwo": f"{p['BINS'][2] / 1000:g}", "IgPieces": num(p["pieces"], 0), "IgPiecesUsed": num(p["pieces used"], 0)})
    step_a(r["step A"])
    step_b(r["step B"])
    step_c(r["step C"], edges)
    table(r["step B"])
    radii_table()
    phase_table()
    figure_geometry(r["step A"])
    figure_effects(r["step B"])
    figure_range(r["step C"], edges)
    (GEN / "intergroup_macros.tex").write_text("\n".join(rf"\newcommand{{\{k}}}{{{v}}}" for k, v in sorted(macros.items())) + "\n")
    print(len(macros), "macros")
