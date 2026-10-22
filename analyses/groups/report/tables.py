"""The tables and the numbers of the report, from the JSON of the analyses: report/generated/*.tex.

Every number quoted in the text of the report is a macro of generated/macros.tex, so that the text
follows the analyses.

    uv run python report/tables.py    # from analyses/groups, after the scripts of the report
"""

import json
from pathlib import Path

import numpy as np
from scipy.stats import binom

HERE = Path(__file__).parent
OUTS = HERE.parent / "out"
MAIN, Q2 = "d120_eh500_ez300_th45_g120_q0.1", "d120_eh500_ez300_th45_g120_q0.2"
GEN = HERE / "generated"
STATS = {  # name in the JSON: (label, digits)
    "H_loc(100 s)": (r"$H_\mathrm{loc}(100~\mathrm{s})$", 3),
    "H_loc(1000 s)": (r"$H_\mathrm{loc}(1000~\mathrm{s})$", 3),
    "time in climb (%)": ("time in climb (\\%)", 1),
    "searches that end in a climb (%)": ("searches ending in a climb (\\%)", 1),
    "climb rate (m/s)": ("climb rate (m/s)", 2),
    "glide speed (km/h)": ("glide speed (km/h)", 1),
    "glide ratio": ("glide ratio", 1),
    "cross-country speed over 30 min (km/h)": ("cross-country speed (km/h)", 1),
    "height lost per km of glide (m/km)": ("height lost per km of glide (m/km)", 0),
}
SINK = "height lost per km of glide (m/km)"
# The statistics of the stratified contrasts: the glide ratio is replaced by its inverse (decompose.SINK).
DEC_STATS = ("H_loc(100 s)", "H_loc(1000 s)", "time in climb (%)", "searches that end in a climb (%)", "climb rate (m/s)",
             "glide speed (km/h)", SINK, "cross-country speed over 30 min (km/h)")
CONTRASTS = {"raw": r"$\G-\Sl$, pooled", "same site-day-hour": r"$\G-\Sl$, same site-day-hour",
             "selection": r"$\A-\Sl$, same site-day-hour", "group, same flight": r"$\G-\A$, same flight",
             "group, same site-day-hour": r"$\G-\A$, same site-day-hour"}
macros = {}


def load(run, name):
    path = OUTS / run / f"{name}.json"
    return json.loads(path.read_text()) if path.exists() else None


def num(x, digits):
    return f"{x:,.{digits}f}".replace(",", r"\,")


def signed(x, digits):
    return ("$+$" if x > 0 else "$-$" if x < 0 else "") + num(abs(x), digits)


def band(v, digits, sign=False):
    f = signed if sign else num
    return f"{f(v['value'], digits)} [{f(v['band'][0], digits)}, {f(v['band'][1], digits)}]"


def macro(name, value):
    macros[name] = value
    return value


def at(r, lag):
    """H_loc of a TA-MSD result at `lag`."""
    return r["H_loc"][int(np.searchsorted(np.array(r["lags"]), lag))]


def at_band(r, lag):
    """The band of H_loc at `lag`."""
    k = int(np.searchsorted(np.array(r["lags"]), lag))
    return r["H_loc_band"][0][k], r["H_loc_band"][1][k]


def write(name, text):
    (GEN / f"{name}.tex").write_text(text + "\n")


def facts_tex():
    f = load(MAIN, "facts")
    p, h = f["para"], f["hang"]
    write("n_flights", f"{num(f['flights']['para'], 0)} paraglider and {num(f['flights']['hang'], 0)} hang-glider flights")
    macro("tauStar", num(f["tau_star_s"] / 60, 1))
    hours = p["group hours by size"]
    total = sum(hours.values())
    shares = ", ".join(f"{k[1:]} pilots {100 * v / total:.0f}\\%" for k, v in hours.items())
    q = f["pair interval minutes (quantiles 10, 50, 90)"]
    e = f["episode flights (quantiles 50, 90, 99, max)"]
    d = f["|ell_med| seconds (quantiles 25, 50, 75, 90)"]
    write("grouping_facts", (
        f"Of the {num(p['eligible flights'], 0)} paraglider flights with a logger of at most 10~s, "
        f"{num(p['flights ever in a group'], 0)} ({100 * p['share ever in a group']:.1f}\\%) fly in a group at some "
        f"point; of the {num(h['eligible flights'], 0)} hang-glider flights, {num(h['flights ever in a group'], 0)} "
        f"({100 * h['share ever in a group']:.1f}\\%). Group time is {100 * p['share of the time in a group, all flights']:.1f}\\% "
        f"of all paraglider flight time after the departure, and on average "
        f"{100 * p['share of the time in a group, group flights']:.0f}\\% of the time of a group flight. "
        f"The {num(f['pair intervals'], 0)} pair intervals last {q[0]:.0f} to {q[2]:.0f}~min (10th to 90th percentile; "
        f"median {q[1]:.0f}~min) and form {num(f['configurations'], 0)} configurations in {num(f['episodes'], 0)} "
        f"episodes. Half of the episodes have two flights, 1\\% have more than {e[2]:.0f}, the largest has {e[3]}. By the size "
        f"of the configuration, the paraglider group time is: {shares} (the largest configuration has "
        f"{p['largest configuration']} pilots). In contact, one pilot is behind the other by a median of "
        f"{d[1]:.0f}~s (75\\% within {d[2]:.0f}~s)."))


def raw_tex():
    """Pooled S, G, A for the drawn statistics."""
    t, b, p = load(MAIN, "tamsd")["para"], load(MAIN, "budget"), load(MAIN, "performance")
    rows = []
    pops = (("solo", r"solo flights $\Sl$"), ("group", r"group time $\G$"), ("group_solo", r"same flights alone $\A$"))
    rows.append("flights (TA-MSD) & " + " & ".join(num(t[k]["flights"], 0) for k, _ in pops) + r" \\")
    rows.append(r"$H$ (10--1000~s) & " + " & ".join(
        "{:.3f} [{:.3f}, {:.3f}]".format(t[k]["H"], *t[k]["H_band"]) for k, _ in pops) + r" \\")
    for lag in (100, 1000):
        rows.append(STATS[f"H_loc({lag} s)"][0] + " & " + " & ".join(
            "{:.3f} [{:.3f}, {:.3f}]".format(at(t[k], lag), *at_band(t[k], lag)) for k, _ in pops) + r" \\")
    for src, names in ((b, ("time in climb (%)", "searches that end in a climb (%)")),
                       (p, ("climb rate (m/s)", "glide speed (km/h)", "glide ratio", "cross-country speed over 30 min (km/h)"))):
        for s in names:
            label, digits = STATS[s]
            rows.append(label + " & " + " & ".join(band(src[k][s], digits) for k, _ in pops) + r" \\")
    write("raw_table", "\n".join([r"\begin{tabular}{@{}l l l l@{}}", r"\toprule",
                                  " & " + " & ".join(lbl for _, lbl in pops) + r" \\", r"\midrule", *rows,
                                  r"\bottomrule", r"\end{tabular}"]))
    for k, name in (("solo", "S"), ("group", "G"), ("group_solo", "A")):
        macro(f"HlocThousand{name}", f"{at(t[k], 1000):.3f}")
        macro(f"HlocHundred{name}", f"{at(t[k], 100):.3f}")
        macro(f"H{name}", f"{t[k]['H']:.3f}")
        macro(f"Climb{name}", num(b[k]["time in climb (%)"]["value"], 1))
        macro(f"Found{name}", num(b[k]["searches that end in a climb (%)"]["value"], 1))
        macro(f"Rate{name}", num(p[k]["climb rate (m/s)"]["value"], 2))
        macro(f"Speed{name}", num(p[k]["glide speed (km/h)"]["value"], 1))
        macro(f"Ratio{name}", num(p[k]["glide ratio"]["value"], 1))
        macro(f"Xc{name}", num(p[k]["cross-country speed over 30 min (km/h)"]["value"], 1))


def decompose_tex(run=MAIN, name="decompose_table"):
    d = load(run, "decompose")
    if d is None:
        return None
    header = (r"\makecell{$\G-\Sl$\\pooled} & \makecell{$\G-\Sl$\\same s-d-h} & \makecell{$\A-\Sl$\\same s-d-h} & "
              r"\makecell{$\G-\A$\\same flight} & \makecell{$\G-\A$\\same s-d-h}")
    rows = []
    short = {"searches that end in a climb (%)": "searches to climb (\\%)", SINK: "height lost per km (m/km)",
             "cross-country speed over 30 min (km/h)": "cross-country speed (km/h)"}
    for s in DEC_STATS:
        label, digits = STATS[s]
        label = short.get(s, label)
        digits += 1 if digits < 3 else 0
        cells = [r"\makecell{" + signed(d[c][s]["value"], digits) + r"\\{\scriptsize[" + signed(d[c][s]["band"][0], digits)
                 + ", " + signed(d[c][s]["band"][1], digits) + "]}}" for c in CONTRASTS]
        rows.append(label + " & " + " & ".join(cells) + r" \\[2pt]")
    write(name, "\n".join([r"\begin{tabular}{@{}l c c c c c@{}}", r"\toprule", r"$\Delta$ & " + header + r" \\",
                           r"\midrule", *rows, r"\bottomrule", r"\end{tabular}"]))
    return d


SIZES = {"n2": "2", "n3": "3", "n4": "4", "n5-6": "5--6", "n7+": "7+"}


def size_tex():
    """Pooled by the size of the configuration: flights, H_loc(1000 s) in group time and for the same
    flights alone, and the drawn ratios."""
    t, b, p = load(MAIN, "size_tamsd")["para"], load(MAIN, "size_budget"), load(MAIN, "size_performance")
    rows = []
    for k, label in (("solo", "solo"), *SIZES.items()):
        alone = t.get(f"{k}/alone")
        rows.append(" & ".join([label, num(t[k]["flights"], 0), f"{t[k]['H']:.3f}", f"{alone['H']:.3f}" if alone else "",
                                f"{at(t[k], 1000):.3f}", f"{at(alone, 1000):.3f}" if alone else "",
                                num(b[k]["time in climb (%)"]["value"], 1),
                                num(b[k]["searches that end in a climb (%)"]["value"], 1),
                                num(p[k]["climb rate (m/s)"]["value"], 2), num(p[k]["glide speed (km/h)"]["value"], 1),
                                num(p[k]["glide ratio"]["value"], 1)]) + r" \\")
    write("size_table", "\n".join([
        r"\begin{tabular}{@{}l r c c c c c c c c c@{}}", r"\toprule",
        r" & & \multicolumn{2}{c}{$H$} & \multicolumn{2}{c}{$H_\mathrm{loc}(1000~\mathrm{s})$} & & & & & \\",
        r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
        r"pilots & flights & $\G_n$ & alone & $\G_n$ & alone & \makecell{time in\\climb (\%)} & "
        r"\makecell{searches\\to climb (\%)} & \makecell{climb\\rate (m/s)} & \makecell{glide speed\\(km/h)} & "
        r"\makecell{glide\\ratio} \\",
        r"\midrule", *rows, r"\bottomrule", r"\end{tabular}"]))
    for k, name in (("n2", "Two"), ("n7+", "Seven")):
        macro(f"HlocThousand{name}", f"{at(t[k], 1000):.3f}")
        macro(f"Ratio{name}", num(p[k]["glide ratio"]["value"], 1))
        macro(f"Rate{name}", num(p[k]["climb rate (m/s)"]["value"], 2))
    alone = [at(t[f"{k}/alone"], 1000) for k in SIZES]
    macro("HlocThousandAloneMin", f"{min(alone):.3f}")
    macro("HlocThousandAloneMax", f"{max(alone):.3f}")


def decompose_size_tex():
    d = load(MAIN, "decompose_size")
    if d is None:
        return
    rows = []
    for s in ("H_loc(100 s)", "H_loc(1000 s)", "time in climb (%)", "searches that end in a climb (%)", "climb rate (m/s)",
              "glide speed (km/h)", SINK):
        label, digits = STATS[s]
        digits += 1 if digits < 3 else 0
        cells = [signed(d[f"{k} - A, same flight"][s]["value"], digits) for k in SIZES]
        cells += [signed(d[f"{k} - n2, same site-day-hour"][s]["value"], digits) for k in list(SIZES)[1:]]
        rows.append(label + " & " + " & ".join(cells) + r" \\")
    write("decompose_size_table", "\n".join([
        r"\resizebox{\textwidth}{!}{\begin{tabular}{@{}l *{9}{l}@{}}", r"\toprule",
        r" & \multicolumn{5}{c}{$\G_n-\A$, same flight} & \multicolumn{4}{c}{$\G_n-\G_2$, same site-day-hour} \\",
        r"\cmidrule(lr){2-6}\cmidrule(l){7-10}",
        "$n$ & " + " & ".join(SIZES.values()) + " & " + " & ".join(list(SIZES.values())[1:]) + r" \\",
        r"\midrule", *rows, r"\bottomrule", r"\end{tabular}}"]))
    for k, name in (("n2", "Two"), ("n7+", "Seven")):
        v = d[f"{k} - A, same flight"]["H_loc(1000 s)"]
        macro(f"DHlocThousandFlight{name}", signed(v["value"], 3))
        macro(f"DHlocThousandFlight{name}Band", f"[{signed(v['band'][0], 3)}, {signed(v['band'][1], 3)}]")
        v = d[f"{k} - A, same flight"][SINK]
        macro(f"DSinkFlight{name}", signed(v["value"], 1))
    middle = [d[f"{k} - n2, same site-day-hour"]["H_loc(1000 s)"]["value"] for k in ("n3", "n4", "n5-6")]
    macro("DHlocThousandMiddleMin", signed(min(middle), 3))
    macro("DHlocThousandMiddleMax", signed(max(middle), 3))
    climb = [d[f"{k} - n2, same site-day-hour"]["time in climb (%)"]["value"] for k in ("n3", "n4", "n5-6", "n7+")]
    macro("DClimbSizeMin", signed(min(climb), 1))
    macro("DClimbSizeMax", signed(max(climb), 1))
    sink = [d[f"{k} - n2, same site-day-hour"][SINK]["value"] for k in ("n3", "n4", "n5-6", "n7+")]
    macro("DSinkSizeMin", signed(max(sink), 0))
    macro("DSinkSizeMax", signed(min(sink), 0))
    v = d["n7+ - n2, same site-day-hour"]["cross-country speed over 30 min (km/h)"]
    macro("DXcSevenTwoSdh", signed(v["value"], 1))
    macro("DXcSevenTwoSdhBand", f"[{signed(v['band'][0], 1)}, {signed(v['band'][1], 1)}]")
    v = d["n7+ - n2, same site-day-hour"]["H_loc(1000 s)"]
    macro("DHlocThousandSevenTwoSdh", signed(v["value"], 3))
    macro("DHlocThousandSevenTwoSdhBand", f"[{signed(v['band'][0], 3)}, {signed(v['band'][1], 3)}]")
    v = d["n7+ - n2, same site-day-hour"][SINK]
    macro("DSinkSevenTwoSdh", signed(v["value"], 1))
    macro("DSinkSevenTwoSdhBand", f"[{signed(v['band'][0], 1)}, {signed(v['band'][1], 1)}]")


def siteday_tex():
    days = load(MAIN, "siteday_tamsd")
    rows = []
    for day, r in days.items():
        rows.append(day.replace("Méruz", r"M\'eruz") + " & " + " & ".join(
            f"{r[k]['H_loc'][-1]:.3f}" if k in r else "" for k in ("solo", *SIZES)) + r" \\")
    write("siteday_table", "\n".join([r"\begin{tabular}{@{}l l l l l l l@{}}", r"\toprule",
                                      "site-day & solo & " + " & ".join(SIZES.values()) + r" \\", r"\midrule", *rows,
                                      r"\bottomrule", r"\end{tabular}"]))


def mechanisms_tex():
    m = load(MAIN, "mechanisms")
    if m is None:
        return
    ph = m["phases"]
    rows = []
    for key, label in (("entry", "entering a group"), ("exit", "leaving a group"), ("time alone", "share of time, alone"),
                       ("time in a group", "share of time, in a group")):
        rows.append(label + " & " + " & ".join("{:.1f} [{:.1f}, {:.1f}]".format(100 * ph[key][c]["value"],
                                                                                  *(100 * x for x in ph[key][c]["band"]))
                                                for c in ("climb", "search", "transition")) + r" \\")
    write("phases_table", "\n".join([r"\begin{tabular}{@{}l l l l@{}}", r"\toprule", r" & climb (\%) & search (\%) & "
                                     r"transition (\%) \\", r"\midrule", *rows, r"\bottomrule", r"\end{tabular}"]))
    c = ph["counts"]
    macro("Entries", num(c["entry"], 0))
    macro("Exits", num(c["exit"], 0))
    macro("AtDeparture", num(c["at departure"], 0))
    macro("AtLanding", num(c["at landing"], 0))
    macro("EntriesEarly", f"{100 * ph['entries within 10 min of the departure']:.0f}")
    for key, name in (("entry", "Entry"), ("exit", "Exit"), ("time alone", "Alone"), ("time in a group", "InGroup")):
        for c_, cn in (("climb", "Climb"), ("search", "Search"), ("transition", "Transition")):
            macro(f"Phase{name}{cn}", f"{100 * ph[key][c_]['value']:.1f}")
    ld = m["leaders"]
    macro("PairConfigurations", num(ld["configurations"], 0))
    macro("Abreast", f"{100 * ld['abreast']:.0f}")
    q = ld["delay quantiles (s)"]
    macro("DelayMedian", f"{q[1]:.0f}")
    macro("DelayNinety", f"{q[3]:.0f}")
    e = ld["leader - follower, same configuration"]
    rows = []
    for s in DEC_STATS:
        label, digits = STATS[s]
        if s in e:
            rows.append(label + " & " + band(e[s], digits + (1 if digits < 3 else 0), True) + r" \\")
    write("leaders_table", "\n".join([r"\begin{tabular}{@{}l l@{}}", r"\toprule", r"$\Delta$ & leader $-$ follower \\",
                                      r"\midrule", *rows, r"\bottomrule", r"\end{tabular}"]))
    for s, name, digits in (("H_loc(1000 s)", "HlocThousand", 3), ("H_loc(100 s)", "HlocHundred", 3),
                            ("searches that end in a climb (%)", "Found", 1), ("time in climb (%)", "Climb", 1),
                            ("climb rate (m/s)", "Rate", 2), ("glide speed (km/h)", "Speed", 1), (SINK, "Sink", 0)):
        d = digits + (1 if digits < 3 else 0)
        macro(f"Lead{name}", band(e[s], d, True))
    h = m["habitual"]
    macro("HabitualPilots", num(h["pilots"], 0))
    macro("HabitualRoles", num(h["roles"], 0))
    macro("HabitualDispersion", f"{h['dispersion']:.2f}")
    macro("HabitualNull", f"{h['null mean']:.2f}")
    macro("HabitualNullHigh", f"{h['null 95%']:.2f}")
    fr, n = np.array(h["lead fractions"]), np.array(h["roles per pilot"])
    macro("HabitualExtreme", f"{100 * np.mean((fr <= 0.2) | (fr >= 0.8)):.0f}")
    low, high = binom.cdf(np.floor(0.2 * n + 1e-9), n, 0.5), binom.sf(np.ceil(0.8 * n - 1e-9) - 1, n, 0.5)
    macro("HabitualExtremeNull", f"{100 * np.mean(low + high):.0f}")
    hours = m["hours"]
    frac = np.array(hours["fraction in a group"])
    macro("GroupShareNoon", f"{100 * np.nanmax(frac[10:15]):.0f}")
    macro("GroupShareNoonHour", f"{10 + int(np.nanargmax(frac[10:15]))}")
    macro("GroupShareEvening", f"{100 * frac[18]:.0f}")
    macro("GroupShareMorning", f"{100 * frac[9]:.0f}")


def robustness_tex():
    """The main numbers with q = 0.1 and q = 0.2, one row each, the two runs in columns."""
    runs = [(load(r, "tamsd"), load(r, "decompose")) for r in (MAIN, Q2)]
    if any(t is None or d is None for t, d in runs):
        return
    rows = [("group flights (TA-MSD)", lambda t, d: num(t["para"]["group"]["flights"], 0))]
    for k, label in (("solo", r"$\Sl$"), ("group_solo", r"$\A$"), ("group", r"$\G$")):
        rows.append((r"$H_\mathrm{loc}(1000~\mathrm{s})$, " + label, lambda t, d, k=k: f"{at(t['para'][k], 1000):.3f}"))
    for s_, c, label, digits in (
            ("H_loc(1000 s)", "group, same flight", r"$\Delta H_\mathrm{loc}(1000~\mathrm{s})$, $\G-\A$ same flight", 3),
            ("H_loc(1000 s)", "group, same site-day-hour", r"$\Delta H_\mathrm{loc}(1000~\mathrm{s})$, $\G-\A$ same s-d-h", 3),
            ("H_loc(1000 s)", "selection", r"$\Delta H_\mathrm{loc}(1000~\mathrm{s})$, $\A-\Sl$ same s-d-h", 3),
            ("H_loc(100 s)", "group, same flight", r"$\Delta H_\mathrm{loc}(100~\mathrm{s})$, $\G-\A$ same flight", 3),
            ("H_loc(100 s)", "group, same site-day-hour", r"$\Delta H_\mathrm{loc}(100~\mathrm{s})$, $\G-\A$ same s-d-h", 3),
            ("searches that end in a climb (%)", "group, same site-day-hour", r"$\Delta$ searches to climb, $\G-\A$ same s-d-h (points)", 1),
            ("climb rate (m/s)", "group, same site-day-hour", r"$\Delta$ climb rate, $\G-\A$ same s-d-h (m/s)", 3),
            (SINK, "group, same site-day-hour", r"$\Delta$ height lost per km, $\G-\A$ same s-d-h (m/km)", 1),
            ("cross-country speed over 30 min (km/h)", "group, same site-day-hour", r"$\Delta$ cross-country speed, $\G-\A$ same s-d-h (km/h)", 2)):
        rows.append((label, lambda t, d, s_=s_, c=c, digits=digits: band(d[c][s_], digits, True)))
    body = [label + " & " + " & ".join(f(t, d) for t, d in runs) + r" \\" for label, f in rows]
    write("robustness_table", "\n".join([r"\begin{tabular}{@{}l l l@{}}", r"\toprule",
                                          r" & $q=0.1$, $\tau^*=38.1$~min & $q=0.2$, $\tau^*=26.4$~min \\", r"\midrule",
                                          *body, r"\bottomrule", r"\end{tabular}"]))


def zones_tex():
    """The fitted H and the differences over every lag, by stratification (zones.py)."""
    z = load(MAIN, "zones")
    if z is None:
        return

    def cell(value, band_, digits=3):
        return r"\makecell{" + signed(value, digits) + r"\\{\scriptsize[" + signed(band_[0], digits) + ", " \
            + signed(band_[1], digits) + "]}}"

    rows = []
    for name, r in z.items():
        S, A = r["S"], r["A"]
        k = int(np.searchsorted(np.array(S["lags"]), 1000))
        loc = lambda e, key: e[key][k]
        cells = [name.replace(" x ", r" $\times$ "), f"{100 * S['share of G kept']:.0f}", f"{S['G']['H']:.3f}",
                 f"{S['other']['H']:.3f}", f"{A['other']['H']:.3f}",
                 signed(S["dH_support"], 3), cell(S["dH"], S["dH_band"]), cell(A["dH"], A["dH_band"]),
                 signed(loc(S, "dH_loc_support"), 3),
                 cell(loc(S, "dH_loc"), [S["dH_loc_band"][0][k], S["dH_loc_band"][1][k]]),
                 cell(loc(A, "dH_loc"), [A["dH_loc_band"][0][k], A["dH_loc_band"][1][k]])]
        rows.append(" & ".join(cells) + r" \\[2pt]")
    write("zones_table", "\n".join([
        r"\begin{tabular}{@{}l r c c c c c c c c c@{}}", r"\toprule",
        r" & & \multicolumn{3}{c}{fitted $H$ (10--1000 s)} & \multicolumn{3}{c}{$\Delta H$} & "
        r"\multicolumn{3}{c}{$\Delta H_\mathrm{loc}(1000~\mathrm{s})$} \\",
        r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}\cmidrule(l){9-11}",
        r"compared within & \makecell{G\\kept (\%)} & $\G$ & $\Sl\mid\G$ & $\A\mid\G$ & "
        r"\makecell{$\G-\Sl$\\as is} & \makecell{$\G-\Sl$\\reweighted} & \makecell{$\G-\A$\\reweighted} & "
        r"\makecell{$\G-\Sl$\\as is} & \makecell{$\G-\Sl$\\reweighted} & \makecell{$\G-\A$\\reweighted} \\",
        r"\midrule", *rows, r"\bottomrule", r"\end{tabular}"]))
    names = {"none": "None", "zone": "Zone", "zone x month": "ZoneMonth", "zone x hour": "ZoneHour",
             "site-day x hour": "Sdh"}
    for name, r in z.items():
        n, S, A = names[name], r["S"], r["A"]
        k = int(np.searchsorted(np.array(S["lags"]), 1000))
        macro(f"ZoneKept{n}", f"{100 * S['share of G kept']:.0f}")
        macro(f"ZoneDHsupport{n}", signed(S["dH_support"], 3))
        macro(f"ZoneRedH{n}", f"{100 * (1 - S['dH'] / S['dH_support']):.0f}")
        macro(f"ZoneRedLoc{n}", f"{100 * (1 - S['dH_loc'][k] / S['dH_loc_support'][k]):.0f}")
        macro(f"ZoneDHlocSupport{n}", signed(S["dH_loc_support"][k], 3))
        for c, e in (("S", S), ("A", A)):
            macro(f"ZoneDH{c}{n}", signed(e["dH"], 3))
            macro(f"ZoneDH{c}{n}Band", f"[{signed(e['dH_band'][0], 3)}, {signed(e['dH_band'][1], 3)}]")
            macro(f"ZoneDHloc{c}{n}", signed(e["dH_loc"][k], 3))
            macro(f"ZoneDHloc{c}{n}Band", f"[{signed(e['dH_loc_band'][0][k], 3)}, {signed(e['dH_loc_band'][1][k], 3)}]")
            macro(f"ZoneDHlocHundred{c}{n}", signed(e["dH_loc"][int(np.searchsorted(np.array(S['lags']), 100))], 3))
        macro(f"ZoneH{n}G", f"{S['G']['H']:.3f}")
        macro(f"ZoneH{n}S", f"{S['other']['H']:.3f}")
        macro(f"ZoneH{n}A", f"{A['other']['H']:.3f}")
    lags = np.array(z["none"]["S"]["lags"])
    low = lags <= 140
    macro("ZoneShortMax", signed(max(max(z[n]["S"]["dH_loc"][i] for i in np.flatnonzero(low)) for n in z), 3))
    macro("ZoneShortMin", signed(min(min(z[n]["S"]["dH_loc"][i] for i in np.flatnonzero(low & (lags >= 20))) for n in z), 3))


def redundancy_tex():
    """Which statistics are drawn and why (redundancy.py)."""
    r = load(MAIN, "redundancy")
    if r is None:
        return

    def tex(name):
        return name.replace("(%)", r"(\%)")

    rows = [f"{tex(c)} & drawn & {r['r2_of_the_kept_on_the_kept'][c]:.2f} " + r"\\" for c in r["kept"]]
    rows += [f"{tex(d['statistic'])} & dropped, number {i + 1}, R$^2$ {d['r2_at_the_time']:.2f} when dropped & "
             f"{r['r2_of_the_dropped_on_the_kept'][d['statistic']]:.2f} " + r"\\" for i, d in enumerate(r["dropped"])]
    write("redundancy_table", "\n".join([r"\begin{tabular}{@{}l l c@{}}", r"\toprule",
                                          r"statistic & status & R$^2$ on the five drawn \\", r"\midrule", *rows,
                                          r"\bottomrule", r"\end{tabular}"]))


def main():
    GEN.mkdir(exist_ok=True)
    facts_tex()
    raw_tex()
    size_tex()
    decompose_size_tex()
    siteday_tex()
    mechanisms_tex()
    robustness_tex()
    zones_tex()
    redundancy_tex()
    d = decompose_tex()
    if d is not None:
        short = {"raw": "Raw", "same site-day-hour": "Sdh", "selection": "Sel", "group, same flight": "Flight",
                 "group, same site-day-hour": "GroupSdh"}
        keys = {"H_loc(100 s)": ("HlocHundred", 3), "H_loc(1000 s)": ("HlocThousand", 3),
                "time in climb (%)": ("Climb", 1), "searches that end in a climb (%)": ("Found", 1),
                "climb rate (m/s)": ("Rate", 2), "glide speed (km/h)": ("Speed", 1), SINK: ("Sink", 0),
                "cross-country speed over 30 min (km/h)": ("Xc", 1)}
        for s, (ks, digits) in keys.items():  # composition by site-day and hour: pooled minus stratified G - S
            macro(f"DComp{ks}", signed(d["raw"][s]["value"] - d["same site-day-hour"][s]["value"],
                                       digits + (1 if digits < 3 else 0)))
        for c, cs in short.items():
            for s, (ks, digits) in keys.items():
                v = d[c][s]
                macro(f"D{ks}{cs}", signed(v["value"], digits + (1 if digits < 3 else 0)))
                macro(f"D{ks}{cs}Band", f"[{signed(v['band'][0], digits + (1 if digits < 3 else 0))}, "
                                         f"{signed(v['band'][1], digits + (1 if digits < 3 else 0))}]")
    (GEN / "macros.tex").write_text("\n".join(rf"\newcommand{{\{k}}}{{{v}}}" for k, v in sorted(macros.items())) + "\n")
    print(f"{len(macros)} macros")


if __name__ == "__main__":
    main()
