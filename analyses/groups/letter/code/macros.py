"""Macros of the letter that the reports of vs-solo and structure do not already write.

The letter reads those reports' macros (vs-solo/generated/comparisons/macros.tex,
vs-solo/generated/macros.tex, vs-solo/generated/structure_macros.tex) and this file's
letter/generated/macros.tex, every name prefixed Lt.

    uv run python letter/code/macros.py [--run NAME]    # from analyses/groups
"""

import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
GROUPS = HERE.parent
sys.path.insert(0, str(GROUPS))
from blocks import SITE_M  # noqa: E402
from longlags import FIXED_LAG  # noqa: E402
from openflights import MIN_OPENING  # noqa: E402
from performance import XC_S  # noqa: E402
from raw import DATASET_LAGS  # noqa: E402
from route import MATCH_S  # noqa: E402
from tail_designs import KS_GRID  # noqa: E402
from twoway import Z  # noqa: E402
from tamsd import H_MAX, HALF_DECADE, SUPPORT, TAU_MAX  # noqa: E402
RUNS = Path("/Volumes/SSD_DISANTE/xc-lab/group-identification/runs")
MIN_FLIGHTS_YEAR = 100  # the period starts at the first year with at least this many flights
LONG = "2000 s"  # the class of tail.py's descriptors that the letter calls the long transitions


def thousands(n):
    return f"{n:,}".replace(",", "\\,")


def macros(run):
    facts = json.loads((GROUPS / "out" / run / "facts.json").read_text())
    para = facts["flights"]["para"]
    summary = dict(facts["para"])
    years = pd.to_datetime(pd.read_parquet(RUNS / run / "flights.parquet", columns=["date"])["date"]).dt.year
    counts = years.value_counts()
    busy = counts[counts >= MIN_FLIGHTS_YEAR].index
    delta, eps_h, eps_z, theta, _, q = re.fullmatch(r"d(\d+)_eh(\d+)_ez(\d+)_th(\d+)_g(\d+)_q([\d.]+)", run).groups()
    params = json.loads((RUNS / run / "calibration.json").read_text())["params"]
    return {
        "LtGapMin": f"{params['gap'] / 60:g}",
        "LtNullRadiusKm": f"{params['r_s'] / 1000:g}",
        "LtSiteKm": f"{SITE_M / 1000:g}",
        "LtLoggerMax": f"{H_MAX:g}",
        "LtDeltaMin": f"{int(delta) / 60:g}",
        "LtEpsH": eps_h,
        "LtEpsZ": eps_z,
        "LtTheta": theta,
        "LtNullShare": f"{100 * float(q):g}",
        "LtFlightsAll": thousands(para),
        "LtFlights": thousands(summary["eligible flights"]),
        "LtGroupFlights": thousands(summary["flights ever in a group"]),
        "LtGroupShare": f"{100 * summary['share ever in a group']:.1f}",
        "LtGroupHours": thousands(round(summary["hours in a group"])),
        "LtEpisodes": thousands(facts["episodes"]),
        "LtPairShare": f"{100 * summary['group hours by size']['n2'] / summary['hours in a group']:.0f}",
        "LtFirstYear": str(busy.min()),
        "LtLastYear": str(busy.max()),
    }


def method_macros(folder):
    """The constants of the MSD method that the supplement states: window of the local exponent, support
    rule, lags and datasets, cross-country speed, matching grid, scan of the Hill threshold."""
    short_window = min(int(f.stem.split("_")[1]) for f in folder.glob("prepost_[0-9]*.json"))
    return {
        "LtDecade": f"{HALF_DECADE:g}",
        "LtBandLevel": f"{100 * math.erf(Z / math.sqrt(2)):.0f}",
        "LtSupport": f"{SUPPORT:g}",
        "LtTauOne": str(TAU_MAX),
        "LtTauTwo": str(FIXED_LAG),
        "LtLagOne": str(DATASET_LAGS[0]),
        "LtLagTwo": str(DATASET_LAGS[1]),
        "LtXcMin": f"{XC_S / 60:g}",
        "LtXcS": str(XC_S),
        "LtMatchStep": str(MATCH_S),
        "LtKsN": str(len(KS_GRID)),
        "LtKsLow": f"{KS_GRID[0]:.0f}",
        "LtKsHigh": f"{KS_GRID[-1]:.0f}",
        "LtWinShortShare": f"{TAU_MAX / short_window:.2f}",
    }


def band(b, digits):
    return f"[{b[0]:.{digits}f}, {b[1]:.{digits}f}]"


def tail_macros(folder):
    """The Hill index of transitions above the xmin of S, where G and S separate, the long transitions."""
    t = json.loads((folder / "tail.json").read_text())
    tr = t["transition"]
    xmin = tr["xmin of S"]
    out = {"LtTailXmin": f"{xmin:.0f}"}
    for pop in ("S", "A", "G"):
        at = tr["populations"][pop]["at xmin of S"]
        out[f"LtTailMu{pop}"], out[f"LtTailMu{pop}Band"] = f"{at['mu']:.2f}", band(at["band"], 2)
        out[f"LtTailRuns{pop}"] = thousands(at["runs above"])
    g, s = tr["populations"]["G"]["curve"], tr["populations"]["S"]["curve"]
    apart = [x for x, gh, sl in zip(g["xmin"], g["band"][1], s["band"][0]) if gh < sl]
    out["LtTailApartFrom"], out["LtTailApartTo"] = f"{min(apart):.0f}", f"{max(apart):.0f}"
    route = t["same route"]
    k = int(np.argmin(np.abs(np.asarray(route["xmin"]) - xmin)))
    out["LtTailRouteRunsG"] = thousands(route["sides"]["G"]["runs above"][k])
    names = {"path (km)": ("Path", 0), "height lost (m/km)": ("Sink", 0), "vertical speed (m/s)": ("Vz", 2),
             "straightness": ("Straight", 2), "time rising (share)": ("Rising", 0)}
    out["LtLongMin"] = LONG.split()[0]
    gaps = t["descriptor gaps"][LONG]
    for key, name, digits in (("height lost (m/km)", "Sink", 0), ("vertical speed (m/s)", "Vz", 2)):
        for gap, tag in (("G - S", "G"), ("A - S", "A")):
            v = gaps[key][gap]
            out[f"LtLong{name}Gap{tag}"] = signed(v["value"], digits)
            out[f"LtLong{name}Gap{tag}Band"] = signed_band(v["band"], digits)
    for pop in ("S", "G"):
        long = t["descriptors"][pop][LONG]
        out[f"LtLongRuns{pop}"] = thousands(long["runs"])
        for key, (name, digits) in names.items():
            value = long[key]["median"] * (100 if name == "Rising" else 1)
            out[f"LtLong{name}{pop}"] = f"{value:.{digits}f}".replace("-", "$-$")
    return out


def signed(x, digits=3):
    return f"{x:+.{digits}f}".replace("+", "$+$").replace("-", "$-$")


def signed_band(b, digits=3):
    return f"[{signed(b[0], digits)}, {signed(b[1], digits)}]"


def terrain_macros(folder):
    """Delta alpha(1000 s) by Hammond class (alpha = 2 H_loc): raw on three classes, same flight on two, and for the table of
    the supplement the raw, same-route and same-flight contrasts by class."""
    t = json.loads((folder / "terrain_compare.json").read_text())
    raw4, flight = t["raw"]["four classes"], t["flight"]
    out = {}
    for name, cls in (("Low", "hills"), ("LowMtn", "low mountains"), ("High", "high mountains")):
        put(out, f"LtTerRaw{name}", raw4[cls]["H_loc(1000 s) gaps"]["G - S"], 3, 2, True)
        out[f"LtTerAlphaS{name}"] = f"{2 * raw4[cls]['H_loc(1000 s)']['S']['value']:.2f}"
    for name, cls in (("Low", "lowland"), ("Mtn", "mountain")):
        put(out, f"LtTerFlight{name}", flight[cls]["effects"]["g"]["H_loc(1000 s)"], 3, 2, True)
        out[f"LtTerFlight{name}Events"] = thousands(flight[cls]["events"])
        out[f"LtTerFlight{name}Pilots"] = thousands(flight[cls]["pilots"])
    out["LtTerRouteLowPieces"] = thousands(t["route"]["lowland"]["counts"]["G"]["pieces"])
    raw = t["raw"]
    for name, d in (("Lowland", raw["binary"]["lowland"]), ("Low", raw4["low mountains"]), ("High", raw4["high mountains"]),
                    ("Mountain", raw["binary"]["mountain"])):
        put(out, f"LtTnRaw{name}", d["H_loc(1000 s) gaps"]["G - S"], 3, 2, True)
    for name, pop, d in (("Lowland", "S", raw["binary"]["lowland"]), ("High", "S", raw4["high mountains"])):
        out[f"LtTnAlpha{pop}{name}"] = f"{2 * d['H_loc(1000 s)'][pop]['value']:.2f}"
    for name, cls in (("Lowland", "lowland"), ("Mountain", "mountain")):
        put(out, f"LtTnRoute{name}", t["route"][cls]["deltas"]["(G - S|G) - (A - S|A)"]["H_loc(1000 s)"], 3, 2, True)
        put(out, f"LtTnFlight{name}", flight[cls]["effects"]["g"]["H_loc(1000 s)"], 3, 2, True)
    return out


DATASETS = {"1000": "One", "2760": "Two", "longest": "Long"}  # dataset of raw.json (pieces of at least 1250 s, 3450 s, T / 0.8): macro tag
FIT_END = {"1000": "1000", "2760": "2760", "longest": "11450"}  # the same datasets in fitted_h_by_end.json
LAGS = {1000: "K", 1840: "Q", 2760: "T"}  # s: the lags quoted, tag of the macros
WINDOWS = {1260: "Short", 1800: "Low", 2400: "Mid", 3000: "Long", 3450: "Max"}  # s: windows of prepost.py, tag of the macros
XC = "cross-country speed over 30 min (km/h)"


def put(out, name, q, digits, scale=1, sign=False):
    """Macros `name` and `nameBand`: the value of the quote q = {value, band} and its band, times `scale`
    (alpha = 2 H_loc), signed for a difference."""
    f = signed if sign else (lambda x, d: f"{x:.{d}f}")
    out[name] = f(scale * q["value"], digits)
    out[name + "Band"] = "[" + ", ".join(f(scale * x, digits) for x in q["band"]) + "]"


def at_lag(c, lag):
    """The quote at `lag` of a curve {lags, value, band: [lows, highs]}, None when the lag is not on it."""
    if lag not in c["lags"]:
        return None
    i = c["lags"].index(lag)
    return {"value": c["value"][i], "band": [c["band"][0][i], c["band"][1][i]]}


def stays_above(lags, values):
    """The first lag from which `values` stay above 0 to the end of `lags`, None when the last one is not."""
    below = [i for i, v in enumerate(values) if v <= 0]
    i = below[-1] + 1 if below else 0
    return lags[i] if i < len(lags) else None


def raw_macros(folder):
    """Raw comparison: alpha of S, A, G and G - S, A - S at the lags the dataset has, the fitted H (10 s to the end lag)
    of each and its differences, the lags where G - S turns positive and where its band clears 0, the samples, and the
    cross-country speed (pieces of at least performance.XC_S / SUPPORT s, no dataset)."""
    raw = json.loads((folder / "raw.json").read_text())
    fits = json.loads((folder / "fitted_h_by_end.json").read_text())
    out = {"LtFitFrom": str(raw["datasets"]["1000"]["populations"]["S"]["curve"]["lags"][0]), "LtTauLong": str(fits["ends"][-1])}
    for name, ds in DATASETS.items():
        for pop in "SAG":
            c = raw["datasets"][name]["populations"][pop]
            put(out, f"LtRawFit{pop}{ds}", {"value": c["H"], "band": c["H_band"]}, 3)
            out[f"LtRawFlights{pop}{ds}"], out[f"LtRawPieces{pop}{ds}"] = thousands(c["flights"]), thousands(c["pieces"])
            for lag, tag in LAGS.items():
                if c.get(f"H_loc({lag} s)"):
                    put(out, f"LtRawAlpha{tag}{pop}{ds}", c[f"H_loc({lag} s)"], 3, 2)
        for gap in ("G - S", "A - S"):
            put(out, f"LtRawFitGap{gap[0]}S{ds}", fits["gaps"][gap][FIT_END[name]], 3, sign=True)
            for lag, tag in LAGS.items():
                if raw["gaps"][name].get(f"H_loc({lag} s)"):
                    put(out, f"LtRawAlphaGap{tag}{gap[0]}S{ds}", raw["gaps"][name][f"H_loc({lag} s)"][gap], 3, 2, True)
        g = raw["gaps"][name]["H_loc"]["G - S"]
        out[f"LtRawTurn{ds}"] = str(stays_above(g["lags"], g["value"]))
        out[f"LtRawClear{ds}"] = str(stays_above(g["lags"], g["band"][0]))
    for pop in "SAG":
        put(out, f"LtRawXc{pop}", raw["xc"][pop]["average first"][XC], 1)
        out[f"LtRawXcFlights{pop}"] = thousands(raw["xc"][pop]["flights"])
    for gap in ("G - S", "A - S"):
        put(out, f"LtRawXcGap{gap[0]}S", raw["xc gaps"][gap], 1, sign=True)
    return out


def route_macros(folder):
    """Same route: Delta alpha of G - S|G, A - S|A and their difference at the lags of each dataset (dataset 1000: the
    main series; 2760: the long series) and the fitted H, the cross-country speed, and the samples."""
    m = json.loads((folder / "matched.json").read_text())["levels"]["all"]
    contrasts = {"GS": "G - S|G", "AS": "A - S|A", "Diff": "(G - S|G) - (A - S|A)"}
    out = {}
    for tag, c in contrasts.items():
        d = m["deltas"][c]
        put(out, f"LtRouteAlphaK{tag}One", d["H_loc(1000 s)"], 3, 2, True)
        put(out, f"LtRouteFit{tag}One", d["H (10 s to 1000 s)"], 3, sign=True)
        put(out, f"LtRouteFit{tag}Two", d["H (10 s to T)"], 3, sign=True)
        long = m["delta curves"][c]["long"]
        for lag, lt in LAGS.items():
            if at_lag(long, lag):
                put(out, f"LtRouteAlpha{lt}{tag}Two", at_lag(long, lag), 3, 2, True)
        put(out, f"LtRouteXc{tag}", d[XC], 1, sign=True)
    gain = m["deltas"][contrasts["Diff"]][XC]["value"] / m["sides"]["S|G"][XC]["value"]
    out["LtRouteXcDiffPct"] = f"{100 * gain:.0f}"
    for side in "GA":
        n = m["counts"][side]
        for key, name in (("pieces", "Pieces"), ("controls", "Controls"), ("flights, %s side" % side, "Flights"),
                          ("pilots, %s side" % side, "Pilots"), ("pilots, control side", "ControlPilots"),
                          ("pool flights", "Pool")):
            out[f"LtRoute{name}{side}One"] = thousands(n[key])
        long = m["sides"][side]["curves"]["long"]
        out[f"LtRoutePieces{side}Two"], out[f"LtRouteControls{side}Two"] = thousands(long["pieces"]), thousands(long["controls"])
        out[f"LtRouteXcPieces{side}"], out[f"LtRouteXcControls{side}"] = (thousands(m["xc counts"][side][k]) for k in ("pieces", "controls"))
    return out


LEVELS = {"beginners": "Beg", "experienced": "Exp", "all": "All"}
CALIPERS = {"month1_hour1_class": "NarrowClass", "month1_hour1_level": "NarrowLevel",
            "month2_hour2_class": "WideClass", "month2_hour2_level": "WideLevel"}
LONGER_TAU, LONGER_BEGINNERS = 3380, 139  # s, pieces: counts of the longer dataset tried once (slide "Why the long dataset stops
# at 2760 s", meetings/2026-11/group_vs_solo); matched.py no longer makes that dataset, so they are not in a JSON


def route_extra_macros(folder):
    """Same route: the point values by wing level, the range of the lines of the match in alpha, the pieces of G and A that
    find a control for each caliper, dataset and level, the route alone match on the open flights, the datasets tried."""
    m = json.loads((folder / "matched.json").read_text())
    c = json.loads((folder / "matched_counts.json").read_text())
    out = {"LtLongerTau": str(LONGER_TAU), "LtLongerBeginners": str(LONGER_BEGINNERS), "LtMinPieces": str(c["min pieces"])}
    for lv, lt in LEVELS.items():
        for tag, k in (("GS", "G - S|G"), ("AS", "A - S|A"), ("Diff", "(G - S|G) - (A - S|A)")):
            out[f"LtRouteLevel{lt}{tag}"] = signed(2 * m["levels"][lv]["deltas"][k]["H_loc(1000 s)"]["value"], 3)
    rob = m["robustness"]
    same = rob["main"]["long T"]
    for lag, tag in ((1000, "K"), (1840, "Q")):
        q = f"H_loc({lag} s)"
        v = [x["quotes"][q] for x in rob.values() if q in x["quotes"] and (lag == 1000 or x["long T"] == same)]
        out[f"LtRobLines{tag}"] = str(len(v))
        out[f"LtRobLow{tag}"], out[f"LtRobHigh{tag}"] = (signed(2 * f([x["value"] for x in v]), 3) for f in (min, max))
        out[f"LtRobZero{tag}"] = str(sum(x["band"][0] <= 0 <= x["band"][1] for x in v))
    out["LtRobLines"] = str(len(rob))
    out["LtRobPiecesLow"], out["LtRobPiecesHigh"] = (thousands(f(x["pieces"] for x in rob.values())) for f in (min, max))
    for ds, dt in (("1000", "One"), ("2760", "Two")):
        for side in "GA":
            for cal, ct in CALIPERS.items():
                for lv, lt in LEVELS.items():
                    out[f"LtCnt{ct}{side}{dt}{lt}"] = thousands(c["pieces with a control"][side][ds][cal][lv]["pieces"])
            for lv, lt in LEVELS.items():
                out[f"LtCntQuery{side}{dt}{lt}"] = thousands(c["query pieces"][side][ds][lv])
    r = json.loads((folder / "route.json").read_text())["any date"]
    put(out, "LtRouteOpenAlpha", r["H_loc(1000 s)"], 3, 2, True)
    put(out, "LtRouteOpenBaseAlpha", r["baseline"]["H_loc(1000 s)"], 3, 2, True)
    return out


def flight_macros(folder):
    """Same flight, for each window of prepost.py that has its JSON: g, d_w, after - before and, with the mock events, the
    mock g and g minus mock: Delta alpha at the lags the window measures, the fitted H, the cross-country speed (windows
    of at least XC_S / SUPPORT s), and the numbers of events and flights."""
    out = {}
    for w, tag in WINDOWS.items():
        path = folder / f"prepost_{w}.json"
        if not path.exists():
            continue
        p = json.loads(path.read_text())
        d = p["differences"]
        out[f"LtWin{tag}"] = thousands(w)
        variants = p.get("mock", {}).get("variants", {})
        mock = variants.get("any site")
        effects = {"G": d["g"], "Dw": d["d w"], "PostPre": d["post - pre"]}
        for name, key in (("In", "D_in"), ("Out", "D_out"), ("CtrlA", "ctrlA2 - ctrlA1"), ("CtrlS", "ctrlS2 - ctrlS1")):
            if key in d:
                effects[name] = d[key]
        for site, v in (("", mock), ("Site", variants.get("same site"))):
            if v:
                effects |= {f"Mock{site}G": v["mock"]["g"], f"Net{site}G": v["real minus mock"]["g"]}
                if "real events with a mock" in v:
                    effects[f"MockSubset{site}G"] = v["real events with a mock"]["g"]
        for name, e in effects.items():
            put(out, f"LtFlt{tag}{name}Fit", e["Delta H"], 3, sign=True)
            for lag, lt in LAGS.items():
                q = at_lag({"lags": p["lags"], **e["dH_loc"]}, lag)
                if q:
                    put(out, f"LtFlt{tag}{name}Alpha{lt}", q, 3, 2, True)
            if XC in e:
                put(out, f"LtFlt{tag}{name}Xc", e[XC], 1, sign=True)
        for name, e in (("", d["g"]), ("PostPre", d["post - pre"]), *((n, effects[n]) for n in ("CtrlA", "CtrlS") if n in effects)):
            out[f"LtFlt{tag}Events{name}"], out[f"LtFlt{tag}Flights{name}"] = thousands(e["events (MSD)"]), thousands(e["flights (MSD)"])
        for site, v in (("", mock), ("Site", variants.get("same site"))):
            if v:
                c = v["counts"]
                out[f"LtFlt{tag}Mock{site}Events"], out[f"LtFlt{tag}Mock{site}Flights"] = thousands(c["mock events (MSD)"]), thousands(c["solo flights (MSD)"])
                out[f"LtFlt{tag}Mock{site}Real"] = thousands(c["real events with a mock event (MSD)"])
    return out


def declared_macros(run):
    """alpha at the longest lag of the dataset to 11450 s on the declared open flights, before the opening threshold."""
    panel = json.loads((GROUPS / "out" / run / "open_declared" / "longlags.json").read_text())["panels"]["longest"]
    out = {"LtDeclTau": str(panel["tau_max"])}
    for pop, key in (("S", "solo"), ("A", "group_solo"), ("G", "group")):
        c = panel["curves"][key]
        put(out, f"LtDeclLongestAlpha{pop}", {"value": c["H_loc"][-1], "band": [c["H_loc_band"][0][-1], c["H_loc_band"][1][-1]]}, 3, 2)
        out[f"LtDeclFlights{pop}"] = thousands(c["flights"])
    return out


def open_macros(run):
    """The selection of the open flights: eligible flights, declared open, and those whose opening is above the threshold."""
    o = json.loads((GROUPS / "out" / run / "open_declared" / "opening.json").read_text())
    every = o["datasets"]["every declared open flight"]["all"]
    return {"LtOpenTotal": thousands(sum(r["flights"] for r in o["labels"])), "LtOpenDeclared": thousands(every["flights"]),
            "LtOpenKept": thousands(every["above the threshold"]), "LtOpenKeptShare": f"{every['above the threshold (%)']:.1f}",
            "LtOpenThreshold": f"{o['min_opening']:g}"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="d120_eh500_ez300_th45_g120_q0.1")
    run = parser.parse_args().run
    folder = GROUPS / "out" / run / f"open_min{MIN_OPENING:g}"
    values = (macros(run) | method_macros(folder) | tail_macros(folder) | terrain_macros(folder) | raw_macros(folder)
              | route_macros(folder) | route_extra_macros(folder) | flight_macros(folder) | open_macros(run) | declared_macros(run))
    out = HERE / "generated" / "macros.tex"
    out.parent.mkdir(exist_ok=True)
    out.write_text("".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in values.items()))
    print(out.read_text())


if __name__ == "__main__":
    main()
