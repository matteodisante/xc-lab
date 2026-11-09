"""Macros of the letter that the reports of vs-solo and structure do not already write.

The letter reads those reports' macros (vs-solo/generated/comparisons/macros.tex,
vs-solo/generated/macros.tex, vs-solo/generated/structure_macros.tex) and this file's
letter/generated/macros.tex, every name prefixed Lt.

    uv run python letter/code/macros.py [--run NAME]    # from analyses/groups
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
GROUPS = HERE.parent
sys.path.insert(0, str(GROUPS))
from openflights import MIN_OPENING  # noqa: E402
from tamsd import H_MAX  # noqa: E402
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
    return {
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
    """Delta H_loc(1000 s) by Hammond class: raw on three classes, same flight on two."""
    t = json.loads((folder / "terrain_compare.json").read_text())
    raw4, flight = t["raw"]["four classes"], t["flight"]
    out = {}
    for name, cls in (("Low", "hills"), ("LowMtn", "low mountains"), ("High", "high mountains")):
        gap = raw4[cls]["H_loc(1000 s) gaps"]["G - S"]
        out[f"LtTerRaw{name}"], out[f"LtTerRaw{name}Band"] = signed(gap["value"]), signed_band(gap["band"])
        out[f"LtTerHlocS{name}"] = f"{raw4[cls]['H_loc(1000 s)']['S']['value']:.2f}"
    for name, cls in (("Low", "lowland"), ("Mtn", "mountain")):
        g = flight[cls]["effects"]["g"]["H_loc(1000 s)"]
        out[f"LtTerFlight{name}"], out[f"LtTerFlight{name}Band"] = signed(g["value"]), signed_band(g["band"])
        out[f"LtTerFlight{name}Events"] = thousands(flight[cls]["events"])
        out[f"LtTerFlight{name}Pilots"] = thousands(flight[cls]["pilots"])
    out["LtTerRouteLowPieces"] = thousands(t["route"]["lowland"]["counts"]["G"]["pieces"])
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="d120_eh500_ez300_th45_g120_q0.1")
    run = parser.parse_args().run
    folder = GROUPS / "out" / run / f"open_min{MIN_OPENING:g}"
    values = macros(run) | tail_macros(folder) | terrain_macros(folder)
    out = HERE / "generated" / "macros.tex"
    out.parent.mkdir(exist_ok=True)
    out.write_text("".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in values.items()))
    print(out.read_text())


if __name__ == "__main__":
    main()
