"""Climbs, glides and cross-country speed of paragliders, solo and in a group.

One pass over the cleaned fixes. Vilpellet's climbs and transitions, each whole in the
population of the stretch that holds its middle (survival.assign), read at their first and
last fix: a climb gains z(end) - z(start) in t(end) - t(start); a transition covers the
straight line between its ends and loses z(start) - z(end). Cross-country speed: on the 10-s
pieces of tamsd.grids that last at least XC_S / SUPPORT, the straight-line distance covered in
XC_S from every origin, averaged and divided by XC_S. Ratios summed over each population, with
the bands of budget.py: climb rate = total gain / total climb time, glide ratio = total
distance / total height lost, so long phases weigh more than short ones.

    uv run python performance.py [--run NAME] [--recompute]    # from this folder; writes out/<run>/performance.*
"""

import json
import os
import sys
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

import numpy as np
import pandas as pd

from budget import budget, plot
from lab.legacy import stream_flights
from phase_msd import RUNS
from survival import assign
from tamsd import CACHE, FIXES, GRID_S, LABELS, OUT, SEED, SUPPORT, folders, grids, stretches

XC_S = 1800  # s
# name: numerator, denominator, scale, decimals shown
STATS = {
    "climb rate (m/s)": ("gain_m", "climb_s", 1, 2),
    "height gained per climb (m)": ("gain_m", "climbs", 1, 0),
    "glide speed (km/h)": ("chord_m", "transition_s", 3.6, 1),
    "glide ratio": ("chord_m", "loss_m", 1, 1),
    "cross-country speed over 30 min (km/h)": ("xc_m", "xc_origins", 3.6 / XC_S, 1),
}
# The statistics drawn: the height gained per climb and the cross-country speed are explained by
# the others and by those of budget.py, and stay in the JSON and the tables (redundancy.py).
PLOTTED = ("climb rate (m/s)", "glide speed (km/h)", "glide ratio")


def flight_terms(fixes, runs, spans):
    """For one flight: z at both ends and the straight line of each run (rows of `runs`, with
    segment_id, t_start, t_end at fixes), and the distance sums of its cross-country pieces with
    their middle instant on the flight clock."""
    fixes = fixes.sort_values(["segment_id", "t"], kind="stable")
    key = fixes["segment_id"].to_numpy() * 1e6 + fixes["t"].to_numpy(float)
    ends = [np.searchsorted(key, runs["segment_id"].to_numpy() * 1e6 + runs[c].to_numpy(float)) for c in ("t_start", "t_end")]
    a, b = (np.clip(e, 0, len(key) - 1) for e in ends)
    z, e, n = (fixes[c].to_numpy(float) for c in ("z", "E", "N"))
    dz, chord = z[b] - z[a], np.hypot(e[b] - e[a], n[b] - n[a])
    k = XC_S // GRID_S
    xc = []
    for span, start, pos in grids(fixes, spans, XC_S / SUPPORT):
        d = np.hypot(*(pos[k:] - pos[:-k]).T)
        xc.append((span.population, d.sum(), len(d), start + (len(pos) - 1) * GRID_S / 2))
    return dz, chord, xc


def _job(flight, fixes, runs, spans):
    return (flight, runs.index.to_numpy(), *flight_terms(fixes, runs, spans))


def measure(flights, runs, spans):
    """Run terms (dz, chord, by run index) and cross-country rows, from one pass over the fixes."""
    ids = flights.loc[flights["discipline"] == "para"].set_index("flight_id")["flight"]
    by_flight, runs_of = dict(tuple(spans.groupby("flight"))), dict(tuple(runs.groupby("flight")))
    dz, chord = pd.Series(np.nan, index=runs.index), pd.Series(np.nan, index=runs.index)
    xc = []

    def store(future):
        flight, rows, d, c, found = future.result()
        dz.loc[rows], chord.loc[rows] = d, c
        xc.extend((flight, *f) for f in found)

    with ProcessPoolExecutor(os.cpu_count()) as pool:
        pending = set()
        for fixes in stream_flights(FIXES["para"], ["segment_id", "t", "E", "N", "z"]):
            flight = ids.get(str(fixes["flight_id"].iat[0]))
            if flight is None or flight not in by_flight:
                continue
            mine = runs_of.get(flight, runs.iloc[:0])
            pending.add(pool.submit(_job, flight, fixes, mine, by_flight[flight]))
            if len(pending) >= 4 * os.cpu_count():
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    store(future)
        for future in pending:
            store(future)
    return dz, chord, pd.DataFrame(xc, columns=["flight", "population", "xc_m", "xc_origins", "t_mid"])


def table(runs, dz, chord, xc):
    """One row per run and per cross-country piece, with the terms of STATS and the middle
    instant t_mid on the flight clock; runs the pass did not reach are left out."""
    reached = ~np.isnan(dz)
    runs, dz, chord = runs[reached], dz[reached], chord[reached]
    climb, glide = (runs["phase"] == "climb").to_numpy(), (runs["phase"] == "transition").to_numpy()
    s = (runs["t_end"] - runs["t_start"]).to_numpy(float)
    rows = pd.DataFrame({"flight": runs["flight"].to_numpy(), "population": runs["population"].to_numpy(),
                         "gain_m": np.where(climb, dz, 0), "climb_s": np.where(climb, s, 0), "climbs": climb * 1.0,
                         "chord_m": np.where(glide, chord, 0), "transition_s": np.where(glide, s, 0),
                         "loss_m": np.where(glide, -dz, 0),
                         "t_mid": ((runs["t_start"] + runs["t_end"]) / 2).to_numpy(float)})
    return pd.concat([rows, xc], ignore_index=True).fillna(0)


def save_terms(path, flights, spans):
    """Measure the terms of STATS for `spans` and save them in `path`."""
    numbers = flights.loc[flights["discipline"] == "para", ["flight_id", "flight"]]
    runs = pd.read_parquet(RUNS["para"], columns=["flight_id", "segment_id", "phase", "t_start", "t_end"])
    runs = assign(runs[runs["phase"].isin(["climb", "transition"])].merge(numbers, on="flight_id"), spans)
    dz, chord, xc = measure(flights, runs, spans)
    table(runs, dz.to_numpy(), chord.to_numpy(), xc).to_parquet(path)


if __name__ == "__main__":
    folders()
    flights = pd.read_parquet(LABELS / "flights.parquet")
    summary = pd.read_parquet(LABELS / "flight_summary.parquet")
    saved = CACHE / "performance_terms.parquet"
    # One pass over 40 GB of fixes: reuse its terms unless asked.
    if "--recompute" in sys.argv or not saved.exists():
        save_terms(saved, flights, stretches(flights, pd.read_parquet(LABELS / "membership.parquet"), summary))
    terms = pd.read_parquet(saved)
    result = budget(terms, summary.set_index("flight")["block"], np.random.default_rng(SEED), STATS)
    (OUT / "performance.json").write_text(json.dumps(result, indent=1))
    plot(result, OUT / "performance.pdf", {name: STATS[name] for name in PLOTTED},
         "Paragliders: climb rate, glide speed and glide ratio")
    for pop, r in result.items():
        print(pop, r["flights"], " | ".join(f"{n}: {r[n]['value']:.2f} {np.round(r[n]['band'], 2).tolist()}" for n in STATS))
