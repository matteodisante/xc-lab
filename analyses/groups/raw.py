"""Point 1 of comparisons_plan.md: S, A and G as they are, on the open flights.

Three datasets of MSD pieces (longlags.py): `1000` (pieces of at least 1250 s, lags to 1000 s), `2760`
(longlags.FIXED_LAG) and `longest` (the longest lag at which each population keeps longlags.MIN_FLIGHTS
flights). For each, the MSD and the H_loc figures with the fitted H (10 s to the end of the lags) in the
legend. On the pieces of dataset `1000`, unit = the flight: the seven phase statistics with the estimator
"average first, ratio after" (budget.unit_rates, budget.mean_ratios) and, as a check, the ratio of sums;
the survival of the durations of climbs, searches that end in a climb and glides. A run of Vilpellet
belongs to the piece that holds its middle (isolated.window_sums). All bands: twoway.py, the same draws of
pilots, blocks and cells for every number.

    uv run python raw.py --open [--run NAME]    # after tamsd.py, longlags.py and performance.py with --open
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import budget
import curves
import decompose
import isolated
import longlags
import survival
import tamsd
import twoway
from blocks import with_takeoffs
from phase_msd import PHASES
from tamsd import COLORS, LABELS, OUT, SEED, read_flights, save

sys.path.insert(0, str(Path(__file__).parents[1] / "phases"))
from durations import trim  # noqa: E402

POPS = {"solo": "S", "group_solo": "A", "group": "G"}  # drawn in this order
NAMES = {"solo": "S, solo flights", "group_solo": "A, group flights alone", "group": "G, group time"}
DATASET_LAGS = (1000, 1840)  # s: the quoted H_loc, on every dataset that has the lag
STATISTICS = ("time in climb (%)", "time in search (%)", "searches that end in a climb (%)", "climbs per hour",
              "climb rate (m/s)", "glide speed (km/h)", decompose.SINK)
DIGITS = {"time in climb (%)": 1, "time in search (%)": 1, "searches that end in a climb (%)": 1,
          "climbs per hour": 2, "climb rate (m/s)": 2, "glide speed (km/h)": 1, decompose.SINK: 0}
SOURCES = ("b", "p")  # the terms of budget.py and of performance.py (isolated.phase_stats)
SURVIVAL_PHASES = {"climb": "climb", "search": "search that ends in a climb", "transition": "glide"}
GAPS = {"G - S": ("group", "solo"), "A - S": ("group_solo", "solo"), "G - A": ("group", "group_solo")}
GAP_COLORS = {"G - S": COLORS["group"], "A - S": COLORS["group_solo"], "G - A": "#8c2f0f"}


def stat_defs():
    """The STATISTICS as {source: {name: (numerator, denominator, scale, digits)}}."""
    every = isolated.phase_stats()
    return {src: {n: (*every[n][1:], DIGITS[n]) for n in STATISTICS if every[n][0] == src} for src in SOURCES}


def pieces_of(p, pop, tau=tamsd.TAU_MAX):
    """Flight, start and end (flight clock, s) of the pieces of population `pop` that last long enough for
    the lag `tau`."""
    keep = longlags.long_enough(p, p["population"] == pop, tau)
    return p["flight"][keep], p["start"][keep], p["start"][keep] + p["duration"][keep]


def in_pieces(table, flight, a, b):
    """The rows of `table` (flight, t_mid) whose middle lies in a piece (flight, a, b), as a mask, and the
    flight of the piece for each (pieces of one flight do not overlap)."""
    rank = isolated.RANK
    order = np.argsort(flight * rank + a, kind="stable")
    start, end = (flight * rank + a)[order], (flight * rank + b)[order]
    key = table["flight"].to_numpy(float) * rank + table["t_mid"].to_numpy(float)
    i = np.searchsorted(start, key, "right") - 1
    return (i >= 0) & (key < end[np.maximum(i, 0)])


def unit_terms(tables, defs, pieces):
    """Per flight with pieces: the phase time T_u (runs in its pieces) and, per source, the sums of the terms
    of `defs` over the runs in its pieces; as frames indexed by flight."""
    flight, a, b = pieces
    flights, k = np.unique(flight, return_inverse=True)
    out = {}
    for src, stats in defs.items():
        cols = sorted({c for num, den, *_ in stats.values() for c in (num, den)})
        per_piece = isolated.window_sums(tables[src], cols, flight, a, b)
        sums = np.zeros((len(flights), len(cols)))
        np.add.at(sums, k, per_piece)
        out[src] = pd.DataFrame(sums, index=flights, columns=cols)
    per_piece = isolated.window_sums(tables["b"], ["s"], flight, a, b)
    time = np.zeros(len(flights))
    np.add.at(time, k, per_piece[:, 0])
    return pd.Series(time, index=flights), out


def phase_statistics(tables, defs, pieces, block, counts):
    """The seven statistics of one population on its pieces: the new estimator and the ratio of sums, each
    with the value and its band over the draws `counts`; and, as "draws", the value and the draws of the new
    estimator (rows), for the gaps."""
    time, sums = unit_terms(tables, defs, pieces)
    time = time[time > 0]
    result = {"flights": int(len(time)), "phase time (h)": float(time.sum() / 3600)}
    for key, scale in (("average first", time), ("ratio of sums", pd.Series(1.0, index=time.index))):
        value, draws = [], []
        for src, stats in defs.items():
            r, q = budget.unit_rates(sums[src].reindex(time.index), time.index, scale, stats)
            v, d = budget.mean_ratios(r, q, block.reindex(time.index).to_numpy(), counts, stats)
            value.append(v)
            draws.append(pd.DataFrame(d, columns=list(stats)))
        value, draws = pd.concat(value), pd.concat(draws, axis=1)
        result[key] = {name: {"value": float(value[name]), "band": twoway.band(float(value[name]), draws[name].to_numpy())}
                       for name in STATISTICS}
        if key == "average first":
            result["draws"] = np.vstack([value[list(STATISTICS)].to_numpy(), draws[list(STATISTICS)].to_numpy()])
    return result


def survival_curves(runs, pieces, block, counts):
    """Per phase: the survival of the counted runs whose middle lies in the pieces, trimmed as
    durations.trim, with its bands; and the number of runs."""
    mine = runs[in_pieces(runs, *pieces)]
    out = {}
    for phase in PHASES:
        r = mine[mine["phase"] == phase]
        x = r["n_fixes"].to_numpy(float)
        keep = np.isin(x, trim(x))
        out[phase] = {"runs": int(keep.sum()), "flights": int(r.loc[keep, "flight"].nunique()),
                      **survival.survival_curve(x[keep], block.reindex(r["flight"]).to_numpy()[keep], counts, band=twoway.share_band)}
    return out


def plot_survival(result, path):
    """P(tau_phase > tau), log-log, one curve per population with its band; one panel per phase."""
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5), constrained_layout=True)
    for ax, phase in zip(axes, PHASES):
        for pop, name in POPS.items():
            s = result[pop][phase]
            t, v, band = np.array(s["grid"]), np.array(s["value"]), np.array(s["band"])
            ok = v > 0
            ax.fill_between(t[ok], np.maximum(band[0][ok], 1e-9), band[1][ok], color=COLORS[pop], alpha=0.2, lw=0)
            ax.plot(t[ok], v[ok], color=COLORS[pop], lw=1.3, label=f"{name} ({s['runs']:,} runs)")
        ax.set(xscale="log", yscale="log", title=SURVIVAL_PHASES[phase], xlabel=r"$\tau$ (s)",
               ylabel=rf"$P(\tau_{{\mathrm{{{phase}}}}} > \tau)$", ylim=(1e-5, 1.2))
        ax.legend(frameon=False, fontsize=7, loc="lower left")
        ax.grid(color="#eeeeee", lw=0.5)
    save(fig, path)


def duration_shares(p, pop, edges):
    """Share of the pieces of population `pop` (all of those in `p`) that fall in each bin of `edges` (s)."""
    count, _ = np.histogram(p["duration"][p["population"] == pop], edges)
    return count / count.sum()


def plot_durations(p, thresholds, path):
    """Histogram of the duration of the pieces of S, A, G, as shares of the pieces of each, on log bins; a
    dashed line at the shortest piece of each dataset (`thresholds`, s)."""
    top = max(p["duration"][p["population"] == pop].max() for pop in POPS)
    edges = np.geomspace(p["duration"].min(), top, 31)
    fig, ax = plt.subplots(figsize=(6, 3.5), constrained_layout=True)
    for pop in POPS:
        mine = p["duration"][p["population"] == pop]
        ax.stairs(duration_shares(p, pop, edges), edges, color=COLORS[pop], lw=1.4,
                  label=f"{NAMES[pop]} ({len(mine):,} pieces, median {np.median(mine):,.0f} s)")
    for t in thresholds:
        ax.axvline(t, color="#888888", lw=0.8, ls="--")
    ax.set(xscale="log", xlabel="duration of the piece (s)", ylabel="share of the pieces")
    ax.set_xticks(ticks := [1250, 2000, 3000, 5000, 10000, 20000, 40000], [f"{t:,}" for t in ticks])
    ax.minorticks_off()
    ax.legend(frameon=False, fontsize=7)
    ax.grid(color="#eeeeee", lw=0.5)
    save(fig, path)


def gap_at(gaps, lag):
    """Each gap of `gaps` (dataset_curves) at `lag`, value and band, or None when the curves stop before it."""
    lags = next(iter(gaps.values()))["lags"]
    if lag not in lags:
        return None
    i = lags.index(lag)
    return {gap: {"value": g["value"][i], "band": [g["band"][0][i], g["band"][1][i]]} for gap, g in gaps.items()}


def hloc_at(c, lag):
    """H_loc at `lag` of a curve of longlags.curve, with its band, or None when the curve stops before it."""
    if lag not in c["lags"]:
        return None
    i = c["lags"].index(lag)
    return {"value": c["H_loc"][i], "band": [c["H_loc_band"][0][i], c["H_loc_band"][1][i]]}


def dataset_curves(p, block, counts, rng):
    """The curves of S, A, G of the three datasets and the differences GAPS of H_loc at every lag, with their
    bands (the populations share the draws): {name: {tau_max, curves: {pop: curve}, gaps: {name: quote}}}."""
    avail = {k: longlags.available(p, m) for k, m in longlags.populations(p).items() if k in longlags.POOLED}
    taus = {"1000": tamsd.TAU_MAX, "2760": longlags.panels(avail)["fixed"], "longest": longlags.panels(avail)["longest"]}
    masks = longlags.populations(p)
    out = {}
    for name, tau in taus.items():
        got = {pop: longlags.curve(p, masks[pop], block, rng, tau, counts, twoway.band, with_draws=True) for pop in POPS}
        rows = {pop: np.vstack([c["H_loc"], c.pop("H_loc_draws")]) for pop, c in got.items()}
        out[name] = {"tau_max": tau, "curves": got,
                     "gaps": {gap: {"lags": got[a]["lags"], **twoway.quote(rows[a] - rows[b])} for gap, (a, b) in GAPS.items()}}
    return out


def draw_datasets(datasets, folder):
    """The MSD and the H_loc figure of each dataset, `raw_<name>_msd` and `raw_<name>_hloc`, and the H_loc
    of S, A, G beside their differences, `raw_<name>_delta`."""
    for name, d in datasets.items():
        drawn = []
        for pop, short in POPS.items():
            c = d["curves"][pop]
            lo, hi = c["H_band"]
            drawn.append(curves.series(c, f"{short} ({c['flights']:,} flights): H = {c['H']:.3f} [{lo:.3f}, {hi:.3f}]",
                                       COLORS[pop]))
        curves.figures(drawn, folder / f"raw_{name}", f"dataset {name}: pieces of at least {d['tau_max'] / tamsd.SUPPORT:g} s")
        left = [{**d["curves"][pop], "label": short, "color": COLORS[pop]} for pop, short in POPS.items()]
        right = [{**d["gaps"][gap], "label": gap, "color": GAP_COLORS[gap]} for gap in GAPS]
        g = d["gaps"]["G - S"]
        below = np.flatnonzero(np.asarray(g["value"]) <= 0)
        turn = below[-1] + 1 if len(below) and below[-1] + 1 < len(g["lags"]) else None  # the lag from which G - S stays above 0
        curves.delta_figure(left, right, folder / f"raw_{name}_delta",
                            f"dataset {name}: pieces of at least {d['tau_max'] / tamsd.SUPPORT:g} s",
                            vline=None if turn is None else (g["lags"][turn], f"G - S > 0 from {g['lags'][turn]:g} s"))


if __name__ == "__main__":
    tamsd.folders()
    flights = with_takeoffs(read_flights())
    code, pilot, block = twoway.cells(flights, pd.read_parquet(LABELS / "flight_summary.parquet"))
    rng = np.random.default_rng(SEED)
    counts = twoway.counts(pilot, block, rng)
    p = dict(np.load(longlags.PIECES))
    datasets = dataset_curves(p, code, counts, rng)
    draw_datasets(datasets, OUT)
    plot_durations(p, [d["tau_max"] / tamsd.SUPPORT for d in datasets.values()], OUT / "raw_durations.pdf")
    tables = isolated.run_terms(flights)
    defs = stat_defs()
    stats, surv = {}, {}
    for pop in POPS:
        pieces = pieces_of(p, pop)
        stats[pop] = phase_statistics(tables, defs, pieces, code, counts)
        surv[pop] = survival_curves(tables["b"][tables["b"]["phase"].ne("search") | tables["b"]["found"].eq(1)],
                                    pieces, code, counts)
    budget.plot({pop: {n: stats[pop]["average first"][n] for n in STATISTICS} for pop in POPS}, OUT / "raw_statistics.pdf",
                {n: (None, None, None, DIGITS[n]) for n in STATISTICS}, "Open paragliders: the seven phase statistics",
                {pop: NAMES[pop] for pop in POPS}, COLORS)
    plot_survival(surv, OUT / "raw_survival.pdf")
    draws = {pop: stats[pop].pop("draws") for pop in POPS}
    result = {"datasets": {name: {"tau_max": d["tau_max"], "populations": {
        POPS[pop]: {"flights": c["flights"], "pieces": c["pieces"], "H": c["H"], "H_band": c["H_band"],
                    **{f"H_loc({lag} s)": hloc_at(c, lag) for lag in DATASET_LAGS}, "curve": c}
        for pop, c in d["curves"].items()}} for name, d in datasets.items()},
        "gaps": {name: {"H_loc": d["gaps"], **{f"H_loc({lag} s)": gap_at(d["gaps"], lag) for lag in DATASET_LAGS}}
                 for name, d in datasets.items()},
        "phase gaps": {gap: {n: twoway.quote(draws[a][:, j] - draws[b][:, j]) for j, n in enumerate(STATISTICS)}
                       for gap, (a, b) in GAPS.items()},
        "phase statistics": {POPS[pop]: s for pop, s in stats.items()},
        "survival": {POPS[pop]: s for pop, s in surv.items()}, "n_boot": tamsd.N_BOOT, "cells": int(len(pilot))}
    (OUT / "raw.json").write_text(json.dumps(result, indent=1))
    for name, d in result["datasets"].items():
        for pop, v in d["populations"].items():
            print(name, pop, v["flights"], "flights", f"H = {v['H']:.3f}", np.round(v["H_band"], 3), v["H_loc(1000 s)"], v["H_loc(1840 s)"])
    for pop, s in result["phase statistics"].items():
        print(pop, s["flights"], "flights", {n: round(v["value"], 3) for n, v in s["average first"].items()})
