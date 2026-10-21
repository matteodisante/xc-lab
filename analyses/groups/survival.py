"""Survival of the phase durations of paragliders, solo and in a group, with the tail exponent mu.

Method of analyses/phases/durations.py, which is Vilpellet's: a run's duration is its number
of fixes (1 Hz, so seconds), a search counts only when the next run of the flight is a climb,
values outside the 0.0001 and 99.9999 percentiles are dropped, and powerlaw.Fit picks xmin
and mu = alpha - 1. Since that xmin moves from one population to another, the populations are
compared above one threshold, the xmin of the solo flights: mu there is the Hill estimate,
with a 5-95% band over N_BOOT draws of the dependency blocks, as in tamsd.py.

A run keeps its whole duration and goes to the population of the stretch (tamsd.stretches)
that holds its middle: cutting it where a flight enters or leaves a group would shorten the
durations the survival measures. Runs whose middle lies in no stretch (the local phase, or a
flight the labels do not see at every instant) are left out. Paragliders only.

    uv run python survival.py [--run NAME]    # from this folder; writes out/<run>/survival.*
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from phase_msd import PHASES, RUNS
from tamsd import COLORS, LABELS, N_BOOT, OUT, POPULATIONS, SEED, save, stretches

sys.path.insert(0, str(Path(__file__).parents[1] / "phases"))
from durations import ccdf, fit, hill, log_resample, trim  # noqa: E402


def counted(runs):
    """Every transition and climb; a search only when the next run of the flight is a climb."""
    runs = runs.sort_values(["flight_id", "segment_id", "t_start"], kind="stable")
    following = runs.groupby("flight_id", sort=False)["phase"].shift(-1)
    return runs[(runs["phase"] != "search") | (following == "climb")]


def assign(runs, spans):
    """Each run with the population of the stretch of its flight that holds its middle."""
    both = runs.assign(middle=(runs["t_start"] + runs["t_end"]) / 2).merge(spans, on="flight")
    return both[(both["a"] <= both["middle"]) & (both["middle"] < both["b"])].reset_index(drop=True)


def mu_band(x, blocks, xmin, rng):
    """5-95% of the Hill mu above xmin over N_BOOT draws of the blocks that have runs there."""
    tail = x >= xmin
    ids, k = np.unique(blocks[tail], return_inverse=True)
    counts, logs = np.bincount(k).astype(float), np.bincount(k, np.log(x[tail] / xmin))
    w = rng.multinomial(len(ids), np.full(len(ids), 1 / len(ids)), size=N_BOOT)
    return np.percentile((w @ counts) / (w @ logs), [5, 95]).tolist()


def plot(data, fits, path):
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8))
    for ax, p in zip(axes, PHASES):
        for pop, label in POPULATIONS.items():
            f, x = fits[pop][p], data[pop][p]
            mu, (lo, hi), xmin = f["mu_common"], f["mu_common_band"], f["xmin_common"]
            ax.scatter(*log_resample(*ccdf(x)), s=14, color=COLORS[pop],
                       label=rf"{label}: $\mu = {mu:.2f}$ [{lo:.2f}, {hi:.2f}]")
            t = np.geomspace(xmin, x.max(), 50)
            ax.plot(t, (x >= xmin).mean() * (t / xmin) ** -mu, color=COLORS[pop], lw=1, ls="--")
        ax.axvline(xmin, color="#999999", lw=0.8, ls=":")
        ax.set(xscale="log", yscale="log", title=f"{p}, paragliders", xlabel=r"$\tau$ (s)",
               ylabel=rf"$P(\tau_{{\mathrm{{{p}}}}} > \tau)$")
        ax.legend(loc="lower left", framealpha=1, edgecolor="none", fontsize=6.5)
    save(fig, path)


if __name__ == "__main__":
    flights = pd.read_parquet(LABELS / "flights.parquet")
    spans = stretches(flights, pd.read_parquet(LABELS / "membership.parquet"),
                      pd.read_parquet(LABELS / "flight_summary.parquet"))
    numbers = flights.loc[flights["discipline"] == "para", ["flight_id", "flight"]]
    runs = assign(counted(pd.read_parquet(RUNS["para"])).merge(numbers, on="flight_id"), spans)
    block = pd.read_parquet(LABELS / "flight_summary.parquet").set_index("flight")["block"]
    rng = np.random.default_rng(SEED)
    data, fits = {pop: {} for pop in POPULATIONS}, {pop: {} for pop in POPULATIONS}
    for p in PHASES:
        for pop in POPULATIONS:  # solo first: its xmin is the common threshold
            mine = runs[(runs["population"] == pop) & (runs["phase"] == p)]
            durations = mine["n_fixes"].to_numpy(float)
            x = trim(durations)
            blocks = block.reindex(mine["flight"]).to_numpy()[np.isin(durations, x)]
            xmin = fits["solo"][p]["xmin"] if pop != "solo" else None
            f = fits[pop][p] = fit(x) | {"flights": int(mine["flight"].nunique()), "median": float(np.median(x)),
                                         "mean": float(x.mean())}
            xmin = f["xmin"] if xmin is None else xmin
            f |= {"xmin_common": xmin, "n_common": int((x >= xmin).sum()), "mu_common": hill(x, xmin),
                  "mu_common_band": mu_band(x, blocks, xmin, rng)}
            data[pop][p] = x
            lo, hi = f["mu_common_band"]
            print(f"{pop:10s} {p:10s} {f['n']:8d} runs of {f['flights']:6d} flights, median {f['median']:5.0f} s, "
                  f"mu = {f['mu']:.2f} above its xmin {f['xmin']:.0f} s; above {xmin:.0f} s: "
                  f"mu = {f['mu_common']:.2f} [{lo:.2f}, {hi:.2f}] ({f['n_common']} runs)")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "survival.json").write_text(json.dumps(fits, indent=1))
    plot(data, fits, OUT / "survival.pdf")
