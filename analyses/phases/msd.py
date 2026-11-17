"""Mean squared displacement per phase, as in Vilpellet's paper.

For every run longer than 180 fixes, the horizontal MSD averaged along the run,
MSD_run(D) = mean_i |r(i + D) - r(i)|^2, with r the east-north position smoothed by his
6-fix Gaussian; at each lag D, the median over runs weighted by the T - D pairs each run
contributes. Search runs followed by a transition are dropped. The phase of a fix comes from
the stored runs (see durations.py), matched by flight, segment and time. Each figure shows two
populations: every labelled run, and the runs of the fixed cohort C_3600.

    uv run python msd.py [--open] [--recompute]    # from this folder; writes out/ (--open: msd_open.*)
"""

import json
import os
import sys
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common.legacy import stream_flights

sys.path.insert(0, str(Path(__file__).parents[1] / "universality"))
import openflights  # noqa: E402

DISK = Path("/Volumes/SSD_DISANTE")
DERIVED = {"para": DISK / "paragliders/ffvl_cfd_igc/derived", "hang": DISK / "hang_gliders/delta_cfd_igc/derived"}
RUNS = "segmentation/vilpellet/phase_segments.parquet"
# C_L: the retained segments of the chapter 3 cohort of lag L that last at least L / support_fraction.
COHORT_LAG = 3600
COHORT = {d: DISK / f"derived-audit/chapter3-fixed-20260917/{d}/cohort-1000.json" for d in DERIVED}
FLIGHTS = DISK / "xc-lab/universality/flights_grid10_support0.8_hmax10_declared_open_dem200m/flights.parquet"
POPULATIONS = {"all": "all 1 Hz flights", "C_3600": r"$C_{3600}$ (1 Hz)"}
POPULATIONS_OPEN = {"all": "open 1 Hz flights", "C_3600": r"$C_{3600}$, open (1 Hz)"}
PHASES = ("transition", "search", "climb")
MIN_FIXES, SMOOTH = 180, 6  # his cut (runs strictly longer) and smoothing window, in fixes
# He takes every lag below min(T, 600); we keep a log-spaced subset of them to fit in memory.
DELTAS = np.unique(np.round(np.geomspace(1, 599, 100)).astype(int))
MAX_PLOT, MIN_RUNS = 360, 100  # his plotted range; a lag needs this many runs to be drawn
OUT = Path(__file__).with_name("out")
COLORS = {"transition": "#2a78d6", "search": "#eb6834", "climb": "#1baf7a"}
NAMES = {"para": ("paragliders", "o"), "hang": ("hang gliders", "*")}


def msd_runs(runs):
    """Runs longer than MIN_FIXES, without the searches whose next phase is a transition
    (his "ARS followed by commute is not research"; a search ending the flight stays)."""
    runs = runs.sort_values(["flight_id", "segment_id", "t_start"], kind="stable").reset_index(drop=True)
    phase, flight = runs["phase"].to_numpy(object), runs["flight_id"].to_numpy(object)
    # Blocks of consecutive runs of one phase; the phase of the next block in the same flight.
    head = np.r_[True, (phase[1:] != phase[:-1]) | (flight[1:] != flight[:-1])]
    first = np.flatnonzero(head)
    same = np.r_[flight[first[1:]] == flight[first[:-1]], False]
    following = np.where(same, np.r_[phase[first[1:]], None], None)[np.cumsum(head) - 1]
    drop = (phase == "search") & (following == "transition")
    return runs[~drop & (runs["n_fixes"] > MIN_FIXES).to_numpy()].reset_index(drop=True)


def msd(pos, deltas):
    """Time-averaged MSD of one run at lags below len(pos), by FFT:
    sum_i |r(i+D) - r(i)|^2 = sum_{i<T-D} |r(i)|^2 + sum_{i>=D} |r(i)|^2 - 2 sum_i r(i).r(i+D)."""
    t = len(pos)
    x = pos - pos.mean(axis=0)
    n = 1 << (2 * t - 1).bit_length()
    f = np.fft.rfft(x, n=n, axis=0)
    auto = np.fft.irfft(np.abs(f) ** 2, n=n, axis=0)[:t].sum(axis=1)
    cum = np.r_[0.0, np.cumsum((x**2).sum(axis=1))]
    d = np.asarray(deltas)
    return (cum[t - d] + cum[t] - cum[d] - 2 * auto[d]) / (t - d)


def smooth(v):
    """His position smoothing; the partial windows at the ends stay NaN, as in his code."""
    return v.rolling(SMOOTH, win_type="gaussian", center=True).mean(std=SMOOTH / 6)


def flight_msds(fixes, runs, deltas=DELTAS):
    """MSD at the lags `deltas` of each run of one flight (NaN where D >= T); a run touching a NaN
    position is skipped, as in his code."""
    fixes = fixes.sort_values(["segment_id", "t"], kind="stable")
    xy = fixes.groupby("segment_id")[["E", "N"]].transform(smooth).to_numpy(float)
    key = fixes["segment_id"].to_numpy() * 1e6 + fixes["t"].to_numpy(float)
    out = np.full((len(runs), len(deltas)), np.nan)
    for k, run in enumerate(runs.itertuples()):
        a = np.searchsorted(key, run.segment_id * 1e6 + run.t_start)
        b = np.searchsorted(key, run.segment_id * 1e6 + run.t_end, side="right")
        assert b - a == run.n_fixes, (run.flight_id, run.segment_id, run.t_start)
        pos = xy[a:b]
        if np.isnan(pos).any():
            continue
        lags = deltas < len(pos)
        out[k, lags] = msd(pos, deltas[lags])
    return out


def _job(span, fixes, runs, deltas):
    return span, flight_msds(fixes, runs, deltas)


def weighted_median(values, weights):
    """His estimator: the first value, in increasing order, whose cumulative weight reaches half."""
    order = np.argsort(values)
    cdf = np.cumsum(weights[order]) / weights.sum()
    return values[order][np.searchsorted(cdf, 0.5)]


def weighted_quantile(values, weights, q):
    """The same estimator at level q: the first value whose cumulative weight reaches q."""
    order = np.argsort(values)
    cdf = np.cumsum(weights[order]) / weights.sum()
    return values[order][np.searchsorted(cdf, q)]


def cohort_segments(discipline, lag=COHORT_LAG):
    """(flight, segment) pairs of C_lag: the segments of the cohort-1000 members that last at
    least lag / support_fraction seconds (chapter 3, fixed population)."""
    cohort = json.loads(COHORT[discipline].read_text())
    shortest = lag / cohort["support_fraction"]
    return {(m["flight_id"], s["segment_id"]) for m in cohort["members"] for s in m["segments"]
            if s["duration_s"] >= shortest}


def open_flights(discipline):
    """Ids of the declared open flights with an opening above openflights.MIN_OPENING."""
    f = pd.read_parquet(FLIGHTS, columns=["discipline", "flight_id", "flight_type", "ends_km", "path_km"])
    f = f[f["discipline"] == discipline]
    return set(f.loc[openflights.is_open(f["flight_type"], f["ends_km"], f["path_km"]), "flight_id"].astype(str))


def in_segments(runs, segments):
    pairs = pd.MultiIndex.from_arrays([runs["flight_id"].astype(str), runs["segment_id"].astype(int)])
    return pairs.isin(list(segments))


def medians(runs, values, deltas=DELTAS, phases=PHASES):
    """Per phase: the weighted median over runs at each lag, its 10th and 90th percentiles over
    runs (q10, q90), how many runs reach the lag, and how many flights and runs the phase has."""
    result = {}
    for phase in phases:
        mask = (runs["phase"] == phase).to_numpy()
        v, t = values[mask], runs.loc[mask, "n_fixes"].to_numpy(float)
        out = {"delta": deltas.tolist(), "msd": [], "q10": [], "q90": [], "runs": [],
               "flights": int(runs.loc[mask, "flight_id"].nunique()), "n_runs": int(mask.sum())}
        for k, d in enumerate(deltas):
            ok = ~np.isnan(v[:, k])
            out["runs"].append(int(ok.sum()))
            for key, q in (("msd", 0.5), ("q10", 0.1), ("q90", 0.9)):
                out[key].append(float(weighted_quantile(v[ok, k].astype(float), t[ok] - d, q)) if ok.any() else None)
        result[phase] = out
    return result


def compute(discipline, only_open=False, deltas=DELTAS, phases=PHASES, lags=(COHORT_LAG,)):
    """Medians for each population, from one pass over the fixes. The run selection reads
    whole flights; C_3600 then keeps the runs inside its segments. With only_open, only the
    flights of open_flights. `lags`: the C_L populations besides all; `phases`: only these runs."""
    runs = msd_runs(pd.read_parquet(DERIVED[discipline] / RUNS))
    if only_open:
        runs = runs[runs["flight_id"].astype(str).isin(open_flights(discipline))]
    runs = runs[runs["phase"].isin(phases)]
    runs = runs.sort_values("flight_id", kind="stable").reset_index(drop=True)
    ids = runs["flight_id"].to_numpy()
    starts = np.flatnonzero(np.r_[True, ids[1:] != ids[:-1]])
    where = dict(zip(ids[starts], zip(starts, np.r_[starts[1:], len(ids)])))
    values = np.full((len(runs), len(deltas)), np.nan, np.float32)

    def store(future):
        (a, b), v = future.result()
        values[a:b] = v

    # This process reads the flights; the workers compute their MSDs. A bounded window of
    # flights in flight keeps memory flat.
    with ProcessPoolExecutor(os.cpu_count()) as pool:
        pending = set()
        for fixes in stream_flights(DERIVED[discipline] / "fixes.parquet", ["segment_id", "t", "E", "N"]):
            span = where.get(str(fixes["flight_id"].iat[0]))
            if span is None:
                continue
            pending.add(pool.submit(_job, span, fixes, runs.iloc[span[0]:span[1]], deltas))
            if len(pending) >= 4 * os.cpu_count():
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    store(future)
        for future in pending:
            store(future)
    result = {"all": medians(runs, values, deltas, phases)}
    for lag in lags:
        keep = in_segments(runs, cohort_segments(discipline, lag))
        result[f"C_{lag}"] = medians(runs[keep], values[keep], deltas, phases)
    return result


def curve(r, n=30, max_lag=MAX_PLOT):
    """His plotted points: n log-spaced lags up to max_lag, log MSD interpolated in log D;
    only lags with at least MIN_RUNS runs."""
    d, m, c = (np.array(r[k], dtype=float) for k in ("delta", "msd", "runs"))
    keep = (c >= MIN_RUNS) & (d <= max_lag)
    d, m = d[keep], m[keep]
    if not len(d):
        return d, m
    grid = np.linspace(0, np.log10(d.max()), n)
    return 10**grid, 10 ** np.interp(grid, np.log10(d), np.log10(m))


def slopes(r, n=30, max_lag=MAX_PLOT):
    """Local exponent d log MSD / d log D on the plotted points."""
    d, m = curve(r, n, max_lag)
    return d, np.gradient(np.log(m), np.log(d))


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=200, bbox_inches="tight")


def plot_msd(results, path, populations=POPULATIONS):
    """Rows: populations; columns: disciplines."""
    fig, axes = plt.subplots(2, 2, figsize=(8, 6.6), sharex=True, sharey=True)
    for row, (pop, label) in zip(axes, populations.items()):
        for ax, (d, (name, marker)) in zip(row, NAMES.items()):
            r = results[pop][d]
            for phase in PHASES:
                ax.scatter(*curve(r[phase]), s=14, marker=marker, color=COLORS[phase], label=phase)
            # Guides as in his figure: slope 2 above the transitions and below the climbs from 1 s,
            # slope 0.6 under the searches at 100 s.
            t = np.geomspace(1, MAX_PLOT, 50)
            tr, cl, se = (curve(r[p]) for p in ("transition", "climb", "search"))
            ax.plot(t, 3 * tr[1][0] * t**2, "k--", lw=1)
            ax.plot(t[t > 3], cl[1][0] / 3 * t[t > 3] ** 2, "k--", lw=1)
            if len(se[0]):
                ax.plot(t, 0.5 * np.interp(100, *se) * (t / 100) ** 0.6, color="#555555", ls=":", lw=1)
            ax.set(xscale="log", yscale="log", title=f"{name}, {label}")
        row[0].set_ylabel(r"MSD (m$^2$)")
    for ax in axes[-1]:
        ax.set_xlabel(r"$\Delta$ (s)")
    handles = [plt.Line2D([], [], color=COLORS[p], marker="o", ls="", label=p) for p in PHASES]
    handles += [plt.Line2D([], [], color="k", ls="--", label=r"$\Delta^2$"),
                plt.Line2D([], [], color="#555555", ls=":", label=r"$\Delta^{0.6}$")]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.04))
    save(fig, path)


def plot_slopes(results, path, populations=POPULATIONS):
    """Local exponent d log MSD / d log D: 2 while ballistic, then the exponent of each phase."""
    fig, axes = plt.subplots(2, 2, figsize=(8, 6.4), sharex=True, sharey=True)
    for row, (pop, label) in zip(axes, populations.items()):
        for ax, (d, (name, marker)) in zip(row, NAMES.items()):
            for phase in PHASES:
                if not len(curve(results[pop][d][phase])[0]):
                    continue
                ax.plot(*slopes(results[pop][d][phase]), color=COLORS[phase], marker=marker, ms=4, lw=1,
                        label=phase)
            for y in (2, 1):
                ax.axhline(y, color="#999999", lw=0.6, ls=":")
            ax.set(xscale="log", title=f"{name}, {label}")
        row[0].set_ylabel(r"$d\,\log\mathrm{MSD} / d\,\log\Delta$")
    for ax in axes[-1]:
        ax.set_xlabel(r"$\Delta$ (s)")
    axes[0, 0].legend(frameon=False)
    save(fig, path)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    only_open = "--open" in sys.argv
    tag, populations = ("_open", POPULATIONS_OPEN) if only_open else ("", POPULATIONS)
    saved = OUT / f"msd{tag}.json"
    # The paraglider pass reads 40 GB of fixes: reuse it unless asked.
    if "--recompute" in sys.argv or not saved.exists():
        by_discipline = {d: compute(d, only_open) for d in DERIVED}
        saved.write_text(json.dumps({pop: {d: by_discipline[d][pop] for d in DERIVED} for pop in populations}))
    results = json.loads(saved.read_text())
    plot_msd(results, OUT / f"msd{tag}.pdf", populations)
    plot_slopes(results, OUT / f"msd_slopes{tag}.pdf", populations)
