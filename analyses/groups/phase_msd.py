"""MSD per phase of solo flights and of group flights during their group time, with slopes.

Method of analyses/phases/msd.py, which is Vilpellet's: runs longer than 180 fixes, searches
followed by a transition dropped (the rule reads whole flights), positions smoothed by his
6-fix Gaussian within each segment, per run the MSD averaged along it, at each lag the median
over runs weighted by the T - D pairs each run contributes. Here a run is first cut to the
stretches of its population (tamsd.stretches), so a group flight's runs count only in its
group time, and the pieces longer than 180 fixes are kept. Phases exist on 1 Hz flights only.

Slopes of log MSD against log D, fitted with equal weight per drawn lag: transitions over
every drawn lag, searches over SEARCH_FIT, the last part of their curve, where the local
exponent levels off. Bands: 5-95% of N_BOOT draws of the dependency blocks of the note, each
draw redoing the medians and the fit on the lags drawn for the whole sample.

    uv run python phase_msd.py [--run NAME] [--recompute]    # from this folder; writes out/<run>/
"""

import json
import os
import sys
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lab.legacy import stream_flights
from lab.msd import tamsd
from tamsd import CACHE, FIXES, LABELS, NAMES, OUT, POPULATIONS, SEED, folders, save, stretches

RUNS = {"para": FIXES["para"].parent / "segmentation/vilpellet/phase_segments.parquet",
        "hang": FIXES["hang"].parent / "segmentation/vilpellet/phase_segments.parquet"}
PHASES = ("transition", "search", "climb")
PHASE_COLORS = {"transition": "#2a78d6", "search": "#eb6834", "climb": "#1baf7a"}
MIN_FIXES, SMOOTH = 180, 6  # his cut (pieces strictly longer) and smoothing window, in fixes
DELTAS = np.unique(np.round(np.geomspace(1, 599, 100)).astype(int))  # as analyses/phases
MAX_PLOT, MIN_RUNS = 360, 100  # his plotted range; a lag needs this many pieces to be drawn
SEARCH_FIT = (100, 360)  # s
N_BOOT = 200


def msd_runs(runs):
    """Runs longer than MIN_FIXES, without the searches whose next phase is a transition
    (his "ARS followed by commute is not research"; a search ending the flight stays)."""
    runs = runs.sort_values(["flight_id", "segment_id", "t_start"], kind="stable").reset_index(drop=True)
    phase, flight = runs["phase"].to_numpy(object), runs["flight_id"].to_numpy(object)
    head = np.r_[True, (phase[1:] != phase[:-1]) | (flight[1:] != flight[:-1])]
    first = np.flatnonzero(head)
    same = np.r_[flight[first[1:]] == flight[first[:-1]], False]
    following = np.where(same, np.r_[phase[first[1:]], None], None)[np.cumsum(head) - 1]
    drop = (phase == "search") & (following == "transition")
    return runs[~drop & (runs["n_fixes"] > MIN_FIXES).to_numpy()].reset_index(drop=True)


def cut(runs, spans):
    """Each run cut to the stretches of its flight: flight, segment_id, phase, population,
    t_start, t_end (first and last fix, 1 Hz) of every piece longer than MIN_FIXES."""
    both = runs.merge(spans, on="flight")
    start = np.maximum(both["t_start"], np.ceil(both["a"]))
    end = np.minimum(both["t_end"], np.ceil(both["b"]) - 1)
    both = both.assign(t_start=start, t_end=end)
    both = both[both["t_end"] - both["t_start"] + 1 > MIN_FIXES]
    return both[["flight", "segment_id", "phase", "population", "t_start", "t_end"]].reset_index(drop=True)


def smooth(v):
    """His position smoothing; the partial windows at the ends stay NaN, as in his code."""
    return v.rolling(SMOOTH, win_type="gaussian", center=True).mean(std=SMOOTH / 6)


def flight_msds(fixes, pieces):
    """MSD at DELTAS of each piece of one flight (NaN where D >= T); a piece touching a NaN
    position is skipped, as in his code."""
    fixes = fixes.sort_values(["segment_id", "t"], kind="stable")
    xy = fixes.groupby("segment_id")[["E", "N"]].transform(smooth).to_numpy(float)
    key = fixes["segment_id"].to_numpy() * 1e6 + fixes["t"].to_numpy(float)
    out = np.full((len(pieces), len(DELTAS)), np.nan)
    for k, p in enumerate(pieces.itertuples()):
        a = np.searchsorted(key, p.segment_id * 1e6 + p.t_start)
        b = np.searchsorted(key, p.segment_id * 1e6 + p.t_end, side="right")
        pos = xy[a:b]
        if np.isnan(pos).any() or b - a <= MIN_FIXES:
            continue
        lags = DELTAS < len(pos)
        out[k, lags] = tamsd(pos, DELTAS[lags])
    return out


def _job(span, fixes, pieces):
    return span, flight_msds(fixes, pieces)


def measure(discipline, flights, spans):
    """Pieces of one discipline and their MSDs, from one pass over its fixes."""
    numbers = flights.loc[flights["discipline"] == discipline, ["flight_id", "flight"]]
    runs = msd_runs(pd.read_parquet(RUNS[discipline])).merge(numbers, on="flight_id")
    pieces = cut(runs, spans).sort_values("flight", kind="stable").reset_index(drop=True)
    ids = pieces["flight"].to_numpy()
    starts = np.flatnonzero(np.r_[True, ids[1:] != ids[:-1]])
    where = dict(zip(ids[starts], zip(starts, np.r_[starts[1:], len(ids)])))
    by_id = numbers.set_index("flight_id")["flight"]
    values = np.full((len(pieces), len(DELTAS)), np.nan, np.float32)

    def store(future):
        (a, b), v = future.result()
        values[a:b] = v

    with ProcessPoolExecutor(os.cpu_count()) as pool:
        pending = set()
        for fixes in stream_flights(FIXES[discipline], ["segment_id", "t", "E", "N"]):
            span = where.get(by_id.get(str(fixes["flight_id"].iat[0]), -1))
            if span is None:
                continue
            pending.add(pool.submit(_job, span, fixes, pieces.iloc[span[0]:span[1]]))
            if len(pending) >= 4 * os.cpu_count():
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    store(future)
        for future in pending:
            store(future)
    return pieces, values


def save_pieces(path, flights, spans):
    """Measure the pieces of every discipline for `spans` and save them in `path`."""
    arrays = {}
    for d in FIXES:
        pieces, values = measure(d, flights, spans)
        arrays.update({f"{d}/{c}": pieces[c].to_numpy(None if pd.api.types.is_numeric_dtype(pieces[c]) else str)
                       for c in pieces.columns})
        arrays[f"{d}/values"] = values
    np.savez_compressed(path, **arrays)


def read_pieces(path, discipline):
    """The pieces of one discipline saved by save_pieces, and their MSDs at DELTAS."""
    data = np.load(path, allow_pickle=True)  # older copies hold object arrays
    columns = {k.split("/")[1]: data[k] for k in data.files if k.startswith(discipline + "/")}
    values = columns.pop("values")
    return pd.DataFrame(columns), values


def weighted_median(values, weights):
    """His estimator: the first value, in increasing order, whose cumulative weight reaches half."""
    order = np.argsort(values)
    cdf = np.cumsum(weights[order]) / weights.sum()
    return values[order][np.searchsorted(cdf, 0.5)]


def medians(pieces, values, phases=PHASES, populations=POPULATIONS):
    """Per population of `populations` and phase: the weighted median over pieces at each lag,
    how many pieces and flights reach it."""
    out = {}
    n = (pieces["t_end"] - pieces["t_start"] + 1).to_numpy(float)
    for pop in populations:
        out[pop] = {}
        for phase in phases:
            mask = ((pieces["population"] == pop) & (pieces["phase"] == phase)).to_numpy()
            v, t = values[mask], n[mask]
            median, count = [], []
            for k, d in enumerate(DELTAS):
                ok = ~np.isnan(v[:, k])
                count.append(int(ok.sum()))
                median.append(float(weighted_median(v[ok, k].astype(float), t[ok] - d)) if ok.any() else None)
            out[pop][phase] = {"delta": DELTAS.tolist(), "msd": median, "runs": count,
                               "flights": int(pieces.loc[mask, "flight"].nunique())}
    return out


def drawn(r):
    """Lags up to MAX_PLOT that at least MIN_RUNS pieces reach, with their MSD."""
    d, m, c = (np.array(r[k], dtype=float) for k in ("delta", "msd", "runs"))
    keep = (c >= MIN_RUNS) & (d <= MAX_PLOT)
    return d[keep], m[keep]


def slope(d, m, lo=0, hi=np.inf):
    """Slope of log MSD against log D over lo <= D <= hi, equal weight per lag."""
    keep = (d >= lo) & (d <= hi)
    return float(np.polyfit(np.log10(d[keep]), np.log10(m[keep]), 1)[0]) if keep.sum() >= 2 else None


def local_exponent(d, m):
    return np.gradient(np.log(m), np.log(d))


FITS = {"transition": (0, np.inf), "search": SEARCH_FIT}


def fitted(results, lags):
    """Slopes of transitions over every lag and of searches over SEARCH_FIT, on the lags
    `lags[pop][phase]` (those drawn for the whole sample); None where too few lags."""
    out = {}
    for pop, r in results.items():
        out[pop] = {}
        for phase, (lo, hi) in FITS.items():
            delta, msd = np.array(r[phase]["delta"]), np.array(r[phase]["msd"], dtype=float)
            d = lags[pop][phase]
            out[pop][phase] = slope(d, msd[np.searchsorted(delta, d)], lo, hi) if d.size else None
    return out


def bands(pieces, values, blocks, lags, rng, populations=POPULATIONS):
    """5-95% of the slopes over N_BOOT draws of whole blocks of pieces."""
    order = np.argsort(blocks, kind="stable")
    starts = np.flatnonzero(np.r_[True, blocks[order][1:] != blocks[order][:-1]])
    members = np.split(order, starts[1:])
    draws = []
    for _ in range(N_BOOT):
        idx = np.concatenate([members[c] for c in rng.integers(0, len(members), len(members))])
        draws.append(fitted(medians(pieces.iloc[idx].reset_index(drop=True), values[idx], tuple(FITS), populations),
                            lags))
    out = {}
    for pop in draws[0]:
        out[pop] = {}
        for phase in FITS:
            v = [d[pop][phase] for d in draws if d[pop][phase] is not None]
            out[pop][phase] = np.percentile(v, [5, 95]).tolist() if len(v) == N_BOOT else None
    return out


def plot(results, path):
    """Rows: MSD per phase, its local exponent; columns: discipline and population."""
    pops = list(POPULATIONS)
    fig, axes = plt.subplots(2, 2 * len(pops), figsize=(4 * len(pops) * 2 / 1.6, 6.5), sharex=True)
    for c, (d, pop) in enumerate((d, p) for d in NAMES for p in pops):
        r = results[d][pop]
        for phase in PHASES:
            x, m = drawn(r[phase])
            if not x.size:
                continue
            axes[0, c].plot(x, m, color=PHASE_COLORS[phase], marker="o", ms=2.5, lw=1, label=f"{phase} ({r[phase]['flights']})")
            axes[1, c].plot(x, local_exponent(x, m), color=PHASE_COLORS[phase], lw=1)
        axes[0, c].set(xscale="log", yscale="log", title=f"{NAMES[d]}\n{POPULATIONS[pop]}")
        axes[0, c].legend(frameon=False, fontsize=7)
        for y in (2, 1):
            axes[1, c].axhline(y, color="#999999", lw=0.6, ls=":")
        axes[1, c].axvspan(*SEARCH_FIT, color=PHASE_COLORS["search"], alpha=0.08, lw=0)
        axes[1, c].set(xscale="log", xlabel=r"$\Delta$ (s)", ylim=(0, 2.3))
    axes[0, 0].set_ylabel(r"MSD (m$^2$)")
    axes[1, 0].set_ylabel(r"$d\,\log\mathrm{MSD} / d\,\log\Delta$")
    save(fig, path)


if __name__ == "__main__":
    folders()
    saved = CACHE / "phase_pieces.npz"
    summary = pd.read_parquet(LABELS / "flight_summary.parquet")
    # Two passes over the fixes take about 6 min: reuse the pieces unless asked.
    if "--recompute" in sys.argv or not saved.exists():
        flights = pd.read_parquet(LABELS / "flights.parquet")
        save_pieces(saved, flights, stretches(flights, pd.read_parquet(LABELS / "membership.parquet"), summary))
    block = summary.set_index("flight")["block"]
    rng = np.random.default_rng(SEED)
    results, slopes = {}, {}
    for d in FIXES:
        pieces, values = read_pieces(saved, d)
        results[d] = medians(pieces, values)
        lags = {pop: {phase: drawn(results[d][pop][phase])[0] for phase in FITS} for pop in POPULATIONS}
        slopes[d] = {"slope": fitted(results[d], lags),
                     "band": bands(pieces, values, block.reindex(pieces["flight"]).to_numpy(), lags, rng)}
    (OUT / "phase_msd.json").write_text(json.dumps(results))
    (OUT / "phase_slopes.json").write_text(json.dumps(slopes, indent=1))
    plot(results, OUT / "phase_msd.pdf")
    print(json.dumps(slopes, indent=1))
