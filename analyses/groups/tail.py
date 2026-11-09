"""The tail of the glide (transition) durations of paragliders, in a group against alone, on the open flights.

Populations and runs as in raw.py: S, A, G on the pieces of dataset 1000 (pieces of at least 1250 s), each
counted run (survival.counted, trimmed as durations.trim) whole in the population of the piece that holds its
middle; a duration is the number of fixes of the run, s. Bands: twoway.py, the draws of raw.py.

1. Hill mu of P(tau > t) ~ t^-mu above a threshold xmin, mu = sum 1 / sum ln(x / xmin) over x >= xmin: against
   XMIN_GRID and at the xmin of S chosen by powerlaw.Fit (as in survival.py), with the two-way band and the number
   of runs above, for glides and, as references, for searches that end in a climb and climbs.
2. The same on the same route, with the weights of matched.py (caliper chosen there, cover 0.8, level "all"):
   a run of a piece weighs 1, a run of a control 1 / (controls of its piece), and the draws weigh the flights
   of the piece and of the controls; weighted mu = sum w / sum w ln(x / xmin) for G, S|G, A, S|A and
   (G - S|G) - (A - S|A).
3. What the long glides are, for glides of at least 1000 s and 2000 s and for all glides: path, net displacement,
   straightness, ground speed along the path, vertical speed of the ends, height lost per km of net
   displacement (the SINK of the lab), share of the time rising; medians with their two-way bands (the
   weighted median over the draws of the cells), quartiles, and the gaps G - S and A - S of the medians with their
   bands. One pass over the fixes of the flights that have a glide in a piece, cached in CACHE (--recompute).

    uv run python tail.py --open [--run NAME] [--sample N]    # after raw.py and matched.py --open
    # --sample N: the flights found in the first row groups of the fixes, up to N; writes tail_sampleN.*
"""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import isolated
import longlags
import matched
import raw
import route
import survival
import twoway
from blocks import with_takeoffs
from common.legacy import stream_flights
from phase_msd import PHASES, RUNS
from tamsd import CACHE, COLORS, FIXES, GRID_S, LABELS, OUT, SEED, folders, read_flights, save

sys.path.insert(0, str(Path(__file__).parents[1] / "phases"))
from durations import fit, trim  # noqa: E402

XMIN_GRID = np.geomspace(300, 4000, 16)  # s
LONG = (0, 1000, 2000)  # s: the glides described (0: all)
REFERENCES = ("search", "climb")
DESCRIPTORS = ("path (km)", "net displacement (km)", "straightness", "ground speed (km/h)",
               "vertical speed (m/s)", "height lost (m/km)", "time rising (share)")
CHUNK = 250  # draws multiplied together
MU_LIMITS = (0, 10)  # drawn range of mu; the bands that leave it are cut
FIX_COLUMNS = ["segment_id", "t", "E", "N", "z"]


def trimmed(x):
    """The durations of `x` that durations.trim keeps (all of them when there are none)."""
    return np.isin(x, trim(x)) if len(x) else np.zeros(0, bool)


def hill_sums(x, unit, n_units, xmins, w=None):
    """Per unit (codes `unit` of the durations `x`, weights `w`, default 1) and xmin: the weight and the weight
    times ln(x / xmin) of the durations from xmin up; array (n_units, 2 len(xmins)), the weights first."""
    w = np.ones(len(x)) if w is None else np.asarray(w, float)
    k, out = len(xmins), np.zeros((n_units, 2 * len(xmins)))
    for j, m in enumerate(xmins):
        t = x >= m
        out[:, j] = np.bincount(unit[t], w[t], minlength=n_units)
        out[:, k + j] = np.bincount(unit[t], w[t] * np.log(x[t] / m), minlength=n_units)
    return out


def ratio(sums):
    """Hill mu from sums with the weights in the first half of the columns (any leading shape)."""
    k = sums.shape[-1] // 2
    with np.errstate(divide="ignore", invalid="ignore"):
        return sums[..., :k] / sums[..., k:]


def weighted_hill(x, w, xmin):
    """mu = sum w / sum w ln(x / xmin) over x >= xmin."""
    return float(ratio(hill_sums(x, np.zeros(len(x), int), 1, [xmin], w).sum(0))[0])


def mu_curve(x, cell, counts, xmins):
    """Hill mu at each of `xmins` for durations `x` of cells `cell`, with the two-way band of the stacked draws
    `counts` (rows) and the number of runs above."""
    k = len(xmins)
    sums = hill_sums(x, cell, counts.shape[1], xmins)
    total = sums.sum(0)
    draws = np.vstack([ratio(np.hstack([counts[a:a + CHUNK] @ sums[:, :k], counts[a:a + CHUNK] @ sums[:, k:]]))
                       for a in range(0, len(counts), CHUNK)])
    value = ratio(total)
    return {"xmin": list(map(float, xmins)), "value": value.tolist(), "band": twoway.band(value, draws),
            "runs": total[:k].astype(int).tolist()}


def glide_descriptors(t, e, n, z, a, b):
    """The DESCRIPTORS of the runs that span the fixes a..b (indices, inclusive) of arrays sorted by time, each
    run within one segment: ground speed is the path over the time, vertical speed the change of z over the time,
    height lost per km of net displacement, time rising the share of the time in steps that gain height."""
    step = np.nan_to_num(np.r_[0, np.hypot(np.diff(e), np.diff(n))])
    dt, dz = np.r_[0, np.diff(t)], np.r_[0, np.diff(z)]
    rising = np.cumsum(np.where(dz > 0, dt, 0.0))
    path, span = np.cumsum(step)[b] - np.cumsum(step)[a], t[b] - t[a]
    net = np.hypot(e[b] - e[a], n[b] - n[a])
    with np.errstate(divide="ignore", invalid="ignore"):
        out = [path / 1000, net / 1000, net / path, path / span * 3.6, (z[b] - z[a]) / span,
               -(z[b] - z[a]) / (net / 1000), (rising[b] - rising[a]) / span]
    return pd.DataFrame(dict(zip(DESCRIPTORS, out)))


def flight_glides(fixes, glides):
    """glide_descriptors of the rows of `glides` (segment_id, t_start, t_end) in the fixes of one flight; rows
    whose ends are not fixes are left out."""
    fixes = fixes.sort_values(["segment_id", "t"], kind="stable")
    key = fixes["segment_id"].to_numpy() * 1e6 + fixes["t"].to_numpy(float)
    want = [glides["segment_id"].to_numpy() * 1e6 + glides[c].to_numpy(float) for c in ("t_start", "t_end")]
    a, b = (np.clip(np.searchsorted(key, w), 0, len(key) - 1) for w in want)
    ok = np.isclose(key[a], want[0]) & np.isclose(key[b], want[1]) & (b > a)
    d = glide_descriptors(*(fixes[c].to_numpy(float) for c in ("t", "E", "N", "z")), a[ok], b[ok])
    return d.set_index(glides.index[ok])


def read_glides(glides, wanted, frames=None):
    """The descriptors of the glides (rows with flight, segment_id, t_start, t_end), from one pass over the fixes
    (`frames`: one frame per flight, default the whole file) that stops when the `wanted` flights are all read;
    indexed as `glides`."""
    ids = pd.read_parquet(LABELS / "flights.parquet", columns=["flight_id", "flight"]).set_index("flight_id")["flight"]
    by_flight, out, left = dict(tuple(glides.groupby("flight"))), [], set(wanted)
    for fixes in stream_flights(FIXES["para"], FIX_COLUMNS) if frames is None else frames:
        flight = ids.get(str(fixes["flight_id"].iat[0]))
        if flight in by_flight:
            out.append(flight_glides(fixes, by_flight[flight]))
            left.discard(flight)
            if not left:
                break
    return pd.concat(out)


def sample_block(flights, n):
    """For a trial: the fixes, one frame per flight, of the open flights of consecutive row groups, enough of
    them to hold n (the flights cut by the block are left out), and their flight numbers."""
    file = pq.ParquetFile(FIXES["para"])
    number = flights.set_index("flight_id")["flight"]
    parts, g, seen = [], file.num_row_groups * 3 // 4, pd.Index([])
    while g < file.num_row_groups:
        parts.append(file.read_row_group(g, columns=["flight_id", *FIX_COLUMNS]).to_pandas())
        g += 1
        seen = seen.append(pd.Index(parts[-1]["flight_id"].unique())).unique()
        ids = seen[1:-1]
        if number.reindex(ids).notna().sum() >= n:
            break
    frames = [f.reset_index(drop=True) for _, f in pd.concat(parts).groupby("flight_id", sort=False)][1:-1]
    return frames, set(number.reindex(ids).dropna().astype(int))


def quartiles(values):
    q = np.nanpercentile(values, [25, 50, 75]) if np.isfinite(values).any() else [np.nan] * 3
    return {"q25": float(q[0]), "median": float(q[1]), "q75": float(q[2])}


def median_draws(v, cell, counts):
    """The weighted median of the values `v` (cell codes `cell`) in each draw, i.e. each row of `counts`: a run
    weighs the count of its cell; the lower median, NaN when a draw has no run."""
    o = np.argsort(v, kind="stable")
    v, cell = v[o], cell[o]
    out, step = np.empty(len(counts)), max(1, int(2e7 // max(len(v), 1)))
    for a in range(0, len(counts), step):
        cum = np.cumsum(counts[a:a + step][:, cell], axis=1, dtype=np.int32)
        out[a:a + step] = v[(2 * cum >= cum[:, -1:]).argmax(axis=1)]
        out[a:a + step][cum[:, -1] == 0] = np.nan
    return out


def describe_glides(table, populations, counts):
    """Per population and minimum duration of LONG, from `table` (a row per glide: flight, population, cell,
    n_fixes and the DESCRIPTORS): runs, flights and, per descriptor, the quartiles, and the median with its
    two-way band (the cells resampled over the rows of `counts`). Also, per minimum duration and descriptor, the
    gaps G - S and A - S of the medians with their bands (the draws of both sides are the same rows)."""
    out, draws, gaps = {}, {}, {}
    for pop, short in populations.items():
        out[short] = {}
        for m in LONG:
            rows = table[(table["population"] == pop) & (table["n_fixes"] >= m)]
            out[short][f"{m} s"] = r = {"runs": int(rows["path (km)"].notna().sum()), "flights": int(rows["flight"].nunique())}
            for name in DESCRIPTORS:
                ok = rows[name].notna().to_numpy() & np.isfinite(rows[name].to_numpy())
                v, c = rows[name].to_numpy()[ok], rows["cell"].to_numpy()[ok]
                med, d = (float(np.median(v)) if len(v) else np.nan), median_draws(v, c, counts) if len(v) else np.full(len(counts), np.nan)
                draws[short, m, name] = (med, d)
                r[name] = quartiles(v) | {"median band": twoway.band(med, d)}
    for m in LONG:
        for name in DESCRIPTORS:
            gaps.setdefault(f"{m} s", {})[name] = {}
            for gap, (x, y) in {"G - S": ("G", "S"), "A - S": ("A", "S")}.items():
                (mx, dx), (my, dy) = draws[x, m, name], draws[y, m, name]
                gaps[f"{m} s"][name][gap] = {"value": mx - my, "band": twoway.band(mx - my, dx - dy)}
    return out, gaps


def read_runs(flights):
    """The runs of the paragliders of `flights`: all of them with their seconds (`every`, for the phase time of a
    window), and the counted ones sorted by the key flight * RANK + middle. Only the columns used are read and
    the flights are numbered through the dictionary of the ids, to keep the memory low."""
    table = pq.read_table(RUNS["para"], columns=["flight_id", "segment_id", "phase", "t_start", "t_end", "n_fixes"])
    ids = table["flight_id"].combine_chunks().dictionary_encode()
    number = flights.set_index("flight_id")["flight"].reindex(ids.dictionary.to_pylist()).to_numpy(float)
    code = ids.indices.to_numpy(zero_copy_only=False)
    flight = number[code]
    keep = np.isfinite(flight)
    runs = table.drop_columns(["flight_id"]).to_pandas().loc[keep].assign(flight=flight[keep].astype(np.int64))
    del table, ids, code, flight
    runs["phase"] = runs["phase"].astype("category")
    runs = runs.assign(t_mid=(runs["t_start"] + runs["t_end"]) / 2, flight_id=lambda r: r["flight"])
    every = runs[["flight", "t_mid"]].assign(s=runs["n_fixes"].to_numpy(float))
    counted = survival.counted(runs)
    del runs
    counted = counted.assign(key=counted["flight"].to_numpy(float) * isolated.RANK + counted["t_mid"].to_numpy(float))
    return every, counted.sort_values("key", kind="stable").reset_index(drop=True).drop(columns="flight_id")


def population_runs(counted, p, wanted):
    """Per population of raw.POPS: the counted runs in its pieces (dataset 1000) of the flights `wanted`."""
    out = {}
    for pop in raw.POPS:
        mine = counted[raw.in_pieces(counted, *raw.pieces_of(p, pop))]
        out[pop] = mine[mine["flight"].isin(wanted)]
    return out


def phase_data(mine, phase, cell):
    """The trimmed durations of one phase of the runs `mine`, their cell codes and rows."""
    r = mine[mine["phase"] == phase]
    keep = trimmed(r["n_fixes"].to_numpy(float))
    r = r[keep]
    return r["n_fixes"].to_numpy(float), cell.reindex(r["flight"]).to_numpy().astype(int), r


def raw_tail(pops, cell, counts):
    """Part 1: per phase, the xmin of S, the glides' survival curves and, per population, mu against XMIN_GRID and
    at the xmin of S."""
    out = {"survival": {}}
    for phase in ("transition", *REFERENCES):
        data = {pop: phase_data(pops[pop], phase, cell)[:2] for pop in raw.POPS}
        xmin = fit(data["solo"][0])["xmin"]
        grid = [*XMIN_GRID, xmin]
        res = {}
        for pop, (x, c) in data.items():
            curve = mu_curve(x, c, counts, grid)
            res[raw.POPS[pop]] = {"runs": len(x), "flights": int(pops[pop]["flight"].nunique()),
                                  "at xmin of S": {"mu": curve["value"][-1], "band": [curve["band"][0][-1], curve["band"][1][-1]],
                                                   "runs above": curve["runs"][-1]},
                                  "curve": {"xmin": curve["xmin"][:-1], "value": curve["value"][:-1],
                                            "band": [curve["band"][0][:-1], curve["band"][1][:-1]], "runs": curve["runs"][:-1]}}
            if phase == "transition":
                out["survival"][raw.POPS[pop]] = {"runs": len(x), **survival.survival_curve(x, c, counts, band=twoway.share_band)}
        out[phase] = {"xmin of S": xmin, "populations": res}
        print(phase, "xmin of S", xmin, {s: round(r["at xmin of S"]["mu"], 2) for s, r in res.items()}, flush=True)
    return out


def windows(ctx, main):
    """What matched.phased and the run counts read for the rows `main` of a match table: the windows of the
    pieces and of the control segments (flight, a, b on the flight clock) and the phase time in each."""
    tr, s, i, j = ctx.tracks, *(main[c].to_numpy() for c in ("solo", "i", "j"))
    g = np.unique(main["piece"].to_numpy())
    w = SimpleNamespace(rows=main.index.to_numpy(), piece=g,
                        win_g=(tr["flight"][g], tr["start"][g], tr["start"][g] + (tr["n"][g] - 1) * GRID_S),
                        win_s=(tr["flight"][s], tr["start"][s] + i * GRID_S, tr["start"][s] + j * GRID_S))
    w.T_g, w.T_s = (isolated.window_sums(ctx.every, ["s"], *win)[:, 0] for win in (w.win_g, w.win_s))
    return w


def glide_sums(ctx, win, xmins):
    """Per window, hill_sums of its trimmed glides: (windows, 2 len(xmins))."""
    unit, idx = matched.window_runs(ctx, *win)
    x, glide = ctx.runs["n_fixes"].to_numpy(float)[idx], ctx.runs["phase"].to_numpy()[idx] == "transition"
    unit, x = unit[glide], x[glide]
    keep = trimmed(x)
    return hill_sums(x[keep], unit[keep], len(win[0]), xmins)


def same_route(ctx, table, line, xmins):
    """Part 2: weighted mu of G, S|G, A, S|A at `xmins` and (G - S|G) - (A - S|A), each as the point estimate and
    its two-way band, and the G runs above each xmin."""
    k, boots, out = len(xmins), {}, {"line": line, "xmin": list(map(float, xmins)), "sides": {}}
    for side, own, control in (("G", "G", "S|G"), ("A", "A", "S|A")):
        main = matched.select(table[table["side"] == side], "1000", **line)
        w = windows(ctx, main)
        sel = matched.phased(w, main)
        pos, rows = np.searchsorted(w.piece, np.unique(sel["piece"].to_numpy())), np.searchsorted(w.rows, sel.index.to_numpy())
        a = glide_sums(ctx, tuple(v[pos] for v in w.win_g), xmins)
        b = glide_sums(ctx, tuple(v[rows] for v in w.win_s), xmins)
        (group, ctrl, _), _ = matched.run_engine(ctx, sel, a, b, False)
        boots[own], boots[control] = ratio(group.T), ratio(ctrl.T)
        out["sides"][side] = {"pieces": int(len(pos)), "controls": int(len(sel)), "runs above": a[:, :k].sum(0).astype(int).tolist(),
                              "control runs above (weighted)": ctrl[:k, 0].tolist()}
    delta = {"G - S|G": boots["G"] - boots["S|G"], "A - S|A": boots["A"] - boots["S|A"]}
    delta["(G - S|G) - (A - S|A)"] = delta["G - S|G"] - delta["A - S|A"]
    out["mu"] = {s: twoway.quote(v) for s, v in {**boots, **delta}.items()}
    return out


def context(flights, summary, tracks_path, every, counted):
    """What matched.run_engine and matched.window_runs read: tracks, universe, weights, runs."""
    data = np.load(tracks_path)
    tracks = {key: data[key] for key in ("flight", "population", "start", "n")}
    universe = np.unique(tracks["flight"])
    pilot, block, _ = isolated.codes(flights, summary, universe)
    cell, of_pilot, of_block = twoway.cell_codes(pilot, block)
    boot = twoway.weights(cell, twoway.counts(of_pilot, of_block, np.random.default_rng(SEED)))
    return SimpleNamespace(tracks=tracks, universe=universe, boot=boot, row=lambda f: np.searchsorted(universe, f),
                           every=every, runs=counted)


def floats(v):
    return np.array(v, float)  # None (a NaN saved in the JSON) becomes NaN


def plot(result, path):
    """(a) survival of the glide durations with the power law above the xmin of S; (b) mu against xmin, the G runs
    above on the top axis; (c) the same route, weighted mu of G and S|G (A and S|A dashed)."""
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(15, 4.4))
    xmin = result["transition"]["xmin of S"]
    for pop, short in raw.POPS.items():
        col, s, res = COLORS[pop], result["survival"][short], result["transition"]["populations"][short]
        t, v, band = floats(s["grid"]), floats(s["value"]), floats(s["band"])
        ok = v > 0
        a.fill_between(t[ok], np.maximum(band[0][ok], 1e-9), band[1][ok], color=col, alpha=0.2, lw=0)
        mu, (lo, hi) = res["at xmin of S"]["mu"], res["at xmin of S"]["band"]
        a.plot(t[ok], v[ok], color=col, lw=1.3, label=rf"{short} ({s['runs']:,} runs): $\mu$ = {mu:.2f} [{lo:.2f}, {hi:.2f}]")
        above = t >= xmin
        a.plot(t[above], v[np.searchsorted(t, xmin)] * (t[above] / t[np.searchsorted(t, xmin)]) ** -mu, color=col, lw=1, ls="--")
        cur = res["curve"]
        b.fill_between(cur["xmin"], *cur["band"], color=col, alpha=0.2, lw=0)
        b.plot(cur["xmin"], cur["value"], color=col, lw=1.3, marker="o", ms=3, label=short)
    a.axvline(xmin, color="#999999", lw=0.8, ls=":")
    a.set(xscale="log", yscale="log", xlabel=r"$\tau$ (s)", ylabel=r"$P(\tau_{\mathrm{glide}} > \tau)$", ylim=(1e-5, 1.2),
          title="glide durations, open paragliders")
    a.legend(frameon=False, fontsize=7, loc="lower left")
    b.set(xscale="log", xlabel=r"$x_{\min}$ (s)", ylabel=r"Hill $\mu$ above $x_{\min}$")
    b.legend(frameon=False, fontsize=8)
    b.axvline(xmin, color="#999999", lw=0.8, ls=":")
    top = b.twiny()
    b.set_xlim(XMIN_GRID[0] * 0.95, XMIN_GRID[-1] * 1.05)
    top.set(xscale="log", xlim=b.get_xlim())
    n = result["transition"]["populations"]["G"]["curve"]["runs"]
    top.set_xticks(result["transition"]["populations"]["G"]["curve"]["xmin"][::3], [f"{n[i]:,}" for i in range(0, len(n), 3)], fontsize=7)
    top.minorticks_off()
    top.set_xlabel("G runs above", fontsize=8)
    r = result["same route"]
    x = r["xmin"][:-1]
    for side, (col, style) in matched.SIDE_STYLE.items():
        q = r["mu"][side]
        v = floats(q["value"])[:-1]
        if side in ("G", "S|G"):
            c.fill_between(x, floats(q["band"][0])[:-1], floats(q["band"][1])[:-1], color=col, alpha=0.2, lw=0)
        c.plot(x, v, color=col, ls="-" if side in ("G", "S|G") else "--", lw=1.3, label=side)
    c.set(xscale="log", xlabel=r"$x_{\min}$ (s)", ylabel=r"weighted Hill $\mu$", title="same route")
    c.legend(frameon=False, fontsize=8)
    for ax in (b, c):
        ax.set(ylim=MU_LIMITS, xlim=(XMIN_GRID[0] * 0.95, XMIN_GRID[-1] * 1.05), xticks=[300, 500, 1000, 2000, 4000], xticklabels=["300", "500", "1000", "2000", "4000"])
        ax.minorticks_off()
        ax.axhline(result["transition"]["populations"]["S"]["at xmin of S"]["mu"], color="#bbbbbb", lw=0.6, ls=":")
    for ax in (a, b, c):
        ax.grid(color="#eeeeee", lw=0.5)
    save(fig, path)


if __name__ == "__main__":
    assert "--open" in sys.argv, "run with --open"
    sample = route.option("--sample", 0)
    if "--redraw" in sys.argv:  # the figure again, from the saved JSON
        name = OUT / f"tail{'_sample' + str(sample) if sample else ''}"
        plot(json.loads(name.with_suffix(".json").read_text()), name.with_suffix(".pdf"))
        sys.exit()
    folders()
    flights = with_takeoffs(read_flights())
    summary = pd.read_parquet(LABELS / "flight_summary.parquet")
    cell, pilot, block = twoway.cells(flights, summary)
    counts = twoway.counts(pilot, block, np.random.default_rng(SEED))
    p = dict(np.load(longlags.PIECES))
    frames = None
    if sample:  # the whole flights of consecutive row groups of the fixes, from three quarters of the file on
        frames, wanted = sample_block(flights, sample)
    else:
        wanted = set(flights["flight"])
    every, counted = read_runs(flights)
    pops = population_runs(counted, p, wanted)
    result = {"n_boot": twoway.N_BOOT, "cells": int(len(pilot)), "xmin grid": XMIN_GRID.tolist(), "sample": sample or None}
    result |= raw_tail(pops, cell, counts)
    chosen = json.loads((OUT / "matched_counts.json").read_text())["chosen caliper"]["name"]
    ctx = context(flights, summary, route.TRACKS, every, counted)
    table = pd.read_parquet(CACHE / "comparisons_matches.parquet")
    if sample:
        table = table[np.isin(ctx.tracks["flight"][table["piece"]], list(wanted))]
    line = {"caliper": chosen, "cover": matched.MAIN_COVER, "isolation": False}
    xmins = [*XMIN_GRID, result["transition"]["xmin of S"]]
    result["same route"] = same_route(ctx, table, line, xmins)
    cache = CACHE / "tail_glides.parquet"  # the descriptors of every glide: the pass over the fixes is the slow part
    if not sample and cache.exists() and "--recompute" not in sys.argv:
        gl = pd.read_parquet(cache)
    else:
        gl = pd.concat([pops[pop].loc[pops[pop]["phase"] == "transition", ["flight", "segment_id", "t_start", "t_end", "n_fixes"]]
                        .assign(population=pop) for pop in raw.POPS]).reset_index(drop=True)
        gl = gl.join(read_glides(gl, set(gl["flight"]), frames))
        if not sample:
            gl.to_parquet(cache)
    gl = gl.assign(cell=cell.reindex(gl["flight"]).to_numpy().astype(int))
    result["descriptors"], result["descriptor gaps"] = describe_glides(gl, raw.POPS, counts)
    tag = f"_sample{sample}" if sample else ""
    (OUT / f"tail{tag}.json").write_text(json.dumps(matched.jsonable(result), indent=1))
    plot(result, OUT / f"tail{tag}.pdf")
    for phase in ("transition", *REFERENCES):
        for s, r in result[phase]["populations"].items():
            q = r["at xmin of S"]
            print(phase, s, f"mu at xmin of S {q['mu']:.2f} [{q['band'][0]:.2f}, {q['band'][1]:.2f}] ({q['runs above']} runs)")
    print({s: [round(v, 2) for v in q["value"][-1:]] for s, q in result["same route"]["mu"].items()})
    print(json.dumps(result["descriptor gaps"]["2000 s"]["height lost (m/km)"]))
