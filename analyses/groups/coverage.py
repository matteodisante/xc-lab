"""Do the bands of twoway.py cover 90% of the time? A simulation on the designs of the comparisons.

The designs are those of the data: point 1 (raw.py, dataset `1000`: the flights of G against the solo flights),
point 2 (matched.py, side G, level "all", dataset `1000`: the pieces, their controls and the reuse of the control
flights) and point 4 (prepost.py --open: the events with the four windows). Each unit gets a synthetic value

    y = a_pilot + b_block + c_flight + e_unit,    independent normal effects of the variances of SCENARIOS,

with no group effect, so the estimate (mean of G minus mean of S, the matched Delta, the mean over the events)
has mean 0 and its true standard deviation is that of REPLICATES draws of the effects. For each replicate the
band of twoway.py (DRAWS per family) and the band of the scheme the points used before (OLD) are computed; the
JSON gives, per design and scenario, the mean of the band's sd over the true sd and the share of bands that hold 0.

    uv run python coverage.py --open    # after raw.py, matched.py and prepost.py --open
"""

import json
import sys

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

import decompose
import isolated
import longlags
import matched
import prepost
import route
import tamsd
import twoway
from blocks import with_takeoffs
from tamsd import CACHE, LABELS, OUT, SEED, read_flights

REPLICATES, DRAWS = 400, 400
SCENARIOS = {  # name: variances of the pilot, block, flight and unit effects
    "unit only": (0, 0, 0, 1),
    "flight only": (0, 0, 1, 0),
    "half flight, half unit": (0.05, 0.15, 0.5, 0.5),
    "pilot strong": (1, 0.2, 0.5, 0.3),
    "block strong": (0.1, 1, 0.5, 0.3),
}
OLD = {"raw": "episodes, 5-95%", "matched": "pilots x blocks on both sides, 5-95%", "prepost": "blocks, 5-95%"}


def one_way(codes, rng, size=DRAWS):
    """Counts (size, codes) of one bootstrap of the codes."""
    return twoway.draw(codes.max() + 1, rng, size)[:, codes]


def flight_codes(flights, summary, universe):
    """Per flight of `universe`: its pilot, its block (episodes and site-days), its cell, its episode block and
    its row; and the stacked counts of twoway.py over the cells."""
    pilot, block, _ = isolated.codes(flights, summary, universe)
    cell, of_pilot, of_block = twoway.cell_codes(pilot, block)
    episode = pd.factorize(summary.set_index("flight")["block"].reindex(universe))[0]
    rng = np.random.default_rng(SEED)
    return {"pilot": pilot, "block": block, "cell": cell, "episode": episode,
            "stacked": twoway.counts(of_pilot, of_block, rng, DRAWS), "rng": rng}


def effects(codes, flight_row, n_units, variances, rng):
    """y of each unit (rows) for every replicate (columns)."""
    vp, vb, vf, ve = variances
    draw = lambda v, n: np.sqrt(v) * rng.standard_normal((n, REPLICATES))
    y = draw(vp, codes["pilot"].max() + 1)[codes["pilot"][flight_row]] + draw(vb, codes["block"].max() + 1)[codes["block"][flight_row]]
    return y + draw(vf, len(codes["pilot"]))[flight_row] + draw(ve, n_units)


def weighted_means(y, code, counts):
    """Mean of y (units x replicates) with weight 1 (row 0) and with the counts of each draw (rows of `counts`,
    columns indexed by `code`)."""
    n = counts.shape[1]
    sums, size = np.zeros((n, y.shape[1])), np.bincount(code, minlength=n).astype(float)
    np.add.at(sums, code, y)
    rows = [y.mean(axis=0)[None]]
    for a in range(0, len(counts), 100):
        c = counts[a:a + 100].astype(float)
        rows.append((c @ sums) / (c @ size)[:, None])
    return np.vstack(rows)


def summary_of(estimates):
    """{scheme: (point estimates, sd of each replicate's band, whether it holds 0)} -> the JSON of one scenario."""
    true = float(np.std(next(iter(estimates.values()))[0]))
    return {"true sd": true, **{k: {"sd / true sd": float(np.mean(sd) / true), "coverage of 90%": float(np.mean(hold))}
                                for k, (_, sd, hold) in estimates.items()}}


def bands(draws, value):
    """The sd and the 0-holding of twoway.py (stacked draws) per replicate."""
    sd = np.sqrt(twoway.variance(draws))
    return sd, np.abs(value) <= twoway.Z * sd


def percentile(draws, value):
    """The same for a 5-95% band of one-way draws (the sd reported is that of the draws)."""
    lo, hi = np.nanpercentile(draws, [5, 95], axis=0)
    return np.nanstd(draws, axis=0), (lo <= 0) & (hi >= 0)


def mean_design(name, code_of, units_g, units_s, codes):
    """Points 1 and 4: the mean over the units of G minus that over S (S absent for point 4), with the band of
    twoway.py on the cells and the OLD band of one-way draws of `code_of` (the old block code)."""
    out = {}
    old_counts = one_way(codes[code_of], codes["rng"])
    n_cells = codes["stacked"].shape[1]
    for scen, variances in SCENARIOS.items():
        rng = np.random.default_rng(1)
        rows = np.r_[units_g, units_s]
        y = effects(codes, rows, len(rows), variances, rng)
        new = weighted_means(y[:len(units_g)], codes["cell"][units_g], codes["stacked"])
        old = weighted_means(y[:len(units_g)], codes[code_of][units_g], old_counts)
        if len(units_s):
            new = new - weighted_means(y[len(units_g):], codes["cell"][units_s], codes["stacked"])
            old = old - weighted_means(y[len(units_g):], codes[code_of][units_s], old_counts)
        out[scen] = summary_of({"two-way, normal": (new[0], *bands(new[1:], new[0])),
                                OLD[name]: (old[0], *percentile(old[1:], old[0]))})
        print(name, scen, json.dumps(out[scen]), flush=True)
    return {"units": int(len(units_g) + len(units_s)), "flights": int(len(np.unique(np.r_[units_g, units_s]))),
            "cells": int(n_cells), "scenarios": out}


def matched_engine(p, s, a, b, wp, wf):
    """matched.py's estimator (isolated.engine) for scalar units, every replicate at once: (weight columns, replicates)."""
    controls = csr_matrix((np.ones(len(p)), (p, s)), shape=(wp.shape[0], wf.shape[0]))
    mass = controls @ wf
    ok = (wp > 0) & (mass > 0)
    w = np.where(ok, wp, 0.0)
    share = np.divide(w, mass, out=np.zeros_like(w), where=ok)
    control = np.column_stack([((csr_matrix((b[:, k], (p, s)), shape=controls.shape) @ wf) * share).sum(axis=0)
                               for k in range(b.shape[1])])
    return (w.T @ a - control) / w.sum(axis=0)[:, None]


def matched_design(flights, summary):
    """Point 2 on the match table of the main result."""
    tracks = route.read_tracks(flights, None)
    table = pd.read_parquet(CACHE / "comparisons_matches.parquet")
    chosen = json.loads((OUT / "matched_counts.json").read_text())["chosen caliper"]["name"]
    sel = matched.select(table[table["side"] == "G"], "1000", chosen)
    universe = np.unique(tracks["flight"])
    codes = flight_codes(flights, summary, universe)
    pieces, p = np.unique(sel["piece"].to_numpy(), return_inverse=True)
    controls, s = np.unique(tracks["flight"][sel["solo"].to_numpy()], return_inverse=True)
    rp, rs = np.searchsorted(universe, tracks["flight"][pieces]), np.searchsorted(universe, controls)
    of, n = twoway.weights(codes["cell"], codes["stacked"])
    new_w = of(np.r_[rp, rs], np.arange(n))
    old = codes["rng"]
    pig = np.vstack([np.ones(len(universe)), one_way(codes["pilot"], old).astype(float) * one_way(codes["block"], old)]).T
    out = {}
    for scen, variances in SCENARIOS.items():
        rng = np.random.default_rng(1)
        a = effects(codes, rp, len(rp), variances, rng)
        b = effects(codes, rs[s], len(s), variances, rng)  # one unit per match row: the control segment
        new = matched_engine(p, s, a, b, new_w[:len(rp)], new_w[len(rp):])
        prev = matched_engine(p, s, a, b, pig[rp], pig[rs])
        out[scen] = summary_of({"two-way, normal": (new[0], *bands(new[1:], new[0])),
                                OLD["matched"]: (prev[0], *percentile(prev[1:], prev[0]))})
        print("matched", scen, json.dumps(out[scen]), flush=True)
    reuse = np.bincount(s, weights=1 / np.bincount(p)[p])
    return {"pieces": int(len(pieces)), "matches": int(len(sel)), "control flights": int(len(controls)),
            "sum K^2 / sum K": float((reuse ** 2).sum() / reuse.sum()), "scenarios": out}


def main():
    flights = with_takeoffs(read_flights())
    summary = pd.read_parquet(LABELS / "flight_summary.parquet")
    result = {"replicates": REPLICATES, "draws per family": DRAWS, "scenarios": SCENARIOS}

    p = dict(np.load(longlags.PIECES))
    masks = longlags.populations(p)
    g, s = (np.unique(p["flight"][longlags.long_enough(p, masks[k], tamsd.TAU_MAX)]) for k in ("group", "solo"))
    universe = np.r_[g, s]
    codes = flight_codes(flights, summary, universe)
    result["raw"] = mean_design("raw", "episode", np.arange(len(g)), np.arange(len(g), len(universe)), codes)

    w = tamsd.read_pieces(CACHE / f"prepost_pieces_{prepost.OPEN_WINDOW_S}_lag{prepost.window_lags(prepost.OPEN_WINDOW_S).max()}.npz")["para"]
    stages = pd.DataFrame({k: w[k] for k in ("flight", "population", "episode")}).drop_duplicates()
    stages = stages[stages["population"].isin(prepost.STAGES)]
    events = stages.groupby(["flight", "episode"])["population"].nunique().loc[lambda n: n == len(prepost.STAGES)].reset_index()
    universe = np.unique(events["flight"])
    codes = flight_codes(flights, summary, universe)
    old_block = decompose.flight_table(flights, summary)[1].reindex(universe)
    codes["old block"] = pd.factorize(old_block)[0]
    result["prepost"] = mean_design("prepost", "old block", np.searchsorted(universe, events["flight"]), np.array([], int), codes)

    result["matched"] = matched_design(flights, summary)
    (OUT / "coverage.json").write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
