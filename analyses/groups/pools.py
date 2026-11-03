"""Two checks of the raw comparison against its fixed population: a pool that shrinks with the lag, and a
cohort of short pieces.

The pieces of every length from MIN_LAG / SUPPORT s are measured once, with sums and origins at
longlags.LONG_LAGS (one pass over the fixes, cached in CACHE). Variable pool: at each lag the pieces of at least
lag / SUPPORT s, so the flights fall as the lag grows and H_loc is taken along a curve whose population changes.
Short pieces: the pieces lasting from SHORT_MIN to SHORT_MAX s, the same at every lag up to SUPPORT x SHORT_MIN.

    uv run python pools.py --open [--sample N] [--recompute] [--curves]    # after longlags.py; writes CACHE/all_pieces.npz and, with --curves, OUT/pools.json
"""

import json
import sys

import numpy as np
import pandas as pd

import longlags
import raw
import tamsd
import twoway
from blocks import with_takeoffs
from tamsd import CACHE, LABELS, OUT, SEED, SUPPORT, ensemble, local_h, stretches

MIN_LAG = 10  # s: the first lag; a piece enters from MIN_LAG / SUPPORT s
SHORT_MIN, SHORT_MAX = 1000, 1250  # s: the cohort of short pieces, lags up to SUPPORT x SHORT_MIN = 800 s
LONGEST = 11450  # s: the longest lag of the variable pool, that of raw.py's longest dataset
SAMPLE = int(sys.argv[sys.argv.index("--sample") + 1]) if "--sample" in sys.argv else None
PIECES = CACHE / (f"all_pieces_sample{SAMPLE}.npz" if SAMPLE else "all_pieces.npz")


def measure(flights, membership, summary, sample=None):
    """The pieces of the pooled populations of at least MIN_LAG / SUPPORT s, with sums and origins at LONG_LAGS;
    with `sample`, only the first `sample` flights that have a stretch."""
    spans = stretches(flights, membership, summary)
    if sample:
        spans = spans[spans["flight"].isin(spans["flight"].drop_duplicates().iloc[:sample])]
    found = tamsd.measure("para", flights, spans, longlags.LONG_LAGS, shortest=MIN_LAG / SUPPORT)
    np.savez(PIECES, **found)
    return found


def pool_curves(p, mask, block, rng, counts, keep_at, lags):
    """H_loc of the pieces of `mask` selected for each lag by `keep_at(lag)` (a mask over the pieces), and the
    number of flights at each lag: the point estimate (row 0) and the bootstrap draws (the other rows), as
    longlags.curve takes them, but with a population that may change along the lags."""
    rows, flights = [], []
    for j, lag in enumerate(lags):
        keep = mask & keep_at(lag)
        ids, k = np.unique(p["flight"][keep], return_inverse=True)
        sums, origins = np.zeros(len(ids)), np.zeros(len(ids))
        np.add.at(sums, k, p["sums"][keep][:, j])
        np.add.at(origins, k, p["origins"][keep][:, j])
        mean, draws = ensemble((sums / origins)[:, None], block.reindex(ids).to_numpy(), rng, counts)
        rows.append(np.vstack([mean, draws])[:, 0])
        flights.append(len(ids))
    msd = np.array(rows).T  # row 0: estimate; the others: draws
    return np.array([local_h(lags, r) for r in msd]), flights


def curves_of(p, block, rng, counts, lags, keep_at):
    """For each population of raw.POPS, the H_loc curve and flights per lag, and the gaps of raw.GAPS, with bands."""
    masks = longlags.populations(p)
    got = {pop: pool_curves(p, masks[pop], block, rng, counts, keep_at, lags) for pop in raw.POPS}
    out = {"lags": lags.tolist(), "populations": {}, "gaps": {}}
    for pop, short in raw.POPS.items():
        q = twoway.quote(got[pop][0])
        out["populations"][short] = {"H_loc": q["value"], "H_loc_band": q["band"], "flights": got[pop][1]}
    for gap, (a, b) in raw.GAPS.items():
        out["gaps"][gap] = {"lags": lags.tolist(), **twoway.quote(got[a][0] - got[b][0])}
    return out


if __name__ == "__main__":
    tamsd.folders()
    if "--recompute" in sys.argv or not PIECES.exists():
        p = measure(tamsd.read_flights(), pd.read_parquet(LABELS / "membership.parquet"),
                    pd.read_parquet(LABELS / "flight_summary.parquet"), SAMPLE)
    else:
        p = dict(np.load(PIECES))
    print(f"{len(p['flight']):,} pieces from {len(np.unique(p['flight'])):,} flights; "
          f"durations {np.min(p['duration']):.0f}-{np.max(p['duration']):.0f} s")
    if "--curves" in sys.argv:
        block, of_pilot, of_block = twoway.cells(with_takeoffs(tamsd.read_flights()),
                                                 pd.read_parquet(LABELS / "flight_summary.parquet"))
        rng = np.random.default_rng(SEED)
        counts = twoway.counts(of_pilot, of_block, rng)
        lags_all = longlags.LONG_LAGS[longlags.LONG_LAGS <= LONGEST]
        lags_short = longlags.LONG_LAGS[longlags.LONG_LAGS <= SUPPORT * SHORT_MIN]
        result = {"variable pool": curves_of(p, block, rng, counts, lags_all,
                                             lambda lag: p["duration"] >= lag / SUPPORT),
                  "short pieces": curves_of(p, block, rng, counts, lags_short,
                                            lambda lag: (p["duration"] >= SHORT_MIN) & (p["duration"] < SHORT_MAX))}
        result["short pieces"]["window"] = [SHORT_MIN, SHORT_MAX]
        (OUT / "pools.json").write_text(json.dumps(result, indent=1))
        for name, r in result.items():
            print(name, {k: (v["flights"][0], v["flights"][-1]) for k, v in r["populations"].items()})
