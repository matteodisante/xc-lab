"""Pieces of group time against the same route flown alone, by wing level.

Matches, strata and weights are those of ../route.py (the piece is the stratum,
the solo segments are reweighted to one piece per stratum), with both sides restricted to the
flights of one level. The result has the layout of tamsd.summarize, with the matched pieces as
"group" and their solo segments as "solo"; "flights" counts pieces and segments.
"""

import numpy as np
import pandas as pd

import decompose
import route
import zones
from blocks import with_takeoffs
from population import CACHE, LABELS, LEVELS, SEED
from tamsd import LAGS


def level_tracks(tracks, classes, level):
    """The pieces of `tracks` that belong to flights of `level` (`pos` stays whole)."""
    mask = np.isin(tracks["flight"], classes.index[classes.isin(LEVELS[level])])
    return {k: v if k == "pos" else v[mask] for k, v in tracks.items()}


def summarize(u, counts):
    """MSD, H and H_loc of the matched pieces and of their solo segments, with 5-95% bands."""
    mp, mq, _, _ = zones.standardized(u, "group", "route", counts)
    out = {}
    for pop, m, n in (("group", mp, len(np.unique(u.stratum))), ("solo", mq, int((u.population == "route").sum()))):
        h_loc, h = zones.curve(m)
        out[pop] = {"flights": n, "lags": LAGS.tolist(), "msd": m[0].tolist(),
                    "msd_band": np.percentile(m[1:], [5, 95], axis=0).tolist(), "H": float(h[0]),
                    "H_band": np.percentile(h[1:], [5, 95]).tolist(), "H_loc": h_loc[0].tolist(),
                    "H_loc_band": np.nanpercentile(h_loc[1:], [5, 95], axis=0).tolist()}
    return out


def results(tracks, classes, sample=None):
    """Per level: the pieces with matches against the solo segments of the same level."""
    flights = with_takeoffs(pd.read_parquet(LABELS / "flights.parquet"))
    summary = pd.read_parquet(LABELS / "flight_summary.parquet")
    _, block, n_blocks = decompose.flight_table(flights, summary)
    counts = decompose.block_counts(n_blocks, np.random.default_rng(SEED))
    out = {}
    for level in LEVELS:
        sub = level_tracks(tracks, classes, level)
        cached = CACHE / f"route_matches_{level}{'' if sample is None else f'_sample{sample}'}.parquet"
        if not cached.exists():
            route.matches(sub, flights, sample).to_parquet(cached)
        found = pd.read_parquet(cached)
        u, _ = route.units(sub, found, flights, block)
        out[level] = summarize(u, counts)
    return out
