"""The separation profile of the pairs: how far apart two pilots are at the same instant.

On a sample of dates, for the real pairs with takeoffs within r_s and for the surrogates of
the null model of run.py: the distance between the two pilots at every grid instant when both
fly after departure, as a share of the exposure in bins of BIN m. Chance comes date by date,
as in run.calibrate, on the dates with surrogates. g(d) is the ratio of the real pairs to
chance at distance d.

    uv run python separation.py [--sample N] [--jobs J]   # from this folder

Writes separation/sample<N>_<name>/ on SSD_DISANTE: one cache per date in days/, profile.json.
"""

import argparse
import json
import pickle

import numpy as np

import contacts
import grid
import run

BIN = 100  # m
BINS = 300  # 30 km; the last bin holds the rest


def histogram(tracks, a, b, lo, hi):
    """Distances at the same instant between a and b (flights or copies) at the grid instants of
    [lo, hi), counted in bins of BIN m."""
    t = np.arange(np.ceil(lo / tracks.step) * tracks.step, hi, tracks.step).astype(np.int64)
    d = np.linalg.norm(tracks.at(a, t)[:, :3] - tracks.at(b, t)[:, :3], axis=1)
    d = d[np.isfinite(d)]
    return np.bincount(np.minimum(d // BIN, BINS - 1).astype(int), minlength=BINS)


def one_date(date, flights, runs, p):
    """Distances at the same instant, real and surrogate, with the exposures and the shifts."""
    out = {"date": date, "exposure_real": 0.0, "exposure_null": 0.0, "real": np.zeros(BINS), "null": np.zeros(BINS),
           "shifts": []}
    table, g = grid.day(flights, runs, p["step"], p["h_max"], p["r_dep"])
    if g is None:
        return out
    tracks = contacts.Tracks(g, p["step"])
    dep, land = table.departure.to_numpy(), table.landing.to_numpy()
    a, b = contacts.near_pairs(table, p["r_s"])
    out["exposure_real"] = float(contacts.exposure(table, a, b).sum())
    for i, j in zip(a, b):
        lo, hi = max(dep[i], dep[j]), min(land[i], land[j])
        if hi > lo:
            out["real"] += histogram(tracks, i, j, lo, hi)
    rng = np.random.default_rng(int(date.replace("-", "")))  # the surrogates of run.one_date
    for i, j, r in contacts.surrogates(table, rng, p, p["step"]):
        lo, hi = max(dep[i], dep[j] - r), min(land[i], land[j] - r)
        out["exposure_null"] += max(hi - lo, 0.0)
        out["shifts"].append(r)
        if hi > lo:
            tracks.shifted(j, r, ("shift", j))
            out["null"] += histogram(tracks, i, ("shift", j), lo, hi)
    return out


def profile(results, step):
    """Share of the exposure in each bin of distance, real and chance, their ratio g, and the
    shifts of the surrogates, on the dates with surrogates."""
    results = [r for r in results if r["exposure_null"] > 0]
    exposure = sum(r["exposure_real"] for r in results)
    real = sum(r["real"] for r in results) * step / exposure
    chance = sum(r["null"] * r["exposure_real"] / r["exposure_null"] for r in results) * step / exposure
    shifts = np.abs(np.concatenate([r["shifts"] for r in results]))
    return {"distance_m": ((np.arange(BINS) + 0.5) * BIN).tolist(), "bin_m": BIN, "real": real.tolist(),
            "chance": chance.tolist(), "ratio": (real / chance).tolist(), "exposure_h": exposure / 3600,
            "dates": len(results), "shift_percentiles_5_50_95_s": np.percentile(shifts, [5, 50, 95]).tolist()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=300)
    parser.add_argument("--jobs", type=int, default=5)
    args = parser.parse_args()
    p = dict(run.STARTING, delta=120)  # the null model of the chosen run
    p["r_min"] = 2 * p["delta"] + p["step"]
    folder = run.DATA / "separation" / f"sample{args.sample}_{run.run_name(p)}"
    days = folder / "days"
    days.mkdir(parents=True, exist_ok=True)
    archive = grid.archive()
    every = sorted(archive["date"].unique())
    dates = sorted(np.random.default_rng(0).choice(every, args.sample, replace=False))  # those of delays.py
    run.describe(folder, p, len(dates))
    run.compute(archive, dates, days, args.jobs, job=one_date, params=p)
    prof = profile([pickle.loads((days / f"{d}.pkl").read_bytes()) for d in dates], p["step"])
    (folder / "profile.json").write_text(json.dumps(prof, indent=1))
    print(f"{prof['dates']} dates, {prof['exposure_h']:.0f} h of real exposure")
