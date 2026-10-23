"""The tolerance profiles of the contacts, to see how far eps_h, eps_z and g separate real
pairs from chance.

On the sample of dates of delays.py, with the delta and the null model of the chosen run, for
the real pairs with takeoffs within r_s and for their surrogates, at every instant t of i in
the window of Phase 1 (whose filter is widened to the largest distance profiled):
- horizontal: the smallest horizontal distance between i at t and j at t + l, |l| <= delta,
  over the l with |dz| < eps_z and headings that pass;
- vertical: the smallest |dz| over the l with horizontal distance < eps_h and headings that
  pass;
- gap: the gaps between consecutive instants of contact of a pair (a_ij = m_ij or m_ji).
Each is counted per second of exposure, chance date by date as in run.calibrate, on the dates
with surrogates; the ratio of real to chance at each value is the profile.

    uv run python tolerances.py [--sample N] [--jobs J]   # from this folder

Writes tolerances/sample<N>_<name>/ on SSD_DISANTE: one cache per date in days/, profile.json.
"""

import argparse
import json
import pickle

import numpy as np

import contacts
import grid
import run

H_BIN, H_BINS = 25, 120  # m: 3 km; the last bin holds the rest
Z_BIN, Z_BINS = 25, 60  # m: 1.5 km
GAP_BINS = 360  # of one grid step: 30 min


def add(histogram, values, width):
    values = values[np.isfinite(values)]
    histogram += np.bincount(np.minimum(values // width, len(histogram) - 1).astype(int), minlength=len(histogram))


def profile_pair(tracks, i, j, instants, p, out):
    """Add to `out` the smallest distances and the gaps of the pair (i, j) over the instants of i."""
    lags = tracks.step * np.arange(-(p["delta"] // tracks.step), p["delta"] // tracks.step + 1)
    cos_max = np.cos(np.radians(p["theta_max"]))
    hits = []
    for start in range(0, len(instants), contacts.CHUNK):
        ti = instants[start:start + contacts.CHUNK]
        xi = tracks.at(i, ti)
        ti, xi = ti[np.isfinite(xi[:, 0])], xi[np.isfinite(xi[:, 0])]
        tj = ti[:, None] + lags[None, :]
        xj = tracks.at(j, tj)
        d = np.linalg.norm(xi[:, None, :3] - xj[..., :3], axis=2)
        dz = np.abs(xi[:, None, 3] - xj[..., 3])
        glide = (xi[:, None, 6] == contacts.GLIDE) & (xj[..., 6] == contacts.GLIDE)
        heading = ~glide | (xi[:, None, 4] * xj[..., 4] + xi[:, None, 5] * xj[..., 5] > cos_max)
        add(out["horizontal"], np.where((dz < p["eps_z"]) & heading, d, np.inf).min(axis=1), H_BIN)
        add(out["vertical"], np.where((d < p["eps_h"]) & heading, dz, np.inf).min(axis=1), Z_BIN)
        ok = (d < p["eps_h"]) & (dz < p["eps_z"]) & heading
        hits += [ti[ok.any(axis=1)], tj[ok]]
    a = np.unique(np.concatenate(hits))
    gaps = np.diff(a) - tracks.step
    add(out["gap"], gaps[gaps > 0].astype(float), tracks.step)


def one_date(date, flights, runs, p):
    """The profiles of the real pairs and of the surrogates, with their exposures."""
    out = {"date": date, "exposure_real": 0.0, "exposure_null": 0.0}
    for kind in ("real", "null"):
        out[kind] = {"horizontal": np.zeros(H_BINS), "vertical": np.zeros(Z_BINS), "gap": np.zeros(GAP_BINS)}
    table, g = grid.day(flights, runs, p["step"], p["h_max"], p["r_dep"])
    if g is None:
        return out
    tracks = contacts.Tracks(g, p["step"])
    wide = dict(p, eps_h=float(H_BIN * H_BINS))  # the filter of Phase 1 misses no distance profiled
    a, b = contacts.near_pairs(table, p["r_s"])
    out["exposure_real"] = float(contacts.exposure(table, a, b).sum())
    near = set(zip(a.tolist(), b.tolist()))
    found = contacts.candidates(g, p["delta"], wide["eps_h"], p["v_max"])
    for (i, j), c in found.groupby(["i", "j"], sort=False):
        if (i, j) in near:
            profile_pair(tracks, i, j, contacts.window(c["tc"].to_numpy(), p["step"]), p, out["real"])
    rng = np.random.default_rng(int(date.replace("-", "")))  # the surrogates of run.one_date
    for i, j, r in contacts.surrogates(table, rng, p, p["step"]):
        copy, instants, (lo, hi) = contacts.surrogate(table, g, tracks, i, j, r, wide)
        out["exposure_null"] += max(hi - lo, 0.0)
        if instants is not None:
            profile_pair(tracks, copy, i, instants, p, out["null"])
    return out


def profile(results, step):
    """Counts per second of real exposure in each bin, real and chance, on the dates with
    surrogates."""
    results = [r for r in results if r["exposure_null"] > 0]
    exposure = sum(r["exposure_real"] for r in results)
    out = {"exposure_h": exposure / 3600, "dates": len(results)}
    for key, width in (("horizontal", H_BIN), ("vertical", Z_BIN), ("gap", step)):
        real = sum(r["real"][key] for r in results) / exposure
        chance = sum(r["null"][key] * r["exposure_real"] / r["exposure_null"] for r in results) / exposure
        out[key] = {"bin": width, "real": real.tolist(), "chance": chance.tolist()}
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=300)
    parser.add_argument("--jobs", type=int, default=5)
    args = parser.parse_args()
    p = dict(run.STARTING, delta=120)  # the chosen run
    p["r_min"] = 2 * p["delta"] + p["step"]
    folder = run.DATA / "tolerances" / f"sample{args.sample}_{run.run_name(p)}"
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
