"""The delay profile of the contacts, to choose delta.

On a sample of dates and with a wide delta, for the real pairs with takeoffs within r_s and
for their surrogates (the null model of run.py): the correspondences (t, t + l) per second of
exposure at each delay l, and the distance of the two pilots at the same instant t when one
is on the path of the other. Chance comes date by date, as in run.calibrate. The ratio of
the real pairs to chance at delay |l| measures how much of what the contact sees there is
flying together; the profile reports the delays at which it falls below RATIOS. The real
excess also has a slow tail (synchronised use of the same thermals), which the null model
removes with its shifts of more than 2 delta, so a share of the excess would pick a delta far
too long.

    uv run python delays.py [--sample N] [--delta S] [--jobs J]   # from this folder

Writes delays/sample<N>_<name>/ on SSD_DISANTE: one cache per date in days/, profile.json
with the curves, profile.pdf.
"""

import argparse
import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import contacts
import grid
import run

RATIOS = (4, 3, 2)
APART_BINS = 120  # of contacts.APART_BIN m: 30 km, the last bin holds the rest


def one_date(date, flights, runs, p):
    """Correspondences per delay and distances at the same instant, real and surrogate."""
    lags = 2 * (p["delta"] // p["step"]) + 1
    out = {"date": date, "exposure_real": 0.0, "exposure_null": 0.0}
    for kind in ("real", "null"):
        out[kind] = {"lags": np.zeros(lags), "apart": np.zeros(APART_BINS)}
    table, g = grid.day(flights, runs, p["step"], p["h_max"], p["r_dep"])
    if g is None:
        return out
    tracks = contacts.Tracks(g, p["step"])
    cos_max = np.cos(np.radians(p["theta_max"]))
    a, b = contacts.near_pairs(table, p["r_s"])
    out["exposure_real"] = float(contacts.exposure(table, a, b).sum())
    near = set(zip(a.tolist(), b.tolist()))
    found = contacts.candidates(g, p["delta"], p["eps_h"], p["v_max"])
    for (i, j), c in found.groupby(["i", "j"], sort=False):
        if (i, j) in near:
            contacts.scan(tracks, i, j, contacts.window(c["tc"].to_numpy(), p["step"]), p["delta"], p["eps_h"],
                          p["eps_z"], cos_max, out["real"])
    rng = np.random.default_rng(int(date.replace("-", "")))
    for i, j, r in contacts.surrogates(table, rng, p, p["step"]):
        copy, instants, (lo, hi) = contacts.surrogate(table, g, tracks, i, j, r, p)
        out["exposure_null"] += max(hi - lo, 0.0)
        if instants is not None:
            contacts.scan(tracks, copy, i, instants, p["delta"], p["eps_h"], p["eps_z"], cos_max, out["null"])
    return out


def profile(results, p):
    """Rates per second of exposure at each |l| (multiple of the step; the mean of +l and
    -l), real and chance, their ratio, and the first |l| at which the ratio falls below each
    of RATIOS."""
    lags = p["step"] * np.arange(-(p["delta"] // p["step"]), p["delta"] // p["step"] + 1)
    real, chance = np.zeros(len(lags)), np.zeros(len(lags))
    apart = {"real": np.zeros(APART_BINS), "null": np.zeros(APART_BINS)}
    exposure = 0.0
    for r in results:
        exposure += r["exposure_real"]
        real += r["real"]["lags"]
        apart["real"] += r["real"]["apart"]
        if r["exposure_null"] > 0:
            chance += r["null"]["lags"] * r["exposure_real"] / r["exposure_null"]
            apart["null"] += r["null"]["apart"] * r["exposure_real"] / r["exposure_null"]
    delays = np.unique(np.abs(lags))
    fold = lambda x: np.array([x[np.abs(lags) == d].mean() for d in delays]) / exposure  # noqa: E731
    real, chance = fold(real), fold(chance)
    ratio = real / chance
    below = {str(x): int(delays[np.argmax(ratio < x)]) if (ratio < x).any() else None for x in RATIOS}
    return {"delay": delays.tolist(), "real": real.tolist(), "chance": chance.tolist(), "ratio": ratio.tolist(),
            "below": below, "exposure_h": exposure / 3600, "apart_bin_m": contacts.APART_BIN,
            "apart_real": apart["real"].tolist(), "apart_chance": apart["null"].tolist()}


def plot(prof, path, size=(13, 3.8)):
    d = np.array(prof["delay"]) / 60
    real, chance = np.array(prof["real"]), np.array(prof["chance"])
    fig, (left, middle, right) = plt.subplots(1, 3, figsize=size)
    left.plot(d, real, color="#2a78d6", label="real pairs")
    left.plot(d, chance, color="#eb6834", ls="--", label="by chance")
    left.set(yscale="log", xlabel="|delay| (min)", ylabel="correspondences per s of exposure")
    left.legend(frameon=False)
    middle.plot(d, np.array(prof["ratio"]), color="#0b0b0b")
    for x in RATIOS:
        middle.axhline(x, color="#52514e", ls=":", lw=1)
    middle.axhline(1, color="#52514e", lw=0.8)
    middle.set(xlabel="|delay| (min)", ylabel="real / chance", ylim=(0, None))
    km = (np.arange(APART_BINS) + 0.5) * prof["apart_bin_m"] / 1000
    for key, color, label in (("apart_real", "#2a78d6", "real pairs"), ("apart_chance", "#eb6834", "by chance")):
        x = np.array(prof[key])
        right.plot(km[:-1], x[:-1] / x.sum(), color=color, label=label, ls="-" if key == "apart_real" else "--")
    right.set(xlabel="distance at the same instant (km)", ylabel="share of instants in contact")
    right.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=200, bbox_inches="tight")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=300)
    parser.add_argument("--delta", type=int, default=900)
    parser.add_argument("--jobs", type=int, default=5)
    args = parser.parse_args()
    p = dict(run.STARTING, delta=args.delta)
    p["r_min"] = 2 * p["delta"] + p["step"]
    folder = run.DATA / "delays" / f"sample{args.sample}_{run.run_name(p)}"
    days = folder / "days"
    days.mkdir(parents=True, exist_ok=True)
    archive = grid.archive()
    every = sorted(archive["date"].unique())
    dates = sorted(np.random.default_rng(0).choice(every, args.sample, replace=False))
    run.describe(folder, p, len(dates))
    run.compute(archive, dates, days, args.jobs, job=one_date, params=p)
    prof = profile([pickle.loads((days / f"{d}.pkl").read_bytes()) for d in dates], p)
    (folder / "profile.json").write_text(json.dumps(prof, indent=1))
    plot(prof, folder / "profile.pdf")
    print("real / chance falls below", ", ".join(f"{x} at {s} s" for x, s in prof["below"].items()))
