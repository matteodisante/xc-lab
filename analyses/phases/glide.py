"""Local exponent of the MSD of the glide (transition) runs, out to the longest lags that enough runs reach.

alpha(D) = d log MSD / d log D, for paragliders and hang gliders, on every labelled run and on the runs of
the fixed cohorts C_3600 and C_10000. Same runs and estimator as msd.py; a lag is drawn while at least
msd.MIN_RUNS runs reach it, and the lower panel counts them.

    uv run python glide.py [--recompute]    # from this folder; writes out/glide.json, out/glide.pdf
"""

import json
import sys

import matplotlib.pyplot as plt
import numpy as np

import msd

PHASE = "transition"
COHORT_LAGS = (3600, 10000)
DELTAS = np.unique(np.round(np.geomspace(1, 4000, 140)).astype(int))
POPULATIONS = {"all": "all 1 Hz flights", "C_3600": r"$C_{3600}$", "C_10000": r"$C_{10000}$"}
STYLES = {"all": "-", "C_3600": "--", "C_10000": ":"}
DISCIPLINES = {"para": ("paragliders", "#2a78d6"), "hang": ("hang gliders", "#eb6834")}
N_POINTS = 45


def plot(results, path):
    """One column per discipline: the local exponent of the median MSD with the band between the
    exponents of its 10th and 90th percentile curves, and the runs reaching each lag."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex="col", sharey="row",
                             gridspec_kw={"height_ratios": [3, 1.3]})
    for col, (d, (name, color)) in enumerate(DISCIPLINES.items()):
        top, bottom = axes[:, col]
        for pop, style in STYLES.items():
            r = results[pop][d][PHASE]
            lag, alpha = msd.slopes(r, N_POINTS, DELTAS.max())
            if not len(lag):
                continue
            low = msd.slopes({**r, "msd": r["q10"]}, N_POINTS, DELTAS.max())[1]
            high = msd.slopes({**r, "msd": r["q90"]}, N_POINTS, DELTAS.max())[1]
            label = f"{POPULATIONS[pop]}: {r['flights']:,} flights, {r['n_runs']:,} runs"
            top.plot(lag, alpha, style, color=color, lw=1.6, label=label)
            top.fill_between(lag, np.minimum(low, high), np.maximum(low, high), color=color, alpha=0.12, lw=0)
            runs = np.array(r["runs"], float)
            bottom.plot(r["delta"], np.where(runs >= msd.MIN_RUNS, runs, np.nan), style, color=color, lw=1.2)
        for y in (2, 1):
            top.axhline(y, color="#999999", lw=0.6, ls=":")
        top.set(xscale="log", title=f"Glide runs (transition), {name}")
        top.legend(frameon=False, fontsize=8, loc="lower left")
        bottom.set(xscale="log", yscale="log", xlabel=r"$\Delta$ (s)")
    axes[0, 0].set_ylabel(r"$d\,\log\mathrm{MSD} / d\,\log\Delta$  (band: 10th-90th percentile of runs)")
    axes[1, 0].set_ylabel("runs reaching the lag")
    msd.save(fig, path)


if __name__ == "__main__":
    msd.OUT.mkdir(exist_ok=True)
    saved = msd.OUT / "glide.json"
    if "--recompute" in sys.argv or not saved.exists():
        by_discipline = {d: msd.compute(d, False, DELTAS, (PHASE,), COHORT_LAGS) for d in DISCIPLINES}
        saved.write_text(json.dumps({pop: {d: by_discipline[d][pop] for d in DISCIPLINES} for pop in POPULATIONS}))
    plot(json.loads(saved.read_text()), msd.OUT / "glide.pdf")
