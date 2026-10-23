"""Survival of the phase durations and their tail exponent mu, as in Vilpellet's paper.

Input: the phase runs that soaring (tag legacy-final) wrote on SSD_DISANTE by applying
Vilpellet's segmenter to the cleaned fixes. A run's duration is its number of fixes,
which at 1 Hz is seconds, as in his code. mu is the exponent of the survival,
P(tau > t) ~ t^-mu for t >= xmin, from powerlaw.Fit (Clauset et al.), mu = alpha - 1.
Each figure shows two populations: every labelled run, and the runs of the fixed cohort
C_10000 (see cohort_segments).

    uv run python durations.py [--refit]    # from this folder; writes out/
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import powerlaw

DISK = Path("/Volumes/SSD_DISANTE")
RUNS = {
    "para": DISK / "paragliders/ffvl_cfd_igc/derived/segmentation/vilpellet/phase_segments.parquet",
    "hang": DISK / "hang_gliders/delta_cfd_igc/derived/segmentation/vilpellet/phase_segments.parquet",
}
COHORT = {d: DISK / f"derived-audit/chapter3-fixed-20260917/{d}/cohort-10000.json" for d in RUNS}
POPULATIONS = {"all": "all 1 Hz flights", "C_10000": r"$C_{10000}$ (1 Hz)"}
PHASES = ("transition", "search", "climb")
OUT = Path(__file__).with_name("out")
STYLE = {"para": ("#eb6834", "o", "paragliders"), "hang": ("#2a78d6", "*", "hang gliders")}
XMIN_GRID = np.geomspace(50, 3000, 40)


def cohort_segments(discipline):
    """(flight, segment) pairs of C_10000: the retained segments of at least 12500 s that
    admit a flight to the cohort (chapter 3, fixed population)."""
    members = json.loads(COHORT[discipline].read_text())["members"]
    return {(m["flight_id"], s["segment_id"]) for m in members for s in m["segments"]}


def in_segments(runs, segments):
    pairs = pd.MultiIndex.from_arrays([runs["flight_id"].astype(str), runs["segment_id"].astype(int)])
    return pairs.isin(list(segments))


def waiting_times(runs, segments=None):
    """Durations per phase. A search counts only when the next run of the flight is a climb
    (his "ARS followed by commute is not research"); climbs and transitions all count.
    With segments, only the runs inside them, after that rule has read the whole flight."""
    runs = runs.sort_values(["flight_id", "segment_id", "t_start"], kind="stable")
    following = runs.groupby("flight_id", sort=False)["phase"].shift(-1)
    kept = runs[(runs["phase"] != "search") | (following == "climb")]
    if segments is not None:
        kept = kept[in_segments(kept, segments)]
    return {p: kept.loc[kept["phase"] == p, "n_fixes"].to_numpy(float) for p in PHASES}


def trim(x):
    """His cut before every fit: strictly inside the 0.0001 and 99.9999 percentiles."""
    lo, hi = np.percentile(x, [0.0001, 99.9999])
    return x[(x > lo) & (x < hi)]


def fit(x):
    f = powerlaw.Fit(x, verbose=0)
    return {"mu": f.power_law.alpha - 1, "xmin": float(f.xmin), "n_tail": int((x >= f.xmin).sum()),
            "n": len(x)}


def hill(x, xmin):
    """Continuous maximum-likelihood mu above xmin: the estimate powerlaw.Fit makes once xmin is set."""
    tail = x[x >= xmin]
    return len(tail) / np.log(tail / xmin).sum()


def ccdf(x):
    d = np.sort(x)
    return d, 1 - np.arange(1, len(d) + 1) / len(d)


def log_resample(x, y, n=30):
    """His plotted points: n log-spaced abscissas, log y interpolated linearly in log x."""
    m = y > 0
    lx, ly = np.log10(x[m]), np.log10(y[m])
    grid = np.linspace(lx.min(), lx.max(), n)
    return 10**grid, 10 ** np.interp(grid, lx, ly)


def load():
    """Trimmed durations by population, discipline and phase."""
    data = {pop: {} for pop in POPULATIONS}
    for d, path in RUNS.items():
        runs = pd.read_parquet(path)
        for pop in POPULATIONS:
            times = waiting_times(runs, cohort_segments(d) if pop == "C_10000" else None)
            data[pop][d] = {p: trim(times[p]) for p in PHASES}
    return data


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=200, bbox_inches="tight")


def plot_survival(data, fits, path):
    fig, axes = plt.subplots(2, 3, figsize=(10, 6.4), sharex="col", sharey="col")
    for row, (pop, label) in zip(axes, POPULATIONS.items()):
        for ax, p in zip(row, PHASES):
            for d in RUNS:
                color, marker, _ = STYLE[d]
                f, x = fits[pop][d][p], data[pop][d][p]
                ax.scatter(*log_resample(*ccdf(x)), s=14, color=color, marker=marker,
                           label=rf"$\mu = {f['mu']:.2f}$, $x_{{\min}} = {f['xmin']:.0f}$ s")
                # The fitted law from xmin, anchored on the empirical survival there.
                t = np.geomspace(f["xmin"], x.max(), 50)
                ax.plot(t, (x >= f["xmin"]).mean() * (t / f["xmin"]) ** -f["mu"], color=color, lw=1, ls="--")
                ax.axvline(f["xmin"], color=color, lw=0.8, ls=":")
            ax.set(xscale="log", yscale="log", title=f"{p}, {label}",
                   ylabel=rf"$P(\tau_{{\mathrm{{{p}}}}} > \tau)$")
            ax.legend(loc="lower left", framealpha=1, edgecolor="none", fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel(r"$\tau$ (s)")
    handles = [plt.Line2D([], [], color=c, marker=m, ls="", label=n) for c, m, n in STYLE.values()]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.03))
    save(fig, path)


def plot_hill(data, fits, path):
    """mu above a moving cutoff: flat for a power law, rising for a tail that bends down."""
    fig, axes = plt.subplots(2, 3, figsize=(10, 6.4), sharex="col", sharey=True)
    for row, (pop, label) in zip(axes, POPULATIONS.items()):
        for ax, p in zip(row, PHASES):
            for d in RUNS:
                color, marker, name = STYLE[d]
                x = data[pop][d][p]
                grid = XMIN_GRID[[(x >= g).sum() >= 100 for g in XMIN_GRID]]
                ax.plot(grid, [hill(x, g) for g in grid], color=color, marker=marker, ms=4, lw=1, label=name)
                xmin = fits[pop][d][p]["xmin"]
                ax.axvline(xmin, color=color, lw=0.8, ls=":")
                ax.text(xmin, 0.97 if d == "para" else 0.88, f"{xmin:.0f} s ", color=color, fontsize=8,
                        transform=ax.get_xaxis_transform(), va="top", ha="right")
            ax.set(xscale="log", title=f"{p}, {label}")
        row[0].set_ylabel(r"$\mu$ above $\tau_{\min}$")
    for ax in axes[-1]:
        ax.set_xlabel(r"$\tau_{\min}$ (s)")
    axes[0, 0].legend(frameon=False, loc="center left")
    save(fig, path)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    data = load()
    saved = OUT / "fits.json"
    fits = json.loads(saved.read_text()) if saved.exists() else {}
    # The paraglider fits take minutes: fit only the populations not saved yet, unless asked.
    for pop in POPULATIONS:
        if "--refit" in sys.argv or pop not in fits:
            fits[pop] = {d: {p: fit(data[pop][d][p]) for p in PHASES} for d in RUNS}
            saved.write_text(json.dumps(fits, indent=2))
    plot_survival(data, fits, OUT / "survival.pdf")
    plot_hill(data, fits, OUT / "hill.pdf")
    print(json.dumps(fits, indent=2))
