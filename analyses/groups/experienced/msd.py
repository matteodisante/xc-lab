"""TA-MSD and H_loc of experienced and beginner paragliders, in a group and with no contact.

Left: the pieces of population.py at each level; MSD, H, H_loc and bands from tamsd.summarize
(equal weight per flight, bootstrap of the dependency blocks). Right: the pieces of group time that
have a match, against the same route flown alone by flights of the same level (route_level.py).

    uv run python msd.py [--run NAME] [--sample N]    # from this folder; writes out/<run>/msd.*
"""

import json
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from population import CACHE, COLORS, LEVELS, LABELS, OUT, POPULATIONS, SEED, keep, load, read_pieces, save
import route  # noqa: E402  (population puts group-identification on the path)
from route_level import results as route_results  # noqa: E402
from tamsd import summarize  # noqa: E402

STYLES = {"experienced": ("-", "o", None), "beginners": ("--", "o", "white")}  # line, marker, marker face
ROUTE = {"group": "group time, with a match", "solo": "same route, no contact"}


def plot(columns, path):
    """Rows: MSD against the lag, H_loc. Columns: (title, labels, results by level). Colour by
    population, style by level."""
    fig, axes = plt.subplots(2, len(columns), figsize=(5.4 * len(columns), 7), sharex=True, sharey="row", squeeze=False)
    for col, (title, labels, results) in enumerate(columns):
        for level, populations in results.items():
            line, marker, face = STYLES[level]
            for pop, r in populations.items():
                lags = np.array(r["lags"])
                lo, hi = r["H_band"]
                for row, y, band in ((0, r["msd"], r["msd_band"]), (1, r["H_loc"], r["H_loc_band"])):
                    ax = axes[row, col]
                    ax.fill_between(lags, *np.array(band), color=COLORS[pop], alpha=0.25, lw=0)
                    ax.plot(lags, y, color=COLORS[pop], ls=line, marker=marker, mfc=face or COLORS[pop], ms=3.5, lw=1,
                            label=f"{level}, {labels[pop]} ({r['flights']}): H = {r['H']:.3f} [{lo:.3f}, {hi:.3f}]"
                            if row == 0 else None)
        axes[0, col].set(xscale="log", yscale="log", title=title)
        axes[0, col].legend(frameon=False, fontsize=6, loc="upper left")
        for h in (0.5, 1):
            axes[1, col].axhline(h, color="#999999", lw=0.6, ls=":")
        axes[1, col].set(xscale="log", xlabel=r"$\tau$ (s)")
    axes[0, 0].set_ylabel(r"MSD (m$^2$)")
    axes[1, 0].set_ylabel(r"$H_{\rm loc}(\tau)$")
    save(fig, path)


if __name__ == "__main__":
    sample = int(sys.argv[sys.argv.index("--sample") + 1]) if "--sample" in sys.argv else None
    summary, classes, alone_flights = load()
    p = read_pieces(CACHE / "pieces.npz")["para"]
    results = {}
    for level in LEVELS:
        mask = keep(p["flight"], p["population"], classes, alone_flights, level)
        results[level] = summarize({k: v[mask] for k, v in p.items()}, summary, np.random.default_rng(SEED), POPULATIONS)
    tracks = route.read_tracks(pd.read_parquet(LABELS / "flights.parquet"), None)
    same_route = route_results(tracks, classes, sample)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "msd.json").write_text(json.dumps({"level": results, "same route": same_route}, indent=1))
    plot([("Paragliders by wing level", POPULATIONS, results), ("Same route, by wing level", ROUTE, same_route)],
         OUT / "msd.pdf")
    for name, group in (("level", results), ("same route", same_route)):
        for level, populations in group.items():
            for pop, r in populations.items():
                print(name, level, pop, r["flights"], f"H = {r['H']:.3f}", f"H_loc(100 s) = {r['H_loc'][r['lags'].index(100)]:.3f}",
                      f"H_loc(1000 s) = {r['H_loc'][-1]:.3f}")
