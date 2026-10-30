"""Solo, group time and the same flights alone over lags longer than 1000 s.

A piece of at least TAU_MAX / SUPPORT s gives the lags up to 1000 s of tamsd.py. A lag tau longer than
that needs pieces of at least tau / SUPPORT, so a curve to the lag T is measured on the pieces of its
population of at least T / SUPPORT s: the population does not change along the curve. The curves of one
panel share T, so they are measured alike: the lag of LONG_LAGS nearest to FIXED_LAG ("fixed"), and the
longest lag at which every population of the panel keeps MIN_FLIGHTS flights ("longest"). Equal weight per
flight and the bootstrap bands of tamsd.py.

Populations (paragliders, as tamsd.py and size.py): solo, group, group_solo; the pieces of the group time by
the number of pilots of the group (n2, ..., n7+, and "<size>/alone", the solo time of the flights that have
that size) are measured too.

    uv run python longlags.py [--run NAME] [--recompute]    # after tamsd.py; writes out/<run>/longlags.json and .pdf
    uv run python longlags.py --redraw                       # the figure again, from the saved JSON
"""

import json
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import curves
import tamsd
from tamsd import CACHE, COLORS, LABELS, OUT, SEED, SIZES, SUPPORT, ensemble, local_h, stretches

# The lags of tamsd.py, then the same geometric step up to 30 000 s.
STEP = tamsd.TAU_MAX ** (1 / 34)
LONG_LAGS = np.unique(np.r_[tamsd.LAGS, np.round(tamsd.TAU_MAX * STEP ** np.arange(1, 17) / 10).astype(int) * 10])
MIN_FLIGHTS = 100  # flights every population of a panel keeps at its longest lag
FIXED_LAG = 4000  # s: the nominal longest lag of the panel "fixed"
POOLED = ("solo", "group", "group_solo")
PIECES = CACHE / "long_pieces.npz"


def measure(flights, membership, summary):
    """The pieces of paragliders of at least TAU_MAX / SUPPORT s, pooled and by size, with sums and
    origins at LONG_LAGS; one pass over the fixes."""
    pooled = stretches(flights, membership, summary)
    sized = stretches(flights, membership, summary, by_size=True)
    spans = pd.concat([pooled, sized[sized["population"].isin(SIZES)]], ignore_index=True)
    found = tamsd.measure("para", flights, spans, LONG_LAGS)
    np.savez(PIECES, **found)


def populations(p):
    """Mask over the pieces of each population."""
    label, flight = p["population"], p["flight"]
    pops = {k: label == k for k in ("solo", "group", "group_solo")}
    pops.update({k: label == k for k in SIZES})
    pops.update({f"{k}/alone": (label == "group_solo") & np.isin(flight, np.unique(flight[label == k])) for k in SIZES})
    return pops


def long_enough(p, mask, tau):
    """The pieces of `mask` that last long enough for the lag `tau`."""
    return mask & (p["duration"] >= tau / SUPPORT)


def available(p, mask):
    """Flights with pieces long enough for each lag of LONG_LAGS from 1000 s."""
    lags = LONG_LAGS[LONG_LAGS >= tamsd.TAU_MAX]
    return {"lags": lags.tolist(), "flights": [int(len(np.unique(p["flight"][long_enough(p, mask, t)]))) for t in lags]}


def longest(avail):
    """The longest lag at which a population keeps MIN_FLIGHTS flights."""
    ok = [t for t, n in zip(avail["lags"], avail["flights"]) if n >= MIN_FLIGHTS]
    return max(ok) if ok else 0


def curve(p, mask, block, rng, tau):
    """MSD, H_loc and bands of the population `mask` on its pieces long enough for the lag `tau`, over
    the lags up to `tau`."""
    keep = long_enough(p, mask, tau)
    lags = LONG_LAGS[LONG_LAGS <= tau]
    flights, k = np.unique(p["flight"][keep], return_inverse=True)
    sums, origins = np.zeros((len(flights), len(lags))), np.zeros((len(flights), len(lags)))
    np.add.at(sums, k, p["sums"][keep][:, :len(lags)])
    np.add.at(origins, k, p["origins"][keep][:, :len(lags)])
    mean, draws = ensemble(sums / origins, block.reindex(flights).to_numpy(), rng)
    h = np.array([local_h(lags, d) for d in draws])
    return {"flights": len(flights), "pieces": int(keep.sum()), "lags": lags.tolist(), "msd": mean.tolist(),
            "msd_band": np.percentile(draws, [5, 95], axis=0).tolist(), "H_loc": local_h(lags, mean).tolist(),
            "H_loc_band": np.percentile(h, [5, 95], axis=0).tolist()}


def panels(available):
    """The longest lag of each panel: the lag of LONG_LAGS nearest to FIXED_LAG, and the longest one at which
    every population of POOLED keeps MIN_FLIGHTS flights."""
    return {"fixed": int(LONG_LAGS[np.argmin(np.abs(LONG_LAGS - FIXED_LAG))]),
            "longest": int(min(longest(available[k]) for k in POOLED))}


def figure(result, path):
    """H_loc(tau) of S, G and A with an inset of MSD, one panel per longest lag. Thin: every piece of tamsd.py,
    up to 1000 s; dashed: the pieces long enough for the longest lag of the panel. Last panel: the flights
    that remain at each lag, against the minimum."""
    main = json.loads((OUT / "tamsd.json").read_text())["para"]
    label = {"solo": "S", "group": "G", "group_solo": "A"}
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.9), constrained_layout=True)
    for ax, panel in zip(axes, result["panels"].values()):
        drawn = []
        for k in POOLED:
            drawn.append(curves.series(main[k], f"{label[k]} ({main[k]['flights']:,})", COLORS[k]))
            drawn.append(curves.series(panel["curves"][k], None, COLORS[k], "--"))
        curves.plot(ax, drawn, ylim=(0.45, 1.0))
        ax.set_title(f"up to {panel['tau_max']:,} s", fontsize=8)
        ax.legend(frameon=False, fontsize=6, loc="lower right")
    ax = axes[-1]
    for k in POOLED:
        v = result["available"][k]
        ax.plot(v["lags"], v["flights"], color=COLORS[k], lw=1.1, label=label[k])
    ax.axhline(result["min_flights"], color="#555555", lw=0.8, ls=":")
    ax.set(xscale="log", yscale="log", xlabel=r"$\tau$ (s)", ylabel="flights with pieces of at least $\\tau/0.8$ s")
    ax.legend(frameon=False, fontsize=6.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=200, bbox_inches="tight")


if __name__ == "__main__":
    if "--redraw" in sys.argv:  # the figure from the saved JSON
        figure(json.loads((OUT / "longlags.json").read_text()), OUT / "longlags.pdf")
        sys.exit()
    if "--recompute" in sys.argv or not PIECES.exists():
        measure(tamsd.read_flights(), pd.read_parquet(LABELS / "membership.parquet"),
                pd.read_parquet(LABELS / "flight_summary.parquet"))
    p = dict(np.load(PIECES))
    block = pd.read_parquet(LABELS / "flight_summary.parquet").set_index("flight")["block"]
    rng = np.random.default_rng(SEED)
    masks = populations(p)
    result = {"min_flights": MIN_FLIGHTS, "available": {k: available(p, m) for k, m in masks.items()}, "panels": {}}
    for name, tau in panels(result["available"]).items():
        assert all(longest(result["available"][k]) >= tau for k in POOLED), f"fewer than {MIN_FLIGHTS} flights at {tau} s"
        result["panels"][name] = {"tau_max": tau, "curves": {k: curve(p, masks[k], block, rng, tau) for k in POOLED}}
        print(name, tau, {k: c["flights"] for k, c in result["panels"][name]["curves"].items()})
    (OUT / "longlags.json").write_text(json.dumps(result, indent=1))
    figure(result, OUT / "longlags.pdf")
