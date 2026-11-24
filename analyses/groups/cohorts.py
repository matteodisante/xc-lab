"""Nested fixed cohorts: alpha(tau) of S, A, G on the cohorts of COHORTS, MSD with one weight per piece.

A cohort to tau_max keeps the pieces of at least tau_max / SUPPORT s (longlags.long_enough) and is measured at
every lag up to tau_max. If alpha at a shared lag moves between cohorts, the length of the pieces selects; the gaps of raw.GAPS
show whether the selection moves the contrasts too.
chained(): MSD* and alpha* by the chain-index construction, a third way to meet the selection by length.
Pieces, cells, seed and two-way bands as raw.py.

    uv run python cohorts.py --open    # after longlags.py --open; writes out/<run>/open_min<m>/cohorts*
    uv run python cohorts.py --declared-open [--min-opening X]    # the declared open flights, or only those with
                                       # an opening above X (the groups are those of the run); writes cohorts[_min<X>]*
    # with --dt5 only the flights up to longlags.MAX_DT s, written to longlags.RESULTS (open_declared_dt5)
"""

import json
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import NullFormatter, ScalarFormatter

import closure
import longlags
import raw
import twoway
from longlags import RESULTS
from tamsd import (COLORS, HALF_DECADE, LABELS, SEED, SUPPORT, TAU_MAX, ensemble, fit_h, local_h, read_flights,
                   save)

COHORTS = (1000, 2760, 3600, 7200, 10800)  # s: tau_max of each cohort
# s: the longest cohort that agrees with the shorter ones within the bands (cohorts_tradeoff). The per-lag curve
# is drawn solid while its pieces are at least as many as this cohort's, dashed beyond (long flights only).
SELECTION_FREE = 3600
QUOTED_LAG = 1000  # s: the lag shared by every cohort
YLIM = (0.80, 1.00)  # alpha axis of the cohort figures, shared so that the runs compare by eye
GAP_YLIM = (-0.035, 0.11)  # the same for the gap figures
POOL_COLOR = "#d62728"


def cohort(p, mask, tau, code, counts, rng):
    """alpha(tau) and its band on the pieces of `mask` long enough for `tau`, one weight per piece."""
    keep = longlags.long_enough(p, mask, tau)
    lags = longlags.LONG_LAGS[longlags.LONG_LAGS <= tau]
    values = p["sums"][keep][:, :len(lags)] / p["origins"][keep][:, :len(lags)]
    mean, draws = ensemble(values, code.reindex(p["flight"][keep]).to_numpy(), rng, counts)
    alpha = local_h(lags, mean)
    rows = np.vstack([alpha, [local_h(lags, d) for d in draws]])  # the draws are shared by the populations
    return {"tau_max": tau, "flights": int(np.unique(p["flight"][keep]).size), "pieces": int(keep.sum()),
            "lags": lags.tolist(), "msd": mean.tolist(), "alpha": np.asarray(alpha).tolist(), "band": np.asarray(twoway.band(alpha, rows[1:])).tolist(),
            "rows": rows}


def window(tau):
    """The lags of LONG_LAGS that local_h uses at `tau`: within HALF_DECADE, or the 3 nearest."""
    x = np.log10(longlags.LONG_LAGS) - np.log10(tau)  # as local_h, so that ties break the same way
    near = np.abs(x) <= HALF_DECADE + 1e-9
    if near.sum() < 3:
        near = np.isin(np.arange(len(x)), np.argsort(np.abs(x), kind="stable")[:3])
    return np.flatnonzero(near)


def per_lag(p, mask, code, counts, rng):
    """alpha at every lag on its own cohort: the pieces of `mask` long enough for the whole window of the lag.
    No composition term, since both ends of the slope see the same pieces; stops when fewer than
    longlags.MIN_FLIGHTS flights remain."""
    lags, rows, pieces = [], [], []
    for tau in longlags.LONG_LAGS:
        cols = window(tau)
        keep = longlags.long_enough(p, mask, longlags.LONG_LAGS[cols].max())
        if np.unique(p["flight"][keep]).size < longlags.MIN_FLIGHTS:
            break
        values = p["sums"][keep][:, cols] / p["origins"][keep][:, cols]
        mean, draws = ensemble(values, code.reindex(p["flight"][keep]).to_numpy(), rng, counts)
        w = longlags.LONG_LAGS[cols]
        rows.append([fit_h(w, mean)] + [fit_h(w, d) for d in draws])
        lags.append(int(tau))
        pieces.append(int(keep.sum()))
    return {"lags": lags, "pieces": pieces, "rows": np.array(rows).T}


def pool(p, mask, code, counts, rng):
    """alpha with the MSD of every lag on all the pieces that reach it: the two ends of a slope see different
    pieces (the composition term of the fixed cohort argument). For comparison only."""
    msd, lags = [], []
    for k, tau in enumerate(longlags.LONG_LAGS):
        keep = longlags.long_enough(p, mask, tau)
        if np.unique(p["flight"][keep]).size < longlags.MIN_FLIGHTS:
            break
        mean, draws = ensemble(p["sums"][keep][:, [k]] / p["origins"][keep][:, [k]],
                               code.reindex(p["flight"][keep]).to_numpy(), rng, counts)
        msd.append(np.r_[mean, draws[:, 0]])
        lags.append(int(tau))
    msd = np.array(msd).T
    return {"lags": lags, "msd": msd[0].tolist(), "rows": np.array([local_h(np.array(lags), m) for m in msd])}


def flight_weights(flight):
    """One weight per piece: 1 over the pieces of its flight among `flight`, so that a flight weighs 1."""
    _, k, n = np.unique(flight, return_inverse=True, return_counts=True)
    return 1.0 / n[k]


def chained(p, mask, code, counts, rng, per_flight=False):
    """MSD* by the chain-index construction and alpha* = local_h of it: MSD*(tau_1) is the mean of all the pieces
    that reach tau_1; each step to the next lag multiplies it by the ratio of the means of the two lags on the
    pieces that reach both. The draws of `counts` are chained alike. Stops when a step keeps fewer than
    longlags.MIN_FLIGHTS flights. rows: alpha* of the point estimate, then of every draw. With `per_flight`, in
    every step the pieces of one flight that enter it share a total weight 1 (flight_weights), not 1 each."""
    lags, pieces, flights = longlags.LONG_LAGS, [], []

    def step(keep, cols):
        values = p["sums"][keep][:, cols] / p["origins"][keep][:, cols]
        w = flight_weights(p["flight"][keep]) if per_flight else None
        return ensemble(values, code.reindex(p["flight"][keep]).to_numpy(), rng, counts, w)

    keep = longlags.long_enough(p, mask, lags[0])
    mean, draws = step(keep, [0])
    log, log_draws = [np.log(mean[0])], [np.log(draws[:, 0])]
    pieces.append(int(keep.sum()))
    flights.append(int(np.unique(p["flight"][keep]).size))
    for k in range(len(lags) - 1):
        keep = longlags.long_enough(p, mask, lags[k + 1])
        if np.unique(p["flight"][keep]).size < longlags.MIN_FLIGHTS:
            break
        mean, draws = step(keep, [k, k + 1])
        log.append(log[-1] + np.log(mean[1] / mean[0]))
        log_draws.append(log_draws[-1] + np.log(draws[:, 1] / draws[:, 0]))
        pieces.append(int(keep.sum()))
        flights.append(int(np.unique(p["flight"][keep]).size))
    lags, msd, msd_draws = lags[:len(log)], np.exp(log), np.exp(np.array(log_draws).T)
    rows = np.vstack([local_h(lags, msd), [local_h(lags, d) for d in msd_draws]])
    return {"lags": lags.tolist(), "pieces": pieces, "flights": flights, "msd": msd, "msd_draws": msd_draws,
            "rows": rows}


def solid(lags):
    """True where the window of the lag ends within SELECTION_FREE: pieces no longer than the reference cohort's."""
    return np.array([longlags.LONG_LAGS[window(t)].max() <= SELECTION_FREE for t in lags])


def per_lag_summary(lag, result, pooled, chain):
    """The per-lag alpha of every population and the gaps of raw.GAPS, with bands; the difference from the
    cohort to SELECTION_FREE at the lags they share (where the per-lag cohort keeps at least its pieces); the
    pooled alpha and its difference from the per-lag one (the composition term); the chained MSD* and alpha*, their
    difference from the per-lag alpha and from the cohort to SELECTION_FREE (where solid), the pool minus alpha*,
    and the gaps of alpha*."""
    n = min(len(v["lags"]) for v in lag.values())
    lags = lag[next(iter(lag))]["lags"][:n]
    out = {"selection_free": SELECTION_FREE, "min_piece_s": TAU_MAX / SUPPORT, "half_decade": HALF_DECADE,
           "support": SUPPORT, "lags": lags, "solid": solid(lags).tolist(), "alpha": {}, "gaps": {},
           "minus cohort": {}, "pool": {}, "pool minus per-lag": {}, "chained": {}, "chained minus per-lag": {},
           "chained gaps": {}, "chained minus cohort": {}, "pool minus chained": {}}
    for pop, v in lag.items():
        out["alpha"][pop] = {"lags": v["lags"], "pieces": v["pieces"], **twoway.quote(v["rows"])}
        q = pooled[pop]
        m = min(len(q["lags"]), len(v["lags"]))
        out["pool"][pop] = {"lags": q["lags"], **twoway.quote(q["rows"])}
        out["pool minus per-lag"][pop] = {"lags": q["lags"][:m], **twoway.quote(q["rows"][:, :m] - v["rows"][:, :m])}
        c = chain[pop]
        m = min(len(c["lags"]), len(v["lags"]))
        out["chained"][pop] = {"lags": c["lags"], "pieces": c["pieces"], "flights": c["flights"],
                               "msd": c["msd"].tolist(), "msd_band": twoway.band(c["msd"], c["msd_draws"]),
                               **twoway.quote(c["rows"])}
        out["chained minus per-lag"][pop] = {"lags": c["lags"][:m],
                                             **twoway.quote(c["rows"][:, :m] - v["rows"][:, :m])}
        ref = next(c for c in result[pop] if c["tau_max"] == SELECTION_FREE)
        out.setdefault("cohort", {})[pop] = {"lags": ref["lags"], "alpha": ref["alpha"]}
        shared = [t for t in v["lags"] if t in ref["lags"] and solid([t])[0]]
        d = v["rows"][:, [v["lags"].index(t) for t in shared]] - ref["rows"][:, [ref["lags"].index(t) for t in shared]]
        out["minus cohort"][pop] = {"lags": shared, **twoway.quote(d)}
        shared = [t for t in c["lags"] if t in ref["lags"] and solid([t])[0]]
        d = c["rows"][:, [c["lags"].index(t) for t in shared]] - ref["rows"][:, [ref["lags"].index(t) for t in shared]]
        out["chained minus cohort"][pop] = {"lags": shared, **twoway.quote(d)}
        m = min(len(q["lags"]), len(c["lags"]))
        out["pool minus chained"][pop] = {"lags": q["lags"][:m], **twoway.quote(q["rows"][:, :m] - c["rows"][:, :m])}
    for name, (a, b) in raw.GAPS.items():
        out["gaps"][name] = twoway.quote(lag[a]["rows"][:, :n] - lag[b]["rows"][:, :n])
        m = min(len(chain[a]["lags"]), len(chain[b]["lags"]))
        out["chained gaps"][name] = {"lags": chain[a]["lags"][:m],
                                     **twoway.quote(chain[a]["rows"][:, :m] - chain[b]["rows"][:, :m])}
    return out


def pool_figure(s, path):
    """One panel per population: the pooled alpha, the per-lag one and the cohort to SELECTION_FREE."""
    fig, axes = plt.subplots(1, len(raw.POPS), figsize=(12, 3.8), sharey=True)
    for ax, (pop, short) in zip(axes, raw.POPS.items()):
        for curve, color, style, label in ((s["pool"][pop], "#888888", "-", "every piece at each lag (pool)"),
                                           (s["alpha"][pop], COLORS[pop], "-", "per-lag cohort")):
            lags, band = np.array(curve["lags"]), np.array(curve["band"])
            ax.fill_between(lags, band[0], band[1], color=color, alpha=0.2, lw=0)
            ax.plot(lags, curve["value"], color=color, lw=1.4, ls=style, label=label)
        ax.plot(s["cohort"][pop]["lags"], s["cohort"][pop]["alpha"], color="k", lw=0.9, ls=":",
                label=f"cohort to {SELECTION_FREE:,} s")
        ax.set(xscale="log", xlabel=r"$\tau$ (s)", title=short)
        ax.legend(frameon=False, fontsize=7)
        ax.grid(color="#eeeeee", lw=0.5)
    axes[0].set(ylabel=r"$\alpha(\tau)$", ylim=YLIM)
    save(fig, path)


def chained_figure(s, path):
    """Top: alpha* (chained) and the per-lag alpha with the pool for reference; bottom: alpha* minus the
    per-lag alpha. One column per population."""
    fig, axes = plt.subplots(2, len(raw.POPS), figsize=(12, 6.8), sharex="col", sharey="row")
    for ax, bot, (pop, short) in zip(axes[0], axes[1], raw.POPS.items()):
        for curve, color, style, label, lw in ((s["pool"][pop], POOL_COLOR, "-", "pool", 0.9),
                                               (s["alpha"][pop], "#555555", "--", r"per-lag $\alpha$", 1.2),
                                               (s["chained"][pop], COLORS[pop], "-", r"chained $\alpha^*$", 1.6)):
            lags, band = np.array(curve["lags"]), np.array(curve["band"])
            if color != POOL_COLOR:
                ax.fill_between(lags, band[0], band[1], color=color, alpha=0.15, lw=0)
            ax.plot(lags, curve["value"], color=color, lw=lw, ls=style, label=label)
        d = s["chained minus per-lag"][pop]
        lags, band = np.array(d["lags"]), np.array(d["band"])
        bot.fill_between(lags, band[0], band[1], color=COLORS[pop], alpha=0.2, lw=0)
        bot.plot(lags, d["value"], color=COLORS[pop], lw=1.5)
        bot.axhline(0, color="k", lw=0.6)
        ax.set(xscale="log", title=short, ylim=(0.80, 1.05))
        bot.set(xscale="log", xlabel=r"$\tau$ (s)")
        ax.legend(frameon=False, fontsize=7)
        for a in (ax, bot):
            a.grid(color="#eeeeee", lw=0.5)
    axes[0, 0].set(ylabel=r"$\alpha(\tau)$")
    axes[1, 0].set(ylabel=r"$\alpha^*-\alpha_\mathrm{per\text{-}lag}$")
    save(fig, path)


def chained_msd_figure(s, pooled, path):
    """Per population: the chained MSD* with its band and the pool MSD."""
    fig, axes = plt.subplots(1, len(raw.POPS), figsize=(12, 3.8), sharey=True)
    for ax, (pop, short) in zip(axes, raw.POPS.items()):
        c = s["chained"][pop]
        lags, band = np.array(c["lags"]), np.array(c["msd_band"])
        ax.fill_between(lags, band[0], band[1], color=COLORS[pop], alpha=0.2, lw=0)
        ax.plot(lags, c["msd"], color=COLORS[pop], lw=1.6, label=r"chained MSD$^*$")
        ax.plot(pooled[pop]["lags"], pooled[pop]["msd"], color=POOL_COLOR, lw=1, label="pool: each lag on its own pieces")
        ax.set(xscale="log", yscale="log", xlabel=r"$\tau$ (s)", title=short)
        ax.legend(frameon=False, fontsize=7)
        ax.grid(color="#eeeeee", lw=0.5)
    axes[0].set(ylabel=r"$\mathrm{MSD}(\tau)$ (m$^2$)")
    save(fig, path)


def chained_pool_figure(s, path):
    """Per population: alpha* and the pooled alpha with bands, and the cohort to SELECTION_FREE."""
    fig, axes = plt.subplots(1, len(raw.POPS), figsize=(12, 3.8), sharey=True)
    for ax, (pop, short) in zip(axes, raw.POPS.items()):
        for curve, color, lw, label in ((s["pool"][pop], POOL_COLOR, 1.0, "pool: each lag on its own pieces"),
                                        (s["chained"][pop], COLORS[pop], 1.6, r"chained $\alpha^*$")):
            lags, band = np.array(curve["lags"]), np.array(curve["band"])
            ax.fill_between(lags, band[0], band[1], color=color, alpha=0.2, lw=0)
            ax.plot(lags, curve["value"], color=color, lw=lw, label=label)
        ax.plot(s["cohort"][pop]["lags"], s["cohort"][pop]["alpha"], color="k", lw=0.9, ls=":",
                label=f"cohort to {SELECTION_FREE:,} s")
        ax.set(xscale="log", xlabel=r"$\tau$ (s)", title=short)
        ax.legend(frameon=False, fontsize=7)
        ax.grid(color="#eeeeee", lw=0.5)
    axes[0].set(ylabel=r"$\alpha(\tau)$", ylim=YLIM)
    save(fig, path)


def solid_dashed(ax, lags, value, color, label=None):
    """Solid where the window ends within SELECTION_FREE, dashed from its last point on."""
    keep = solid(lags)
    ax.plot(lags[keep], value[keep], color=color, lw=1.5, label=label)
    tail = np.arange(len(keep)) >= keep.sum() - 1  # the solid part is a prefix: the window grows with tau
    ax.plot(lags[tail], value[tail], color=color, lw=1.5, ls="--")


def chained_summary_figure(s, result, path):
    """Left: alpha* of S, A, G (solid up to SELECTION_FREE, dashed beyond) with the cohort to SELECTION_FREE
    dotted. Right: the gaps of alpha* of raw.GAPS."""
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 3.8))
    for pop, short in raw.POPS.items():
        a = s["chained"][pop]
        lags, band = np.array(a["lags"]), np.array(a["band"])
        left.fill_between(lags, band[0], band[1], color=COLORS[pop], alpha=0.2, lw=0)
        solid_dashed(left, lags, np.array(a["value"]), COLORS[pop], short)
        ref = next(c for c in result[pop] if c["tau_max"] == SELECTION_FREE)
        left.plot(ref["lags"], ref["alpha"], color=COLORS[pop], lw=0.9, ls=":")
    left.set(xscale="log", xlabel=r"$\tau$ (s)", ylabel=r"$\alpha^*(\tau)$", ylim=YLIM)
    left.plot([], [], color="k", lw=0.9, ls=":", label=f"cohort to {SELECTION_FREE:,} s")
    left.legend(frameon=False, fontsize=7)
    for name, g in s["chained gaps"].items():
        lags, band = np.array(g["lags"]), np.array(g["band"])
        right.fill_between(lags, band[0], band[1], color=raw.GAP_COLORS[name], alpha=0.2, lw=0)
        solid_dashed(right, lags, np.array(g["value"]), raw.GAP_COLORS[name], name.replace(" - ", " $-$ "))
    right.axhline(0, color="k", lw=0.6)
    right.set(xscale="log", xlabel=r"$\tau$ (s)", ylabel=r"$\Delta\alpha^*(\tau)$", ylim=GAP_YLIM)
    right.legend(frameon=False, fontsize=7)
    for ax in (left, right):
        ax.grid(color="#eeeeee", lw=0.5)
    save(fig, path)


def per_lag_figure(s, result, path):
    """Left: alpha of S, A, G, each lag on its own cohort (solid up to SELECTION_FREE, dashed beyond), with the
    cohort to SELECTION_FREE dotted. Right: the gaps of raw.GAPS."""
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 3.8))
    for pop, short in raw.POPS.items():
        a = s["alpha"][pop]
        lags, value, band, keep = np.array(a["lags"]), np.array(a["value"]), np.array(a["band"]), solid(a["lags"])
        left.fill_between(lags, band[0], band[1], color=COLORS[pop], alpha=0.2, lw=0)
        left.plot(lags[keep], value[keep], color=COLORS[pop], lw=1.5, label=short)
        tail = np.arange(len(keep)) >= keep.sum() - 1  # the solid part is a prefix: the window grows with tau
        left.plot(lags[tail], value[tail], color=COLORS[pop], lw=1.5, ls="--")
        ref = next(c for c in result[pop] if c["tau_max"] == SELECTION_FREE)
        left.plot(ref["lags"], ref["alpha"], color=COLORS[pop], lw=0.9, ls=":")
    left.set(xscale="log", xlabel=r"$\tau$ (s)", ylabel=r"$\alpha(\tau)$", ylim=YLIM)
    left.plot([], [], color="k", lw=0.9, ls=":", label=f"cohort to {SELECTION_FREE:,} s")
    left.legend(frameon=False, fontsize=7)
    lags, keep = np.array(s["lags"]), np.array(s["solid"])
    for name, g in s["gaps"].items():
        value, band = np.array(g["value"]), np.array(g["band"])
        right.fill_between(lags, band[0], band[1], color=raw.GAP_COLORS[name], alpha=0.2, lw=0)
        right.plot(lags[keep], value[keep], color=raw.GAP_COLORS[name], lw=1.5, label=name.replace(" - ", " $-$ "))
        tail = np.arange(len(keep)) >= keep.sum() - 1
        right.plot(lags[tail], value[tail], color=raw.GAP_COLORS[name], lw=1.5, ls="--")
    right.axhline(0, color="k", lw=0.6)
    right.set(xscale="log", xlabel=r"$\tau$ (s)", ylabel=r"$\Delta\alpha(\tau)$", ylim=GAP_YLIM)
    right.legend(frameon=False, fontsize=7)
    for ax in (left, right):
        ax.grid(color="#eeeeee", lw=0.5)
    save(fig, path)


def gaps(result):
    """Per gap of raw.GAPS and cohort: the difference of alpha at every lag, with its band."""
    return {gap: [{"tau_max": ca["tau_max"], "lags": ca["lags"], **twoway.quote(ca["rows"] - cb["rows"])}
                  for ca, cb in zip(result[a], result[b])] for gap, (a, b) in raw.GAPS.items()}


def shades():
    return plt.cm.viridis(np.linspace(0.85, 0.0, len(COHORTS)))


def draw_gaps(axes, gap):
    for ax, (name, cs) in zip(axes, gap.items()):
        for c, color in zip(cs, shades()):
            lags, band = np.array(c["lags"]), np.array(c["band"])
            ax.fill_between(lags, band[0], band[1], color=color, alpha=0.2, lw=0)
            ax.plot(lags, c["value"], color=color, lw=1.4, label=f"to {c['tau_max']:,} s")
        ax.axhline(0, color="k", lw=0.6)
        ax.set(xscale="log", xlabel=r"$\tau$ (s)", title=name.replace(" - ", " $-$ "))
        ax.legend(frameon=False, fontsize=7)
        ax.grid(color="#eeeeee", lw=0.5)
    axes[0].set(ylabel=r"$\Delta\alpha(\tau)$", ylim=GAP_YLIM)


def gap_figure(gap, path):
    """The gaps of alpha of every cohort, one panel per gap."""
    fig, axes = plt.subplots(1, len(gap), figsize=(12, 3.8), sharey=True)
    draw_gaps(axes, gap)
    save(fig, path)


def gap_msd_figure(result, gap, path):
    """Top: the MSD ratio of the two populations of each gap, per cohort; bottom: the gaps of alpha."""
    fig, axes = plt.subplots(2, len(gap), figsize=(12, 6.8), sharex="col", sharey="row")
    for ax, (name, (a, b)) in zip(axes[0], raw.GAPS.items()):
        for ca, cb, color in zip(result[a], result[b], shades()):
            ax.plot(ca["lags"], np.array(ca["msd"]) / np.array(cb["msd"]), color=color, lw=1.4,
                    label=f"to {ca['tau_max']:,} s")
        ax.axhline(1, color="k", lw=0.6)
        ax.set(xscale="log", yscale="log", title=name.replace(" - ", " $-$ "))
        ax.legend(frameon=False, fontsize=7)
        ax.grid(color="#eeeeee", lw=0.5)
    axes[0, 0].set(ylabel=r"$\mathrm{MSD}_X/\mathrm{MSD}_Y$", yticks=[1, 1.1, 1.2, 1.3, 1.4])
    axes[0, 0].yaxis.set_major_formatter(ScalarFormatter())
    axes[0, 0].yaxis.set_minor_formatter(NullFormatter())
    draw_gaps(axes[1], gap)
    save(fig, path)


def draw_pool(ax, q, key):
    """The pool (every piece that reaches each lag) in red; its band when the curve is alpha."""
    lags = np.array(q["lags"])
    if key == "value":
        ax.fill_between(lags, *q["band"], color=POOL_COLOR, alpha=0.15, lw=0)
    y = np.array(q["msd" if key == "msd/tau" else key])
    ax.plot(lags, y / lags if key == "msd/tau" else y, color=POOL_COLOR, lw=1.8,
            label="whole dataset (pool, no fixed cohort)")


def draw_alpha(axes, result, pooled=None):
    for ax, (pop, short) in zip(axes, raw.POPS.items()):
        if pooled:
            draw_pool(ax, pooled[pop], "value")
        for c, color in zip(result[pop], shades()):
            lags, band = np.array(c["lags"]), np.array(c["band"])
            ax.fill_between(lags, band[0], band[1], color=color, alpha=0.2, lw=0)
            ax.plot(lags, c["alpha"], color=color, lw=1.4, label=f"to {c['tau_max']:,} s ({c['pieces']:,} pieces)")
        ax.set(xscale="log", xlabel=r"$\tau$ (s)", title=short)
        ax.legend(frameon=False, fontsize=7)
        ax.grid(color="#eeeeee", lw=0.5)
    ylim = YLIM
    if pooled:  # the pool must be fully visible
        band = np.concatenate([np.array(q["band"]).ravel() for q in pooled.values()])
        ylim = (min(YLIM[0], np.floor(band.min() * 100) / 100), max(YLIM[1], np.ceil(band.max() * 100) / 100))
    axes[0].set(ylabel=r"$\alpha(\tau)$", ylim=ylim)


def figure(result, path):
    """alpha(tau) of every cohort, one panel per population."""
    fig, axes = plt.subplots(1, len(raw.POPS), figsize=(12, 3.8), sharey=True)
    draw_alpha(axes, result)
    save(fig, path)


def msd_figure(result, pooled, path):
    """Rows: MSD(tau), MSD(tau)/tau, alpha(tau) of every cohort. One column per population."""
    fig, axes = plt.subplots(3, len(raw.POPS), figsize=(12, 10), sharex="col", sharey="row")
    for k, row in enumerate(axes[:2]):
        for ax, (pop, short) in zip(row, raw.POPS.items()):
            draw_pool(ax, pooled[pop], "msd/tau" if k else "msd")
            for c, color in zip(result[pop], shades()):
                lags, msd = np.array(c["lags"]), np.array(c["msd"])
                ax.plot(lags, msd / lags if k else msd, color=color, lw=1.4, label=f"to {c['tau_max']:,} s")
            ax.set(xscale="log", yscale="log", title=short)
            ax.legend(frameon=False, fontsize=7)
            ax.grid(color="#eeeeee", lw=0.5)
    axes[0, 0].set(ylabel=r"$\mathrm{MSD}(\tau)$ (m$^2$)")
    axes[1, 0].set(ylabel=r"$\mathrm{MSD}(\tau)/\tau$ (m$^2$/s)")
    draw_alpha(axes[2], result, pooled)
    save(fig, path)


def trade_off(result, path):
    """Left: alpha(QUOTED_LAG) against the tau_max of the cohort, with its band. Right: the pieces of each cohort."""
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 3.6))
    for k, (pop, short) in enumerate(raw.POPS.items()):
        taus = np.array([c["tau_max"] for c in result[pop]]) * (1 + 0.03 * (k - 1))  # side by side
        i = [c["lags"].index(QUOTED_LAG) for c in result[pop]]
        value = np.array([c["alpha"][j] for c, j in zip(result[pop], i)])
        lo, hi = (np.array([c["band"][b][j] for c, j in zip(result[pop], i)]) for b in (0, 1))
        left.errorbar(taus, value, [value - lo, hi - value], fmt="o-", ms=4, lw=1, capsize=2,
                      color=COLORS[pop], label=short)
        right.plot(taus, [c["pieces"] for c in result[pop]], "o-", ms=4, lw=1, color=COLORS[pop], label=short)
    left.set(xscale="log", xlabel=r"cohort $\tau_\mathrm{max}$ (s)", ylabel=rf"$\alpha({QUOTED_LAG}$ s$)$")
    right.set(xscale="log", yscale="log", xlabel=r"cohort $\tau_\mathrm{max}$ (s)", ylabel="pieces")
    for ax in (left, right):
        ax.set_xticks(COHORTS, [f"{t:,}" for t in COHORTS], fontsize=7)
        ax.minorticks_off()
        ax.legend(frameon=False, fontsize=8)
        ax.grid(color="#eeeeee", lw=0.5)
    save(fig, path)


if __name__ == "__main__":
    RESULTS.mkdir(parents=True, exist_ok=True)
    flights = raw.with_takeoffs(read_flights())
    code, pilot, block = twoway.cells(flights, pd.read_parquet(LABELS / "flight_summary.parquet"))
    rng = np.random.default_rng(SEED)
    counts = twoway.counts(pilot, block, rng)
    p = longlags.load_pieces()
    tag = ""
    if "--min-opening" in sys.argv:
        x = float(sys.argv[sys.argv.index("--min-opening") + 1])
        opening = closure.flight_table().set_index("flight")["ratio"]
        keep = opening.reindex(p["flight"]).to_numpy() > x
        p = {k: v[keep] if len(v) == len(keep) else v for k, v in p.items()}
        tag = f"_min{x:g}"
    masks = longlags.populations(p)
    all_mask = np.ones(len(p["flight"]), dtype=bool)  # every piece of every flight, S+A+G together
    result = {pop: [cohort(p, masks[pop], tau, code, counts, rng) for tau in COHORTS] for pop in raw.POPS}
    result["all"] = [cohort(p, all_mask, tau, code, counts, rng) for tau in COHORTS]
    gap = gaps(result)
    lag = {pop: per_lag(p, masks[pop], code, counts, rng) for pop in raw.POPS}
    pooled = {pop: pool(p, masks[pop], code, counts, rng) for pop in raw.POPS}
    pooled["all"] = pool(p, all_mask, code, counts, rng)
    chain = {pop: chained(p, masks[pop], code, counts, rng) for pop in raw.POPS}  # after the calls above
    summary = per_lag_summary(lag, result, pooled, chain)
    (RESULTS / f"perlag{tag}.json").write_text(json.dumps(summary))
    per_lag_figure(summary, result, RESULTS / f"perlag{tag}.pdf")
    pool_figure(summary, RESULTS / f"perlag_pool{tag}.pdf")
    chained_figure(summary, RESULTS / f"perlag_chained{tag}.pdf")
    chained_msd_figure(summary, pooled, RESULTS / f"chained_msd{tag}.pdf")
    chained_pool_figure(summary, RESULTS / f"chained_pool{tag}.pdf")
    chained_summary_figure(summary, result, RESULTS / f"chained{tag}.pdf")
    for cs in result.values():
        for c in cs:
            c.pop("rows")
    pool_out = {pop: {"lags": q["lags"], "msd": q["msd"], **twoway.quote(q["rows"])} for pop, q in pooled.items()}
    aggregate = {"alpha": result.pop("all"), "pool": pool_out.pop("all")}  # all pieces together: for the slides
    (RESULTS / f"cohorts{tag}.json").write_text(json.dumps({"alpha": result, "gaps": gap, "pool": pool_out, "all": aggregate}))
    gap_figure(gap, RESULTS / f"cohorts_gaps{tag}.pdf")
    figure(result, RESULTS / f"cohorts{tag}.pdf")
    msd_figure(result, pool_out, RESULTS / f"cohorts_msd{tag}.pdf")
    gap_msd_figure(result, gap, RESULTS / f"cohorts_gaps_msd{tag}.pdf")
    trade_off(result, RESULTS / f"cohorts_tradeoff{tag}.pdf")
