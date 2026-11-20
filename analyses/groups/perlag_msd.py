"""MSD of the pool and of the per-lag cohort, the curves whose slopes give the alpha of perlag_pool.

Pool: MSD(tau_k) on every piece that reaches tau_k, one curve. Per-lag: at each tau, the MSD of every lag of the
window on the pieces that reach tau+, a short curve per window, which meets the pool at tau+. Plain means, one
weight per piece, as cohorts.py; no bands.

    uv run python perlag_msd.py --declared-open    # after longlags.py; writes out/<run>/open_declared/perlag_msd.pdf
    # (open_declared_dt5 with --dt5: only the flights up to longlags.MAX_DT s)
"""

import matplotlib.pyplot as plt
import numpy as np

import longlags
import raw
from cohorts import window
from tamsd import COLORS, save

EVERY = 3  # a per-lag window every EVERY lags of the grid: adjacent windows barely overlap


def msd(p, keep, cols):
    return (p["sums"][keep][:, cols] / p["origins"][keep][:, cols]).mean(axis=0)


def enough(p, keep):
    return np.unique(p["flight"][keep]).size >= longlags.MIN_FLIGHTS


def curves(p, mask):
    lags = longlags.LONG_LAGS
    pool = [(t, msd(p, keep, [k])[0]) for k, t in enumerate(lags)
            if enough(p, keep := longlags.long_enough(p, mask, t))]
    windows = []
    for tau in lags[np.flatnonzero(lags >= 1000)[0]::EVERY]:  # below 1000 s the two coincide: every piece counts
        cols = window(tau)
        keep = longlags.long_enough(p, mask, lags[cols].max())
        if not enough(p, keep):
            break
        windows.append((lags[cols], msd(p, keep, cols)))
    return np.array(pool).T, windows


def main():
    p = longlags.load_pieces()
    masks = longlags.populations(p)
    fig, axes = plt.subplots(1, len(raw.POPS), figsize=(12, 3.8), sharey=True)
    for ax, (pop, short) in zip(axes, raw.POPS.items()):
        (lags, pool), windows = curves(p, masks[pop])
        ax.plot(lags, pool, color="#888888", lw=1.4, label="pool: each lag on its own pieces")
        for k, (w, m) in enumerate(windows):
            ax.plot(w, m, "o-", color=COLORS[pop], ms=2.5, lw=1.2,
                    label=r"per-lag: a window on the pieces of $\tau^+$" if k == 0 else None)
        ax.set(xscale="log", yscale="log", xlabel=r"$\tau$ (s)", title=short)
        ax.legend(frameon=False, fontsize=7)
        ax.grid(color="#eeeeee", lw=0.5)
    axes[0].set(ylabel=r"$\mathrm{MSD}(\tau)$ (m$^2$)")
    save(fig, longlags.RESULTS / "perlag_msd.pdf")


if __name__ == "__main__":
    main()
