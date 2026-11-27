"""Scheme of the pool, the per-lag cohort and the chained MSD* on toy pieces: which pieces enter each lag, the MSDs,
their alpha. perlag_scheme.pdf has the pool and the per-lag cohort; chained_scheme.pdf the pool and the chained MSD*.

Toy pieces: durations T_p log-uniform on [T_MIN, T_MAX], MSD_p(tau) = c_p tau^(2 ALPHA) with c_p = (T_p/T0)^BETA
(the longest flights are straighter at every scale). The local exponent of the pool has the closed form
alpha + BETA/2 g(y), g(y) = 1/y - 1/(e^y - 1), y = BETA ln(T_MAX SUPPORT / tau) (group_story.tex, Which cohort).

    uv run python perlag_scheme.py    # writes out/<run>/perlag_scheme.pdf, chained_scheme.pdf and perlag_scheme.json
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from tamsd import COLORS, HALF_DECADE, LABELS, SUPPORT, save

TAU = 1e4  # s: the window drawn in the top row and in the MSD panel
ALPHA, BETA = 0.9, 0.3  # exponent of every piece; growth of its prefactor with the duration
T0, T_MIN, T_MAX = 1e4, 1000 / SUPPORT, 1e5  # s; every piece reaches 1000 s, as in the data
N_PIECES, N_LAGS = 400, 7
COHORTS = (2000, 5000, 10000, 20000, 50000)  # s: tau_max of the nested toy cohorts
GRAY, PURPLE = "#888888", "#7B2D8E"
OUT = Path(__file__).with_name("out") / LABELS.name


def window(tau):
    return tau * 10 ** np.linspace(-HALF_DECADE, HALF_DECADE, N_LAGS)


def sets(dur, lags):
    """Pool: piece p at lag k if it reaches tau_k. Per-lag cohort: if it reaches tau+, at every lag."""
    pool = dur[:, None] >= lags[None, :] / SUPPORT
    return pool, np.repeat(pool[:, -1:], len(lags), axis=1)


def msd(c, used, lags):
    return np.array([c[used[:, k]].mean() for k in range(len(lags))]) * lags ** (2 * ALPHA)


def cohort_mean(dur, c, lags):
    """Mean of the MSDs of the cohorts that reach each lag, one weight per cohort; nan where none does."""
    out = []
    for t in lags:
        reach = [c[dur >= tmax / SUPPORT].mean() for tmax in COHORTS if tmax >= t]
        out.append(np.mean(reach) * t ** (2 * ALPHA) if reach else np.nan)
    return np.array(out)


def half_slope(lags, m):
    return np.polyfit(np.log(lags), np.log(m), 1)[0] / 2


def pieces():
    dur = np.logspace(np.log10(T_MIN), np.log10(T_MAX), N_PIECES)
    return dur, (dur / T0) ** BETA


def fits(tau, dur, c):
    """Half-slopes over the window of tau: pool, per-lag cohort, mean of the cohorts (nan if a lag has none)."""
    lags = window(tau)
    return [half_slope(lags, msd(c, used, lags)) for used in sets(dur, lags)] + [
        half_slope(lags, m) if np.isfinite(m := cohort_mean(dur, c, lags)).all() else np.nan]


def theory(tau):
    """Local exponent of the pool: alpha while every piece reaches tau, then alpha + BETA/2 g(y)."""
    y = BETA * np.log(T_MAX * SUPPORT / np.maximum(tau, T_MIN * SUPPORT))
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(tau <= T_MIN * SUPPORT, ALPHA, ALPHA + BETA / 2 * (1 / y - 1 / np.expm1(y)))


def chain_grid(dur, c):
    """MSD* on a log grid from 100 s to T_MAX SUPPORT (the last lag) with the spacing of window(): the first lag is
    the mean of every piece that reaches it, each step multiplies by the ratio of the means of the two lags on
    P(tau_{k+1})."""
    r = 10 ** (2 * HALF_DECADE / (N_LAGS - 1))
    lags = (T_MAX * SUPPORT / r ** np.arange(int(np.log(T_MAX * SUPPORT / 100) / np.log(r)) + 1))[::-1]
    reach = lambda t: dur >= t / SUPPORT
    msd = [c[reach(lags[0])].mean() * lags[0] ** (2 * ALPHA)]
    for a, b in zip(lags[:-1], lags[1:]):
        used = reach(b)
        msd.append(msd[-1] * (c[used].mean() * b ** (2 * ALPHA)) / (c[used].mean() * a ** (2 * ALPHA)))
    return lags, np.array(msd)


def chained_msd(dur, c, lags):
    """MSD* at `lags`, interpolating log MSD* in log tau on the grid."""
    g, m = chain_grid(dur, c)
    return np.exp(np.interp(np.log(lags), np.log(g), np.log(m)))


def chained_fit(tau, dur, c):
    lags = window(tau)
    return half_slope(lags, chained_msd(dur, c, lags))


def grid(ax, used, lags, color, title):
    ax.imshow(used, aspect="auto", cmap=plt.matplotlib.colors.ListedColormap(["#f2f2f2", color]),
              origin="lower", interpolation="nearest")
    ax.set(title=title, xticks=range(N_LAGS), xticklabels=[f"{t:,.0f}" for t in lags], yticks=[])
    ax.tick_params(axis="x", labelsize=6, rotation=45)


def main():
    dur, c = pieces()
    lags = window(TAU)
    pool, perlag = sets(dur, lags)
    fig, axes = plt.subplots(2, 2, figsize=(9, 7.2), gridspec_kw={"height_ratios": [1, 1.1]})
    grid(axes[0, 0], pool, lags, GRAY, r"Pool: each lag on $P(\tau_k)$")
    grid(axes[0, 1], perlag, lags, COLORS["group"], r"Per-lag cohort: every lag on $P(\tau^+)$")
    axes[0, 0].set(ylabel="pieces, short to long (row)")

    ax = axes[1, 0]
    series = ((msd(c, pool, lags), GRAY, "pool"), (msd(c, perlag, lags), COLORS["group"], "per-lag"),
              (cohort_mean(dur, c, lags), PURPLE, "mean of the cohorts"))
    for m, color, name in series:
        a = half_slope(lags, m)
        ax.loglog(lags, m, "o", color=color, ms=4)
        ax.loglog(lags, m[-1] * (lags / lags[-1]) ** (2 * a), color=color, lw=1.2,
                  label=rf"{name}: fit $\hat\alpha={a:.3f}$")
    ax.set(xlabel=r"$\tau_k$ (s)", ylabel=r"$\mathrm{MSD}(\tau_k)$ (toy units)",
           title=rf"The window of $\tau={TAU:,.0f}$ s")
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    ax.grid(color="#eeeeee", lw=0.5, which="both")

    ax = axes[1, 1]
    taus = np.logspace(2.5, np.log10(T_MAX * SUPPORT) - HALF_DECADE - 0.02, 40)
    curves = np.array([fits(t, dur, c) for t in taus])
    dense = np.logspace(2.5, np.log10(T_MAX * SUPPORT) - 0.01, 400)
    ax.plot(dense, theory(dense), color=GRAY, lw=1, label=r"pool, local: $\alpha+\frac{\beta}{2}\,g(y)$")
    ax.plot(taus, curves[:, 0], "o", color=GRAY, ms=3, label="pool: fit over the window")
    ax.plot(taus, curves[:, 1], "o", color=COLORS["group"], ms=3, label="per-lag: fit over the window")
    ax.plot(taus, curves[:, 2], "s", color=PURPLE, ms=3, mfc="none", label="mean of the cohorts: fit over the window")
    ax.axhline(ALPHA, color="k", lw=0.8, ls=":", label=rf"every piece: $\alpha={ALPHA}$")
    ax.axvline(TAU, color="#cccccc", lw=0.8, ls="--")
    ax.set(xscale="log", xlabel=r"$\tau$ (s)", ylabel=r"$\alpha(\tau)$", title="Every window")
    ax.legend(frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    ax.grid(color="#eeeeee", lw=0.5)
    save(fig, OUT / "perlag_scheme.pdf")
    a_pool, a_perlag, a_mean = fits(TAU, dur, c)
    (OUT / "perlag_scheme.json").write_text(json.dumps({
        "alpha": ALPHA, "beta": BETA, "t0": T0, "t_min": T_MIN, "t_max": T_MAX, "tau": TAU, "cohorts": COHORTS,
        "pool": a_pool, "perlag": a_perlag, "chained": chained_fit(TAU, dur, c), "mean": a_mean, "mean_max": float(np.nanmax(curves[:, 2])),
        "jump": float(theory(T_MIN * SUPPORT * 1.0001))}, indent=1))
    print(f"window {TAU:.0f} s: pool {a_pool:.3f}, per-lag {a_perlag:.3f}, mean of cohorts {a_mean:.3f}, "
          f"max of the mean {np.nanmax(curves[:, 2]):.3f}, every piece {ALPHA}")


def chained_main():
    dur, c = pieces()
    lags = window(TAU)
    pool = sets(dur, lags)[0]
    steps = dur[:, None] >= lags[None, 1:] / SUPPORT  # step k -> k+1 on P(tau_{k+1})
    fig, axes = plt.subplots(2, 2, figsize=(9, 7.2), gridspec_kw={"height_ratios": [1, 1.1]})
    grid(axes[0, 0], pool, lags, GRAY, r"Pool: each lag on $P(\tau_k)$")
    ax = axes[0, 1]
    ax.imshow(steps, aspect="auto", cmap=plt.matplotlib.colors.ListedColormap(["#f2f2f2", COLORS["group"]]),
              origin="lower", interpolation="nearest")
    ax.set(title=r"Chained: step $\tau_k\to\tau_{k+1}$ on $P(\tau_{k+1})$", yticks=[], xticks=range(N_LAGS - 1),
           xticklabels=[f"{a:,.0f}→{b:,.0f}" for a, b in zip(lags[:-1], lags[1:])])
    ax.tick_params(axis="x", labelsize=6, rotation=45)
    axes[0, 0].set(ylabel="pieces, short to long (row)")

    ax = axes[1, 0]
    series = ((msd(c, pool, lags), GRAY, "pool"), (chained_msd(dur, c, lags), COLORS["group"], "chained"),
              (cohort_mean(dur, c, lags), PURPLE, "mean of the cohorts"))
    for m, color, name in series:
        a = half_slope(lags, m)
        ax.loglog(lags, m, "o", color=color, ms=4)
        ax.loglog(lags, m[-1] * (lags / lags[-1]) ** (2 * a), color=color, lw=1.2,
                  label=rf"{name}: fit $\hat\alpha={a:.3f}$")
    ax.set(xlabel=r"$\tau_k$ (s)", ylabel=r"$\mathrm{MSD}(\tau_k)$ (toy units)",
           title=rf"The window of $\tau={TAU:,.0f}$ s")
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    ax.grid(color="#eeeeee", lw=0.5, which="both")

    ax = axes[1, 1]
    taus = np.logspace(2.5, np.log10(T_MAX * SUPPORT) - HALF_DECADE - 0.02, 40)
    curves = np.array([[fits(t, dur, c)[0], chained_fit(t, dur, c), fits(t, dur, c)[2]] for t in taus])
    dense = np.logspace(2.5, np.log10(T_MAX * SUPPORT) - 0.01, 400)
    ax.plot(dense, theory(dense), color=GRAY, lw=1, label=r"pool, local: $\alpha+\frac{\beta}{2}\,g(y)$")
    ax.plot(taus, curves[:, 0], "o", color=GRAY, ms=3, label="pool: fit over the window")
    ax.plot(taus, curves[:, 1], "o", color=COLORS["group"], ms=3, label="chained: fit over the window")
    ax.plot(taus, curves[:, 2], "s", color=PURPLE, ms=3, mfc="none", label="mean of the cohorts: fit over the window")
    ax.axhline(ALPHA, color="k", lw=0.8, ls=":", label=rf"every piece: $\alpha={ALPHA}$")
    ax.axvline(TAU, color="#cccccc", lw=0.8, ls="--")
    ax.set(xscale="log", xlabel=r"$\tau$ (s)", ylabel=r"$\alpha(\tau)$", title="Every window")
    ax.legend(frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    ax.grid(color="#eeeeee", lw=0.5)
    save(fig, OUT / "chained_scheme.pdf")
    return float(np.abs(curves[:, 1] - ALPHA).max())


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    main()
    print(f"chained: fit at {TAU:.0f} s {chained_fit(TAU, *pieces()):.6f}, max |fit - alpha| {chained_main():.2e}")
