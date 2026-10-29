"""H_loc(tau) of populations on one axis, with an inset of MSD(tau): the drawing shared by the
documents of vs-solo.

A series is a dict of the form tamsd.summarize gives (lags, msd, H_loc, H_loc_band, flights) with a
label, a colour and a line style.
"""

import numpy as np

INSET = (0.08, 0.08, 0.36, 0.36)  # left, bottom, width, height of the inset, in axes fractions
RAMP = ("#f7c59f", "#f2a065", "#eb6834", "#c24a1d", "#8c2f0f")  # orange, light to dark: more pilots


def series(result, label, color, ls="-"):
    """A series from one population of a result of tamsd.summarize (or longlags.curve)."""
    return {**result, "label": label, "color": color, "ls": ls}


def plot(ax, curves, ylim=None, inset=INSET):
    """H_loc against the lag of every series in `curves`, with the 5-95% band, and the MSD of the
    same series, log-log, in an inset. The legend holds the series that have a label."""
    small = ax.inset_axes(inset) if inset else None
    for c in curves:
        lags = np.asarray(c["lags"])
        ax.fill_between(lags, *c["H_loc_band"], color=c["color"], alpha=0.15, lw=0)
        ax.plot(lags, c["H_loc"], color=c["color"], ls=c["ls"], lw=1.3, label=c.get("label"))
        if small is not None:
            small.plot(lags, c["msd"], color=c["color"], ls=c["ls"], lw=1)
    low = min(np.min(c["H_loc_band"][0]) for c in curves)
    ax.set(xscale="log", ylim=ylim or (min(0.45, np.floor(low * 20) / 20), 1.0), xlabel=r"$\tau$ (s)", ylabel=r"$H_{\rm loc}(\tau)$")
    ax.axhline(0.5, color="#999999", lw=0.6, ls=":")
    ax.grid(color="#eeeeee", lw=0.5)
    if small is not None:
        small.set(xscale="log", yscale="log")
        small.set_title(r"MSD (m$^2$)", fontsize=6, pad=1)
        small.tick_params(labelsize=5.5, length=2, pad=1)
        small.patch.set_alpha(0.85)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    return small
