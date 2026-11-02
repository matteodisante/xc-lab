"""H_loc(tau) of populations on one axis, with an inset of MSD(tau): the drawing shared by the
documents of vs-solo.

A series is a dict of the form tamsd.summarize gives (lags, msd, H_loc, H_loc_band, flights) with a
label, a colour and a line style.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

INSET = (0.08, 0.08, 0.36, 0.36)  # left, bottom, width, height of the inset, in axes fractions
RAMP = ("#f7c59f", "#f2a065", "#eb6834", "#c24a1d", "#8c2f0f")  # orange, light to dark: more pilots
CROWDED = 6  # more curves than this: the legend goes beside the axes


def series(result, label, color, ls="-"):
    """A series from one population of a result of tamsd.summarize (or longlags.curve)."""
    return {**result, "label": label, "color": color, "ls": ls}


def plot_msd(ax, curves):
    """MSD against the lag of every series in `curves`, log-log, with the 5-95% band; the legend holds the
    series that have a label."""
    for c in curves:
        lags = np.asarray(c["lags"])
        ax.fill_between(lags, *c["msd_band"], color=c["color"], alpha=0.15, lw=0)
        ax.plot(lags, c["msd"], color=c["color"], ls=c["ls"], lw=1.3, label=c.get("label"))
    ax.set(xscale="log", yscale="log", xlabel=r"$\tau$ (s)", ylabel=r"MSD (m$^2$)")
    ax.grid(color="#eeeeee", lw=0.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def figures(curves, stem, title=None):
    """Two figures of `curves`, `<stem>_msd` and `<stem>_hloc` (pdf and png): plot_msd, and plot without inset."""
    low = min(np.min(c["H_loc_band"][0]) for c in curves)
    high = max(np.max(c["H_loc_band"][1]) for c in curves)
    zoom = (np.floor((low - 0.01) * 50) / 50, high + 0.04)
    for suffix, draw in (("msd", plot_msd), ("hloc", lambda ax, c: plot(ax, c, zoom, inset=None))):
        fig, ax = plt.subplots(figsize=(4.2, 3.3), constrained_layout=True)
        draw(ax, curves)
        if len(curves) > CROWDED:
            ax.legend(frameon=False, fontsize=6.5, loc="upper left", bbox_to_anchor=(1.0, 1.0))
        else:
            ax.legend(frameon=False, fontsize=6.5, loc="upper left" if suffix == "msd" else "best")
        if title:
            ax.set_title(title, fontsize=8)
        path = Path(f"{stem}_{suffix}.pdf")
        fig.savefig(path, bbox_inches="tight")
        fig.savefig(path.with_suffix(".png"), dpi=200, bbox_inches="tight")
        plt.close(fig)


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


def delta_figure(left, right, path, title=None):
    """A comparison over every lag (`path`.pdf and .png): left, H_loc(tau) of the series `left` (no bands); right,
    the differences `right` (dicts: lags, value, label, color, and optionally band, ls) with their bands, and 0."""
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 2.7), constrained_layout=True)
    for c in left:
        axes[0].plot(c["lags"], c["H_loc"], color=c["color"], ls=c.get("ls", "-"), lw=1.4, label=c["label"])
    axes[0].set(xscale="log", xlabel=r"$\tau$ (s)", ylabel=r"$H_{\rm loc}(\tau)$")
    for d in right:
        if d.get("band") is not None:
            axes[1].fill_between(d["lags"], *d["band"], color=d["color"], alpha=0.18, lw=0)
        axes[1].plot(d["lags"], d["value"], color=d["color"], ls=d.get("ls", "-"), lw=d.get("lw", 1.4), label=d["label"])
    axes[1].axhline(0, color="#999999", lw=0.8)
    axes[1].set(xscale="log", xlabel=r"$\tau$ (s)", ylabel=r"$\Delta H_{\rm loc}(\tau)$")
    for ax, drawn in zip(axes, (left, right)):
        if len(drawn) > CROWDED:
            ax.legend(frameon=False, fontsize=6.5, loc="upper left", bbox_to_anchor=(1.0, 1.0))
        else:
            ax.legend(frameon=False, fontsize=6.5, loc="best")
        ax.grid(color="#eeeeee", lw=0.5)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    if title:
        fig.suptitle(title, fontsize=8)
    path = Path(path).with_suffix(".pdf")
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
