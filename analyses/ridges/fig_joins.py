"""Small multiples of random joins over a hillshade, with the cells below the height limit and the streams."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

import build
import geo

PAD = 400.0  # m around a join
SEED = 7


def hillshade(z, az=315.0, alt=45.0):
    gy, gx = np.gradient(z, geo.CELL)
    slope, aspect = np.arctan(np.hypot(gx, gy)), np.arctan2(-gx, gy)
    a, h = np.radians(360 - az + 90), np.radians(alt)
    return np.sin(h) * np.cos(slope) + np.cos(h) * np.sin(slope) * np.cos(a - aspect)


def load(region):
    """(curve polylines, joins file, hand raster, x0, y0) of a region from the cache folder."""
    d = build.folder()
    c = dict(np.load(d / f"curves_{region}.npz"))
    curves_ = [c["xy"][i:j] for i, j in zip(c["off"][:-1], c["off"][1:])]
    return curves_, dict(np.load(d / f"joins_{region}.npz")), np.load(d / f"hand_{region}.npy", mmap_mode="r"), float(c["x0"]), float(c["y0"])


HALF = 450.0  # half the side of every panel, m: all panels have the same scale


def draw(ax, curves_, hand, hx0, hy0, a, b, color="#E69F00", valley="#00A6C8", low="#9ECAE1"):
    mid = (np.asarray(a) + np.asarray(b)) / 2
    lo, hi = mid - HALF, mid + HALF
    z, (zx, zy) = geo.window(lo[0], lo[1], hi[0], hi[1])
    ny, nx = z.shape
    ext = (zx - 12.5, zx + 25 * nx - 12.5, zy - 25 * ny + 12.5, zy + 12.5)
    ax.imshow(hillshade(z), cmap="gray", extent=ext, vmin=0, vmax=1.4)
    rr = np.rint((hy0 - (zy - 25 * np.arange(ny))) / 25).astype(int)
    cc = np.rint(((zx + 25 * np.arange(nx)) - hx0) / 25).astype(int)
    h = np.asarray(hand[np.ix_(rr, cc)], float)
    ax.imshow(np.ma.masked_where(h >= 20, np.ones_like(h)), cmap=ListedColormap([low]), alpha=0.45, extent=ext)
    ax.imshow(np.ma.masked_where(h > 0, np.ones_like(h)), cmap=ListedColormap([valley]), alpha=1.0, extent=ext)
    for p in curves_:
        if p[:, 0].max() >= lo[0] and p[:, 0].min() <= hi[0] and p[:, 1].max() >= lo[1] and p[:, 1].min() <= hi[1]:
            ax.plot(p[:, 0], p[:, 1], color=color, lw=1.4)
    ax.plot([a[0], b[0]], [a[1], b[1]], color="red", lw=2.0)
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    return lo, hi


def joins_figure(region, path, n=8, ncols=4, seed=SEED, loaded=None, color="#E69F00", valley="#00A6C8", low="#9ECAE1"):
    """n joins that span the range of lengths, each in a window of the same size, with a shared legend."""
    curves_, J, hand, hx0, hy0 = loaded or load(region)
    order = np.argsort(J["length"])
    quant = np.linspace(0.05, 0.95, n)
    rng = np.random.default_rng(seed)
    pick = [order[min(int(q * len(order)) + int(rng.integers(0, 20)), len(order) - 1)] for q in quant]
    rows = int(np.ceil(n / ncols))
    fig, axs = plt.subplots(rows, ncols, figsize=(6.3, 6.3 / ncols * rows + 0.5))
    for ax, k, letter in zip(axs.ravel(), pick, "abcdefghij"):
        lo, hi = draw(ax, curves_, hand, hx0, hy0, J["a"][k], J["b"][k], color, valley, low)
        ax.text(0.04, 0.96, f"{letter}  {J['length'][k]:.0f} m", transform=ax.transAxes, ha="left", va="top", fontsize=8, fontweight="bold",
                bbox=dict(facecolor="white", alpha=0.8, pad=1.0, edgecolor="none"))
        ax.plot([lo[0] + 40, lo[0] + 240], [lo[1] + 45] * 2, color="white", lw=4, solid_capstyle="butt")
        ax.plot([lo[0] + 40, lo[0] + 240], [lo[1] + 45] * 2, color="black", lw=2, solid_capstyle="butt")
        ax.text(lo[0] + 140, lo[1] + 70, "200 m", ha="center", va="bottom", fontsize=8, bbox=dict(facecolor="white", alpha=0.75, pad=0.6, edgecolor="none"))
    handles = [plt.Line2D([], [], color=color, lw=1.5, label="crest curve"), plt.Line2D([], [], color="red", lw=2, label="join"),
               plt.Rectangle((0, 0), 1, 1, fc=valley, label="stream cell"), plt.Rectangle((0, 0), 1, 1, fc=low, alpha=0.6, label="cell less than 20 m above a stream")]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(pad=0.3, rect=(0, 0.07, 1, 1))
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return len(J["length"])


if __name__ == "__main__":
    import sys
    from pathlib import Path
    out = Path(__file__).parent / "out"
    out.mkdir(exist_ok=True)
    print(joins_figure(sys.argv[1], out / f"joins_{sys.argv[1]}.png"))
