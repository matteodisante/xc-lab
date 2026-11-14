"""Small multiples of random joins over a hillshade, with the cells below the height limit and the streams."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

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


def draw(ax, curves_, hand, hx0, hy0, a, b):
    lo, hi = np.minimum(a, b) - PAD, np.maximum(a, b) + PAD
    z, (zx, zy) = geo.window(lo[0], lo[1], hi[0], hi[1])
    ny, nx = z.shape
    ext = (zx - 12.5, zx + 25 * nx - 12.5, zy - 25 * ny + 12.5, zy + 12.5)
    ax.imshow(hillshade(z), cmap="gray", extent=ext, vmin=0, vmax=1.4)
    rr = np.rint((hy0 - (zy - 25 * np.arange(ny))) / 25).astype(int)
    cc = np.rint(((zx + 25 * np.arange(nx)) - hx0) / 25).astype(int)
    h = np.asarray(hand[np.ix_(rr, cc)], float)
    ax.imshow(np.ma.masked_where(h >= 20, np.ones_like(h)), cmap="Blues", alpha=0.35, extent=ext, vmin=0, vmax=1.5)
    ax.imshow(np.ma.masked_where(h > 0, np.ones_like(h)), cmap="winter", alpha=0.9, extent=ext, vmin=0, vmax=1)
    for p in curves_:
        if p[:, 0].max() >= lo[0] and p[:, 0].min() <= hi[0] and p[:, 1].max() >= lo[1] and p[:, 1].min() <= hi[1]:
            ax.plot(p[:, 0], p[:, 1], color="#e8a317", lw=1.0)
    ax.plot([a[0], b[0]], [a[1], b[1]], color="red", lw=1.6)
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_xticks([])
    ax.set_yticks([])


def joins_figure(region, path, n=30, ncols=6, seed=SEED, loaded=None):
    curves_, J, hand, hx0, hy0 = loaded or load(region)
    pick = np.random.default_rng(seed).choice(len(J["length"]), min(n, len(J["length"])), replace=False)
    fig, axs = plt.subplots(int(np.ceil(len(pick) / ncols)), ncols, figsize=(2.4 * ncols, 2.4 * np.ceil(len(pick) / ncols)))
    for ax in axs.ravel():
        ax.axis("off")
    for ax, k in zip(axs.ravel(), pick):
        ax.axis("on")
        draw(ax, curves_, hand, hx0, hy0, J["a"][k], J["b"][k])
        ax.set_title(f"{J['length'][k]:.0f} m", fontsize=8, pad=2)
    fig.tight_layout(pad=0.3)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return len(J["length"])


if __name__ == "__main__":
    import sys
    from pathlib import Path
    out = Path(__file__).parent / "out"
    out.mkdir(exist_ok=True)
    print(joins_figure(sys.argv[1], out / f"joins_{sys.argv[1]}.png"))
