"""Two maps of France: where the pieces of solo flights and of group time are.

The middle of every piece of tamsd.py (paragliders) is placed on a grid of CELL_DEG degrees, smoothed by a
Gaussian of SIGMA cells and divided by the area of a cell, so a pixel is the pieces per 1000 km^2. Both
maps share the colour scale. The zones of regions.py are drawn on them. The coast and borders are those of
the basemap of xc-viewer (read only).

    uv run python maps.py [--run NAME]    # after tamsd.py and regions.py; writes out/<run>/maps.pdf
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import PolyCollection
from matplotlib.colors import LogNorm
from scipy.ndimage import gaussian_filter

import tamsd
import zones
from blocks import with_takeoffs
from tamsd import CACHE, LABELS, OUT

BASEMAP = Path(__file__).resolve().parents[2].parent / "xc-viewer/src/xc_viewer/assets/basemap.json"
EXTENT = (-5.5, 41.0, 10.0, 51.5)  # west, south, east, north, degrees
CELL_DEG = 0.1
SIGMA = 1.5  # cells
VMIN = 0.5  # pieces per 1000 km^2 under which a pixel is blank
PANELS = (("solo", "solo flights S"), ("group", "group time G"))


def density(lat, lon):
    """Pieces per 1000 km^2 on the grid of EXTENT, smoothed; the grid edges and the cell areas."""
    west, south, east, north = EXTENT
    xe, ye = np.arange(west, east + CELL_DEG / 2, CELL_DEG), np.arange(south, north + CELL_DEG / 2, CELL_DEG)
    count, _, _ = np.histogram2d(lon, lat, bins=(xe, ye))
    mid = (ye[:-1] + ye[1:]) / 2
    area = (111.32 * CELL_DEG * np.cos(np.radians(mid))) * (111.2 * CELL_DEG)  # km^2 of a cell, by latitude
    return xe, ye, gaussian_filter(count, SIGMA) / area[None, :] * 1000


def corners(z, km):
    """Longitude and latitude of the corners of a zone cell."""
    x = np.array([0, 1, 1, 0]) * km + z["x0_km"]
    y = np.array([0, 0, 1, 1]) * km + z["y0_km"]
    return 2.5 + x / (111.32 * np.cos(np.radians(46.5))), 46.5 + y / 111.2


if __name__ == "__main__":
    flights = with_takeoffs(pd.read_parquet(LABELS / "flights.parquet"))
    p = tamsd.read_pieces(CACHE / "pieces.npz")["para"]
    lat, lon = zones.lonlat(p, flights.set_index("flight")[["lat0", "lon0"]])
    panels = json.loads(BASEMAP.read_text())["panels"]
    rings = panels["world"]["rings"] + panels["france"]["rings"]
    regions = json.loads((OUT / "regions.json").read_text())
    grids = {k: density(lat[p["population"] == k], lon[p["population"] == k]) for k, _ in PANELS}
    top = max(g[2].max() for g in grids.values())
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 4.3), constrained_layout=True)
    for ax, (k, title) in zip(axes, PANELS):
        xe, ye, d = grids[k]
        ax.add_collection(PolyCollection([np.asarray(r) for r in rings], facecolors="#f1f0ec", edgecolors="#b5b3ac", linewidths=0.3))
        image = ax.pcolormesh(xe, ye, np.where(d >= VMIN, d, np.nan).T, norm=LogNorm(vmin=VMIN, vmax=top), cmap="magma_r", rasterized=True)
        for z in regions["zones"]:
            ax.fill(*corners(z, regions["cell_km"]), fill=False, edgecolor="#1a7f5a", lw=0.8)
        n = int((p["population"] == k).sum())
        ax.set(xlim=EXTENT[::2], ylim=EXTENT[1::2], title=f"{title}: {n:,} pieces", xticks=[], yticks=[])
        ax.title.set_fontsize(8.5)
        ax.set_aspect(1 / np.cos(np.radians(46.3)))
    fig.colorbar(image, ax=axes, shrink=0.7, label="pieces per 1000 km$^2$")
    fig.savefig(OUT / "maps.pdf", bbox_inches="tight")
    fig.savefig(OUT / "maps.png", dpi=200, bbox_inches="tight")
