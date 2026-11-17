"""Interior crests of a 5 m trial window (run_massif.py --window) against run B of dem_resolution.py.

    RIDGES_FLATS=blm RIDGES_DEM=5 RIDGES_CACHE=<trial cache> uv run --with numba python check_window.py alps1

Run B (same DEM, margin and constants) used the flood rule for the flats, the trial the rule of Barnes, Lehman and Mulla
(2014): the crest sides and the curve points inside the interior differ only where the two rules route flats differently.
Prints the sides and the points of each that the other lacks.
"""

import sys
from pathlib import Path

import numpy as np

import build
import curves
import dem_resolution as dr
import geo
import ridges
import run_tiled


B_CACHE = Path("/Volumes/SSD_DISANTE/xc-lab/ridges/dem_resolution")  # where dem_resolution.py --stage run wrote run B


def sides(e, box):
    """Bridged crest sides of an edges_<region>.npz inside the box, as rows of 4 integers (end points in half metres)."""
    cell = float(e["cell"])
    h, w = e["shape"]
    x0, y0 = float(e["x0"]), float(e["y0"])
    s = curves.bridge(e["acc"], e["flat"], ridges.cells(curves.BRIDGE_M, cell))
    vid, inv = np.unique(s.astype(np.int64).ravel(), return_inverse=True)
    inv = inv.reshape(-1, 2)
    xy = np.c_[x0 - cell / 2 + cell * (vid % (w + 1)), y0 + cell / 2 - cell * (vid // (w + 1))]
    ends = np.rint(2 * np.c_[xy[inv[:, 0]], xy[inv[:, 1]]]).astype(np.int64)
    ends = np.where((ends[:, :2] <= ends[:, 2:]).all(axis=1)[:, None], ends, ends[:, [2, 3, 0, 1]])
    return ends[dr.inside(xy[inv[:, 0]], box) & dr.inside(xy[inv[:, 1]], box)]


def main(window):
    region, box = dr.WINDOWS[window]
    d = build.folder(5)
    dr.CACHE = B_CACHE
    b = dr.load(window, "B")
    key = lambda e: np.ascontiguousarray(e).view([("", np.int64)] * 4).ravel()
    kt, kb = key(sides(np.load(d / f"edges_{region}.npz"), box)), key(b["ends"])
    c = np.load(d / f"curves_{region}.npz")
    pts = [q for p in np.split(c["xy"], c["off"][1:-1]) for q in dr.clip(p, box)]
    dt, db = dr.nearest(dr.points(pts), dr.points(b["curves"])), dr.nearest(dr.points(b["curves"]), dr.points(pts))
    print(f"{window}: interior crest sides, trial {len(kt)}, run B {len(kb)}; "
          f"missing in B {np.mean(~np.isin(kt, kb)):.2%}, missing in the trial {np.mean(~np.isin(kb, kt)):.2%}")
    print(f"curves: trial {len(pts)} ({dr.length_km(pts):.1f} km), run B {len(b['curves'])} ({dr.length_km(b['curves']):.1f} km); "
          f"trial points > 1 m from B {np.mean(dt > 1):.2%}, > 25 m {np.mean(dt > 25):.2%}; "
          f"B points > 1 m from the trial {np.mean(db > 1):.2%}, > 25 m {np.mean(db > 25):.2%}")


if __name__ == "__main__":
    main(sys.argv[1])
