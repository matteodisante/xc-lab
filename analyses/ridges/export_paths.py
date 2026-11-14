"""Every joined crest path of a region, for the intermittent model: paths_all_<region>.npz in the crest folder.

Same joins, trees and paths as hgeo.py (same order: row i of tamsd_rows_<region>_paths.npz is path i here), so the
model and H_geo use the same population. Stored points are float32 offsets from `origin` (Lambert-93).

    uv run --with numba python export_paths.py alps [pyrenees]
"""

import sys

import numpy as np

import build
import joined
import tamsd


def export(region):
    d = build.folder()
    c = np.load(d / f"curves_{region}.npz")
    hand = np.load(d / f"hand_{region}.npy", mmap_mode="r")
    net = joined.Net(c["xy"], c["off"], c["jc"], c["jxy"])
    joined.make_joins(net, hand, float(c["x0"]), float(c["y0"]))
    emap = joined.edge_map(net)
    origin = np.array([float(c["x0"]), float(c["y0"])])
    pts, tree = [], []
    for t in joined.trees(net):
        for j in t["tips"]:
            p, _ = joined.path_polyline(net, t, j, emap)
            pts.append((p - origin).astype(np.float32))
            tree.append(t["id"])
    ref = np.load(d / f"tamsd_rows_{region}_paths.npz")
    length = np.array([(len(p) - 1) * tamsd.STEP for p in pts])
    assert np.allclose(length, ref["length"]) and (np.array(tree) == ref["group"]).all(), "paths differ from the TAMSD rows"
    np.savez_compressed(d / f"paths_all_{region}.npz", xy=np.concatenate(pts), off=np.r_[0, np.cumsum([len(p) for p in pts])],
                        tree=np.array(tree), origin=origin, length=length)
    print(region, len(pts), "paths", flush=True)


if __name__ == "__main__":
    for r in sys.argv[1:]:
        export(r)
