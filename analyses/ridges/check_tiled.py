"""Compares a tiled run (run_tiled.py) with the in-memory routing of ridges.py on the same DEM (FLATS = "blm").

    RIDGES_FLATS=blm RIDGES_CACHE=<cache> uv run --with numba python check_tiled.py <region> --dem 25 --tile 400

Prints the number of cells or sides that differ for filled, d, acc, the partition into links, the link forest, sz,
hand and the crest sides (acc, flat, vid, near). The in-memory arrays of the Alps take about 3 GB.
"""

import argparse
import resource

import numpy as np
from numba import njit

import build
import geo
import ridges
import tiled

STRIP = 1000  # rows compared at a time


@njit(cache=True)
def count_cells(d, order, w):
    acc = np.ones(d.size, np.int32)
    for m in range(order.size - 1, -1, -1):
        i = order[m]
        if d[i] >= 0:
            acc[i + ridges.DR[d[i]] * w + ridges.DC[d[i]]] += acc[i]
    return acc


def diff(grid, ref, dtype, nan=False):
    """Cells of the disk raster that differ from the flat reference array (NaN equal to NaN if nan)."""
    H, W = grid.shape
    n = 0
    for r in range(0, H, STRIP):
        a = grid.read(r, min(H, r + STRIP), 0, W)
        b = ref[r * W:min(H, r + STRIP) * W].reshape(a.shape)
        n += int((a != b).sum() - (np.isnan(a) & np.isnan(b)).sum() if nan else (a != b).sum())
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("region", choices=list(geo.REGIONS))
    ap.add_argument("--dem", type=int, default=25)
    ap.add_argument("--tile", type=int, default=400)
    a = ap.parse_args()
    z, (x0, y0) = geo.mosaic(a.region)
    H, W = z.shape
    folder = geo.CACHE / f"tiled_{a.region}_{a.dem}m_T{a.tile}"
    run = tiled.Run(folder, z.shape, a.tile, xsec=ridges.XSEC_CELLS, out=build.folder(), name=a.region)
    assert np.array_equal(run.z.array(), z, equal_nan=True)
    out = {}
    filled, d, order = ridges.route(z, "blm")
    out["filled"] = diff(run.grid("filled", np.float32), filled, np.float32, nan=True)
    out["d"] = diff(run.grid("d", np.int8), d, np.int8)
    acc = count_cells(d, order, W)
    out["acc"] = diff(run.grid("acc", np.int32), acc, np.int32)
    del acc
    lid, down = ridges._links(filled, d, order, W, ridges.valley_cells(ridges.VALLEY_KM2, ridges.CELL))
    del d, order
    sz = filled
    out["sz"] = diff(run.grid("sz", np.float32), sz, np.float32, nan=True)
    pairs, Lid = np.empty(0, np.int64), run.grid("lid", np.int32)
    for r in range(0, H, STRIP):
        t = Lid.read(r, min(H, r + STRIP), 0, W).ravel().astype(np.int64)
        pairs = np.unique(np.r_[pairs, t << 32 | lid[r * W:min(H, r + STRIP) * W].astype(np.int64)])
    link = dict(zip(pairs >> 32, pairs & (2**32 - 1)))
    out["links_not_one_to_one"] = int(pairs.size - len(link)) + int(pairs.size - np.unique(pairs & (2**32 - 1)).size)
    tdown = np.load(run.dir / "valley.npz")["down"]
    out["forest"] = sum(down[link[k]] != link[tdown[k]] for k in range(1, tdown.size))
    out["n_links"] = (int(tdown.size - 1), int(down.size - 1))
    tin, tout = ridges._euler(down)
    ref_edges = ridges.crest_edges(z, lid, sz, tin, tout)
    hand = run.grid("hand", np.int16, run.out / f"hand_{a.region}.npy")
    n = 0
    for r in range(0, H, STRIP):
        s = slice(r * W, min(H, r + STRIP) * W)
        zz, ll, ss = z[r:r + STRIP], lid[s].reshape(-1, W), sz[s].reshape(-1, W)
        ref = np.where(ll > 0, np.clip(np.rint(zz - ss), -32767, 32767), build.HAND_NONE)
        n += int((hand.read(r, r + zz.shape[0], 0, W) != ref).sum())
    out["hand"] = n
    del lid, sz, filled
    vid = np.unique(np.r_[ref_edges[0].ravel(), ref_edges[1].ravel()])
    import curves
    near = curves.near_nodata(z, vid // (W + 1), vid % (W + 1), curves.EDGE_CELLS)
    e = np.load(run.out / f"edges_{a.region}.npz")
    for k, ref in (("acc", ref_edges[0]), ("flat", ref_edges[1]), ("vid", vid), ("near", near)):
        out["edges_" + k] = int(abs(len(e[k]) - len(ref))) if e[k].shape != ref.shape else int((e[k] != ref).sum())
    out["edges_x0y0shape"] = int(e["x0"] != x0) + int(e["y0"] != y0) + int((e["shape"] != z.shape).any())
    out["sizes"] = dict(acc=len(ref_edges[0]), flat=len(ref_edges[1]))
    for k, v in out.items():
        print(f"{k:24s} {v}")
    print(f"peak RAM {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30:.2f} GB")


if __name__ == "__main__":
    main()
