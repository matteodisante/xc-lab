"""Mountain area from the 5 m DEM against the one from the 25 m DEM, on the 1 km cells of a region.

    RIDGES_FLATS=blm RIDGES_CACHE=<cache> uv run --with numba python compare_masks.py alps [pyrenees]

The 5 m block means are read from the VRT strip by strip (the 5 m DEM is not written), the 25 m ones from the mosaic of
the BD ALTI tiles; both go through mountains.mountain_area_h with the departments' own tiles. Prints the areas, the
cells that differ and the 1 km mean elevations; saves `mask_compare_<region>.npz` (h5, h25, m5, m25) in the cache.
A 5 m block starts 10 m east and 10 m north of a 25 m one (cell centres on the tile corner): the cells are compared by index.
"""

import sys
import time

import numpy as np
import rasterio

import dem_resolution as dr
import geo
import mountains
import run_tiled


def means5(region):
    h, w, (x0, y0) = run_tiled.box_shape(region, 5)
    r0, c0 = run_tiled.origin5(x0, y0)
    with rasterio.open(dr.DEM5) as src:
        return run_tiled.block_means(lambda a, b, c, d: run_tiled.read_dem5(src, r0 + a, r0 + b, c0 + c, c0 + d), h, w, 5.0)


def means25(region):
    z, _ = geo.mosaic(region)
    return mountains.block_mean(z, mountains.block_cells(25.0))


def compare(region):
    t = time.time()
    own = geo.own_cells(region)
    h5 = means5(region)
    print(f"{region}: 5 m block means in {time.time() - t:.0f} s", flush=True)
    h25 = means25(region)
    m5, m25 = mountains.mountain_area_h(h5.copy(), own), mountains.mountain_area_h(h25.copy(), own)
    np.savez_compressed(geo.CACHE / f"mask_compare_{region}.npz", h5=h5, h25=h25, m5=m5, m25=m25)
    diff = m5 != m25
    both = np.isfinite(h5) & np.isfinite(h25) & own
    d = (h5 - h25)[both]
    print(f"{region}: mountain area {m5.sum()} km2 at 5 m, {m25.sum()} km2 at 25 m; cells that differ {diff.sum()} "
          f"({diff.sum() / (m5 | m25).sum():.2%} of the union): {(m5 & ~m25).sum()} only at 5 m, {(~m5 & m25).sum()} only at 25 m")
    print(f"{region}: 1 km mean elevation, 5 m minus 25 m over {both.sum()} cells: mean {d.mean():.2f} m, "
          f"|difference| 95th percentile {np.percentile(np.abs(d), 95):.1f} m, max {np.abs(d).max():.0f} m; "
          f"cells with data only at 5 m {(np.isfinite(h5) & ~np.isfinite(h25) & own).sum()}, only at 25 m {(~np.isfinite(h5) & np.isfinite(h25) & own).sum()}")
    x0, x1, y0, y1 = geo.region_box(region)
    nodata5 = ~np.isfinite(h5) & np.isfinite(h25)
    print(f"{region}: of the {diff.sum()} cells that differ, {(diff & nodata5).sum()} have no 5 m data (outside the 5 m mosaic), "
          f"{(diff & ~nodata5).sum()} have data in both ({(diff & ~nodata5 & m5).sum()} only at 5 m, {(diff & ~nodata5 & m25).sum()} only at 25 m)")
    for name, (reg, (xa, ya, xb, yb)) in dr.WINDOWS.items():  # the interiors of dem_resolution.py, Lambert-93 m
        if reg == region:
            i, j = slice(int((y1 * 1000 - yb) // 1000), int((y1 * 1000 - ya) // 1000)), slice(int((xa - x0 * 1000) // 1000), int((xb - x0 * 1000) // 1000))
            print(f"{region}: window {name} ({(xb - xa) // 1000} x {(yb - ya) // 1000} km): cells that differ {diff[i, j].sum()}, "
                  f"mountain {m5[i, j].sum()} at 5 m and {m25[i, j].sum()} at 25 m")


if __name__ == "__main__":
    for r in sys.argv[1:]:
        compare(r)
