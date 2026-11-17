"""The tiled routing equals the in-memory one (ridges.py, FLATS = "blm") on synthetic DEMs and a real window."""

import numpy as np
import pytest
from scipy import ndimage as ndi

import geo
import ridges
import tiled


def _acc(d, order, w):
    acc = np.ones(d.size, np.int64)
    for i in order[::-1]:
        if d[i] >= 0:
            acc[i + ridges.DR[d[i]] * w + ridges.DC[d[i]]] += acc[i]
    return acc


def reference(z, valley_km2, cell, xsec):
    w = z.shape[1]
    filled, d, order = ridges.route(z, "blm")
    acc = _acc(d, order, w)
    sz = filled.copy()
    lid, down = ridges._links(sz, d, order, w, ridges.valley_cells(valley_km2, cell))
    tin, tout = ridges._euler(down)
    return dict(filled=filled, d=d, acc=acc, lid=lid, sz=sz, down=down,
                edges=ridges.crest_edges(z, lid, sz, tin, tout, xsec))


def run_tiled(z, T, folder, valley_km2, cell, xsec):
    run = tiled.Run(folder, z.shape, T, cell=cell, valley_km2=valley_km2, xsec=xsec, keep=True, log=lambda m: None)
    run.z.write(0, 0, z)
    tiled.fill(run)
    tiled.dirs(run)
    tiled.acc(run)
    tiled.links(run)
    tiled.edges(run, 0.0, 0.0)
    return run


def check(z, T, folder, valley_km2=0.01, cell=25.0, xsec=6, ref=None):
    ref = ref or reference(z, valley_km2, cell, xsec)
    run = run_tiled(z, T, folder, valley_km2, cell, xsec)
    G = lambda n, t: run.grid(n, t).array().ravel()
    assert np.array_equal(G("filled", np.float32), ref["filled"], equal_nan=True)
    assert np.array_equal(G("d", np.int8), ref["d"])
    assert np.array_equal(G("acc", np.int32).astype(np.int64), np.minimum(ref["acc"], tiled.INT32_MAX))
    lid = G("lid", np.int32)
    pairs = np.unique(lid.astype(np.int64) << 32 | ref["lid"].astype(np.int64))
    assert pairs.size == np.unique(lid).size == np.unique(ref["lid"]).size  # the same partition into links
    link = dict(zip(pairs >> 32, pairs & (2**32 - 1)))
    down = np.load(run.dir / "valley.npz")["down"]
    assert all(ref["down"][link[a]] == link[down[a]] for a in range(1, down.size))  # the same link forest
    assert np.array_equal(G("sz", np.float32), ref["sz"], equal_nan=True)
    e = np.load(run.out / "edges_tiled.npz")
    for k, a in zip(("acc", "flat"), ref["edges"]):
        assert np.array_equal(e[k], a), k
    return run, ref


def random_dem(h, w, seed, quant=0.0, holes=3, pits=6):
    rng = np.random.default_rng(seed)
    z = ndi.gaussian_filter(rng.normal(size=(h, w)), 5) * 400 + 0.3 * np.add.outer(np.arange(h), np.arange(w))
    z += ndi.gaussian_filter(rng.normal(size=(h, w)), 1.5) * 30
    for _ in range(pits):
        r, c = rng.integers(10, h - 10), rng.integers(10, w - 10)
        z[r - 3:r + 3, c - 4:c + 4] -= 60
    if quant:
        z = np.round(z / quant) * quant  # many flats
    for _ in range(holes):
        r, c = rng.integers(0, h - 12), rng.integers(0, w - 12)
        z[r:r + rng.integers(2, 12), c:c + rng.integers(2, 12)] = np.nan
    return z.astype(np.float32)


@pytest.mark.parametrize("T", [20, 33, 64])
def test_random_terrain(tmp_path, T):
    check(random_dem(110, 131, 1), T, tmp_path)


@pytest.mark.parametrize("T", [20, 37])
def test_flats_and_holes(tmp_path, T):
    check(random_dem(120, 127, 2, quant=25.0, holes=6), T, tmp_path)


def test_lake_over_four_tiles_and_flat_at_a_corner(tmp_path):
    z = random_dem(100, 100, 3, quant=0.0, holes=0, pits=0)
    z[30:70, 30:70] = z[30:70, 30:70].mean() - 20  # a flat lake over the corner of four 50-cell tiles
    z[10:25, 45:60] = z[10:25, 45:60].max() + 5  # a plateau across the edge of two tiles
    run, ref = check(z, 50, tmp_path)
    assert (np.load(run.dir / "crossing.npz")["cells"] > 1000).any()


def test_tile_of_one_cell_wide_remainder(tmp_path):
    check(random_dem(61, 62, 4, quant=10.0), 30, tmp_path)  # tiles of 1 and 2 cells at the end


def test_edge_of_data(tmp_path):
    z = random_dem(90, 90, 5)
    z[:, 60:] = np.nan  # no data on a third of the grid
    check(z, 25, tmp_path)


def test_real_window(tmp_path):
    if not geo.DEM.exists():
        pytest.skip("no disk")
    z, _ = geo.window(946000, 6513000, 958000, 6525000)  # about 480 x 480 cells of the Alps
    check(z, 150, tmp_path, valley_km2=ridges.VALLEY_KM2)


def test_rerun_after_a_crash_gives_the_same(tmp_path):
    z = random_dem(100, 90, 6, quant=15.0)
    run, ref = check(z, 32, tmp_path)
    for stage in ("fill2", "dirs", "acc1", "lid2", "edges"):  # forget some tiles and the stages after them
        for f in sorted(run.sdir(stage).glob("[0-9]*[0-9].npz"))[::3]:
            f.unlink()
    for f in ("incoming.npz", "exits.npz", "dirs_patched"):
        (run.dir / f).unlink()
    tiled.dirs(run), tiled.acc(run), tiled.links(run), tiled.edges(run, 0.0, 0.0)
    e = np.load(run.out / "edges_tiled.npz")
    assert np.array_equal(e["acc"], ref["edges"][0]) and np.array_equal(e["flat"], ref["edges"][1])


def test_valleys_only_after_links_without_keep(tmp_path):
    z = random_dem(80, 80, 7)
    run = tiled.Run(tmp_path, z.shape, 30, valley_km2=0.01, log=lambda m: None)
    run.z.write(0, 0, z)
    tiled.fill(run), tiled.dirs(run), tiled.acc(run), tiled.links(run)
    assert not any((tmp_path / f"{n}.npy").exists() for n in ("filled", "acc", "d", "fl_lab"))
    assert (tmp_path / "lid.npy").exists() and (tmp_path / "sz.npy").exists()


def test_one_tile_and_tiles_of_the_whole_height(tmp_path):
    z = random_dem(70, 75, 8, quant=20.0)
    check(z, 200, tmp_path / "a")
    check(z, 7, tmp_path / "b", xsec=3)  # tiles narrower than the halo
