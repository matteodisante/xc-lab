"""Grid constants in metres on the 25 and the 5 m grid, side ids beyond 2^31, the 5 m reader and the driver."""

import subprocess
import sys

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

import build
import curves
import dem_resolution as dr
import geo
import joined
import mountains
import ridges
import run_massif
import run_tiled

X0, Y0 = -1000.0, 2000.0


def test_every_grid_constant_goes_back_to_the_25_m_cells():
    assert (ridges.XSEC_CELLS, curves.BRIDGE_EDGES, curves.PRUNE_EDGES, curves.EDGE_CELLS) == (6, 40, 4, 4)
    assert mountains.BLOCK == 40 and mountains.block_cells(25.0) == 40
    assert ridges.valley_cells(ridges.VALLEY_KM2, 25.0) == 800


def test_the_5_m_grid_has_the_same_lengths():
    c = 5.0
    assert (ridges.cells(ridges.XSEC_M, c), ridges.cells(curves.BRIDGE_M, c), ridges.cells(curves.PRUNE_M, c),
            ridges.cells(curves.EDGE_M, c)) == (30, 200, 20, 20)
    assert mountains.block_cells(c) == 200 and ridges.valley_cells(ridges.VALLEY_KM2, c) == 20000
    assert ridges.cells(1.0, 5.0) == 1  # never zero cells


def test_parameter_folders_keep_the_25_m_name_and_tag_the_5_m_one(tmp_path, monkeypatch):
    monkeypatch.setattr(geo, "CACHE", tmp_path)
    monkeypatch.setattr(ridges, "FLATS", "blm")
    n25, n5 = build.folder(25).name, build.folder(5).name
    tail = "_defl135-200_min1000_step5_smooth100_join1000-45-200_flats-blm"
    assert n25 == "bdalti25_valley0.5_xsec6-10_hand20_gap40_prune4" + tail
    assert n5 == "rgealti5_valley0.5_xsec30-10_hand20_gap200_prune20" + tail


def test_parameters_file_lists_metres_and_cells(tmp_path):
    build.write_parameters(tmp_path, 5.0)
    t = (tmp_path / "PARAMETERS.md").read_text()
    assert "RGE ALTI 5 m" in t and "1000 m (200 sides)" in t and "150 m (30 cells)" in t and "200 x 200" in t


# --- joins on the 5 m grid -----------------------------------------------------------------------------------

def line(x0, y0, x1, y1):
    n = int(np.hypot(x1 - x0, y1 - y0) // 5)
    t = np.linspace(0, 1, n + 1)[:, None]
    return np.array([x0, y0]) + t * np.array([x1 - x0, y1 - y0])


def gap_net():
    a, b = line(0, 0, 2000, 0), line(2300, 0, 4500, 0)
    return joined.Net(np.concatenate([a, b]), np.r_[0, len(a), len(a) + len(b)], np.zeros((0, 2), int), np.zeros((0, 2)))


def raster5(valley=None, stream=None):
    h = np.full((1000, 1500), 100, np.int16)  # 5 m cells: x = X0 + 5 c, y = Y0 - 5 r
    if valley:
        h[:, valley[0]:valley[1]] = 0
    if stream is not None:
        h[:, stream] = 0
    return h


def test_cells_crossed_scales_with_the_cell():
    a, b = np.array([0.0, 0.0]), np.array([100.0, 0.0])
    assert len(joined.cells_crossed(a, b, 0.0, 0.0, 25.0)[1]) == 5
    r, c = joined.cells_crossed(a, b, 0.0, 0.0, 5.0)
    assert set(c) == set(range(21)) and set(r) == {0}


def test_join_across_a_5_m_valley_is_refused_and_across_a_hill_accepted():
    assert joined.make_joins(gap_net(), raster5(), X0, Y0, cell=5.0) == 1
    assert joined.make_joins(gap_net(), raster5(valley=(610, 650)), X0, Y0, cell=5.0) == 0  # x = 2050 .. 2250 m
    assert joined.make_joins(gap_net(), raster5(stream=630), X0, Y0, cell=5.0) == 0  # a one-cell stream, 5 m wide


def test_a_5_m_raster_read_as_25_m_would_misplace_the_crossing():
    # the default cell is the 25 m one: the same raster then puts the stream far from the join, which goes through
    assert joined.make_joins(gap_net(), raster5(stream=630), X0, Y0) == 1


def test_cell_of_a_curves_file(tmp_path):
    np.savez(tmp_path / "a.npz", xy=np.zeros(1))
    np.savez(tmp_path / "b.npz", xy=np.zeros(1), cell=5.0)
    assert joined.cell_of(np.load(tmp_path / "a.npz")) == 25.0
    assert joined.cell_of(np.load(tmp_path / "b.npz")) == 5.0


# --- side ids beyond 2^31 --------------------------------------------------------------------------------------

def test_bridge_keeps_64_bit_ids():
    base = 2**32 + 12345
    acc = np.array([[base, base + 1], [base + 1, base + 2]], np.int64)
    flat = np.array([[base + 2, base + 3], [base + 3, base + 4], [base + 4, base]], np.int64)  # closes a loop through acc
    out = curves.bridge(acc, flat, 10)
    assert out.dtype == np.int64 and out.max() == base + 4 and len(out) == 5


def test_bridge_keeps_32_bit_ids_for_small_grids():
    acc = np.array([[1, 2], [2, 3]], np.int32)
    flat = np.array([[3, 4], [4, 1]], np.int32)
    assert curves.bridge(acc, flat, 10).dtype == np.int32


def test_curves_from_a_grid_with_more_than_2_pow_31_corners(tmp_path, monkeypatch):
    monkeypatch.setattr(build, "MASK_SHARE", 0.5)
    H, W, r = 3_000_000, 2000, 2_999_000  # (H + 1)(W + 1) = 6e9 corners; a crest of 1999 sides on row r
    ids = r * (W + 1) + np.arange(W)
    acc = np.c_[ids, ids + 1].astype(np.int64)
    vid = np.unique(acc)
    x0, y0, cell = 810000.0, 6500000.0, 5.0
    np.savez(tmp_path / "edges_t.npz", acc=acc, flat=np.zeros((0, 2), np.int64), vid=vid, near=np.zeros(len(vid), bool),
             shape=np.array([H, W]), x0=x0, y0=y0, cell=cell)
    np.savez(tmp_path / "mask_t.npz", mask=np.ones((4, 4), bool), left=x0 - 2.5, top=y0 + 2.5, cell=1000.0)
    build.build_curves("t", tmp_path)
    c = np.load(tmp_path / "curves_t.npz")
    assert len(c["off"]) == 2 and float(c["cell"]) == 5.0
    xy = c["xy"]
    assert np.allclose(xy[:, 1], y0 + 2.5 - cell * r, atol=1e-3)  # the row survives the ids
    assert abs(np.ptp(xy[:, 0]) - cell * (W - 1)) < 12  # the 1999 sides, shortened by the smoothing at the ends only


# --- the 5 m reader ----------------------------------------------------------------------------------------------

@pytest.fixture
def tiff(tmp_path):
    z = np.arange(60 * 80, dtype=np.float32).reshape(60, 80)
    z[10:20, 10:20] = run_tiled.NODATA5
    f = tmp_path / "m.tif"
    with rasterio.open(f, "w", driver="GTiff", height=60, width=80, count=1, dtype="float32", nodata=run_tiled.NODATA5,
                       transform=from_origin(0, 300, 5, 5)) as dst:
        dst.write(z, 1)
    return f, z


def test_reader_fills_what_sticks_out_with_no_data(tiff):
    f, z = tiff
    with rasterio.open(f) as src:
        a = run_tiled.read_dem5(src, -7, 70, -9, 95)  # beyond the north, south, west and east borders
    assert a.shape == (77, 104)
    assert np.array_equal(a[7:67, 9:89], np.where(z == run_tiled.NODATA5, np.nan, z), equal_nan=True)
    assert np.isnan(a[:7]).all() and np.isnan(a[67:]).all() and np.isnan(a[:, :9]).all() and np.isnan(a[:, 89:]).all()
    with rasterio.open(f) as src:
        assert np.isnan(run_tiled.read_dem5(src, 100, 120, 0, 10)).all()  # a box with nothing inside
        assert np.isnan(run_tiled.read_dem5(src, 0, 10, -30, -10)).all()


VRT = dr.DEM5


@pytest.mark.skipif(not VRT.exists(), reason="no disk")
def test_alps_box_beyond_the_mosaic_is_no_data_and_inside_matches_the_window_reader():
    h, w, (x0, y0) = run_tiled.box_shape("alps", 5)
    assert (h, w, x0, y0) == (80000, 55000, 825000.0, 6600000.0) and x0 + 5 * (w - 1) > 1090000  # beyond the VRT (1080 km)
    r0, c0 = run_tiled.origin5(x0, y0)
    with rasterio.open(VRT) as src:
        width = src.width
        assert c0 + w > width
        a = run_tiled.read_dem5(src, r0 + 40000, r0 + 40500, c0, c0 + w)  # a strip at y = 6400 .. 6397.5 km, x 825 .. 1100 km
    assert a.shape == (500, w)
    east = a[:, width - c0:]
    assert east.size > 0 and np.isnan(east).all()
    assert np.isfinite(a[:, :20000]).any()
    b, org = dr.read5((x0, y0 - 5 * 40499, x0 + 20000, y0 - 5 * 40000))  # the window reader of dem_resolution
    assert org == (x0, y0 - 5 * 40000)
    n = min(b.shape[1], 4001)
    assert np.array_equal(a[:, :n], b[:500, :n], equal_nan=True)


@pytest.mark.skipif(not VRT.exists(), reason="no disk")
def test_pyrenees_box_west_of_the_mosaic_is_not_shifted():
    h, w, (x0, y0) = run_tiled.box_shape("pyrenees", 5)
    assert x0 == 300000.0
    r0, c0 = run_tiled.origin5(x0, y0)
    assert c0 == -2000  # 10 km west of the first column
    with rasterio.open(VRT) as src:
        a = run_tiled.read_dem5(src, r0 + 20000, r0 + 20500, c0, c0 + 6000)
        b = src.read(1, window=((r0 + 20000, r0 + 20500), (0, 4000))).astype(np.float32)
    assert np.isnan(a[:, :2000]).all()
    b[b == run_tiled.NODATA5] = np.nan
    assert np.array_equal(a[:, 2000:], b, equal_nan=True)


def test_boxes_have_more_than_2_pow_31_corners_at_5_m():
    for region in geo.REGIONS:
        h, w, _ = run_tiled.box_shape(region, 5)
        assert (h + 1) * (w + 1) > 2**31


def test_trial_window_boxes_are_35_km_squares_on_whole_km():
    region, (x0, x1, y0, y1) = run_tiled.window_box("alps1")
    assert region == "alps" and (x0, x1, y0, y1) == (941, 976, 6508, 6543)
    assert run_tiled.box_shape("alps", 5, "alps1") == (7000, 7000, (941000.0, 6543000.0))


# --- the mountain mask from 5 m block means ----------------------------------------------------------------------

def test_block_means_of_the_5_m_and_25_m_grids_agree_on_block_averaged_terrain():
    rng = np.random.default_rng(0)
    nb = (30, 40)
    coarse = np.kron(rng.uniform(200, 3200, nb), np.ones((200, 200)))  # fine terrain constant per km, plus relief
    fine = (coarse + rng.normal(0, 20, coarse.shape)).astype(np.float32)
    fine[:200, 200:400] = np.nan  # a no-data block
    h5 = run_tiled.block_means(lambda r0, r1, c0, c1: fine[r0:r1, c0:c1], *fine.shape, 5.0)
    # 25 m cells are the 5 x 5 means of the fine ones: the same 1 km means, the same mountain area
    w = fine.reshape(nb[0] * 40, 5, nb[1] * 40, 5)
    z25 = np.where(np.isnan(w).any(axis=(1, 3)), np.nan, w.mean(axis=(1, 3))).astype(np.float32)
    h25 = run_tiled.block_means(lambda r0, r1, c0, c1: z25[r0:r1, c0:c1], *z25.shape, 25.0)
    assert h5.shape == h25.shape == nb
    ok = np.isfinite(h5)
    assert np.array_equal(ok, np.isfinite(h25)) and not ok[0, 1]
    assert np.abs(h5[ok] - h25[ok]).max() < 0.05
    assert np.array_equal(mountains.mountain_area_h(h5.copy()), mountains.mountain_area_h(h25.copy()))


# --- the driver ----------------------------------------------------------------------------------------------------

def test_driver_runs_the_stages_in_order_and_skips_the_dimension_on_request():
    names = [n for n, _ in run_massif.stages("alps", 5, 2500, None, True)]
    assert names == ["tiled-dem", "tiled-mask", "tiled-fill", "tiled-flats", "tiled-dirs", "tiled-acc", "tiled-links",
                     "tiled-edges", "curves", "hgeo", "paths-all", "dim-crests", "dim-network", "clean"]
    assert [n for n, _ in run_massif.stages("alps", 5, 2500, "alps1", False)][-3:] == ["curves", "hgeo", "clean"]
    cmd = dict(run_massif.stages("pyrenees", 5, 3000, "pyr1", False))["tiled-fill"]
    assert cmd[-6:] == ["--tile", "3000", "--window", "pyr1", "--stage", "fill"]


@pytest.mark.skipif(not __import__("os").path.exists("/usr/bin/time"), reason="no /usr/bin/time")
def test_driver_stage_gives_seconds_and_peak_memory_and_failure_stops(tmp_path):
    lines = []
    s, peak = run_massif.run_stage("t", [sys.executable, "-c", "x = bytearray(60_000_000); print('hello')"],
                                   lambda m, st="": lines.append(m))
    assert "hello" in "".join(lines) and s >= 0 and peak is not None and peak > 0.05
    with pytest.raises(SystemExit):
        run_massif.run_stage("t", [sys.executable, "-c", "raise SystemExit(3)"], lambda m, st="": None)


def test_clean_stage_keeps_stats_only_and_does_not_create_the_dem_raster(tmp_path, monkeypatch):
    monkeypatch.setattr(geo, "CACHE", tmp_path)
    monkeypatch.setattr(geo, "DEM_M", 5.0)
    monkeypatch.setattr(ridges, "FLATS", "blm")
    run = tmp_path / "tiled_941-976-6508-6543_5m_T2500"
    (run / "fill1").mkdir(parents=True)
    for f in ("fill1/0000_0000.npz", "valley.npz", "lid.npy", "stats.json"):
        (run / f).write_text("x")
    monkeypatch.setattr(sys, "argv", ["run_tiled.py", "alps", "--dem", "5", "--window", "941,976,6508,6543", "--stage", "clean"])
    run_tiled.main()
    assert sorted(p.name for p in run.iterdir()) == ["stats.json"]
