"""compare_dem.py: the zoom search on synthetic curves, and the numbers of the four runs against the report (skipped without the disk)."""

import numpy as np
import pandas as pd
import pytest

import compare_dem as cd


def line(x0, y0, x1, y1, step=5.0):
    n = int(np.hypot(x1 - x0, y1 - y0) / step) + 1
    return np.c_[np.linspace(x0, x1, n), np.linspace(y0, y1, n)]


def test_zoom_square_holds_the_points_that_only_one_run_has():
    common = line(0, 500, 9000, 500)
    only3 = line(6000, 1000, 6000, 4000)  # a crest that C3 has and C4 lacks, in the square x 4500-7500
    only4 = line(6200, 1000, 6200, 4000) + [500.0, 0.0]  # 700 m away: C4's own
    p3 = np.vstack([common, only3, line(0, 8000, 9000, 8000)])
    p4 = np.vstack([common, only4, line(0, 8000, 9000, 8000)])
    z = cd.zoom_place(p3, p4)
    x0, y0, x1, y1 = z["box"]
    assert x1 - x0 == cd.ZOOM_M and y1 - y0 == cd.ZOOM_M
    assert x0 <= 6000 <= x1 - 1 and x0 <= 6700 <= x1 - 1 and y0 <= 2000 <= y1
    assert z["only_c3"] >= 500 and z["only_c4"] >= 500  # 2.5 km of crest every 5 m fit in a 3 km square on the 500 m grid


def test_delta_prints_zero_as_a_bound():
    assert cd.delta(-0.0004) == "$<0.001$"
    assert cd.delta(-0.0123) == "0.012"


def test_differences_use_the_printed_values():
    res = dict(hgeo={"alps_C1": {"paths": {"w": [0.93049, 0, 1]}}, "alps_C2": {"paths": {"w": [0.93051, 0, 1]}}})
    assert cd.hdiff(res, "alps", "C1", "C2", "w") == pytest.approx(0.001)  # 0.931 - 0.930, as the table shows


disk = pytest.mark.skipif(not cd.geo.CACHE.exists() or not any(cd.geo.CACHE.glob("rgealti5_*")), reason="caches not mounted")


@disk
def test_four_distinct_folders():
    assert len({cd.folder(r) for r in cd.RUNS}) == 4


@disk
@pytest.mark.parametrize("region, curves, paths, trees", [("alps", 13542, 12981, 2040), ("pyrenees", 6438, 6202, 841)])
def test_c1_counts_are_those_of_the_report(region, curves, paths, trees):
    c = cd.counts(cd.folder("C1"), region)
    assert (c["curves"], c["paths"], c["trees"]) == (curves, paths, trees)


@disk
@pytest.mark.parametrize("run", cd.RUNS)
def test_windows_count_the_paths_of_the_tamsd_rows(run):
    d = cd.folder(run)
    w = pd.read_csv(d / "hgeo_windows.csv")
    for region in cd.REGIONS:
        n = w[(w.region == region) & (w.kind == "paths")].n.unique()
        assert list(n) == [cd.counts(d, region)["paths"]]


@disk
def test_c1_box_dimension_is_that_of_section_4():
    d = cd.dimension(cd.folder("C1"), "alps")
    assert f"{d['windows']['from 0.1 km']['box'][0]:.3f}" == "1.051"  # generated/dimension_table_main.tex
