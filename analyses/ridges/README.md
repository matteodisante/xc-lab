# Ridges: how much of H the terrain alone explains

Question: how much of the Hurst exponent H of a mountain flight does the geometry of the crests alone explain? The
geometric exponent H_geo is that of a ball at constant speed (35 km/h) along the crest paths of the French Alps and
Pyrenees, extracted from the 25 m IGN DEM; the active report compares it with the paraglider flights in time and along
their route. The intermittent model of the flights is a separate, frozen report.

| Question | Scripts | Document |
|---|---|---|
| H_geo of the crests: DEM to mountain area, divides, curves, joins, trees, paths, TAMSD | `geo.py` `mountains.py` `ridges.py` `curves.py` `joined.py` `tamsd.py` `hgeo.py` `build.py`; figures and macros `report_figures.py` `fig_joins.py` | active: `crests/crest_geometry.pdf`; results in Section 2 (`crests/sections/geometry.tex`), the construction in Section 5 (`method.tex`) and Appendix A (`appendix.tex`) |
| H of the paraglider flights in time (H_dyn) and along the route (H_route) against H_geo | `flights.py` `run_flights.py` `pauses.py` `glide_speed.py` `decompose.py` `report.py --part route` (figures `route_*`, tables, macros) | active: `crests/crest_geometry.pdf`, Section 3 (`crests/sections/flights_route.tex`), flight selection and rule in Sections 5.3-5.4, checks in Appendix C |
| Does the 25 m DEM change the crests or H_geo? 4 windows of 25 x 25 km at 25 m and 5 m (RGE ALTI) | `dem_resolution.py` | active: `crests/crest_geometry.pdf`, appendix "Where the numbers hold", subsection DEM resolution (`appendix.tex`) |
| Box dimension of the crests and the test D = 1/H_route | `boxdim.py` `run_dimension.py` | active: `crests/crest_geometry.pdf`, Section 4 (`crests/sections/dimension.tex`), details in Appendix D (`dimension_appendix.tex`) |
| Phases, glide and pause laws, intermittent ball, split of 1 - H, null models (H_dyn vs H_route, subordinated process X(t) = R(S(t))) | `pauses.py` `model.py` `glide_speed.py` `decompose.py` `report.py --part model` | on hold: `intermittency/crest_intermittency.pdf`, a frozen snapshot with its own copies of figures and generated files; it refers to the active report for the crests and the flight selection |
| Recipe and definitions of the flight comparison | | `paper/FLIGHTS.md` |

`paper/` holds the folder of an earlier paper on the same question (`main.tex`, `FLIGHTS.md`); the pipeline here is
built from scratch and does not use or reproduce its numbers.

## Procedure

1. **Ground.** IGN BD ALTI v2, 25 m, Lambert-93, `/Volumes/SSD_DISANTE/IGN_France_Elevation_25m/` (read-only). A
   region is the departments of its massif (Alps 04, 05, 06, 26, 38, 73, 74, 83, 84; Pyrenees 09, 11, 31, 64, 65,
   66); the mosaic covers the bounding box of their tiles.
2. **Mountain area.** UNEP-WCMC classes on 1 km cells (local range within 7 km, enclosed basins up to 25 km2, largest
   piece), computed on the tiles of the departments themselves: they define the massif, and the neighbouring tiles of
   the box serve only the water routing.
3. **Crests as divides.** Priority flood, D8 steepest descent, valleys of at least 0.5 km2 cut into links at the
   confluences; a side between two cells draining into different links, neither downstream of the other, is a divide.
   Tests: ground 6 cells (150 m) away on each side at least 10 m lower; crest at least 20 m above the streams.
4. **Curves.** Flat stretches up to 40 sides bridged, dead ends under 4 sides pruned, strokes through junctions
   (deflection at most 135 deg, direction measured 200 m out), smoothing spline with a cut-off of 100 m (the scale
   below which the shape is set by the grid, where the fit starts), resampled every 5 m, curves under 1 km dropped,
   at least half in the mountain area.
5. **Joins.** Straight, at most 1 km, within 45 deg of the run-out direction, snapped to a dead end within 200 m, up
   to 3 other components per dead end, shortest first, one per dead end. A join is accepted only if every DEM cell the
   segment touches (corners included) is at least 20 m above the streams: it never crosses a valley, however narrow.
6. **Trees and paths.** Curves, junctions and joins form a graph; a tree is a component, its root one end of its
   longest path, its paths run from the root to every other end.
7. **TAMSD.** 80 log-spaced lags up to a quarter of each path, equal weight per path, lags with at least 10 paths,
   local H(s) = half the log-log slope of the mean, H_geo on windows from 0.1 km, 0.1-1, 1-10 and beyond 10 km,
   1000 bootstrap resamples of whole trees (of curves for the curves).

Every constant is a named constant at the top of its module; `PARAMETERS.md` in the cache folder lists them. Every grid
constant is a length in metres (`ridges.XSEC_M`, `curves.BRIDGE_M`, `PRUNE_M`, `EDGE_M`, `mountains.BLOCK_M`) and goes
to cells with `ridges.cells(metres, cell)` at the cell size of the run (25 or 5 m): the 25 m runs give the same numbers
as before (cross-section 6 cells, bridges 40 sides, pruning 4 sides, edge 4 cells, 40 x 40 cells per 1 km cell), the 5 m
runs 30, 200, 20, 20 and 200 x 200. `RIDGES_DEM=5` (default 25) makes `build.folder()` the 5 m folder
(`rgealti5_..._flats-blm`, grid constants in cells as in the 25 m name); `curves_`/`edges_<region>.npz` carry the cell
size (`cell`, 25 when absent) and `hgeo.py`/`export_paths.py` read it. Side ids (corner `r (w + 1) + c`) are int64 as soon
as a grid has more than 2^31 corners (both 5 m boxes have 3-4 x 10^9).

## Run

```
uv run --with numba python build.py <alps|pyrenees>   # DEM -> mask, crest sides, curves (about 3 min)
uv run --with numba python hgeo.py <alps|pyrenees>    # joins, trees, paths, TAMSD, H_geo, exports (about 1 min)
uv run --with numba --with pyproj python report_figures.py   # figures, macros and tables of the crest part
uv run --with numba python report.py                          # route figures, tables and macros of the flight part (--part model: the frozen report)
cd crests && latexmk -pdf crest_geometry.tex                  # the active report
cd intermittency && latexmk -pdf crest_intermittency.tex      # the on-hold report (needs crests/crest_geometry.aux for its references)
uv run --with numba python -m pytest -q               # synthetic DEMs, curves, joins, TAMSD; DEM tests skip without the disk
```

The rule for the cells of the flats is `RIDGES_FLATS=flood` (default: they drain to the cell that reached them in the
priority flood) or `blm` (gradients of Barnes, Lehman and Mulla 2014 on the filled surface; the folder name gets
`_flats-blm`); `RIDGES_CACHE` moves the cache root. Set both for every script of the pipeline.

Out of core (DEMs that do not fit in memory, e.g. the 5 m RGE ALTI): `tiled.py` is an exact tiled version of the
routing (priority flood of Barnes 2016, flats of Barnes, Lehman and Mulla 2014, accumulation of Barnes 2017, link of
each cell, crest sides), `run_tiled.py <region> --dem 25|5 --tile T --stage dem|mask|fill|flats|dirs|acc|links|edges|all`
runs it per tile, resumable, with `RIDGES_FLATS=blm`; at 25 m it writes `mask_`, `edges_` and `hand_<region>` into the
parameter folder so that `build.py --stage curves` and `hgeo.py` run on them unchanged. `check_tiled.py` compares a
tiled run with the in-memory arrays (0 differences: Alps at T = 400, Pyrenees at T = 400 and 1000); `test_tiled.py`
does it on synthetic DEMs and a real window. Link numbers differ from the in-memory ones (the partition and the
link forest are the same).

The 5 m run of a massif is one command (`RIDGES_FLATS`, `RIDGES_DEM` and `RIDGES_CACHE` are set by the driver):

```
uv run --with numba python run_massif.py alps        # then: ... pyrenees (one region at a time)
uv run --with numba python run_massif.py alps --window alps1    # trial on a 35 x 35 km window (or --window x0,x1,y0,y1 in km)
```

`run_massif.py` runs, each stage in its own process under `/usr/bin/time -l`: the tiled stages (`run_tiled.py`, tile 2500
cells), `build.py --stage curves`, `hgeo.py`, `export_paths.py` and `run_dimension.py crests|network` (inputs of the box
dimension; `--no-dimension` skips them) and `run_tiled.py --stage clean` (the big rasters; `mask_`, `edges_` and `hand_` stay
in the parameter folder). It resumes after a crash (run the same command: finished stages are in `driver_<region>.json`,
the tiled stages resume tile by tile) and logs every line with a timestamp to `run.log` in the folder
`/Volumes/SSD_DATA/xc-lab/ridges/rgealti5_..._flats-blm/` (a window: `trial_<window>/`). The 5 m mosaic is read from the VRT of
`/Volumes/SSD_DATA/dem/rge_alti_5m/` (README there): what lies outside it, or has no tile, is no data and no BD ALTI cell
enters. The mountain area is recomputed from the 5 m block means (200 x 200 cells per 1 km cell; `compare_masks.py` compares
it with the 25 m one); `check_tiled.py --dem 5 --window <w>` compares the tiled run of a window with the in-memory routing
and `check_window.py <w>` the interior crests with run B of `dem_resolution.py`.

Trials of the driver (2026-11-17, T = 2500): on the 35 x 35 km window `alps1` the chain takes about 75 s with a peak of
0.4 GB per stage (the tiled run equals the in-memory routing bit for bit, T = 1700 and 2500; its interior crests differ from
run B of `dem_resolution.py` by 0.7% and 1.0% of the sides, the flat rule); on a 100 x 100 km box of the Alps (6,957 km2 of
data) 14 min, 0.74 GB (`paths-all`), 5.2 GB of disk at the peak; `build.py` read 0.47 GB for 1.5 M sides. Mountain area from
the 5 m block means: Alps 40,460 km2 against 47,754 at 25 m, Pyrenees 15,494 against 20,640; of the cells that differ, 98% lie
where the 5 m mosaic has no data (the 25 m tiles reach beyond the border), and on the footprint of the 5 m mosaic the two masks
differ in 118 and 27 cells of 1 km (`compare_masks.py`); the four windows of `dem_resolution.py` are identical.

One heavy process at a time. Peak resident memory (`getrusage`, recorded in `peak_ram.json`): `build.py` 1.8 GB for
the Alps and 1.4 GB for the Pyrenees (`/usr/bin/time -l` read 2.4 and 2.1 GB), `hgeo.py` 0.4 and 0.3 GB.

## Files in the cache folder

`/Volumes/SSD_DISANTE/xc-lab/ridges/bdalti25_valley0.5_xsec6-10_hand20_gap40_prune4_defl135-200_min1000_step5_smooth100_join1000-45-200/`

| File | Content |
|---|---|
| `mask_<region>.npz` | mountain area on 1 km cells: `mask` (rows go south), `left`, `top` (west and north edges, Lambert-93 m), `cell` = 1000 |
| `tamsd_rows_<region>_<kind>.npz` | kind `curves` or `paths`: `rows` (TAMSD in m2 of every curve or path on the common lags, NaN beyond a quarter of it), `k` (lags in samples), `s_m`, `group` (what the bootstrap resamples: the curve, or the tree of a path), `tree`, `length` (m), `edge` (reaches the data edge), `step` |
| `hgeo_<region>_<kind>.csv` | per lag: `s_m`, `tau_s` at 35 km/h, `n`, `mean`, `q05`, `q95` (over curves), `H`, `H_lo`, `H_hi` (95% bootstrap) |
| `hgeo_windows.csv` | H on the windows of s, per region and kind (`curves`, `paths`, `_noedge`), with 95% interval |
| `paths_<region>.npz` | longest path of every tree with one of at least 10 km, every 5 m: `xy` float32 offsets from `origin` (Lambert-93 = `xy` + `origin`), `off`, `tree`, `length` |
| `joins_<region>.npz` | the joins (`a`, `b` end points, `length`, `tree`) and the longest tree (`big_*`) |
| `curves_<region>.npz`, `edges_*`, `hand_*` | intermediate: curves every 5 m, crest sides, height above the streams (int16 m) |
| `peak_ram.json`, `PARAMETERS.md` | peak memory of each stage; the parameters |

## Results

Counts (Alps / Pyrenees): mountain area 47,754 / 20,640 km2; curves 13,542 / 6,438 (33,470 / 16,397 km, median
1.94 / 2.04 km); joins 1,789 / 705, none touching a cell below 20 m above the streams on an independent 0.25 m
sampling; trees 2,040 / 841; paths 12,981 / 6,202 (median 14 / 15 km, longest 114 / 91 km); trees with a path of at
least 10 km exported 309 / 159.

H_geo of the paths (95% bootstrap interval over trees):

| window of s | Alps | Pyrenees |
|---|---|---|
| from 0.1 km | 0.933 (0.929-0.940) | 0.930 (0.925-0.933) |
| 0.1-1 km | 0.960 (0.959-0.961) | 0.960 (0.959-0.962) |
| 1-10 km | 0.927 (0.921-0.934) | 0.917 (0.912-0.922) |
| beyond 10 km | 0.867 (0.812-0.933) | 0.931 (0.774-0.974) |

Single curves (from 0.1 km): 0.974 / 0.965. Their 1-10 km window mixes populations (only the longest curves reach
the large lags) and is not used.

Local H(s) of the paths, Alps / Pyrenees: 0.983 / 0.982 at 100 m, 0.951 / 0.950 at 500 m, 0.940 / 0.933 at 1 km,
0.935 / 0.927 at 2 km, 0.920 / 0.912 at 5 km, 0.905 / 0.850 at 10 km. Without the curves and paths that reach the
data edge the windows are the same to the third decimal (report, Appendix B).

Flights: paragliders and hang gliders that declared a distance flight, from the cleaned fixes of xc-dynamics, kept at
one fix per second (median interval 1 s), opening at least 0.1, at least 80% of the fixes over one massif and phases
from Vilpellet's segmenter. The route replaces each thermal pause, the climb runs of the segmenter (consecutive runs
joined; search runs stay in the route), by its chord. Selected paragliders: 15,190 in the Alps and 1,406 in the
Pyrenees (hang gliders: 266 and 2, not analysed). Cache:
`/Volumes/SSD_DISANTE/xc-lab/ridges/flights_cleaned_v35_open0.1_dt1_reg0.8_pause-vilpellet/` (`run_flights.cache_dir()`);
the first run, with a turn detector and median interval up to 5 s, is in `flights_cleaned_v35_open0.1_dt5_reg0.8/`
and `check_raw.py` still uses that detector.

Limits: the DEM stops at the border (edge-flagged curves and paths), the shape below 100 m is set by the grid, and the
last window rests on few trees (42 in the Alps and 16 in the Pyrenees at a lag of 10 km).
