# How straight is a ridge? Paper, figures, data and code

The geometric Hurst exponent H_geo of the crests of the French Alps and Pyrenees: a ball runs
at constant speed (35 km/h) along crest paths extracted from the 25 m IGN DEM, and its TAMSD
is fitted from 100 m (4 DEM cells) up. This folder holds everything needed to read the paper,
redraw its figures, rebuild it from the DEM, and run the same analysis on paragliding flights.

| Path | What it is |
|---|---|
| `main.tex` | the paper; `make` builds `main.pdf` (pdflatex twice, no bibtex) |
| `Makefile` | `make`, `make clean`; `make code` copies `../src` again (development machine only) |
| `figs/` | the six figures of the paper, PDF and PNG (`code/fig_paper.py`) |
| `data/` | the numbers behind the paper (`code/hgeo.py`), listed below |
| `code/` | the pipeline from the DEM to the figures, plus `flights.py` and its test |
| `FLIGHTS.md` | instructions for the flight analysis (written for Claude on another PC) |

## data/

All coordinates are Lambert-93 (EPSG:2154), stored as float32 offsets from `origin`.

| File | Content |
|---|---|
| `hgeo_windows.csv` | H_geo per region, kind (curves, paths, and `_noedge`: without those reaching the data edge) and window of arc length, with 95% bootstrap interval (Table 1) |
| `hgeo_<region>_<kind>.csv` | per lag: arc length `s_m`, `tau_s` at 35 km/h, number of curves, mean TAMSD and 5-95% band, local H(s) with 95% interval, 1/D at eps = R(s) with 95% interval |
| `boxcount_<region>.csv` | per box size: mean box count of crests and of straight segments, D(eps) and 95% interval, for curves and paths |
| `tamsd_rows_<region>_<kind>.npz` | the TAMSD of every curve or path on the common lags (NaN beyond its last lag), with group (tree) ids and lengths: refit any window with the same bootstrap |
| `crests_<region>.npz` | the crest curves of the region, a point every 25 m (`xy`, `off`, `length`, `edge`) |
| `paths_<region>.npz` | long crest paths every 5 m, the longest path of every tree that has one of at least 10 km (Alps 193, Pyrenees 69), for `test_flights.py` |
| `mask_<region>.npz` | the mountain area on 1 km cells (`mask`, `left`, `top`, `cell`), to assign flights to a massif |

## code/

Run order, from the DEM to the paper (times on 8 cores of the development machine):

1. `./run.sh build.py alps` (and `pyrenees`): crest network and curves, `results/<region>/curves.npz` (Alps about 10 min)
2. `./run.sh tamsd.py <region>`: TAMSD of the curves (minutes)
3. `./run.sh pauses.py <region>`: the same with Lomax pauses (not used in the paper)
4. `./run.sh joined.py <region>`: joined trees and paths and their TAMSD (Alps about 8 min, Pyrenees 5)
5. `./run.sh boxcount.py <region>`: box counting of curves and paths
6. `./run.sh hgeo.py`: everything in `data/`
7. `./run.sh fig_paper.py`: everything in `figs/`

| Module | Role |
|---|---|
| `geo.py` | DEM tiles, mosaics, Lambert-93 |
| `ridges.py` | flow routing (priority flood), divides, crest network |
| `curves.py` | strokes through junctions, splines every 5 m |
| `mountains.py` | the UNEP-WCMC mountain area on 1 km cells, the data edge |
| `build.py` | the regions and the extraction parameters `P` |
| `maps.py` | loading results, colours, polylines |
| `tamsd.py` | TAMSD, local exponent, fit from `S_RES` = 100 m, bootstrap |
| `pauses.py` | intermittent motion with Lomax phases (outlook only) |
| `joined.py` | gap joins, trees, paths |
| `boxdim.py`, `boxcount.py` | box counting with the straight-segment control |
| `hgeo.py` | H_geo on windows, local H(s) bands, 1/D, the exports in `data/` |
| `fig_paper.py` | the figures of the paper; `fig_massif.py`, `fig_test.py`, `fig_joined.py` for the maps |
| `flights.py` | the flight analysis (IGC files), standalone: numpy, scipy, matplotlib |
| `test_flights.py` | synthetic IGC flights from `data/paths_<region>.npz`, run through `flights.py` |

Steps 1-7 need the DEM in `<root>/IGN_France_Elevation_25m/mainland_france/geotiff/D0xx/`,
where the root is this folder or `$RIDGES_ROOT`; results go to `<root>/results/`. Figures and
data need only `data/` (and, for the maps, `results/`). `flights.py` needs only `data/`.

Requirements: `code/requirements.txt` (Python 3.12; GDAL only to read the DEM).

## Data source

IGN, BD ALTI version 2.0, 25 m (ASCII grids, Lambert-93, IGN69), Licence Ouverte / Open
Licence 2.0, downloaded from the IGN Geoplateforme on 24 September 2026. Departments and
edition dates: D004 2023-08-08, D005 2021-08-04, D006 2023-08-08, D026 2022-12-16,
D038 2020-11-13, D073 2020-10-15, D074 2020-10-15, D083 2022-12-05, D084 2022-12-16 (Alps);
D009 2023-10-04, D011 2023-10-04, D031 2021-05-12, D064 2021-04-19, D065 2020-02-11,
D066 2023-10-04 (Pyrenees). The mosaic of a massif uses every tile in the bounding box of
these departments, so tiles of neighbouring departments are read too.

## Test of flights.py (test_flights.py, 60 Alps paths)

The paths' own TAMSD gives H = 0.939 from 100 m (0.955 on 0.1-1 km, 0.928 on 1-10 km).

| Synthetic flights | route | track | dyn |
|---|---|---|---|
| no circling, no noise | 0.940 | 0.940 | 0.938 |
| circling (40 m, 25 s turns) | 0.938 | 0.906 | 0.906 (dyn_route 0.913) |
| circling + 3 m GPS noise + 2 min on the ground | 0.937 | 0.904 | - |

The route recovers H_geo within 0.004; the circles must be collapsed before comparing.
