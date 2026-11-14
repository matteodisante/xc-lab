# Flight analysis: Hurst exponents of paragliding flights against the crest geometry

Instructions for Claude, running on the user's personal PC, where the IGC flight files are.
Read this file in full, then `README.md`, then the paper (`main.tex`, Sections 3-6). Talk to
the user in Italian; write every file, comment and figure label in English.

## 1. Goal

The paper measures H_geo(s): the Hurst exponent of a ball running at constant speed along the
crest paths of the French Alps and Pyrenees, from the TAMSD on lags of arc length s, fitted
from 100 m (4 DEM cells) up. The question is how much of the Hurst exponent of a real ridge
flight the terrain alone explains. Your job: compute the same quantities for the user's
flights, with exactly the same definitions, and compare.

Reference values (paths, `data/hgeo_windows.csv`):

| window of s | Alps H_geo | Pyrenees H_geo |
|---|---|---|
| from 0.1 km | 0.922 (0.918-0.929) | 0.916 (0.910-0.931) |
| 0.1-1 km | 0.960 (0.959-0.961) | 0.957 (0.953-0.960) |
| 1-10 km | 0.925 (0.920-0.933) | 0.910 (0.900-0.925) |
| beyond 10 km | 0.868 (0.773-0.877) | 0.859 (0.841-0.946) |

Local H(s) of the Alps paths: 0.987 at 100 m, 0.95 at 550 m, 0.94 at 2 km, 0.89 at 4.6 km,
0.87 at 11 km. No plateau: compare curves H(s), not only single numbers.

## 2. What is in this folder

- `code/flights.py`: the reference implementation. Standalone (numpy, scipy, matplotlib; no
  GDAL, no pyproj). Read its docstring and constants first.
- `code/test_flights.py`: synthetic IGC flights built from the crest paths; the benchmark.
- `data/`: `tamsd_rows_<region>_paths.npz` (H_geo is refitted from these on the flights'
  windows, with the same bootstrap), `mask_<region>.npz` (mountain area, to assign flights to
  a massif), `hgeo_*.csv`, `paths_<region>.npz`, `crests_<region>.npz`.
- The rest of `code/` rebuilds the crests from the DEM. You do not need it, nor the DEM.

## 3. Definitions (do not change them without asking the user)

- Positions in Lambert-93 (EPSG:2154), the projection of the DEM (`flights.lambert93`).
- TAMSD of one piece of track: delta^2(lag) = mean over t of |r(t + lag) - r(t)|^2, horizontal
  only, on 80 log-spaced lags up to a quarter of the piece (`MAX_FRAC`).
- Across flights: mean TAMSD with equal weight per piece; only lags with at least 10 pieces.
- Local exponent H(lag) = (d ln mean / d ln lag) / 2.
- H on a window: least-squares slope of ln mean against ln lag, divided by 2; windows of arc
  length 100 m-inf, 0.1-1 km, 1-10 km, beyond 10 km; time windows are those divided by v.
- 95% intervals: 1000 bootstrap resamples of whole flights (crest paths: whole trees).
- Four kinds per flight:
  - `dyn`: positions every 1 s in time, circles included: H_dyn(tau);
  - `dyn_route`: the same with each thermalling circle replaced by uniform motion along its
    chord;
  - `track`: the track resampled every 5 m of arc length after a smoothing spline (rms
    tolerance 3 m x sqrt 2, against GPS noise): H_track(s);
  - `route`: the same with each circle replaced by its chord: H_route(s). **This is the one
    to compare with H_geo.**
- Thermalling circles: heading turning faster than 8 deg/s averaged over 20 s, runs merged
  across gaps < 10 s, kept if longer than 20 s.

## 4. Recipe

1. Install: Python >= 3.10 with numpy, scipy, matplotlib (`code/requirements.txt`).
2. Benchmark first: `cd code && python test_flights.py alps 60`. Expect route H within about
   0.004 of the paths' own H (Section 8). If not, stop and find out why.
3. Look at the real files before running everything. On 10-20 flights check:
   - the IGC dialect: B records `BHHMMSSDDMMmmmNDDDMMmmmEV...`; HFDTE date header (both
     `HFDTE010726` and `HFDTEDATE:010726,01` exist); fixes flagged `V`;
   - the sampling interval (`dt_median_s` in `flights.csv`): circle detection needs about
     1-2 s; above about 5 s it is unreliable, so tell the user and treat those flights apart;
   - gaps, GPS jumps, the cut to the time in the air (`duration_s`, `track_km`);
   - the circle detection by eye: plot a few tracks with the circling runs in another colour
     (`circling_frac` should typically be 0.2-0.5 for cross-country flights);
   - the region assignment (`frac_alps`, `frac_pyrenees`, `region`).
   Fix the parser or the thresholds only after showing the user what goes wrong.
4. Run: `python code/flights.py IGC_DIR --out flights_out --jobs N` (N <= the free cores).
5. Read `flights_windows.csv` (H, H_geo, dH = H - H_geo with intervals, per region, kind and
   window) and the figures `flights_<region>.png`.

## 5. Decisions to ask the user before the full run

- Which flights: all, only cross-country (e.g. track > 20 km), only some years or sites;
  whether to drop flights with long ground segments or gaps.
- Thermalling circles: collapsed (default, needed to compare the route with the terrain) or
  kept; the thresholds of Section 3 if the visual check is poor.
- The speed v that maps time lags to arc length (`--v-kmh`, default 35 km/h, the speed of
  the paper's ball): the paper's value, or the mean speed of each pilot or flight.
- `--region-min` (default 0.8): fraction of a flight that must lie over a massif's mountain
  area for the flight to count for it.
- Whether to split by pilot, site or season.

## 6. The comparison

- Terrain: H_geo(s) from the crest paths.
- Route: H_route(s) - H_geo(s), per window and as curves H(s). Near zero means the pilots fly
  as straight as the ridges; below zero, that they bend more (crossing to other ridges,
  zig-zags); above zero, that they straighten the ridges (shortcuts across valleys).
- Track vs route: what the circles add (expected below 1 km; the benchmark shows -0.03 on the
  whole range).
- Time: H_dyn(tau) against H_route(s = v tau). If the progress along the route over a lag tau
  grows as tau^beta, H_dyn(tau) ~ beta(tau) H_route. Estimate beta(tau) from the ratio, and
  compare with the phases in `phases.csv` (circling and gliding durations, whole phases only;
  `flights.py` prints a Lomax fit of each, as in `pauses.py`): with a finite mean pause beta
  should return to 1 at lags much longer than a glide plus a climb. The Lomax fits are poorly
  constrained (in the benchmark mu came out between 5.7 and 16): report them with intervals
  or as a survival plot, not as single numbers.

## 7. Outputs to produce

- `flights_out/` from `flights.py` (tables, rows, figures).
- A table like Table 1 of the paper with H_route, H_track, H_dyn and H_geo per window and
  massif, with intervals, and the number of flights.
- A figure: local H(s) of route, track and crests on the same axes (top axis arc length,
  bottom axis lag at v), one panel per massif; a second with H_dyn(tau) and beta(tau).
- A short Markdown note of what was found, with the decisions taken. Ask the user before
  deciding anything that changes a definition.
- Figures: essentials only, text at least 8 pt at print width, nothing outside a panel.

## 8. Benchmark (test_flights.py, 60 Alps paths, development machine)

The paths' own TAMSD: H = 0.939 from 100 m, 0.955 on 0.1-1 km, 0.928 on 1-10 km.

| synthetic flights | route (whole / 0.1-1 / 1-10 km) | track (whole / 0.1-1 km) | dyn (whole / 0.1-1 km) |
|---|---|---|---|
| no circling, no noise | 0.940 / 0.957 / 0.927 | as route | 0.938 / 0.953 |
| circling, 40 m radius, 25 s per turn | 0.938 / 0.951 / 0.926 | 0.906 / 0.911 | 0.906 / 0.911 (dyn_route 0.913 / 0.936) |
| circling + AR(1) GPS error 3 m, 60 s + 2 min on the ground | 0.937 / 0.953 / 0.924 | 0.904 / 0.910 | - (58 of 60 flights assigned to the Alps) |

Phase detection: mean glide 245 s (truth 239 s), mean circling 77-82 s (truth about 69 s,
rounded up to whole turns). Synthetic flights use Lomax glides (mu 3.93, tau0 700 s) and
climbs (mu 3.88, tau0 200 s) at 35 km/h; real flights will differ.
