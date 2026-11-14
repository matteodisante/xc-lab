# Universality

Is the Hurst exponent H of a cross-country flight the same for paragliders (para) and hang gliders (hang)? On
declared open flights with an opening above 0.1, both disciplines, whole pieces on a 10-s grid, raw (all terrain) and
by terrain class (median ground elevation under the points of the flight, IGN BD ALTI 25 m) and by region.

| Question | Scripts | Document |
|---|---|---|
| Which flights, which curves | `measure.py` (pass over the fixes), `tamsd.py`, `openflights.py`, `twoway.py` | `universality.pdf` sections 1 and A |
| What terrain does a flight fly over | `terrain.py` (DEM to 200 m blocks, class of a flight), `example.py` | section 2 |
| Same H, all terrain, fixed population or all flights | `analysis.py` (`pair`, `curve`) | section 3 |
| Same H within terrain classes and regions | `analysis.py`, `regions.py` | sections 4 and 5 |
| Does the shrinking population change the answer | `analysis.py` (designs A and B) | section 6 |
| Figures, tables, macros | `figures.py`, `report.py` | `generated/`, `out/full/` |

## Answer

On a fixed population at T = 1000 s, H is 0.922 (para) and 0.898 (hang), difference +0.024 with a 90% band
[+0.017, +0.031]. The difference is positive in 24 of 24 populations and designs (all terrain, four classes, Alps; four designs), between
+0.007 and +0.042, with a band that excludes 0 in 22. At the longest T of the raw comparison (11 450 s, 172 hang flights)
it is +0.007 [-0.006, +0.020]. The terrain mix does not account for it. H falls from plains to mountains in both disciplines
(para 0.941 to 0.907, hang 0.932 to 0.880); the gap between the gliders is of the same order as the gap between terrains. A
shrinking population (design B) lowers H by up to 0.018 and leaves the difference between the disciplines unchanged within the
bands. Only the Alps have enough hang flights among the four regions (Alps, Pyrenees, Channel Coast, Champagne-Lorraine).

## Data and code version

- Flights: table of the run `d120_eh500_ez300_th45_g120_q0.1` of `methods/group-identification` (interval of the logger, departure,
  landing, date), catalogue types and pilots from the catalogues of both disciplines, cleaned fixes `fixes.parquet` of
  `/Volumes/SSD_DISANTE/{paragliders/ffvl_cfd_igc,hang_gliders/delta_cfd_igc}/derived`. xc-dynamics at `8476644` (not used).
  xc-lab at `ba25b28` plus the uncommitted files of this folder.
- DEM: `/Volumes/SSD_DISANTE/IGN_France_Elevation_25m/mainland_france` (read-only).
- Caches, with their `PARAMETERS.md`: `/Volumes/SSD_DISANTE/xc-lab/universality/dem_bdalti25_mean200m_sea0nodata/` (elevation grid, 129 MB)
  and `/Volumes/SSD_DISANTE/xc-lab/universality/flights_grid10_support0.8_hmax10_declared_open_dem200m/` (flights, pieces; `sample300/` for the trial).
- Nothing is written to `SSD_DATA`.

## Rebuild

```bash
cd analyses/universality
uv run python terrain.py --build            # once: 200 m elevation grid (70 s, 0.8 GB)
uv run python measure.py [--sample N]       # pass over the fixes (3 min, 0.8 GB with 8 workers)
uv run python analysis.py [--sample N]      # curves and exponents: out/full/results.json (20 s, 0.9 GB)
uv run python example.py                    # the real flight of the terrain figure
uv run python report.py [--sample N]        # figures (usetex, needs LaTeX), tables and macros in generated/
latexmk -pdf universality.tex
uv run pytest                               # synthetic fBm, terrain, opening, fixed population, regions, bootstrap
```

One heavy job at a time (8.6 GB Mac). Peak memory measured with `/usr/bin/time -l`: `terrain.py --build` 0.82 GB,
`measure.py --sample 300` 0.48 GB, `measure.py` 0.77 GB, `analysis.py` 0.91 GB.

## Scripts

- `terrain.py`: elevation of 200 m blocks from the 25 m tiles (mean of 8 x 8 cells, blocks over half NoData or exactly 0 m are
  NaN), `ground()` under Lambert-93 points, `classify()` of a flight: median of the ground elevations of its 10-s points, edges
  300, 800 and 1500 m, unknown when under half of the points are on the DEM.
- `measure.py`: for each declared open eligible flight, its ends (opening), its pieces with TAMSD sums and origins at 37 lags, and its
  median ground elevation. 55 879 para and 1 417 hang open flights.
- `analysis.py`: designs A (fixed population, T = 1000 s and the longest T where both populations keep 100 flights) and B (all
  flights), per population and pair of populations; alpha(tau), H, differences with paired two-way bootstrap bands.
- `regions.py`: the four boxes of the earlier project (`soaring/analysis/regions.py`) and the rule box + terrain class.
- `report.py`, `figures.py`: everything in `universality.tex` comes from `out/full/results.json`.

The tests cover every number the document quotes: `test_tamsd.py` (power law and fBm of known H give H back, fixed population,
pooling of pieces), `test_terrain.py` (blocks, class edges, unknown rule, median of a synthetic track), `test_openflights.py`,
`test_twoway.py`, `test_regions.py`, `test_analysis.py` (known exponents and their difference, strata, regions), `test_report.py`
(macros against the results).
