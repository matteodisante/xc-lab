# Plan: group time against isolated solo flights on the same route

Folder: `xc-lab/analysis/groups/group-identification/`. File to rewrite: `isolated.py`.

`isolated.py` already holds `sphere()` (positions on the sphere, in m), the index of every flight of
the archive (`read_index`, cache `CACHE/archive_index.npz`, 30 M positions, one per minute),
`levels()` (wing level per flight) and `isolated()` (the 10-km check). Its `matches()` looks for
matches on the **same day** and finds almost none (5 pieces): replace it.

## Goal

Compare each piece of group time (paragliders, cache `route.TRACKS`) with segments of truly solo
flights on the same route, on **other days**, in the same season and hour, with the same wing
level; measure how much of the group - solo difference remains.

## Control rules

A solo segment k matches the piece g when all hold:

1. **Route**: `route.coverage` >= `route.COVER` (0.8) with `route.RHO` = 500 m, same direction
   through `route.segment`, duration >= `route.SHORTEST`.
2. **Date**: different from the date of g. No condition on the site.
3. **Season and hour**: month within +-1 (circular) and local hour within +-2
   (`decompose.local_hour`), both on the middles of g and of the segment, as in
   `route.units(season=True)`.
4. **Wing level**: the same, from `levels()` (beginners A/B, experienced C/D/CCC).
5. **Solo flight**: k belongs to a flight that never had a contact (population "solo" of
   `route.TRACKS`).
6. **Isolation**: at every point of the segment, one every `route.MATCH_S` s, no other flight of
   the index within 10 km at the same instant, i.e. within `TIME_S = 60` s, since the index has one
   position per minute. Use `isolated()`.

Keep **all** matches, no cap per piece.

## Statistics

Per level (`beginners`, `experienced`, `all`): group, control and difference, with bands.

- **MSD(tau)** and **H_loc(tau)** over every lag, fitted **H**, **H_loc(100 s)**, **H_loc(1000 s)**.
- **Phases**: time in glide (transition) (%), time in search (%), time in climb (%), searches that
  end in a climb (%), climb rate (m/s), glide speed (km/h), glide ratio, height lost per km of
  glide (m/km), climbs per hour. Definitions of `budget.STATS`, `performance.STATS`,
  `decompose.SINK`; time in search = (`found_s` + `vain_s`) / `s`. The glide ratio is the inverse
  of the height lost per km: both go in the JSON.

## Steps

**0. Cleanup.** Delete `CACHE/diag_noWindow.parquet` and `CACHE/isolated_matches.parquet` (trials
of the same-day version). Replace `TIMES_S` with `TIME_S = 60`, remove `WINDOW_S`, rewrite the
module docstring with the rules above.

**1. Matching with no site or day condition.**
- Per wing level, one cKDTree of the points of every solo segment of that level, in `sphere()`
  coordinates; apply rules 1-5.
- Queries with `workers=6`. The matching runs in **one process**: threads share the tree,
  processes would copy it.
- First print the number of points and measure the peak with `/usr/bin/time -l` on `--sample 500`.
- Swap may be used a little beyond the 8.6 GB of RAM, but must not fill. If the expected peak of
  the full run exceeds about 7 GB, split into 0.5 deg tiles by the middle of the piece, one tree
  per tile holding the solo segments with at least one point in the tile widened by 50 km.

**2. Isolation in batches, in parallel.**
- Group the matches that pass rules 1-5 by the **date of the candidate**.
- Check isolation with `ProcessPoolExecutor(6)`, one task per day.
- Pass each task only the slice of the index of its day (+-2 h margin) and the segments to check,
  never the whole index.

**3. Fixed population, longest possible lags.** A lag tau needs pieces and segments lasting at
least tau / `SUPPORT` (0.8).
- **Main series**: lags `tamsd.LAGS` up to 1000 s, every matched piece and its segments.
- **Long series**: the log spacing of `tamsd.LAGS` extended to `LONG_TAU_MAX`.
  - For tau_max in (2000, 3000, 5000, 8000) s, print per level how many matched pieces remain
    when only pieces and segments lasting at least tau_max / 0.8 are kept.
  - `LONG_TAU_MAX` = the largest tau_max with at least `MIN_PIECES = 200` pieces per level.
  - On that fixed subset, MSD and H_loc over every lag up to `LONG_TAU_MAX`.
- MSD per segment with `common.msd.tamsd_sums` on the 10-s grid positions, as in `route.units`.
- H and H_loc with the functions of `tamsd` (`fit_h`, `local_h`) or `zones.curve`. If they assume
  `tamsd.LAGS`, add an optional `lags` argument whose default leaves the current results unchanged.

**4. Phases.**
- Vilpellet's runs: `phase_msd.RUNS["para"]` with the terms of `budget.columns`, and the terms of
  `CACHE/performance_terms.parquet` (gain_m, climb_s, chord_m, transition_s, loss_m, climbs).
- Each run goes to the window (piece of group time or control segment) that holds its **middle**,
  with the definition of `survival.assign`: reuse it, adapted to arbitrary windows if needed.
- Phases exist on 1 Hz flights only: pieces and segments without phases leave only the phase
  statistics. Report how many remain.

**5. Estimate, reweighted to the piece** (stratum = piece, as `zones.standardized`). With unit
weights w (1 for the point estimate):
- **MSD.** Group side: `sum_g w_g MSD_g / sum_g w_g`. Control side: per piece
  `M_g = sum_s w_s MSD_s / sum_s w_s` over its controls, then `sum_g w_g M_g / sum_g w_g`.
  Logarithm and slopes **after** the means.
- **Phase ratios.** Group side: `sum_g w_g num_g / sum_g w_g den_g`. Control side: per piece the
  mean num and den of its controls weighted by w_s, then `sum_g w_g mean_num_g / sum_g w_g mean_den_g`.
- A piece enters only when **both sides have positive weight**.

**6. Two-way bootstrap** (pigeonhole, Owen 2007), written in `isolated.py`, without changing
`decompose` or `zones`.
- **Pilot**: column `pilot` of the catalog
  `/Volumes/SSD_DATA/xc-viewer-data/paragliders/catalog/catalog.csv`, joined on `flight_id`; a
  flight with no pilot is a pilot of its own.
- **Block**: the current one of `decompose.flight_table` (episodes and site-days).
- **Each draw**: `c_p ~ Multinomial(n_pilots)` and `c_k ~ Multinomial(n_blocks)`, independent;
  every piece and control segment gets `w = c_pilot * c_block` of its flight; redo step 5.
- **The same draws for every statistic, level and series** (fixed seed, `tamsd.SEED`).
- `N_BOOT = 1000`; band = 5th and 95th percentiles.

**7. Stability across years.** For each year (date of the piece), redo the point estimate leaving
out the pieces of that year and the controls of that year. Report, for Delta H_loc(1000 s) and
every phase Delta, the min and max over the years next to the band.

**8. Share explained.** With `route.baseline(...)`, the same pieces against the solo flights of
their site-day and hour. Share explained at 1000 s with the formula of `\RouteExplained`: find it
in `report/facts.py` and reuse it, do not rewrite it.

**9. Output.**
- Match cache `CACHE / f"isolated_matches{tag}.parquet"`: `level, piece, solo, i, j, cover`.
- `OUT / f"isolated{tag}.json"`:
  - counts: pieces with a match, matches, distinct solo flights, distinct pilots per side, median
    matches per piece, pieces and segments with phases, pieces of the long series and
    `LONG_TAU_MAX`;
  - per statistic: group, control, Delta, band, min/max over the years;
  - the share explained.
- Results are reported as figures, each in the form that best shows what it reports: curves for
  what depends on the lag, forest plots for differences with their bands, bars or points for
  counts. Numbers stay in the JSON; the figures carry the reading.
- Figure `isolated_dhloc{tag}.pdf`, its own figure: Delta H_loc(tau) against the lag (log axis),
  one curve per level with its 5-95% band, the main series solid up to 1000 s and the long series
  dashed up to `LONG_TAU_MAX`, a line at zero, and the min/max over the years as a thin envelope.
- Figure `isolated{tag}.pdf`:
  - left: H_loc(tau) of group and control, one pair of curves per level, long series dashed;
  - right: forest plot of the Deltas of the non-redundant statistics (`decompose.DRAWN` plus time
    in search), per level, with bands and the min/max over the years. The others stay in the JSON
    only.

**10. Run.**
1. `uv run python isolated.py --sample 2000` with `/usr/bin/time -l`.
2. `uv run pytest` from the root of xc-lab, to check the shared functions did not change.
3. The full run, only if the peak is acceptable.

**11. README.** Add a line for `isolated.py` to the README of the folder.

## Constraints

- Style of route.py: constants at the top, short docstrings that state the contract, no long
  comments.
- Heavy caches only in `CACHE` on SSD_DISANTE; JSON and figure in `out/<run>/`; temporary trials
  in the scratchpad.
- Do not touch `report/` or the thesis. No commit unless asked.
- Before each edit and each command, say in plain words what it does. One small change at a time.

## Report at the end

Give the paths of the figures first, then, per level:
- pieces with a match, pieces with phases, pieces of the long series and `LONG_TAU_MAX`;
- Delta H_loc(100 s), Delta H_loc(1000 s) and Delta H_loc at `LONG_TAU_MAX`, each with band and
  min/max over the years;
- the share explained at 1000 s, against the 71% of route.py;
- the Delta of every phase statistic, with its band.
