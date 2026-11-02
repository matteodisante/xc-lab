# Plan: the comparisons report, group against solo step by step

Folder: `xc-lab/analyses/groups/`. Report: `vs-solo/comparisons/`, numbers and tables from
`vs-solo/code/comparisons.py`. Everything runs on the open flights: every script with `--open` (declared
open, opening above `openflights.MIN_OPENING` = 0.1); results in `out/<run>/open_min0.1/`, caches in
`analyses_open_min0.1/`.

## Goal

The report holds the premise (declared open flights, their opening, the threshold 0.1) and the raw
H_loc of S, G, A. Extend it to four comparisons of group time with solo flight:

1. **raw**: S, A, G as they are;
2. **matched**: pieces of G against solo segments on the same route, season, hour and wing;
3. **decomposition**: how much of the raw gap route, season and hour, and wing each account for, and how
   much is left to the group (Shapley);
4. **before, during, after**: the same flight around its group time.

## Rules shared by every point

**Populations.** S, G, A of `tamsd.py`. Points 2 and 3 draw controls from the pool of `route.py`: open
flights with no contact of any length (`route.eligible`), on another date than the piece. The isolation
of `isolated.py` (rule 6, nobody within 10 km) is off in the main results and on in one robustness line
of point 2.

**MSD datasets.** Lag grid `longlags.LONG_LAGS`. A dataset to the lag T keeps the pieces (segments,
windows) of at least T / `SUPPORT` s and measures them at every lag up to T: the population does not
change along a curve.

- dataset `1000`: T = 1000 s (pieces of at least 1250 s), the pieces of `tamsd.py`;
- dataset `2760`: T = 2760 s (at least 3450 s), `longlags.FIXED_LAG`;
- point 1 also keeps the dataset `longest` of `longlags.py`;
- point 4: windows of 2400 s, T = 1840 s (the largest lag of the grid within 0.8 x 2400 s).

Quoted lags: H_loc(1000 s) on dataset `1000`, and H_loc(1840 s) on
dataset `2760` (its +-0.25-decade window, 1035 to 3272 s, is cut at 2760 s, `tamsd.local_h`: say so in the captions).
Point 4 quotes H_loc(1000 s) only.

**MSD figures.** For every dataset of every point, two separate figures:

- `<name>_msd.pdf`: MSD(tau) log-log, one curve per population with its 5-95% band; in the legend the
  population, its flights (or pieces) and H fitted on every lag of the curve, from 10 s to T, with its
  band (`tamsd.fit_h`; `longlags.curve` already returns `H` and `H_band`);
- `<name>_hloc.pdf`: H_loc(tau) with bands, a dotted line at 0.5, same colours.

Add `plot_msd(ax, curves)` to `curves.py` beside `plot`, and call `plot` with `inset=None`; every point
draws with these two.

**Phase statistics.** Seven, from Vilpellet's runs (1-Hz flights only), each run placed in the unit that
holds its middle (`survival.assign`, `isolated.window_sums`):

| statistic | numerator n | denominator d | terms of |
|---|---|---|---|
| time in climb (%) | time in climbs | phase time | `budget.columns` |
| time in search (%) | time in searches (found + vain) | phase time | `budget.columns` |
| searches that end in a climb (%) | searches followed by a climb | searches | `budget.columns` |
| climbs per hour | climbs | phase time (h) | `budget.columns` |
| climb rate (m/s) | height gained in climbs | time in climbs | `performance_terms.parquet` |
| glide speed (km/h) | straight-line distance of glides | time in glides | `performance_terms.parquet` |
| height lost per km of glide (m/km) | height lost in glides | distance of glides | `decompose.SINK` |

The phase time T_u of a unit u is the sum of the durations of its runs (column `s` of `budget.columns`).

**Estimator: average first, ratio after**, one weight per unit, as the MSD. Unit: the flight in point 1
(its runs on its pieces of dataset `1000` in that population), the piece of group time or the control
segment in points 2-3, the window in point 4. With w_u the weight of the unit (1 for the point estimate,
the bootstrap counts for a draw):

    r_u = n_u / T_u ,   q_u = d_u / T_u
    stat = sum_u w_u r_u / sum_u w_u q_u

For the two time shares and the climbs per hour q_u is a constant and stat is the mean over units of
the unit's share or rate; for the other four it does not divide by the few events of one unit (one
search gives 0% or 100%). In points 2-3 the control side takes, per piece, the mean of r and of q over
its controls, then the same ratio: the reweighting of `isolated.engine`, fed with r and q instead of n
and d. The JSON also keeps the ratio of sums, sum n / sum d (the estimator of `budget.py`), as a check;
the report uses the new one only.

**Survival of the phase durations.** Climb, search (only those followed by a climb, `survival.counted`)
and transition; the duration is the number of fixes (1 Hz, so seconds), trimmed by `durations.trim`.
P(tau_phase > tau) log-log, one curve per population, pointwise 5-95% bands from the bootstrap draws of
the point on a log grid of tau. No fit, no mu, no threshold line. Every run weighs 1; in points 2-3 the
runs of a control segment weigh 1 / (number of controls of its piece), so that the control side has the
composition of the pieces. Same units as the phase statistics. One figure per point,
`<name>_survival.pdf`, three panels.

**Bands.** 5-95% of `N_BOOT` = 1000 draws, the same draws for every statistic, curve and cell of a
point: point 1 the dependency blocks of `tamsd.py`; points 2-3 pilots x blocks
(`isolated.bootstrap_weights`); point 4 `decompose.block_counts`. Points 2-3 also give the min-max over
the years, leaving one year out (`isolated.year_weights`).

**Forest plots.** Point 1: the values of the seven statistics per population (the form of
`budget.plot`). Points 2-4: the differences, Delta H_loc(1000 s), Delta H_loc(1840 s) where it exists,
and the seven Delta, with bands and the min-max over the years where present.

## Step 0. Fixes and caches

1. `survival.py` (line 103) and `prepost.py` (line 192) read `flights.parquet` directly and ignore
   `--open`: with `--open` they would write results of every flight into `open_min0.1/`. Read
   `tamsd.read_flights()` instead (`with_takeoffs` around it in prepost). Grep the other scripts of this
   plan for the same.
2. With `--open` every cache must land in `analyses_open_min0.1/`: `performance_terms.parquet` (one pass
   over the fixes, about 12 min), `route_tracks.npz`, the matches. The archive index
   (`SHARED/archive_index.npz`) stays shared.
3. The estimator: a function in `budget.py` that turns rows (unit, terms) into r and q per unit; keep
   `budget()` as it is. Tests in `test_groups.py`: on synthetic units with known shares the result is
   the mean of the shares; a unit with a single search does not move the result more than any other.

## Point 1. Raw

- `tamsd.py --open`, `longlags.py --open`: datasets `1000`, `2760`, `longest`, each with its MSD and
  H_loc figures.
- The seven statistics per population on the pieces of dataset `1000`, forest plot.
- Survival of S, A, G.
- `out/<run>/open_min0.1/raw.json`.

## Point 2. Matched

Extend `isolated.py`; `route.py` keeps its own results.

1. **Calipers as parameters.** Strict: month +-1 (circular), local hour +-1 (middles of piece and
   segment), the same wing class (A, B, C, D, CCC of the catalog; a flight with no class leaves). Loose:
   month +-2, hour +-2, the same level (beginners A, B; experienced C, D, CCC). Route (rho = 500 m, cover
   0.8, same direction, `route.segment`) and other date unchanged. `isolated.py` gets its own constants;
   `route.MONTHS`, `route.HOURS` stay for `route.py`.
2. **Match once, on the route only.** One match table with, per match: piece, solo piece, i, j, cover,
   Delta month, Delta hour, wing class of both sides, isolation flag (`solo_flags`). Every caliper, and
   every cell of point 3, is a filter on it. Drop the season pre-filter of `matches()` and the split by
   level (a cell without W matches across levels). First measure with `--sample 500` under
   `/usr/bin/time -l`; if the full run would pass about 7 GB, split into tiles of 0.5 deg as in step 1 of
   `isolated_plan.md`.
3. **Counts before any Delta.** A table of the pieces with at least one control, per level, on datasets
   `1000` and `2760`, for the calipers in this order: (+-1 month, +-1 h, class), (+-1, +-1, level),
   (+-2, +-2, class), (+-2, +-2, level). The main caliper is the first with at least `MIN_PIECES` = 200
   such pieces per level on dataset `2760`. Print the table, save it in the JSON, and go on with the rule.
4. **A matched too.** Build `route.TRACKS` with the pieces of A (`group_solo` of the flights with group
   time) as well, and match them as the pieces of G. Check that `route.py` selects by population, so
   that its results do not change. Report Delta(G - S | match), Delta(A - S | match) and their
   difference.
5. **Lags.** Main series on dataset `1000`, long series on dataset `2760`: replace `LONG_CANDIDATES` and
   `LAGS_EXTENDED` by `longlags.LONG_LAGS` and T = 2760 s; if a level keeps fewer than `MIN_PIECES`
   pieces there, the largest lag of the grid where it keeps them, as `long_tau` does now.
6. The seven statistics with the new estimator, survival with the weights, the figures of the shared
   rules.
7. **Robustness**, for Delta H_loc(1000 s), Delta H_loc(1840 s) and the seven Delta: the other calipers of
   the table; isolation on; rho 300 and 1000 m, cover 0.7 and 0.9 (rho needs a new match table, cover
   is a filter). One forest plot, `matched_robustness.pdf`, one row per line.

## Point 3. Decomposition

Factors R (route), T (season and hour), W (wing), with the caliper chosen in point 2; F = {R, T, W}. For
each of the 8 subsets C of F, Delta(C) = G minus the controls that satisfy only the conditions of C, for
Delta H_loc(1000 s), Delta H_loc(1840 s) and the seven Delta. Delta(empty set) is the raw gap on these
pieces, Delta(F) the result of point 2.

So that the cells compare:

- the same pieces of G in every cell: those with at least one control in the cell F;
- the same pool in every cell (`route.eligible`, open flights, other date);
- cells with R: the controls of the match table that satisfy C;
- cells without R: the control is a window of a pool piece as long as the piece of G. If T is in C, the
  pool piece is within the month caliper and the window is placed at random among the positions whose
  middle is within the hour caliper; otherwise at random. If W is in C, same wing class (or level). At
  most K = 50 controls per piece, drawn with `tamsd.SEED` among the pool pieces that satisfy C and are
  long enough. MSD from the 10-s positions of `route.TRACKS` (`isolated.msd_of`), phase terms by
  `window_sums`;
- the same bootstrap draws in every cell.

Shapley value of each factor k, on the point estimate and on every draw (hence the bands):

    phi_k = sum over C in F without k of  |C|! (2 - |C|)! / 3!  x  [Delta(C) - Delta(C with k)]
            (weights 1/3 for |C| = 0, 1/6 for |C| = 1, 1/3 for |C| = 2)

    Delta(empty set) = Delta(F) + phi_R + phi_T + phi_W        (exact)

phi_k is the part of the raw gap that comes from G and S differing in k.

Tests: on any synthetic Delta(C) the identity holds to 1e-12; with additive Delta(C) = g + sum of a_k
over the k not in C, phi_k = a_k.

Output: `decomposition.json` with the 8 Delta(C) and the phi per statistic, with bands. Figures:

- `decomposition.pdf`: per statistic, Delta(empty set) split into phi_W, phi_T, phi_R and the part left,
  Delta(F), each with its band;
- MSD and H_loc figures of G and of the controls of the 8 cells, datasets `1000` and `2760`, colours by
  the number of conditions.

The report gives Delta(empty set) beside the raw gap of point 1: they differ (pieces, pool, windows).

## Point 4. Before, during, after

`prepost.py --open --window 2400`:

- lags up to 1840 s: `tamsd.measure(d, flights, wins, lags, shortest)` with
  `lags = LONG_LAGS[LONG_LAGS <= 1840]` and `shortest = 2400`;
- the seven statistics with the new estimator, unit = window;
- MSD and H_loc figures of pre, start, end, post, and of the two control pairs;
- differences D_in = start - pre, D_out = end - post, post - pre, the controls (ctrlA2 - ctrlA1,
  ctrlS2 - ctrlS1) and, on the events that have all four windows,

      g   = (D_in + D_out) / 2      the group effect, free of a drift linear in time
      d w = (D_in - D_out) / 2      the drift over one window

  each with its band;
- forest plot of Delta H_loc(1000 s) and the seven Delta for D_in, D_out, post - pre, g, d w and the
  controls; beside g, Delta H_loc(1000 s) of point 2 (the same lag).

## Report

Rebuild `vs-solo/comparisons/` (sections in `sections/`, every number a macro of `comparisons.py`):

- keep aim, flights and measures, declared open flights, opening and threshold;
- "Flights and measures" gets the table of the seven statistics (numerator, denominator, unit, runs
  counted, unit of the average) and the estimator;
- new sections: raw (point 1), matched (2), decomposition (3), before and after (4). Each opens with its
  question and closes with "What we keep"; every table and figure gives its flights or pieces; every
  caption names the dataset, the units and the estimator;
- appendices, one per tool, with formulas and a small worked example: TA-MSD and ensemble MSD; fitted H
  and H_loc, with the window cut at the edge; datasets and the tau / 0.8 rule; the phase statistics,
  average then ratio; the survival function and its weighted version; route matching (coverage,
  direction, segment) and calipers; the reweighting to the piece; the bootstrap of dependency blocks and
  the two-way pilot x block bootstrap; leave one year out; the Shapley decomposition (definition, the
  weights for three factors, the exact identity, why the order does not matter, the additive example);
  the drift-free estimator of point 4;
- about 10 core pages plus appendices; figures first, little text.

## Order and runs

Step 0, point 1, point 4, point 2, point 3, report; one point per session. Before every full run,
`--sample` under `/usr/bin/time -l` (8.6 GB of RAM: never two heavy runs at once). `uv run pytest` from
the root of xc-lab after each step. Each step ends by printing its key numbers and the paths of its
figures.

## Constraints

- Style of the folder: constants at the top, docstrings that state the contract, short comments.
- New arguments get defaults that leave the current outputs unchanged (`budget()`, `curves.plot`,
  `tamsd.measure`, `route.py`).
- Tests in `test_groups.py` for every new function whose numbers the report quotes: the estimator, the
  weighted survival, the windows without route, Shapley, the drift-free estimator.
- Heavy caches on SSD_DISANTE (`CACHE`), JSON and figures in `out/<run>/open_min0.1/`, trials in the
  scratchpad.
- No commit unless asked. Before each edit and each command, say in plain words what it does; one small
  change at a time. A logbook entry at the end of each session.
