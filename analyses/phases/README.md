# Phases

Statistics of the flight phases (climb, search, transition) on our paragliders and hang
gliders, computed as in Vilpellet's paper (arXiv:2601.01293) and compared with his values.

**Data.** The phase runs that soaring (tag `legacy-final`) wrote on SSD_DISANTE with
Vilpellet's segmenter over the cleaned fixes (`derived/segmentation/vilpellet/phase_segments.parquet`).
His segmenter labels only 1 Hz segments of flights with at least 3600 fixes, so every figure
is on 1 Hz flights, as in his paper. xc-dynamics at `aaf9a6d` (not used).

**Populations.** Each figure has two rows:

- all 1 Hz flights: every labelled run, 104 636 paraglider and 1 670 hang-glider flights;
- C_10000 (1 Hz): the runs inside the segments that admit a flight to the fixed cohort of
  chapter 3 (a retained segment of at least 12 500 s,
  `derived-audit/chapter3-fixed-20260917/*/cohort-10000.json`). Of its 46 273 paraglider and
  2 326 hang-glider members, 33 450 and 690 are labelled.

The run selection (which searches count) reads whole flights; C_10000 then keeps its segments.

## Phase durations: `durations.py`

Survival of τ_transition, τ_search, τ_climb and the tail exponent μ (`out/survival.pdf`, his
panels c, f, i), and μ above a moving cutoff (`out/hill.pdf`). Updated 2026-10-20.

**Result.** With his estimator, μ ≈ 4.3 for transitions (both disciplines), 4.27 and 3.29 for
search, 10.7 and 5.45 for climbs: the same order as his values. The tails are not power laws:
μ above a cutoff keeps rising with the cutoff, so each μ describes only the 0.03–5 % of runs
above the xmin that Clauset's method picks. Only paraglider transitions level off, at μ ≈ 4.3
above about 1500 s. Climbs bend down like an exponential. In C_10000 the curves of μ against
the cutoff are almost those of all flights; its μ differ mainly because Clauset's xmin moves
(hang-glider transitions: 740 s instead of 1010 s, μ 3.38 instead of 4.34).

| | μ para | μ hang | μ Vilpellet (para, hang) | xmin para, hang (s) | runs above xmin, para, hang |
|---|---|---|---|---|---|
| transition | 4.32 | 4.34 | 3.93, 4.79 | 1851, 1010 | 8482, 823 |
| search | 4.27 | 3.29 | 3.88, 2.25 | 648, 192 | 1772, 714 |
| climb | 10.66 | 5.45 | 8.28, 5.33 | 1161, 608 | 707, 1087 |
| C_10000 transition | 4.14 | 3.38 | | 1648, 740 | 8119, 1460 |
| C_10000 search | 3.84 | 4.02 | | 478, 243 | 2478, 125 |
| C_10000 climb | 9.08 | 5.92 | | 1032, 637 | 966, 529 |

Method, as in his code:

- τ is the number of fixes in a run (1 fix = 1 s); a run ends at a phase change or a time gap.
- A search counts only when the next run of the flight is a climb; every climb and transition counts.
- Values outside the 0.0001 and 99.9999 percentiles are dropped, then `powerlaw.Fit` (Clauset
  et al.) picks xmin by minimum Kolmogorov–Smirnov distance and fits α by maximum likelihood;
  μ = α − 1 is the exponent of the survival, P(τ > t) ~ t^−μ.
- `powerlaw` is pinned to 1.5: version 2.0 bounds α to [0, 3], so μ could not exceed 2.
- Plotted points: the empirical survival resampled at 30 log-spaced τ. The dashed line is the
  fitted law from xmin (dotted vertical line).

## MSD per phase: `msd.py`

Horizontal mean squared displacement within each phase against the lag Δ (`out/msd.pdf`, his
figure "vario_regime" without sailplanes), and its local exponent d log MSD / d log Δ
(`out/msd_slopes.pdf`). Updated 2026-10-20.

**Result.** Transitions stay ballistic up to 357 s (local exponent ≥ 1.89 para, 1.85 hang).
Climbs drop to an exponent of 0.45 (para) and 0.15 (hang) at 17 s, about one turn of the
spiral, then rise again (1.59 and 1.44 at 100 s) as the climb drifts. Searches slow down to
0.70 and 0.69 at 100 s, near his Δ^0.6 guide. √MSD at 1 s gives the horizontal speed of each
phase: 10.4, 8.6, 9.4 m/s for paragliders (transition, search, climb), 17.6, 13.5, 14.8 m/s
for hang gliders. C_10000 gives the same picture, slightly faster: speeds 10.8, 8.7, 9.5 and
18.6, 13.8, 15.2 m/s; exponents at 100 s 1.97, 0.74, 1.57 (para) and 1.96, 0.72, 1.47 (hang);
the climb minimum 0.47 and 0.15 at 17 s.

The run pool of this MSD changes with Δ only above 180 s: every run is longer than 180 fixes,
so up to there all of them contribute, beyond it only the longer ones. C_10000 fixes the
flights, not this.

Method, as in his code:

- Positions: east and north of the cleaned fixes, smoothed by his 6-fix Gaussian (σ = 1 fix)
  within each preprocessing segment; a run touching the unsmoothed ends is skipped, as in his code.
- Runs longer than 180 fixes; search runs whose next phase is a transition are dropped.
- Per run, the time-averaged MSD(Δ) = mean_i |r(i + Δ) − r(i)|² for Δ < min(T, 600);
  at each Δ, the median over runs weighted by the T − Δ pairs of each run.
- Plotted points: 30 log-spaced Δ up to 360 s, log MSD interpolated in log Δ. The guides Δ²
  and Δ^0.6 are placed on the curves, as his were by hand.

Differences from his figure:

- MSD is computed at 72 log-spaced lags (every second up to 18 s) instead of every second up
  to 599 s, to fit the paraglider runs in memory.
- His paraglider curve averages the curves of the wing classes; ours takes all paragliders at once.
- A lag is drawn only when at least 100 runs reach it. He cut the hang-glider search by hand
  at 180 s: up to there all 764 runs contribute, beyond it only the longest searches, and the
  curve bends down (exponent below 0 after about 250 s).

## Tests and rebuild

`test_durations.py` checks the selection rules and their order with the cohort, the estimator
on a synthetic Pareto, the cohort sizes, every number of the table, and the stored runs of one
hang-glider flight against the legacy segmenter.
`test_msd.py` checks the FFT against the direct average, the run selection, his weighted
median, a straight glide (MSD = v²Δ²), every number quoted above, and that the hang-glider
result in `out/msd.json` is reproduced from the disk.

```
uv run python durations.py           # from this folder, needs SSD_DISANTE; reuses out/fits.json
uv run python durations.py --refit   # refits too: about 10 min
uv run python msd.py                 # reuses out/msd.json
uv run python msd.py --recompute     # reads all fixes again: about 6 min
uv run pytest analyses/phases        # from the root of xc-lab
```
