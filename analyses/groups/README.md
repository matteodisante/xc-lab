# Groups

The motion of paragliders and hang gliders alone and in a group: TA-MSD with its exponent H
and local exponent H_loc, the largest groups one by one, and the MSD of each flight phase.

**Data.** The group labels of [`ideas/group-identification`](../../ideas/group-identification/),
run `d120_eh500_ez300_th45_g120_q0.1` (δ = 2 min, q = 0.1: a flight is in a group only in a
contact of at least τ* = 38.1 min; see its `parameter_tuning.pdf`), and the cleaned tracks of
soaring (tag `legacy-final`) on SSD_DISANTE. xc-dynamics at `d72e1b9` (not used).

**Results.** Figures and numbers are in `out/<run>/` of this folder (not in git), with the
`PARAMETERS.md` of the run: `tamsd.pdf`, `episodes.pdf`, `phase_msd.pdf` and their JSON. The
measured pieces (`pieces.npz`, `phase_pieces.npz`), too heavy for the repository, stay on the
disk in `/Volumes/SSD_DISANTE/xc-lab/group-identification/runs/<run>/analyses/`, so the figures
are redrawn without reading the fixes again. `--run NAME` picks another run; the run with the
starting values of the note (δ = 5 min, τ* = 89.5 min) has its own, with the same picture as
below.

**Populations**, after the departure and on loggers of at most 10 s, where the labels see every
instant:

- solo: flights never in a group;
- group, group time: flights with group time, only their stretches in a group;
- group, solo time: the solo stretches of the same flights, a control with the same pilots,
  days and loggers.

## TA-MSD, H and H_loc: `tamsd.py`

Horizontal TA-MSD against the lag τ (`tamsd.pdf`), with the conventions of the fixed
transport of chapter 3: positions on a 10-s grid, the lags of its common grid from 10 to
1000 s, a stretch enters only if it lasts at least 1000 s / 0.8 = 1250 s inside one segment, so
that no population changes with τ; equal weight per flight, each flight's origins pooled over
its stretches. H is half the slope of log MSD against log τ over all lags, with equal weight per
lag; H_loc the same within ±0.25 decades. Bands: 5–95% of 1000 bootstrap draws of the
dependency blocks of the note (flights tied by episodes; a solo flight is a block alone).

**Result.** Group flights have a larger H than solo flights, 0.916 against 0.899
(paragliders), and nearly the same in their solo time, 0.914: most of the difference comes from
the flights and days that have groups, not from flying in one. Flying in a group changes the
long lags: H_loc at 1000 s is 0.893 in group time against 0.860 for the same flights alone and
0.837 for solo flights. Below about 300 s the group time is slightly less ballistic than the
solo time of the same flights (H_loc(100 s) 0.927 against 0.932). Hang gliders show the same
pattern, with wide bands at long lags (418 group flights).

| | flights | H, 10–1000 s | H_loc(100 s) | H_loc(1000 s) |
|---|---|---|---|---|
| para, solo | 119 129 | 0.899 [0.898, 0.899] | 0.919 | 0.837 |
| para, group, group time | 33 435 | 0.916 [0.915, 0.917] | 0.927 | 0.893 |
| para, group, solo time | 29 611 | 0.914 [0.913, 0.915] | 0.932 | 0.860 |
| hang, solo | 5 555 | 0.881 [0.880, 0.882] | 0.905 | 0.817 |
| hang, group, group time | 418 | 0.899 [0.894, 0.903] | 0.914 | 0.860 |
| hang, group, solo time | 373 | 0.889 [0.885, 0.893] | 0.912 | 0.823 |

The group stretches all last at least τ*, the solo stretches only 1250 s; how much of the long
lags this explains is not checked.

**Coarser blocks: `blocks.py`.** The blocks above treat solo flights of one day as independent,
though they share the weather. `blocks.py` redoes the bands with the same pieces and seed and
coarser blocks: flights whose takeoffs lie within 3 km on the same day (chained, the departure
radius of the note) join one block, 64 128 blocks instead of 139 684; or all the flights of a
day, 6 209 blocks. No band of H or of H_loc at 100 and 1000 s widens by more than 0.002
(paragliders, solo, H_loc(1000 s): [0.836, 0.839] by day against [0.837, 0.838]), so the
comparisons above stand.

## The largest groups: `episodes.py`

The MSD of the five episodes with most flights (`episodes.pdf`), each over the group time of
its members, same conventions; the band resamples the members. A member enters with its pieces
of at least 1250 s on a logger of at most 10 s, so a few members drop out.

**Result.** H varies from group to group well beyond the band of each: from 0.879 to 0.926 for
the paragliders. The three hang gliders of 2013-04-24 give 0.861 [0.821, 0.890].

| episode | flights (largest configuration) | duration | paragliders in the MSD | H |
|---|---|---|---|---|
| 2013-04-24 | 53 (16) | 5.8 h | 49 | 0.926 [0.920, 0.931] |
| 2015-04-11 | 52 (19) | 5.1 h | 52 | 0.912 [0.907, 0.917] |
| 2016-03-19 | 49 (22) | 5.2 h | 49 | 0.879 [0.868, 0.888] |
| 2022-03-26 | 47 (16) | 3.8 h | 47 | 0.883 [0.869, 0.896] |
| 2018-04-21 | 47 (19) | 6.5 h | 47 | 0.923 [0.916, 0.929] |

## MSD per phase: `phase_msd.py`

The MSD of transitions, searches and climbs (`phase_msd.pdf`, with their local exponent),
by the method of [`analyses/phases/msd.py`](../phases/): Vilpellet's pieces longer than 180
fixes, his 6-fix smoothing and his median over pieces weighted by the T − Δ pairs; lags
drawn up to 360 s where at least 100 pieces reach them. Here a phase run is first cut to the
stretches of its population, so a group flight's runs count only in its group time. Phases
exist on 1 Hz flights only. Slopes of log MSD against log Δ, equal weight per drawn lag:
transitions over every drawn lag, searches over 100–360 s, where their local exponent
levels off near Vilpellet's Δ^0.6 guide. Bands: 5–95% of 200 bootstrap draws of the blocks.

**Result.** Transitions are slightly more ballistic in group flights, 1.977 against 1.962
for solo flights (paragliders), and equally so in their solo time, 1.977. Search slopes do not
separate: 0.606 in group time, 0.715 in the solo time of the same flights, 0.619 for solo
flights, with bands that overlap. Hang gliders have too few group searches for a slope, and
their solo searches lose their pool beyond 180 s (as in `analyses/phases`), so their search
slope means nothing.

| | flights (transitions, searches) | transition slope | search slope, 100–360 s |
|---|---|---|---|
| para, solo | 80 186, 29 243 | 1.962 [1.961, 1.962] | 0.619 [0.585, 0.668] |
| para, group, group time | 23 944, 5 084 | 1.977 [1.976, 1.978] | 0.606 [0.537, 0.685] |
| para, group, solo time | 22 843, 9 003 | 1.977 [1.976, 1.978] | 0.715 [0.647, 0.787] |
| hang, solo | 1 535, 663 | 1.963 [1.960, 1.966] | (pool lost beyond 180 s) |
| hang, group, group time | 121, 11 | 1.977 [1.966, 1.990] | – |
| hang, group, solo time | 114, 44 | 1.960 [1.952, 1.970] | – |

## Phase durations: `survival.py`

Survival of the durations of transitions, searches and climbs (`survival.pdf`), paragliders only,
by the method of [`analyses/phases/durations.py`](../phases/): a duration is the number of fixes
of a run (1 Hz), a search counts only when the next run is a climb, the 0.0001 and 99.9999
percentiles are cut. A run keeps its whole duration and goes to the population of the stretch
that holds its middle: cutting it where a flight enters or leaves a group would shorten the
durations measured. Clauset's x_min moves from one population to another, so μ is compared
above one threshold, the x_min of the solo flights, with the Hill estimate; bands: 5–95% of
1000 draws of the dependency blocks.

**Result.** The survivals coincide up to about 30 min, and so do the medians. Only the tail of
transitions in group time is heavier: above 31 min μ = 3.52 against 4.54 for solo flights and
4.37 for the same flights in their solo time, with bands that do not overlap. Long glides are
more frequent while flying in a group, as the long lags of the TA-MSD; searches and climbs do
not change. Whether part of this comes from the definition (a contact must last at least τ*,
which a long glide together meets easily) is not checked.

| | runs (flights) | median (s) | μ above the solo x_min |
|---|---|---|---|
| transition, solo | 1 388 316 (80 241) | 194 | 4.54 [4.43, 4.65] above 1857 s |
| transition, group time | 269 867 (24 097) | 196 | 3.52 [3.36, 3.72] |
| transition, group flights' solo time | 392 750 (23 208) | 213 | 4.37 [4.18, 4.58] |
| search, solo | 420 780 (72 277) | 51 | 4.02 [3.88, 4.16] above 495 s |
| search, group time | 87 614 (20 681) | 53 | 4.10 [3.81, 4.45] |
| search, group flights' solo time | 122 354 (20 163) | 53 | 4.04 [3.83, 4.30] |
| climb, solo | 1 258 867 (79 545) | 130 | 8.85 [8.48, 9.26] above 995 s |
| climb, group time | 268 561 (24 034) | 132 | 9.54 [8.51, 10.76] |
| climb, group flights' solo time | 362 061 (22 415) | 130 | 8.83 [8.02, 9.83] |

## Time budget: `budget.py`

How paragliders share their time among the phases (`budget.pdf`): the same runs and
populations as `survival.py`, ratios summed over the runs of each population, bands 5–95% of
1000 draws of the dependency blocks. A search in vain is one that the next run of the flight
does not turn into a climb.

**Result.** In group time pilots climb more and search in vain less: 33.0% of the time in climb
against 30.8% for solo flights and 29.1% for the same flights alone, 4.7% in a search in vain
against 6.3% and 5.7%, 7.03 climbs per hour against 6.51 and 6.21. A search ends in a climb
49.5% of the time in a group, 43.3% alone (45.9% for the same flights). A climb starts after a
search as often as alone (about a third), and that search lasts as long (`survival.py`): with
others around a thermal is still entered and centred, but fewer searches are wasted. The same
flights alone glide more and climb less than solo flights, a flight and day effect, as for H.

| | solo | group time | group flights, solo time |
|---|---|---|---|
| flights | 80 278 | 24 099 | 23 469 |
| time in transition (%) | 58.4 [58.4, 58.5] | 57.6 [57.3, 57.9] | 60.8 [60.7, 61.0] |
| time in climb (%) | 30.8 [30.7, 30.9] | 33.0 [32.8, 33.2] | 29.1 [28.9, 29.2] |
| time in a search that ends in a climb (%) | 4.4 [4.4, 4.4] | 4.7 [4.6, 4.8] | 4.4 [4.3, 4.5] |
| time in a search in vain (%) | 6.3 [6.3, 6.4] | 4.7 [4.6, 4.8] | 5.7 [5.6, 5.8] |
| climbs per hour | 6.51 [6.50, 6.52] | 7.03 [6.99, 7.06] | 6.21 [6.18, 6.23] |
| climbs that start after a search (%) | 33.5 [33.4, 33.6] | 32.7 [32.3, 33.1] | 33.8 [33.5, 34.0] |
| searches that end in a climb (%) | 43.3 [43.2, 43.4] | 49.5 [49.2, 49.7] | 45.9 [45.7, 46.1] |

## Climbs, glides and cross-country speed: `performance.py`

One pass over the cleaned fixes (`performance.pdf`). Climbs and transitions are the runs of
`survival.py`, read at their first and last fix: a climb gains the difference of altitude in its
duration, a transition covers the straight line between its ends and loses the difference of
altitude. Ratios summed over each population, so long phases weigh more: climb rate = total
gain / total climb time, glide ratio = total distance / total height lost. Cross-country speed:
the straight-line distance covered in 30 min from every origin of the 10-s pieces of at least
37.5 min (30 min / 0.8, as for the TA-MSD), averaged and divided by 30 min; it uses every logger
of at most 10 s (112 292 solo flights), the phases only the 1 Hz ones (80 242). Bands as above.

**Result.** Flying in a group improves the glides: glide ratio 12.9 in group time against 11.3
for the same flights alone and 10.4 for solo flights, at a similar speed (33.3, 32.8 and
30.1 km/h). Climbs are a little worse than for the same pilots alone: 1.26 m/s and 212 m per
climb against 1.33 m/s and 223 m (solo flights: 1.22 m/s, 207 m). The two balance:
cross-country speed is 20.0 km/h in group time and 20.0 km/h for the same flights alone, against
17.2 km/h for solo flights, so the faster progress of group flights is a flight and day effect.
Part of the better glides may come from the definition: two pilots in glide must keep within
45° of heading to stay in contact, so glides in a group are glides along the same line.

| | solo | group time | group flights, solo time |
|---|---|---|---|
| climb rate (m/s) | 1.22 [1.22, 1.22] | 1.26 [1.25, 1.27] | 1.33 [1.32, 1.34] |
| height gained per climb (m) | 207 [206, 207] | 212 [210, 214] | 223 [221, 225] |
| glide speed (km/h) | 30.1 [30.1, 30.2] | 33.3 [33.1, 33.5] | 32.8 [32.6, 32.9] |
| glide ratio | 10.4 [10.4, 10.4] | 12.9 [12.8, 13.0] | 11.3 [11.2, 11.4] |
| cross-country speed over 30 min (km/h) | 17.2 [17.2, 17.3] | 20.0 [19.8, 20.3] | 20.0 [19.8, 20.1] |

## Tests and rebuild

`test_groups.py` checks the FFT against the direct average, the populations on a synthetic
membership, the 1250-s rule and the segments, MSD = v²τ² on a straight glide, H and H_loc on
power laws, the block bootstrap, the cut of the phase runs, the slopes on power laws, the sites
and blocks of `blocks.py`, the runs that count and their population in `survival.py`, the
terms of `budget.py`, the climbs, glides and 30-min distances of `performance.py` on a synthetic
flight, and every number quoted here.

```
uv run python tamsd.py [--run NAME]      # from this folder, needs SSD_DISANTE; --recompute: about 5 min
uv run python episodes.py [--run NAME]   # after tamsd.py
uv run python phase_msd.py [--run NAME]  # --recompute: about 6 min; the bootstrap of the slopes about 45 min
uv run python blocks.py [--run NAME]     # after tamsd.py, about 30 s
uv run python survival.py [--run NAME]   # about 30 s
uv run python budget.py [--run NAME]     # about 20 s
uv run python performance.py [--run NAME]  # --recompute: about 12 min (one pass over the fixes)
uv run pytest analyses/groups            # from the root of xc-lab
```
