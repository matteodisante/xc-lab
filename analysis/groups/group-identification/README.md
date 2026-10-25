# Groups

The motion of paragliders and hang gliders alone and in a group: TA-MSD with its exponent H
and local exponent H_loc, the largest groups one by one, and the MSD of each flight phase; then
the same by the number of pilots of the group, pooled and on chosen site-days.

**Data.** The group labels of [`ideas/group-identification`](../../../ideas/group-identification/),
run `d120_eh500_ez300_th45_g120_q0.1` (δ = 2 min, q = 0.1: a flight is in a group only in a
contact of at least τ* = 38.1 min; see sections 5–6 of its `group_identification.pdf`), and the cleaned tracks of
soaring (tag `legacy-final`) on SSD_DISANTE. xc-dynamics at `d72e1b9` (not used).

**Results.** Figures and numbers are in `out/<run>/` of this folder (not in git), with the
`PARAMETERS.md` of the run: `tamsd.pdf`, `episodes.pdf`, `phase_msd.pdf` and their JSON, and the
`size_*` and `siteday_*` figures of `size.py`. The measured pieces (`pieces.npz`,
`phase_pieces.npz`, the `size_*` ones), too heavy for the repository, stay on the
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
pattern, with wide bands at long lags (418 group flights). `decompose.py` separates these
differences by stratification: the long-lag effect survives with the flight fixed and with the
site-day and hour fixed; the short-lag one vanishes at the same hour, since the solo time of group
flights falls at other hours.

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

| episode | flights (largest group) | duration | paragliders in the MSD | H |
|---|---|---|---|---|
| 2013-04-24 | 53 (16) | 5.8 h | 49 | 0.926 [0.920, 0.931] |
| 2015-04-11 | 52 (19) | 5.1 h | 52 | 0.912 [0.907, 0.917] |
| 2016-03-19 | 49 (22) | 5.2 h | 49 | 0.879 [0.868, 0.888] |
| 2022-03-26 | 47 (16) | 3.8 h | 47 | 0.883 [0.869, 0.896] |
| 2018-04-21 | 47 (19) | 6.5 h | 47 | 0.923 [0.916, 0.929] |

## MSD per phase: `phase_msd.py`

The MSD of transitions, searches and climbs (`phase_msd.pdf`, with their local exponent),
by the method of [`analysis/phases/msd.py`](../../phases/): Vilpellet's pieces longer than 180
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
their solo searches lose their pool beyond 180 s (as in `analysis/phases`), so their search
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
by the method of [`analysis/phases/durations.py`](../../phases/): a duration is the number of fixes
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
does not turn into a climb. The figure draws two of the seven rows, time in climb and searches
that end in a climb; the others are explained by these and by the statistics of
`performance.py` ([which are drawn](#which-statistics-are-drawn-redundancypy)), and stay in the
table and in `budget.json`.

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

One pass over the cleaned fixes (`performance.pdf`, which draws climb rate, glide speed and
glide ratio; the height gained per climb and the cross-country speed are explained by the others,
[see below](#which-statistics-are-drawn-redundancypy), and stay in the table and in
`performance.json`). Climbs and transitions are the runs of
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

These pooled ratios weight each flight by its time in the population. Compared flight by flight,
or at the same site-day and hour (`decompose.py`), two readings change: the climb rate is
slightly higher in a group, not lower, and the cross-country speed is higher in a group than for
the same flights alone. The pooled comparison is a composition effect (see the report).

| | solo | group time | group flights, solo time |
|---|---|---|---|
| climb rate (m/s) | 1.22 [1.22, 1.22] | 1.26 [1.25, 1.27] | 1.33 [1.32, 1.34] |
| height gained per climb (m) | 207 [206, 207] | 212 [210, 214] | 223 [221, 225] |
| glide speed (km/h) | 30.1 [30.1, 30.2] | 33.3 [33.1, 33.5] | 32.8 [32.6, 32.9] |
| glide ratio | 10.4 [10.4, 10.4] | 12.9 [12.8, 13.0] | 11.3 [11.2, 11.4] |
| cross-country speed over 30 min (km/h) | 17.2 [17.2, 17.3] | 20.0 [19.8, 20.3] | 20.0 [19.8, 20.1] |

## Which statistics are drawn: `redundancy.py`

`budget.pdf` and `performance.pdf` draw only the statistics that carry information the others
do not; the rest stay in the tables above and in their JSON. Unit: one population (solo, group
time, solo time of the group flights) on one site-day, with at least 25 flights in both tables:
119 units (11 solo, 55 group time, 53 solo time), each statistic a ratio of sums as in the
figures. A statistic is redundant when the others explain most of it across the units: the R² of
its linear regression on them, all standardised. Backward elimination drops the statistic most
explained by the others until every R² is at most 0.5. The four shares of time sum to 100%
exactly, so one of them always goes first (the first of the table on a tie).

**Result.** Five statistics remain, two of the budget (time in climb, searches that end in a
climb) and three of the performance (climb rate, glide speed, glide ratio); each is explained by
the other four at most to 0.41. The others, in the order they go:

| dropped | R² when dropped | R² on the five kept |
|---|---|---|
| time in transition (%) | 1.00 | 0.83 |
| height gained per climb (m) | 0.98 | 0.91 |
| climbs that start after a search (%) | 0.93 | 0.77 |
| cross-country speed over 30 min (km/h) | 0.93 | 0.90 |
| time in a search in vain (%) | 0.88 | 0.71 |
| climbs per hour | 0.72 | 0.71 |
| time in a search that ends in a climb (%) | 0.65 | 0.65 |

The cross-country speed and the height per climb are almost functions of the five (0.90, 0.91);
the share in a search in vain keeps 29% of its variance of its own, and is a different
reading of the same conversion of searches into climbs. The units vary mostly with the weather,
so "explained" means across site-days. The choice is not unique: on the 55 units of group time
alone the elimination keeps the share in a search in vain in place of the climb rate, and on the
53 of solo time it keeps climbs per hour and neither the climb rate nor the glide ratio; time in
climb, searches that end in a climb and glide speed are in every set.

## By the number of pilots: `size.py`

The group time of `tamsd.py` is cut where the group changes, so that every stretch has
one number of pilots, and each goes to a population: 2, 3, 4, 5–6 and 7+ pilots. A flight with
stretches of several sizes is in several populations, one weight each. The solo and solo-time
stretches are those of the scripts above; only the group stretches were measured again, in three
passes over the fixes (`size_*` in the cache folder, about 8 min). Every figure draws the solo
flights, the sizes and, in the pooled ones, a control for each size: the solo time of the flights
that have it (dashed). A population enters only with 15 flights in at least 5 blocks of the note;
the stretches of a group must still last 1250 s for the TA-MSD, so the larger
groups, which are shorter, lose more of their time.

**TA-MSD** (`size_tamsd.pdf`, paragliders; hang gliders have only 2 and 3 pilots). H and
H_loc(1000 s) rise with the number of pilots, while the same flights alone stay at 0.86–0.87
whatever the size: the dependence on size belongs to the group time, not to the flights. A
group of n pilots must keep all of them within contact for at least τ*, so the larger
ones are those that fly one line together; how much of the rise this selection explains is not
checked.

| pilots | flights | H | H_loc(1000 s) | H_loc(1000 s), same flights alone |
|---|---|---|---|---|
| solo | 119 129 | 0.899 | 0.837 | |
| 2 | 26 411 | 0.917 | 0.889 | 0.861 |
| 3 | 9 480 | 0.923 | 0.906 | 0.867 |
| 4 | 4 073 | 0.924 | 0.910 | 0.865 |
| 5–6 | 2 686 | 0.925 | 0.915 | 0.870 |
| 7+ | 1 142 | 0.933 [0.926, 0.940] | 0.944 [0.933, 0.953] | 0.871 [0.860, 0.882] |

Hang gliders, H_loc(1000 s): 0.857 [0.842, 0.872] with 2 pilots (385 flights) and 0.856
[0.807, 0.895] with 3 (60), against 0.817 solo.

**Time budget and performance** (`size_budget.pdf`, `size_performance.pdf`, the statistics of
`redundancy.py`). In group time the share of time in climb and of searches that end in a climb
are higher than for the same flights alone from 2 pilots on, and grow little with the size. The
climb rate falls a little with the size and stays under that of the same flights alone at every
size. The glide ratio rises steadily with the size, at a glide speed that stays near 33–35 km/h.
Part of this may come from the definition: two pilots in glide must keep within 45° of heading.

| pilots | time in climb (%) | searches that end in a climb (%) | climb rate (m/s) | glide speed (km/h) | glide ratio |
|---|---|---|---|---|---|
| solo | 30.8 | 43.3 | 1.22 | 30.1 | 10.4 |
| 2 | 32.7 | 49.0 | 1.27 | 33.2 | 12.5 |
| 3 | 33.6 | 49.9 | 1.26 | 33.5 | 13.1 |
| 4 | 33.6 | 51.0 | 1.25 | 33.5 | 13.7 |
| 5–6 | 33.1 | 50.1 | 1.23 | 33.1 | 14.1 |
| 7+ | 34.6 | 51.7 | 1.20 | 34.6 | 15.9 |
| same flights alone, over the sizes | 27.8–29.1 | 45.8–46.1 | 1.33–1.35 | 32.5–33.1 | 11.3–11.6 |

**Phases** (`size_phase_msd.pdf`, `size_survival.pdf`, paragliders). The MSD per phase does not
separate by size: the local exponent of transitions at the last drawn lag, 357 s, is 1.85 for
solo flights and 1.90 to 1.95 for every size, the curves of searches and climbs overlap (more
noisily for the large sizes, which have few searches). The tail of the transitions is heavier
than for solo flights from 2 to 4 pilots, with no order in the size: μ above 1857 s is 4.54
[4.43, 4.65] for solo flights, 3.64 [3.44, 3.88] for 2 pilots (795 runs), 3.20 [2.85, 3.67] for 3
(206), 3.38 [2.91, 3.92] for 4 (90) and 3.81 [3.31, 4.61] for 5–6 (84). A μ is shown only with 50
runs above the threshold: for transitions up to 5–6 pilots, for searches up to 3 (4.15 [3.77,
4.59] and 4.04 [3.40, 4.90], against 4.02 solo) and for climbs only with 2 (9.23 [8.03, 10.72]
against 8.85).

**Site-days** (`siteday_tamsd.pdf`, `siteday_budget.pdf`, `siteday_performance.pdf`). Flights
whose takeoffs lie within 3 km on one day share the weather and the site, so the size is the
only thing that varies. Six are drawn, by a fixed rule: most sizes with 15 flights or more,
then most flights, at most two per takeoff site (`--sitedays "Name:YYYY-MM-DD,..."` picks
others). The archive is dominated by Saint Hilaire sud: Méruz is the only other site that
reaches three sizes, and Chalvet Ouest and La Sambuy two. Bands resample the flights, since a large episode is a single block of the note on one
day, so flights of one group are not independent and the bands are too narrow. Searches are
almost absent from a site-day (0 to 16 pieces per size), so there is no phase figure.

H_loc(1000 s) by size on each site-day:

| site-day (flights) | solo | 2 | 3 | 4 | 5–6 | 7+ |
|---|---|---|---|---|---|---|
| Saint Hilaire sud, 2022-03-26 (108) | 0.750 | 0.853 | 0.894 | 0.818 | | 0.994 |
| Saint Hilaire sud, 2017-03-28 (64) | 0.854 | 0.931 | 0.901 | 0.956 | | 0.873 |
| Méruz, 2019-05-01 (73) | | 0.926 | 0.939 | | 0.938 | |
| Méruz, 2016-04-28 (44) | | 0.894 | 0.958 | | 0.957 | |
| Chalvet Ouest, 2016-07-19 (61) | | 0.951 | | | 0.892 | |
| La Sambuy, 2020-07-25 (54) | | 0.941 | 0.901 | | | |

**Result.** Where a site-day has solo flights (the two Saint Hilaire days) they have the lowest
H_loc(1000 s). The sizes differ among themselves by up to 0.18 on one day, with no common
order: the 7+ pilots that give 0.994 on one day give 0.873 on the other. The rise with the size
seen pooled does not show within a site-day, so it may come from the days and sites where the
large groups form. The budget and performance lines are scattered the same way, and the
differences between the six days (climb rate from 0.79 to 2.11 m/s) are far larger than those
between sizes.

## Group effect, selection, day, site and hour: `decompose.py`

Takes the difference between group time (G) and solo flights (S) apart, with the same flights
alone (A) as control. Each contrast is computed inside strata and averaged over the strata with
both populations, weights n_P n_Q / (n_P + n_Q): G − S pooled; G − S and A − S (selection) within
site-day and local hour; G − A within the flight and within site-day and hour (two estimates of
the group effect that fail in opposite ways: the first mixes the hour, the second the pilots).
For the TA-MSD the difference is that of log MSD at every lag, so ΔH and ΔH_loc are differences
of slopes. Bands: 200 draws of blocks that join the episodes and the site-days, the same draws
for every statistic. The same contrasts by the size of the group: each size minus A
within the flight, each size minus 2 pilots within site-day and hour (`decompose_size.*`). The
TA-MSD pieces and the performance terms carry their start or middle instant for the hour.

## Entries, exits, leaders and hours: `mechanisms.py`

The phase (Vilpellet run, read on all the runs of the flight) at the instant a flight enters or
leaves a group, against the time shares; in groups of two pilots, leader minus follower
within the group, the leader being the pilot the other passes after (signed median delay
of the pair interval, |delay| ≥ 15 s, else abreast); whether some pilots lead more than chance
(dispersion of the lead fractions of pilots with at least 10 roles against 1000 random swaps of
the roles); and the time of each population over the local hour.

## Group or zone: `zones.py`

Do the zones where groups fly explain the difference between group time (G) and solo flights (S)?
Each piece of track goes to a stratum (cell of 25 km of its middle, with the month, with the hour,
or the site-day and hour); in every stratum with both populations the ensemble MSD of S (and of A,
the same flights alone) is reweighted to the mix of G over the strata (direct standardisation), and
H_loc over every lag and the fitted H of the three are compared, with 5-95% bands from blocks that
join episodes and site-days. "As is" is the same comparison without the reweighting. Result: the
long-lag gap (ΔH_loc(1000 s) = 0.055) falls by only 6, 15 and 19% when S is reweighted to the
cells, the cells and months, or the cells and hours where groups fly, and by 14% at the same
site-day and hour (53% of G kept); the gap opens above 150 s. It is not a zone effect.

## Group or route: `route.py`

Is the straighter progress of groups the route they fly? The pieces of group time against the same
route flown alone on another day. Candidates: the 52 371 paraglider flights with no contact of any
length; a site joins the takeoffs within 1 km across dates, and its flights share one east-north
frame. A piece and a candidate match when at least 80% of the points of the piece (one every 30 s)
lie within 500 m of the candidate's track and the candidate flew them the same way; its segment, at
least 1250 s long, is measured as the pieces of `tamsd.py`; at most 20 matches per piece. The pieces
and the segments are compared by direct standardisation, one piece per stratum, with bands from
blocks; the baseline compares the same pieces with the solo flights of their site-day and hour.
Result, on a random sample of 10 000 of the 45 476 pieces: 4 933 pieces have matches (52 912
segments); ΔH_loc(1000 s) is +0.018 [+0.014, +0.021] against the same route alone and +0.016
[+0.012, +0.021] with the same season and hour (4 183 pieces), against +0.060 [+0.055, +0.066] for
the same pieces against the solo flights of their site-day and hour. Fixing the route lowers the
long-lag gap of these pieces by about 70% (71% and 74%); the rest stays with flying together. At
100 s group time is a little less straight than the same route alone (−0.004). On
2000 pieces, radii of 300 and 1000 m and covers of 70 and 90% give +0.014 to +0.022 against the
route and 66 to 76% explained.

## Group or moment: `prepost.py`

The same flight before, during and after a group. Each stretch of group time of at least W
seconds (default 3000, `--window SECONDS`) is an event with windows of W s of the flight clock:
before and after (the W s next to it, in a solo stretch of the same flight at least that long),
start and end of the group (its first and last W s). Controls: two consecutive windows with no
change of state, one pair per stretch, drawn in the solo time of the group flights (A) and in
solo flights (S). On the events that have both windows the ensemble MSD of each is compared over
every lag (H_loc and fitted H) and on the phase statistics (runs and glides placed by their
middle), with 5-95% bands from blocks that join episodes and site-days. Result (W = 3000 s,
6 524 entries, 13 643 exits): ΔH_loc(1000 s) = +0.027 [+0.023, +0.031] entering and +0.031 [+0.028,
+0.033] leaving, against −0.002 for the controls and −0.003 [−0.007, +0.001] for after minus
before; nothing at 100 s; the fitted H does not leave the drift; glides faster and flatter in both
directions. The straighter progress follows the group in time; what brings pilots together is not
controlled. With W = 1260 s, H_loc(1000 s) rests on 26 origins and drifts by −0.007 to −0.011
between consecutive windows.

## Size, duration and events: `describe.py`

The groups by number of pilots, the durations of groups and episodes, and how
each group ends: join, leave (in flight), landing (a member removed lands within 5 s,
the grid step), split, merge. Rates are groups ending per hour spent at each size, with
the bands of the blocks. All disciplines (0.9% of the groups hold a hang glider).

- 40 051 groups in 17 169 episodes. Groups of 2 pilots are 63% of them and 77%
  of the time; the largest has 27. 74% of the episodes are one group; 5.0% hold a split,
  4.4% a merge.
- Median duration: group 30 min, episode 58 min; 12% of the episodes last 2 h or more.
  Episodes cannot be shorter than τ* = 38 min; groups can, cut by changes of members.
- Of the 18 218 groups with no predecessor, 34% form at a member's departure. Of the
  18 376 that end the group, 91% end in flight, 9% at a landing.
- At 2 pilots a group ends 0.96 times per hour by leave, 0.31 by join, 0.098 by landing,
  0.085 by merge. Join, leave, split and merge grow with size (split from 0.48/h at 4 pilots to
  2.7/h at 8+), so the median group lasts 41 min at 2 pilots and 4 min at 8+.

Large groups last minutes: someone joins, leaves or splits off every few minutes. The
episode is the unit that lasts.

## Cohesion: `cohesion.py`

Per date, with the grid and the surrogate pairs of the note (the same seed; the surrogate
intervals match the run on 30 dates): the delay ℓ*(t) inside the 34 082 pair intervals of near
pairs (takeoffs within 10 km) and inside the 134 202 surrogate contacts, 980 of them lasting at
least τ*; and the distance at the same instant after the first contact of each near pair. The
per-date cache is in `analyses/cohesion/all/days/` of the run on SSD_DISANTE.

- D(s) = ⟨(ℓ(t+s) − ℓ(t))²⟩ fitted by an Ornstein-Uhlenbeck process: groups 1/k = 3.9 min
  [3.8, 4.0], σ = 54 s [53, 54]; chance contacts of at least τ*, 4.4 min [4.2, 4.7] and 78 s
  [77, 79]. The diffusivity 2kσ² is 24.5 [23.9, 25.1] s²/s in groups, 46 [44, 48] for chance
  contacts of at least τ*, 69 [68, 70] for all of them; 24.2, 24.5 and 25.0 at 2, 3 and 4+ pilots.
- The drift of ℓ, made symmetric, pulls back to 0 with 1/k = 4.7 min [4.6, 4.7] in groups and
  6.4 min [5.9, 6.9] for chance contacts of at least τ*.
- After a first contact of at least τ*, the median distance is 1.0 km at 60 min against 2.4 km by
  chance, 2.9 against 5.8 km at 2 h. After any first contact, 3.7 against 5.1 km at 30 min.

Groups do not pull back much harder than two pilots kept in contact by chance: most of the pull
is the selection of staying in contact. They stay together because their delay wanders half as
fast: they keep the same pace. The baseline 2 MSD₁ of the plan was dropped: it ignores the drift
two pilots share along the route; the surrogate pairs are the baseline.

## Hierarchy: `hierarchy.py`

Order in groups of 3 pilots or more: each edge points from the pilot ahead (|ℓ_med| ≥ 15 s),
against the same graphs with directions at random (50 draws). Switches of the pilot ahead along
ℓ*(t) (cache of `cohesion.py`, runs shorter than 60 s dropped). Who leads in groups of two
named pilots: earlier flights in the catalog and wing class (A = 1 to competition = 5).

- Directed triangles are cyclic 2.5% [2.1, 3.0] of the time, against 25% at random; graphs with
  every edge directed (33% of the time) are acyclic 98.5%, against 88%; with one leader 80.6%
  [79.6, 81.7], against 77.0% [76.5, 77.6]. One leader: 86% at 3 pilots, 68% at 4, 50% at 5+.
- The pilot ahead changes 3.4 times per hour of contact [3.3, 3.4] in groups, 4.4 [4.2, 4.5] in
  chance contacts of at least τ*; the pilot most often ahead holds 73% of the directed time
  (median), 69% by chance.
- The leader has more earlier flights in 58.2% [57.5, 58.8] of 15 937 pairs, the higher wing class
  in 63.9% [62.8, 64.9] of 8 304.

Pilots in a group are in a line along the route, as positions on a route must be: the low share of
cycles is geometry, not rank. Leadership is fluid, a little steadier than chance, and goes more
often to the experienced pilot on the faster wing.

## Chains or compact groups: `chains.py`

The configurations of each group of 3 pilots or more (its graph between two changes of its
edges): density, diameter, and shape (complete, tree, between). Then H and H_loc of the TA-MSD
pieces of `size.py` by the shape of the configuration at their middle (paragliders, bands of the blocks).

- At 3 pilots the graph is a triangle 53% of the time and a chain 47%. The share of complete
  graphs falls from 27% at 4 pilots to 2.5% at 8+; the mean density from 0.84 to 0.45; at 8+ the
  diameter is 4 or more 61% of the time. Every stretch is connected.
- H_loc(1000 s): triangles 0.906 [0.901, 0.910] (5 322 flights), chains 0.907 [0.902, 0.911]
  (4 882); 4+ complete 0.915 [0.905, 0.923] (1 638), trees 0.907 [0.894, 0.920] (1 171).

Large groups are strings along the route, not clouds. The straighter progress does not need
every pilot in contact with every other.

## Repeated partners: `partners.py`

Named paraglider pilots (catalog); partners on a day when their flights share a pair interval.
Chance redraws the partners within each site-day (joined when a pair interval links two), each
pilot keeping its number of partners that day (configuration model, 100 draws).

- 30 550 pair-days, 4 525 pilots, 22 513 pairs; one pair flew together on 69 days.
- Pair-days on pairs that meet on more than one day: 37.6%, against 25.7% [25.3, 26.0] at random;
  such pairs: 3 437 against 2 501 [2 453, 2 544].
- Pair-days of two pilots of the same club (each pilot's main club): 32.1%, against 24.0%
  [23.8, 24.2].

With who flies where and when fixed, pilots still choose the same partners and their club mates.

## Fission-fusion: `fission.py`

T(n), the share of the time in groups of n pilots, against a birth-death chain,
T(n+1)/T(n) = g(n)/l(n+1), with g the joins and l the leaves and landings per hour at size n.

- With the measured rates the chain reproduces T(n) within 4% from 2 to 8 pilots, inside its
  bands except at 3 pilots (0.145 against 0.148), although splits and merges are 6% of the changes
  at 2 pilots and 46% at 8+.
- Per member, joins are constant, a = 0.162/h [0.158, 0.167]; leaves fall with size, from 0.53/h
  at 2 pilots to 0.25/h at 8+: μ n^-γ with γ = 0.30 [0.28, 0.33]. With γ = 0 (log-series, x = 0.25)
  T(10) is 290 times too small; with γ = 0.30 it is 4 times too small.
- Fitted to T: n^-α exp(-n/n_c) with α = 3.4, n_c = 5 (Bonabeau et al. 1999).

The size distribution follows from one pilot joining or leaving at a time; its tail is heavy
because members of larger groups leave less often.

## Report: `report/`

`report/group_vs_solo.pdf` is a technical report (10 pages of text and figures, then appendices
with the tables): what is compared, the raw difference between group and solo flight, four tests that
separate the group from the pilots, the place, the route and the moment, then size and leader, for
reading and as the base of a presentation. `report/facts.py` writes the
facts of the grouping, `report/tables.py` the tables and the numbers of the text (macros in
`report/generated/`) from the JSON of `out/<run>/`, `report/figures.py` the figures of the
report, drawn at the text width (`out/<run>/report_*`); `latexmk -pdf group_vs_solo.tex` in `report/`
builds it. The figures are read from `out/<run>/` and from `ideas/group-identification/figures/`.

What groups are, what keeps two pilots together, the shape of groups, who leads, repeated
partners and where the sizes come from are in section 7 of the note
`ideas/group-identification/group_identification.pdf`, with the graph theory and the two models
in its appendices F–G. `report/structure_tables.py` writes the macros the note reads
(`generated/structure_*.tex`); it also reads the macros of `tables.py`. The note reads copies of
the figures, in `ideas/group-identification/figures/structure/`.

## To do

- Game theory, postponed: who pays for the search of thermals (producer-scrounger) and whether
  pilots take turns ahead (Voelkl et al. 2015), from the switches of `hierarchy.py`.
- Communities of the pilot network, against clubs (needs a graph library).

## Tests and rebuild

`test_groups.py` checks the FFT against the direct average, the populations on a synthetic
membership, the 1250-s rule and the segments, MSD = v²τ² on a straight glide, H and H_loc on
power laws, the block bootstrap, the cut of the phase runs, the slopes on power laws, the sites
and blocks of `blocks.py`, the runs that count and their population in `survival.py`, the
terms of `budget.py`, the climbs, glides and 30-min distances of `performance.py` on a synthetic
flight, the elimination of `redundancy.py`, the groups by size and the choice of the site-days
of `size.py`, the events of `describe.py` on the episode figure of the note, the delay statistics
of `cohesion.py` (an Ornstein-Uhlenbeck fit recovers its parameters), the shapes and diameters of
`chains.py`, the order, triangles and switches of `hierarchy.py`, the coverage, segments and sites
of `route.py`, and every number quoted here.

```
uv run python tamsd.py [--run NAME]      # from this folder, needs SSD_DISANTE; --recompute: about 5 min
uv run python episodes.py [--run NAME]   # after tamsd.py
uv run python phase_msd.py [--run NAME]  # --recompute: about 6 min; the bootstrap of the slopes about 45 min
uv run python blocks.py [--run NAME]     # after tamsd.py, about 30 s
uv run python survival.py [--run NAME]   # about 30 s
uv run python budget.py [--run NAME]     # about 20 s
uv run python performance.py [--run NAME]  # --recompute: about 12 min (one pass over the fixes)
uv run python redundancy.py [--run NAME] # after budget.py and performance.py, about 1 min
uv run python size.py [--run NAME]       # after the scripts above; first run: three passes over the fixes, about 8 min, then about 4 min
uv run python size.py --only slopes      # the slopes of the phase MSD by size, about an hour
uv run python decompose.py [--run NAME] # after size.py; about 40 min (by size included)
uv run python mechanisms.py [--run NAME] # after size.py, about 5 min
uv run python zones.py [--run NAME]      # after tamsd.py, about 3 min
uv run python route.py --sample 10000    # after tamsd.py; one pass over the fixes (about 4 min), then the
                                         # matching, about 45 min and 2.6 GB: run one at a time
uv run python route.py --sample 2000 [--rho M] [--cover SHARE]   # the variants, about 10 min each
uv run python prepost.py [--run NAME] [--window SECONDS]  # after performance.py; first run per window: one pass over the fixes, about 6 min
uv run python size.py --recompute-pieces # TA-MSD pieces by size only, about 3 min
uv run python describe.py [--run NAME]   # about 15 s
uv run python cohesion.py [--jobs J]     # one pass over the grid of every date, about 25 min with 6 jobs
uv run python chains.py [--run NAME]     # after size.py, about 40 s
uv run python hierarchy.py [--run NAME]  # after cohesion.py, about 25 s
uv run python partners.py [--run NAME]   # about 4 min
uv run python fission.py [--run NAME]    # about 10 s
uv run python report/structure_tables.py   # then rebuild the note (README of ideas/group-identification)
uv run python report/facts.py; uv run python report/tables.py; uv run python report/figures.py
cd report && latexmk -pdf group_vs_solo.tex
uv run pytest analysis/groups/group-identification   # from the root of xc-lab
```
