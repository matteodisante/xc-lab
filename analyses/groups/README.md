# Groups

Paragliders and hang gliders alone and in a group, on the labels of
[`methods/group-identification`](../../methods/group-identification/). The scripts are one
pipeline on the labels and import each other; the questions are split by their documents:

| question | scripts | document |
|---|---|---|
| Does flying in a group change the flight? | tamsd, episodes, phase_msd, survival, budget, performance, redundancy, size, decompose, zones, route, isolated, prepost, mechanisms (leader and follower) | `vs-solo/group_vs_solo.pdf` |
| What are the groups? | describe, mechanisms, cohesion, hierarchy, chains, partners, fission | `structure/group_structure.pdf` |
| Does it depend on the wing level? | `experienced/` | none |

`tamsd.py` (labels, pieces, bands) and `blocks.py` (dependency blocks) serve every question.
The documents hold the full results, as macros the scripts write; below, each script in a few
lines with its key result. `@ 6e45daf` names the commit of the documents these lines summarise.

**Data.** Run `d120_eh500_ez300_th45_g120_q0.1` of the method (δ = 2 min, q = 0.1, τ* = 38.1
min) and the cleaned tracks of soaring (tag `legacy-final`) on SSD_DISANTE; xc-dynamics at
`d72e1b9` (not used). `--run NAME` picks another run.

**Output.** Figures and JSON in `out/<run>/` (git-ignored), with the run's `PARAMETERS.md`. The
measured pieces (`pieces.npz`, `phase_pieces.npz`, `size_*`) stay on
`/Volumes/SSD_DISANTE/xc-lab/group-identification/runs/<run>/analyses/`.

**Populations**, after the departure, on loggers of at most 10 s:

- S, solo: flights never in a group;
- G, group time: the stretches in a group of the flights that have one;
- A, the same flights alone: their solo stretches, a control with the same pilots, days and
  loggers.

**Conventions.** TA-MSD on a 10-s grid, lags 10–1000 s, a stretch enters only if it lasts
1250 s in one segment; equal weight per flight. H: half the slope of log MSD over all lags;
H_loc: the same within ±0.25 decades. Bands: 5–95% of bootstrap draws of the dependency blocks
of the method (flights tied by episodes; a solo flight is a block alone).

## Does flying in a group change the flight?

Results in `vs-solo/group_vs_solo.pdf @ 6e45daf`.

- **`tamsd.py`**: TA-MSD, H and H_loc of S, G, A. Paragliders: H 0.899 (S), 0.916 (G), 0.914
  (A): the gap in H is the flights and days. H_loc(1000 s) 0.837, 0.893, 0.860: at long lags
  it is the group. Hang gliders alike, with wide bands (418 group flights).
- **`blocks.py`**: the bands with coarser blocks (takeoffs within 3 km on a day, or a whole
  day): no band widens by more than 0.002.
- **`episodes.py`**: the MSD of the five episodes with most flights, each over the group time
  of its members (not in the report):

  | episode | flights (largest group) | duration | paragliders in the MSD | H |
  |---|---|---|---|---|
  | 2013-04-24 | 53 (16) | 5.8 h | 49 | 0.926 [0.920, 0.931] |
  | 2015-04-11 | 52 (19) | 5.1 h | 52 | 0.912 [0.907, 0.917] |
  | 2016-03-19 | 49 (22) | 5.2 h | 49 | 0.879 [0.868, 0.888] |
  | 2022-03-26 | 47 (16) | 3.8 h | 47 | 0.883 [0.869, 0.896] |
  | 2018-04-21 | 47 (19) | 6.5 h | 47 | 0.923 [0.916, 0.929] |

  H varies from group to group well beyond the band of each; the three hang gliders of
  2013-04-24 give 0.861 [0.821, 0.890].
- **`phase_msd.py`**: MSD of transitions, searches and climbs by the method of
  [`analyses/phases/msd.py`](../phases/), a phase run cut to the stretches of its population;
  1 Hz flights only. Transitions slightly more ballistic in G and A (1.977) than in S (1.962);
  search slopes overlap.
- **`survival.py`**: phase durations by the method of
  [`analyses/phases/durations.py`](../phases/), a run whole in the population of its middle; μ
  above the x_min of S. Only the tail of transitions in G is heavier: μ 3.52 against 4.54 (S)
  and 4.37 (A).
- **`budget.py`**: shares of time in each phase. In G more time in climb (33.0% against 30.8% S,
  29.1% A) and fewer searches in vain; a search ends in a climb 49.5% of the time (43.3% S).
- **`performance.py`**: climb rate, glide speed, glide ratio, height per climb, cross-country
  speed over 30 min (one pass over the fixes). Pooled: glide ratio 12.9 (G), 11.3 (A), 10.4
  (S); within the flight or the site-day and hour (`decompose.py`) the climb rate and the
  cross-country speed are also higher in G: the pooled readings are composition.
- **`redundancy.py`**: keeps the statistics that the others do not explain (R² ≤ 0.5 by backward
  elimination over site-day units): time in climb, searches that end in a climb, climb rate,
  glide speed, glide ratio. Only these are drawn.
- **`size.py`**: everything by the number of pilots of the group (2, 3, 4, 5–6, 7+), pooled and
  on six site-days. Pooled, H_loc(1000 s) rises with the size (0.889 at 2, 0.944 at 7+) while A
  stays at 0.86–0.87; within a site-day there is no common order.
- **`decompose.py`**: G − S, A − S (selection) and G − A within the flight and within site-day
  and hour; the same by size. The long-lag effect survives both strata; the short-lag one
  vanishes at the same hour.
- **`zones.py`**: S reweighted to the 25-km cells (with month, hour, site-day and hour) of G. The
  long-lag gap (0.055) falls by 6–19%: not a zone effect.
- **`route.py`**: pieces of G against the same route flown on another day by flights with no
  contact (80% of the points within 500 m, same direction), 10 000 pieces sampled. ΔH_loc(1000 s)
  +0.018 against the route, +0.060 against S of the same site-day and hour: the route explains
  about 70%, on every radius (300–1000 m) and cover (70–90%) tried.
- **`isolated.py`**: the same with controls nobody came within 10 km of (same minute), on other
  days, same season, hour and wing level; all pieces (not yet in the report; plan in
  `isolated_plan.md`). ΔH_loc(1000 s) at the same site-day and hour, then against the isolated
  controls:

  | | pieces | against S | against isolated | explained |
  |---|---|---|---|---|
  | experienced | 10 824 | 0.051 | 0.016 | 69% |
  | beginners | 4 715 | 0.068 | 0.022 | 68% |
  | all | 15 539 | 0.058 | 0.017 | 70% |

  As with `route.py`: the route explains 70%, the rest stays with flying together. At short lags
  G is less ballistic than the controls (ΔH_loc ≈ −0.02 at 10 s).
- **`prepost.py`**: the same flight before, during and after a group (windows of 3000 s,
  `--window`), against two consecutive windows with no change. ΔH_loc(1000 s) +0.027 entering,
  +0.031 leaving, −0.003 after minus before: the effect follows the group in time.
- **`mechanisms.py`**: phase at entry and exit, leader minus follower in groups of two, habitual
  leaders, hours (the last three serve the structure note too).

## What are the groups?

Results in `structure/group_structure.pdf @ 6e45daf`; numbers as macros from
`vs-solo/structure_tables.py`, figures copied to `structure/figures/structure/`.

- **`describe.py`**: sizes, durations and how each group ends (join, leave, landing, split,
  merge). Pairs are 63% of the groups and 77% of the time; median group 30 min, episode 58 min.
- **`cohesion.py`**: the delay ℓ*(t) inside pair intervals against the surrogate contacts of the
  null model (per-date cache in `runs/<run>/analyses/cohesion/`). Groups keep the same pace (delay
  diffusivity half that of chance contacts as long) rather than pulling back.
- **`hierarchy.py`**: order in groups of 3+, switches of the pilot ahead, who leads in pairs. A
  line along the route; the lead changes 3.3 times per hour and goes more often to the more
  experienced pilot (58.5%) on the higher wing (64.3%).
- **`chains.py`**: shape of the configurations (complete, tree, between). Large groups are
  strings; the shape does not change H_loc.
- **`partners.py`**: pilots return to the same partners (37.6% of pair-days against 25.7% at
  random) and to club mates (32.1% against 24.0%).
- **`fission.py`**: the size distribution from a birth-death chain with the measured join and
  leave rates (within 4% from 2 to 8 pilots); the tail is heavy because members of larger groups
  leave less often (γ = 0.30).

## Does it depend on the wing level? `experienced/`

Experienced (C, D, CCC) against beginners (A, B), group flights against flights with no contact
of any length (`population.py`); not in a document yet.

| | group | no contact | group, with a route match | same route, no contact |
|---|---|---|---|---|
| experienced, H | 0.920 | 0.905 | 0.912 | 0.910 |
| beginners, H | 0.903 | 0.888 | 0.893 | 0.895 |
| experienced, glide ratio | 13.0 | 10.2 | | |
| beginners, glide ratio | 12.4 | 8.9 | | |

`count.py`, `glide.py`, `msd.py` (with `route_level.py` for the same-route columns),
`phase_survival.py` (phase durations fitted by `families.py` and `lomax.py`).

## To do

- Game theory, postponed: who pays for the search of thermals (producer-scrounger) and whether
  pilots take turns ahead (Voelkl et al. 2015), from the switches of `hierarchy.py`.
- Communities of the pilot network, against clubs (needs a graph library).

## Tests and rebuild

`test_groups.py` checks the methods on synthetic data (FFT against the direct average, the
1250-s rule, H on power laws, the bootstrap, phase cuts, budget and performance terms, the
elimination, sizes, events, the Ornstein-Uhlenbeck fit, shapes, order, route coverage) and the
numbers quoted here, except those of `isolated.py` and `experienced/`, which no test checks yet;
`experienced/test_*.py` checks the fits.

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
uv run python size.py --recompute-pieces # TA-MSD pieces by size only, about 3 min
uv run python decompose.py [--run NAME]  # after size.py; about 40 min (by size included)
uv run python mechanisms.py [--run NAME] # after size.py, about 5 min
uv run python zones.py [--run NAME]      # after tamsd.py, about 3 min
uv run python route.py --sample 10000    # after tamsd.py; one pass over the fixes (about 4 min), then the
                                         # matching, about 45 min and 2.6 GB: run one at a time
uv run python route.py --sample 2000 [--rho M] [--cover SHARE]   # the variants, about 10 min each
uv run python isolated.py [--sample N]   # after route.py, performance.py and the archive index; about 25 min and 2 GB
uv run python prepost.py [--run NAME] [--window SECONDS]  # after performance.py; first run per window: one pass over the fixes, about 6 min
uv run python describe.py [--run NAME]   # about 15 s
uv run python cohesion.py [--jobs J]     # one pass over the grid of every date, about 25 min with 6 jobs
uv run python chains.py [--run NAME]     # after size.py, about 40 s
uv run python hierarchy.py [--run NAME]  # after cohesion.py, about 25 s
uv run python partners.py [--run NAME]   # about 4 min
uv run python fission.py [--run NAME]    # about 10 s
uv run python vs-solo/facts.py; uv run python vs-solo/tables.py; uv run python vs-solo/figures.py
uv run python vs-solo/structure_tables.py
cd vs-solo && latexmk -pdf group_vs_solo.tex
cd structure && latexmk -pdf group_structure.tex   # after the method note, for the gi- labels
uv run pytest analyses/groups   # from the root of xc-lab
```
