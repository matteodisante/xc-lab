# Plan: the group, or the gliders around it? Intra-group against inter-group effect

Folder: `xc-lab/analyses/groups/`. New script `intergroup.py`, tests `test_intergroup.py`, new
section of the report `vs-solo/comparisons` (blue title). Asked by the author on 2026-11-12.

## Question

Group time is more persistent than solo time. A group is defined by a strict rule (contact within
2 min of delay and 500 m, for at least tau*), so gliders a few km ahead are not members, yet a
pilot can see them. How much of the "group effect" belongs to the group itself (intra-group) and how
much to the presence of other gliders, in other groups or alone (inter-group)? If the inter-group
effect exists and is the stronger one, it goes in the letter.

## Definitions

- **Outsider** of a pilot at time t: any other glider of the archive in the air that is not a
  member of the pilot's configuration at t. For a solo pilot, any other glider. An outsider is "in
  another group" or "solo" at t.
- **Positions**: the archive index of `isolated.py` (`read_index`, one position per minute for
  every flight, para and hang, on the sphere via `sphere()`). Import it, do not modify
  `isolated.py`. "At the same instant" means within `isolated.TIME_S`.
- **Direction of motion** u(t) of a pilot: unit vector of its horizontal displacement between
  t - T_DIR/2 and t + T_DIR/2, with T_DIR a constant of the script (10 min: long enough to average
  the circling). For a group, the same on the centroid of its members.
- **Ahead / behind**: an outsider is ahead of a pilot when its position relative to the pilot has a
  positive component along u(t), behind otherwise.
- **Gap ahead** d_a(t): horizontal distance to the nearest outsider ahead; **gap behind** d_b(t)
  likewise. Infinite when there is none within R_MAX (20 km).

## Step A: geometry (the necessary condition)

For every group, every minute it exists:
- length L_par: extent of the members along u (leader to last);
- width L_perp: extent of the members across u;
- gap from the leader to the nearest outsider ahead of the leader, split by kind of outsider
  (member of another group, solo), and the same gap to the nearest *other group* only;
- the cross-track offset of that outsider.
Report: the distributions of L_par, L_perp and of the gaps (medians, quantiles 10 and 90), the
share of group time with an outsider ahead within R for R in (1, 2, 5, 10) km, and the same shares
for solo time (populations A and S). Chance level: the same shares on the time-shift surrogates of
the group identification if they are cheap to reuse (pairs of flights that never met); if not,
skip and say so.
The condition for an inter-group effect is that the gaps are within sight. No sight range is
assumed: the comparison is made at the four radii, and Step C scans the distance.

## Step B: separating the two effects

Units: the pieces the report already uses in the raw design on the open flights (group pieces G,
solo pieces of A and of S, long enough for H_loc(1000 s)). For every piece, from the index:
f_a(R) = share of its minutes with an outsider ahead within R; f_b(R) the same behind.
Neighbourhood class of a piece, at R = 5 km in the main line (and at 2 and 10 km as variants):
- ALONE: f_a < F_LOW and f_b < F_LOW;
- AHEAD: f_a >= F_HIGH (others ahead most of the time);
- BEHIND: f_b >= F_HIGH and f_a < F_LOW (others only behind: the pilot leads them);
- MIXED: the rest (counted, shown, not interpreted).
F_LOW = 0.1, F_HIGH = 0.5, constants of the script.

Effects, on H_loc(1000 s) first, then on the phase statistics of the report (time in climb,
searches that end in a climb, glide speed, height lost per km):
- intra-group = (group, ALONE) - (solo, ALONE): a group with nobody around against a pilot with
  nobody around;
- inter-group = (solo, AHEAD) - (solo, ALONE): no group, but gliders ahead;
- crowd control = (solo, BEHIND) - (solo, ALONE): gliders behind cannot show the way, so this
  measures what a frequented route gives without any information;
- information = (solo, AHEAD) - (solo, BEHIND), and the same inside group time;
- both = (group, AHEAD) - (solo, ALONE).
Selection is handled as in the report, in two lines:
1. raw, with the stratification the raw design of the report already uses;
2. within the flight: only flights that have pieces in both classes of a contrast, difference per
   flight, then averaged with equal weight per flight (in analogy with the same-flight design).
Bands: `twoway.py`, as everywhere (pilots, blocks, cells; 90%).
Report every count: pieces, flights and pilots per cell. Some cells will be thin: show them with
their bands and do not merge classes to rescue them.

## Step C: range of the interaction

Event-based, so no minimum length is needed:
- every search (from the phase labels): did it end in a climb, against the gap ahead d_a at its
  start, in bins (0-0.5, 0.5-1, 1-2, 2-5, 5-10, 10-20 km, none), for solo time and for group time,
  and against the gap behind d_b in the same bins as the control;
- every glide: speed and height lost per km against the same bins;
- H_loc(1000 s) of the pieces by the bin of their median d_a (same bins, merged where thin).
The curve against d_a, compared with the curve against d_b, gives the range over which gliders
ahead matter. Within-flight version where the counts allow.

## Outputs

- `intergroup.py` with `--sample N` for trials, results as JSON in the usual place of the run
  (`out/<run>/open_min<...>/intergroup.json`), heavy caches by the repo's convention.
- Macros in a new generated file `vs-solo/generated/comparisons/intergroup_macros.tex`, written by a
  small new script in `vs-solo/code/`, names prefixed `Ig`, checked against clashes with the four
  macro files the letter loads.
- Figures by the report's figure code: (1) distributions of L_par, L_perp and of the gap ahead on
  one distance axis; (2) forest plot of the effects of Step B, raw and within the flight; (3) the
  curves of Step C against d_a and d_b.
- Report section `vs-solo/comparisons/sections/intergroup.tex`, input after the other new sections
  and before the summary; blue title; opens with the question, defines every quantity with its
  formula, results as figures, one table of the key numbers, limits, "What we keep", which states
  whether an inter-group effect exists, how it compares with the intra-group one, and whether the
  ahead-behind contrast supports information or a frequented route.

## Limits to state in the section

- The index has one position per minute: distances below a few hundred metres and delays below a
  minute are not resolved.
- Only gliders that declared their flight to the federation are in the archive: "alone" means no
  recorded glider around.
- A piece changes neighbourhood along its length: the classes are shares of time, the lags reach
  1000 s.
- Being near others is itself selected (good days, frequented routes): the BEHIND control and the
  within-flight line are the two guards, and neither is a randomised comparison.
