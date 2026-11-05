# Plan: the group structure note, from pairs to pilots across days

Folder: `xc-lab/analyses/groups/`. Note: `structure/group_structure.tex`, numbers as macros in
`structure_macros.tex`. Chosen run: `d120_eh500_ez300_th45_g120_q0.1`.

## Goal

Make the note complete in what is worth studying: every new study answers a question the note does not
answer yet, with a null model; nothing is added that only confirms a known result.

## Rules shared by every study

- **Interaction against shared air.** Two pilots in the same air correlate without seeing each other.
  Every effect is measured against the surrogate pairs of the method (same place, never met) or a
  time shift of the other pilot; a number without its null does not enter the note.
- **Bands.** 5--95% of 200 draws of the dependency blocks, as in the rest of the note.
- **Fixes.** Geometry and velocity studies on 1 Hz flights with phases; the 5 s grid only where it is
  enough, said in the caption.
- **Constants** (radii, delays, windows) chosen from a measured distribution and written in the script
  header and in the note; none invented.
- **Each study** starts with a short design (question, quantity, null, expected result, what would
  refute it), agreed before coding.

## New outline of the note

1. **Two pilots.** Cohesion (kept). Velocity correlations (new, A2).
2. **The group.** Sizes, durations, changes, birth--death chain (kept) and rates by episode age (new,
   B1). Shape (kept) and breathing (new, B2). Order (kept) and leader rotation in groups of 3+ (new, B3).
3. **Between groups** (new, C).
4. **Pilots across days.** Partners, clubs, habitual leaders (kept, moved here). Pilot network (new, D).
5. **What the data support.**

Core about 15 pages; phase, hour and details in appendices. Each section opens with its question and
closes with "What we keep".

## Step 0: feasibility

- episodes with 3+ flights: count and share of group time; those that change members;
- share of group time on 1 Hz flights with phases;
- neighbours not in contact: distribution of the number of gliders within R of a group pilot, for a
  few R (1, 3, 5 km), and the share of group time with at least one.

If neighbours are rare, C shrinks to what the data allow.

**Results (2026-11-05).**
- 4405 of 17 169 episodes have 3+ flights (26%), with 52% of the pilot time in groups; all of them
  hold more than one group. Flights per 3+ episode: median 3, 90th percentile 6, 99th 15.
- 72% of the group time is on 1 Hz flights with phases; in 51% of the groups of 3+ episodes every
  member is.
- Neighbours (300 random dates, 364 k group pilot-minutes, minute index of `isolated.py`, 3D
  distance): at least one glider outside the own group within 1, 3, 5 km for 34%, 60%, 71% of the
  pilot-minutes (mean 0.7, 2.2, 3.6); at least one from another group for 7%, 17%, 23%. So C is
  feasible; most neighbours fly solo.

## Studies

**A2. Velocity correlations (pairs).** Do pilots respond to each other's turns and speed changes, and
who responds to whom? In glide only (circling dominates the heading):
`C_ij(tau) = < u_i(t) . u_j(t+tau) >`, u the unit horizontal velocity; the tau of the maximum gives
who follows and with what delay (Nagy et al. 2010). Cross-check with the sign of l*: agreement makes the
leadership result stronger. Null: surrogate pairs. Not done: correlation length (groups of 2--8).

**B1. Rates by episode age.** Do joins and leaves depend on the age a of the episode, beyond the size n?
`g(n, a)`, `l(n, a)` per member per hour, a in minutes from the start. Flat in a: the chain stands.
Rising leaves with a: episodes have a life cycle, and the chain needs it.
*Done (2026-11-05, `age.py`, not committed):* joins fall from 0-15 to 15 min and stay flat (pilots
who took off together); leaves of pairs fall with age (1.27 to 0.78 per member-hour, in every hour
class); landings rise, mostly with the time of day. Leaves, landings and splits cannot occur before
tau*, so their exposure starts there; the dead time after a join is not removed, which biases toward a
rise, so the fall is conservative. Selection by partnership checked (`age.py`, key `partners`): pairs
that meet on several days leave less (0.98 against 1.57 at 30-60 min), but within each class leaves
still fall, 120+ / 30-60 = 0.69 [0.64, 0.73] and 0.65 [0.61, 0.68], against 0.61 [0.59, 0.64] for all
pairs. Other unobserved heterogeneity (day, route) is not excluded.

**B2. Breathing.** Does a group contract and stretch, and is it more than kinematics? Gyration tensor of
the members' positions around the centroid, `L_par`, `L_perp` along and across the mean velocity, plus
the vertical spread. Trap: at constant time delays `L_par(t) ~ v(t) * (l_max(t) - l_min(t))`, so it
shrinks in thermals by speed alone. The real question is whether the spread of the delays changes.
Tools: L and delay spread against the share of members climbing; spectrum of L against the
climb--glide period.
*Done (2026-11-05, `breathing.py`, not committed):* 11 472 groups of 2+ lasting 30 min or more. At 3
pilots L_par is 252 m all gliding and 46.5 m all climbing; L_perp 69 to 42 m; L_z rises in thermals
(stacking). At a thermal entry L_par falls to 0.40 of its value in 1.5 min while sigma_l stays flat
(1.05). Follow-the-leader test (each member where the head was one delay earlier): observed over
predicted L_par 1.00 in glide, 1.09 in thermals at 3 pilots (1.01-1.06 at 5+); at entries the
prediction falls to 0.29, the observed to 0.40. So the breathing is about 90% the accordion of
constant delays; followers contract slightly less than retracing would. No common period of L and the
climbing share. The prediction needs an order: 55% of the instants.

**B2b. Breathing in glide.** Seen in the viewer: in glide the group stretches and shrinks along and
across the route. In glide L_par follows the constant-delay prediction (ratio 1.00), so its changes
come from the delays (sigma_l: a pilot speeds up) or from the head's speed; L_perp is not kinematic
at all (choice of line). Glides: f = 0 for at least a minimum duration. Per glide: L_par, L_perp, L_z
and sigma_l against the time from the glide start and to the glide end (next entry), relative to
their value at the start; their fluctuation within a glide (coefficient of variation) and its time
scale (autocorrelation). Question for L_perp: does the group fan out across the route while
searching and close up before the next thermal? Null for L_perp: surrogate pairs on the same terrain,
or pairs of the same group with times shifted, to be chosen at the design.
*Done (2026-11-05, `glide.py`, not committed):* 34 917 glides of pairs, 6081 of 3+, 1481 of chance
contacts (surrogate pairs, bootstrap over dates); median glide about 5 min. Along the route, changes of
log L_par over 120 s (30 s is noise): delays account for 0.45 (pairs) and 0.61 (3+), head speed 0.11 and
0.26 (Cov(y, k)/Var(y)). Across: RMS cross-track offset of the followers from the head's track, at
s = 0.05 / 0.55 / 0.95 of the glide: pairs 107 / 143 / 152 m, 3+ 106 / 160 / 175 m, chance 159 / 180 / 166
m. The spread grows after the thermal exit and levels off; no closing before the entry; groups
follow tighter lines than chance pairs early and mid-glide (pairs / chance 0.67 to 0.91). The first
bin is partly selection (the passage must lie inside the glide). L_z flat in glide: pairs 28-31 m, 3+
47-52 m.

**B3. Leader rotation (3+).** Who leads, keeps leading? Rank along the string over time; Kendall tau
of the order against the lag; Markov chain over head, middle, tail; does the head change at the
start of a climb (who finds the thermal leads)? Three time scales: one thermal, one episode, across
days (habitual leaders, kept). Null: ranks permuted within the group.
*Done (2026-11-05, `rotation.py`, not committed):* 2133 groups of 3+ lasting 30 min or more, all
members labelled. Kendall tau of the order 0.70 at 1 min, 0.40 at 5, 0.20 at 15, 0.13-0.17 at 30 (all
sizes alike): the order renews in about 5-10 min and keeps a weak lasting part. The most frequent head
holds 0.68 [0.67, 0.69] of the time at 3 pilots, against 0.65 [0.65, 0.66] with random hand-overs (same
spells): barely more than few changes give. About 6 head changes per hour, median spell 3-4 min.
To fix before the note: (1) drop the null of shuffled instants (it mixes close instants and
overstates the plateau; tau falls "below" it after 10 min): reference tau = 0 and the plateau;
(2) climbs: the change rate is 6.2/h before a climb start, 8.1 in the first minute, 7.3 through the
climb, so "near climb vs elsewhere" (1.57) mostly measures circling, where delays are noisy; compare
within phases (climb start against mid-climb, glide against glide) before quoting it. First climber
becomes head 29.7% [28.8, 30.7] against 26.4%: weak.
*Fixed (2026-11-05):* shuffled null removed. Head changes per hour by group phase: all gliding 4.6
[4.45, 4.77], mixed 8.3 [8.07, 8.56], all climbing 5.9 [5.57, 6.06]. The lead changes most when the
group straddles a thermal (some in, some out). Climb start against mid-climb: 1.08 [1.02, 1.15], so
little beyond circling.

**B4. Thermal exits (user's request, 2026-11-05).** Do followers leave a thermal at the same height
as the leader, below or above? Group thermal: f > 0 between two glides of the whole group. Per
member: exit altitude and time at the end of its last climb run there. Leader: the head before the
thermal, and the head of the glide after. dz = z_exit(follower) - z_exit(leader), shares above /
same / below (tolerance from the GNSS altitude noise), dt, dz over the leader's gain. Null: random
leader (median 0). Link with rotation: does the highest or the first to leave become the next head?
Script `exits.py`; in the note with rotation (the lead changes most as a group leaves a thermal).
*Done (2026-11-05, `exits.py`):* 35 294 thermals with 2+ climbers, 35 694 followers (leader before).
Median dz -5.7 m [-6.5, -4.9] (leader after: -1.2 m); the gain, free of logger offsets, is 10 m
smaller for the follower; entry heights equal (+1.3 m). At 20 m: 42% below, 34% above, against 38%
each with a random leader. dz grows with dt: who stays longer leaves higher. The first to leave heads
the next glide in 66% (near the definition of the head), the highest in 48%, against 48% at random.

**C. Between groups.** Do pilots use other groups, not only their own? Neighbours: gliders within R at
the same instant, not in contact.
- C1 thermal marking: a climb starts within r of where a non-member climbed in the last dt, real
  against the other pilot shifted in time;
- C2 advantage and density: the group advantage of vs-solo against the number of neighbours, compared
  within the same site-day (density goes with the quality of the day);
- C3 encounters: when two groups come within R, how often they merge.
Limit: only uploaded tracks are seen; the density is underestimated and the effect diluted.
Literature: social information in soaring birds (storks, vultures), to verify.

**D. Pilot network.** Nodes: named pilots; edges: pair-days. Degree and strength over the number of
flights; communities against clubs (NMI); assortativity by experience and wing class. Null: the
configuration model of `partners.py`.

**Dropped:** degree profile per episode (three episodes in four are pairs, and its shape follows from
the definition), generic centralities.

## Order and workflow

Step 0, then B1 and B3, B2, A2, D, C. For each study: design and review with Opus; implementation in a
fresh Sonnet session that reads this plan; `/code-review medium` on the diff; numbers in the text
checked against the JSON. One literature agent (Sonnet) once for all studies.
