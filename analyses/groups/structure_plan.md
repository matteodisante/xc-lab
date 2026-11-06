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
*Design (agreed 2026-11-06).* Trap: pilots strung along one track turn at the same places, so the
heading peak sits at tau = l* with no interaction, and the sign check with l* holds almost by
definition. The question becomes: in glide, does the follower turn at the same place as the head
(tau_max = l*: retracing, or the same air) or at the same time (tau_max short and independent of l*:
a response to seeing the other)? Quantity: dtheta_i(t) = theta_i(t) - route direction of the glide
(theta: horizontal heading on the 5 s grid of glide.py, enough to tell tau = l*, tens of s, from
a few s; 1 Hz only if a short-tau peak appears); C_ij(tau) = corr(dtheta_i(t), dtheta_j(t + tau)), tau > 0 when j
follows i; tau_max and C_max per pair-glide; l* = median delay of the pair in that glide. Second
signal dv (horizontal speed about its glide mean), same code, kept only if it differs. Population:
the glides of glide.py (pairs, every pair of the 3+ groups, chance pairs). Tau range from the
measured |l*| in glide (95th percentile). Null: chance pairs in the same |l*| class. Figure:
tau_max against l* (medians per class with bands, line 1:1), real against chance; beside it the
mean C(tau) for two or three l* classes. Reading: no peak above the null, independent lines; slope
1 and real = chance, the same air; slope 1 with C_max real > chance, retracing; slope < 1 or a
short-tau peak absent in chance pairs, a response. The directed "who responds to whom" network in
3+ only if a response appears (else it repeats B3).
*Done (2026-11-06, `headings.py`):* 47 215 real pair-glides (pairs and every pair of the 3+ groups),
1041 chance; TAU_MAX 120 s (95th percentile of l, which the contact window delta = 120 s bounds),
l classes cut at 10, 30, 65 s. The pooled heading peak sits at tau = l in every class (pairs 1.6, 17,
46, 84 s at l = 5, 20, 45, 100 s; the speed peak the same and closer): the follower turns and changes
speed where the head did, not when; no short-tau peak, C(0) is the tail of the peak. Heading C(l):
pairs 0.36 to 0.19 as l grows, 3+ 0.33 to 0.29 (the tail keeps the head's line through the members
between), chance 0.04 and 0.11 at l = 5, 20 s, 0.32 and 0.21 at 50, 107 s: close pilots share the
line more than chance, far ones as much as the place gives. Speed: real above chance at every l, but a
chance pair passes r later, in changed air, so it is no sign of interaction. The per-pair slope of
tau_max on l (0.28) is not used: single peaks are noisy. No response, so no directed network.
Figure: C(tau - l) of the pairs by class, and C(l) by kind.

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
*Design (2026-11-06, the user delegated the choice).* C2 dropped: the density is underestimated (only
uploaded tracks), goes with the quality of the day, and within one site-day few flights remain; C1
already asks whether others help. C1 marking: share of climb starts with a marker, a glider not in
contact with the pilot climbing within r of the start in the dt before it; for group members (markers:
other groups, solo gliders) and solo pilots; r and dt the 90th percentiles of the distance and delay
between the entries of two members of one group thermal (exits.py), the scale of "same thermal". Null:
the markers shifted in time by +-s; the share against s shows how long a place keeps climbing, and S is
where it levels off. Excess over the null is marking or two pilots finding one active thermal; the
data do not separate them, said in the note. C3 encounters: per close pass within R with headings
within theta_max of the run, the probability that a pair interval forms within T, against the
closest distance, for solo-solo, solo-group, group-group; the solo-solo curve is the reference
(groups absorb if solo-group lies above it, stay closed if group-group lies below).
*C1 done (2026-11-06, `marking.py`):* 214 395 climb starts of group members, 1 280 161 of solo pilots
(5 s grid; markers: labelled flights); R = 829 m, DT = 152 s (90th percentiles over 49 557 pairs of
entries of group thermals); the contact and the pilot's kind are read at t - DT, so a pilot who joins
its marker still counts. Marked starts: group members 0.245, solo pilots 0.145. "Now" ratio, real over
the markers shifted by 5 min (the first window that does not overlap): group members 1.02 [1.01, 1.03]
(other-group markers 0.92 [0.90, 0.93], solo gliders 1.05), solo pilots 1.09 [1.08, 1.09]. A glider
climbing right then adds little over one that climbed at the same place minutes before or after:
places keep producing climbs (the share halves in 46 min for group members, 39 for solo pilots, an
upper bound as shifted markers leave their flights). Marking from within about 1.5 km (DT at glide
speed) is a few percent at most; marking seen from farther falls in the null too. Dropped: the ratio
at large shifts (about 2 at 45 min), mostly dilution. Oddity left open: other-group markers below 1
for group members (groups in file along one route pass a thermal minutes apart?).

**D. Pilot network.** Nodes: named pilots; edges: pair-days. Degree and strength over the number of
flights; communities against clubs (NMI); assortativity by experience and wing class. Null: the
configuration model of `partners.py`.
*Design (agreed 2026-11-06).* Known already (partners.py): repeated partners and same-club pairs above
chance; D asks what that leaves open. Population and null as partners.py (named paraglider pilots,
pair-days, configuration model within the site-day pools, which keeps who flies in a group where and
when). D1 loyalty per pilot, general or a core? k distinct partners over s pair-days, s fixed by the
null; per pilot the quantile of k_real among the null draws (mid-rank for ties); its histogram is
uniform without loyalty, shifted if loyalty is general, a spike at 0 over a uniform rest if a core;
by classes of s. D2 how much of the loyalty clubs hold: the excess of pair-days on repeated pairs
(real - null) split into same club, different clubs, club unknown. D3 who flies with whom: mean
partner experience (earlier flights in the catalog at the date, log) against own experience, and the
same for the wing class (WING of hierarchy.py), real against null; a curve above the null for
beginners is mentoring, a stronger diagonal is assortativity; for the wing, assortativity may be
physical (similar speed keeps two gliders together longer), said as a hypothesis. Figure: (a)
histogram of the quantiles, (b) the excess inside and outside clubs, (c) partner experience against
own. Communities (Louvain, NMI with clubs) dropped: same question as D2, a less readable number, and a
new dependency.
*Done (2026-11-06, `network.py`):* first a fix of the null of partners.py: in 289 pools (27% of the
pair-days) every TRIES draw held a self or double pair and the draw dropped them, so each draw lost
about 1300 pair-days and s changed for up to 898 pilots. Such pools are now mixed by degree-preserving
swaps (SWAPS = 10 per pair; the share of original pairs left levels off at 0.43 from 5); s is fixed in
every draw. Partners at random: repeat share 26.7% (was 25.7%), same club 24.2% (24.0%); README and test
updated, the note's partners macros change at the next structure_tables run. D1 (randomized PIT, flat
without loyalty): share of pilots with q < 0.05 is 0.11, 0.26, 0.58 for s <= 5, 6-14, > 14 (1413, 1100,
1147 pilots); median k / k_null 1.00, 1.00, 0.91. Loyalty grows with activity: a small core among
occasional pilots (spike at 0 over a flat rest), the habit of most active ones. D2: excess pair-days
on repeated pairs 2279 [2212, 2332] same club, 1032 [974, 1106] different clubs (69% / 31%), against 59%
same club among the null's repeated pair-days: clubs hold a little more than their share; a third of
the loyalty crosses clubs. D3: experience assortative (corr 0.238 against 0.160-0.172), no mentoring
(beginners' partners 1.25 against 1.32-1.34 in log10(1 + E)); wing class corr 0.489 against
0.330-0.339, same class 0.457 against 0.395-0.401.

**Dropped:** degree profile per episode (three episodes in four are pairs, and its shape follows from
the definition), generic centralities.

## Order and workflow

Step 0, then B1 and B3, B2, A2, D, C. For each study: design and review with Opus; implementation in a
fresh Sonnet session that reads this plan; `/code-review medium` on the diff; numbers in the text
checked against the JSON. One literature agent (Sonnet) once for all studies.
