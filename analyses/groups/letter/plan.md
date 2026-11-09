# Letter plan: group flight

A four-page letter (PRL format, about 3500 words, 4 figures) with a Supplement, written and
computed in xc-lab. Numbers below: run `d120_eh500_ez300_th45_g120_q0.1`, paragliders, open flights
(opening > 0.1) unless said.

## Message

**Claim.** A group straightens a pilot's route by putting them on the right one. Raw, group time is
more persistent than solo flight and has longer transitions; on the same route, once who flies in
groups is accounted for, the persistence gap is gone; within one flight, persistence rises when the
pilot joins a group and falls when they leave. What the group adds on a given route is lift: more
time in climb.

**One line for the reader.** Flying together does not change how a pilot flies a route; it changes
which route they fly.

**Not claimed.** That the group causes the route rather than gathering on it (the same-flight test
narrows this, the encounters study says groups form mostly as chance meetings on shared lines); that
pilots in a group take more risk.

## Title candidates

1. Together, glider pilots find the way
2. Flying together puts glider pilots on the right route
3. The group chooses the route
4. Soaring pilots in groups fly the same way on a better route

## The three comparisons

| | design | persistence, ΔH_loc(1000 s) | other |
|---|---|---|---|
| raw | G − S, open flights | +0.018 [+0.015, +0.021]; < 0 below 170 s | transitions: tail μ 3.52 [3.36, 3.72] (G) against 4.54 [4.43, 4.65] (S), 4.37 (A), all flights |
| same route | G − S\|G, route, month ±2, hour ±2, wing class | +0.010 [+0.004, +0.017]; net of selection +0.004 [−0.007, +0.015], at 1840 s −0.006 [−0.028, +0.016] | time in climb net +2.4 [+0.8, +4.0] pp; transition survival of G on S\|G |
| same flight | g = (D_in + D_out)/2, windows of 2400 s | +0.026 [+0.020, +0.032]; post − pre +0.002 | climb +2.8 pp, searches ending in a climb +2.7 pp, glide +1.25 km/h and −5.5 m/km, all bands off 0 |

Selection = A − S|A, the same pilots flying alone on the same routes. The route explains about 70%
of the raw long-lag gap on all flights too (`route.py`, `isolated.py`).

On the same route the glides of G are slower and steeper, the reverse of the same flight; the same
deficit appears in A − S|A, so it belongs to the pilots. One sentence in the letter, details in the
Supplement.

## What goes in, from the two reports

Chosen for the claim; everything else goes to the Supplement or stays out.

From `vs-solo`:

- the three comparisons above, and the 170 s crossover (median glide 201 s, climb 155 s);
- the route explains about 70% of the raw gap, on every radius and cover tried;
- size: H_loc(1000 s) rises with group size pooled (0.889 at 2, 0.944 at 7+), with no common order
  within a site-day: big groups form on good routes (Supplement);
- experienced and beginners show the same pattern (Supplement).

From `structure`:

- groups are lines along the route; the lead changes 3.3 times per hour and goes more often to the
  more experienced pilot (58.5%) on the higher wing (64.3%): who sets the line;
- followers turn where the leader turned, not when (heading correlation peaks at τ = ℓ): the line is
  passed in space;
- following the leader's track with one's own delay explains about 90% of the contraction in a
  thermal; followers leave the thermal at the leader's altitude;
- a place keeps producing climbs for 40–50 min, and a glider climbing there now adds little to one
  that climbed 5 min earlier: lift is tied to places, a route is a chain of them;
- groups form mostly as chance meetings on shared lines (pairs attract ×1.42 against independent
  meetings);
- partners return (37.6% of pair-days against 26.7% at random) and the size distribution follows a
  birth-death chain with the measured rates (Supplement).

## Structure

1. **Lead.** Soaring flight crosses invisible, short-lived lift; soaring birds use each other to find
   it. Whether grouping helps is hard to measure in the field because groups form where flying is
   good. Here: 20 years of French tracks, groups found by a rule, three comparisons that remove the
   route and then the pilot.
2. **Data and groups** (Fig. 1). 154 598 paraglider flights, 33 501 ever in a group, 53 293 hours
   in a group; the rule in one paragraph, the rest in the Supplement.
3. **Raw** (Fig. 2). More persistent beyond one leg, less below; heavier tail of transitions.
4. **Same route** (Fig. 2). The persistence gap shrinks, and net of selection it is gone; the tail
   (to check).
5. **Same flight** (Fig. 2). Joining switches persistence on, leaving off.
6. **How the line passes** (Fig. 4). Leaders, spatial following, places that hold lift; more time
   in climb on the same route.
7. **Terrain** (Fig. 4, if the counts allow). The same in mountains and in plains and hills.
8. **Discussion.** The group as a way to find the route; the parallel with birds; what stays open.

## Figures

1. **Groups and the design.** (a) One episode on a map, solo and group stretches coloured, climbs
   marked; (b) its membership over time; (c) sketch of the three comparisons.
2. **Persistence.** (a) ΔH_loc(τ) for raw, same route net of selection, same flight on one axis,
   bands, vertical line at 170 s; (b) H_loc(1000 s) at pre, start, end, post with the control pairs.
3. **Transitions.** (a) Survival of transition durations S, A, G with the power law above x_min;
   (b) Hill μ against the threshold with bands; (c) same route, G against S\|G.
4. **How.** (a) Heading correlation C(τ − ℓ) of pairs; (b) forest of the phase statistics in the
   three designs; (c) the persistence gap by terrain class.

## New computations

All need SSD_DISANTE mounted.

1. **Tail of transitions.** Hill μ against x_min (1000–4000 s) with block-bootstrap bands for S, A, G
   on open flights; the number of runs above each threshold; the same on the same route with the
   match weights (G against S\|G and A against S\|A). What the long transitions are: duration,
   length, mean vertical speed, straightness, so that "longer transition" can be read as a glide or
   as flying along a line of lift.
2. **Terrain.** First check the DEM already on disk (product, resolution ≤ 250 m, coverage with
   Pyrenees and Corsica, datum); Copernicus GLO-90 if it falls short. Classes by Hammond (1964),
   automated as in Karagulle et al. (2017): local relief in a ~10 km window and share of gentle
   slope give plains, hills, low mountains, high mountains (thresholds checked in `references.md`);
   Kapos et al. (2000) as a check, mountain or not. A class map on a grid, each fix reads its cell;
   a piece takes a class when about 80% of its fixes fall in it, otherwise it is mixed and left out;
   in the same flight the four windows share the class. Measure first how many flights change
   class: if few, class whole flights. Then the counts of pieces of G, A, S per class, and raw, same
   route and same flight per class if each class keeps enough pieces (letter: plains and hills
   against mountains; four classes in the Supplement).
3. **Same flight, more statistics.** Cross-country speed over 30 min, glide length and height lost
   per glide in `prepost.py --open`, as g with its band.
4. The figures, from the JSON.

## Supplement

- S1 Data: the FFVL archive, cleaning, sampling, open flights.
- S2 Phases: Vilpellet's labelling at 1 Hz, tracks up to 5 s splined.
- S3 Groups: the identification rule (δ, ε_h, ε_z, θ, q, τ*), the time-shift null model, one scheme.
- S4 Persistence: TA-MSD, H and H_loc, the pieces and their minimum length.
- S5 Uncertainty: two-way bootstrap of pilots and blocks (Cameron, Gelbach, Miller 2011) and its
  coverage check.
- S6 The three designs in full: route match (ρ, cover, direction), calipers, the event windows, g and
  d w, window length.
- S7 Tail: the Hill estimator, threshold choice, bands.
- S8 Terrain classes: Hammond's landforms from the DEM, the class of a piece, flights that change class.
- S9 Group structure: the numbers quoted in the letter and their nulls.
- S10 Robustness: calipers, radius and cover, years left out, experience, size, hang gliders.

## Files

- `letter.tex`: the draft (revtex, PRL); pending results in red. Reads the macros of
  `../vs-solo/generated/` and `generated/macros.tex`.
- `supplement.tex`: the skeleton of S1–S10, each section naming its source note.
- `code/macros.py`: the `Lt*` macros (counts, period, grouping parameters).
- `code/figures.py`: Fig. 2 (`figures/persistence.pdf`) from `raw.json`, `matched.json`,
  `prepost_2400.json`.
- `references.md`, `letter.bib`: verified references.
- `code/figures.py` also draws Fig. 3 (`figures/tail.pdf`, from `tail.json`) and Fig. 4
  (`figures/how.pdf`, from `headings.json`, `prepost_2400.json`, `terrain_compare.json`).
- `../tail.py`: Hill index of transitions against the threshold, same route, descriptors of the
  long transitions. `../terrain.py`: Hammond classes from the IGN BD ALTI 25 m (cache in
  SSD_DISANTE/xc-lab/terrain/hammond_200m), `--compare` for the three designs by class.
- To come: Fig. 1; bands of the descriptor medians (Fig. 3c); the Supplement text.

## Results added on 2026-11-09

- Tail: Hill index above 1704 s, G 3.33 [2.96, 3.71], S 4.58 [4.39, 4.77]; bands apart from 1194 to
  2383 s; same route not resolvable (59 G transitions). Transitions above 2000 s: ~25 km, median
  vertical speed −0.15 m/s in G, rising 42% of the time: lines of lift, straighter in G.
- Terrain, against the expectation of a stronger effect in plains: raw +0.010 (plains and hills),
  +0.011 (low mountains), +0.046 (high mountains); same flight +0.007 [−0.005, +0.018] (lowland,
  158 events) against +0.032 [+0.025, +0.040] (mountain). Same route in lowland: 13 pieces.

## Literature

Cited only where it carries a sentence of the letter. Each item checked (authors, year, journal, DOI,
what it shows) in `references.md` before use.
