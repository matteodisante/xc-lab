# Letter review: open items (2026-11-12)

Review of `letter.tex` (title since this review: "Good company is a good route"). This file lists what was fixed and
what is still open, in priority order. Paths are relative to `xc-lab/analyses/groups/`.

## Remaining work, in order (state at commit 57efdac, 2026-11-12)

Everything below this list is a record: items marked Result or Decided are done, do not redo
them. `analyses/groups/intergroup_plan.md` is executed (report section "Point 7"); read it only
as background.

1. Compile the letter (only when the author asks): `latexmk -pdf letter.tex` in `letter/`. Check
   length (PRL: about 3750 words plus figures), that Tables I and II fit one column, no undefined
   references. If too long, move Fig. `survival` to the supplement first, then shorten "How a
   group holds a route".
2. Compile the report in place (only when the author asks): `latexmk -pdf
   group_vs_solo_comparisons.tex` in `vs-solo/comparisons/`; it already compiles in a temporary
   folder (37 pages).
3. Fig. 1 of the letter (item 5 below): design with the author first, then code in
   `letter/code/figures.py`.
4. Write `supplement.tex`: every section is still a red placeholder; the sentences moved out of the
   letter are already there. Turn each section into text and one table, macros only.
5. Number density of the letter (item 5 of the list below, "Number density").
6. Smaller points (item 6 below): terrain gain relative to the headroom, the "$-$0.000" format of
   `\PreCAHK`, a paraglider-only episode count, the venue.
7. Report only, optional: the open variants of the inter-group analysis (item 4c below).

## Rules for whoever works on this

- Every number reaches the text through a macro. Never type a number, never edit `generated/` by
  hand. Letter macros come from `letter/code/macros.py`; report macros from
  `vs-solo/generated/comparisons/macros.tex`, `vs-solo/generated/macros.tex`,
  `vs-solo/generated/structure_macros.tex`.
- Before citing a macro, open the report section or the code line that defines it and check sample,
  lag and units.
- Do not run `latexmk` unless the author asks. Do not commit unless asked.
- Items marked **author decides** change the message: prepare the options, do not apply them.
- Say in plain words what each edit does before making it; one small change at a time.
- Heavy runs: one process at a time, measure the memory peak first (8.6 GB machine).

## Fixed on 2026-11-12 (do not redo; the draft was not recompiled after these edits)

1. Abstract: "the gap in persistence vanishes" -> "is no longer detectable".
2. Same route: added the band `\MatDeltaHKABand` to A - S at 1000 s; said that 1840 s is read on
   the longer pieces; the sentence "more persistent also when alone" now rests on the 1840 s value,
   the only one whose band excludes 0.
3. Same route: "a pilot in a group is as persistent as alone" -> "we detect no effect, within these
   bands" (the band of the effect at 1000 s reaches +0.015, close to the raw gap +0.018). Caption of
   Fig. `persistence`(b) aligned.
4. Same route: the 71% (`\RouteExplained`) now comes with its two numbers and its sample
   (`\RouteBaseDHloc` -> `\RouteDHloc`, the route-only match of `vs-solo/general/section_route.tex`;
   formula in `vs-solo/code/tables.py:461`). Before, it contradicted +0.018 -> +0.010.
5. Same flight: cited the existing control pairs `\PreCAHK`, `\PreCSHK` (two consecutive windows
   drawn at random in solo time; `vs-solo/comparisons/sections/prepost.tex`).
6. How the line passes: on the same route glide speed and height lost per km do not change
   (`\MatDeltaGlideD`, `\MatDeltaSinkD`).
7. How the line passes: cited the null of the heading correlation (no second peak near zero lag;
   surrogate pairs correlate as much from `\StHdLChanceThree` s of delay; source
   `structure/sections/structure.tex`, lines 76-135).
8. Discussion: "faster and flatter glides" was attributed to "a given route"; it holds within a
   flight only. Sentence corrected.

## Open, in priority order

### 1. Placebo for the same-flight design (decides the message; do this first)

**Problem.** `g = (D_in + D_out)/2` cancels a linear drift along the flight, nothing else. If
persistence follows an inverted U along a flight (low after takeoff and before landing, or the daily
cycle of thermals), four windows before / start / end / after give `g > 0` and `d_w = 0` with no
group effect: with a profile `-a t^2` and windows at t = -1.5, -0.5, +0.5, +1.5 one gets
`D_in = D_out = 2a`. This is the pattern the letter reports (`\PreGHK` +0.026, `\PreDwHK` -0.001).
The existing control pairs (fix 5) are two consecutive windows at random positions: they test
drift, not this.

**Status (2026-11-12).** The code is written and tested (`prepost.py --open` now adds "mock
events"; 16 new tests in `test_groups.py`, 314 pass). On a synthetic inverted-U profile with no
group effect the real `g` is +0.012, the mock `g` +0.011 and their difference is compatible with
zero; on a flat profile with a true effect the mock `g` is zero. A trial on 500 flights ran in
115 s with 1.3 GB, and the existing values of the JSON did not change. Mocks are drawn among the
flights never in a group, at the same flight-clock times, within the calipers of the same-route
design (month, hour, wing class), 5 per real event, with a second variant on the same site.
**Result (full run, 2026-11-12): the placebo holds.** On 7 666 mock events for 1 534 of the 1 595
real events, `g` of `H_loc(1000 s)` is +0.000 [-0.002, +0.003]; the real `g` on the same events is
+0.028 [+0.021, +0.034] and `g` minus mock +0.027 [+0.021, +0.034]. With mocks from the same site:
mock +0.003 [-0.001, +0.006], net +0.026 [+0.019, +0.034]. The 2 383 values already in
`prepost_2400.json` did not change (backup compared). Macros regenerated with
`vs-solo/code/comparisons.py` (`\PreMockGHK`, `\PreNetGHK`, `\PreMockSubsetGHK`, `\PreMockEvents`,
`\PreMockRealEvents`, `\PreMockFlights`, and the `Site` variants). In the letter: two rows of
Table I and one sentence of "Same flight"; the control pairs moved to the supplement. The message
and the title stand.
Still to run for item 3 (cross-country speed, code ready): `uv run python raw.py --open`, then
`/usr/bin/time -l uv run python matched.py --open --stats`, then `vs-solo/code/comparisons.py`.
Note: regenerating the macros also updated five opening macros (`\FlightsFixed`, `\MedianFixed`,
`\MedianFixedG`, `\MedianFixedS`, `\BelowFixed`) and one row of `opening_table.tex`: the opening
JSON had been rewritten on 3 November for the dataset up to 2 760 s and the macros were older.

**Design, as specified.** Build pseudo-events in flights with no group:
- for each real event (`\PreEvents` of them), take one or more flights of S, and separately the
  solo time of flights of A away from their episodes;
- place four windows of `\PreWindow` s at the same time since takeoff as the real event, with the
  same gap between the "start" and "end" windows (the episode duration); require the same month
  +-2 and hour +-2 as in the route match, and the same terrain class if the sample allows;
- compute `D_in`, `D_out`, `g`, `d_w` of `H_loc(1000 s)` with the same code as the real events
  (`vs-solo/code/comparisons.py`, the part behind `sections/prepost.tex`) and the same bootstrap;
- export macros (for example `\PrePlaceboGHK`, `\PrePlaceboGHKBand`, `\PrePlaceboEvents`).

**Done when.** The placebo `g` with its band is known. If the band includes 0: cite it in one
sentence of the "Same flight" paragraph, replacing or joining the control-pair sentence. If the
placebo `g` is comparable to +0.026: report `g - g_placebo` and tell the author, since the message
of the letter changes.

### 2. Title, last sentence of the abstract, claim of the introduction (decided 2026-11-12)

**Status.** The author chose the title "Good company is a good route" (in `letter.tex` and
`supplement.tex`). Aligned with it: the last sentence of the abstract ("What a group gives a pilot
is the route"), the claim of the introduction ("the advantage of company is the route it flies"),
the opening of the Discussion ("Two features of the effect tie it to the route"). Nothing to do
here unless the placebo of item 1 fails; then re-read these three sentences. The reasoning is kept
below as a record.

The title and "A group helps a pilot by putting them on the right route" are causal. The Discussion
says the data do not separate "the group leads pilots to the good line" from "the group gathers on
it", and the letter's own numbers lean to the second: pairs attract only `\StEnSoloPair` (1.42)
times more than independent meetings; a glider climbing now adds a factor `\StMkNowGroup` (1.02)
over the place itself; followers do not answer the leader's turns (fix 7). Also "The three
comparisons agree" is loose: the same-route design finds no effect, the same-flight design finds a
jump; they fit one reading, they do not agree.
Option to propose: message "the advantage of company is the route" (the group is where the route
is), with a title in that sense.

The author wants a striking title that the paper supports. Three candidates, all true whether the
group leads to the line or gathers on it, and whatever the placebo of item 1 gives:
- "In soaring flight, good company is a good route": the three designs (the raw advantage of
  company reduces to the route). Recommended.
- "Follow the line, not the leader": the heading result (followers retrace the line and do not
  answer the leader's turns) and the marking of places (factor `\StMkNowGroup`).
- "The group is where the route is": the closing sentence of the Discussion.
The present title, "Together, glider pilots find the way", says the group finds the route; the
data do not show that.

### 3. One tangible outcome is missing (needs a run)

The introduction asks whether company makes flight "more efficient"; the only outcome is
`Delta H_loc` of 0.02-0.03 on 0.9. A tangible quantity exists in the general report only:
"cross-country speed over 30 min (km/h)" (`vs-solo/code/tables.py:132` and `:516`), macros
`\XcS` 17.2, `\XcG` 20.0, `\XcA` 20.0, `\DXcRaw` +2.81 [+2.59, +3.01], `\DXcSdh` (same site-day and
hour), `\DXcSel` (selection), `\DXcFlight` +0.70 [+0.60, +0.78] (group, same flight).
`\SpeedS/G/A` is glide speed, not cross-country speed.
- Do not cite these in the letter as they are: the sample and the same-flight estimator of the
  general report differ from the three designs of the letter.
- To do: compute the same quantity in the three designs of `vs-solo/code/comparisons.py` (raw on
  the letter's sample, same route with the selection term, same flight with `g` on the windows of
  `\PreWindow` s), export macros, then add one sentence in "Raw comparison", one in "Same route",
  one in "Same flight".
- Tell the author: `\XcA` equals `\XcG`, so in km/h the flights that join groups go as fast alone
  as in a group; only +0.70 km/h is left within the flight. This bears on item 2.

### 4. Structure, topics, figures and numbers (restructured 2026-11-12; the author still has to read it)

The author's requirements: the order of the arguments must follow one thread; the choice of topics
and of plots matters most; the text must not be full of numbers and intervals; two appendices, on
finding the groups and on the uncertainty bands. What was done in `letter.tex`:
- Order: intro; "Data and groups"; "Measuring persistence"; "Raw comparison"; "Same route"; "Same
  flight"; "How a group holds a route" (places, files, followers retrace, chance meetings, phases);
  "Terrain"; "Discussion"; Appendix A (`appendix_groups.tex`), Appendix B
  (`appendix_uncertainty.tex`), both after the bibliography.
- Topics: the tail of transitions (Hill index, Fig. `tail`) left the main text. It is a raw-only
  result in a letter whose point is that raw comparisons mislead, and on the same route it cannot be
  resolved. Its paragraph and figure are now in `supplement.tex`; the Discussion keeps two sentences.
- Figures of the main text: Fig. 1 design (still a placeholder), Fig. 2 `persistence.pdf`, Fig. 3
  `how.pdf`.
- Numbers: the persistence results of the three designs are in Table I (`tab:persistence`) with
  their bands; the text cites the table and one or two numbers per paragraph. Everything removed
  from the text was moved, as sentences with macros, into the matching sections of
  `supplement.tex` (matching parameters, 1840 s values, route-alone match, phases, terrain, leaders).

Still to do here:
- Add the placebo row to Table I once the macros of item 1 exist (`\PreMockGHK`, `\PreNetGHK`).
- The two appendices (about 340 words each, two equations each) were written by a Sonnet agent
  from the internal reports and the code, then reviewed: no `\pending{}` is left; three macros were
  added to `letter/code/macros.py` (`\LtGapMin`, `\LtNullRadiusKm`, `\LtSiteKm`) and
  `letter/generated/macros.tex` was regenerated (only those three lines changed). The author still
  has to read them. Facts fixed by the review, to keep in mind when editing: the contact is
  two-sided (delay before or after); a block joins episodes and site-days; the band is
  `estimate +- z_0.95 sqrt(V)` with `V = max(V_P + V_B - V_C, V_P, V_B)`, not a percentile band;
  the coverage check covers the three designs only, not the terrain contrasts or the tail.
- PRL prints appendices as "End Matter" after the references: check the current length rules of the
  journal before submitting.
- Recompile (when the author asks) and check: length, float placement, that Table I fits one
  column, undefined references.
- Turn the sentences moved to the supplement into proper sections and tables.

### 3b. Cross-country speed: result (full data, 2026-11-12)

Cross-country speed over 30 min (km/h), existing values of `raw.json` and `matched.json`
unchanged: raw G 21.9, A 21.5, S 20.5, G - S +1.4 [+1.0, +1.7]; same route, net of selection, +1.3
[+0.4, +2.2] (1 127 pieces of G, 368 of A, pieces of at least 2250 s); same flight g +1.4
[+1.0, +1.8], mock +0.2 [+0.0, +0.4], net +1.2 [+0.7, +1.6]. Unlike persistence, speed survives
the same-route design. In the letter: one sentence in each design paragraph and in "How a group
holds a route"; all bands in the supplement.
**Decided (2026-11-12):** the abstract and the last sentence of the introduction now say that on
the same route the group still helps (more time in climb, faster progress) and that the
same-flight jump is absent in the mock events. The title stays "Good company is a good route": the
author replaces it only with something equally striking.

### 4b. Tails of the transitions in the three designs (asked by the author on 2026-11-12, in progress)

The author wants the tail back in the letter in a stronger form: the Hill index `mu` of the
transition durations with an uncertainty band in the three designs (raw: S, A, G; same route: G and
A against their matched solo pieces; same flight: before, in group, after), one survival plot, and
the full analysis in the report `vs-solo/comparisons` (new sections with blue titles: tails,
terrain). A Sonnet agent is doing the analysis and the report sections; the letter insert is
written after its results are reviewed.
- Hypothesis of the author: before and after similar, in group a smaller `mu` (longer glides more
  likely), because in a group pilots dare more.
- Caution: the numbers in hand point the other way for the reading, not for the fact. Long
  transitions in a group lose less height per km and rise a larger share of the time, which fits
  lines of rising air more than daring. The agent tests both readings (total height spent, arrival
  height). The letter states the reading the numbers support.
- Same route: only `\LtTailRouteRunsG` group transitions exceed the threshold, so that band will be
  wide.
- When the results are in: add a short paragraph after "Same flight" with a small table of `mu` in
  the three designs and the survival figure (`letter/figures/survival.pdf`), load the new macro
  file `../vs-solo/generated/comparisons/tail_macros.tex` in the preamble, and shorten the two
  sentences on the tail in the Discussion.

**Result (full data, 2026-11-12).** The author's hypothesis is not supported. Raw: mu G 3.33
[2.96, 3.71] against S 4.58 [4.39, 4.77]. Same route: effect +2.78 [-0.80, +6.37] (59 group
transitions). Same flight: before 3.30, in group 4.24, after 4.95; in - mean(before, after) +0.12
[-0.66, +0.89]; after - before +1.66 [+0.92, +2.39], because 44% of the long transitions after a
group are final glides to landing. The raw heavy tail comes from transitions that cross a group
boundary (mu 3.52) or start before the departure. Long transitions in G cost less height (34
against 72 m/km), and so do solo controls on the same route: cheapness belongs to the route.
"Dare more" is not supported (paths not longer, total height lower). In the letter: paragraph
"Long glides" after "Same flight", Table II (`tab:tail`), Fig. `survival`; Discussion shortened.
Report: sections "Point 5" (tails) and "Point 6" (terrain), blue titles, compiled in a temporary
folder (33 pages, no errors); the report PDF was not rebuilt.
Open: the letter is now four figures and two tables; check the length at the next compile and,
if it runs long, move Fig. `survival` to the supplement and keep Table II.

### 4c. Intra-group against inter-group effect (asked by the author on 2026-11-12, in progress)

Design in `analyses/groups/intergroup_plan.md`; a Sonnet agent implements it (`intergroup.py`) and
writes a new section of the report with a blue title. Three steps: the geometry (length and width
of a group against the gap to the gliders ahead of it), the two effects on pieces classed by state
(solo, group) and neighbourhood (alone, others ahead, others only behind), and the range of the
interaction against the distance to the glider ahead. The ahead-behind contrast tells information
from a frequented route. It enters the letter only if the inter-group effect exists and is the
stronger one; decide after the results are reviewed.

**Result (full data, 2026-11-12): not in the letter.** Groups are small (length 277 m, width
117 m, medians) and the nearest glider ahead of the leader is at 2.5 km (median), so an
inter-group effect is possible by distance. It does not show on persistence: inter-group
(solo, others ahead) - (solo, alone) is -0.024 [-0.027, -0.022] raw and +0.016 [-0.005, +0.036]
within the flight; intra-group +0.009 [+0.005, +0.013] raw, +0.011 [-0.008, +0.030] within the
flight. Ahead minus behind has the sign opposite to information (-0.047 within the flight). The
only hint of information: searches end in a climb 2-3 points more often with a glider ahead within
1 km, reversed beyond. Report section "Point 7" (blue title), appendix `app:intergroup`.
Open, for the report only: the chance level of the geometry shares, a within-flight version of
the search curves, variants of the direction window and of F_LOW / F_HIGH, an A-only variant.

### 5. Fig. 1

Still a placeholder: (a) one group episode on a map, (b) its membership over time, (c) the three
designs. Needs the tracks (disk `SSD_DATA`) for (a) and (b); (c) is a scheme (TikZ). Add it to
`letter/code/figures.py`.

### 6. Smaller points

- Terrain: "less to gain in plains" (ceiling) and "route harder to find in mountains" are two
  readings; the text mixes them. A gain relative to the headroom `1 - H_loc` would separate them
  (rough values from the macros: 0.17 plains and hills, 0.11 low mountains, 0.31 high mountains);
  compute it properly with bands in `letter/code` before using it.
- `\LtEpisodes` (17 169) counts the episodes of paragliders and hang gliders together: the run has
  6 094 hang-glider flights and `vs-solo/code/facts.py` takes `len(episodes)`. The letter is about
  paragliders, so the macro is no longer cited anywhere. To cite a number of episodes, add a
  paraglider-only count to `facts.py` and `letter/code/macros.py` first.
- `\PreCAHK` prints "$-$0.000" in the text: the author decides the format in the code that writes
  it (`signed` in the vs-solo code).
- Still pending in the source: Fig. 1 and its caption, affiliation, acknowledgments, every section
  of `supplement.tex`.
- Venue: for PRL the physics content is thin (a change of 0.02 in an exponent). Decide between PRL
  and a journal such as J. R. Soc. Interface or PNAS before polishing, since limits differ.
