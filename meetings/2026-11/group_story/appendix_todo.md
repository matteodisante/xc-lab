# Appendix of the group story deck: what is left

From an independent audit (2026-11-27). A fix agent stopped halfway (usage limit): its edits compile but are
unreviewed. Goal: an appendix with no stale or wrong material, every number a macro, readable at slide size.
Verify each item on the JSON before fixing; fix the same sentence in `analyses/groups/story/group_story.tex`; put
`\isnew` on every changed frame.

## Text applied in the deck, to check (the note is not done)

- t-cycle: cycle range from `\PhCycleMin`/`Max` (checked on m-scale: 7.7–9.4, from the JSON).
- t-tails3: "on every flight no difference; per class two exceptions" (high-mtn searches, hill climbs).
- t-durnum: climbs slightly longer (+3 s); why m-units (per-flight mean, +1.1 [−0.3, 2.5]) differs.
- t-joinstab: "departures"; here "search" includes exit and unknown runs.
- t-openthreshold, t-opencohorts, t-opengaps: "cohort to 1 h". t-ogtab, t-og: "normalised".
- t-phperf: except slightly in the high mountains. t-phsearch: "share of the search-labelled runs".
- t-long: "320 s, the lower lag of the fit". t-gapchain: "negligible".

## Still to do

1. t-hillmethod, TAILS label of `slide_tails` (`story_slides.py` ~960), note ~706 and ~1378: the search tail holds
   found, refound and failed searches, not "searches that end in a climb". Redraw.
2. t-kindmain: the lower median is mostly because those flights are longer (equal medians from 9000 s), not where
   they fly.
3. t-longtab: resolved in every class (plains −0.007*).
4. t-refit: `\RefitSearchPerHourOurs` = search runs per hour before classification.
5. t-routeclasses: "60–98%", "(40%)", "least in the hills" typed and not reproducible: macros from
   `route_alpha.json` or remove.
6. t-phmain2: take on high mountains and every flight; A−S climb rate resolved (+0.0205 high mtn, +0.070 all).
7. t-phcomp: text cites the climb share, figure draws climbs per hour: make them agree.
8. t-routecheck: quote the shift of G−A between length and time. t-cycle / t-memorytab: say which peak (normalised or
   absolute). t-ogclass: 78% vs table 77%.
9. Links: orphans t-where, t-tails3, t-hill, t-groundalpha (pills from m-ground, m-tails); every appendix frame
   returns to every body frame that links to it (t-survival, t-tailnum, t-phsearch, t-phmain, t-durnum,
   t-chainedC, t-memorypath, m-kinds); t-phmain2, t-tools linked from the body.
10. Dynamics order: tools, phsearch, refit, phcomp, joinstab, phperf, phmain, phmain2, phall, searchkm, kmexample,
    thermalsmain, thermals, wind, hillmethod, hill, tails3, survival, tailnum, durnum.
11. Titles: drop "Which flights? ·" and "Persistent? ·" prefixes for the current section names. t-index: add
    Kaplan–Meier, joins, durations; drop "route gaps" if it has no frame.
12. Unreadable: redraw slide-sized (few panels, large fonts) only the frames linked from the body among t-weights,
    t-cohort, t-cohgaps, t-openingdef, t-og, t-windows, t-long, t-phperf, t-survival, t-hill, t-memorytab, t-phall.
13. Stop generating the unused `phases_joins_all_table.tex`.

Then: compile deck and note, render changed pages, `uv run pytest test_groups.py test_tail.py`.

## Also

- Note `group_story.tex`: still the pooled mean (`phases_duration_table`, `PhDurMean*`); use the per-flight mean
  and pooled median as the deck does (t-estimators), with the same two-estimator paragraph.
- Pills wrap to two lines on 19, 21, 24; on 21 the thread line touches "Details".
