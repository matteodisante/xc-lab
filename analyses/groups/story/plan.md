# Plan of the story: group time and solo time

## Principles

- One question at a time. Every analysis exists because a question needs it; a chapter states why the question
  matters, which tool answers it and why that tool.
- Each chapter: **Q** · **Why** · **Tool** (and why this tool) · **Result** · **Observation** · **Deduction** ·
  **Take home**. Method boxes for the tools, kept apart from the observations.
- Few metrics. Persistence is read on one quantity, $\Delta\alpha^*$; the phases on a fixed list.
- At the end: what we found, what the data say and do not say, which tools are still missing.

## The object and the identity

Three populations of flight time, paragliders, declared open flights:

- **S**: solo flights (no group time);
- **A**: the solo time of the flights that have group time;
- **G**: their group time.

For any statistic $X$ (an exponent, a phase statistic, a tail exponent), with the same estimator for the three,

$$ X_G - X_S = (X_G - X_A) + (X_A - X_S). $$

- **G − A**: two moments of the *same flights* (same pilot, wing, day, site, flight shape). The group effect
  proper, up to the moment of the flight (Ch. 6) and the direction (Ch. 9).
- **A − S**: *different flights*, both alone: which pilots, days, places and flight types have group time
  (selection), plus any carry-over of the group.
- **G − S**: the raw total.

The identity replaces the three designs of the comparisons report: same flight is G − A (refined by the
timeline). A − S differs by who flies, where, when and which flight; each control removes one source and acts on
A − S only, since G − A holds them fixed: same pilot (A against the solo flights of the same pilots), same route
(A against solo flights along the same route), ground class (a coarse "where"). The old same-route design
measured G − S_route = (G − A) + (A − S_route). Same route also controls the place within the flight in G − A:
group time and solo time of one flight fall on different parts of its path, so G can be compared with the solo
time of other flights on the same stretches.

## Data

- Every declared open paraglider flight with a logger interval **up to 5 s** (those at 10 s, about 2.4%, are
  dropped), so that every flight has phases. All chapters on this one set.
- Group time: the run `d120_eh500_ez300_th45_g120_q0.1` as it is. It is built on every flight (a partner logged
  at 10 s still makes a flight's time group time); we only analyse the tracks of the flights up to 5 s.
- Ground class and opening per flight as now.

## Statistics

**Persistence.** The chained exponent $\alpha^*(\tau)$ and its gaps $\Delta\alpha^*(\tau)$, two-way bands. In
Ch. 3 only, to compare the ground classes whose $\alpha^*$ differs, the gap normalised by $1-\alpha^*_b$, stated
as our choice. No displacement ratio: above 1 000 s it is fixed by $\Delta\alpha^*$ and adds nothing.

**Phases** (labels of our refit, Sec. "Phase labels"). Unit = the flight, "average first, ratio after":

- glide (transition): share of time; glide speed; height lost per km;
- search: share of time; share of searches that end in a climb;
- climb: climbs per hour; climb rate; share of time; vertical gain per climb; horizontal distance per climb
  (its chord) and the drift speed, chord / duration, a proxy of the wind at the height of the thermals (are group
  flights in windier air?).

Glide speed: speed-to-fly (MacCready) grows with the climb expected next, so a group that makes the next climb
more certain may fly faster with the bar. But faster in the same air loses more height per km (the polar); group
time is faster *and* loses less: better air in glide (shared lines of lift, the pilot ahead showing where the air
rises) or tailwind. The drift speed tests the wind part.

**Durations** of glides, searches that end in a climb, climbs: survival $P(X>x)$ and the tail exponent $\mu$
(Hill). Threshold: per phase, **one threshold for S, A and G, the largest of their three Kolmogorov–Smirnov
thresholds** (Clauset, Shalizi, Newman 2009), re-chosen in every bootstrap draw so that the band includes it.
Text to state:

> We read $\mu$ of S, A and G above one threshold per phase, the largest of their Kolmogorov–Smirnov thresholds,
> re-chosen in every bootstrap draw so that the band includes it: above it every curve follows its power law,
> and the three exponents describe the same durations. Separate thresholds would give each tail its own best
> estimate, over different ranges of durations, and their difference would mix the range with the population.

The per-curve thresholds and the Hill plots $\mu(x_\mathrm{min})$ go to an appendix as the check. One threshold
for the three populations also keeps the identity exact for $\mu$.

**Geometry.** The opening $\mathcal O$ of the flight; the route (each climb replaced by its chord, arc length)
and its exponent $\alpha_\mathrm{route}$, as in the ridges work.

**Moderators** everywhere: the ground class (plains, hills, low and high mountains) and the lag $\tau$.

## Phase labels

We use our own labels; Vilpellet's model is the comparison.

- **Scope**: every flight with a logger interval up to 5 s, paragliders and hang gliders, the whole catalogue
  (not only the declared open flights).
- **Refit** of his three-state HMM (same features, same decoding): Baum–Welch to convergence with the transition
  matrix estimated (his training never updates it), several random starts, the likelihood on held-out flights;
  training flights from every ground class (his parameters come from 2019 flights of four Alpine take-offs).
- **Training on 1-Hz flights only**: a spline smooths the track (small turns, short changes of height), so a
  model trained on splined tracks would learn what an interpolated flight looks like. Flights at 2–5 s are
  splined to 1 s and labelled by the 1-Hz model.
- **One model per discipline**: hang gliders fly faster, on wider circles, with a better glide; Vilpellet reuses
  the paraglider emissions for them. If the hang gliders cannot be refitted (too few at 1 Hz, or a fit that does
  not converge or loses the meaning of the states), they are labelled with the paraglider refit, changing only
  the discipline's straightness threshold as Vilpellet does. Check: the paraglider model applied to the hang
  gliders.
- **Storage, in the folder the viewer reads**: runs (flight, start, end, state), not one label per second,
  below 300 MB in all, in `xc-viewer-data/<discipline>/derived/segmentation/refit/` next to
  `segmentation/vilpellet/`, same file layout (`phase_segments.parquet`, `phase_coverage.parquet`) and the
  conventions of the folder (manifest, inputs identified by place and content), with a PARAMETERS.md (training
  flights, iterations, starts, commit). Only a new subfolder is written; nothing existing is touched. xc-lab
  reads the labels from there: one copy. The model parameters also in `methods/phase-labelling/out/` and, for
  the viewer, in `configs/segmentation_refit.yaml`.
- **Viewer**: the refit as a second segmentation next to Vilpellet's. On a single track, both shown together
  (two phase ribbons along the track, or the two colourings side by side, with the fixes where they differ
  marked) and a switch for which one the other tabs use; the share of fixes that agree, per phase, in the
  panel. The single-track decode runs the same code as the batch labelling (one implementation, imported or
  ported bit for bit and tested on the saved runs).
- **Meaning of the states**: climb is the state that rises while turning, transition the straight one, search
  the rest; checked on the emissions and on the climb rate.
- **Cadence**: tracks up to 5 s splined to 1 s, as in methods/phase-labelling (with his labels: 99.0% of the
  1-Hz labels back at 2 s, 97.5% at 5 s); the check repeated with our model, per discipline: 1-Hz flights
  thinned to 2–5 s, splined back, labels compared.
- **Against his labels**: agreement per state and per ground class; the statistics of Ch. 7 and 8 recomputed
  with his labels. Appendix "Phase labels".
- **Grouping**: the phases enter it only through the heading condition, checked when both pilots glide and
  both flights are 1 Hz. The run stays as it is; on a sample of days the contacts are recomputed with our
  labels and the change of group time is reported.

## Chapters

**0. Measures.** Definitions (pieces, populations, TA-MSD, support, the window of $\alpha$); bands; the fixed
cohort and the problem of the pool; one curve per population: the chained MSD and $\alpha^*$ (the proof on the
toy); ground class; opening; phases and their statistics; tails. *Exists; to reorder.*

**1. Is group time straighter?**
Q: does a pilot's motion persist more while flying in a group than when flying alone?
Why: the starting fact; without it there is no story.
Tool: $\alpha^*_G$, $\alpha^*_S$ and $\Delta\alpha^*_{G-S}(\tau)$. *Exists (all flights); rerun on ≤ 5 s.*

**2. The group or the flights?**
Q: is the gap a property of the moment in a group or of the flights that have group time?
Why: G − S mixes the effect of flying with others and the selection of pilots, days and places.
Tool: the identity; $G-A$ against $A-S$ at every lag. *Exists; rerun.*

**3. Where?**
Q: does the effect depend on the ground under the flight?
Why: groups fly more in the mountains (72% of G, 58% of S) and the ground changes $\alpha^*$ itself: a moderator
and a confounder.
Tool: ground class (DEM 200 m, median under the 10-s points without the first and last 10% of the duration);
the gaps per class, absolute and normalised; the lag of the peak. Simpson's paradox on $A-S$. From here every
comparison at fixed class. *Exists; rerun.* Phases (after the refit): only climbs per hour, i.e. the length of a
glide + climb cycle per class, against the lag of the peak (hypothesis: the effect peaks at the scale of one
cycle, shorter in the mountains; caution: in the plains the peak lies near the end of the solid range).

**4. Which flights?**
Q: does the effect depend on the shape of the flight?
Why: a threshold on the opening shrinks G − S: is the effect a matter of path shape?
Tool: the gaps per class × threshold on $\mathcal O$; $\mathcal O$ of S, A, G per class. G − A holds the flight
shape fixed; A − S does not. Caution: a threshold on $\mathcal O$ selects on the outcome. *Exists; rerun.*

**5. Who?**
Q: does A − S come from the pilots?
Why: A − S compares different flights; the pilots who fly in groups may be other pilots.
Tool: A − S′, with S′ the solo flights of the pilots who also have group flights, per class. *To do.*

**6. When in the flight?**
Q: is group time straighter than the solo time just before and just after it?
Why: G − A compares group time with any solo time of the same flight; the moment of the flight may matter.
Tool: windows of fixed length before, at the start, at the end and after each group stretch (prepost.py), per
class; equal-length windows need no chain. *Exists for H; redo with the class and ≤ 5 s.* Phases: only the
shares of time in glide, search and climb per window, as a control of composition (if groups form in climbs, the
"start" window holds a climb by construction); the same shares answer Ch. 9 (does the group form in a climb?).
Last step: the carry-over of the group on the solo time around it, A(d) − S by the distance from the group
(carry.py), moved here from Ch. 2 since it uses the same windows.

**7. How: the dynamics.**
Q: which phases does the group change?
Why: $\alpha^*$ in time mixes the shape of the path and the way it is flown; the phases tell the second.
Tool: the phase statistics and the tails, $G-A$ and $A-S$, per class, in two steps. The phase composition is
to $G-A$ what the ground is to $A-S$: the grouping catches pilots in the same thermal, so group time may hold
more climbs.
1. *Composition*: is $G-A$ a composition effect? Shares of time in glide, search, climb; climbs per hour.
2. *Performance within the phase*: glide speed, height lost per km, share of searches ending in a climb, climb
   rate, horizontal distance gained in climb; durations and $\mu$.
Main figures: the non-redundant statistics (`redundancy.py`); the full table in an appendix. The hypothesis of
the solo time (climbs go to G) is the composition step, with no analysis of its own. Not used in Ch. 4 (the
opening) nor in Ch. 5 (who) unless $A-S'$ survives, and then at fixed day (climb rate and glide speed depend on the
day more than on the pilot). *To do (after the refit).*

**8. How: the geometry.**
Q: is the path of the group straighter, or the same path flown more regularly?
Why: separates shape from timing.
Tool: $\alpha_\mathrm{route}$ of S, A, G per class; $\Delta\alpha^*_\mathrm{time} - \Delta\alpha_\mathrm{route}$ is the
share of the dynamics. *To do (after the refit).*

**9. Which way round?**
Q: does the group straighten the stretch, or does a straight stretch make the group?
Tool: the timeline of Ch. 6 (is "before" already straight?), and what the data cannot say. *Partly exists.*

**10. What we found.** What the data say, what they do not, the tools still missing (direction; geometry against
dynamics beyond the route, as in the ridges work; weather).

**Appendices.** Per-lag cohort; phase labels (model, refit, cadence, agreement, grouping check); tail thresholds;
one weight per piece or per flight.

## Modules and who does them

Opus (main session): design, specifications, review of every result, the text. Sonnet agents: code and runs,
one module each, reusing what exists.

| # | Module | Reuses | Feeds |
|---|---|---|---|
| M1 | `contrasts.py`: S, A, G under any stratum (class, opening, pilot set, ≤ 5 s), $\alpha^*$, gaps, normalised gaps, bands, macros | `cohorts.chained` (folds the former `ground_gaps.py`, `ground_opening.py`) | Ch. 1–5 |
| M2 | HMM refit and labels of every ≤ 5 s flight; agreement with Vilpellet per class | methods/phase-labelling `phases.py`, `cadence.py` | Ch. 7–8 |
| M3 | Phase statistics and tails per population × class, both label sets | `raw.py`, `budget`, `tail.py`, `tail_designs.py` | Ch. 7 |
| M4 | Same pilot | M1 | Ch. 5 |
| M5 | Timeline per class | `prepost.py` | Ch. 6, 9 |
| M6 | Route exponent per population | `ridges/decompose.py` | Ch. 8 |
| M7 | Grouping with our labels, sample of days | group-identification `contacts.py` | Appendix |
| M8 | Viewer: the refit next to Vilpellet's on a track, agreement, switch (repo xc-viewer) | `data.load_vilpellet_phases`, `core/vilpellet` | Checks by eye |

Order: (1) skeleton of group_story.tex (chapters with Q, Why, Tool, "To do"; existing material moved); (2) M1 and the
rerun of Ch. 1–4 on ≤ 5 s; (3) M4, M5 → Ch. 5, 6; (4) M2, then M8 (look at the labels before using them) and
M7; (5) M3, M6 → Ch. 7, 8; (6) Ch. 9, 10.

## Risks

- The ≤ 5 s set changes every number already in the text: all figures and macros are rerun.
- The level of MSD$^*$ beyond 1 000 s assumes that the short flights would grow as the long ones: only its slope
  is read.
- A threshold on the opening selects on the outcome; few mountain flights remain at $\mathcal O>0.5$.
- An unsupervised refit has no true labels: the criteria above decide, and Vilpellet's labels stay as the
  comparison.

## Outside the story

- `tamsd.summarize` and `longlags.curve` still pool the origins per flight (one weight per flight). No figure or
  macro of group_story.tex uses them (its curves come from `cohorts.py`, one weight per piece), but `raw`, `pools`,
  `size`, `episodes`, `curvature` do.
