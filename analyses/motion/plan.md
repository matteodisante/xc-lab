# Plan: the motion of a mountain flight, topology and dynamics

Status: plan, 2026-11-27, after an independent review. Nothing moved yet. This topic replaces `analyses/ridges/` and
`analyses/phases/`. Keep it simple: a check only where a result depends on it.

## Questions

Inside the flight:

1. How much of the motion of a mountain flight comes from the shape of its path (topology), and how much from the
   way the path is flown in time (dynamics)?
2. Is the motion self-similar: one exponent $H$ for every scale of the increments and every moment?
3. If not, does the topology or the clock break it?
4. Which laws do the durations of glides, searches, climbs and breaks follow?

The flights against the crests:

5. Do the routes have the same geometry as the crests (bend, direction memory, $D$), and how much of the bend do the
   crests explain?
6. Do the flights follow the crests: how close, for how long, with a pull towards them?
7. A ball on the crests with the clock of the real flights: is it the observed motion?

A model:

8. Is the motion an intermittent correlated random walk: the direction memory $C(u)$ of the paths, flown with an
   intermittent clock?
9. Where does $H$ go at lags longer than the flights: is the observed $H$ a crossover or already the limit?

| question | step |
|---|---|
| 1, 2, 3 | Decomposition |
| 4 | Durations |
| 5 | Crests; Flights against crests: geometry, fractal dimension |
| 6 | Flights against crests: pairing |
| 7 | Flights against crests: clock on the crests |
| 8, 9 | Model |

## Data

- Paragliders only; declared open flights, logger interval up to 5 s; all mountain flights together.
- Unit: the S, A, G pieces of the flights, as in `groups` (piece boundaries of `longlags.py`), group run
  `d120_eh500_ez300_th45_g120_q0.1`. Every result apart for S, A, G. Why: group time has a different shape.
- A piece is kept when at least 90% of its route lies inside the masks of the Alps or the Pyrenees.
- Positions: the tracks of the refit labels, on their 1 s time base (5 s loggers splined to 1 s). Why: the cache of
  `longlags.py` holds sums on a 10 s grid, too coarse for the objects below; new sums are built.
- Phase labels: our refit (`xc-viewer-data/paragliders/derived/segmentation/refit/`).
- Routes and crest paths resampled every 10 m of arc length. Why: the arc length depends on the resolution; one step
  for both makes $H(s)$ and $D$ comparable.

## Conventions

- MSD: the chained MSD* of `groups/cohorts.chained`. Why: each step $\tau_k \to \tau_{k+1}$ is taken on the same
  pieces (no composition term), and every piece is used up to its length.
- Main result: the local exponent $\alpha^*(\tau)$, a straight line at 1. $H$: one fit of MSD* $\propto \tau^{2H}$ on
  a window common to every object and to S, A, G, from 60 s to the largest lag where all keep enough flights. Why
  60 s: past one turn of a spiral. Why one window: $\alpha^*$ is not flat, so $H$ depends on the window.
- Bands: 90%, the pilot × block draws of `groups/twoway.py`. Comparisons with the crests also resample the crest
  paths.
- Compare exponents, not MSD levels. Why: the level depends on when in the flight the window starts (increments are
  not stationary).
- The error of a model as a share of the observed exponent, lag by lag: $(\alpha^*_\mathrm{model} -
  \alpha^*_\mathrm{flight}) / \alpha^*_\mathrm{flight}$. No pass/fail threshold.
- Matching the exponent does not mean matching the motion: the models drop the motion inside the breaks, so they
  cannot give the MSD conditioned on the phase. Stated in the note.

## Glide and break

A **break** is everything between two glides: searches and climbs in any order. A break cut by a gap or the end of
a segment stops there.

The break is flown along its **chord** (start of its first run to end of its last) at constant speed. Why: time and
net displacement stay exact, spirals and drift inside are dropped, and the motion stays continuous. A stasis + jump
would add an MSD growing as $\tau$ at lags shorter than a break, an artefact right in the fit window.

First measure (logbook 2026-11-26, 2961 flights in approximate boxes): net speed 8.5 m/s in glide, 2.2 m/s in break;
at equal duration the glide goes 2.0 times farther at 30–60 s, 7.6 times at 10–20 min. Breaks hold 17% of the chord
length; their drift is coherent (0.46) and in open flights points with the flight (12% of the net displacement).

## Decomposition (first step, Q1 to Q3)

The **route** $R(s)$: glides and break chords in turn, against their arc length $s$. $L$ the route length, $T$ the
duration of the piece, $S(t)$ the arc length reached at time $t$.

| name | object | holds |
|---|---|---|
| route | $R(vt)$, $v = L/T$ | topology |
| route + clock | $R(S(t))$: glides at their real speed, breaks along their chord | topology + dynamics |
| clock | $S(t)$ on a straight line | dynamics |
| flight | the real trajectory | + the motion inside the breaks |
| route + other clock | $R(S'(t))$, $S'$ the clock of another piece of the same kind (S, A, G) | interaction check |

$v = L/T$ gives route and route + clock the same duration, ends and mean speed. In arc length they are the same
curve: their difference in time comes only from the clock.

Reading, on the $\alpha^*(\tau)$ curves:

- route: how far the topology bends the path from a straight line;
- route + clock against route: the effect of the clock. Composing the two multiplies the exponents
  ($\alpha^*_\mathrm{route+clock} \approx \alpha^*_\mathrm{route} \times h$, $h$ the scaling of the arc length
  flown), so the effect is read as a ratio, not a difference;
- clock: $h$ directly. It tends to 1 at long lags (the clock only moves forward), so the clock acts at intermediate
  lags;
- route + other clock against route + clock: if equal, route and clock are independent;
- flight against route + clock: the error of the chords, as a share.

**Beyond $H$.** The same objects, on the increments $|\Delta r|$ at 60 s, 5 min, 20 min, 1 h:

- distributions rescaled by $\tau^H$: they collapse if the motion is self-similar;
- structure functions $\langle|\Delta r|^q\rangle \propto \tau^{\zeta(q)}$, $q$ from 0.5 to 4: $\zeta(q) = qH$ for a
  monofractal.

Reading of $\zeta(q)$ (Q2), and on which object it departs from $qH$ (Q3):

- linear, $\zeta(q) = qH$: monofractal, $H$ describes the whole motion;
- bilinear, $\zeta(q) = q\nu$ below $q^*$ and $q - c$ above: strong anomalous diffusion, the mark of a Lévy walk
  (Castiglione et al. 1999, Physica D 134, 75; Rebenshtok et al. 2014, PRL 112, 110601). High moments are set by the
  long glides at near constant speed; to read with the duration laws (Q4);
- concave and smooth: a cascade-like multifractal.

If not linear, the $H$ of the MSD is only $\zeta(2)/2$, one moment of the motion: stated in the note. The model (Q8)
must give $\zeta(q)$, not only $H$.

Increments of each piece divided by its $v = L/T$ before pooling. Why: pooling slow and fast pieces makes fat tails
and a false multifractal; one scale per piece, independent of $\tau$, keeps the test honest. The pieces covering
1 h, with their number per population (G dropped here if too few). $q \le 4$. Why: high moments rest on a few events.

## Durations (second step, Q4)

Laws of the durations of glides, searches, climbs and breaks (search + climb), apart for S, A, G. Searches as in
`groups/phase_terms.py`: found (glide-search-climb) and refound (climb-search-climb) are found searches; failed
(glide-search-glide) ones are censored; exits (climb-search-glide) apart.

- Five families: exponential, gamma, Weibull, lognormal, Lomax (power-law tail). Parameters by maximum likelihood.
- A phase cut by a gap is censored. The last phase of a flight ends at landing: not censored.
- A phase whose group state changes goes whole to the state where it starts. Why: simple; censoring at the change
  would be biased if groups form or break up in a given phase.
- Fit above $x_0 = 15$ s (three intervals of the coarsest logger), conditional on $x > x_0$; repeated at 30 s. Why:
  the shortest phases are mostly flicker of the labelling, in whole seconds, and would steer the fit.
- Choice by AIC. Tail score: $\ell_\mathrm{tail} = \sum_{x_i > x_t} \log[f(x_i)/S(x_t)]$ with the parameters of the
  whole fit, $x_t$ the Kaplan–Meier 90th percentile; censored phases enter with $S(x_i)/S(x_t)$. AIC chooses the law
  that describes the durations, $\ell_\mathrm{tail}$ the law of a simulated clock; a disagreement is stated. Why: the
  long breaks weigh on $H$.
- Mean and median, S, A, G: median from Kaplan–Meier, mean restricted to one horizon common to S, A, G.
- One figure per phase: Kaplan–Meier survival with the fits overlaid.

## Crests (third step)

**Method.** Why from the DEM: no complete vector map of the crests exists (OSM and BD TOPO are partial, global
products are 90 m rasters); the existing maps serve as the check. Wanted: the crests between summits, long, not cut
in pieces by small changes; side spurs left out.

The **divide network** of Scherler & Schwanghart (2020, ESurf 8, 245):

1. Streams: D8 flow, cells draining more than a threshold area. Divides: the cell sides between cells draining to
   different stream links. Already in `ridges.py`.
2. Network: endpoints and junctions, broken segments merged, by the node rules of their Table A1. Replaces our
   straight gap joins.
3. Order: the divide distance $d_d$, the longest distance from an endpoint along the sorted network; it grows
   linearly with the distance down both sides to a common stream point, so it measures how main a crest is.
4. Selection (ours): whole segments kept with $d_d$ (at their end) above a threshold and median hillslope relief HR
   above a minimum. Why: main crests have high $d_d$, side spurs low; flat divides have low HR. Never single cells,
   so no gaps.

Three constants: stream threshold, $d_d$, HR. Steps 2–3 written from the paper (TopoToolbox has `DIVIDEobj` only in
MATLAB). Computed on the whole DEM of each range with a margin. Why: $d_d$ is not local; a cut basin makes a false
divide and a short $d_d$. Segments whose $d_d$ starts at the edge are flagged.

Differences from the paper: depressions filled, not carved; RGE ALTI, not SRTM 30 m; the selection of step 4; spline
smoothing; the strokes below. Methods left aside: topological persistence (heavy tool); geomorphons (a raster,
thinning breaks long crests).

**Test of the method**, on a few tiles against the current `ridges`: precision and recall against the OSM lines
`natural=ridge` and `natural=arete` within a distance; pieces per OSM crest; each threshold at ×0.5 and ×2.

**Crest paths.** At each junction the path goes on along the branch that turns least, up to a maximum deflection:
the "good continuation" strokes of Thomson & Richardson (1999). Why: a long crest stays one path, and we study open
flights. Each path kept whole. The maximum deflection is fixed before any comparison with the flights, as the one
that keeps the OSM crests in one piece. Why: tuning it on the route lengths would also set $H_\mathrm{crest}$.

- Lengths of the crest paths against the routes; opening (net displacement over length) of both.

**$D$ of the crests.** Box counting of single paths (`boxdim.py`), bands by bootstrap; against the basin perimeters
of Breyer & Snow (1992, Geomorphology 5, 143): $D \approx 1.06$–$1.12$ (from a secondary citation, to check on the
paper). They used the divider method, so the comparison is approximate; stated.

## Flights against crests (fourth step)

**Geometry (Q5).** Route against crest paths in arc length, $100\ \mathrm{m} \le s \le s_\max$, $s_\max$ from the
path lengths: $\alpha^*(s)$, $\langle\cos\Delta\theta(s)\rangle$. Why 100 m: the cut-off of the crest spline. The
crest path is compared with the route: both hold topology only. The second half of Q5 as a number: the share of the bend explained,
$(1-\alpha^*_\mathrm{crest}(s))/(1-\alpha^*_\mathrm{route}(s))$.

**Fractal dimension.** $D$ by box counting on the routes, against $D$ of the crests. The relation with $H$
depends on the curve:

| curve | relation | exponent |
|---|---|---|
| self-similar | $D = 1/H$ | of the displacement along the arc length |
| self-affine (a main direction: the start–end chord) | $D = 2 - H_\perp$ at small scales | of the displacement across the chord against the distance along it |

Both give $D \to 1$ at large scales: only the small scales tell them apart.

**Pairing (Q6).** Each point of the route projected on its nearest selected crest: $d(s)$, the signed horizontal distance.
Distribution of $d$; length held on one crest before a switch. Pull towards the crest: $E[\Delta d \mid d]$ against
$d$, real against the same route moved and turned at random inside the mask. Why: $d$ is bounded, so only the
difference from the null shows a pull.

**Clock on the crests (Q7).** Crest route + clock: the clock $S(t)$ of a piece applied to a crest path of the same
range at least as long as its route, 20 draws of (path, start, direction) per piece. Against route + clock: same
clock, only the topology changes; against the flight: the whole answer. Pieces with no path long enough: share
reported by $\tau$, results repeated without them. The crests have no wind: the drift of the breaks is a declared
missing ingredient.

## Model: an intermittent correlated random walk (last step, Q8 and Q9, only if Q7 holds)

The simplified motion, written as a model:

- Geometry: a correlated random walk in arc length (Kareiva & Shigesada 1983; the worm-like chain of polymer
  physics), steps of 10 m whose direction has the memory $C(u) = \langle\cos\Delta\theta(u)\rangle$ of the real
  paths. A glide is many such steps, not one segment: the route follows its real shape.
- Clock: intermittent (Bénichou et al. 2011, Rev. Mod. Phys. 83, 81): fast phases (glides at their speed) and slow
  phases (breaks flown along their chord at constant speed, so the small net advance of a break is kept). It enters
  by subordination: $s \to S(t)$.
- What the model ignores: the motion inside the breaks (spirals, drift off the chord). Zoomed out, a break is one
  slow straight stretch.
- Name: a Lévy walk only if the duration laws come out power-law; otherwise "intermittent correlated random walk".

Can the model say where $H$ goes as $\tau \to \infty$, beyond the length of real flights?

- The geometry follows from $C(u)$ alone: $\mathrm{MSD}(s) = 2\int_0^s (s-u)\,C(u)\,du$. Exponential decay of $C$
  gives $H_\infty = 1/2$; $C \sim u^{-\beta}$ with $\beta < 1$ gives $H_\infty = 1 - \beta/2$; $C \to$ a positive
  constant gives $H_\infty = 1$. So the work is the form of the tail of $C(u)$, fitted on the data.
- $C(u)$ of the routes (already in the decomposition; open flights keep a direction by selection, so stated as a
  bias) and of the crest paths (not selected, longer): the two predictions side by side.
- The clock: with a finite mean break the clock goes on linearly and leaves $H_\infty$ unchanged; with an infinite
  mean (power-law tail exponent below 1) $H_\infty$ is lower.
- Simulation only as the check: synthetic paths with the fitted $C(u)$, the simulated clock on them, $\alpha^*(\tau)$
  far beyond the flights.
- Stated limit: beyond the largest real scale the answer follows from the fitted form of $C(u)$, not from data (the
  Alps are finite and bend as an arc). It tells whether the observed $H$ is a crossover or already the limit.

## Later, only if needed

- No chords (weight of the drift of the breaks).
- A simulated clock (on the routes and on the crests): the laws of "Durations"; chord length given the duration;
  glide speed; coupling of successive durations (first check, logbook 2026-11-26: weak; only climb → next glide,
  ρ ≈ 0.12–0.17).

## What exists and where it goes

| now | what | goes to |
|---|---|---|
| `ridges/` geo, mountains, ridges, curves, joined, tiled, hgeo, tamsd, boxdim, dem_resolution, compare_dem | crest pipeline, $H_\mathrm{geo}$, $D$ | crests |
| `ridges/` flights, run_flights, `crests/` sections flights_route, dimension | flights in time and along the route | decomposition, Q5 |
| `ridges/` pauses, glide_speed | glides and pauses, laws | durations |
| `ridges/` model, decompose, `intermittency/` | intermittent ball, null models | Q7 |
| `phases/` durations, msd, glide | Vilpellet's statistics reproduced | durations, as the comparison |
| `groups/experienced/` families, lomax | laws of durations | `common/` |
| `groups/` route_arc, route_alpha, legs | route and direction memory by S, A, G | decomposition, Q5 (copied) |

## Document

One LaTeX note, `motion/story/motion_story.tex`, in the line of `groups/story`, shorter: plots, tables, bullets, no prose.

- Opening: the nine questions; the definitions once (break, route, route + clock, clock, flight, MSD*, $H$,
  $\alpha^*$).
- One section per step, always in five blocks: **Question**, **Method and tools**, **Results** (one figure, one
  table if needed), **Observations**, **Take home**. The macros `\Q`, `\Tool`, `\Res`, `\Obs`, `\Home` of
  `groups/story`.
- Every choice with its reason, in one short line (a `\Why` bullet).
- Short, plain sentences: one idea each.
- Numbers only from generated tables and macros (`generated/`), never typed.
- Closing: one bullet per question with its answer. Details and checks in appendices.
- An appendix on the crest method: each step with its reference (Scherler & Schwanghart 2020; O'Callaghan & Mark
  1984 for D8; Thomson & Richardson 1999 for the strokes), and a table of what we take from the paper as it is and
  what we change or add, each with its reason. A short section in the body points to it.
- Each step adds its section as soon as its results exist.

## Order

1. Decomposition (route, route + clock, clock, flight, route + other clock), with "Beyond $H$". First section.
2. Durations.
3. Crests: method, test on tiles, paths, $D$.
4. Flights against crests: geometry, fractal dimension, pairing, clock on the crests.
5. Move what `groups` imports from `phases/durations.py` (`ccdf`, `fit`, `hill`, `log_resample`, `trim`) to `common/`;
   then `git mv` of `ridges` and `phases`. The caches on disk keep their folder names.
6. `families.py`, `lomax.py` and the laws of `pauses.py` into one module of `common/`.
7. Model (intermittent correlated random walk) and asymptotic motion, if the crest + clock model reproduces the
   flights.
8. "Later, only if needed", as the results ask.

## Open decisions

- Layout: code by part and one document in parts, or flat code and one document per part.
- Ch. 7 and 8 of the group story: stay in `groups`, or move here and the story cites them.
