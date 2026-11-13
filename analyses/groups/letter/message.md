# Message of the letter after the redesign (2026-11-13)

Approved by the author: the centre is the jump within the same flight, absent in the mock events.
The same route is a control. The route is a plausible mechanism, argued in the Discussion.
Numbers below are written as `<<...>>`: replace each with the `\Lt...` macro of
`generated/macros.tex` that holds it (check its definition in `code/macros.py` first). Main
dataset = lags to 2760 s (`Two`, same flight: window 3450 s, `Max`); check = dataset 1000 (`One`,
same flight: window 2400 s, `Mid`).

## Title (chosen by the author on 2026-11-13)

The value of flying together

## Abstract (use as written)

Soaring pilots and birds travel across country by chaining invisible, short-lived thermals, and
they often fly in groups. Whether company helps is hard to measure in the field, since groups form
where and when flying is good. We identify groups in <<flights>> cross-country paraglider flights
recorded in France between <<first year>> and <<last year>>, with a rule calibrated against chance
encounters, and compare group and solo flight in three designs. In a group the horizontal motion is
more persistent, at time lags from tens of seconds to three quarters of an hour, and cross-country
speed is higher. The gap remains against solo pilots who flew the same route in the same season and
at the same hour, and about half of it is also found in the flights that join groups while they fly
alone. Within a single flight, persistence and speed rise when a pilot joins a group and fall back
when they leave, a jump that flights without a group do not show at the same times of flight.
Followers retrace the line of the pilot ahead, and lift stays tied to places for tens of minutes:
what company provides may be the route.

## End of the INTRODUCTION (replaces the paragraph "Here we measure it on human pilots...")

Here we measure it on human pilots. We find the pilots who fly together with a rule calibrated
against chance encounters, and compare group and solo flight three times: as they are; on the same
route, against solo pilots who flew it, with the flights that join groups taken while alone as a
control for selection; and within one flight, before, during and after a group, against mock
events placed at the same times of flight in flights without a group. The three designs agree: in
company pilots fly straighter and faster, and within a flight the gain comes and goes with the
group.

## What each results section must say (facts; the prose is written from these)

RAW COMPARISON. On the main dataset group time is more persistent than solo flight at every lag
from <<lag where the band of G - S clears zero, main>> s to the end; alpha(1000 s) differs by
<<raw G - S, main>>, the fitted H by <<...>>. On the shorter pieces of dataset 1000 the gap opens
only beyond <<clear lag, dataset 1000>> s (one sentence, no "one leg" argument). The flights that
join groups, while alone, have the alpha of solo flights (A - S) but are faster by <<A - S speed>>
km/h, against <<G - S speed>> for group time: a first sign of selection. Fig. msd shows the MSD
and alpha(tau) of S, A, G with the fitted H; Fig. designs(a) the differences against the lag.

SAME ROUTE. Matching as now (other pilots who share only the route; season, hour, wing class). The
gap remains: G - S|G at 1000 s and 1840 s, fitted H; it does not shrink with respect to the raw
gap on the main dataset, and is about half the raw one on dataset 1000. The flights that join
groups, alone, keep about half of it (A - S|A): selection. The difference, the effect of the group
net of selection, is not resolved for alpha or for the fitted H on either dataset: say that its
band includes zero and also values as large as the raw gap, so the design does not decide. It is
resolved for speed: <<route speed difference>> km/h. Sample sizes of both datasets in the text or
table.

SAME FLIGHT. Design as now (four windows, g, d_w, mock events with their calipers: month +-2, hour
+-2 h, same wing level). Net of the mock events alpha rises by <<net alpha(1000 s), Max>> at 1000 s
and <<net alpha(1840 s), Max>> at 1840 s, the fitted H by <<...>>, speed by <<...>> km/h, on
<<events>> events; with windows of 2400 s, <<events Mid>> events, the same at 1000 s
(<<net alpha(1000 s), Mid>>). The mock events are small but not always zero (give speed). The
phase statistics net of the mock events (`\PreNetG<Stat>`, window 2400 s): more time in climb, more
searches that end in a climb, faster and flatter glides; the climb rate is not resolved. Do not
write that the same-route band stays below the jump: the two designs are compatible.

HOW A GROUP HOLDS A ROUTE. Keep the content, with the corrections of the check (review_todo.md,
"Check of 2026-11-13", items 3, 4, 5, 9, 10, 12): the marking ratio is a share of climb starts
with a glider climbing nearby, against the same gliders shifted by 5 min before or after; 46 min
is at least how long a place keeps being used for climbs; the lead changes less often than in
chance contacts; the factor 1.42 is said for what it measures (a solo glider joins an existing
pair 1.42 times more often than expected, a group of three or more as often as expected) or left
out; terrain: the gain is largest in high mountains; long glides as now.

## DISCUSSION (use as written, fill the citations and macros)

In company, paraglider pilots fly straighter and faster. A raw comparison could not establish it,
since groups form on good days and good routes and gather pilots who fly well. Two results carry
the claim. Against solo pilots on the same route the gap remains. And within one flight
persistence and speed rise at the entry into a group and return at the exit, while flights
without a group show no such jump at the same times of flight. How much of the gap belongs to the
group is less certain: on the same route the flights that join groups keep about half of it when
they fly alone, and the remainder is resolved for speed and not for persistence.

What does company provide? Our data favour the route over the instant. Lift stays tied to places
for tens of minutes, and a glider climbing now adds little to what the place gives. Followers
retrace the line of the pilot ahead and do not answer its turns. Long glides that cost little
height are found by solo pilots on the same route as well. Faster glides are what speed-to-fly
theory prescribes when the next climb is surer or stronger~\cite{MacCready1954,Cochrane1999}, and
what vultures do when others reveal the next thermal~\cite{Williams2018}. We cannot tell a group
that leads its members to a good line from a group that gathers on it. Our groups are defined by
flight alone: pilots who arranged to fly together, who talk by radio or who follow the known
routes of a site are among them, and the archive does not tell them apart.

## CONCLUDING REMARKS (use as written)

The same-flight design needs only tracks and a rule for who flies with whom. Each flight is its
own control, and mock events remove what the course of a flight produces by itself. It applies as
it is to soaring birds tagged in flocks, where the value of company has been read from leaders
and followers~\cite{Flack2018} or from single thermals~\cite{Williams2018,Sassi2024}. For pilots
it puts a number on a question asked at every takeoff: within a flight, company is worth about
<<net speed, same flight, main>> km/h of cross-country speed.

## Style

Short declarative sentences; say what was measured; no em-dash asides; no "X is A, not B"
sentence shape; "we chose" for choices; every number through a macro; bands in tables and
captions, one or two numbers per paragraph in the text; cite the supplement by section
("Supplemental Material, Sec. S2"), never generically.
