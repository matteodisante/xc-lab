# Group identification

**Status:** method note, not run on flight data. Started 2026-10-14, updated 2026-10-20.

**Question.** How do we tell, at every instant of every flight, whether a pilot flies in a
group, and with whom?

**Answer.** Two pilots are in contact when one of them is on the path of the other within
δ = 5 min, 500 m horizontally and 300 m vertically, with headings compared only in glide;
altitude keeps the delay well defined in thermals. At each instant the groups are the connected
components of the contact graph: a configuration is a fixed set of flights over a maximal
interval, an episode a chain of configurations that share members. Method note only: nothing
has run on flight data yet.

**xc-dynamics commit.** The note reads its input as the pre-processing of `d72e1b9` leaves it
(segments through step 8, the frame of the thesis); figures and tests use synthetic tracks only.

## Files

- `group_identification.pdf`, built from `group_identification.tex` and `sections/`: the note,
  English version with figures of the Italian note of 14 October 2026. Figures 7, 9 and 18 are
  animated inside the PDF (LaTeX package `animate`): they play in Acrobat Reader and other
  viewers that run PDF JavaScript; elsewhere they show the static figure.
- `models.py`: the synthetic scenarios of the figures and the sweep that builds configurations,
  episodes and blocks.
- `plots.py`: writes `figures/` (static figures, and `*_frames.pdf` with `*_frames.tln` for the
  animations in the PDF) and `animations/`.
- `test_note.py`: checks every number the note quotes.

## Animations

The contact of a pair: the set of correspondences fills the (t, t′) plane above the diagonal,
because j passes where i was two minutes later.

![contact](animations/contact.gif)

Altitude as a clock: in a thermal every turn gives an equal minimum of the horizontal distance,
while the altitude selects one window of delays.

![altitude clock](animations/altitude-clock.gif)

An example day: the contact graph, the configurations c1–c4 and the episodes E1, E2, event by
event.

![example day](animations/example-day.gif)

## Rebuild

```
uv run python plots.py                       # from this folder; needs LaTeX and ImageMagick
latexmk -pdf group_identification.tex
uv run pytest ideas/group-identification   # from the root of xc-lab
```

`uv run python plots.py --static` skips the GIFs.
