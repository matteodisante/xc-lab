# Phase labelling

**Status:** method note, and a cadence check on 300 flights logged at 1 Hz. Started 2026-10-20,
updated 2026-10-20.

**Question.** How does Vilpellet's hidden Markov model label every fix as transition, search or
climb, and can it label tracks logged every 2 or 5 seconds?

**Answer.** Three yes/no features over 30 s (straight, rising, turning one way) feed a three-state
HMM decoded by Viterbi, then a 90-fix majority vote; we keep it as his code runs it, one
pre-processing piece at a time. Interpolating a piece onto a 1-s grid with a cubic spline gives
back the 1-Hz labels on 99.0% of the fixes at 2 s and 97.5% at 5 s; linear interpolation loses
climbs from 4 s on, and at 10 s neither works.

**xc-dynamics commit.** `d72e1b9` (the cadence run, with the take-off and landing rule).

## Files

- `phase_labelling.pdf`, built from `phase_labelling.tex` and `sections/`: the note, with hidden
  Markov models explained from zero, his method step by step, where his paper and his code
  disagree, what we keep and change, and the cadence check. Figure 7, the 30-fix window on a
  synthetic flight, is animated inside the PDF (LaTeX package `animate`): it plays in Acrobat
  Reader and other viewers that run PDF JavaScript; elsewhere it shows the static figure.
- `phases.py`: the method, rewritten from his inference script; same labels, bit for bit.
- `cadence.py`: thins 1-Hz flights and compares the labels (writes `out/cadence.csv`).
- `mechanism.py`: why linear interpolation loses climbs at 5 s (`out/mechanism.yaml`).
- `plots.py`: writes `figures/`. Without the disk it redraws only the figures that do not read
  flights (emissions, interpolation, window, cadence curves from `out/cadence.csv`).
- `test_note.py`: checks every number the note quotes.

## Rebuild

From the root of xc-lab, with `/Volumes/SSD_DATA` mounted:

```
uv run python methods/phase-labelling/cadence.py
uv run python methods/phase-labelling/mechanism.py
uv run python methods/phase-labelling/plots.py
uv run pytest methods/phase-labelling
```

then `latexmk -pdf phase_labelling.tex` in this folder.
