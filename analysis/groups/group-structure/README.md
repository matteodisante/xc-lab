# Group structure

**Status:** working note, from the section "What we find" of
`ideas/group-identification` (note of the method), 2026-10-29. The analyses that give its numbers
and figures are in [`../group-identification`](../group-identification/).

**Question.** Once the groups are labelled, what are they: how large and lasting, what holds
their pilots together, who leads, with whom pilots fly, and where the sizes come from?

The note studies the groups in themselves: not the method that builds them (that is
`ideas/group-identification`), not the difference between group and solo flight (that is
`../group-identification/report`).

## Files

- `group_structure.pdf`, built from `group_structure.tex` and `sections/`: the note. It reads the
  labels of the note `ideas/group-identification` as `gi-<label>` (package `xr-hyper`), which exist once
  that note has been compiled; the note reads this one as `gs-<label>`.
- `sections/structure.tex`: the study; appendices `graphs.tex` (graph terms of shape and order),
  `models.tex` (Ornstein–Uhlenbeck and birth–death chain), `evidence.tex` (phase, hour, habitual
  leaders).
- `figures/structure/`: copies of the figures of the analyses. `assessment_structure.pdf` is read
  from `ideas/group-identification/figures/`.

The numbers are macros written by `../group-identification`
(`report/generated/macros.tex` and `structure_macros.tex`).

```bash
cp ../group-identification/out/d120_eh500_ez300_th45_g120_q0.1/{mechanisms_phases,mechanisms_hours,cohesion,chains,hierarchy,mechanisms_habitual,partners,fission}.pdf figures/structure/   # after the analyses
latexmk -pdf group_structure.tex
```
