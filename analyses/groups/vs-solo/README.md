# Group against solo

**Question.** Does flying in a group change how a pilot moves, against the same pilots and routes
flown alone? The analyses are the scripts of [`..`](../); here are the documents that report them.

## Files

- `general/`: `group_vs_solo.pdf`, the general report (setup, raw contrasts, zones, route, before
  and after the group, leaders and followers), built from `group_vs_solo.tex`, `section_*.tex`,
  `abstract.tex` and `appendices.tex`.
- `comparisons/`: `group_vs_solo_comparisons.pdf`, a short report of selected comparisons of group
  time with solo flight, each with the size of its populations: the definitions of solo, group and
  the same flights alone, their raw difference, and the same route flown alone by isolated solo
  flights (`../isolated.py`), by wing level. Built from `group_vs_solo_comparisons.tex` and
  `sections/`; more comparisons will be added as sections.
- `code/`: the scripts that turn the JSON of the analyses into what the documents read.
  `tables.py` writes the tables and the macros, `figures.py` the report figures, `facts.py` the
  facts of the grouping, `structure_tables.py` the macros of [`../structure`](../structure/).
- `generated/`: tables and macros written by `code/` (never edited by hand), read by every
  document of this folder and by `../structure`.

Figures come from `../out/<run>/`.

```bash
cd analyses/groups
uv run python vs-solo/code/facts.py; uv run python vs-solo/code/tables.py; uv run python vs-solo/code/figures.py   # after the analyses
cd vs-solo/general && latexmk -pdf group_vs_solo.tex
cd ../comparisons && latexmk -pdf group_vs_solo_comparisons.tex   # after isolated.py and tables.py
```
