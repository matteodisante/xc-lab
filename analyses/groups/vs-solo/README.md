# Group against solo

**Question.** Does flying in a group change how a pilot moves, against the same pilots and routes
flown alone? The analyses are the scripts of [`..`](../); here are the documents that report them.

## Files

- `general/`: `group_vs_solo.pdf`, the general report (setup, raw contrasts, zones, route, before
  and after the group, leaders and followers), built from `group_vs_solo.tex`, `section_*.tex`,
  `abstract.tex` and `appendices.tex`.
- `comparisons/`: `group_vs_solo_comparisons.pdf`, group time against solo flight on open flights: the premise
  (declared open flights, why their long lags bend, the threshold on the opening) and four comparisons, raw
  (`../raw.py`), same route, season, hour and wing (`../matched.py`), the decomposition of the raw gap
  (`../decomposition.py`) and before, during and after (`../prepost.py --open`). Built from
  `group_vs_solo_comparisons.tex` and `sections/` (aim, flights, declared, opening, raw, matched, decomposition,
  prepost, conclusions, appendices); every number is a macro of `code/comparisons.py`, tables and macros in
  `generated/comparisons/` (checked against the JSON by `test_groups.py`).
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
# --declared-open: tamsd, longlags --recompute, opening; --open: tamsd, longlags --recompute, performance,
# then raw.py, prepost.py --window 2400, matched.py (counts, then --stats), decomposition.py, each with --open
uv run python vs-solo/code/comparisons.py   # macros and tables of comparisons/
cd vs-solo/comparisons && latexmk -pdf group_vs_solo_comparisons.tex
```
