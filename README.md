# xc-lab

Quick experiments and weekly meeting material for an internship on cross-country
paraglider and hang-glider flights.

The reviewed thesis code lives in [xc-dynamics](https://github.com/matteodisante/xc-dynamics).
This repository tests ideas fast; results that reach the thesis are rewritten there.

## Layout

- `methods/<slug>/`: one method we build per folder, with its code, tests and notes; `methods/README.md`
  lists them with their status.
- `analyses/<topic>/`: quantities computed on the flight data, as figures and numbers; one
  folder per topic, `analyses/README.md` lists them.
- `meetings/YYYY-MM/<talk>/`: the slides of one talk, grouped by month.
- `logbook/`: the lab diary, one file per month, in Italian.
- `common/`: helpers shared by the methods and the analyses, with their tests. `common/legacy.py` borrows the Vilpellet
  segmenter from [soaring-anomalous-transport](https://github.com/matteodisante/soaring-anomalous-transport)
  at tag `legacy-final`, until xc-dynamics has it.

## Setup

The lab expects its two dependencies as sibling folders:

```
xc-dynamics/       git clone https://github.com/matteodisante/xc-dynamics
soaring-legacy/    git -C soaring-anomalous-transport worktree add --detach ../soaring-legacy legacy-final
xc-lab/
```

Then:

```
uv sync
uv run pytest
```

Tests that need flight data skip when `/Volumes/SSD_DATA` is not mounted.
