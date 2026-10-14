# xc-lab

Quick experiments and weekly meeting material for an internship on cross-country
paraglider and hang-glider flights.

The reviewed thesis code lives in [xc-dynamics](https://github.com/matteodisante/xc-dynamics).
This repository tests ideas fast; results that reach the thesis are rewritten there.

## Layout

- `ideas/YYYY-MM-DD-slug/`: one idea per folder, with its code, tests and notes.
- `meetings/YYYY-MM-DD/`: the slides for one weekly meeting.
- `lab/`: helpers shared by several ideas. `lab/legacy.py` borrows the Vilpellet
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
