# xc-lab

The lab where Matteo tests ideas for the weekly internship meetings. Speed comes first:
Matteo reads the conclusions, not the code, so the tests are what make a result
trustworthy.

## Neighbours

- `../xc-dynamics`: reviewed thesis code. Import from it freely.
- `../soaring-legacy`: the old monorepo frozen at tag `legacy-final`. Reach it only
  through `lab/legacy.py`, so that porting a piece to xc-dynamics is a one-file change.

Both are read-only from here: edit them only in a session opened in their own repo. A
bug found in soaring gets a workaround in `lab/legacy.py`, with a comment saying what
the port to xc-dynamics must fix.

## Ideas

One idea, one folder: `ideas/YYYY-MM-DD-slug/`, self-contained.

- `README.md`: the question, the answer in three lines, and the xc-dynamics commit it
  ran on (`git -C ../xc-dynamics rev-parse --short HEAD`).
- The code, with `test_*.py` beside it covering every number the answer rests on.
- `out/` for generated files (git-ignored). A note PDF, when the idea gets one, sits in
  the folder and is committed.

An idea is done when `uv run pytest` is green and its README states the answer. An idea
that is only a note still gets its folder.

Copy code between ideas freely. The third copy of a function moves into `lab/`.

## Meetings

`meetings/YYYY-MM-DD/` holds one meeting's slides, which point to the idea folders they
report. Slides are in English, like the earlier decks in `../soaring-legacy/presentations/`.

## Promotion

A result headed for the thesis is rewritten and reviewed in xc-dynamics. Lab code stays
in the lab.

## Data

Flights live under `/Volumes/SSD_DATA` (paths in `../xc-dynamics/configs/download.yaml`;
raw `.igc` files under `xc-viewer-data/<discipline>/raw/igc/`). A test that needs the disk
skips when it is not mounted.
