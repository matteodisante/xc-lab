"""Why linear interpolation loses climbs at 5 s: measured on the first 40 flights of
out/cadence.csv. Run after cadence.py, from the root of xc-lab."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import phases  # noqa: E402
from plots import flight  # noqa: E402

ALPHA = phases.ALPHA["para"]


def minority_in_top(turn, window=phases.WINDOW):
    """Number of minority-sign values among the window//2 largest |turn| of each window."""
    half = window // 2
    w = np.lib.stride_tricks.sliding_window_view(np.pad(turn, (half, half), mode="reflect"), window)[: len(turn)]
    top = np.take_along_axis(w, np.argpartition(-np.abs(w), kth=half - 1, axis=1)[:, :half], axis=1)
    return np.minimum((top > 0).sum(axis=1), (top < 0).sum(axis=1))


def main():
    ids = pd.read_csv(HERE / "out/cadence.csv").flight_id.unique()[:40]
    flips = {False: [], True: []}
    minority = {"1 Hz": [], False: [], True: []}
    heading = {5: [], 10: []}
    for fid in ids:
        t, xyz = flight(fid)
        for grid, vals in phases.regular_grid(t, xyz, 4.0):
            if len(grid) < 300:
                continue
            obs, turn, _ = phases.features(grid, *vals, ALPHA)
            climb = phases.label(grid, *vals, ALPHA)[0] == 0
            minority["1 Hz"].append(minority_in_top(turn)[climb])
            for k in heading:
                keep = grid % k == 0
                step = np.arctan2(np.diff(vals[1, keep]), np.diff(vals[0, keep]))
                change = (np.diff(step) + np.pi) % (2 * np.pi) - np.pi
                heading[k].append(np.abs(change[climb[keep][1:-1]]))
            keep = grid % 5 == 0
            for cubic in (False, True):
                for g5, v5 in phases.regular_grid(grid[keep], vals[:, keep], 10.0, cubic):
                    if len(g5) < phases.WINDOW:
                        continue
                    obs5, turn5, _ = phases.features(g5, *v5, ALPHA)
                    idx = (g5 - grid[0]).astype(int)
                    on = climb[idx]
                    flips[cubic].append((obs[idx] @ [4, 2, 1] == 3)[on] & (obs5 @ [4, 2, 1] == 2)[on])
                    minority[cubic].append(minority_in_top(turn5)[on])
    out = {
        "climb_fixes_011_to_010": {"linear": float(np.concatenate(flips[False]).mean()),
                                   "cubic": float(np.concatenate(flips[True]).mean())},
        "climb_windows_two_minority": {"native": float((np.concatenate(minority["1 Hz"]) >= 2).mean()),
                                       "linear": float((np.concatenate(minority[False]) >= 2).mean()),
                                       "cubic": float((np.concatenate(minority[True]) >= 2).mean())},
        "climb_heading_change_rad": {k: {"median": float(np.median(np.concatenate(v))),
                                         "above_2": float((np.concatenate(v) > 2.0).mean()),
                                         "above_2.5": float((np.concatenate(v) > 2.5).mean())}
                                     for k, v in heading.items()},
        "flights": len(ids),
    }
    (HERE / "out/mechanism.yaml").write_text(yaml.safe_dump(out, sort_keys=False))
    print(yaml.safe_dump(out, sort_keys=False))


if __name__ == "__main__":
    main()
