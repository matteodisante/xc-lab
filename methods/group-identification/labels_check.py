"""The grouping of the chosen run with the refit phase labels in place of Vilpellet's, on a sample of days.

    uv run python labels_check.py [--days 20] [--seed 0] [--jobs 2]

The phases enter only through the heading condition, checked when both pilots glide. Everything else is the
chosen run (parameters and tau* of its calibration.json). The sample: one random day in each of `--days`
chronological chunks of the busiest tenth of the dates (by flights). Each date is cached in
SSD_DATA/xc-lab/labels-check/days/, so an interrupted run resumes. Writes out/labels_check.json.
"""

import argparse
import json
import pickle
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

import contacts
import grid
import run
import sweep

CHOSEN = run.DATA / "runs" / "d120_eh500_ez300_th45_g120_q0.1"
VIEWER = Path("/Volumes/SSD_DATA/xc-viewer-data")
REFIT = {"para": VIEWER / "paragliders/derived/segmentation/refit/phase_segments.parquet",
         "hang": VIEWER / "hang_gliders/derived/segmentation/refit/phase_segments.parquet"}
CACHE = Path("/Volumes/SSD_DATA/xc-lab/labels-check")
OUT = Path(__file__).parent / "out" / "labels_check.json"
BUSIEST = 0.1  # share of the dates, by flights, the sample is drawn from
RUN_COLUMNS = ["flight_id", "segment_id", "t_start", "t_end", "state"]


def sample(n, seed):
    """The dates: one random day in each of n chronological chunks of the busiest dates."""
    count = pd.read_parquet(CHOSEN / "flights.parquet", columns=["date"])["date"].value_counts()
    busy = np.sort(count.index[count >= count.quantile(1 - BUSIEST)].to_numpy())
    rng = np.random.default_rng(seed)
    return [str(rng.choice(chunk)) for chunk in np.array_split(busy, n)]


def refit_runs(discipline, ids):
    """The refit runs of some flights. Its segment_id is not that of fixes.parquet: runs are matched on time only."""
    r = pd.read_parquet(REFIT[discipline], columns=["flight_id", "phase", "t_start", "t_end"], filters=[("flight_id", "in", ids)])
    r["state"] = r.pop("phase").map(grid.STATES).astype(np.int8)
    return r.assign(segment_id=0)[RUN_COLUMNS]


def labelled_runs(flights):
    """{label set: {discipline: runs}} of the given flights (rows of grid.archive())."""
    out = {"vilpellet": {}, "refit": {}}
    for d, f in flights.groupby("discipline"):
        ids = f["flight_id"].tolist()
        v = grid.phase_runs(d)
        out["vilpellet"][d] = v[v["flight_id"].isin(ids)][RUN_COLUMNS]
        out["refit"][d] = refit_runs(d, ids)
    return out


def intervals_of(flights, runs, p, zero_segment):
    """Phases 0-3 on one date: the flight table, the pair intervals and the state of every grid row."""
    states = grid.states
    if zero_segment:
        grid.states = lambda seg, t, r: states(np.zeros_like(seg), t, r)
    try:
        table, g = grid.day(flights, runs, p["step"], p["h_max"], p["r_dep"])
    finally:
        grid.states = states
    if g is None:
        return table, pd.DataFrame(columns=["a", "b", "t_on", "t_off"]), g
    iv = contacts.real_intervals(table, g, contacts.Tracks(g, p["step"]), p)
    return table, iv, g[["f", "t", "state"]].astype({"t": np.int64})


def one_date(date, flights, runs, p):
    start = time.time()
    out = {"date": date}
    for name in ("vilpellet", "refit"):
        out[name] = intervals_of(flights, runs[name], p, name == "refit")
    out["seconds"] = time.time() - start
    return out


def group_seconds(iv, n, tau):
    """Per flight the intervals [start, end) it spends in a group, and the number of groups and episodes."""
    kept = iv[iv["duration"] >= tau] if len(iv) else iv
    rows = [(a, b, on, off) for a, b, on, off in kept[["a", "b", "t_on", "t_off"]].itertuples(index=False)]
    groups, _, links = sweep.groups(rows)
    mine = {f: [] for f in range(n)}
    for m, start, end in groups:
        for f in m:
            mine[f].append((start, end))
    return mine, len(groups), len(set(sweep.episodes(len(groups), links))) if groups else 0


def overlap(a, b):
    return sum(max(0, min(e1, e2) - max(s1, s2)) for s1, e1 in a for s2, e2 in b)


def summarise(results, tau):
    """The totals of the two label sets over the dates, and the agreement of their group time."""
    tot = {k: dict(flights=0, labelled=0, with_group=0, seconds=0, groups=0, episodes=0, intervals=0) for k in ("vilpellet", "refit")}
    shared = union = 0
    gained = lost = both = 0
    rows, same, rows_both = 0, 0, 0
    per_flight = []
    for r in results:
        mine = {}
        for k in tot:
            table, iv, g = r[k]
            iv = iv.assign(duration=iv["t_off"] - iv["t_on"])
            mine[k], n_groups, n_episodes = group_seconds(iv, len(table), tau)
            secs = {f: sum(e - s for s, e in v) for f, v in mine[k].items()}
            t = tot[k]
            t["flights"] += len(table)
            t["labelled"] += int(table["labelled"].sum())
            t["with_group"] += sum(s > 0 for s in secs.values())
            t["seconds"] += sum(secs.values())
            t["groups"] += n_groups
            t["episodes"] += n_episodes
            t["intervals"] += int((iv["duration"] >= tau).sum())
        for f in mine["vilpellet"]:
            a, b = mine["vilpellet"][f], mine["refit"][f]
            i = overlap(a, b)
            sa, sb = sum(e - s for s, e in a), sum(e - s for s, e in b)
            shared += i
            union += sa + sb - i
            gained += sa == 0 and sb > 0
            lost += sa > 0 and sb == 0
            both += sa > 0 and sb > 0
            if sa + sb > 0:
                per_flight.append(i / (sa + sb - i))
        gv, gr = r["vilpellet"][2], r["refit"][2]
        if gv is not None and gr is not None:
            m = gv.merge(gr, on=["f", "t"], suffixes=("_v", "_r"))
            m = m[(m["state_v"] != grid.UNCLASSIFIED) & (m["state_r"] != grid.UNCLASSIFIED)]
            rows_both += len(m)
            same += int((m["state_v"] == m["state_r"]).sum())
    v, o = tot["vilpellet"], tot["refit"]
    return {"vilpellet": v, "refit": o, "flights_change_pct": 100 * (o["with_group"] / v["with_group"] - 1),
            "time_change_pct": 100 * (o["seconds"] / v["seconds"] - 1),
            "seconds_per_flight": {"vilpellet": v["seconds"] / v["flights"], "refit": o["seconds"] / o["flights"]},
            "seconds_per_group_flight": {"vilpellet": v["seconds"] / v["with_group"], "refit": o["seconds"] / o["with_group"]},
            "episodes_change_pct": 100 * (o["episodes"] / v["episodes"] - 1),
            "shared_seconds": shared, "shared_of_vilpellet_pct": 100 * shared / v["seconds"],
            "shared_of_refit_pct": 100 * shared / o["seconds"], "shared_of_union_pct": 100 * shared / union,
            "flights_gaining_group": int(gained), "flights_losing_group": int(lost), "flights_in_group_in_both": int(both),
            "mean_flight_jaccard": float(np.mean(per_flight)),
            "grid_state_agreement_pct": 100 * same / rows_both, "grid_rows_labelled_in_both": rows_both}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--jobs", type=int, default=2)
    args = parser.parse_args()
    calibration = json.loads((CHOSEN / "calibration.json").read_text())
    p, tau = calibration["params"], calibration["tau_star"]
    dates = sample(args.days, args.seed)
    cache = CACHE / "days"
    cache.mkdir(parents=True, exist_ok=True)
    todo = [d for d in dates if not (cache / f"{d}.pkl").exists()]
    print(f"{len(dates)} dates, {len(todo)} to compute: {dates}", flush=True)
    archive = grid.archive()
    if todo:
        mine = archive[archive["date"].isin(todo)]
        runs = labelled_runs(mine)
        with ProcessPoolExecutor(args.jobs) as pool:
            futures = {}
            for d in todo:
                f = mine[mine["date"] == d]
                r = {k: {disc: v[v["flight_id"].isin(f.loc[f["discipline"] == disc, "flight_id"])] for disc, v in by.items()}
                     for k, by in runs.items()}
                futures[d] = pool.submit(one_date, d, f, r, p)
            for d, future in futures.items():
                result = future.result()
                (cache / f"{d}.pkl").write_bytes(pickle.dumps(result))
                print(f"  {d}: {len(result['vilpellet'][0])} flights, {result['seconds']:.0f} s", flush=True)
    results = [pickle.loads((cache / f"{d}.pkl").read_bytes()) for d in dates]
    # the Vilpellet run recomputed here against the intervals the chosen run stored for these dates
    stored = [pickle.loads((CHOSEN / "days" / f"{d}.pkl").read_bytes())["intervals"] for d in dates]
    reproduced = all(len(s) == len(r["vilpellet"][1]) and (s[["a", "b", "t_on", "t_off"]].to_numpy() == r["vilpellet"][1][["a", "b", "t_on", "t_off"]].to_numpy()).all()
                     for s, r in zip(stored, results))
    out = {"run": CHOSEN.name, "tau_star_s": tau, "days": len(dates), "dates": dates, "seed": args.seed,
           "sampled_from": f"the busiest {BUSIEST:.0%} of the dates by flights, one per chronological chunk",
           "recomputed_vilpellet_equals_stored_run": bool(reproduced), **summarise(results, tau)}
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k not in ("dates", "vilpellet", "refit")}, indent=1))
