"""The phase runs of the paragliders of the story under one label set, with the terms of the statistics of
phase_gaps.py.

Label sets (LABEL_SETS, `--labels NAME`, default `refit`): the runs of our refit of Vilpellet's model and his own,
both read from the folder of the viewer, `phase_segments.parquet` (his are the same file as in phase_msd.RUNS). A run
is in the time base of the viewer's fixes, which is the flight clock of the cache of longlags.py (both start at
`ground_phase_start_s`); `--check` shows it on a few flights. A run lasts s = t_end - t_start + the logger interval
of its flight (his: n_fixes); its segment number is not used: runs are ordered by time within a flight.

Per run: the terms of budget.columns (time, climbs, searches that end in a climb) and, for climbs and glides, from
one pass over the fixes (cached in CACHE/phase_terms.parquet, both label sets together): gain_m (z of the end minus
z of the start of a climb), climb_t (its duration, end minus start), drift_m (its horizontal chord), chord_m (the
chord of a glide), glide_t (its duration) and loss_m (its height lost), and climbs_m (1 for a climb that has them). A
run whose ends are not both fixes of the flight (the labels reach past the cleaned track) has none of these terms:
it counts in the time and the counts of budget.columns only.

Rule for the searches (`--rule context`, the default, or `plain`): with `context`, runs are first merged where
same-phase neighbours touch, then each search is classified by its neighbours in the labelled sequence (same segment,
adjacent when less than max(MAX_GAP, 2 x interval) s lie between them, the labelling's own rule for a piece):
transition-search-climb is a search that is found, transition-search-transition a search that fails, climb-search-climb
is a re-centring and is merged into the climb (the two climbs and the search become one climb run), climb-search-
transition is an exit (its own category) and a search with an unknown neighbour (the start or end of a segment) is
dropped. With `plain` the runs are as labelled and a search is found when the next run of the flight is a climb.

    uv run python phase_terms.py --declared-open --dt5 [--recompute]    # the pass over the fixes, about 10 min
    uv run python phase_terms.py --declared-open --dt5 --check          # the time base on a few flights
"""

import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

import budget
import longlags
from blocks import with_takeoffs
from common.legacy import stream_flights
from tamsd import CACHE, DISK, FIXES, read_flights

SEGMENTATION = Path("/Volumes/SSD_DATA/xc-viewer-data/paragliders/derived/segmentation")
LABEL_SETS = {"refit": SEGMENTATION / "refit/phase_segments.parquet",
              "vilpellet": SEGMENTATION / "vilpellet/phase_segments.parquet"}
SET = sys.argv[sys.argv.index("--labels") + 1] if "--labels" in sys.argv else "refit"
RULE = sys.argv[sys.argv.index("--rule") + 1] if "--rule" in sys.argv else "context"
assert RULE in ("context", "plain")
PLAIN = "_plain" if RULE == "plain" else ""  # the suffix of the outputs of the old rule
TERMS = CACHE / ("phase_terms.parquet" if RULE == "plain" else "phase_terms_context.parquet")
MAX_GAP = 4  # s between two runs of one segment that are still adjacent (search_context.py)
INDEX = DISK / "paragliders/ffvl_cfd_igc/derived/viewer/thermal-planes/thermal-cells.sqlite3"  # row groups of each flight
MOVE_TERMS = ["gain_m", "climb_t", "drift_m", "climbs_m", "chord_m", "glide_t", "loss_m"]
COLUMNS = (["transition_s", "search_s", "climb_s", "climbs", "found", "searches"] if RULE == "plain" else
           ["transition_s", "search_s", "climb_s", "exit_s", "climbs", "searches", "found", "exits", "recentres", "unknown"]) + MOVE_TERMS
# the terms the statistics of phase_gaps.py sum, kept as float32
MOVING = ("climb", "transition")  # the phases that have a gain or a chord
SHIFTS = (-60, -30, -10, 0, 10, 30, 60)  # s: the shifts of --check
CHECK_FLIGHTS = 12  # per logger interval, in --check


def kept_flights():
    """The declared open flights with a logger interval up to longlags.MAX_DT s, indexed by their FFVL number (int):
    flight (the number of the cache), dt_native (s)."""
    flights = with_takeoffs(read_flights())
    flights = flights[flights["dt_native"] <= longlags.MAX_DT + 1e-6]
    return flights.set_index(flights["flight_id"].astype("int64"))[["flight", "dt_native"]]


def read_runs(label_set, flights):
    """The runs of `label_set` of the flights of `flights` (kept_flights): fid, flight, phase, t_start, t_end and
    s, sorted by flight and time."""
    t = pq.read_table(LABEL_SETS[label_set], columns=["flight_id", "segment_id", "phase", "t_start", "t_end"])
    fid = pc.cast(t["flight_id"], pa.int64()).to_numpy()
    keep = np.isin(fid, flights.index.to_numpy())
    phase = pc.dictionary_encode(t["phase"]).combine_chunks()
    runs = pd.DataFrame({"fid": fid[keep], "phase": pd.Categorical.from_codes(phase.indices.to_numpy()[keep], phase.dictionary.to_pylist()),
                         "seg": t["segment_id"].to_numpy()[keep], "t_start": t["t_start"].to_numpy()[keep], "t_end": t["t_end"].to_numpy()[keep]})
    runs["flight"] = flights["flight"].reindex(runs["fid"]).to_numpy()
    runs["dt"] = flights["dt_native"].reindex(runs["fid"]).to_numpy()
    runs["s"] = runs["t_end"] - runs["t_start"] + runs["dt"]
    return runs.sort_values(["flight", "seg", "t_start"], kind="stable", ignore_index=True)


KINDS = {"climb": 0, "search": 1, "transition": 2}
NAMES = ("climb", "search", "transition")


def neighbours(r):
    """Phase code of the previous and the next run of each run (-1 when not adjacent: another segment, or more than
    max(MAX_GAP, 2 x interval) s away), and whether each run follows its previous one."""
    reach = np.maximum(MAX_GAP, 2 * r["dt"].to_numpy())
    adj = ((r["flight"].to_numpy()[1:] == r["flight"].to_numpy()[:-1]) & (r["seg"].to_numpy()[1:] == r["seg"].to_numpy()[:-1])
           & (r["t_start"].to_numpy()[1:] - r["t_end"].to_numpy()[:-1] <= reach[:-1]))
    code = r["phase"].astype(str).map(KINDS).to_numpy()
    return np.where(np.r_[False, adj], np.r_[-1, code[:-1]], -1), np.where(np.r_[adj, False], np.r_[code[1:], -1], -1), adj


def collapse(r, joined):
    """The runs of `r` with the runs j and j - 1 joined where joined[j]: the first row of each group with the end of
    the last, and the group's size."""
    first = np.flatnonzero(~joined)
    last = np.r_[first[1:] - 1, len(r) - 1]
    out = r.iloc[first].reset_index(drop=True)
    out["t_end"] = r["t_end"].to_numpy()[last]
    out["s"] = out["t_end"] - out["t_start"] + out["dt"]
    return out, last - first + 1


def contextual(r):
    """The rule `context` on the runs `r` (read_runs): same-phase neighbours merged, then each search classified by its
    neighbours: found, failed, exit or unknown, and the re-centrings merged into the climbs (recentres counts them).
    Phase is a string: transition, climb, found, failed, exit, unknown."""
    _, _, adj = neighbours(r)
    same = np.r_[False, adj & (r["phase"].to_numpy()[1:] == r["phase"].to_numpy()[:-1])]
    r, _ = collapse(r, same)
    before, after, _ = neighbours(r)
    code = r["phase"].astype(str).map(KINDS).to_numpy()
    srch = code == 1
    kind = np.select([srch & (before == 2) & (after == 0), srch & (before == 2) & (after == 2), srch & (before == 0) & (after == 2)],
                     ["found", "failed", "exit"], "unknown")
    recentre = srch & (before == 0) & (after == 0)
    joined = np.r_[False, recentre[1:] | recentre[:-1]] & np.r_[False, (code[1:] == 0) | (code[:-1] == 0)]
    joined &= np.r_[False, ~(srch[1:] & srch[:-1])]
    out, size = collapse(r.assign(phase=np.where(srch, kind, np.array(NAMES)[code])), joined)
    merged = size > 1
    out["phase"] = np.where(merged, "climb", out["phase"])
    out["recentres"] = np.where(merged, (size - 1) // 2, 0)
    return out


def run_terms(fixes, runs):
    """Gain (z of the end minus z of the start) and horizontal chord of each of `runs` (t_start, t_end at fixes of
    the flight; NaN where an end is not a fix)."""
    fixes = fixes.sort_values("t", kind="stable")
    t, z, e, n = (fixes[c].to_numpy(float) for c in ("t", "z", "E", "N"))
    a, b = (np.clip(np.searchsorted(t, runs[c].to_numpy(float)), 0, len(t) - 1) for c in ("t_start", "t_end"))
    ok = np.isclose(t[a], runs["t_start"]) & np.isclose(t[b], runs["t_end"])
    return np.where(ok, z[b] - z[a], np.nan), np.where(ok, np.hypot(e[b] - e[a], n[b] - n[a]), np.nan)


def processed(label_set, flights):
    """The runs of `label_set` as the rule of the run has them: read_runs, or contextual(read_runs)."""
    runs = read_runs(label_set, flights)
    return contextual(runs) if RULE == "context" else runs


def measure(flights):
    """One pass over the fixes: the terms of run_terms of the climbs and glides of every label set, as a frame
    (labels, flight, t_start, t_end, dz, chord)."""
    runs = {k: processed(k, flights) for k in LABEL_SETS}
    runs = {k: r[r["phase"].isin(MOVING)].reset_index(drop=True) for k, r in runs.items()}
    number, out = flights["flight"], []
    for k, fixes in enumerate(stream_flights(FIXES["para"], ["t", "E", "N", "z"])):
        flight = number.get(int(fixes["flight_id"].iat[0]))
        if flight is None:
            continue
        for label_set, r in runs.items():
            lo, hi = np.searchsorted(r["flight"], flight, "left"), np.searchsorted(r["flight"], flight, "right")
            if hi > lo:
                dz, chord = run_terms(fixes, r.iloc[lo:hi])
                out.append(r.iloc[lo:hi][["flight", "t_start", "t_end"]].assign(labels=label_set, dz=dz, chord=chord))
        if k % 10000 == 0:
            print(k, "flights read", flush=True)
    return pd.concat(out, ignore_index=True)


def table(label_set, flights):
    """The runs of `label_set` with the terms of the module docstring, sorted by flight and time: flight, phase, t_mid,
    s, hours and COLUMNS."""
    if not TERMS.exists() or "--recompute" in sys.argv:
        measure(flights).to_parquet(TERMS)
    terms = pd.read_parquet(TERMS)
    terms = terms[terms["labels"] == label_set].drop(columns="labels")
    runs = processed(label_set, flights).merge(terms, on=["flight", "t_start", "t_end"], how="left")
    ok = runs["dz"].notna().to_numpy()
    kind = runs["phase"].astype(str)
    climb, glide = ((kind == k).to_numpy() & ok for k in MOVING)
    runs = runs.assign(gain_m=np.where(climb, runs["dz"], 0.0), climb_t=np.where(climb, runs["t_end"] - runs["t_start"], 0.0),
                       drift_m=np.where(climb, runs["chord"], 0.0), climbs_m=climb * 1.0, chord_m=np.where(glide, runs["chord"], 0.0),
                       glide_t=np.where(glide, runs["t_end"] - runs["t_start"], 0.0), loss_m=np.where(glide, -runs["dz"], 0.0))
    if RULE == "plain":
        runs = budget.columns(runs.assign(phase=kind, n_fixes=runs["s"], flight_id=runs["flight"], segment_id=0).drop(columns=["dz", "chord", "fid"]))
        runs = runs.assign(search_s=runs["found_s"] + runs["vain_s"])
    else:
        found, failed, exit_, unknown = ((kind == k).to_numpy() for k in ("found", "failed", "exit", "unknown"))
        runs["s"] = np.where(unknown, 0.0, runs["s"])
        runs = runs.assign(hours=runs["s"] / 3600, transition_s=runs["s"] * (kind == "transition"), climb_s=runs["s"] * (kind == "climb"),
                           search_s=runs["s"] * (found | failed), exit_s=runs["s"] * exit_, climbs=(kind == "climb") * 1.0,
                           searches=(found | failed) * 1.0, found=found * 1.0, exits=exit_ * 1.0, unknown=unknown * 1.0)
    runs = runs.assign(t_mid=(runs["t_start"] + runs["t_end"]) / 2, phase=kind.astype("category"))
    runs[COLUMNS] = runs[COLUMNS].astype(np.float32)
    return runs[["flight", "phase", "t_mid", "s", "hours", *COLUMNS]]


# ---- the time base ----

def codes(runs, grid):
    """The phase code (0 none, 1 climb, 2 search, 3 transition) at each instant of `grid` of one flight whose runs are `runs`."""
    code = np.zeros(len(grid), int)
    for r in runs.itertuples():
        code[(grid >= r.t_start) & (grid <= r.t_end)] = ("climb", "search", "transition").index(r.phase) + 1
    return code


def read_flight(fid):
    """t, z of one flight of the fixes (via the row groups of the index of the viewer)."""
    db = sqlite3.connect(f"file:{INDEX}?mode=ro", uri=True)
    groups = [g for (g,) in db.execute("SELECT row_group FROM flight_groups WHERE discipline = 'paragliders' AND flight_id = ?", (str(fid),))]
    db.close()
    f = pq.ParquetFile(FIXES["para"]).read_row_groups(sorted(groups), columns=["flight_id", "t", "z"]).to_pandas()
    return f[f["flight_id"] == str(fid)].sort_values("t")


def check(flights):
    """The time base on CHECK_FLIGHTS flights per logger interval (1, 2 and 5 s), for each shift of SHIFTS s of the runs
    of the refit: the mean vertical speed of its climbs and of its glides, read on the fixes (positive in climbs,
    negative in glides, extreme at 0 if the base is right); and, on the 1-Hz flights both label sets cover, the share
    of instants where the refit and his labels agree when his are shifted (the peak at 0)."""
    rng = np.random.default_rng(0)
    refit, his = read_runs("refit", flights), read_runs("vilpellet", flights)
    speed = {phase: {s: [] for s in SHIFTS} for phase in MOVING}
    agree = {s: [] for s in SHIFTS}
    for dt in (1.0, 2.0, 5.0):
        pick = flights[(flights["dt_native"] == dt) & flights["flight"].isin(refit["flight"])]
        for fid in rng.choice(pick.index.to_numpy(), CHECK_FLIGHTS, replace=False):
            number = flights["flight"].at[fid]
            mine = refit[refit["flight"] == number]
            f = read_flight(fid)
            t, z = f["t"].to_numpy(float), f["z"].to_numpy(float)
            for phase in MOVING:
                r = mine[mine["phase"] == phase]
                for s in SHIFTS:
                    dz = np.interp(r["t_end"] + s, t, z) - np.interp(r["t_start"] + s, t, z)
                    speed[phase][s].append(dz.sum() / (r["t_end"] - r["t_start"]).sum())
            theirs = his[his["flight"] == number]
            if len(theirs):
                grid = np.arange(0, t.max())
                a, b = codes(mine, grid), codes(theirs, grid)
                for s in SHIFTS:
                    c = np.roll(b, s)
                    both = (a > 0) & (c > 0)
                    agree[s].append((a == c)[both].mean())
            print(f"flight {fid} (dt {dt:g} s): {len(mine)} runs, from {mine['t_start'].min():g} to {mine['t_end'].max():g} s, fixes {t.min():g} to {t.max():g} s", flush=True)
    for phase in MOVING:
        print(f"mean vertical speed of the {phase}s of the refit (m/s) by shift (s):", {s: round(float(np.mean(v)), 3) for s, v in speed[phase].items()})
    print("share of instants with the same label, his shifted (s):", {s: round(float(np.mean(v)), 4) for s, v in agree.items()}, f"on {len(agree[0])} flights")


if __name__ == "__main__":
    kept = kept_flights()
    if "--check" in sys.argv:
        check(kept)
    else:
        t = table(SET, kept)
        print(SET, len(t), "runs,", t["flight"].nunique(), "flights; mean climb rate", round(t["gain_m"].sum() / t["climb_t"].sum(), 2), "m/s")
