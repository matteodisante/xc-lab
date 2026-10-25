"""Group identification on the cleaned tracks of the old repository, date by date.

    uv run python run.py [--jobs J] [--sample N | --dates D1,D2] [--set k=v,...] [--out DIR]

Every run writes to its own folder on SSD_DISANTE, named by its parameters (run_name):
runs/<name> for all dates, trials/sample<N>/<name> for a sample. Phases 0-3 and the null
model run per date in parallel; each date is kept in <folder>/days/, so an interrupted run
resumes. Then tau* comes from the null model over the dates done, and Phase 4 writes the
tables of the note (section Outputs): pair_intervals (T1), groups (T2), membership (T3),
episodes (T4), flight_summary (T5), configurations (T6), and flights, the key of the flight
index used by all of them. calibration.json holds tau* and the false discovery rate, PARAMETERS.md every
parameter with its meaning.
"""

import argparse
import json
import os
import pickle
import shutil
import subprocess
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

import numpy as np
import pandas as pd

import contacts
import grid
import sweep

# Starting values of the note (table "Symbols and parameters"); v_max is that of the filter
# of Phase 1, whose pieces are split at faster steps, so the lemma holds on the old tracks.
PARAMS = dict(step=5, h_max=10, r_dep=3000.0, delta=300, eps_h=500.0, eps_z=300.0, theta_max=45.0, gap=120,
              v_max=100 / 3.6, r_s=10_000.0, q=0.05, n_null=200)
PARAMS["r_min"] = 2 * PARAMS["delta"] + PARAMS["step"]
STARTING = dict(PARAMS)
TAUS = np.arange(0, 4 * 3600 + 1, PARAMS["step"])  # durations at which the FDR is evaluated, s
DATA = Path("/Volumes/SSD_DISANTE/xc-lab/group-identification")
# The parameters that define a contact and tau*, in the folder name of every run.
NAMED = {"delta": "d", "eps_h": "eh", "eps_z": "ez", "theta_max": "th", "gap": "g", "q": "q"}
MEANINGS = {
    "delta": ("δ", "s", "largest delay between the two pilots of a correspondence"),
    "eps_h": ("ε_h", "m", "how close one pilot passes, horizontally, to where the other was"),
    "eps_z": ("ε_z", "m", "the same in altitude"),
    "theta_max": ("θ_max", "°", "largest angle between the headings when both glide"),
    "gap": ("g", "s", "gaps of contact shorter than this are joined"),
    "q": ("q", "", "false discovery rate accepted; fixes τ*"),
    "r_dep": ("R_dep", "m", "departure: the first fix farther than this from takeoff"),
    "step": ("Δ", "s", "step of the common time grid"),
    "h_max": ("h_max", "s", "longest time between two fixes across which a position is interpolated"),
    "v_max": ("v_max", "m/s", "speed of the candidate filter; pieces end at faster steps"),
    "r_s": ("R_s", "m", "null model: pairs whose takeoffs are closer than this"),
    "n_null": ("", "", "null model: surrogate pairs per date, at most"),
    "r_min": ("r_min", "s", "null model: smallest time shift, 2δ + Δ"),
}


def run_name(p):
    """Folder name of a run: the parameters of NAMED, then any other one changed from the
    starting values."""
    parts = [f"{short}{p[key]:g}" for key, short in NAMED.items()]
    parts += [f"{key}{p[key]:g}" for key in STARTING if key not in NAMED and key != "r_min"
              and p[key] != STARTING[key]]
    return "_".join(parts)


def describe(folder, p, dates, calibration=None):
    """Write folder/PARAMETERS.md: the parameters of the run with their meaning, and tau*."""
    here = Path(__file__).parent
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=here, capture_output=True, text=True)
    dirty = subprocess.run(["git", "status", "--porcelain", "."], cwd=here, capture_output=True, text=True)
    version = commit.stdout.strip() + (" with uncommitted changes" if dirty.stdout.strip() else "")
    lines = [f"# Group identification: {folder.name}", "",
             f"`ideas/group-identification/run.py` of xc-lab ({version}) on {dates} dates of the cleaned "
             "tracks of soaring (tag legacy-final) on SSD_DISANTE.", "",
             "| parameter | value | meaning |", "|---|---|---|"]
    lines += [f"| {symbol or key} | {p[key]:g} {unit} | {meaning} |" for key, (symbol, unit, meaning) in MEANINGS.items()]
    if calibration is not None and calibration["tau_star"] is not None:
        tau = calibration["tau_star"]
        lines.append(f"| τ* | {tau} s ({tau / 60:.1f} min) | shortest pair interval kept: the first duration "
                     "with FDR ≤ q |")
    lines += ["", "Tables of the note (section Outputs): `pair_intervals`, `groups`, `membership`, "
              "`episodes`, `flight_summary`, `configurations`, `flights`; `calibration.json`; `days/` caches each "
              "date."]
    (folder / "PARAMETERS.md").write_text("\n".join(lines) + "\n")


def one_date(date, flights, runs, p):
    """Phases 0-3 and the null model on one date."""
    start = time.time()
    table, g = grid.day(flights, runs, p["step"], p["h_max"], p["r_dep"])
    out = {"date": date, "table": table, "intervals": pd.DataFrame(columns=["a", "b", "t_on", "t_off", "ell_med", "d_dec"]),
           "null_pairs": pd.DataFrame(columns=["i", "j", "r", "exposure"]),
           "null_intervals": pd.DataFrame(columns=["i", "j", "duration"]), "exposure": 0.0}
    if g is not None:
        tracks = contacts.Tracks(g, p["step"])
        out["intervals"] = contacts.real_intervals(table, g, tracks, p)
        a, b = contacts.near_pairs(table, p["r_s"])
        out["exposure"] = float(contacts.exposure(table, a, b).sum())
        rng = np.random.default_rng(int(date.replace("-", "")))
        out["null_pairs"], out["null_intervals"] = contacts.null_intervals(table, g, tracks, rng, p)
    out["seconds"] = time.time() - start
    return out


def compute(flights, dates, days, jobs, job=None, params=None):
    """Run `job` (one_date by default) with `params` (PARAMS) on every date not yet in `days`,
    keeping each result there; a result is a dict with the date under "date"."""
    job, params = job or one_date, params or PARAMS
    todo = [d for d in dates if not (days / f"{d}.pkl").exists()]
    print(f"{len(dates)} dates, {len(todo)} to compute", flush=True)
    if not todo:
        return
    flights = flights[flights["date"].isin(todo)]
    runs = {}
    for d in grid.DERIVED:
        r = grid.phase_runs(d)
        r = r.merge(flights.loc[flights["discipline"] == d, ["flight_id", "date"]], on="flight_id")
        runs[d] = dict(tuple(r.groupby("date")))
    with ProcessPoolExecutor(jobs) as pool:
        pending, done = set(), 0
        for date in todo + [None]:
            if date is not None:
                mine = flights[flights["date"] == date]
                r = {d: runs[d].get(date, pd.DataFrame(columns=["flight_id", "segment_id", "t_start", "t_end", "state"]))
                     for d in grid.DERIVED}
                pending.add(pool.submit(job, date, mine, r, params))
            while pending and (len(pending) >= 2 * jobs or date is None):
                finished, pending = wait(pending, return_when=FIRST_COMPLETED)
                for f in finished:
                    result = f.result()
                    (days / f"{result['date']}.pkl").write_bytes(pickle.dumps(result))
                    done += 1
                    if done % 100 == 0 or done == len(todo):
                        print(f"  {done}/{len(todo)} dates", flush=True)


def load(days, dates):
    """The per-date results, with flights numbered across dates: the flight table, the raw
    intervals (a, b as global flight numbers), and per date the null model: the exposure of
    its real pairs within r_s, of its surrogate pairs, and the durations of their intervals."""
    tables, intervals, null = [], [], []
    offset = 0
    for date in dates:
        r = pickle.loads((days / f"{date}.pkl").read_bytes())
        t = r["table"].assign(date=date, flight=np.arange(offset, offset + len(r["table"])))
        tables.append(t)
        iv = r["intervals"].astype({"a": np.int64, "b": np.int64}).assign(date=date)
        iv[["a", "b"]] += offset
        intervals.append(iv)
        null.append((r["exposure"], float(r["null_pairs"]["exposure"].sum()),
                     r["null_intervals"]["duration"].to_numpy(np.int64)))
        offset += len(r["table"])
    flights = pd.concat(tables, ignore_index=True)
    intervals = pd.concat(intervals, ignore_index=True)
    intervals["duration"] = intervals["t_off"] - intervals["t_on"]
    return flights, intervals, null


def at_least(durations, taus):
    """How many durations are at least each tau."""
    return len(durations) - np.searchsorted(np.sort(durations), taus, side="left")


def calibrate(intervals, null, p):
    """tau* = the shortest duration at which the estimated false discovery rate falls to q
    (Eq. fdr of the note): the intervals lasting at least tau that chance alone gives the real
    pairs with takeoffs within r_s, sum over dates of rho0 H (rho0 the rate of surrogate
    intervals per second of surrogate exposure, H the exposure of the real pairs), over N,
    the number of real intervals of those pairs lasting at least tau. Surrogates are a sample
    of the pairs of each date, so the rate is taken date by date."""
    near = intervals.loc[intervals["d_dec"] < p["r_s"], "duration"].to_numpy()
    real = at_least(near, TAUS)
    chance = np.zeros(len(TAUS))
    for exposure, surrogate, durations in null:
        if surrogate > 0:
            chance += at_least(durations, TAUS) / surrogate * exposure
    fdr = np.divide(chance, real, out=np.full(len(TAUS), np.inf), where=real > 0)
    ok = np.flatnonzero(fdr <= p["q"])
    return {"tau_star": int(TAUS[ok[0]]) if ok.size else None, "tau": TAUS.tolist(), "fdr": fdr.tolist(),
            "real": real.tolist(), "chance": chance.tolist(), "exposure_real": sum(n[0] for n in null),
            "exposure_null": sum(n[1] for n in null), "null_intervals": sum(len(n[2]) for n in null)}


def label(flights, intervals, tau_star):
    """Phase 4 on every date: tables T1-T6 of the note, with global flight, group,
    configuration, episode and block numbers."""
    kept = intervals[intervals["duration"] >= tau_star]
    gs, cs, ms = [], [], []
    block = np.zeros(len(flights), np.int64)
    n_group = n_config = n_episode = n_block = 0
    for date, table in flights.groupby("date", sort=False):
        first = table["flight"].iat[0]
        local = table.reset_index(drop=True)
        iv = kept[kept["date"] == date]
        rows = [(a - first, b - first, on, off) for a, b, on, off in iv[["a", "b", "t_on", "t_off"]].itertuples(index=False)]
        groups, configs, links = sweep.groups(rows)
        episode = sweep.episodes(len(groups), links)
        blocks = sweep.blocks(len(local), groups, episode)
        block[first:first + len(local)] = blocks + n_block
        for g, (m, start, end) in enumerate(groups):
            gs.append((date, n_group + g, n_episode + episode[g], start, end, len(m), sorted(f + first for f in m)))
        for c, (m, edges, start, end, g) in enumerate(configs):
            cs.append((date, n_config + c, n_group + g, start, end, len(m), sorted(f + first for f in m),
                       sorted([a + first, b + first] for a, b in edges)))
        mb = sweep.membership(local, groups)
        mb["flight"] += first
        mb["group_id"] = np.where(mb["group"] >= 0, mb["group"] + n_group, -1)
        ms.append(mb.drop(columns="group"))
        n_group += len(groups)
        n_config += len(configs)
        n_episode += int(episode.max()) + 1 if len(groups) else 0
        n_block += int(blocks.max()) + 1
    t2 = pd.DataFrame(gs, columns=["date", "group_id", "episode_id", "t_start", "t_end", "n", "members"])
    t6 = pd.DataFrame(cs, columns=["date", "config_id", "group_id", "t_start", "t_end", "n", "members", "edges"])
    t3 = pd.concat(ms, ignore_index=True)
    t3 = t3.merge(t2[["group_id", "episode_id", "n"]], on="group_id", how="left")
    t3["episode_id"] = t3["episode_id"].fillna(-1).astype(np.int64)
    t3["n"] = t3["n"].fillna(1).astype(np.int64)
    first_group = t2.groupby("episode_id").agg(date=("date", "first"), t_start=("t_start", "min"), t_end=("t_end", "max"),
                                               n_max=("n", "max"), n_groups=("group_id", "size"))
    flights_of = t2.explode("members").groupby("episode_id")["members"].agg(lambda m: len(set(m)))
    t4 = first_group.assign(n_flights=flights_of).reset_index()
    t4["block"] = block[[m[0] for m in t2.groupby("episode_id")["members"].first()]]
    t5 = summary(flights, t2, t3, block)
    t1 = kept[["date", "a", "b", "t_on", "t_off", "duration", "ell_med", "d_dec"]].reset_index(drop=True)
    return t1, t2, t3, t4, t5, t6


def summary(flights, t2, t3, block):
    """T5: per flight the fraction of its time after departure spent in a group (phi_G), the
    number of distinct companions (kappa), of entries into a group, of episodes; its block;
    and whether it is local."""
    after = t3[t3["state"] != "L"]
    span = (after["t_end"] - after["t_start"]).groupby(after["flight"]).sum()
    g = after[after["state"] == "G"]
    in_group = (g["t_end"] - g["t_start"]).groupby(g["flight"]).sum()
    # An entry is a G stretch that does not follow another G stretch of the same flight.
    previous = t3.groupby("flight")["state"].shift()
    entries = t3[(t3["state"] == "G") & (previous != "G")].groupby("flight").size()
    n_episodes = g.groupby("flight")["episode_id"].nunique()
    pairs = t2.explode("members")[["group_id", "members"]].rename(columns={"members": "flight"})
    pairs = pairs.merge(pairs, on="group_id")
    pairs = pairs[pairs["flight_x"] != pairs["flight_y"]]
    kappa = pairs.groupby("flight_x")["flight_y"].nunique()
    out = pd.DataFrame({"flight": flights["flight"].to_numpy()})
    out["phi_G"] = (in_group.reindex(out["flight"]).fillna(0) / span.reindex(out["flight"])).to_numpy()
    out["kappa"] = kappa.reindex(out["flight"]).fillna(0).astype(int).to_numpy()
    out["n_entries"] = entries.reindex(out["flight"]).fillna(0).astype(int).to_numpy()
    out["n_episodes"] = n_episodes.reindex(out["flight"]).fillna(0).astype(int).to_numpy()
    out["block"] = block
    out["local"] = ~np.isfinite(flights["departure"].to_numpy())
    return out


def write(out, flights, t1, t2, t3, t4, t5, t6, calibration):
    flights.drop(columns=["X0", "Y0", "Z0"]).to_parquet(out / "flights.parquet", index=False)
    for name, t in [("pair_intervals", t1), ("groups", t2), ("membership", t3), ("episodes", t4),
                    ("flight_summary", t5), ("configurations", t6)]:
        t.to_parquet(out / f"{name}.parquet", index=False)
    (out / "calibration.json").write_text(json.dumps({"params": PARAMS, **calibration}, indent=1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    parser.add_argument("--sample", type=int, help="a random sample of dates (seed 0), for trials")
    parser.add_argument("--dates", help="comma-separated dates YYYY-MM-DD")
    parser.add_argument("--out", type=Path, help="default: the folder of the run on SSD_DISANTE")
    parser.add_argument("--set", default="", help="parameters to change, e.g. eps_h=200,delta=120")
    parser.add_argument("--days", type=Path, help="reuse the days/ of a run that differs only in q")
    args = parser.parse_args()
    for item in filter(None, args.set.split(",")):
        key, value = item.split("=")
        PARAMS[key] = type(PARAMS[key])(float(value))
    PARAMS["r_min"] = 2 * PARAMS["delta"] + PARAMS["step"]
    if args.out is None:
        trial = f"sample{args.sample}" if args.sample else "dates" if args.dates else None
        args.out = DATA / ("trials/" + trial if trial else "runs") / run_name(PARAMS)
    days = args.out / "days"
    days.mkdir(parents=True, exist_ok=True)
    saved = days / "params.json"
    if saved.exists() and json.loads(saved.read_text()) != PARAMS:
        raise SystemExit(f"{days} was computed with other parameters: choose another --out")
    saved.write_text(json.dumps(PARAMS))
    if args.days:  # q changes only tau* and Phase 4, never the per-date results
        other = json.loads((args.days / "params.json").read_text())
        if {k: v for k, v in other.items() if k != "q"} != {k: v for k, v in PARAMS.items() if k != "q"}:
            raise SystemExit(f"{args.days} differs from these parameters in more than q")
        for cached in args.days.glob("*.pkl"):
            if not cached.name.startswith(".") and not (days / cached.name).exists():
                shutil.copy2(cached, days / cached.name)
    archive = grid.archive()
    every = sorted(archive["date"].unique())
    if args.dates:
        dates = sorted(args.dates.split(","))
    elif args.sample:
        dates = sorted(np.random.default_rng(0).choice(every, args.sample, replace=False))
    else:
        dates = every
    describe(args.out, PARAMS, len(dates))
    compute(archive, dates, days, args.jobs)
    flights, intervals, null = load(days, dates)
    intervals.to_parquet(args.out / "intervals_raw.parquet", index=False)
    calibration = calibrate(intervals, null, PARAMS)
    print(f"tau* = {calibration['tau_star']} s from {calibration['null_intervals']} surrogate intervals", flush=True)
    tables = label(flights, intervals, calibration["tau_star"] if calibration["tau_star"] is not None else TAUS[-1])
    write(args.out, flights, *tables, calibration)
    describe(args.out, PARAMS, len(dates), calibration)
    t5 = tables[4]
    print(f"{len(flights)} flights, {len(tables[1])} groups, {len(tables[5])} configurations, {len(tables[3])} episodes; "
          f"flights ever in a group: {(t5['n_entries'] > 0).sum()}", flush=True)
