"""Label every flight logged at up to 5 s with the refitted model and write the runs in the layout of
xc-viewer-data/<discipline>/derived/segmentation/vilpellet/ (phase_segments.parquet, phase_coverage.parquet),
into the new folder segmentation/refit/.

Flights go through the xc-dynamics pre-processing (steps 1 to 9); segment_id is its segment number. Times are in
the time base of the viewer's fixes.parquet: seconds from ground_phase_start_s of its flights_meta.parquet (the
take-off of the old cleaning); a flight the viewer does not hold is counted from the first fix kept by our
take-off and landing rule. Our first fixes can lie a few seconds before the viewer's, so t can be negative. Each segment is cut into pieces at gaps longer than
max(4 s, 2 x interval); a piece is put on a 1-s grid (linear at 1 Hz, cubic spline above), decoded as in
phases.py with the model of out/refit_params_<discipline>.json, and the real fixes take the labels. A flight
logged at more than 5 s is listed in the coverage and stays unlabelled. Work goes in chunks of flights, kept
in CACHE, so a run can be stopped and resumed.

    uv run python methods/phase-labelling/label.py para|hang [--sample N] [--workers W] [--time-only]
"""

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import phases  # noqa: E402
import refit  # noqa: E402

CHUNK = 2000  # flights per chunk
SAMPLE_SEED = 2026
RUNS_SCHEMA = pa.schema([("source", pa.large_string()), ("flight_id", pa.large_string()), ("segment_id", pa.int64()),
                         ("phase", pa.large_string()), ("t_start", pa.float64()), ("t_end", pa.float64()),
                         ("duration_s", pa.float64()), ("n_fixes", pa.int64()), ("left_censored", pa.bool_()),
                         ("right_censored", pa.bool_())])
COVERAGE_SCHEMA = pa.schema([("source", pa.large_string()), ("flight_id", pa.large_string()), ("segment_id", pa.int64()),
                             ("t_start", pa.float64()), ("t_end", pa.float64()), ("n_native_fixes", pa.int64()),
                             ("native_cadence_s", pa.float64()), ("n_classified_fixes", pa.int64()), ("status", pa.large_string())])
STATUS = dict(classified="classified", slow="cadence_above_gate", short="segment_shorter_than_persistence_window")

_STATE = {}


def init(disc):
    model, alpha, _ = refit.load_params(disc)
    meta = pd.read_parquet(refit.DISCIPLINES[disc]["tracks"] / "derived/flights_meta.parquet", columns=["flight_id", "ground_phase_start_s"])
    offset = pd.Series(meta.ground_phase_start_s.to_numpy(), index=meta.flight_id.astype(int)).dropna()
    _STATE.update(disc=disc, model=model, alpha=alpha, cfg=refit.config(), offset=offset.to_dict())


def runs_of(state, piece):
    """Runs of one segment: (first index, last index, state, left censored, right censored). A run ends at a
    change of state or piece; it is censored on the side where it meets the end of its piece."""
    n = len(state)
    piece_start = np.r_[True, piece[1:] != piece[:-1]]
    first = np.flatnonzero(piece_start | np.r_[True, state[1:] != state[:-1]])
    last = np.r_[first[1:] - 1, n - 1]
    keep = state[first] >= 0
    first, last = first[keep], last[keep]
    piece_end = np.r_[piece_start[1:], True]
    return first, last, state[first], piece_start[first], piece_end[last]


def label_flight(item):
    """One flight, given as (flight id, track path): its run rows and coverage rows as lists of tuples."""
    fid, path = item
    disc = _STATE["disc"]
    source = refit.DISCIPLINES[disc]["source"]
    flight, row = refit.read_flight(path, disc, _STATE["cfg"])
    if flight is None:
        return fid, [], [], "dropped"
    k = float(row["sampling_s"])
    own = fid not in _STATE["offset"]
    shift = row["t_takeoff"] if own else _STATE["offset"][fid]
    runs, coverage = [], []
    for seg, t, xyz in refit.segments(flight):
        dt = np.diff(t)
        cadence = float(np.median(dt[dt > 0])) if (dt > 0).any() else np.nan
        if k > refit.MAX_INTERVAL_S:
            state, piece = np.full(len(t), -1), np.full(len(t), -1)
        else:
            state, piece = refit.label_segment(t, xyz, k, _STATE["alpha"], _STATE["model"])
        n_labelled = int((state >= 0).sum())
        status = STATUS["slow"] if k > refit.MAX_INTERVAL_S else STATUS["classified"] if n_labelled else STATUS["short"]
        coverage.append((source, str(fid), seg, float(t[0] - shift), float(t[-1] - shift), len(t), cadence, n_labelled, status))
        if n_labelled:
            first, last, st, left, right = runs_of(state, piece)
            for a, b, s, lc, rc in zip(first, last, st, left, right):
                runs.append((source, str(fid), seg, phases.STATES[s], float(t[a] - shift), float(t[b] - shift), float(t[b] - t[a]), int(b - a + 1),
                             bool(lc), bool(rc)))
    return fid, runs, coverage, "own_time_base" if own else "ok"


def flight_list(disc):
    """(flight id, track path) of the flights that pass the pre-processing (the sampling check included)."""
    root = refit.DISCIPLINES[disc]["root"]
    t = pd.read_csv(root / "derived/preprocessing/airborne/flights.csv")
    ok = t.gnss_ok.astype(bool) & t.speed_ok.astype(bool) & t.airborne_ok.fillna(False).astype(bool) & t.sampling_ok.fillna(False).astype(bool)
    cat = refit.catalogue(disc).set_index("flight_id").local_path
    ids = np.sort(t[ok].flight_id.to_numpy())
    return [(int(f), refit.DISCIPLINES[disc]["tracks"] / cat[f]) for f in ids], t


def write_text(path, text):
    """Write a small file atomically."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def write_table(rows, schema, path):
    columns = list(zip(*rows)) if rows else [[] for _ in schema]
    table = pa.table({f.name: pa.array(c, type=f.type) for f, c in zip(schema, columns)}, schema=schema)
    tmp = path.with_suffix(".tmp")
    pq.write_table(table, tmp, compression="zstd")
    tmp.replace(path)


def merge(shards, name, schema, target):
    """Concatenate the chunk files into one parquet file, one row group per chunk."""
    tmp = target.with_suffix(".tmp")
    with pq.ParquetWriter(tmp, schema, compression="zstd") as w:
        for s in shards:
            w.write_table(pq.read_table(s / name, schema=schema))
    tmp.replace(target)


def parameters_md(disc, record, n_flights, n_labelled, own_time_base, started, seconds):
    t = record["training"]
    best = next(r for r in record["starts"] if r["start"] == record["best_start"])
    return f"""# Refit of Vilpellet's phase model, {refit.DISCIPLINES[disc]['folder']}

Written by `xc-lab/methods/phase-labelling/label.py` on {date.today()}. Same file layout as `../vilpellet/`.
Not registered in `manifest.json`.

- **Model.** Three-state HMM of Vilpellet (three binary features over 30 fixes, Viterbi, 90-fix majority vote),
  parameters estimated again by Baum-Welch on our flights, transition matrix updated. States in the order
  climb, search, transition. Parameters in `model/parameters.json`.
- **Used for this discipline.** `{record['used']}`{': ' + record['why'] if 'why' in record else ''}. Straightness
  threshold {record['alpha_rad']} rad per fix.
- **Training.** {t['train_flights']} flights logged at 1 Hz (mean step below {refit.MAX_MEAN_STEP_S} s, at least {refit.MIN_FIXES} fixes),
  {t['train_fixes']} fixes, drawn at random over years and ground strata (seed {record['settings']['seed']}); {t['held_out_flights']}
  held-out flights ({t['held_out_fixes']} fixes). {record['settings']['random_starts']} random starts, {record['settings']['perturbed_starts']} starts perturbed from his parameters and his parameters as one start. Kept: the best held-out likelihood among the starts whose states keep his meaning, start `{record['best_start']}` ({len(record['starts_near_best'])} starts reach it):
  {best['iterations']} iterations, converged {best['converged']} (relative change of the log-likelihood below {record['settings']['rel_tol']}, at most {record['settings']['max_iter']} iterations).
  Held-out log-likelihood {record['held_out_log_likelihood']['refit']:.1f} (his parameters: {record['held_out_log_likelihood']['his']:.1f}).
- **Input.** The tracks of `xc-viewer-data/{refit.DISCIPLINES[disc]['folder']}/raw/igc` through the xc-dynamics pre-processing
  (steps 1 to 9, `configs/preprocessing.yaml`). `segment_id` is the segment number of that pre-processing, not that of
  `derived/fixes.parquet`. `t_start` and `t_end` are in the time base of `derived/fixes.parquet`: seconds from `ground_phase_start_s`
  of `derived/flights_meta.parquet`; {own_time_base} flights that the viewer does not hold are counted from the first fix our take-off
  and landing rule keeps. The first fixes of a flight can lie before the viewer's, so a time can be negative. `duration_s` is `t_end - t_start`.
- **Cadence rule.** Flights logged every 1 to {refit.MAX_INTERVAL_S:g} s are labelled. A segment with a median step of 1 s is decoded directly
  (gaps up to {refit.MIN_GAP_S:g} s filled linearly), a segment with a longer step is put on a 1-s grid with a cubic spline first. A gap longer than
  max({refit.MIN_GAP_S:g} s, {refit.GAP_STEPS} x interval) ends a piece; a piece of fewer than {phases.WINDOW} s stays unlabelled. Only real fixes carry labels.
  Flights logged at more than {refit.MAX_INTERVAL_S:g} s are in `phase_coverage.parquet` with status `cadence_above_gate`.
  No minimum number of fixes per flight. No selection of runs (tail, short runs, searches).
- **Files.** `phase_segments.parquet`: one row per run of one phase; a run ends at a change of phase or at the end of its piece (a gap
  longer than the rule above, or the end of the segment) and is censored on the side where it meets such an end.
  `phase_coverage.parquet`: one row per segment.
- **Size.** {n_flights} flights processed, {n_labelled} with at least one label; {seconds / 3600:.2f} h of wall time on {refit.WORKERS} workers (sum of the chunks), started {started}.
- **Code.** xc-lab `{record['xc_lab']['commit']}`{' (uncommitted changes)' if record['xc_lab']['uncommitted_changes'] else ''},
  xc-dynamics `{record['xc_dynamics']['commit']}` (training run); parameters `out/refit_params_{disc}.json`.
"""


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("discipline", choices=list(refit.DISCIPLINES))
    p.add_argument("--sample", type=int, help="label this many flights drawn at random, into the scratch folder")
    p.add_argument("--workers", type=int, default=refit.WORKERS)
    p.add_argument("--time-only", action="store_true", help="with --sample: do not merge, only report the time")
    args = p.parse_args()
    disc = args.discipline
    flights, _ = flight_list(disc)
    if args.sample:
        pick = np.random.default_rng(SAMPLE_SEED).choice(len(flights), size=min(args.sample, len(flights)), replace=False)
        flights = [flights[i] for i in sorted(pick)]
    work = refit.CACHE / ("scratch" if args.sample else "chunks") / disc
    work.mkdir(parents=True, exist_ok=True)
    chunks = [flights[i:i + CHUNK] for i in range(0, len(flights), CHUNK)]
    started, t0 = time.strftime("%Y-%m-%d %H:%M"), time.time()
    done = 0
    with ProcessPoolExecutor(args.workers, initializer=init, initargs=(disc,)) as pool:
        for c, chunk in enumerate(chunks):
            shard = work / f"{c:04d}"
            if (shard / "ok").exists():
                done += len(chunk)
                print(f"{disc}: chunk {c + 1}/{len(chunks)} already done", flush=True)
                continue
            shard.mkdir(exist_ok=True)
            t_chunk = time.time()
            runs, coverage, dropped, own = [], [], 0, 0
            for _, r, cov, status in pool.map(label_flight, chunk, chunksize=10):
                runs += r
                coverage += cov
                dropped += status == "dropped"
                own += status == "own_time_base"
            write_table(runs, RUNS_SCHEMA, shard / "runs.parquet")
            write_table(coverage, COVERAGE_SCHEMA, shard / "coverage.parquet")
            write_text(shard / "ok", json.dumps(dict(flights=len(chunk), runs=len(runs), dropped=dropped, own_time_base=own, seconds=time.time() - t_chunk)))
            done += len(chunk)
            elapsed = time.time() - t0
            print(f"{disc}: chunk {c + 1}/{len(chunks)}, {done}/{len(flights)} flights, {len(runs)} runs, "
                  f"{elapsed / 60:.1f} min, about {elapsed / done * (len(flights) - done) / 60:.0f} min left", flush=True)
    seconds = time.time() - t0
    if args.sample and args.time_only:
        print(f"{disc}: {len(flights)} flights in {seconds:.0f} s on {args.workers} workers: {seconds / len(flights):.3f} s per flight")
        return
    shards = sorted(work.glob("[0-9]*"))
    done_info = [json.loads((s_ / "ok").read_text()) for s_ in shards]
    seconds, own_time_base = sum(d["seconds"] for d in done_info), sum(d["own_time_base"] for d in done_info)
    target = work / "merged" if args.sample else refit.DISCIPLINES[disc]["tracks"] / "derived/segmentation/refit"
    (target / "model").mkdir(parents=True, exist_ok=True)
    merge(shards, "runs.parquet", RUNS_SCHEMA, target / "phase_segments.parquet")
    merge(shards, "coverage.parquet", COVERAGE_SCHEMA, target / "phase_coverage.parquet")
    record = json.loads((refit.OUT / f"refit_params_{disc}.json").read_text())
    write_text(target / "model/parameters.json", (refit.OUT / f"refit_params_{disc}.json").read_text())
    n_labelled = int(pq.read_table(target / "phase_coverage.parquet", columns=["flight_id", "n_classified_fixes"]).to_pandas()
                     .groupby("flight_id").n_classified_fixes.sum().gt(0).sum())
    write_text(target / "PARAMETERS.md", parameters_md(disc, record, len(flights), n_labelled, own_time_base, started, seconds))
    print(f"{disc}: written to {target}")


if __name__ == "__main__":
    main()
