"""Agreement of the refit labels with Vilpellet's, on the flights that both label.

Both label sets are runs (flight, first and last second, phase), so the comparison counts seconds on which the
two runs overlap: every common flight is logged at 1 Hz, so a second is a fix up to the few seconds missing from
a track. Writes out/refit_agreement.json.

    uv run python methods/phase-labelling/agreement.py [--baseline N]

--baseline N also labels N flights per discipline with his own parameters through our pre-processing, to split
the disagreement into the effect of the input (his cleaned fixes against ours) and that of the model.
"""

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import phases  # noqa: E402
import refit  # noqa: E402

KEY = 1 << 21  # seconds per flight in the sort key; flight * KEY + t
CHUNK = 4000  # flights per chunk
CLASSES = ("plains", "hills", "low mountains", "high mountains", "unknown", "not in the ground table")
QUANTILES = (0.05, 0.1, 0.25, 0.5, 0.75, 0.9)
BASELINE_SEED = 2026


def folder(disc, name):
    return refit.DISCIPLINES[disc]["tracks"] / "derived/segmentation" / name


def load_runs(path):
    """Runs sorted by flight and start: dict of flight (int), phase (0 climb, 1 search, 2 transition), ts, te, n."""
    t = pq.read_table(path, columns=["flight_id", "phase", "t_start", "t_end", "n_fixes"])
    flight = pc.cast(t["flight_id"], pa.int64()).to_numpy()
    runs = dict(flight=flight, phase=pc.index_in(t["phase"], value_set=pa.array(phases.STATES)).to_numpy().astype(np.int8),
                ts=t["t_start"].to_numpy().astype(np.int64), te=t["t_end"].to_numpy().astype(np.int64), n=t["n_fixes"].to_numpy())
    order = np.lexsort((runs["ts"], flight))
    return {k: v[order] for k, v in runs.items()}


def restrict(runs, flights):
    keep = np.isin(runs["flight"], flights)
    return {k: v[keep] for k, v in runs.items()}


def ground_class(disc, flights):
    """The ground class of each flight (paragliders), as in CLASSES; one class for all hang gliders."""
    if disc != "para":
        return np.full(len(flights), "all hang gliders", dtype=object)
    ground = pd.read_parquet(refit.GROUND, columns=["flight", "class"])
    ids = pd.read_parquet(refit.GROUND_FLIGHTS, columns=["flight", "flight_id"]).merge(ground, on="flight")
    ids["flight_id"] = ids.flight_id.astype(np.int64)
    cls = pd.Series(flights).map(ids.drop_duplicates("flight_id").set_index("flight_id")["class"])
    return cls.fillna("not in the ground table").to_numpy(dtype=object)


def overlap_chunk(a, b, flights):
    """Per flight of the chunk: the 3x3 seconds (rows a, columns b) where both label, flattened."""
    ka0, ka1 = a["flight"] * KEY + a["ts"], a["flight"] * KEY + a["te"] + 1
    kb0, kb1 = b["flight"] * KEY + b["ts"], b["flight"] * KEY + b["te"] + 1
    bounds = np.unique(np.concatenate([ka0, ka1, kb0, kb1]))
    start, length = bounds[:-1], np.diff(bounds)

    def label_at(k0, k1, phase):
        i = np.searchsorted(k0, start, side="right") - 1
        ok = (i >= 0) & (start < k1[np.maximum(i, 0)])
        return np.where(ok, phase[np.maximum(i, 0)], -1)

    la, lb = label_at(ka0, ka1, a["phase"]), label_at(kb0, kb1, b["phase"])
    both = (la >= 0) & (lb >= 0)
    fi = np.searchsorted(flights, start[both] // KEY)
    cell = fi * 9 + la[both] * 3 + lb[both]
    return np.bincount(cell, weights=length[both], minlength=len(flights) * 9).reshape(len(flights), 9)


def confusions(a, b, flights):
    """Seconds per flight (rows of flights) and cell (a state * 3 + b state)."""
    out = np.zeros((len(flights), 9))
    edges = np.searchsorted(flights, flights[::CHUNK])
    for lo in edges:
        part = flights[lo:lo + CHUNK]
        rows = lambda r: {k: v[np.searchsorted(r["flight"], part[0]):np.searchsorted(r["flight"], part[-1], side="right")]
                          for k, v in r.items()}
        out[lo:lo + len(part)] = overlap_chunk(rows(a), rows(b), part)
    return out


def summarise(conf):
    """Agreement numbers of a (flights x 9) table of seconds."""
    total = conf.sum(axis=0).reshape(3, 3)
    flight = conf.reshape(-1, 3, 3)
    same = np.trace(flight, axis1=1, axis2=2)
    n = flight.sum(axis=(1, 2))
    per = same[n > 0] / n[n > 0]
    names = phases.STATES
    return dict(flights=int((n > 0).sum()), seconds=int(total.sum()), same_label=float(np.trace(total) / total.sum()),
                confusion_seconds_rows_his_columns_refit=total.astype(int).tolist(), states=list(names),
                his_state_kept={names[i]: float(total[i, i] / total[i].sum()) for i in range(3)},
                refit_state_in_his={names[j]: float(total[j, j] / total[:, j].sum()) for j in range(3)},
                time_share_his={names[i]: float(total[i].sum() / total.sum()) for i in range(3)},
                time_share_refit={names[j]: float(total[:, j].sum() / total.sum()) for j in range(3)},
                per_flight_agreement=dict(mean=float(per.mean()), **{f"q{int(q * 100):02d}": float(np.quantile(per, q)) for q in QUANTILES}))


def statistics(runs, flights):
    """Phase statistics of the flights of one label set: pooled and per-flight time shares, climbs per hour,
    median run length in fixes, per phase."""
    fi = np.searchsorted(flights, runs["flight"])
    cell = fi * 3 + runs["phase"]
    fixes = np.bincount(cell, weights=runs["n"], minlength=len(flights) * 3).reshape(-1, 3)
    count = np.bincount(cell, minlength=len(flights) * 3).reshape(-1, 3)
    hours = fixes.sum(axis=1) / 3600
    ok = hours > 0
    names = phases.STATES
    out = dict(flights=int(ok.sum()), hours=float(hours.sum()),
               time_share_pooled={names[i]: float(fixes[:, i].sum() / fixes.sum()) for i in range(3)},
               time_share_flight_mean={names[i]: float((fixes[ok, i] / fixes[ok].sum(axis=1)).mean()) for i in range(3)},
               runs_per_hour_pooled={names[i]: float(count[:, i].sum() / hours.sum()) for i in range(3)},
               runs_per_hour_flight_median={names[i]: float(np.median(count[ok, i] / hours[ok])) for i in range(3)},
               median_run_fixes={names[i]: float(np.median(runs["n"][runs["phase"] == i])) for i in range(3)})
    return out


def compare(his, ours, flights):
    """Statistics of both sets on `flights`, and the change refit minus his."""
    a, b = statistics(restrict(his, flights), flights), statistics(restrict(ours, flights), flights)
    change = {k: {s: b[k][s] - a[k][s] for s in b[k]} for k in b if isinstance(b[k], dict)}
    return dict(his=a, refit=b, change=change)


# ---- his parameters through our pre-processing -------------------------------------------------

def baseline_item(args):
    """Per fix of one flight: the labels of his viewer, of his parameters on our input, and of the refit."""
    path, disc, models, ts, te, phase, shift = args
    flight, _ = refit.read_flight(path, disc, refit.config())
    alpha, conf = phases.ALPHA[disc], np.zeros((3, 3, 3), dtype=int)  # [pair][row][column]
    for _, t, xyz in refit.segments(flight):
        k = 1
        mine = {name: refit.label_segment(t, xyz, k, alpha, m)[0] for name, m in models.items()}
        tv = t - shift  # the viewer's time base
        i = np.searchsorted(ts, tv, side="right") - 1
        viewer = np.where((i >= 0) & (tv <= te[np.maximum(i, 0)]), phase[np.maximum(i, 0)], -1)
        for p, (x, y) in enumerate(((viewer, mine["his"]), (mine["his"], mine["refit"]), (viewer, mine["refit"]))):
            ok = (x >= 0) & (y >= 0)
            np.add.at(conf[p], (x[ok], y[ok]), 1)
    return conf


def baseline(disc, his, flights, n):
    model, _, _ = refit.load_params(disc)
    models = {"his": refit.HIS, "refit": model}
    rng = np.random.default_rng(BASELINE_SEED)
    pick = np.sort(rng.choice(flights, size=min(n, len(flights)), replace=False))
    cat = refit.catalogue(disc).set_index("flight_id").local_path
    tracks = refit.DISCIPLINES[disc]["tracks"]
    lo, hi = np.searchsorted(his["flight"], pick), np.searchsorted(his["flight"], pick, side="right")
    meta = pd.read_parquet(tracks / "derived/flights_meta.parquet", columns=["flight_id", "ground_phase_start_s"])
    shift = pd.Series(meta.ground_phase_start_s.to_numpy(), index=meta.flight_id.astype(int))
    items = [(tracks / cat[int(f)], disc, models, his["ts"][a:b], his["te"][a:b], his["phase"][a:b], shift[int(f)])
             for f, a, b in zip(pick, lo, hi)]
    with ProcessPoolExecutor(refit.WORKERS) as pool:
        conf = sum(pool.map(baseline_item, items, chunksize=4))
    names = ("his_viewer_vs_his_parameters_on_our_input", "his_parameters_vs_refit_on_our_input", "his_viewer_vs_refit")
    return dict(flights=len(pick), **{n_: dict(same_label=float(np.trace(c) / c.sum()), fixes=int(c.sum()), confusion=c.tolist())
                                       for n_, c in zip(names, conf)})


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--baseline", type=int, default=0, help="flights per discipline for the baseline")
    p.add_argument("--discipline", nargs="*", default=list(refit.DISCIPLINES))
    args = p.parse_args()
    path = refit.OUT / "refit_agreement.json"
    result = json.loads(path.read_text()) if path.exists() else {}
    for disc in args.discipline:
        his, ours = load_runs(folder(disc, "vilpellet") / "phase_segments.parquet"), load_runs(folder(disc, "refit") / "phase_segments.parquet")
        flights = np.intersect1d(np.unique(his["flight"]), np.unique(ours["flight"]))
        his, ours = restrict(his, flights), restrict(ours, flights)
        conf = confusions(his, ours, flights)
        cls = ground_class(disc, flights)
        res = dict(common_flights=len(flights), his_flights=int(len(np.unique(his["flight"]))), overall=summarise(conf),
                   statistics=compare(his, ours, flights), per_class={})
        for c in sorted(set(cls)):
            m = cls == c
            res["per_class"][c] = dict(summarise(conf[m]), statistics=compare(his, ours, flights[m]))
        if args.baseline:
            res["baseline"] = baseline(disc, his, flights, args.baseline)
        result[disc] = res
        path.write_text(json.dumps(result, indent=1))
        print(disc, json.dumps({k: res["overall"][k] for k in ("flights", "same_label", "his_state_kept")}))


if __name__ == "__main__":
    main()
