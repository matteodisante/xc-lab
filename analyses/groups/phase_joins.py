"""Where in the cycle pilots join a group and leave it: the phase of the member at the instant it enters group time and at
the instant it leaves it, against what random times would give.

Populations as in phase_gaps.py (declared open paragliders, logger interval up to 5 s, `--dt5`, the runs of the label set refit
under the rule `context` of phase_terms.py, a search of any kind being a search). An entry is the first second of a stretch of
group time (tamsd.stretches, the grouping of the run) that follows a stretch alone of the same flight, an exit its last second
when a stretch alone follows; group time that starts at the departure or ends at the landing is counted apart. The phase at that
instant is that of the run holding it; an instant inside no run has none and is counted apart.

Baseline: if the instants were random in the time alone of the flight (A time, the stretches alone of the flights with a group),
the events of a flight would fall in each phase as its own A time does, so the baseline share of a phase is the share of the
events expected there, summed over the flights: sum_f n_f a_fk / sum_f n_f, with n_f the events of flight f and a_fk the share
of its A time in phase k. The share observed is sum_f n_fk / sum_f n_f; the ratio is observed over baseline. Per ground class and
on every flight; bands are those of twoway.py, draws of the cells of the flights shared with phase_gaps.py (a ratio is taken
draw by draw).

    uv run python phase_joins.py --declared-open --dt5 [--draws N]    # writes phase_joins.json, about 1 min
"""

import json
import sys

import numpy as np
import pandas as pd

import isolated
import longlags
import phase_gaps
import phase_terms
from blocks import with_takeoffs
from phase_gaps import CLASS_KEYS, DRAWS, rows_of
from tamsd import LABELS, read_flights, stretches

PHASES = ("glide", "search", "climb")
NAME = "phase_joins"


def phase_codes(runs):
    """The code of PHASES of each run of the rule `context` (a search of any kind is a search)."""
    return runs["phase"].map({"transition": 0, "climb": 2}).fillna(1).astype(int).to_numpy()


def events(spans):
    """The entries and exits of group time in `spans` (tamsd.stretches): a frame (flight, kind, t) on the flight clock, kind
    "join" or "leave" for those inside the flight, "join at departure" and "leave at landing" for the others."""
    s = spans.sort_values(["flight", "a"], kind="stable")
    group = (s["population"] == "group").to_numpy()
    before, after = (s.groupby("flight")["population"].shift(k).notna().to_numpy() for k in (1, -1))
    parts = [(group & before, "join", "a", 0), (group & ~before, "join at departure", "a", 0),
             (group & after, "leave", "b", -1), (group & ~after, "leave at landing", "b", -1)]
    return pd.concat([pd.DataFrame({"flight": s["flight"].to_numpy()[m], "kind": kind, "t": s[c].to_numpy()[m] + shift})
                      for m, kind, c, shift in parts], ignore_index=True)


def phase_at(runs, flight, t):
    """The code of PHASES of the run of `flight` that holds the instant `t` (the run lasts until its last fix plus the logger
    interval), -1 where none does; arrays of the same length."""
    key = runs["flight"].to_numpy(float) * isolated.RANK + runs["t_start"].to_numpy(float)
    end = runs["flight"].to_numpy(float) * isolated.RANK + runs["t_end"].to_numpy(float) + runs["dt"].to_numpy(float)
    i = np.maximum(np.searchsorted(key, flight * isolated.RANK + t, "right") - 1, 0)
    return np.where((key[i] <= flight * isolated.RANK + t) & (flight * isolated.RANK + t < end[i]), phase_codes(runs)[i], -1)


def time_in(runs, flight, a, b):
    """The seconds spent in each phase of PHASES between a and b (arrays) of `flight`: (len(a), 3)."""
    key = runs["flight"].to_numpy(float) * isolated.RANK + runs["t_start"].to_numpy(float)
    end = runs["flight"].to_numpy(float) * isolated.RANK + runs["t_end"].to_numpy(float) + runs["dt"].to_numpy(float)
    code = phase_codes(runs)
    cum = np.vstack([np.r_[0.0, np.cumsum((end - key) * (code == k))] for k in range(3)])

    def upto(t):  # the time of each phase before t, of the flight of the interval
        x = flight * isolated.RANK + t
        i = np.searchsorted(key, x, "right")
        last = np.maximum(i - 1, 0)
        cut = np.where(i > 0, np.maximum(end[last] - x, 0.0), 0.0)
        return np.column_stack([cum[k][i] - cut * (code[last] == k) for k in range(3)])
    return upto(b) - upto(a)


def weighted(counts, values, step=250):
    """The sums of `values` (cells, m) over the cells in the point estimate (row 0) and in each draw (a row of `counts`,
    the count of each cell): (1 + rows of counts, m)."""
    return np.vstack([values.sum(axis=0), *[counts[a:a + step] @ values for a in range(0, len(counts), step)]])


def terms(runs, spans, cell):
    """Per flight with an entry or an exit: its cell, and for each of "join" and "leave" the events by phase n_fk and the
    events expected by phase n_f a_fk (arrays (flights, 3)), as {kind: (observed, expected)}, and the flights."""
    ev = events(spans)
    ev["phase"] = phase_at(runs, ev["flight"].to_numpy(float), ev["t"].to_numpy(float))
    alone = spans[spans["population"] == "group_solo"]
    seconds = pd.DataFrame(time_in(runs, alone["flight"].to_numpy(float), alone["a"].to_numpy(float), alone["b"].to_numpy(float)),
                           columns=PHASES).groupby(alone["flight"].to_numpy()).sum()
    flights = np.unique(ev.loc[ev["kind"].isin(("join", "leave")), "flight"])
    share = seconds.div(seconds.sum(axis=1).replace(0, np.nan), axis=0).reindex(flights)
    out = {}
    for kind in ("join", "leave"):
        e = ev[(ev["kind"] == kind) & (ev["phase"] >= 0)]
        observed = np.zeros((len(flights), 3))
        np.add.at(observed, (np.searchsorted(flights, e["flight"].to_numpy()), e["phase"].to_numpy()), 1.0)
        out[kind] = observed, observed.sum(axis=1, keepdims=True) * share.to_numpy()
    counts = {k: int((ev["kind"] == k).sum()) for k in ("join", "leave", "join at departure", "leave at landing")}
    counts["without a phase"] = int(((ev["phase"] < 0) & ev["kind"].isin(("join", "leave"))).sum())
    return flights, out, counts


def statistics(observed, expected, cell, counts):
    """The share of the events in each phase, the share expected by chance and their ratio, as {phase: {share, baseline, ratio}}
    of {value, band}, from the events of flights (rows of `observed` and `expected`, in the cells `cell`)."""
    n = counts.shape[1]
    by_cell = []
    for x in (observed, expected):
        total = np.zeros((n, 3))
        np.add.at(total, cell, np.nan_to_num(x))
        by_cell.append(weighted(counts, total))
    share, base = (x / x.sum(axis=1, keepdims=True) for x in by_cell)
    return {p: {"share": rows_of(share[:, k]), "baseline": rows_of(base[:, k]), "ratio": rows_of(share[:, k] / base[:, k])}
            for k, p in enumerate(PHASES)}


if __name__ == "__main__":
    _, code, counts, classes = phase_gaps.context()
    kept = phase_terms.kept_flights()
    runs = phase_terms.processed("refit", kept).sort_values(["flight", "t_start"], kind="stable", ignore_index=True)
    flights = with_takeoffs(read_flights())
    spans = stretches(flights, pd.read_parquet(LABELS / "membership.parquet"), pd.read_parquet(LABELS / "flight_summary.parquet"))
    spans = spans[spans["flight"].isin(kept["flight"])]
    fl, got, n = terms(runs, spans, code)
    cell = code.reindex(fl).to_numpy()
    where = classes.reindex(fl).fillna("unknown").to_numpy()
    result = {"labels": phase_terms.SET, "rule": phase_terms.RULE, "draws": DRAWS, "classes": CLASS_KEYS, "phases": PHASES, "events": n,
              "baseline": "share of the A time of the flight in the phase, weighted by the events of the flight"}
    for key in CLASS_KEYS:
        m = np.ones(len(fl), bool) if key == "all" else where == key
        result.setdefault("flights", {})[key] = int(m.sum())
        result.setdefault("events by class", {})[key] = {kind: int(got[kind][0][m].sum()) for kind in ("join", "leave")}
        for kind in ("join", "leave"):
            observed, expected = got[kind]
            result.setdefault(kind, {})[key] = statistics(observed[m], expected[m], cell[m], counts)
        print(key, int(m.sum()), "flights", {k: {p: round(v["share"]["value"], 3) for p, v in r[key].items()} for k, r in result.items() if k in ("join", "leave")}, flush=True)
    longlags.RESULTS.mkdir(parents=True, exist_ok=True)
    (longlags.RESULTS / f"{NAME}.json").write_text(json.dumps(result, indent=1))
