"""Context of the search runs, for his labels and the refit, on the paragliders both label.

A run is adjacent to the next one of its flight when they share the segment and less than MAX_GAP s lie between
them; same-phase neighbours are merged first. Each search run gets its previous run (climb, transition, start)
and next run (climb, transition, end). Statistics are given as labelled and with the climb->search->transition
runs ("exit-like") counted as transition. Writes out/search_context.json and out/search_context.pdf.

    uv run python methods/phase-labelling/search_context.py
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.compute as pc  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import agreement  # noqa: E402
import phases  # noqa: E402

MAX_GAP = 4  # s between two runs of one segment still counted as adjacent
C, S, T = 0, 1, 2
NONE = 3  # no adjacent run
PREV, NEXT = ((C, "climb"), (T, "transition"), (NONE, "start")), ((C, "climb"), (T, "transition"), (NONE, "end"))
SETS = {"his": "vilpellet", "refit": "refit"}


def load(name, flights):
    """Runs of the flights, merged where same-phase neighbours touch: flight, segment, phase, ts, te (seconds)."""
    t = pq.read_table(agreement.folder("para", name) / "phase_segments.parquet",
                      columns=["flight_id", "segment_id", "phase", "t_start", "t_end"])
    r = dict(flight=pc.cast(t["flight_id"], pa.int64()).to_numpy(), seg=t["segment_id"].to_numpy(),
             phase=pc.index_in(t["phase"], value_set=pa.array(phases.STATES)).to_numpy().astype(np.int8),
             ts=t["t_start"].to_numpy().astype(np.int64), te=t["t_end"].to_numpy().astype(np.int64))
    keep = np.isin(r["flight"], flights)
    r = {k: v[keep] for k, v in r.items()}
    o = np.lexsort((r["ts"], r["seg"], r["flight"]))
    r = {k: v[o] for k, v in r.items()}
    new = np.r_[True, ~(adjacent(r) & (r["phase"][1:] == r["phase"][:-1]))]
    first = np.flatnonzero(new)
    last = np.r_[first[1:] - 1, len(new) - 1]
    return dict(flight=r["flight"][first], seg=r["seg"][first], phase=r["phase"][first], ts=r["ts"][first], te=r["te"][last])


def adjacent(r):
    """adjacent(r)[i]: run i+1 follows run i within MAX_GAP s in the same segment."""
    return (r["flight"][1:] == r["flight"][:-1]) & (r["seg"][1:] == r["seg"][:-1]) & (r["ts"][1:] - r["te"][:-1] <= MAX_GAP)


def context(r):
    """Previous and next phase of every run (NONE if not adjacent), seconds of every run."""
    adj = adjacent(r)
    prev = np.where(np.r_[False, adj], np.r_[0, r["phase"][:-1]], NONE).astype(np.int8)
    nxt = np.where(np.r_[adj, False], np.r_[r["phase"][1:], 0], NONE).astype(np.int8)
    return prev, nxt, r["te"] - r["ts"] + 1


def dist(d):
    if not len(d):
        return dict(median=None, q25=None, q75=None)
    q = np.quantile(d, [0.25, 0.5, 0.75])
    return dict(median=float(q[1]), q25=float(q[0]), q75=float(q[2]))


def summarise(phase, prev, nxt, sec, flight):
    """All numbers of one set of runs (a full set or one ground class)."""
    hours = sec.sum() / 3600
    srch = phase == S
    exit_like = srch & (prev == C) & (nxt == T)
    ctx = {}
    for i, p in PREV:
        for j, n in NEXT:
            m = srch & (prev == i) & (nxt == j)
            ctx[f"{p}>search>{n}"] = dict(count=int(m.sum()), share_of_searches=float(m.sum() / srch.sum()),
                                          share_of_search_time=float(sec[m].sum() / sec[srch].sum()), seconds=dist(sec[m]))
    cnt = lambda a, b: int(((phase == a) & (nxt == b)).sum())
    names = phases.STATES
    switches = {f"{names[a]}>{names[b]}": dict(count=cnt(a, b), per_hour=cnt(a, b) / hours) for a in range(3) for b in range(3) if a != b}
    n_s = int(srch.sum())
    # searches that remain when the exit-like ones become transition; a search is next-known unless it ends a segment
    keep = srch & ~exit_like
    stat = lambda m, tr_extra: dict(
        search_time_share=float(sec[m].sum() / sec.sum()),
        transition_time_share=float((sec[phase == T].sum() + tr_extra) / sec.sum()),
        searches_per_hour=float(m.sum() / hours),
        share_ending_in_climb=float((m & (nxt == C)).sum() / m.sum()),
        share_ending_in_climb_of_known_next=float((m & (nxt == C)).sum() / (m & (nxt != NONE)).sum()),
        median_search_s=float(np.median(sec[m])))
    return dict(flights=int(len(np.unique(flight))), hours=float(hours), searches=n_s,
                exit_like=dict(count=int(exit_like.sum()), share_of_searches=float(exit_like.sum() / n_s),
                               share_of_search_time=float(sec[exit_like].sum() / sec[srch].sum()), seconds=dist(sec[exit_like]),
                               per_hour=float(exit_like.sum() / hours)),
                contexts=ctx, switches=switches,
                time_share=dict(zip(names, (float(sec[phase == k].sum() / sec.sum()) for k in range(3)))),
                as_labelled=stat(srch, 0.0), exit_like_as_transition=stat(keep, float(sec[exit_like].sum())))


def figure(data, path):
    keys = ["climb>search>transition", "transition>search>climb", "climb>search>climb", "transition>search>transition"]
    titles = ["climb, search, transition\n(exit-like)", "transition, search, climb\n(search proper)", "climb, search, climb", "transition, search, transition"]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    colors = {"his": "#4c72b0", "refit": "#dd8452"}
    for s, (name, off) in enumerate(zip(SETS, (-0.2, 0.2))):
        d = data[name]["durations"]
        bp = ax.boxplot([d[k] for k in keys], positions=np.arange(len(keys)) + off, widths=0.32, whis=(5, 95), showfliers=False,
                        patch_artist=True, medianprops=dict(color="k"))
        for b in bp["boxes"]:
            b.set_facecolor(colors[name])
        for i, k in enumerate(keys):
            ax.text(i + off, 1.1, f"n={len(d[k]) / 1000:.0f}k", ha="center", fontsize=7)
    ax.set_yscale("log")
    ax.set_ylim(1, 3000)
    ax.set_xticks(range(len(keys)), titles, fontsize=8)
    ax.set_ylabel("search duration (s)")
    ax.legend([plt.Rectangle((0, 0), 1, 1, color=c) for c in colors.values()], ["Vilpellet", "refit"], loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path)


def main():
    his0 = agreement.load_runs(agreement.folder("para", "vilpellet") / "phase_segments.parquet")
    ours0 = agreement.load_runs(agreement.folder("para", "refit") / "phase_segments.parquet")
    flights = np.intersect1d(np.unique(his0["flight"]), np.unique(ours0["flight"]))
    del his0, ours0
    cls_of = agreement.ground_class("para", flights)
    result, plot = dict(common_flights=int(len(flights)), gap_s=MAX_GAP), {}
    for name, folder in SETS.items():
        r = load(folder, flights)
        prev, nxt, sec = context(r)
        cls = cls_of[np.searchsorted(flights, r["flight"])]
        res = summarise(r["phase"], prev, nxt, sec, r["flight"])
        res["per_class"] = {c: summarise(r["phase"][m], prev[m], nxt[m], sec[m], r["flight"][m])
                            for c in sorted(set(cls)) for m in [cls == c]}
        result[name] = res
        srch = r["phase"] == S
        plot[name] = dict(durations={f"{p}>search>{n}": sec[srch & (prev == i) & (nxt == j)]
                                     for i, p in PREV for j, n in NEXT})
        print(name, json.dumps(res["exit_like"]), json.dumps(res["as_labelled"]), json.dumps(res["exit_like_as_transition"]))
    (HERE / "out/search_context.json").write_text(json.dumps(result, indent=1))
    figure(plot, HERE / "out/search_context.pdf")


if __name__ == "__main__":
    main()
