"""Rotation: in groups of 3 pilots or more, does who leads keep leading? The scale of one group.

Population: the groups of n >= 3 whose members are all 1 Hz flights with phases and that last at
least MIN_GROUP s (the largest lag, so that every lag sees the same groups), in sizes 3, 4, 5+.

Order. At each grid instant the delays l*(t) of the pair intervals alive among the members
(cohesion.py; none outside a correspondence or in a search) give the position x_i of each pilot
along the string, the passage time of a common point in s: least squares on the edges, minimise
sum (x_i - x_j - l_ij)^2 with mean 0, l_ij the delay by which i follows j (positions()). An
instant whose edges do not connect all members has no order. Rank 0, the smallest x, is the head;
ties are broken at random.

Measures, by size:
- Kendall tau between the orders at t and at t + s (mean over member pairs of the product of the
  signs of x_i - x_j), over the instants where both are ordered, against s in LAGS_MIN. Reference:
  tau = 0 for independent orders, and the plateau at the longest lags.
- Head spells. The head at each instant, runs shorter than MIN_RUN s dropped and neighbours of one
  pilot joined (hierarchy.py); a spell is counted in ordered seconds. Per group the share of the
  pilot most often head, against 1/n and against the same spells with each new head drawn at
  random among the other members (N_NULL draws); head changes per hour; the median spell.
- Roles. Head, middle, tail (rank 0, 1 to n - 2, n - 1) at t and t + ROLE_S: the Markov matrix.
- Phase. The climbing share f(t) of the members (climbing(): a member climbs inside a climb run
  lasting at least MIN_RUN s). Head changes per hour of ordered time by group phase: all gliding
  (f = 0), mixed, all climbing (f = 1).
- Climb starts. A climb start is that of a climb run of a member. Head changes per hour in the
  first FIRST_S after the starts against MID_S after the same starts while that member still
  climbs (instants of the first window left out of the second). For the head changes with a climb
  start within WINDOW_S, the share where the new head is the pilot who started climbing first,
  against the share expected if the new head were any pilot but the old one.

Bands: 5-95% of N_BOOT draws of the dependency blocks of the note (episodes).

    uv run python rotation.py [--run NAME] [--sample N]    # from this folder, after cohesion.py
"""

import argparse
import json
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cohesion import CACHE as COHESION, dense
from hierarchy import MIN_RUN
from phase_msd import RUNS
from tamsd import LABELS, OUT, SEED, save

LAGS_MIN = (1, 2, 5, 10, 15, 20, 30)  # lags of the Kendall tau, min
MIN_GROUP = max(LAGS_MIN) * 60  # shortest group kept, s
ROLE_S = 60  # lag of the role transitions, s
WINDOW_S = 300  # half-window around a head change to look for a climb start, s
FIRST_S, MID_S = 60, (120, 300)  # first minute after a climb start; mid-climb, s after it
N_NULL, N_BOOT = 50, 200  # draws of random heads per group; draws of the blocks
TIE_S = 1e-6  # jitter that breaks ties of positions, s
PHASES = ("gliding", "mixed", "climbing")  # group phase: f = 0, 0 < f < 1, f = 1
SIZES = ("3", "4", "5+")
COLORS = {"3": "#eb6834", "4": "#2a78d6", "5+": "#1baf7a"}


def positions(n, edges, delays):
    """Least-squares positions x (mean 0) of n members from the delays on the edges (i, j):
    minimise sum (x_i - x_j - l_ij)^2. `delays` has one row per instant and one column per edge,
    NaN where the edge has none; each row uses its finite edges. Rows whose edges do not connect
    the n members are NaN."""
    edges, delays = np.asarray(edges), np.atleast_2d(np.asarray(delays, float))
    x = np.full((len(delays), n), np.nan)
    patterns, row = np.unique(np.isfinite(delays), axis=0, return_inverse=True)
    for k, live in enumerate(patterns):
        d = np.zeros((live.sum(), n))
        d[np.arange(len(d)), edges[live, 0]] = 1
        d[np.arange(len(d)), edges[live, 1]] = -1
        if np.linalg.matrix_rank(d) == n - 1:  # the incidence matrix of a connected graph
            here = row.ravel() == k
            x[here] = delays[here][:, live] @ np.linalg.pinv(d).T
    return x


def spells(heads, step):
    """The head spells of a series of heads (-1 where there is none): runs of the ordered instants,
    those shorter than MIN_RUN s dropped, neighbours of one pilot joined. Returns the pilot, the
    first instant and the length in instants of each spell."""
    at = np.flatnonzero(heads >= 0)
    h = heads[at]
    first = np.r_[0, np.flatnonzero(np.diff(h)) + 1]
    n = np.diff(np.r_[first, len(h)])
    keep = n * step >= MIN_RUN
    first, n = first[keep], n[keep]
    new = np.r_[True, h[first][1:] != h[first][:-1]] if len(first) else np.zeros(0, bool)
    return h[first][new], at[first][new], np.add.reduceat(n, np.flatnonzero(new)) if len(first) else n


def group_delays(group, real, flight_of, step):
    """Edges (i, j) as member indices and the delays l*(t) (T, m) of the pair intervals of the
    group over its instants, where i follows j by l_ij."""
    index = {f: k for k, f in enumerate(group.members)}
    t0, steps = group.t_start, (group.t_end - group.t_start) // step
    columns, edges = [], []
    for a, b, on, off, t, lag, search in real:
        a, b = flight_of[a], flight_of[b]
        if a not in index or b not in index:
            continue
        lo, hi = max(on, t0), min(off, group.t_end)
        if hi <= lo:
            continue
        x = dense(t, lag, search, on, off, step)[(lo - on) // step:(hi - on) // step]
        edge, sign = ((index[b], index[a]), 1) if index[b] > index[a] else ((index[a], index[b]), -1)
        if edge not in edges:
            edges.append(edge)
            columns.append(np.full(steps, np.nan))
        columns[edges.index(edge)][(lo - t0) // step:(hi - t0) // step] = sign * x
    return edges, np.array(columns).T.reshape(steps, len(edges))


def kendall(sign, ordered, lags):
    """Per lag (in instants), the sum over the instants t with t and t + lag ordered of the mean over
    member pairs of sign_t sign_(t+lag), and the number of those instants."""
    out = []
    for k in lags:
        ok = ordered[:-k] & ordered[k:] if k < len(ordered) else np.zeros(0, bool)
        out.append(((sign[:-k] * sign[k:]).mean(axis=1)[ok].sum() if ok.any() else 0.0, ok.sum()))
    return np.array(out, float)


def head_stats(pilot, length, n, rng):
    """The share of the time of the pilot most often head, and its mean when every spell, after the
    first, passes the head to a member drawn at random among the others (N_NULL draws); NaN without
    spells."""
    if not len(pilot):
        return np.nan, np.nan
    total = length.sum()
    jump = rng.integers(1, n, (N_NULL, len(pilot)))
    jump[:, 0] = rng.integers(0, n, N_NULL)
    null = np.mean([np.bincount(d, length, n).max() for d in jump.cumsum(axis=1) % n]) / total
    return np.bincount(pilot, length, n).max() / total, null


def climbing(group, step, runs):
    """Whether each member of the group is in a climb run at each instant of its grid
    (t_start + k step): a bool array (steps, n). `runs` maps a flight to the sorted starts and
    ends of its climb runs (climb_runs())."""
    t = group.t_start + step * np.arange((group.t_end - group.t_start) // step)
    out = np.zeros((len(t), len(group.members)), bool)
    for k, f in enumerate(group.members):
        if f in runs:
            start, end = runs[f]
            i = np.searchsorted(start, t, side="right") - 1
            out[:, k] = (i >= 0) & (t < end[np.maximum(i, 0)])
    return out


def covered(lo, hi, steps):
    """The instants of 0..steps - 1 inside any of the intervals [lo, hi)."""
    lo, hi = np.clip(lo, 0, steps), np.clip(np.maximum(hi, lo), 0, steps)
    mark = np.zeros(steps + 1, int)
    np.add.at(mark, lo, 1)
    np.add.at(mark, hi, -1)
    return np.cumsum(mark)[:steps] > 0


def climb_events(group, step, runs):
    """The climb runs of the members that start inside the group: the first instant, the end
    instant (both on its grid) and the member index of each, in time order."""
    e = [(int((s - group.t_start) // step), k, int(-(-(en - group.t_start) // step)))
         for k, f in enumerate(group.members) if f in runs for s, en in zip(*runs[f])
         if group.t_start <= s < group.t_end]
    e = np.array(sorted(e), int).reshape(-1, 3)
    return e[:, 0], e[:, 2], e[:, 1]


def one_group(group, real, flight_of, step, runs, rng):
    """The sums of one group that the summary adds: see names(). Also the lengths of its head spells."""
    n = len(group.members)
    edges, delays = group_delays(group, real, flight_of, step)
    steps = len(delays)
    x = positions(n, edges, delays) if len(edges) else np.full((steps, n), np.nan)
    x += rng.uniform(-TIE_S, TIE_S, x.shape)
    ordered = np.isfinite(x[:, 0])
    rank = x.argsort(axis=1).argsort(axis=1)
    i, j = np.triu_indices(n, 1)
    sign = np.nan_to_num(np.sign(x[:, i] - x[:, j]))
    tau = kendall(sign, ordered, [m * 60 // step for m in LAGS_MIN])
    pilot, first, length = spells(np.where(ordered, x.argmin(axis=1), -1), step)
    share, null = head_stats(pilot, length, n, rng)
    change = np.zeros(steps, bool)
    change[first[1:]] = True
    role = np.where(rank == 0, 0, np.where(rank == n - 1, 2, 1))
    m = ROLE_S // step
    both = ordered[:-m] & ordered[m:]
    roles = np.bincount((role[:-m] * 3 + role[m:])[both].ravel(), minlength=9)
    hours = lambda mask: mask.sum() * step / 3600
    # head changes by group phase, and in the first minute after a climb start against mid-climb
    f = climbing(group, step, runs).mean(axis=1)
    phase = np.where(f == 0, 0, np.where(f < 1, 1, 2))
    by_phase = [(change & (phase == k)).sum() for k in range(3)] + [hours(ordered & (phase == k)) for k in range(3)]
    e, end, who = climb_events(group, step, runs)
    first_min = covered(e, e + FIRST_S // step, steps)
    mid = covered(e + MID_S[0] // step, np.minimum(e + MID_S[1] // step, end), steps) & ~first_min
    # head changes with a climb start within WINDOW_S: is the new head the first climber?
    w = WINDOW_S // step
    seen = hit = expected = 0
    for k in range(1, len(pilot)):
        lo, hi = np.searchsorted(e, [first[k] - w, first[k] + w])
        if hi > lo:
            seen, hit = seen + 1, hit + (pilot[k] == who[lo])
            expected += (pilot[k - 1] != who[lo]) / (n - 1)
    stats = np.r_[tau[:, 0], tau[:, 1], len(pilot) > 0, share, null, (len(pilot) > 0) / n,
                  len(pilot) - 1 if len(pilot) else 0, hours(ordered), hours(np.ones(steps, bool)), roles, by_phase,
                  (change & first_min).sum(), hours(ordered & first_min), (change & mid).sum(), hours(ordered & mid),
                  seen, hit, expected]
    return np.nan_to_num(stats), length * step


def names():
    """The columns of the vector of one_group, with the lags named by their index."""
    nl = len(LAGS_MIN)
    return ([f"tau{k}" for k in range(nl)] + [f"pairs{k}" for k in range(nl)]
            + ["headed", "dominant", "dominant_null", "inverse_n", "changes", "ordered_h", "group_h"]
            + [f"role{k}" for k in range(9)] + [f"{k}_changes" for k in range(3)] + [f"{k}_h" for k in range(3)]
            + ["first_changes", "first_h", "mid_changes", "mid_h", "first_seen", "first_hit", "first_expected"])


def climb_runs(flights, members):
    """The climb runs of the `members` (flight numbers) lasting at least MIN_RUN s and not opening
    the record: by flight, the sorted starts and ends in UTC seconds."""
    out = {}
    for d, path in RUNS.items():
        r = pd.read_parquet(path, columns=["flight_id", "phase", "t_start", "duration_s", "left_censored"],
                            filters=[("phase", "==", "climb")])
        r = r[~r["left_censored"] & (r["duration_s"] >= MIN_RUN)]
        f = flights.loc[(flights["discipline"] == d) & flights["flight"].isin(members), ["flight_id", "flight", "utc0"]]
        r = r.merge(f, on="flight_id")
        r = r.assign(start=r["utc0"] + r["t_start"]).sort_values("start")
        out |= {k: (v["start"].to_numpy(), (v["start"] + v["duration_s"]).to_numpy()) for k, v in r.groupby("flight")}
    return out


def measure(groups, flights, step, rng):
    """Per group of `groups`: its columns (see names) and the lengths of its head spells."""
    runs = climb_runs(flights, {f for m in groups["members"] for f in m})
    by_date = dict(tuple(flights[flights["date"].isin(groups["date"])].groupby("date")))
    rows, lengths = [], []
    for number, (date, day) in enumerate(groups.groupby("date", sort=False)):
        real = pickle.loads((COHESION / "all" / "days" / f"{date}.pkl").read_bytes())["real"]
        flight_of = dict(zip(by_date[date]["flight_id"], by_date[date]["flight"]))
        for group in day.itertuples():
            s, length = one_group(group, real, flight_of, step, runs, rng)
            rows.append(s)
            lengths.append(length)
        if number % 100 == 0:
            print(f"{number} dates, {len(rows)}/{len(groups)} groups", flush=True)
    return np.array(rows), lengths


def wquantile(x, weights, q):
    """The q quantile of x under each row of `weights` (draws x values)."""
    o = np.argsort(x)
    c = np.cumsum(weights[:, o], axis=1)
    return x[o][(c >= q * c[:, -1:]).argmax(axis=1)]


def summarize(rows, lengths, groups, flights, block, rng):
    """The sums of the groups by size, with the 5-95% bands of the draws of the blocks."""
    col = {k: i for i, k in enumerate(names())}
    n = groups["n"].to_numpy()
    cls = {"3": n == 3, "4": n == 4, "5+": n >= 5, "all": n >= 3}
    ids, k = np.unique(block, return_inverse=True)
    w = rng.multinomial(len(ids), np.full(len(ids), 1 / len(ids)), size=N_BOOT).astype(float)

    def ratio(num, den, sel):
        a, b = (np.bincount(k[sel], x[sel], len(ids)) for x in (num, den))
        draws = (w @ a) / (w @ b)
        return {"value": float(a.sum() / b.sum()), "band": np.nanpercentile(draws, [5, 95]).tolist()}, draws

    c = lambda name: rows[:, col[name]]
    out = {"groups": {}, "tau": {}, "head": {}, "roles": {}, "climbs": {}}
    for size, sel in cls.items():
        out["groups"][size] = {"groups": int(sel.sum()), "hours": float(c("group_h")[sel].sum()),
                               "ordered_share": float(c("ordered_h")[sel].sum() / c("group_h")[sel].sum())}
        out["tau"][size] = {"lags_min": list(LAGS_MIN), "pairs": [float(c(f"pairs{i}")[sel].sum()) for i in range(len(LAGS_MIN))],
                            "tau": [ratio(c(f"tau{i}"), c(f"pairs{i}"), sel)[0] for i in range(len(LAGS_MIN))]}
        headed = c("headed")
        spell = np.concatenate([lengths[g] for g in np.flatnonzero(sel)])
        owner = np.repeat(np.arange(len(lengths))[sel], [len(lengths[g]) for g in np.flatnonzero(sel)])
        wt = w[:, k[owner]]
        out["head"][size] = {
            "groups_with_spells": int(headed[sel].sum()),
            "dominant": ratio(c("dominant"), headed, sel)[0], "dominant_null": ratio(c("dominant_null"), headed, sel)[0],
            "one_over_n": ratio(c("inverse_n"), headed, sel)[0]["value"],
            "changes_per_hour": ratio(c("changes"), c("ordered_h"), sel)[0],
            "spells": int(len(spell)), "spell_median_min": float(np.median(spell) / 60),
            "spell_median_band": np.percentile(wquantile(spell, wt, 0.5) / 60, [5, 95]).tolist(),
            "spell_time_median_min": float(wquantile(spell, spell[None], 0.5)[0] / 60),
            "spell_time_median_band": np.percentile(wquantile(spell, wt * spell, 0.5) / 60, [5, 95]).tolist()}
        trans = np.array([c(f"role{i}")[sel].sum() for i in range(9)]).reshape(3, 3)
        out["roles"][size] = {"count": trans.tolist(), "matrix": (trans / trans.sum(1, keepdims=True)).tolist(),
                              "occupancy": (trans.sum(1) / trans.sum()).tolist(), "diagonal_band": []}
        for r in range(3):
            out["roles"][size]["diagonal_band"].append(
                ratio(c(f"role{4 * r}"), sum(c(f"role{3 * r + j}") for j in range(3)), sel)[0])
        first, d_first = ratio(c("first_changes"), c("first_h"), sel)
        mid, d_mid = ratio(c("mid_changes"), c("mid_h"), sel)
        hit, _ = ratio(c("first_hit"), c("first_seen"), sel)
        expected, _ = ratio(c("first_expected"), c("first_seen"), sel)
        out["climbs"][size] = {
            "by_group_phase_per_hour": {p: ratio(c(f"{k}_changes"), c(f"{k}_h"), sel)[0] for k, p in enumerate(PHASES)},
            "group_phase_hours": {p: float(c(f"{k}_h")[sel].sum()) for k, p in enumerate(PHASES)},
            "first_minute_per_hour": first, "mid_climb_per_hour": mid,
            "first_over_mid": {"value": first["value"] / mid["value"],
                               "band": np.nanpercentile(d_first / d_mid, [5, 95]).tolist()},
            "first_changes": float(c("first_changes")[sel].sum()), "mid_changes": float(c("mid_changes")[sel].sum()),
            "first_climber_new_head": hit, "expected_if_any_but_old_head": expected,
            "changes_with_a_climb_start": int(c("first_seen")[sel].sum())}
    return out


def plot(r, path):
    """Left: Kendall tau against the lag; middle: time share of the most frequent head, against random
    hand-overs and 1/n; right: head changes per hour by group phase and, at the climb starts, in the
    first minute against mid-climb."""
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(13, 3.9))
    for size in SIZES:
        t = r["tau"][size]
        a.fill_between(t["lags_min"], *np.array([x["band"] for x in t["tau"]]).T, color=COLORS[size], alpha=0.15, lw=0)
        a.plot(t["lags_min"], [x["value"] for x in t["tau"]], color=COLORS[size], marker="o", ms=4, lw=1.2, label=f"{size} pilots")
    a.axhline(0, color="#999999", lw=0.6)
    a.set(xlabel="lag $s$ (min)", ylabel=r"Kendall $\tau$ of the orders at $t$ and $t+s$")
    a.legend(frameon=False, fontsize=8)
    x = np.arange(len(SIZES))
    for dx, key, color, label in ((-0.18, "dominant", "#eb6834", "observed"), (0.18, "dominant_null", "#2a78d6", "heads at random")):
        v = np.array([r["head"][s][key]["value"] for s in SIZES])
        band = np.array([r["head"][s][key]["band"] for s in SIZES]).T
        b.bar(x + dx, v, width=0.34, color=color, label=label)
        b.errorbar(x + dx, v, yerr=[v - band[0], band[1] - v], fmt="none", ecolor="#333333", lw=1)
    b.plot(x, [r["head"][s]["one_over_n"] for s in SIZES], "_", color="#333333", ms=28, mew=1.5, label="$1/n$")
    b.set_xticks(x, [f"{s} pilots" for s in SIZES])
    b.set(ylabel="time share of the most frequent head", ylim=(0, 1))
    b.legend(frameon=False, fontsize=8)
    cats = [(f"group {p}", lambda k, p=p: k["by_group_phase_per_hour"][p]) for p in PHASES]
    cats += [("first minute\nafter a start", lambda k: k["first_minute_per_hour"]),
             ("2-5 min after\n(mid-climb)", lambda k: k["mid_climb_per_hour"])]
    for i, size in enumerate(SIZES):
        v = np.array([f(r["climbs"][size])["value"] for _, f in cats])
        band = np.array([f(r["climbs"][size])["band"] for _, f in cats]).T
        at = np.arange(len(cats)) + (i - 1) * 0.2
        c.errorbar(at, v, yerr=[v - band[0], band[1] - v], fmt="o", ms=4, lw=1.2, color=COLORS[size], label=f"{size} pilots")
    c.axvline(2.5, color="#999999", lw=0.6)
    c.set_xticks(np.arange(len(cats)), [n for n, _ in cats], fontsize=7)
    c.set(ylabel="head changes per hour", ylim=(0, None))
    c.legend(frameon=False, fontsize=8)
    save(fig, path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", help="read by tamsd")
    parser.add_argument("--sample", type=int, help="a random sample of dates (seed 0)")
    args = parser.parse_args()
    step = json.loads((LABELS / "days" / "params.json").read_text())["step"]
    flights = pd.read_parquet(LABELS / "flights.parquet")
    groups = pd.read_parquet(LABELS / "groups.parquet")
    labelled = flights.set_index("flight")["labelled"]
    groups = groups[(groups["n"] >= 3) & (groups["t_end"] - groups["t_start"] >= MIN_GROUP)]
    groups = groups[groups["members"].map(lambda m: bool(labelled.loc[list(m)].all()))].reset_index(drop=True)
    if args.sample:
        dates = np.random.default_rng(0).choice(sorted(groups["date"].unique()), args.sample, replace=False)
        groups = groups[groups["date"].isin(dates)].reset_index(drop=True)
    rng = np.random.default_rng(SEED)
    rows, lengths = measure(groups, flights, step, rng)
    block = pd.read_parquet(LABELS / "episodes.parquet").set_index("episode_id")["block"]
    result = summarize(rows, lengths, groups, flights, block.reindex(groups["episode_id"]).to_numpy(), rng)
    OUT.mkdir(parents=True, exist_ok=True)
    name = f"rotation_sample{args.sample}" if args.sample else "rotation"
    (OUT / f"{name}.json").write_text(json.dumps(result, indent=1))
    plot(result, OUT / f"{name}.pdf")
    print(json.dumps(result["groups"]), flush=True)
