"""What groups are: the size and duration of configurations and episodes, and how a configuration
ends.

Sizes: configurations by number of pilots, counted and weighted by duration; episodes by their
largest configuration. Durations: the share of configurations and of episodes lasting at least d.

Events. The configurations of an episode that end at an instant and those that start at it,
linked when they share a member (the links of the note), make one event per connected
component. The outcome of each configuration that ends: join (one to one, members added), leave
(one to one or none, members removed in flight), landing (the same, with a removed member landing
within STEP s), split (one to several), merge (several to one), other. A configuration with no
predecessor forms its episode, at departure if a member departs within STEP s. Rates: the
configurations of size n ending with each outcome per hour spent at size n (sizes above N_MAX
pooled).

Bands: 5-95% of N_BOOT draws of the blocks of the note.

    uv run python describe.py [--run NAME]    # from this folder, about 1 min
"""

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from tamsd import LABELS, OUT, SEED, save

STEP = 5  # s, the grid of the labels
N_MAX = 8
N_BOOT = 200
OUTCOMES = ("join", "leave", "landing", "split", "merge", "other")
COLORS = dict(zip(OUTCOMES, ["#1baf7a", "#2a78d6", "#52514e", "#eb6834", "#8e5bd0", "#d4a017"]))
MARKERS = dict(zip(OUTCOMES, ["^", "v", "s", "o", "D", "x"]))
DURATIONS_MIN = np.geomspace(1, 1200, 60)


def outcome(before, after, landed):
    """Outcome of the configurations with member sets `before` ending where those of `after` start
    (empty: dissolution); `landed` holds the members landing at that instant."""
    if len(before) == 1 and len(after) <= 1:
        old, new = before[0], (after[0] if after else set())
        if new > old:
            return "join"
        if new < old:
            return "landing" if (old - new) & landed else "leave"
        return "other"
    if len(after) == 1:
        return "merge"
    if len(before) == 1:
        return "split"
    return "other"


def events(confs, flights):
    """Per configuration (rows of `confs`): outcome, whether it dissolves (no successor), and its
    formation ('' if it has a predecessor, else 'departure' or 'flight')."""
    n = len(confs)
    members = [set(m.tolist()) for m in confs["members"]]
    pos = pd.Series(np.arange(n), index=confs["conf_id"].to_numpy())
    pairs = confs[["episode_id", "conf_id", "t_end"]].merge(
        confs[["episode_id", "conf_id", "t_start"]], left_on=["episode_id", "t_end"],
        right_on=["episode_id", "t_start"], suffixes=("_e", "_s"))
    e, s = pos[pairs["conf_id_e"]].to_numpy(), pos[pairs["conf_id_s"]].to_numpy()
    shared = np.array([bool(members[a] & members[b]) for a, b in zip(e, s)], bool)
    e, s = e[shared], s[shared]
    graph = coo_matrix((np.ones(len(e)), (e, s + n)), shape=(2 * n, 2 * n))
    label = connected_components(graph, directed=False)[1]
    landing = dict(zip(flights["flight"], flights["landing"]))
    departure = dict(zip(flights["flight"], flights["departure"]))
    t_end, t_start = confs["t_end"].to_numpy(), confs["t_start"].to_numpy()

    def landed(group, t):
        return {f for f in group if abs(landing[f] - t) <= STEP}

    out = np.full(n, "", object)
    for _, g in pd.DataFrame({"e": e, "s": s, "k": label[e]}).groupby("k"):
        before, after = np.unique(g["e"]), np.unique(g["s"])
        old = set().union(*(members[a] for a in before))
        out[before] = outcome([members[a] for a in before], [members[b] for b in after], landed(old, t_end[before[0]]))
    alone = np.flatnonzero(out == "")
    out[alone] = [outcome([members[a]], [], landed(members[a], t_end[a])) for a in alone]
    has_before = np.zeros(n, bool)
    has_before[s] = True
    form = np.where(has_before, "", np.where(
        [any(abs(departure[f] - t_start[a]) <= STEP for f in members[a]) for a in range(n)], "departure", "flight"))
    dissolves = np.ones(n, bool)
    dissolves[e] = False
    return pd.DataFrame({"outcome": out, "dissolves": dissolves, "form": form}, index=confs.index)


def rates(table, rng):
    """Per size 2..N_MAX (the last pooled): hours, configurations ending with each outcome, rate per
    hour and its 5-95% band, resampling blocks."""
    ids, k = np.unique(table["block"], return_inverse=True)
    j = np.clip(table["n"].to_numpy(), 2, N_MAX) - 2
    o = pd.Categorical(table["outcome"], OUTCOMES).codes
    hours = np.zeros((len(ids), N_MAX - 1))
    counts = np.zeros((len(ids), N_MAX - 1, len(OUTCOMES)))
    np.add.at(hours, (k, j), table["hours"].to_numpy())
    np.add.at(counts, (k, j, o), 1)
    w = rng.multinomial(len(ids), np.full(len(ids), 1 / len(ids)), size=N_BOOT).astype(float)
    draws = np.einsum("bk,kso->bso", w, counts) / (w @ hours)[:, :, None]
    total = counts.sum(axis=0)
    return {"sizes": list(range(2, N_MAX + 1)), "hours": hours.sum(axis=0).tolist(),
            "counts": {x: total[:, i].astype(int).tolist() for i, x in enumerate(OUTCOMES)},
            "rate": {x: (total[:, i] / hours.sum(axis=0)).tolist() for i, x in enumerate(OUTCOMES)},
            "band": {x: np.percentile(draws[:, :, i], [5, 95], axis=0).tolist() for i, x in enumerate(OUTCOMES)}}


def survival(minutes):
    return [(minutes >= d).mean() for d in DURATIONS_MIN]


def describe(confs, episodes, flights, rng):
    ev = events(confs, flights)
    minutes = (confs["t_end"] - confs["t_start"]).to_numpy() / 60
    ep_minutes = (episodes["t_end"] - episodes["t_start"]).to_numpy() / 60
    block = episodes.set_index("episode_id")["block"]
    table = pd.DataFrame({"n": confs["n"], "hours": minutes / 60, "outcome": ev["outcome"],
                          "block": block.reindex(confs["episode_id"]).to_numpy()})
    discipline = flights.set_index("flight")["discipline"]
    with_hang = confs["members"].map(lambda m: (discipline.reindex(m) == "hang").any())
    by_ep = ev.groupby(confs["episode_id"])["outcome"]
    n = confs["n"].to_numpy()
    return {
        "configurations": len(confs), "episodes": len(episodes), "with_hang": float(with_hang.mean()),
        "size_count": {int(k): int(v) for k, v in confs["n"].value_counts().sort_index().items()},
        "size_hours": {int(k): float(v) for k, v in table.groupby("n")["hours"].sum().items()},
        "episode_n_max": {int(k): int(v) for k, v in episodes["n_max"].value_counts().sort_index().items()},
        "durations_min": DURATIONS_MIN.tolist(),
        "survival_conf": survival(minutes), "survival_episode": survival(ep_minutes),
        "median_conf_min": float(np.median(minutes)), "median_episode_min": float(np.median(ep_minutes)),
        "episode_2h": float((ep_minutes >= 120).mean()),
        "median_conf_min_by_size": {int(k): float(np.median(minutes[np.clip(n, 2, N_MAX) == k]))
                                    for k in range(2, N_MAX + 1)},
        "outcomes": {x: int((ev["outcome"] == x).sum()) for x in OUTCOMES},
        "formations": {x: int((ev["form"] == x).sum()) for x in ("departure", "flight")},
        "dissolutions": {x: int((ev["dissolves"] & (ev["outcome"] == x)).sum()) for x in OUTCOMES},
        "episode_single": float((episodes["n_config"] == 1).mean()),
        "episode_with_split": float(by_ep.apply(lambda o: (o == "split").any()).mean()),
        "episode_with_merge": float(by_ep.apply(lambda o: (o == "merge").any()).mean()),
        "rates": rates(table, rng),
    }


def plot(r, tau_star, path):
    """Left: sizes; middle: durations, with tau*; right: rate of each outcome by size (a split needs
    at least 4 pilots)."""
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(13, 3.9))
    for key, label, color, marker in [("size_count", "configurations", "#2a78d6", "o"),
                                      ("size_hours", "time in configurations", "#1baf7a", "s"),
                                      ("episode_n_max", "episodes (largest size)", "#eb6834", "^")]:
        sizes = np.array(list(map(int, r[key])))
        share = np.array(list(r[key].values()), float)
        a.plot(sizes, share / share.sum(), color=color, marker=marker, ms=4, lw=1, label=label)
    a.set(xscale="log", yscale="log", xlabel="pilots", ylabel="share")
    a.set_xticks([2, 3, 5, 10, 20], labels=["2", "3", "5", "10", "20"])
    a.minorticks_off()
    a.legend(frameon=False, fontsize=8)
    for key, label, color in [("survival_conf", "configurations", "#2a78d6"),
                              ("survival_episode", "episodes", "#eb6834")]:
        b.plot(r["durations_min"], r[key], color=color, lw=1.5, label=label)
    b.axvline(tau_star / 60, color="#52514e", ls="--", lw=1)
    b.text(tau_star / 60, 2e-5, r" $\tau^*$", color="#52514e")
    b.set(xscale="log", yscale="log", xlabel="duration $d$ (min)", ylabel=r"share lasting at least $d$")
    b.legend(frameon=False, fontsize=8)
    rt = r["rates"]
    sizes = np.array(rt["sizes"])
    for x in OUTCOMES[:-1]:
        keep = sizes >= (4 if x == "split" else 2)
        lo, hi = np.array(rt["band"][x])[:, keep]
        c.fill_between(sizes[keep], lo, hi, color=COLORS[x], alpha=0.2, lw=0)
        c.plot(sizes[keep], np.array(rt["rate"][x])[keep], color=COLORS[x], marker=MARKERS[x], ms=5, lw=1.2, label=x)
    c.set(yscale="log", xlabel=f"pilots (last: {N_MAX}+)", ylabel="configurations ending per hour")
    c.set_xticks(sizes)
    c.legend(frameon=False, fontsize=8, ncol=2)
    save(fig, path)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    confs = pd.read_parquet(LABELS / "configurations.parquet")
    episodes = pd.read_parquet(LABELS / "episodes.parquet")
    flights = pd.read_parquet(LABELS / "flights.parquet")
    result = describe(confs, episodes, flights, np.random.default_rng(SEED))
    (OUT / "describe.json").write_text(json.dumps(result, indent=1))
    plot(result, json.loads((LABELS / "calibration.json").read_text())["tau_star"], OUT / "describe.pdf")
    print({k: result[k] for k in ("configurations", "episodes", "outcomes", "formations", "episode_single")}, flush=True)
