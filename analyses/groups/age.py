"""Rates of joins, leaves and landings by the age of the episode.

The birth-death chain of fission.py takes the rates of a member as functions of the size n
alone. Here they also depend on the age a of the episode, the time since its t_start: flat in
a, the chain stands; losses rising with a, episodes have a life cycle.

The outcome of each group that ends is that of describe.py (join, leave, landing, split,
merge); its age is the time from the start of the episode to the end of the group. The
exposure is the member-time at size n (n times the duration of the group), cut at the age
edges. Rate: groups of size n ending with an outcome per member-hour at size n, by size class
(sizes above N_MAX pooled, plus all sizes together) and age bin. A leave, landing or split
ends a link, and a link lasts at least tau* (as does an episode): none occurs before the age
tau*, so their exposure starts there and the bins before it are empty.

Hour control: age goes with the time of day, and late leaves are landings. The rates are
repeated within classes of local time of day (Europe/Paris, matched.time_of_day), the
exposure cut at the class edges as well. Cells with fewer than MIN_HOURS member-hours are not
quoted. Bands: 5-95% of N_BOOT draws of the blocks of the note (the draws of fission.py).

    uv run python age.py [--run NAME]    # from this folder; writes out/<run>/age.json and age.pdf
"""

import json
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from describe import OUTCOMES, events
from matched import time_of_day
from partners import named
from tamsd import LABELS, OUT, SEED, save

KINDS = OUTCOMES[:-1]  # the outcomes of a group that ends, without "other"
LATE = ("leave", "landing", "split")  # outcomes that cannot occur before the age tau*
N_MAX = 8  # sizes 2..N_MAX, the last pooled (describe.py)
N_BOOT = 200
MIN_HOURS = 10  # member-hours of a cell below which it is not quoted (fission.py)
AGE_MIN = [0, 15, 30, 60, 120, np.inf]  # edges of the age bins, min since the start of the episode
HOUR_H = [0, 13, 15, 24]  # edges of the classes of local time, h: 19, 42 and 39% of the member-time
AGES = ["0-15", "15-30", "30-60", "60-120", "120+"]
HOURS = ["all day"] + [f"{a}-{b} h" for a, b in zip(HOUR_H[:-1], HOUR_H[1:])]
SIZES = [str(n) for n in range(2, N_MAX)] + [f"{N_MAX}+", "all"]
PLOTTED = ("join", "leave", "landing")
TITLES = {"join": "joins", "leave": "leaves in flight", "landing": "landings"}
PLOT_SIZES = SIZES[:-1]
PLOT_MIN_EVENTS = 30  # a line of the figure needs this many events in all
COLORS = dict(zip(SIZES, ["#2a78d6", "#1baf7a", "#d4a017", "#eb6834", "#c2410c", "#8e5bd0", "#9a9a96", "#0b0b0b"]))


def exposure(start, dur, tod, age_edges, hour_edges):
    """Seconds that each interval spends in each age bin and hour class: intervals x age bins x
    classes. `start` is the age and `tod` the local time of day (s) at the start of an interval of
    length `dur`; the edges are in s."""
    def part(edges, at):  # offsets from the start of the interval where each bin begins and ends
        return np.maximum(edges[:-1] - at[:, None], 0), np.minimum(edges[1:] - at[:, None], dur[:, None])

    (a0, a1), (h0, h1) = part(age_edges, start), part(hour_edges, tod)
    return np.maximum(np.minimum(a1[:, :, None], h1[:, None]) - np.maximum(a0[:, :, None], h0[:, None]), 0)


def cells(groups, flights, episodes, tau_star, subset=None):
    """Per block: groups ending with each outcome and the member-hours at risk of it (both blocks x
    kinds x sizes x ages x classes), sizes 2..N_MAX; `tau_star` in s. `subset` (boolean per group)
    keeps some groups after their outcomes are found among all."""
    ev = events(groups, flights)
    if subset is not None:
        groups, ev = groups[subset], ev[subset]
    ep = episodes.set_index("episode_id").reindex(groups["episode_id"])
    start = groups["t_start"].to_numpy(float) - ep["t_start"].to_numpy(float)
    dur = (groups["t_end"] - groups["t_start"]).to_numpy(float)
    age_s, hour_s = np.array(AGE_MIN) * 60, np.array(HOUR_H) * 3600.0
    n = groups["n"].to_numpy()
    ids, k = np.unique(ep["block"], return_inverse=True)
    j = np.clip(n, 2, N_MAX) - 2
    tod = time_of_day(groups["t_start"]) * 3600
    late = np.maximum(start, tau_star) - start  # the part of the group before tau* is not at risk of LATE
    expo = np.zeros((len(ids), 2, N_MAX - 1, len(AGES), len(HOURS) - 1))
    for r, x in enumerate((exposure(start, dur, tod, age_s, hour_s),
                           exposure(start + late, np.maximum(dur - late, 0), tod + late, age_s, hour_s))):
        np.add.at(expo[:, r], (k, j), x * n[:, None, None] / 3600)
    expo = expo[:, np.isin(KINDS, LATE).astype(int)]
    o = pd.Categorical(ev["outcome"], OUTCOMES).codes
    a = np.searchsorted(age_s, start + dur, side="right") - 1
    h = np.searchsorted(hour_s, time_of_day(groups["t_end"]) * 3600, side="right") - 1
    counts = np.zeros((len(ids), len(KINDS), N_MAX - 1, len(AGES), len(HOURS) - 1))
    keep = o < len(KINDS)
    np.add.at(counts, (k[keep], o[keep], j[keep], a[keep], h[keep]), 1)
    return counts, expo


def partnership(groups, flights, intervals):
    """Per group: 'repeat' for two named paraglider pilots who share a pair interval on more than one
    day, 'once' for two who share one on this day only, '' otherwise."""
    who = named(flights)["pilot"]
    e = intervals[intervals["a"].isin(who.index) & intervals["b"].isin(who.index)]
    pa, pb = who.loc[e["a"]].to_numpy(), who.loc[e["b"]].to_numpy()
    days = pd.DataFrame({"p": np.minimum(pa, pb), "q": np.maximum(pa, pb), "date": e["date"].to_numpy()})
    n_days = days[days["p"] != days["q"]].drop_duplicates().groupby(["p", "q"]).size()
    out = np.full(len(groups), "", object)
    for i, (m, n) in enumerate(zip(groups["members"], groups["n"])):
        if n == 2 and m[0] in who.index and m[1] in who.index and who[m[0]] != who[m[1]]:
            p, q = sorted((who[m[0]], who[m[1]]))
            out[i] = "repeat" if n_days.get((p, q), 0) > 1 else "once"
    return out


def totals(x, size_axis):
    """`x` with the sum over the hour classes put first, then the sum over sizes (axis `size_axis`) last."""
    x = np.concatenate([x.sum(-1, keepdims=True), x], -1)
    return np.concatenate([x, x.sum(size_axis, keepdims=True)], size_axis)


def nest(x, *names):
    """Dicts keyed by `names` (one list per leading axis of `x`), the last axes as lists, NaN as None."""
    if not names:
        return np.where(np.isnan(x), None, x).tolist()
    return {name: nest(v, *names[1:]) for name, v in zip(names[0], x)}


def age_rates(counts, expo, rng):
    """Counts, member-hours, rates per member-hour with 5-95% bands, and the ratio of the rate in the
    last age bin to that in the first with exposure, by hour class, outcome, size class and age bin."""
    counts, expo = totals(counts, 2), totals(expo, 2)  # blocks x kinds x sizes x ages x hours
    nb = len(counts)
    w = rng.multinomial(nb, np.full(nb, 1 / nb), size=N_BOOT).astype(float)
    c, e = counts.sum(0), expo.sum(0)
    cd = (w @ counts.reshape(nb, -1)).reshape(N_BOOT, *c.shape)
    ed = (w @ expo.reshape(nb, -1)).reshape(N_BOOT, *e.shape)
    ok = e[:, -1, :, 0] >= MIN_HOURS  # all sizes, all day: the age bins with exposure to each outcome
    base = ok.argmax(-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        rate = np.where(e >= MIN_HOURS, c / e, np.nan)  # kinds x sizes x ages x hours
        draws = cd / ed
        ratio = np.stack([draws[:, q, :, -1] / draws[:, q, :, b] for q, b in enumerate(base)], 1)
        point = np.stack([rate[q, :, -1] / rate[q, :, b] for q, b in enumerate(base)])
    point = np.where(np.isfinite(point), point, np.nan)
    band = np.percentile(draws, [5, 95], axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # cells without exposure are all NaN
        ratio_band = np.nanpercentile(np.where(np.isfinite(ratio), ratio, np.nan), [5, 95], axis=0)
    shown, kept = np.isfinite(rate), np.isfinite(point)
    first = lambda x: np.moveaxis(x, -1, 0)  # hour classes first
    by_age = {"rate": rate, "band_5": np.where(shown, band[0], np.nan), "band_95": np.where(shown, band[1], np.nan),
              "counts": c, "member_hours": e}
    by_ratio = {"ratio": point, "ratio_5": np.where(kept, ratio_band[0], np.nan),
                "ratio_95": np.where(kept, ratio_band[1], np.nan)}
    return {"ages_min": AGES, "sizes": SIZES, "hours": HOURS, "kinds": list(KINDS), "n_blocks": nb,
            "ratio_of": {kind: f"{AGES[-1]} / {AGES[b]}" for kind, b in zip(KINDS, base)},
            **{k: nest(first(v), HOURS, KINDS, SIZES) for k, v in {**by_age, **by_ratio}.items()}}


def plot(r, path):
    """Rate per member per hour against the age of the episode, all day: one line per size class with
    its 5-95% band, drawn when it rests on at least PLOT_MIN_EVENTS events; panels joins, leaves in
    flight and landings (shaded: before tau*)."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    x = np.arange(len(AGES))
    for ax, kind in zip(axes, PLOTTED):
        for s in PLOT_SIZES + ["all"]:
            get = lambda key: np.array(r[key]["all day"][kind][s], float)
            if np.nansum(get("counts")) < PLOT_MIN_EVENTS:
                continue
            ax.fill_between(x, get("band_5"), get("band_95"), color=COLORS[s], alpha=0.15, lw=0)
            ax.plot(x, get("rate"), color=COLORS[s], marker="o", ms=3.5, lw=1.2, ls=":" if s == "all" else "-",
                    label="all sizes" if s == "all" else s)
        if kind in LATE:
            ax.axvspan(-0.5, 1.5, color="#e5e5e5", lw=0)
            ax.text(0.5, 0.04, r"before $\tau^*$", ha="center", transform=ax.get_xaxis_transform(), color="#52514e")
        ax.set(yscale="log", xlabel="age of the episode (min)", title=TITLES[kind])
        ax.set_xticks(x, AGES)
        ax.minorticks_off()
    axes[0].set_ylabel("per member per hour")
    axes[0].legend(frameon=False, fontsize=7, title="pilots", title_fontsize=7, ncol=2)
    save(fig, path)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    tau_star = json.loads((LABELS / "calibration.json").read_text())["tau_star"]
    groups, flights = pd.read_parquet(LABELS / "groups.parquet"), pd.read_parquet(LABELS / "flights.parquet")
    episodes = pd.read_parquet(LABELS / "episodes.parquet")
    counts, expo = cells(groups, flights, episodes, tau_star)
    result = {"tau_star_min": tau_star / 60, **age_rates(counts, expo, np.random.default_rng(SEED))}
    # selection or bond: pairs that meet on several days against one-off pairs
    kind = partnership(groups, flights, pd.read_parquet(LABELS / "pair_intervals.parquet"))
    result["partners"] = {c: age_rates(*cells(groups, flights, episodes, tau_star, kind == c),
                                       np.random.default_rng(SEED)) for c in ("repeat", "once")}
    (OUT / "age.json").write_text(json.dumps(result, indent=1))
    plot(result, OUT / "age.pdf")
    for kind in PLOTTED:
        for s in SIZES:
            print(kind, s, np.round(np.array(result["rate"]["all day"][kind][s], float), 3), flush=True)
    for c, r in result["partners"].items():
        for h in HOURS:
            print("leave pairs", c, h, np.round(np.array(r["rate"][h]["leave"]["2"], float), 3),
                  np.round(np.array(r["member_hours"][h]["leave"]["2"], float), 0), flush=True)
