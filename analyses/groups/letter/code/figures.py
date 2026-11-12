"""Figures of the letter, from the JSON of the analyses (raw.py, matched.py, prepost.py, all with --open).
Fig. 1 (design) reads the run's labels, the pieces cache of prepost.py --open and the fixes of one episode.

    uv run python letter/code/figures.py [--run NAME] [--only NAME]    # from analyses/groups; writes letter/figures/
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from matplotlib.ticker import NullFormatter, NullLocator

HERE = Path(__file__).resolve().parents[1]
GROUPS = HERE.parent
sys.path.insert(0, str(GROUPS))
import prepost  # noqa: E402
import tamsd  # noqa: E402
from openflights import MIN_OPENING, open_flights  # noqa: E402
from phase_msd import RUNS  # noqa: E402
from tamsd import COLORS  # noqa: E402

GROUP, ALONE, CONTROL = COLORS["group"], COLORS["group_solo"], "#8a8a8a"
WINDOW = 2400  # s, the windows of prepost.py --open quoted in the letter
DATASET = "1000"  # the dataset of raw.py and matched.py whose pieces give H_loc(1000 s)
STAGES = ("pre", "start", "end", "post")
XMIN_TICKS = (300, 500, 1000, 2000, 4000)  # s, labelled thresholds of the Hill panel

plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
                     "ytick.major.width": 0.6, "legend.frameon": False, "legend.fontsize": 7})


def crossing(curve):
    """The first lag after the last one where the curve is not above 0, as vs-solo/code/comparisons.py."""
    above = np.asarray(curve["value"]) > 0
    return curve["lags"][int(np.flatnonzero(~above)[-1]) + 1]


def draw(ax, lags, entry, color, label, dashed=False):
    lo, hi = entry["band"]
    ax.fill_between(lags, lo, hi, color=color, alpha=0.18, lw=0)
    ax.plot(lags, entry["value"], color=color, lw=1.4, ls="--" if dashed else "-", label=label)


def persistence(folder, out):
    raw = json.loads((folder / "raw.json").read_text())["gaps"][DATASET]["H_loc"]
    matched = json.loads((folder / "matched.json").read_text())["levels"]["all"]["delta curves"]
    prepost = json.loads((folder / f"prepost_{WINDOW}.json").read_text())
    lags_pp, diff = prepost["lags"], prepost["differences"]

    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.3), sharey=True, constrained_layout=True)
    draw(axes[0], raw["G - S"]["lags"], raw["G - S"], GROUP, "G $-$ S")
    draw(axes[0], raw["A - S"]["lags"], raw["A - S"], ALONE, "A $-$ S", dashed=True)
    for gap, color, dashed in (("G - S|G", GROUP, False), ("A - S|A", ALONE, True)):
        main = matched[gap]["main"]
        draw(axes[1], main["lags"], main, color, gap.replace("-", "$-$").replace("|", "$|$"), dashed)
    draw(axes[2], lags_pp, diff["g"]["dH_loc"], GROUP, "$g$, group")
    draw(axes[2], lags_pp, diff["d w"]["dH_loc"], CONTROL, "drift, $d_w$", dashed=True)

    cross = crossing(raw["G - S"])
    for ax, title in zip(axes, ("(a) raw", "(b) same route", "(c) same flight")):
        ax.set_xscale("log")
        ax.set_xlim(10, 2000)
        ax.axhline(0, color="#bbbbbb", lw=0.6)
        ax.axvline(cross, color="#888888", lw=0.7, ls=":")
        ax.set_title(title, fontsize=8)
        ax.set_xlabel(r"$\tau$ (s)")
        ax.legend(loc="lower right", handlelength=1.8)
    axes[0].set_ylabel(r"$\Delta H_\mathrm{loc}(\tau)$")

    stages = prepost["curves"]["stages"]
    inset = axes[2].inset_axes([0.17, 0.63, 0.34, 0.28])
    values = [stages[s]["H_loc(1000 s)"] for s in STAGES]
    bands = np.array([stages[s]["H_loc(1000 s) band"] for s in STAGES]).T
    x = np.arange(len(STAGES))
    inset.errorbar(x, values, yerr=[np.subtract(values, bands[0]), np.subtract(bands[1], values)], color=GROUP,
                   lw=1, marker="o", ms=3, capsize=0)
    inset.set_xticks(x, STAGES, fontsize=6)
    inset.tick_params(labelsize=6, length=2)
    inset.set_title(r"$H_\mathrm{loc}(1000\ \mathrm{s})$", fontsize=6, pad=2)

    out.mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"persistence.{ext}", dpi=300)
    plt.close(fig)


def tail(folder, out):
    t = json.loads((folder / "tail.json").read_text())
    colors = {"S": COLORS["solo"], "A": ALONE, "G": GROUP}
    xmin = t["transition"]["xmin of S"]
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.3), constrained_layout=True)

    ax = axes[0]
    for pop in ("S", "A", "G"):
        s = t["survival"][pop]
        ax.fill_between(s["grid"], *s["band"], color=colors[pop], alpha=0.18, lw=0)
        ax.plot(s["grid"], s["value"], color=colors[pop], lw=1.4, ls="--" if pop == "A" else "-", label=pop)
    ax.set(xscale="log", yscale="log", xlim=(100, 1e4), ylim=(1e-5, 1.5), xlabel=r"$\tau$ (s)",
           ylabel=r"$P(\tau_\mathrm{transition} > \tau)$")
    ax.legend(loc="lower left")

    ax = axes[1]
    for pop in ("S", "A", "G"):
        c = t["transition"]["populations"][pop]["curve"]
        ax.fill_between(c["xmin"], *c["band"], color=colors[pop], alpha=0.18, lw=0)
        ax.plot(c["xmin"], c["value"], color=colors[pop], lw=1.4, ls="--" if pop == "A" else "-", label=pop)
    ax.set(xscale="log", xlim=(300, 4000), ylim=(0, 8), xlabel=r"$x_\mathrm{min}$ (s)", ylabel=r"Hill $\mu$")
    ax.set_xticks(XMIN_TICKS, [f"{x:g}" for x in XMIN_TICKS])
    ax.xaxis.set_minor_formatter(NullFormatter())
    g = t["transition"]["populations"]["G"]["curve"]
    top = ax.secondary_xaxis("top")
    shown = [i for i, x in enumerate(g["xmin"]) if x <= 4000][::4]
    top.set_xticks([g["xmin"][i] for i in shown], [f"{g['runs'][i]:,}" for i in shown], fontsize=6)
    top.xaxis.set_minor_locator(NullLocator())
    top.set_xlabel("transitions of G above", fontsize=7)
    ax.legend(loc="lower right")

    ax = axes[2]
    classes, labels = ("0 s", "1000 s", "2000 s"), ("all", r"$\geq$1000 s", r"$\geq$2000 s")
    for k, pop in enumerate(("S", "A", "G")):
        d = [t["descriptors"][pop][c]["height lost (m/km)"] for c in classes]
        x = np.arange(len(classes)) + (k - 1) * 0.22
        med = [v["median"] for v in d]
        lo, hi = np.array([v["median band"] for v in d]).T
        ax.errorbar(x, med, yerr=[np.subtract(med, lo), np.subtract(hi, med)], color=colors[pop], marker="o",
                    ms=3.5, lw=0, elinewidth=1.2, label=pop)
    ax.set_xticks(range(len(classes)), labels)
    ax.set(ylabel="height lost (m per km)", xlabel="transition duration")
    ax.axhline(0, color="#bbbbbb", lw=0.6)
    ax.legend(loc="upper right")

    for ax, letter, x in zip(axes, "abc", (0.88, 0.03, 0.03)):
        ax.text(x, 0.96, f"({letter})", transform=ax.transAxes, va="top", fontsize=8)
    for ax in axes[:2]:
        ax.axvline(xmin, color="#888888", lw=0.7, ls=":")
    for ext in ("pdf", "png"):
        fig.savefig(out / f"tail.{ext}", dpi=300)
    plt.close(fig)


def survival(folder, out):
    """Survival of the transition durations of S, A and G (raw design), with the threshold of S: one column."""
    t = json.loads((folder / "tail.json").read_text())
    colors = {"S": COLORS["solo"], "A": ALONE, "G": GROUP}
    fig, ax = plt.subplots(figsize=(3.4, 2.5), constrained_layout=True)
    for pop in ("S", "A", "G"):
        s = t["survival"][pop]
        ax.fill_between(s["grid"], *s["band"], color=colors[pop], alpha=0.18, lw=0)
        ax.plot(s["grid"], s["value"], color=colors[pop], lw=1.4, ls="--" if pop == "A" else "-", label=pop)
    xmin = t["transition"]["xmin of S"]
    ax.axvline(xmin, color="#888888", lw=0.7, ls=":")
    ax.text(xmin * 1.08, 0.5, rf"$x_\mathrm{{min}}$ = {xmin:.0f} s", fontsize=6.5, color="#555555")
    ax.set(xscale="log", yscale="log", xlim=(100, 1e4), ylim=(1e-5, 1.5), xlabel=r"$\tau$ (s)",
           ylabel=r"$P(\tau_\mathrm{transition} > \tau)$")
    ax.legend(loc="lower left")
    for ext in ("pdf", "png"):
        fig.savefig(out / f"survival.{ext}", dpi=300)
    plt.close(fig)


def relative(g, pre, post):
    """g as a percentage of the level outside the group, the mean of the windows before and after."""
    level = (pre["value"] + post["value"]) / 2
    return 100 * g["value"] / level, [100 * b / level for b in g["band"]]


def forest(ax, rows, color_of):
    """Rows of (label, value, band) from the top, a marker with its band each."""
    for k, (label, value, band, key) in enumerate(rows):
        ax.errorbar(value, -k, xerr=[[value - band[0]], [band[1] - value]], color=color_of(key), marker="o",
                    ms=3.5, lw=0, elinewidth=1.2)
    ax.set_yticks([-k for k in range(len(rows))], [r[0] for r in rows])
    ax.axvline(0, color="#bbbbbb", lw=0.6)
    ax.set_ylim(-len(rows) + 0.4, 0.6)


def how(folder, out):
    groups_out = folder.parent
    h = json.loads((groups_out / "headings.json").read_text())
    prepost = json.loads((folder / f"prepost_{WINDOW}.json").read_text())
    terrain = json.loads((folder / "terrain_compare.json").read_text())
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.4), constrained_layout=True,
                             gridspec_kw={"width_ratios": [1.1, 1, 1]})

    ax = axes[0]
    tau = np.asarray(h["tau"], dtype=float)
    pairs = h["classes"]["pairs"]
    shades = plt.cm.Oranges(np.linspace(0.45, 0.95, len(pairs)))
    for c, shade in zip(pairs, shades):
        ell = c["l_median"]["value"]
        curve = c["heading"]["curve"]
        x = tau - ell
        ax.fill_between(x, [p["band"][0] for p in curve], [p["band"][1] for p in curve], color=shade, alpha=0.2, lw=0)
        ax.plot(x, [p["value"] for p in curve], color=shade, lw=1.3, label=rf"$\ell \approx {ell:.0f}$ s")
    ax.axvline(0, color="#888888", lw=0.7, ls=":")
    ax.set(xlabel=r"$\tau - \ell$ (s)", ylabel=r"heading correlation $C(\tau)$")
    ax.legend(loc="upper left", fontsize=6, handlelength=1.4)

    stats = (("searches that end in a climb (%)", "searches ending\nin a climb"), ("time in climb (%)", "time in climb"),
             ("climb rate (m/s)", "climb rate"), ("glide speed (km/h)", "glide speed"),
             ("height lost per km of glide (m/km)", "height lost\nper km"))
    stages, g = prepost["phase statistics"]["stages"], prepost["differences"]["g"]
    rows = [(label, *relative(g[key], stages["pre"][key], stages["post"][key]), "G") for key, label in stats]
    forest(axes[1], rows, lambda _: GROUP)
    axes[1].set_xlabel("change in the group (%)")

    raw4, flight = terrain["raw"]["four classes"], terrain["flight"]
    rows = [("raw, plains and hills", *quote(raw4["hills"]["H_loc(1000 s) gaps"]["G - S"]), "low"),
            ("raw, low mountains", *quote(raw4["low mountains"]["H_loc(1000 s) gaps"]["G - S"]), "mid"),
            ("raw, high mountains", *quote(raw4["high mountains"]["H_loc(1000 s) gaps"]["G - S"]), "high"),
            ("flight, plains and hills", *quote(flight["lowland"]["effects"]["g"]["H_loc(1000 s)"]), "low"),
            ("flight, mountains", *quote(flight["mountain"]["effects"]["g"]["H_loc(1000 s)"]), "high")]
    relief = dict(zip(("low", "mid", "high"), plt.cm.Greys(np.linspace(0.45, 0.95, 3))))
    forest(axes[2], rows, relief.get)
    axes[2].set_xlabel(r"$\Delta H_\mathrm{loc}(1000\ \mathrm{s})$")

    for ax, letter in zip(axes, "abc"):
        ax.set_title(f"({letter})", fontsize=8, loc="left")
    for ext in ("pdf", "png"):
        fig.savefig(out / f"how.{ext}", dpi=300)
    plt.close(fig)


def quote(entry):
    return entry["value"], entry["band"]


# Fig. 1: the episode drawn is the one of the design that satisfies these, closest to the median duration
MEMBERS = 3  # pilots in the group during the focal stretch, the focal pilot included
MIN_CLIMBS = 2  # climbs of the focal pilot inside the stretch
MIN_PRESENT = 0.5  # share of the stretch each member is in the group with the focal pilot
MEMBER_LEAVES = True  # some member joins late or leaves early, so that the membership strip shows something
MAX_GAP_S = 5  # longest hole in the fixes of the plotted tracks
MAX_INTERPOLATED = 0.05  # share of interpolated fixes in the plotted tracks
FIX_COLUMNS_MAP = ["flight_id", "segment_id", "t", "E", "N", "interpolated"]
SOLO_COLOR, MEMBER = "#b3b3b3", "#7a2e0e"
WINDOW_ROWS = (("pre", "before"), ("start", "start"), ("end", "end"), ("post", "after"))


def design_events():
    """Events of the same-flight design (prepost.py --open) that have the four windows measured in its cache:
    flight, window stratum, t_in, t_out (focal flight clock, s), duration, with the tables they come from."""
    run = tamsd.LABELS
    flights = pd.read_parquet(run / "flights.parquet")
    keep = open_flights(run, tamsd.SHARED / "flight_ends.parquet", tamsd.FIXES["para"].parent, True)
    flights = flights[flights["flight"].isin(keep)]
    mem = pd.read_parquet(run / "membership.parquet")
    spans = tamsd.stretches(flights, mem, pd.read_parquet(run / "flight_summary.parquet"))
    spans = spans.sort_values(["flight", "a"], kind="stable").reset_index(drop=True)
    wins = prepost.windows(spans, np.random.default_rng(tamsd.SEED), WINDOW)
    events = prepost.event_times(wins)
    cache = run / f"analyses_open_min{MIN_OPENING:g}" / f"prepost_pieces_{WINDOW}_lag{prepost.window_lags(WINDOW).max()}.npz"
    pieces = np.load(cache)
    made = pd.DataFrame({k: pieces[f"para/{k}"] for k in ("population", "episode")}).drop_duplicates()
    four = made[made["population"].isin(prepost.STAGES)].groupby("episode")["population"].nunique()
    events = events[events["episode"].isin(four.index[four == len(prepost.STAGES)])].copy()
    events["dur"] = events["t_out"] - events["t_in"]
    return events, flights.set_index("flight"), mem[mem["state"] == "G"]


def companions(event, flights, mem):
    """The other pilots in the group of the focal stretch: rows (flight, t_start, t_end) of their group time, clipped
    to the stretch, on the focal clock."""
    u0 = flights.at[event.flight, "utc0"]
    a, b = u0 + event.t_in, u0 + event.t_out
    mine = mem[(mem["flight"] == event.flight) & (mem["t_end"] > a) & (mem["t_start"] < b)]
    rows = mem[mem["group_id"].isin(mine["group_id"].unique()) & (mem["flight"] != event.flight)
               & (mem["t_end"] > a) & (mem["t_start"] < b)]
    return pd.DataFrame({"flight": rows["flight"], "t_start": np.maximum(rows["t_start"], a) - u0,
                         "t_end": np.minimum(rows["t_end"], b) - u0})


def read_tracks(flight_ids):
    t = pq.read_table(tamsd.FIXES["para"], columns=FIX_COLUMNS_MAP, filters=[("flight_id", "in", list(flight_ids))])
    return {k: g.sort_values("t") for k, g in t.to_pandas().groupby("flight_id")}


def clean(x):
    """Whether the fixes `x` form one segment with no hole above MAX_GAP_S and few interpolated fixes."""
    return (len(x) > 1 and x["segment_id"].nunique() == 1 and np.diff(x["t"].to_numpy()).max() <= MAX_GAP_S
            and x["interpolated"].mean() <= MAX_INTERPOLATED)


def choose_episode():
    """The first event, by closeness of its duration to the median of the design's events, that meets the constants
    above. Returns it with everything the drawing needs."""
    events, flights, mem = design_events()
    median = events["dur"].median()
    print(f"design events with the four windows: {len(events)}, median stretch duration {median:.0f} s")
    for event in events.iloc[(events["dur"] - median).abs().argsort()].itertuples():
        comp = companions(event, flights, mem)
        others = comp["flight"].unique()
        if len(others) != MEMBERS - 1 or not all(
                flights.at[m, "discipline"] == "para" and flights.at[m, "dt_native"] <= tamsd.H_MAX
                and flights.at[m, "labelled"] for m in (event.flight, *others)):
            continue
        present = comp.assign(d=comp["t_end"] - comp["t_start"]).groupby("flight")["d"].sum() / event.dur
        if present.min() < MIN_PRESENT or (MEMBER_LEAVES and present.min() >= 1 - 1e-9):
            continue
        ids = {m: str(flights.at[m, "flight_id"]) for m in (event.flight, *others)}
        tracks = read_tracks(ids.values())
        u0 = {m: flights.at[m, "utc0"] for m in ids}
        lo, hi = event.t_in - WINDOW, event.t_out + WINDOW
        focal = tracks[ids[event.flight]]
        focal = focal[(focal["t"] >= lo) & (focal["t"] <= hi)]
        inside = lambda m: tracks[ids[m]][(tracks[ids[m]]["t"] + u0[m] - u0[event.flight] >= event.t_in)
                                          & (tracks[ids[m]]["t"] + u0[m] - u0[event.flight] <= event.t_out)]
        if not (clean(focal) and all(clean(inside(m)) for m in others)):
            continue
        runs = pq.read_table(RUNS["para"], columns=["phase", "t_start", "t_end"],
                             filters=[("flight_id", "=", ids[event.flight])]).to_pandas()
        climbs = runs[runs["phase"] == "climb"]
        n_in = ((climbs["t_start"] >= event.t_in) & (climbs["t_end"] <= event.t_out)).sum()
        if n_in < MIN_CLIMBS:
            continue
        print(f"episode chosen: flight {event.flight} (stratum {event.episode}), members {sorted(others)}, "
              f"duration {event.dur:.0f} s, {n_in} climbs inside, presence {present.round(2).tolist()}")
        return event, flights, comp, tracks, ids, climbs
    raise SystemExit("no episode meets the criteria")


def design(folder, out):
    """Fig. 1: (a) the map of one group episode, (b) its members over time with the four windows of the same-flight design."""
    event, flights, comp, tracks, ids, climbs = choose_episode()
    u0 = flights.at[event.flight, "utc0"]
    t0 = flights.at[event.flight, "takeoff"] - u0
    x = tracks[ids[event.flight]]
    x = x[(x["t"] >= event.t_in - WINDOW) & (x["t"] <= event.t_out + WINDOW)]
    t, e, n = (x[c].to_numpy(float) for c in ("t", "E", "N"))
    e0, n0 = np.interp(event.t_in, t, e), np.interp(event.t_in, t, n)
    km = lambda v, v0: (v - v0) / 1000
    together = (t >= event.t_in) & (t <= event.t_out)
    others = sorted(ids.keys() - {event.flight})
    drawn = False

    fig = plt.figure(figsize=(3.4, 5.0))
    top = fig.add_axes([0.17, 0.46, 0.78, 0.49])
    top.plot(km(e, e0), km(n, n0), color=SOLO_COLOR, lw=0.9, solid_capstyle="round", label="focal pilot, alone")
    for m in others:
        y = tracks[ids[m]]
        tm = y["t"].to_numpy(float) + flights.at[m, "utc0"] - u0
        for r in comp[comp["flight"] == m].itertuples():
            s = (tm >= r.t_start) & (tm <= r.t_end)
            top.plot(km(y["E"].to_numpy(float)[s], e0), km(y["N"].to_numpy(float)[s], n0), color=MEMBER, lw=0.45, zorder=3,
                     label="other members" if not drawn else None)
            drawn = True
    top.plot(km(e[together], e0), km(n[together], n0), color=GROUP, lw=1.5, label="focal pilot, in the group")
    inwin = climbs[(climbs["t_end"] >= t[0]) & (climbs["t_start"] <= t[-1])]
    centers = [(km(e[(t >= r.t_start) & (t <= r.t_end)].mean(), e0), km(n[(t >= r.t_start) & (t <= r.t_end)].mean(), n0))
               for r in inwin.itertuples()]
    top.scatter(*zip(*centers), s=16, color="#222222", edgecolor="white", linewidth=0.5, zorder=5, label="climbs")
    top.set_aspect("equal", adjustable="datalim")
    top.set_xlabel("east (km)")
    top.set_ylabel("north (km)")
    top.set_title("(a)", fontsize=8, loc="left", x=-0.17)
    top.legend(loc="lower right", handlelength=1.6, fontsize=6.5, borderaxespad=0.2)

    bot = fig.add_axes([0.2, 0.11, 0.75, 0.25])
    minutes = lambda s: (np.asarray(s, dtype=float) - t0) / 60
    rows = [w for w, _ in WINDOW_ROWS] + ["focal", *others]
    labels = [lab for _, lab in WINDOW_ROWS] + ["focal pilot", *(f"pilot {k + 2}" for k in range(len(others)))]
    spans = {"pre": (event.t_in - WINDOW, event.t_in), "start": (event.t_in, event.t_in + WINDOW),
             "end": (event.t_out - WINDOW, event.t_out), "post": (event.t_out, event.t_out + WINDOW),
             "focal": (event.t_in, event.t_out)}
    bot.axvspan(*minutes((event.t_in, event.t_out)), color=GROUP, alpha=0.1, lw=0)
    for k, key in enumerate(rows):
        if key in spans:
            a, b = spans[key]
            color = prepost.LINES[key][0] if key in prepost.LINES else GROUP
            bot.barh(-k, (b - a) / 60, left=minutes(a), height=0.62, color=color, lw=0)
        else:
            for r in comp[comp["flight"] == key].itertuples():
                bot.barh(-k, (r.t_end - r.t_start) / 60, left=minutes(r.t_start), height=0.62, color=GROUP, lw=0)
    gap = len(WINDOW_ROWS) - 0.5
    bot.axhline(-gap, color="#bbbbbb", lw=0.5)
    bot.set_yticks([-k for k in range(len(rows))], labels)
    bot.set_ylim(-len(rows) + 0.5, 0.5)
    bot.set_xlim(*minutes((event.t_in - WINDOW, event.t_out + WINDOW)))
    bot.set_xlabel("time since takeoff of the focal pilot (min)")
    bot.set_title("(b)", fontsize=8, loc="left", x=-0.2)
    bot.tick_params(axis="y", length=0)
    for side in ("top", "right"):
        bot.spines[side].set_visible(False)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"design.{ext}", dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="d120_eh500_ez300_th45_g120_q0.1")
    parser.add_argument("--only", choices=("persistence", "tail", "survival", "how", "design"))
    args = parser.parse_args()
    run = args.run
    folder = GROUPS / "out" / run / f"open_min{MIN_OPENING:g}"
    for name in ("persistence", "tail", "survival", "how", "design"):
        if args.only in (None, name):
            globals()[name](folder, HERE / "figures")


if __name__ == "__main__":
    main()
