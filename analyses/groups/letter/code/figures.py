"""Figures of the letter and of its supplement (sm_*), from the JSON of the analyses (raw.py, matched.py, prepost.py,
all with --open). Fig. 1 (design) reads the run's labels, the pieces cache of prepost.py --open and the fixes of one
episode; sm_opening reads the flight tables of the run for the histogram of the opening. alpha = 2 x the stored H_loc.

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
XMIN_TICKS = (300, 500, 1000, 2000, 4000)  # s, labelled thresholds of the Hill panel

plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
                     "ytick.major.width": 0.6, "legend.frameon": False, "legend.fontsize": 7})


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
    marking = json.loads((groups_out / "marking.json").read_text())
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
    for c, shade in zip(h["classes"]["chance"], shades):
        curve = c["heading"]["curve"]
        ax.plot(tau - c["l_median"]["value"], [p["value"] for p in curve], color=shade, lw=1.1, ls="--")
    ax.plot([], [], color="#555555", lw=1.1, ls="--", label="surrogate pairs")
    ax.axvline(0, color="#888888", lw=0.7, ls=":")
    ax.set(xlabel=r"$\tau - \ell$ (s)", ylabel=r"heading correlation $C(\tau)$")
    ax.legend(loc="upper left", fontsize=7, handlelength=1.4)

    stats = (("searches that end in a climb (%)", "searches ending\nin a climb"), ("time in climb (%)", "time in climb"),
             ("climb rate (m/s)", "climb rate"), ("glide speed (km/h)", "glide speed"),
             ("height lost per km of glide (m/km)", "height lost\nper km"))
    stages = prepost["phase statistics"]["stages"]
    g = prepost["mock"]["variants"]["any site"]["real minus mock"]["g"]
    rows = [(label, *relative(g[key], stages["pre"][key], stages["post"][key]), "G") for key, label in stats]
    forest(axes[1], rows, lambda _: GROUP)
    axes[1].set_xlabel("change in the group, net of mock events (%)")

    ax = axes[2]
    shifts = np.asarray(marking["params"]["shifts_min"], dtype=float)
    for who, label in (("group", "group members"), ("solo", "solo pilots")):
        curve = marking["curve"][who]["any"]
        m0 = curve[0]["value"]
        ax.fill_between(shifts, [c["band"][0] / m0 for c in curve], [c["band"][1] / m0 for c in curve],
                        color=POP_COLOR["G" if who == "group" else "S"], alpha=0.2, lw=0)
        ax.plot(shifts, [c["value"] / m0 for c in curve], color=POP_COLOR["G" if who == "group" else "S"], lw=1.3,
                marker="o", ms=2.5, label=label)
    ax.axhline(0.5, color="#888888", lw=0.7, ls=":")
    ax.set(xlabel="shift of the markers (min)", ylabel="marked share / at no shift", ylim=(0, 1.05))
    ax.legend(loc="upper right", fontsize=7, handlelength=1.4)

    for ax, letter in zip(axes, "abc"):
        ax.set_title(f"({letter})", fontsize=8, loc="left")
    for ext in ("pdf", "png"):
        fig.savefig(out / f"how.{ext}", dpi=300)
    plt.close(fig)


def quote(entry, scale=1):
    """Value and band of a {value, band} entry times `scale` (2: alpha = 2 H_loc)."""
    return scale * entry["value"], [scale * b for b in entry["band"]]


# Fig. 1: one episode, fixed. The images are made by the viewer (xc-viewer, scripts/episode_snapshot.py) on the orthophoto
# of IGN; CACHE is any folder outside the data folder (it receives a copy of the route index and the downloaded tiles).
DESIGN_EPISODE, DESIGN_FLIGHT = 11462, 114542  # episode of the viewer (groups/episodes.parquet) and its focal flight
DESIGN_WINDOWS = (3450, 2400)  # s, the main window of the design, then the one drawn if the episode has no valid event at it
DESIGN_VIEWER = "cd ../../xc-viewer && uv run python scripts/episode_snapshot.py 11462"  # relative to the xc-lab folder
DESIGN_COMMON = ("--root /Volumes/SSD_DATA/xc-viewer-data --cache $CACHE --hide-panels --no-numbers --no-markers "
                 "--surface aerial --aerial-side 5000 --mute 0.8")
DESIGN_COMMANDS = {  # the commands that wrote figures/design_overview.png and figures/design_close.png (viewer branch feature/group-episodes)
    "design_overview.png": f"{DESIGN_VIEWER} OUT {DESIGN_COMMON} --size 4200 1500 --extent whole --width 9 --outline 2.5 "
                           "--elevation 30 --zoom 0.47 --vertical 1.6 --clip 1648127355 1648131730 --azimuth -107 "
                           "--utc 1648128860 --arrows 114542 5000 --arrow-px 64",
    "design_close.png": f"{DESIGN_VIEWER} OUT {DESIGN_COMMON} --size 3600 2400 --clip 1648128500 1648128900 "
                        "--centre 300 150 2080 --extent trail --width 12 --outline 3 --trail 5 --utc 1648128875 "
                        "--azimuth -120 --elevation 26 --distance 2700 --hide 114534 --detail 4000 500 5000"}
DESIGN_WIDTH, DESIGN_HEIGHTS = 7.0, (2.5, 2.35)  # in: the figure; the overview, then the close-up and the strip
DESIGN_FONT = 7  # pt, at the size of the figure


def design_events(window):
    """Events of the same-flight design (prepost.py --open) that have the four windows of `window` s measured in its
    cache: flight, window stratum, t_in, t_out (focal flight clock, s), duration, with the tables they come from."""
    run = tamsd.LABELS
    flights = pd.read_parquet(run / "flights.parquet")
    keep = open_flights(run, tamsd.SHARED / "flight_ends.parquet", tamsd.FIXES["para"].parent, True)
    flights = flights[flights["flight"].isin(keep)]
    mem = pd.read_parquet(run / "membership.parquet")
    spans = tamsd.stretches(flights, mem, pd.read_parquet(run / "flight_summary.parquet"))
    spans = spans.sort_values(["flight", "a"], kind="stable").reset_index(drop=True)
    wins = prepost.windows(spans, np.random.default_rng(tamsd.SEED), window)
    events = prepost.event_times(wins)
    cache = run / f"analyses_open_min{MIN_OPENING:g}" / f"prepost_pieces_{window}_lag{prepost.window_lags(window).max()}.npz"
    pieces = np.load(cache)
    made = pd.DataFrame({k: pieces[f"para/{k}"] for k in ("population", "episode")}).drop_duplicates()
    four = made[made["population"].isin(prepost.STAGES)].groupby("episode")["population"].nunique()
    events = events[events["episode"].isin(four.index[four == len(prepost.STAGES)])].copy()
    return events, wins, flights.set_index("flight"), mem[mem["state"] == "G"]


def design_episode():
    """The fixed episode of Fig. 1: the window length drawn, the event (flight, stratum, t_in, t_out on the clock of the
    focal flight), its four windows [a, b) by name, the clock origin t0 (takeoff of the focal flight), and the time in
    the group of each other member, clipped to the stretch. The first of DESIGN_WINDOWS that has the event is used."""
    for window in DESIGN_WINDOWS:
        events, wins, flights, mem = design_events(window)
        u0 = flights.at[DESIGN_FLIGHT, "utc0"]
        mine = mem[(mem["flight"] == DESIGN_FLIGHT) & (mem["episode_id"] == DESIGN_EPISODE)]
        e = events[(events["flight"] == DESIGN_FLIGHT) & (events["t_out"] + u0 > mine["t_start"].min())
                   & (events["t_in"] + u0 < mine["t_end"].max())]
        if len(e):
            break
    else:
        raise SystemExit(f"flight {DESIGN_FLIGHT} has no event of the design at {DESIGN_WINDOWS} s")
    event = e.iloc[0]
    w = wins[(wins["flight"] == DESIGN_FLIGHT) & (wins["episode"] == event["episode"])]
    windows = {r.population: (r.a, r.b) for r in w.itertuples() if r.population in prepost.STAGES}
    a, b = u0 + event["t_in"], u0 + event["t_out"]
    rows = mem[mem["group_id"].isin(mine[(mine["t_end"] > a) & (mine["t_start"] < b)]["group_id"]) & (mem["flight"] != DESIGN_FLIGHT)
               & (mem["t_end"] > a) & (mem["t_start"] < b)]
    comp = pd.DataFrame({"flight": rows["flight"], "t_start": np.maximum(rows["t_start"], a) - u0,
                         "t_end": np.minimum(rows["t_end"], b) - u0})
    t0 = flights.at[DESIGN_FLIGHT, "takeoff"] - u0
    return window, event, windows, t0, comp


def design(folder, out):
    """Fig. 1 (two columns): (a) the group stretch on the orthophoto, the direction of travel shown by arrowheads;
    (b) the shared thermal of the same group; (c) the members and the four windows of the same-flight design against
    the time since takeoff of the focal pilot. (a) and (b) are the viewer's images of figures/DESIGN_COMMANDS."""
    window, event, windows, t0, comp = design_episode()
    print(f"design: episode {DESIGN_EPISODE}, windows of {window} s")
    members = sorted({DESIGN_FLIGHT, *comp["flight"].unique()})
    tab20 = plt.get_cmap("tab20")  # the colours of the viewer: its members in the order of their flights
    color = {m: tab20(2 * k) for k, m in enumerate(members)}
    others = [m for m in members if m != DESIGN_FLIGHT]
    names = {DESIGN_FLIGHT: "focal pilot", **{m: f"pilot {k + 2}" for k, m in enumerate(others)}}
    here = Path(out)
    top, bottom = (plt.imread(here / n) for n in ("design_overview.png", "design_close.png"))
    h_top, h_bottom = DESIGN_HEIGHTS
    total = h_top + h_bottom + 0.08
    fig = plt.figure(figsize=(DESIGN_WIDTH, total))
    half = DESIGN_WIDTH / 2
    axes = {"a": fig.add_axes([0, 1 - h_top / total, 1, h_top / total]),
            "b": fig.add_axes([0, 0, (half - 0.05) / DESIGN_WIDTH, h_bottom / total])}
    for key, image in (("a", top), ("b", bottom)):
        ax = axes[key]
        ax.imshow(image, aspect="auto", interpolation="lanczos")
        ax.set(xticks=[], yticks=[])
        for side in ax.spines.values():
            side.set_linewidth(0.4)
            side.set_color("#999999")
    strip = fig.add_axes([(half + 0.95) / DESIGN_WIDTH, 0.5 / total, (half - 1.05) / DESIGN_WIDTH, (h_bottom - 0.65) / total])
    minutes = lambda s: (np.asarray(s, dtype=float) - t0) / 60
    t_in, t_out = event["t_in"], event["t_out"]
    lo, hi = windows["pre"][0], windows["post"][1]
    strip.axvspan(*minutes((t_in, t_out)), color="#f0ece6", lw=0, zorder=0)
    rows = [-0.9 * k for k in range(len(members))]
    for y, m in zip(rows, [DESIGN_FLIGHT, *others]):
        spans = [(t_in, t_out)] if m == DESIGN_FLIGHT else list(zip(*(comp[comp["flight"] == m][k] for k in ("t_start", "t_end"))))
        for a, b in spans:
            strip.barh(y, (b - a) / 60, left=minutes(a), height=0.62, color=color[m], lw=0)
    order = [DESIGN_FLIGHT, *others]
    levels = {"pre": (-5.0, 1.0), "start": (-4.7, 0.5), "end": (-5.3, 0.5), "post": (-5.0, 1.0)}
    labels = {"pre": "before", "start": "start", "end": "end", "post": "after"}
    for stage, (a, b) in windows.items():
        y, h = levels[stage]
        strip.barh(y, (b - a) / 60, left=minutes(a), height=h, color="#555555", alpha=0.35, lw=0)
        ha, x = {"start": ("left", a), "end": ("right", b)}.get(stage, ("center", (a + b) / 2))
        strip.text(minutes(x) + {"left": 1.5, "right": -1.5, "center": 0}[ha], y, labels[stage], ha=ha, va="center",
                   fontsize=DESIGN_FONT)
    strip.set_yticks([*rows, -5.0], [*(names[m] for m in order), "windows"], fontsize=DESIGN_FONT)
    for tick, m in zip(strip.get_yticklabels(), order):
        tick.set_color(color[m])
    strip.set_ylim(-5.8, 0.5)
    strip.set_xlim(*minutes((lo, hi)))
    strip.set_xlabel("time since takeoff of the focal pilot (min)", fontsize=DESIGN_FONT)
    strip.tick_params(labelsize=DESIGN_FONT, axis="x")
    strip.tick_params(axis="y", length=0)
    for side in ("top", "right"):
        strip.spines[side].set_visible(False)
    for letter, ax in (("(a)", axes["a"]), ("(b)", axes["b"]), ("(c)", strip)):
        ax.text(0.01 if letter != "(c)" else -0.42, 0.97 if letter != "(c)" else 1.02, letter, transform=ax.transAxes,
                fontsize=DESIGN_FONT + 1, fontweight="bold", va="top", ha="left",
                bbox=dict(fc="white", ec="none", pad=1.5, alpha=0.8) if letter != "(c)" else None)
    for ext in ("pdf", "png"):
        fig.savefig(here / f"design.{ext}", dpi=300)
    plt.close(fig)


# Figures of the redesigned letter and of its supplement. Main dataset: lags to 2760 s (pieces of at least 3450 s;
# same flight: window 3450 s); check: dataset 1000 (same flight: window 2400 s). alpha = 2 x the stored H_loc.
MAIN_SET, CHECK_SET = "2760", "1000"  # datasets of raw.json
MAIN_WIN, CHECK_WIN = 3450, 2400  # s, windows of prepost.py
POP_COLOR = {"S": COLORS["solo"], "A": ALONE, "G": GROUP}
NET_COLOR = "#222222"
SPEED = "cross-country speed over 30 min (km/h)"
PHASE_STATS = (("time in climb (%)", "time in climb (%)"), ("time in search (%)", "time in search (%)"),
               ("searches that end in a climb (%)", "searches ending\nin a climb (%)"), ("climbs per hour", "climbs per hour"),
               ("climb rate (m/s)", "climb rate (m/s)"), ("glide speed (km/h)", "glide speed (km/h)"),
               ("height lost per km of glide (m/km)", "height lost\nper km (m/km)"))
HEADLINE = (("a1000", r"$\Delta\alpha(1000\ \mathrm{s})$"), ("a1840", r"$\Delta\alpha(1840\ \mathrm{s})$"),
            ("H", r"$\Delta H$ (10 s to end lag)"), (SPEED, "$\\Delta$ speed (km/h)"))
THRESHOLDS = (("open_declared", 0.0, "none"), ("open_min0.1", 0.1, "0.1"), ("open_min0.5", 0.5, "0.5"))


def load(folder, name):
    return json.loads((folder / f"{name}.json").read_text())


def save(fig, out, name):
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{name}.{ext}", dpi=300)
    plt.close(fig)


def num(n):
    """An integer with a thin space between the thousands from 10 000 on."""
    return f"{n:,}".replace(",", "\u2009") if n >= 10000 else str(n)


def lag_axis(ax, hi=2760, pad=1.08):
    ax.set_xscale("log")
    ax.set_xlim(10, hi * pad)
    ticks = [t for t in (10, 100, 1000, 10000) if t <= hi * pad]
    ax.set_xticks(ticks, [str(t) for t in ticks])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel(r"lag $\tau$ (s)")


def line(ax, lags, value, band, color, label=None, ls="-", lw=1.4, scale=1.0):
    """A curve with its 90% band as a shaded area; `scale` 2 turns the stored H_loc into alpha."""
    if band is not None:
        ax.fill_between(lags, scale * np.asarray(band[0]), scale * np.asarray(band[1]), color=color, alpha=0.18, lw=0)
    ax.plot(lags, scale * np.asarray(value), color=color, ls=ls, lw=lw, label=label)


def thin(ax, curve, color, scale=1.0):
    """The curve of the check dataset: a thin line without band."""
    ax.plot(curve["lags"], scale * np.asarray(curve["value"]), color=color, lw=0.7, zorder=1)


def tag(ax, text):
    ax.set_title(text, fontsize=8, loc="left")


def at(curve, lag):
    """The quote {value, band} of a curve {lags, value, band: [lows, highs]} at `lag`, None when it is not on it."""
    if lag not in curve["lags"]:
        return None
    i = list(curve["lags"]).index(lag)
    return {"value": curve["value"][i], "band": [curve["band"][0][i], curve["band"][1][i]]}


def twice(q):
    """alpha = 2 H_loc: the quote with value and band doubled."""
    return None if q is None else {"value": 2 * q["value"], "band": [2 * b for b in q["band"]]}


def msd(folder, out):
    """Fig. 2: (a) MSD of S, A and G on the main dataset with the power law fitted from 10 s; (b) alpha(tau)."""
    pops = load(folder, "raw")["datasets"][MAIN_SET]["populations"]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7), constrained_layout=True)
    ax = axes[0]
    for pop in "SAG":
        c = pops[pop]["curve"]
        lags, m = np.array(c["lags"], float), np.array(c["msd"])
        slope, intercept = np.polyfit(np.log10(lags), np.log10(m), 1)
        assert abs(slope / 2 - pops[pop]["H"]) < 1e-9
        line(ax, lags, m, c["msd_band"], POP_COLOR[pop], f"{pop}, $H$ = {pops[pop]['H']:.3f}", "--" if pop == "A" else "-")
        ax.plot([10, 2760], 10 ** (intercept + slope * np.log10([10, 2760])), color="black", lw=0.6, ls=":", zorder=3)
    ax.set(yscale="log", ylabel=r"MSD (m$^2$)")
    lag_axis(ax)
    tag(ax, "(a)")
    for lo, hi, box in ((10, 20, [0.58, 0.08, 0.38, 0.4]), (1500, 2760, [0.1, 0.52, 0.38, 0.4])):
        zoom = ax.inset_axes(box)
        for pop in "SAG":
            c = pops[pop]["curve"]
            lags = np.array(c["lags"], float)
            keep = (lags >= lo) & (lags <= hi)
            band = [np.asarray(b)[keep] for b in c["msd_band"]]
            line(zoom, lags[keep], np.asarray(c["msd"])[keep], band, POP_COLOR[pop], ls="--" if pop == "A" else "-")
        zoom.set(xscale="log", yscale="log")
        zoom.tick_params(labelsize=6, which="both")
        zoom.xaxis.set_minor_formatter(NullFormatter())
        ax.indicate_inset_zoom(zoom, edgecolor="#777777")
    ax = axes[1]
    for pop in "SAG":
        c = pops[pop]["curve"]
        line(ax, c["lags"], c["H_loc"], c["H_loc_band"], POP_COLOR[pop], pop, "--" if pop == "A" else "-", scale=2)
    lo, hi = ax.get_ylim()
    for ref, text in ((1, "diffusion"), (2, "straight flight")):
        if lo < ref < hi:
            ax.axhline(ref, color="#999999", lw=0.6, ls=":")
            ax.text(11, ref, text, fontsize=7, va="bottom", color="#555555")
    ax.set_ylabel(r"$\alpha(\tau)$")
    lag_axis(ax)
    ax.legend(loc="upper right", handlelength=1.8)
    tag(ax, "(b)")
    save(fig, out, "msd")


def designs(folder, out):
    """Fig. 3: differences of alpha against the lag in the three designs, main dataset, with the check dataset in a
    thin line of the colour of the main contrast."""
    raw = load(folder, "raw")["gaps"]
    cur = load(folder, "matched")["levels"]["all"]["delta curves"]
    main_p, check_p = (load(folder, f"prepost_{w}") for w in (MAIN_WIN, CHECK_WIN))
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.8), constrained_layout=True)

    ax = axes[0]
    for gap, color, ls, label in (("G - S", GROUP, "-", "G $-$ S"), ("A - S", ALONE, "--", "A $-$ S")):
        c = raw[MAIN_SET]["H_loc"][gap]
        line(ax, c["lags"], c["value"], c["band"], color, label, ls, scale=2)
    thin(ax, raw[CHECK_SET]["H_loc"]["G - S"], GROUP, 2)
    tag(ax, "(a) raw")

    ax = axes[1]
    names = (("G - S|G", GROUP, "-", "G $-$ S|G"), ("A - S|A", ALONE, "--", "A $-$ S|A"),
             ("(G - S|G) - (A - S|A)", NET_COLOR, "-", "difference"))
    for key, color, ls, label in names:
        c = cur[key]["long"]
        line(ax, c["lags"], c["value"], c["band"], color, label, ls, scale=2)
    thin(ax, cur["(G - S|G) - (A - S|A)"]["main"], NET_COLOR, 2)
    thin(ax, cur["G - S|G"]["main"], GROUP, 2)
    tag(ax, "(b) same route")

    ax = axes[2]
    mock = main_p["mock"]["variants"]["any site"]
    for label, e, color, ls in (("$g$", main_p["differences"]["g"], GROUP, "-"), ("$g$, mock events", mock["mock"]["g"], CONTROL, "--"),
                                ("$g$ $-$ mock", mock["real minus mock"]["g"], NET_COLOR, "-")):
        line(ax, main_p["lags"], e["dH_loc"]["value"], e["dH_loc"]["band"], color, label, ls, scale=2)
    e = check_p["mock"]["variants"]["any site"]["real minus mock"]["g"]["dH_loc"]
    thin(ax, {"lags": check_p["lags"], "value": e["value"]}, NET_COLOR, 2)
    tag(ax, "(c) same flight")

    for ax, note in zip(axes, ("thin: dataset 1000", "thin: dataset 1000", "thin: window 2400 s")):
        ax.axhline(0, color="#bbbbbb", lw=0.6, zorder=0)
        lag_axis(ax)
        ax.plot([], [], color="#555555", lw=0.7, label=note)
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, hi + 0.42 * (hi - lo))  # room for the legend above the curves
        ax.legend(loc="upper left", handlelength=1.8)
    axes[0].set_ylabel(r"$\Delta\alpha(\tau)$")
    save(fig, out, "designs")


# Forest plots. A row is {"label", "color", "data": {quantity: {"main": quote, "check": quote}}}; a header is {"header": text}.
def raw_quotes(raw, fits, gap):
    g = raw["gaps"]
    return {"a1000": {"main": twice(g[MAIN_SET]["H_loc(1000 s)"][gap]), "check": twice(g[CHECK_SET]["H_loc(1000 s)"][gap])},
            "a1840": {"main": twice(g[MAIN_SET]["H_loc(1840 s)"][gap])},
            "H": {"main": fits["gaps"][gap][MAIN_SET], "check": fits["gaps"][gap][CHECK_SET]},
            SPEED: {"main": raw["xc gaps"][gap]}} | {s: {"main": raw["phase gaps"][gap][s]} for s, _ in PHASE_STATS}


def route_quotes(m, contrast):
    d, long = m["deltas"][contrast], m["delta curves"][contrast]["long"]
    return {"a1000": {"main": twice(at(long, 1000)), "check": twice(d["H_loc(1000 s)"])},
            "a1840": {"main": twice(at(long, 1840))},
            "H": {"main": d["H (10 s to T)"], "check": d["H (10 s to 1000 s)"]},
            SPEED: {"main": d[SPEED]}} | {s: {"main": d[s]} for s, _ in PHASE_STATS}


def flight_effect(p, which):
    """g of the real events, of the mock events, or their difference (any site); None when the window has no mock events."""
    if which == "g":
        return p["differences"]["g"]
    mock = p.get("mock", {}).get("variants", {}).get("any site")
    return mock and (mock["mock"]["g"] if which == "mock" else mock["real minus mock"]["g"])


def flight_quotes(p, which):
    e = flight_effect(p, which)
    if e is None:
        return {}
    curve = {"lags": p["lags"], **e["dH_loc"]}
    return {"a1000": twice(at(curve, 1000)), "a1840": twice(at(curve, 1840)), "H": e["Delta H"],
            **{s: e.get(s) for s in (SPEED, *(p for p, _ in PHASE_STATS))}}


def both(main, check):
    """{quantity: {"main", "check"}} from two {quantity: quote} dictionaries, the quantities missing in one left out."""
    return {k: {n: q[k] for n, q in (("main", main), ("check", check)) if q.get(k)} for k in main}


def forest_grid(rows, keys, ncols, figsize, out, name, names=("main dataset", "check dataset")):
    """One panel per quantity of `keys` ((key, title) pairs), the rows of `rows` from the top: main dataset filled,
    check dataset open, 90% band; a quantity with one sample only has one marker per row."""
    nr = -(-len(keys) // ncols)
    fig, axes = plt.subplots(nr, ncols, figsize=figsize, sharey=True, constrained_layout=True, squeeze=False)
    slots = {id(r): k for k, r in enumerate(rows)}
    heads = [r["header"] for r in rows if "header" in r]
    for ax in axes.flat[len(keys):]:
        ax.set_visible(False)
    for ax, (key, title) in zip(axes.flat, keys):
        lows, highs = [0.0], [0.0]
        for r in rows:
            if "header" in r:
                continue
            y, d = slots[id(r)], r["data"].get(key, {})
            two = "main" in d and "check" in d
            for n, q in d.items():
                if q is None:
                    continue
                v, (lo, hi) = q["value"], q["band"]
                yy = y + ((-0.17 if n == "main" else 0.17) if two else 0)
                ax.errorbar(v, yy, xerr=[[v - lo], [hi - v]], fmt="o", color=r["color"], mfc=r["color"] if n == "main" else "white",
                            ms=3.5, mew=0.9, elinewidth=1.1, capsize=0)
                lows.append(lo)
                highs.append(hi)
        pad = 0.08 * (max(highs) - min(lows))
        ax.set_xlim(min(lows) - pad, max(highs) + pad)
        ax.axvline(0, color="#bbbbbb", lw=0.6, zorder=0)
        for k, r in enumerate(rows):
            if "header" in r and k:
                ax.axhline(k - 0.5, color="#dddddd", lw=0.6, zorder=0)
        ax.xaxis.set_major_locator(plt.MaxNLocator(3))
        ax.set_title(title, fontsize=8)
    ax = axes[0, 0]
    ax.set_yticks(range(len(rows)), [r.get("header") or r["label"] for r in rows])
    for t in ax.get_yticklabels():
        if t.get_text() in heads:
            t.set_fontweight("bold")
    ax.set_ylim(len(rows) - 0.4, -0.6)
    ax.tick_params(axis="y", length=0)
    handles = [plt.Line2D([], [], marker="o", color="#555555", ls="", ms=3.5, label=names[0]),
               plt.Line2D([], [], marker="o", color="#555555", mfc="white", ls="", ms=3.5, label=names[1])]
    fig.legend(handles=handles, loc="outside lower center", ncol=2, handletextpad=0.2)
    save(fig, out, name)


def designs_forest(folder, out):
    """Fig. 4: alpha at two lags, fitted H and cross-country speed in the three designs."""
    raw, fits, m = load(folder, "raw"), load(folder, "fitted_h_by_end"), load(folder, "matched")["levels"]["all"]
    main_p, check_p = (load(folder, f"prepost_{w}") for w in (MAIN_WIN, CHECK_WIN))
    rows = [{"header": "raw"},
            {"label": "G $-$ S", "color": GROUP, "data": raw_quotes(raw, fits, "G - S")},
            {"label": "A $-$ S", "color": ALONE, "data": raw_quotes(raw, fits, "A - S")},
            {"header": "same route"}]
    for key, color, label in (("G - S|G", GROUP, "G $-$ S|G"), ("A - S|A", ALONE, "A $-$ S|A"),
                              ("(G - S|G) - (A - S|A)", NET_COLOR, "difference")):
        rows.append({"label": label, "color": color, "data": route_quotes(m, key)})
    rows.append({"header": "same flight"})
    for key, color, label in (("g", GROUP, "$g$"), ("mock", CONTROL, "mock events"), ("net", NET_COLOR, "$g$ $-$ mock")):
        rows.append({"label": label, "color": color, "data": both(flight_quotes(main_p, key), flight_quotes(check_p, key))})
    forest_grid(rows, HEADLINE, 4, (7.0, 3.6), out, "forest",
                ("main: lags to 2760 s, window 3450 s", "check: lags to 1000 s, window 2400 s"))


def opening_values():
    """The opening of the declared open flights of the eligible paragliders (the set of opening.py)."""
    from openflights import opening
    flights = pd.read_parquet(tamsd.LABELS / "flights.parquet", columns=["flight", "dt_native", "n_grid"])
    o = opening(tamsd.LABELS, tamsd.SHARED / "flight_ends.parquet", tamsd.FIXES["para"].parent).merge(flights, on="flight")
    return o.loc[o["declared_open"] & (o["dt_native"] <= tamsd.H_MAX + 1e-6) & (o["n_grid"] > 0), "opening"].to_numpy()


def sm_opening(folder, out):
    """S1: (a) the opening of the flights declared open with the threshold of the letter; (b) fitted H and (c) alpha(1000 s)
    of S, A and G on the dataset 1000 against the threshold."""
    run = folder.parent
    shares = load(run / "open_declared", "opening")["datasets"]["every declared open flight"]["all"]
    ratio = opening_values()
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.5), constrained_layout=True, gridspec_kw={"width_ratios": [1.3, 1, 1]})
    ax = axes[0]
    ax.hist(np.clip(ratio, 0, 1), bins=np.linspace(0, 1, 41), color="#bbbbbb", lw=0)
    for x, share, ha in ((MIN_OPENING, shares["above the threshold (%)"], "left"), (0.5, shares["above the stricter threshold (%)"], "left")):
        ax.axvline(x, color="black", lw=0.9, ls="-" if x == MIN_OPENING else ":")
        ax.text(x + 0.02, 0.93 if x == MIN_OPENING else 0.66, f"threshold {x:g}\n{share:.1f}% above", transform=ax.get_xaxis_transform(),
                fontsize=7, va="top", ha=ha, bbox=dict(fc="white", ec="none", pad=1.0, alpha=0.8))
    ax.set(xlabel="opening (takeoff to landing over path flown)", ylabel="flights", xlim=(0, 1))
    tag(ax, f"(a) {num(len(ratio))} flights declared open")
    xs = {t: k for k, (_, _, t) in enumerate(THRESHOLDS)}
    for ax, what in ((axes[1], "H"), (axes[2], "alpha")):
        for k, pop in enumerate("SAG"):
            for x, (sub, _, name) in enumerate(THRESHOLDS):
                c = load(run / sub, "tamsd")["para"][{"S": "solo", "A": "group_solo", "G": "group"}[pop]]
                if what == "H":
                    v, (lo, hi) = c["H"], c["H_band"]
                else:
                    i = c["lags"].index(1000)
                    v, lo, hi = 2 * c["H_loc"][i], 2 * c["H_loc_band"][0][i], 2 * c["H_loc_band"][1][i]
                ax.errorbar(x + 0.12 * (k - 1), v, yerr=[[v - lo], [hi - v]], fmt="o", color=POP_COLOR[pop], ms=3.5, elinewidth=1.1,
                            label=pop if x == 0 else None)
        ax.set_xticks(range(len(THRESHOLDS)), [t for _, _, t in THRESHOLDS])
        ax.set_xlabel("threshold on the opening")
        ax.set_xlim(-0.5, 2.5)
    axes[1].set_ylabel(r"fitted $H$ (10 s to 1000 s)")
    axes[2].set_ylabel(r"$\alpha(1000\ \mathrm{s})$")
    axes[1].legend(loc="upper left", handletextpad=0.2)
    tag(axes[1], "(b)")
    tag(axes[2], "(c)")
    save(fig, out, "sm_opening")


def sm_datasets(folder, out):
    """S2: MSD and alpha(tau) of S, A and G on the three datasets."""
    sets = load(folder, "raw")["datasets"]
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.5), constrained_layout=True, sharey="row")
    for col, (name, ds) in enumerate(sets.items()):
        top, bottom = axes[0, col], axes[1, col]
        for pop in "SAG":
            p = ds["populations"][pop]
            c = p["curve"]
            ls = "--" if pop == "A" else "-"
            line(top, c["lags"], c["msd"], c["msd_band"], POP_COLOR[pop], f"{pop}  {p['H']:.3f}  ({num(p['flights'])})", ls)
            line(bottom, c["lags"], c["H_loc"], c["H_loc_band"], POP_COLOR[pop], None, ls, scale=2)
        top.set_yscale("log")
        top.legend(title=r"$H$  (flights)", loc="lower right", handlelength=1.6, alignment="right", title_fontsize=7)
        tag(top, f"({'abc'[col]}) lags to {num(ds['tau_max'])} s")
        for ax in (top, bottom):
            lag_axis(ax, ds["tau_max"], 1.3)
        top.set_xlabel("")
    axes[0, 0].set_ylabel(r"MSD (m$^2$)")
    axes[1, 0].set_ylabel(r"$\alpha(\tau)$")
    save(fig, out, "sm_datasets")


def sm_durations(folder, out):
    """S2: flights with a piece of at least d = tau / SUPPORT s, the pieces that give a lag tau, with the thresholds of the datasets."""
    av = load(folder, "longlags")["available"]
    fig, ax = plt.subplots(figsize=(3.4, 2.7), constrained_layout=True)
    for key, pop in (("solo", "S"), ("group_solo", "A"), ("group", "G")):
        keep = np.array(av[key]["flights"]) > 0
        ax.plot((np.array(av[key]["lags"]) / tamsd.SUPPORT)[keep], np.array(av[key]["flights"])[keep], color=POP_COLOR[pop], marker="o", ms=2.5, lw=1.2,
                ls="--" if pop == "A" else "-", label=pop)
    for tau, text in ((1000, "dataset 1000"), (2760, "2760"), (11450, "11 450")):
        ax.axvline(tau / tamsd.SUPPORT, color="#888888", lw=0.7, ls=":")
        ax.text(tau / tamsd.SUPPORT * 1.06, 1.5, text, rotation=90, fontsize=7, color="#555555", va="bottom")
    ax.set(xscale="log", yscale="log", ylim=(1, 1e5), xlabel=r"piece duration $d=\tau/0.8$ (s)", ylabel=r"flights with a piece $\geq d$")
    ax.set_xticks([1000, 3000, 10000, 30000], ["1000", "3000", "10000", "30000"])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.legend(loc="upper right", handlelength=1.6)
    save(fig, out, "sm_durations")


def fit_to(c, end):
    """H fitted from 10 s to `end` on the MSD curve `c`, as tamsd.fit_h."""
    lags = np.array(c["lags"], float)
    k = lags <= end
    return np.polyfit(np.log10(lags[k]), np.log10(np.array(c["msd"])[k]), 1)[0] / 2


def sm_fit_by_end(folder, out):
    """S3: the fitted H of S, A, G and the gaps against the end lag, each end lag on its own pieces (points with bands) and
    on the pieces of the dataset 2760 (dotted)."""
    fits = load(folder, "fitted_h_by_end")
    pops = load(folder, "raw")["datasets"][MAIN_SET]["populations"]
    ends = fits["ends"]
    grid = [e for e in pops["S"]["curve"]["lags"] if e >= 210]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7), constrained_layout=True)
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xticks([300, 1000, 3000, 10000], ["300", "1000", "3000", "10000"])
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel("end lag of the fit (s)")
    for k, pop in enumerate("SAG"):
        ax = axes[0]
        ax.plot(grid, [fit_to(pops[pop]["curve"], e) for e in grid], color=POP_COLOR[pop], lw=0.9, ls=":")
        v = np.array([fits["H"][pop][str(e)]["value"] for e in ends])
        b = np.array([fits["H"][pop][str(e)]["band"] for e in ends]).T
        ax.errorbar(np.array(ends) * (1 + 0.04 * (k - 1)), v, yerr=[v - b[0], b[1] - v], fmt="o", color=POP_COLOR[pop], ms=3.5, elinewidth=1.1, label=pop)
    axes[0].set_ylabel(r"fitted $H$ (10 s to end lag)")
    axes[0].legend(loc="upper left", handletextpad=0.2)
    for gap, color, ls in (("G - S", GROUP, "-"), ("A - S", ALONE, "--")):
        ax = axes[1]
        a, b = gap.split(" - ")
        ax.plot(grid, [fit_to(pops[a]["curve"], e) - fit_to(pops[b]["curve"], e) for e in grid], color=color, lw=0.9, ls=":")
        v = np.array([fits["gaps"][gap][str(e)]["value"] for e in ends])
        bd = np.array([fits["gaps"][gap][str(e)]["band"] for e in ends]).T
        ax.errorbar(np.array(ends) * (1 + 0.04 * (gap == "A - S")), v, yerr=[v - bd[0], bd[1] - v], fmt="o", color=color, ms=3.5, elinewidth=1.1,
                    label=gap.replace("-", "$-$"))
    axes[1].axhline(0, color="#bbbbbb", lw=0.6, zorder=0)
    axes[1].set_ylabel(r"$\Delta H$ (10 s to end lag)")
    axes[1].legend(loc="upper left", handletextpad=0.2)
    axes[0].plot([], [], color="#555555", lw=0.9, ls=":", label="dataset 2760")
    axes[0].legend(loc="upper left", handletextpad=0.2)
    axes[1].plot([], [], color="#555555", lw=0.9, ls=":", label="dataset 2760")
    axes[1].legend(loc="upper left", handletextpad=0.2)
    tag(axes[0], "(a)")
    tag(axes[1], "(b)")
    save(fig, out, "sm_fit_by_end")


WINDOWS = (1260, 1800, 2400, 3000, 3450)  # s, the windows of prepost.py


def sm_windows(folder, out):
    """S7: g and g minus mock of the same-flight design against the window length, with the numbers of events."""
    pre = {w: load(folder, f"prepost_{w}") for w in WINDOWS}
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9), constrained_layout=True)
    for ax, key, label in ((axes[0], "a1000", r"$\Delta\alpha(1000\ \mathrm{s})$"), (axes[1], "H", r"$\Delta H$ (10 s to $0.8\times$ window)")):
        for which, color, name, shift in (("g", GROUP, r"$g$", -25), ("net", NET_COLOR, r"$g$ $-$ mock", 25)):
            for w in WINDOWS:
                q = flight_quotes(pre[w], which).get(key)
                if q:
                    edge = key == "a1000" and w == 1260  # 1000 s is the last lag of this window
                    ax.errorbar(w + shift, q["value"], yerr=[[q["value"] - q["band"][0]], [q["band"][1] - q["value"]]], fmt="o", color=color,
                                mfc="white" if edge else color, ms=3.5, mew=0.9, elinewidth=1.1, label=name if w == 3450 else None)
        ax.axhline(0, color="#bbbbbb", lw=0.6, zorder=0)
        ax.set_xticks(WINDOWS, [f"{w}\n{num(pre[w]['differences']['g']['events (MSD)'])}\n"
                                + (num(flight_effect(pre[w], "net")["events (MSD)"]) if flight_effect(pre[w], "net") else "-") for w in WINDOWS])
        ax.set_xlim(1100, 3600)
        ax.set_ylabel(label)
        ax.set_xlabel("window length (s)\nevents of $g$\nevents of $g$ $-$ mock")
    axes[0].text(0.03, 0.97, "open: 1000 s is the last lag", transform=axes[0].transAxes, fontsize=7, va="top")
    axes[0].legend(loc="lower right", handletextpad=0.2)
    axes[1].legend(loc="lower right", handletextpad=0.2)
    tag(axes[0], "(a)")
    tag(axes[1], "(b)")
    save(fig, out, "sm_windows")


def sm_flight_forest(folder, out):
    """S7: the same-flight design, every quantity: g, mock events and their difference; windows 3450 s (filled) and 2400 s (open)."""
    main_p, check_p = (load(folder, f"prepost_{w}") for w in (MAIN_WIN, CHECK_WIN))
    rows = [{"label": label, "color": color, "data": both(flight_quotes(main_p, key), flight_quotes(check_p, key))}
            for key, color, label in (("g", GROUP, "$g$"), ("mock", CONTROL, "mock events"), ("net", NET_COLOR, "$g$ $-$ mock"))]
    forest_grid(rows, HEADLINE + tuple(PHASE_STATS), 4, (7.0, 4.0), out, "sm_flight_forest", ("window 3450 s", "window 2400 s"))


def sm_route_forest(folder, out):
    """S6: the same-route design, every quantity: G - S|G, A - S|A and their difference; datasets 2760 (filled) and 1000 (open)."""
    m = load(folder, "matched")["levels"]["all"]
    rows = [{"label": label, "color": color, "data": route_quotes(m, key)}
            for key, color, label in (("G - S|G", GROUP, "G $-$ S|G"), ("A - S|A", ALONE, "A $-$ S|A"),
                                      ("(G - S|G) - (A - S|A)", NET_COLOR, "difference"))]
    forest_grid(rows, HEADLINE + tuple(PHASE_STATS), 4, (7.0, 4.0), out, "sm_route_forest", ("lags to 2760 s", "lags to 1000 s"))


def sm_coverage(folder, out):
    """S5: coverage of the 90% bands on synthetic data with no effect, by design and mix of variances."""
    cov = load(folder, "coverage")
    names = list(cov["scenarios"])
    designs = (("raw", "(a) raw"), ("matched", "(b) same route"), ("prepost", "(c) same flight"))
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.6), constrained_layout=True, sharey=True)
    se = np.sqrt(0.9 * 0.1 / cov["replicates"])
    for ax, (key, title) in zip(axes, designs):
        sc = cov[key]["scenarios"]
        ax.axvspan(0.9 - 2 * se, 0.9 + 2 * se, color="#dddddd", lw=0)
        ax.axvline(0.9, color="black", lw=0.8)
        old = [k for k in sc[names[0]] if k not in ("true sd", "two-way, normal")][0]
        for method, marker, color, label, shift in (("two-way, normal", "o", GROUP, "two-way band, as in the letter", -0.12), (old, "s", CONTROL, "earlier scheme", 0.12)):
            ax.plot([sc[n][method]["coverage of 90%"] for n in names], np.arange(len(names)) + shift, marker, color=color, ms=4, label=label)
        ax.set(xlim=(0.5, 1.02), xlabel="coverage of the 90% band")
        tag(ax, title)
    axes[0].set_yticks(range(len(names)), names)
    axes[0].set_ylim(len(names) - 0.5, -0.5)
    axes[0].legend(loc="lower left", handletextpad=0.2)
    save(fig, out, "sm_coverage")


FIGURES = {"msd": msd, "designs": designs, "forest": designs_forest, "how": how, "survival": survival, "tail": tail,
           "design": design, "sm_opening": sm_opening, "sm_datasets": sm_datasets, "sm_durations": sm_durations,
           "sm_fit_by_end": sm_fit_by_end, "sm_windows": sm_windows, "sm_flight_forest": sm_flight_forest,
           "sm_route_forest": sm_route_forest, "sm_coverage": sm_coverage}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="d120_eh500_ez300_th45_g120_q0.1")
    parser.add_argument("--only", choices=list(FIGURES))
    args = parser.parse_args()
    folder = GROUPS / "out" / args.run / f"open_min{MIN_OPENING:g}"
    for name, figure in FIGURES.items():
        if args.only in (None, name):
            figure(folder, HERE / "figures")


if __name__ == "__main__":
    main()
