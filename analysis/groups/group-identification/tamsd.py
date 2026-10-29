"""TA-MSD of solo flights and of group flights during their group time, with H and H_loc.

Labels: a run of methods/group-identification/run.py on SSD_DISANTE (RUN). Tracks: the
cleaned east and north E, N of the old repository on SSD_DISANTE. Conventions of the fixed
transport of chapter 3 (soaring, legacy-final): positions on a 10-s grid, the lags of its
common grid up to TAU_MAX, a stretch enters only if it lasts at least TAU_MAX / 0.8 so that
no population changes with the lag, equal weight per flight, H half the slope of log MSD
against log lag fitted with equal weight per lag, H_loc the same slope within +/- 0.25
decades (the 3 nearest lags when fewer lie there).

Populations, after the departure and with loggers of at most H_MAX:
- solo: flights never in a group;
- group: flights with group time, only their stretches in a group (consecutive G rows of the
  membership table, which lie in one episode);
- group, solo time: the solo stretches of the group flights, a control with the same pilots
  and days.

    uv run python tamsd.py [--run NAME] [--recompute]    # from this folder; writes out/<run>/
"""

import json
import os
import shutil
import sys
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lab.legacy import stream_flights
from lab.msd import tamsd_sums

# The labels are a run of methods/group-identification on SSD_DISANTE, named by its grouping
# parameters (see its PARAMETERS.md). Figures and numbers go to out/<run>/ with a copy of
# PARAMETERS.md; the measured pieces, too heavy for the repository, stay in <run>/analyses.
RUNS = Path("/Volumes/SSD_DISANTE/xc-lab/group-identification/runs")
RUN = "d120_eh500_ez300_th45_g120_q0.1"  # the run the README quotes; --run NAME picks another
LABELS = RUNS / (sys.argv[sys.argv.index("--run") + 1] if "--run" in sys.argv else RUN)
OUT = Path(__file__).with_name("out") / LABELS.name
CACHE = LABELS / "analyses"
DISK = Path("/Volumes/SSD_DISANTE")
FIXES = {"para": DISK / "paragliders/ffvl_cfd_igc/derived/fixes.parquet",
         "hang": DISK / "hang_gliders/delta_cfd_igc/derived/fixes.parquet"}
GRID_S, SUPPORT, TAU_MAX, HALF_DECADE = 10, 0.8, 1000, 0.25
LAGS = np.unique(np.r_[np.round(np.geomspace(1, 1000, 35)).astype(int) * 10, 100, 1000])  # chapter 3
LAGS = LAGS[LAGS <= TAU_MAX]
H_MAX = 10  # loggers up to this interval are on the grid of the labels at every instant
POPULATIONS = {"solo": "solo flights", "group": "group flights, group time",
               "group_solo": "group flights, solo time"}
COLORS = {"solo": "#2a78d6", "group": "#eb6834", "group_solo": "#1baf7a"}
# Group time by the number of pilots of the group (stretches(by_size=True)).
SIZES = {"n2": "2 pilots", "n3": "3 pilots", "n4": "4 pilots", "n5-6": "5-6 pilots", "n7+": "7+ pilots"}
NAMES = {"para": "paragliders", "hang": "hang gliders"}
N_BOOT, SEED = 1000, 20261020


def size_of(n):
    """The key of SIZES of a group of n pilots."""
    return np.select([n == 2, n == 3, n == 4, n <= 6], ["n2", "n3", "n4", "n5-6"], "n7+")


def stretches(flights, membership, summary, by_size=False):
    """Stretches of every eligible flight on its own clock (seconds from the trimmed start):
    flight, population, episode (-1 when solo), a, b. With by_size, a group stretch ends where
    the group changes and its population is the key of SIZES of the group."""
    ok = flights[(flights["dt_native"] <= H_MAX + 1e-6) & (flights["n_grid"] > 0)]
    mb = membership[membership["flight"].isin(ok["flight"]) & (membership["state"] != "L")]
    mb = mb.sort_values(["flight", "t_start"], kind="stable")
    # Consecutive rows of one state are one stretch; G rows in a row change only the group.
    change = mb["state"] != mb.groupby("flight")["state"].shift()
    if by_size:
        change |= (mb["state"] == "G") & (mb["group_id"] != mb.groupby("flight")["group_id"].shift())
    columns = dict(flight=("flight", "first"), state=("state", "first"), a=("t_start", "min"), b=("t_end", "max"),
                   episode=("episode_id", "first"), **({"n": ("n", "first")} if by_size else {}))
    s = mb.groupby(change.cumsum()).agg(**columns)
    grouped = summary.set_index("flight")["n_entries"].reindex(s["flight"]).to_numpy() > 0
    in_group = size_of(s["n"].to_numpy()) if by_size else "group"
    s["population"] = np.where(s["state"] == "G", in_group, np.where(grouped, "group_solo", "solo"))
    s["episode"] = np.where(s["state"] == "G", s["episode"], -1)
    utc0 = flights.set_index("flight")["utc0"].reindex(s["flight"]).to_numpy()
    s["a"], s["b"] = s["a"] - utc0, s["b"] - utc0
    return s[["flight", "population", "episode", "a", "b"]].reset_index(drop=True)


def grids(fixes, spans, shortest):
    """For one flight: every piece of a stretch inside one segment that lasts at least
    `shortest` s, as its stretch, its first instant on the flight clock and its positions E, N
    on the 10-s grid. `spans` holds the stretches of the flight (rows of stretches())."""
    fixes = fixes.sort_values("t", kind="stable")
    t = fixes["t"].to_numpy(float)
    seg = fixes["segment_id"].to_numpy()
    for s in np.unique(seg):
        q = seg == s
        ts = t[q]
        for span in spans.itertuples():
            g = np.arange(np.ceil(max(ts[0], span.a) / GRID_S) * GRID_S, min(ts[-1], span.b - 1e-6) + 1e-6, GRID_S)
            if len(g) < 2 or (len(g) - 1) * GRID_S < shortest:
                continue
            yield span, g[0], np.column_stack([np.interp(g, ts, fixes[c].to_numpy(float)[q]) for c in ("E", "N")])


def pieces(fixes, spans):
    """The pieces of grids() that last at least TAU_MAX / SUPPORT: population, episode, first
    instant (flight clock), duration, the east and north of the middle of the piece (frame of the
    flight), and their sums and origins at LAGS."""
    out = []
    for span, start, pos in grids(fixes, spans, TAU_MAX / SUPPORT):
        sums, origins = tamsd_sums(pos, LAGS // GRID_S)
        mid = pos[len(pos) // 2]
        out.append((span.population, span.episode, start, (len(pos) - 1) * GRID_S, mid[0], mid[1], sums, origins))
    return out


def _job(flight, fixes, spans):
    return flight, pieces(fixes, spans)


def measure(discipline, flights, spans):
    """Pieces of every flight of one discipline, from one pass over its fixes."""
    ids = flights.loc[flights["discipline"] == discipline].set_index("flight_id")["flight"]
    by_flight = dict(tuple(spans.groupby("flight")))
    rows = []

    def store(future):
        flight, found = future.result()
        rows.extend((flight, *p) for p in found)

    with ProcessPoolExecutor(os.cpu_count()) as pool:
        pending = set()
        for fixes in stream_flights(FIXES[discipline], ["segment_id", "t", "E", "N"]):
            flight = ids.get(str(fixes["flight_id"].iat[0]))
            if flight is None or flight not in by_flight:
                continue
            pending.add(pool.submit(_job, flight, fixes, by_flight[flight]))
            if len(pending) >= 4 * os.cpu_count():
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    store(future)
        for future in pending:
            store(future)
    flight, population, episode, start, duration, mid_e, mid_n, sums, origins = zip(*rows) if rows else ([],) * 9
    return {"flight": np.array(flight, np.int64), "population": np.array(population), "episode": np.array(episode),
            "start": np.array(start), "duration": np.array(duration), "mid_e": np.array(mid_e),
            "mid_n": np.array(mid_n), "sums": np.array(sums), "origins": np.array(origins)}


def save_pieces(path, flights, spans):
    """Measure the pieces of every discipline for `spans` and save them in `path`."""
    measured = {d: measure(d, flights, spans) for d in FIXES}
    np.savez(path, **{f"{d}/{k}": v for d, m in measured.items() for k, v in m.items()})


def read_pieces(path):
    """The pieces saved by save_pieces: per discipline, a dict of arrays."""
    data = np.load(path)
    return {d: {k.split("/")[1]: data[k] for k in data.files if k.startswith(d + "/")} for d in FIXES}


def per_flight(p, mask):
    """Each flight's MSD at LAGS, pooling the origins of its pieces in `mask`."""
    flights, k = np.unique(p["flight"][mask], return_inverse=True)
    sums, origins = np.zeros((len(flights), len(LAGS))), np.zeros((len(flights), len(LAGS)))
    np.add.at(sums, k, p["sums"][mask])
    np.add.at(origins, k, p["origins"][mask])
    return flights, sums / origins


def fit_h(lags, msd):
    """Half the slope of log MSD against log lag, equal weight per lag."""
    return np.polyfit(np.log10(lags), np.log10(msd), 1)[0] / 2


def local_h(lags, msd):
    """H within +/- HALF_DECADE of each lag; the 3 nearest lags when fewer lie there."""
    x = np.log10(lags)
    out = []
    for c in x:
        near = np.abs(x - c) <= HALF_DECADE + 1e-9
        if near.sum() < 3:
            near = np.isin(np.arange(len(x)), np.argsort(np.abs(x - c), kind="stable")[:3])
        out.append(fit_h(lags[near], msd[near]))
    return np.array(out)


def ensemble(values, blocks, rng):
    """Equal-weight mean over flights, and N_BOOT bootstrap means resampling whole blocks."""
    ids, k = np.unique(blocks, return_inverse=True)
    sums, counts = np.zeros((len(ids), values.shape[1])), np.bincount(k, minlength=len(ids)).astype(float)
    np.add.at(sums, k, values)
    draws = []
    for start in range(0, N_BOOT, 50):
        w = rng.multinomial(len(ids), np.full(len(ids), 1 / len(ids)), size=min(50, N_BOOT - start))
        draws.append((w @ sums) / (w @ counts)[:, None])
    return values.mean(axis=0), np.concatenate(draws)


def summarize(p, summary, rng, populations=POPULATIONS):
    """Per population of `populations`: flights, pieces, MSD with its 5-95% bootstrap band, H and H_loc."""
    block = summary.set_index("flight")["block"]
    out = {}
    for pop in populations:
        mask = p["population"] == pop
        if not mask.any():
            continue
        flights, values = per_flight(p, mask)
        mean, draws = ensemble(values, block.reindex(flights).to_numpy(), rng)
        h = np.array([fit_h(LAGS, d) for d in draws])
        hl = np.array([local_h(LAGS, d) for d in draws])
        out[pop] = {"flights": len(flights), "pieces": int(mask.sum()), "lags": LAGS.tolist(), "msd": mean.tolist(),
                    "msd_band": np.percentile(draws, [5, 95], axis=0).tolist(), "H": fit_h(LAGS, mean),
                    "H_band": np.percentile(h, [5, 95]).tolist(), "H_loc": local_h(LAGS, mean).tolist(),
                    "H_loc_band": np.percentile(hl, [5, 95], axis=0).tolist()}
    return out


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=200, bbox_inches="tight")


def plot(results, path):
    """Top: MSD against the lag; bottom: H_loc. Columns: disciplines."""
    fig, axes = plt.subplots(2, 2, figsize=(9, 7), sharex=True)
    for col, (d, name) in enumerate(NAMES.items()):
        for pop, label in POPULATIONS.items():
            r = results.get(d, {}).get(pop)
            if r is None:
                continue
            lags = np.array(r["lags"])
            band = np.array(r["msd_band"])
            lo, hi = r["H_band"]
            axes[0, col].fill_between(lags, band[0], band[1], color=COLORS[pop], alpha=0.25, lw=0)
            axes[0, col].plot(lags, r["msd"], color=COLORS[pop], marker="o", ms=3, lw=1,
                              label=f"{label} ({r['flights']}): H = {r['H']:.3f} [{lo:.3f}, {hi:.3f}]")
            band = np.array(r["H_loc_band"])
            axes[1, col].fill_between(lags, band[0], band[1], color=COLORS[pop], alpha=0.25, lw=0)
            axes[1, col].plot(lags, r["H_loc"], color=COLORS[pop], marker="o", ms=3, lw=1)
        axes[0, col].set(xscale="log", yscale="log", title=name, ylabel=r"MSD (m$^2$)")
        axes[0, col].legend(frameon=False, fontsize=7, loc="upper left")
        axes[1, col].axhline(0.5, color="#999999", lw=0.6, ls=":")
        axes[1, col].axhline(1, color="#999999", lw=0.6, ls=":")
        axes[1, col].set(xscale="log", xlabel=r"$\tau$ (s)", ylabel=r"$H_{\rm loc}(\tau)$")
    save(fig, path)


def folders():
    """Create OUT and CACHE; OUT gets the PARAMETERS.md of the run."""
    for folder in (OUT, CACHE):
        folder.mkdir(parents=True, exist_ok=True)
    shutil.copy2(LABELS / "PARAMETERS.md", OUT)


if __name__ == "__main__":
    folders()
    flights = pd.read_parquet(LABELS / "flights.parquet")
    membership = pd.read_parquet(LABELS / "membership.parquet")
    summary = pd.read_parquet(LABELS / "flight_summary.parquet")
    saved = CACHE / "pieces.npz"
    # The paraglider pass reads 40 GB of fixes: reuse it unless asked.
    if "--recompute" in sys.argv or not saved.exists():
        save_pieces(saved, flights, stretches(flights, membership, summary))
    measured = read_pieces(saved)
    rng = np.random.default_rng(SEED)
    results = {d: summarize(measured[d], summary, rng) for d in FIXES}
    (OUT / "tamsd.json").write_text(json.dumps(results, indent=1))
    plot(results, OUT / "tamsd.pdf")
    for d, r in results.items():
        for pop, v in r.items():
            print(f"{d} {pop}: {v['flights']} flights, {v['pieces']} pieces, H = {v['H']:.3f} {np.round(v['H_band'], 3)}")
