"""Refit of Vilpellet's three-state HMM on our flights, and the checks of the refit.

The model keeps his three binary features, his Viterbi decoding and his 90-fix vote
(phases.py); only the parameters (emissions, transition matrix, initial law) are estimated
again, by Baum-Welch on flights logged at 1 Hz, per discipline. A model is a dict with the
keys emission (3x8), transition (3x3) and initial (3), states in the order of phases.STATES.

    uv run python methods/phase-labelling/refit.py train para|hang
    uv run python methods/phase-labelling/refit.py cadence para|hang
"""

import argparse
import json
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from numba import njit

from xc_dynamics.preprocessing.flight import preprocess

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import phases  # noqa: E402

LAB = HERE.parents[1]
DYNAMICS = LAB.parent / "xc-dynamics"
OUT = HERE / "out"
CACHE = Path("/Volumes/SSD_DATA/xc-lab/phase-labelling")
VIEWER = Path("/Volumes/SSD_DATA/xc-viewer-data")
GROUND = LAB / "analyses/groups/out/d120_eh500_ez300_th45_g120_q0.1/open_declared/ground.parquet"
# Map of the ground table: its flight number to the FFVL flight id (read-only).
GROUND_FLIGHTS = Path("/Volumes/SSD_DISANTE/xc-lab/group-identification/runs/d120_eh500_ez300_th45_g120_q0.1/flights.parquet")

# Per discipline: the data folder, the tracks folder, the folder and the source name of the viewer.
DISCIPLINES = {
    "para": dict(root=Path("/Volumes/SSD_DATA/paragliders/ffvl_cfd_igc"), tracks=VIEWER / "paragliders",
                 folder="paragliders", source="paraglider"),
    "hang": dict(root=Path("/Volumes/SSD_DATA/hang_gliders/delta_cfd_igc"), tracks=VIEWER / "hang_gliders",
                 folder="hang_gliders", source="hangglider"),
}

MAX_INTERVAL_S = 5.0  # a logger slower than this stays unlabelled
MIN_GAP_S, GAP_STEPS = 4.0, 2  # a gap longer than max(MIN_GAP_S, GAP_STEPS * interval) ends a piece
CADENCES = (2, 3, 4, 5)  # s, thinning of the cadence check

# Training sample.
TRAIN_FLIGHTS = {"para": 2000, "hang": 2000}  # flights drawn, held-out included
HELD_OUT = 0.2
MIN_FIXES, MAX_MEAN_STEP_S = 3600, 1.2  # his selection of training flights
SEED = 2026
WORKERS = 3

# Baum-Welch.
STARTS = 10  # random starts, and as many perturbed ones, plus his parameters
PERTURB = 0.5  # weight of a random start in a perturbed one: (1 - PERTURB) * his + PERTURB * random
MAX_ITER, REL_TOL = 500, 1e-6  # stop when |change of log-likelihood| / |log-likelihood| < REL_TOL
FLOOR = 1e-12  # smallest probability after an update
BEST_TOL = 1e-5  # a start "reaches the best" within this relative log-likelihood

# A state keeps its meaning when the probability that a feature is 1 in the state is at least MEANING_HIGH for
# the features that define it and at most MEANING_LOW for those that exclude it: climb is rising, one-sign and
# not straight; transition is straight; search is not straight and not a climb.
MEANING_HIGH, MEANING_LOW = 0.8, 0.2
OWN_HANG_MIN_FLIGHTS = 200

HIS = dict(emission=phases.EMISSION, transition=phases.TRANSITION, initial=phases.INITIAL)
SYMBOL_BITS = np.array([[(s >> 2) & 1, (s >> 1) & 1, s & 1] for s in range(8)])  # straight, rising, one sign


# ---- decoding ---------------------------------------------------------------------------------

@njit(cache=True)
def viterbi_nb(symbols, emission, transition, initial):
    """phases.viterbi_symbols, compiled: same scores, same renormalisation, ties to the lower state."""
    n, s_len = len(symbols), emission.shape[0]
    psi = np.zeros((n, s_len), dtype=np.int8)
    delta = np.empty(s_len)
    new = np.empty(s_len)
    for j in range(s_len):
        delta[j] = initial[j] * emission[j, symbols[0]]
    for k in range(1, n):
        for j in range(s_len):
            best, arg = delta[0] * transition[0, j], 0
            for i in range(1, s_len):
                score = delta[i] * transition[i, j]
                if score > best:
                    best, arg = score, i
            psi[k, j] = arg
            new[j] = best * emission[j, symbols[k]]
        total = 0.0
        for j in range(s_len):
            total += new[j]
        for j in range(s_len):
            delta[j] = new[j] / total
    path = np.empty(n, dtype=np.int64)
    arg = 0
    for j in range(1, s_len):
        if delta[j] > delta[arg]:
            arg = j
    path[n - 1] = arg
    for k in range(n - 1, 0, -1):
        path[k - 1] = psi[k, path[k]]
    return path


def decode(symbols, model):
    """Final states of a symbol sequence: Viterbi, then the 90-fix vote."""
    return phases.majority(viterbi_nb(symbols, model["emission"], model["transition"], model["initial"]))


def symbols_of(obs):
    return (obs @ np.array([4, 2, 1])).astype(np.int8)


# ---- Baum-Welch -------------------------------------------------------------------------------

@njit(cache=True)
def em_stats(sym, off, emission, transition, initial):
    """Expected counts of every sequence sym[off[q]:off[q+1]] under the model, and the log-likelihood.
    Returns (log-likelihood, emission counts, transition counts, initial counts)."""
    n_state, n_sym = emission.shape
    count_b, count_a, count_0 = np.zeros((n_state, n_sym)), np.zeros((n_state, n_state)), np.zeros(n_state)
    ll = 0.0
    beta, nxt = np.empty(n_state), np.empty(n_state)
    for q in range(len(off) - 1):
        o = sym[off[q]:off[q + 1]]
        n = len(o)
        alpha, scale = np.empty((n, n_state)), np.empty(n)
        for j in range(n_state):
            alpha[0, j] = initial[j] * emission[j, o[0]]
        for t in range(n):
            if t > 0:
                for j in range(n_state):
                    s = 0.0
                    for i in range(n_state):
                        s += alpha[t - 1, i] * transition[i, j]
                    alpha[t, j] = s * emission[j, o[t]]
            scale[t] = 0.0
            for j in range(n_state):
                scale[t] += alpha[t, j]
            for j in range(n_state):
                alpha[t, j] /= scale[t]
            ll += np.log(scale[t])
        for j in range(n_state):
            beta[j] = 1.0
            count_b[j, o[n - 1]] += alpha[n - 1, j]
        for t in range(n - 2, -1, -1):
            for i in range(n_state):
                s = 0.0
                for j in range(n_state):
                    w = transition[i, j] * emission[j, o[t + 1]] * beta[j] / scale[t + 1]
                    count_a[i, j] += alpha[t, i] * w
                    s += w
                nxt[i] = s
            for i in range(n_state):
                beta[i] = nxt[i]
                g = alpha[t, i] * beta[i]
                count_b[i, o[t]] += g
                if t == 0:
                    count_0[i] += g
    return ll, count_b, count_a, count_0


@njit(cache=True)
def log_likelihood(sym, off, emission, transition, initial):
    """Log-likelihood of the sequences sym[off[q]:off[q+1]] (scaled forward pass)."""
    n_state = emission.shape[0]
    ll = 0.0
    prev, cur = np.empty(n_state), np.empty(n_state)
    for q in range(len(off) - 1):
        o = sym[off[q]:off[q + 1]]
        for t in range(len(o)):
            for j in range(n_state):
                if t == 0:
                    cur[j] = initial[j] * emission[j, o[0]]
                else:
                    s = 0.0
                    for i in range(n_state):
                        s += prev[i] * transition[i, j]
                    cur[j] = s * emission[j, o[t]]
            total = 0.0
            for j in range(n_state):
                total += cur[j]
            for j in range(n_state):
                prev[j] = cur[j] / total
            ll += np.log(total)
    return ll


def normalised(x):
    x = np.maximum(x, FLOOR)
    return x / x.sum(axis=-1, keepdims=True)


def baum_welch(sym, off, start, max_iter=MAX_ITER, rel_tol=REL_TOL):
    """Baum-Welch from the model `start`, emissions, transition matrix and initial law all updated.
    Returns the model, the log-likelihood trace, and whether it converged."""
    model = {k: np.array(v, dtype=float) for k, v in start.items()}
    trace, converged = [], False
    for _ in range(max_iter):
        ll, cb, ca, c0 = em_stats(sym, off, model["emission"], model["transition"], model["initial"])
        trace.append(ll)
        if len(trace) > 1 and abs(trace[-1] - trace[-2]) < rel_tol * abs(trace[-2]):
            converged = True
            break
        for key, count in (("emission", cb), ("transition", ca), ("initial", c0)):
            new = normalised(count / np.maximum(count.sum(axis=-1, keepdims=True), 1e-300))
            model[key] = np.where(count.sum(axis=-1, keepdims=True) > 0, new, model[key])
    return model, trace, converged


def random_start(rng):
    """Emission rows from a flat Dirichlet law, sticky random transitions, random initial law."""
    stay = rng.uniform(0.9, 0.995, 3)
    transition = np.empty((3, 3))
    for i in range(3):
        other = rng.dirichlet([1.0, 1.0]) * (1 - stay[i])
        transition[i] = np.insert(other, i, stay[i])
    return dict(emission=rng.dirichlet(np.ones(8), 3), transition=transition, initial=rng.dirichlet(np.ones(3)))


# ---- meaning of the states --------------------------------------------------------------------

def identify(emission):
    """Permutation p with p[0], p[1], p[2] the rows holding climb, search, transition.
    Climb: the row giving most weight to (curved, rising, one sign); transition: of the others, the
    straightest; search: the remaining one."""
    climb = int(np.argmax(emission[:, 3]))
    rest = [i for i in range(3) if i != climb]
    straight = emission[:, SYMBOL_BITS[:, 0] == 1].sum(axis=1)
    transition = max(rest, key=lambda i: straight[i])
    search = [i for i in rest if i != transition][0]
    return [climb, search, transition]


def permute(model, p):
    return dict(emission=model["emission"][p], transition=model["transition"][np.ix_(p, p)], initial=model["initial"][p])


def feature_shares(emission):
    """Per state (climb, search, transition): the probability that each feature is 1."""
    return {name: {feat: float(emission[i] @ SYMBOL_BITS[:, f]) for f, feat in enumerate(("straight", "rising", "one_sign"))}
            for i, name in enumerate(phases.STATES)}


def meaning_ok(emission):
    """Whether the rows, in the order climb, search, transition, keep the meaning of the names (see
    MEANING_HIGH)."""
    c, se, tr = (feature_shares(emission)[name] for name in phases.STATES)
    climb = c["rising"] >= MEANING_HIGH and c["one_sign"] >= MEANING_HIGH and c["straight"] <= MEANING_LOW
    search = se["straight"] <= MEANING_LOW and not (se["rising"] >= MEANING_HIGH and se["one_sign"] >= MEANING_HIGH)
    return bool(climb and search and tr["straight"] >= MEANING_HIGH)


def perturbed_start(rng):
    """His parameters mixed with a random start."""
    r = random_start(rng)
    return {k: (1 - PERTURB) * np.asarray(HIS[k]) + PERTURB * r[k] for k in HIS}


# ---- flights ----------------------------------------------------------------------------------

@lru_cache(maxsize=1)
def config():
    return yaml.safe_load((DYNAMICS / "configs/preprocessing.yaml").read_text())


def max_gap(k):
    return max(MIN_GAP_S, GAP_STEPS * k)


def read_flight(path, disc, cfg):
    """The flight after the xc-dynamics pre-processing: (dict of segment, t, xyz, or None if it is
    dropped, and the row of its counters). t is whole seconds from the first fix of the file,
    xyz the east, north, up of the fixes in the frame of the first fix kept."""
    fixes, row = preprocess(path, cfg["altitude"], cfg["speed"]["median_max_kmh"][disc], cfg["speed"]["max_kmh"][disc],
                            cfg["excursion"]["window_s"], cfg["vertical"]["max_ms"][disc], cfg["airborne"],
                            cfg["sampling"]["max_s"])
    if not (row["gnss_ok"] and row["speed_ok"] and row.get("airborne_ok", False)):
        return None, row
    xyz = np.vstack(phases.enu(fixes.lat.to_numpy(), fixes.lon.to_numpy(), fixes.gnss_alt.to_numpy()))
    return dict(segment=fixes.segment.to_numpy(), t=np.round(fixes.t.to_numpy(float)), xyz=xyz), row


def segments(flight):
    """(segment number, times, xyz) of every segment of a flight."""
    for seg in np.unique(flight["segment"]):
        s = flight["segment"] == seg
        yield int(seg), flight["t"][s], flight["xyz"][:, s]


def grids(t, xyz, k):
    """The 1-s grids of a segment of a flight logged every k seconds. The step of the segment is its median
    step: linear interpolation at 1 s, cubic spline above. A gap longer than max_gap of the larger of k and
    the step ends a piece; pieces of fewer than phases.WINDOW seconds are left out."""
    dt = np.diff(t)
    if not (dt > 0).any():
        return []
    step = float(np.median(dt[dt > 0]))
    return [(g, v) for g, v in phases.regular_grid(t, xyz, max_gap(max(k, step)), cubic=step > 1) if len(g) >= phases.WINDOW]


def label_segment(t, xyz, k, alpha, model):
    """Final state of every fix of a segment (-1 where unlabelled) and the start time of its piece."""
    state, piece = np.full(len(t), -1), np.full(len(t), -1)
    for grid, vals in grids(t, xyz, k):
        obs, _, _ = phases.features(grid, *vals, alpha)
        states = decode(symbols_of(obs), model)
        sel = (t >= grid[0]) & (t <= grid[-1])
        state[sel] = states[(t[sel] - grid[0]).astype(int)]
        piece[sel] = int(grid[0])
    return state, piece


def training_item(args):
    """Symbols of the pieces of one 1-Hz flight: (flight id, symbols, piece lengths), or (id, None,
    reason) if the flight fails his selection."""
    fid, path, disc = args
    flight, row = read_flight(path, disc, config())
    if flight is None:
        return fid, None, "dropped"
    steps = [np.diff(t) for _, t, _ in segments(flight)]
    steps = np.concatenate([s for s in steps if len(s)]) if steps else np.empty(0)
    if len(flight["t"]) < MIN_FIXES or not len(steps) or steps.mean() >= MAX_MEAN_STEP_S:
        return fid, None, "not_1hz"
    alpha = phases.ALPHA[disc]
    sym, lens = [], []
    for _, t, xyz in segments(flight):
        for grid, vals in grids(t, xyz, 1):
            obs, _, _ = phases.features(grid, *vals, alpha)
            sym.append(symbols_of(obs))
            lens.append(len(grid))
    if not sym:
        return fid, None, "no_piece"
    return fid, np.concatenate(sym), np.array(lens)


# ---- training sample --------------------------------------------------------------------------

def catalogue(disc):
    return pd.read_csv(DISCIPLINES[disc]["root"] / "catalog/catalog.csv", usecols=["flight_id", "date", "season_year", "local_path"])


def candidates(disc):
    """The flights logged at 1 Hz that pass the pre-processing, with their year and ground stratum:
    the ground class where known (paragliders), else the tercile of the take-off altitude."""
    root = DISCIPLINES[disc]["root"]
    t = pd.read_csv(root / "derived/preprocessing/airborne/flights.csv")
    ok = t.gnss_ok.astype(bool) & t.speed_ok.astype(bool) & t.airborne_ok.fillna(False).astype(bool)
    t = t[ok & (t.sampling_s == 1.0) & (t.fixes_read >= MIN_FIXES)][["flight_id"]]
    cat = catalogue(disc)
    year = pd.to_numeric(cat.date.str[:4], errors="coerce")
    cat["year"] = year.where(year > 2000, cat.season_year)
    t = t.merge(cat[["flight_id", "year", "local_path"]], on="flight_id")
    alt = pd.read_parquet(DISCIPLINES[disc]["tracks"] / "derived/flights_meta.parquet", columns=["flight_id", "alt0"])
    alt["flight_id"] = alt.flight_id.astype(int)
    t = t.merge(alt, on="flight_id", how="left")
    edges = t.alt0.quantile([1 / 3, 2 / 3]).to_numpy()
    t["stratum"] = np.where(t.alt0.isna(), "altitude_unknown", np.where(t.alt0 < edges[0], "altitude_low",
                            np.where(t.alt0 < edges[1], "altitude_mid", "altitude_high")))
    if disc == "para" and GROUND.exists():
        ground = pd.read_parquet(GROUND, columns=["flight", "class"])
        ids = pd.read_parquet(GROUND_FLIGHTS, columns=["flight", "flight_id"])
        ids = ids.merge(ground, on="flight")
        ids["flight_id"] = ids.flight_id.astype(int)
        t = t.merge(ids[["flight_id", "class"]], on="flight_id", how="left")
        known = t["class"].notna() & (t["class"] != "unknown")
        t["stratum"] = np.where(known, t["class"], t["stratum"])
    return t.reset_index(drop=True)


def allocate(sizes, n):
    """Quotas summing to min(n, sum(sizes)), as equal as the sizes allow (water filling)."""
    quota = np.zeros(len(sizes), dtype=int)
    open_ = np.ones(len(sizes), bool)
    left = min(n, int(sum(sizes)))
    while left > 0 and open_.any():
        share = max(left // open_.sum(), 1)
        for i in np.flatnonzero(open_):
            take = min(share, sizes[i] - quota[i], left)
            quota[i] += take
            left -= take
            open_[i] = quota[i] < sizes[i]
            if left == 0:
                break
    return quota


def draw_sample(pool, n, seed=SEED):
    """n flights: equal quotas over the ground strata, within each stratum spread over the years in
    proportion to the pool, uniformly at random inside every (stratum, year) cell."""
    rng = np.random.default_rng(seed)
    chosen = []
    strata = sorted(pool.stratum.unique())
    quota = allocate([int((pool.stratum == s).sum()) for s in strata], n)
    for s, q in zip(strata, quota):
        cell = pool[pool.stratum == s]
        years = sorted(cell.year.unique())
        sizes = np.array([int((cell.year == y).sum()) for y in years])
        want = q * sizes / sizes.sum()
        take = np.floor(want).astype(int)
        for i in np.argsort(-(want - take))[: q - take.sum()]:
            take[i] += 1
        for y, m in zip(years, take):
            ids = cell[cell.year == y].flight_id.to_numpy()
            chosen += list(rng.choice(ids, size=min(m, len(ids)), replace=False))
    return np.array(sorted(chosen))


def build_training(disc, n):
    """Draw the sample, read the flights and write the training symbols to CACHE; returns the path."""
    pool = candidates(disc)
    sample = draw_sample(pool, n)
    paths = pool.set_index("flight_id").local_path
    tracks = DISCIPLINES[disc]["tracks"]
    items = [(int(f), tracks / paths[f], disc) for f in sample]
    kept, lost = {}, {}
    with ProcessPoolExecutor(WORKERS) as pool_:
        for n_done, (fid, sym, lens) in enumerate(pool_.map(training_item, items, chunksize=8), 1):
            if sym is None:
                lost[fid] = lens
            else:
                kept[fid] = (sym, lens)
            if n_done % 200 == 0:
                print(f"{disc}: {n_done}/{len(items)} flights read", flush=True)
    ids = np.array(sorted(kept))
    held = np.zeros(len(ids), bool)
    held[np.random.default_rng(SEED + 1).permutation(len(ids))[: int(round(HELD_OUT * len(ids)))]] = True
    arrays = {}
    for name, mask in (("train", ~held), ("held", held)):
        syms = [kept[f][0] for f in ids[mask]]
        lens = np.concatenate([kept[f][1] for f in ids[mask]])
        arrays[f"{name}_sym"] = np.concatenate(syms)
        arrays[f"{name}_off"] = np.r_[0, np.cumsum(lens)]
        arrays[f"{name}_flights"] = ids[mask]
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"training_{disc}.npz"
    np.savez(path, **arrays, drawn=len(sample), pool=len(pool), lost=np.array(sorted(lost.items(), key=str), dtype=object),
             strata=pool.set_index("flight_id").loc[ids].stratum.to_numpy().astype(str), strata_ids=ids)
    return path, pool, sample, lost


def fit_start(args):
    """One Baum-Welch run, in a worker: (label, trace, converged, model, held-out log-likelihood)."""
    disc, label, start = args
    z = np.load(CACHE / f"training_{disc}.npz", allow_pickle=True)
    model, trace, converged = baum_welch(z["train_sym"], z["train_off"], start)
    held = log_likelihood(z["held_sym"], z["held_off"], model["emission"], model["transition"], model["initial"])
    return label, trace, converged, model, held


# ---- checks on the held-out flights -----------------------------------------------------------

def state_statistics(args):
    """For one held-out flight and one model: per state, the fixes, the fixes with each feature, the
    sums of the climb rate and of the absolute turn, on the 1-s grid."""
    path, disc, model = args
    flight, _ = read_flight(path, disc, config())
    acc = np.zeros((3, 6))
    for _, t, xyz in segments(flight):
        for grid, vals in grids(t, xyz, 1):
            obs, turn, vz = phases.features(grid, *vals, phases.ALPHA[disc])
            states = decode(symbols_of(obs), model)
            for s in range(3):
                m = states == s
                good = np.isfinite(turn) & np.isfinite(vz)
                acc[s] += [m.sum(), *(obs[m, c].sum() for c in range(3)), np.nansum(vz[m & good]), np.nansum(np.abs(turn[m & good]))]
    return acc


def meaning_table(disc, ids, models):
    """Per model and state on the held-out flights: share of fixes, features, mean climb rate, mean
    absolute turn."""
    cat = catalogue(disc).set_index("flight_id").local_path
    tracks = DISCIPLINES[disc]["tracks"]
    out = {}
    with ProcessPoolExecutor(WORKERS) as pool:
        for name, model in models.items():
            acc = sum(pool.map(state_statistics, [(tracks / cat[f], disc, model) for f in ids], chunksize=4))
            n = acc[:, 0]
            out[name] = {phases.STATES[s]: dict(time_share=float(n[s] / n.sum()), straight=float(acc[s, 1] / n[s]),
                                                rising=float(acc[s, 2] / n[s]), one_sign=float(acc[s, 3] / n[s]),
                                                mean_climb_rate_ms=float(acc[s, 4] / n[s]),
                                                mean_abs_turn_rad=float(acc[s, 5] / n[s])) for s in range(3)}
    return out


def commit(repo):
    sha = subprocess.run(["git", "-C", str(repo), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "-C", str(repo), "status", "--porcelain"], capture_output=True, text=True).stdout.strip())
    return dict(commit=sha, uncommitted_changes=dirty)


def as_lists(model):
    return {k: np.asarray(v).tolist() for k, v in model.items()}


def as_arrays(model):
    return {k: np.array(model[k], dtype=float) for k in ("emission", "transition", "initial")}


def load_params(disc):
    """The model, its straightness threshold and the whole record of out/refit_params_<disc>.json."""
    record = json.loads((OUT / f"refit_params_{disc}.json").read_text())
    return as_arrays(record["model"]), record["alpha_rad"], record


# ---- training ---------------------------------------------------------------------------------

def train(disc, rebuild=False):
    OUT.mkdir(exist_ok=True)
    path = CACHE / f"training_{disc}.npz"
    if rebuild or not path.exists():
        build_training(disc, TRAIN_FLIGHTS[disc])
    z = np.load(path, allow_pickle=True)
    n_fix = int(len(z["train_sym"])), int(len(z["held_sym"]))
    print(f"{disc}: {len(z['train_flights'])} training and {len(z['held_flights'])} held-out flights, {n_fix} fixes", flush=True)
    rng = np.random.default_rng(SEED + 2)
    starts = [("his", HIS)] + [(f"random{i}", random_start(rng)) for i in range(STARTS)]
    starts += [(f"perturbed{i}", perturbed_start(rng)) for i in range(STARTS)]
    with ProcessPoolExecutor(WORKERS) as pool_:
        results = list(pool_.map(fit_start, [(disc, label, start) for label, start in starts]))
    ll_his = {name: log_likelihood(z[f"{name}_sym"], z[f"{name}_off"], HIS["emission"], HIS["transition"], HIS["initial"])
              for name in ("train", "held")}
    table, canonical = [], {}
    for label, trace, converged, model, held in results:
        p = identify(model["emission"])
        canonical[label] = permute(model, p)
        table.append(dict(start=label, iterations=len(trace), converged=bool(converged), train_ll=trace[-1], held_ll=held,
                          held_ll_per_fix=held / n_fix[1], states_found=p, meaning_ok=meaning_ok(canonical[label]["emission"]),
                          model=as_lists(canonical[label])))
    # The model kept: the best held-out likelihood among the starts whose states keep his meaning (all of them
    # if none does).
    eligible = [r for r in table if r["meaning_ok"]] or table
    best = max(eligible, key=lambda r: r["held_ll"])
    best_train = max(r["train_ll"] for r in eligible)
    near = [r["start"] for r in eligible if abs(r["train_ll"] - best_train) < BEST_TOL * abs(best_train)]
    for r in table:
        r["reaches_best"] = r["start"] in near
    spread = {key: np.ptp(np.stack([canonical[s][key] for s in near]), axis=0).max().item() for key in HIS}
    overall = max(table, key=lambda r: r["held_ll"])
    model = canonical[best["start"]]
    ids = z["held_flights"]
    record = dict(discipline=disc, used="own", model=as_lists(model), alpha_rad=phases.ALPHA[disc], states=list(phases.STATES),
                  his_model=as_lists(HIS), best_start=best["start"], best_start_any_meaning=overall["start"],
                  training=dict(flights_drawn=int(z["drawn"]), pool=int(z["pool"]), train_flights=len(z["train_flights"]),
                                held_out_flights=len(ids), train_fixes=n_fix[0], held_out_fixes=n_fix[1],
                                lost=len(z["lost"]), strata=pd.Series(z["strata"]).value_counts().to_dict(),
                                train_flight_ids=z["train_flights"].tolist(), held_out_flight_ids=ids.tolist()),
                  settings=dict(seed=SEED, random_starts=STARTS, perturbed_starts=STARTS, perturbation=PERTURB, max_iter=MAX_ITER,
                                rel_tol=REL_TOL, floor=FLOOR, best_tol=BEST_TOL, meaning_high=MEANING_HIGH, meaning_low=MEANING_LOW, transition_updated=True),
                  starts=table, starts_near_best=near, spread_across_best_starts_max_abs_difference=spread,
                  held_out_log_likelihood=dict(refit=best["held_ll"], his=ll_his["held"], best_any_meaning=overall["held_ll"]),
                  train_log_likelihood=dict(refit=best["train_ll"], his=ll_his["train"]),
                  feature_shares=feature_shares(model["emission"]), meaning_ok=best["meaning_ok"],
                  feature_shares_best_any_meaning=feature_shares(np.array(overall["model"]["emission"])),
                  date=datetime.now().astimezone().isoformat(timespec="seconds"), xc_lab=commit(LAB), xc_dynamics=commit(DYNAMICS))
    record["meaning_on_held_out"] = meaning_table(disc, ids, {"refit": model, "his": HIS})
    if disc == "hang":
        record |= decide_hang(record, z)
    (OUT / f"refit_params_{disc}.json").write_text(json.dumps(record, indent=1, default=float))
    summary = ("used", "best_start", "best_start_any_meaning", "held_out_log_likelihood", "spread_across_best_starts_max_abs_difference")
    print(json.dumps({k: record[k] for k in summary}, indent=1))


def pair_confusion(args):
    """Confusion matrix (rows: model a, columns: model b) of the labels of two models on the grid of one flight."""
    path, disc, a, b = args
    flight, _ = read_flight(path, disc, config())
    conf = np.zeros((3, 3), dtype=int)
    for _, t, xyz in segments(flight):
        for grid, vals in grids(t, xyz, 1):
            sym = symbols_of(phases.features(grid, *vals, phases.ALPHA[disc])[0])
            np.add.at(conf, (decode(sym, a), decode(sym, b)), 1)
    return conf


def decide_hang(record, z):
    """Own hang-glider fit if there are enough flights, the best start converged and the states keep their
    meaning; else the paraglider refit with the hang-glider threshold. Also reports the paraglider refit on
    the held-out hang gliders: likelihood, states, and agreement with the own fit."""
    para, _, para_record = load_params("para")
    ll_para = log_likelihood(z["held_sym"], z["held_off"], para["emission"], para["transition"], para["initial"])
    best = next(r for r in record["starts"] if r["start"] == record["best_start"])
    n = len(z["train_flights"]) + len(z["held_flights"])
    reasons = []
    if n < OWN_HANG_MIN_FLIGHTS:
        reasons.append(f"only {n} flights at 1 Hz")
    if not best["converged"]:
        reasons.append("the best fit did not converge")
    if not record["meaning_ok"]:
        reasons.append("the states lose their meaning")
    ids = z["held_flights"]
    cat = catalogue("hang").set_index("flight_id").local_path
    tracks = DISCIPLINES["hang"]["tracks"]
    own = as_arrays(record["model"])
    with ProcessPoolExecutor(WORKERS) as pool:
        conf = sum(pool.map(pair_confusion, [(tracks / cat[f], "hang", own, para) for f in ids], chunksize=4))
    out = dict(held_out_log_likelihood=record["held_out_log_likelihood"] | dict(para_refit=ll_para),
               meaning_on_held_out=record["meaning_on_held_out"] | meaning_table("hang", ids, {"para_refit": para}),
               own_vs_para_refit_labels=dict(confusion_rows_own_columns_para=conf.tolist(), same_label=float(np.trace(conf) / conf.sum())))
    if reasons:
        out |= dict(used="para_refit", model=para_record["model"], why="; ".join(reasons))
    else:
        out |= dict(used="own", why=f"{n} flights at 1 Hz, converged, states keep their meaning")
    return out


# ---- cadence ----------------------------------------------------------------------------------

def cadence_item(args):
    """One held-out flight: the 1-Hz labels of every model against the labels from fixes thinned to k s."""
    path, disc, models = args
    flight, _ = read_flight(path, disc, config())
    alpha, rows = phases.ALPHA[disc], []
    for seg, t, xyz in segments(flight):
        for name, model in models.items():
            native = label_segment(t, xyz, 1, alpha, model)[0]
            for k in CADENCES:
                keep = (t % k) == 0
                coarse = label_segment(t[keep], xyz[:, keep], k, alpha, model)[0]
                ref = native[keep]
                both = (ref >= 0) & (coarse >= 0)
                conf = np.zeros((3, 3), dtype=int)
                np.add.at(conf, (ref[both], coarse[both]), 1)
                rows.append(dict(model=name, k=k, segment=seg, compared=int(both.sum()),
                                 **{f"c{a}{b}": int(conf[a, b]) for a in range(3) for b in range(3)}))
    return rows


def cadence_summary(table):
    """Per model and k: pooled share of same labels, climb and search kept, climb share, 10th percentile of flights."""
    out = {}
    for (model, k), g in table.groupby(["model", "k"]):
        c = g[[f"c{a}{b}" for a in range(3) for b in range(3)]].sum().to_numpy().reshape(3, 3)
        flight = g.groupby("flight_id").apply(lambda x: x[[f"c{a}{a}" for a in range(3)]].sum().sum() / x.compared.sum(), include_groups=False)
        out.setdefault(model, {})[int(k)] = dict(fixes=int(c.sum()), same_label=float(np.trace(c) / c.sum()),
                                                  climb_kept=float(c[0, 0] / c[0].sum()), search_kept=float(c[1, 1] / c[1].sum()),
                                                  transition_kept=float(c[2, 2] / c[2].sum()),
                                                  climb_share_1hz=float(c[0].sum() / c.sum()), climb_share_thinned=float(c[:, 0].sum() / c.sum()),
                                                  worst_10_percent=float(flight.quantile(0.1)))
    return out


def cadence(disc):
    model, alpha, record = load_params(disc)
    models = {"refit": model, "his": HIS}
    ids = record["training"]["held_out_flight_ids"]
    cat = catalogue(disc).set_index("flight_id").local_path
    tracks = DISCIPLINES[disc]["tracks"]
    rows = []
    with ProcessPoolExecutor(WORKERS) as pool:
        for n, (fid, res) in enumerate(zip(ids, pool.map(cadence_item, [(tracks / cat[f], disc, models) for f in ids], chunksize=4)), 1):
            rows += [dict(discipline=disc, flight_id=fid, **r) for r in res]
            if n % 50 == 0:
                print(f"{disc}: {n}/{len(ids)} flights", flush=True)
    table = pd.DataFrame(rows)
    path = OUT / "refit_cadence.csv"
    old = pd.read_csv(path) if path.exists() else pd.DataFrame()
    pd.concat([old[old.discipline != disc] if len(old) else old, table]).to_csv(path, index=False)
    summary_path = OUT / "refit_cadence.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    summary[disc] = dict(flights=len(ids), model_used=record["used"], results=cadence_summary(table))
    summary_path.write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary[disc], indent=1))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("step", choices=["train", "cadence"])
    p.add_argument("discipline", choices=list(DISCIPLINES))
    p.add_argument("--rebuild", action="store_true", help="train: draw and read the training flights again")
    args = p.parse_args()
    if args.step == "train":
        train(args.discipline, args.rebuild)
    else:
        cadence(args.discipline)


if __name__ == "__main__":
    main()
