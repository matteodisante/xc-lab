"""How much of 1 - H of real flights is the crest geometry and how much is intermittency.

For each region and discipline, four mean squared displacements as functions of the time lag tau:
  crests  the TAMSD of the crest paths against arc length s, with tau = s / v  (v: mean ground speed in glides);
  model   an intermittent ball on the same paths (model.py), Lomax glides and pauses fitted on the flights;
  dyn     the flights, positions every 1 s (flights.py);
  route   the flights with each circle replaced by its chord, against arc length, tau = s / v (reference).
H of a curve is half the least-squares slope of ln MSD against ln tau from the end of the initial ballistic
regime to the last lag every curve reaches: start = the first lag, not below FLOOR_M / v, where the local exponent
alpha = 1/2 d ln MSD / d ln tau of that curve falls below ALPHA_END. The same rule is applied to the four curves
and, in each bootstrap resample, to the resampled curve. Crests and model share the resamples of the trees; the
flights are resampled as whole flights.

    1 - H_dyn = (1 - H_crests) + (H_crests - H_model) + (H_model - H_dyn)

    uv run python decompose.py [--cache DIR] [--data DIR]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import flights as fl
import model as md
import pauses as pa

REGIONS, DISCS = ("alps", "pyrenees"), ("para", "hang")
DISC_LONG = {"para": "paragliders", "hang": "hang_gliders"}
ALPHA_END, FLOOR_M = 0.98, 100.0
N_BOOT, N_REAL, SEED = 1000, 3, 20261113
MIN_FLIGHTS = 30  # a group with fewer flights is shown but not interpreted
HERE = Path(__file__).parent
CURVES = ("geo", "model", "dyn", "route")


def newest_cache():
    return max(Path("/Volumes/SSD_DISANTE/xc-lab/ridges").glob("flights_cleaned_*"), key=lambda d: (d / "flights.csv").stat().st_mtime)


def load_paths(data, region):
    """Every joined crest path of the region (export_paths.py; Lambert-93, every 5 m) and the tree of each."""
    z = np.load(Path(data) / f"paths_all_{region}.npz")
    xy = z["xy"].astype(float) + z["origin"]
    return [xy[a:b] for a, b in zip(z["off"][:-1], z["off"][1:])], z["tree"]


# bootstrap on log mean TAMSD
def draw_counts(m, n_boot, rng):
    """(n_boot + 1, m) numbers of times each of m groups is drawn: row 0 is the data itself."""
    return np.vstack([np.ones(m), rng.multinomial(m, np.full(m, 1 / m), size=n_boot)])


def log_means(rows, group, counts):
    """ln of the mean over rows with equal weight, per resample of the groups; NaN below fl.MIN_PIECES rows."""
    v = np.isfinite(rows)
    x, v = np.where(v, rows, 0.0), v.astype(float)
    out = []
    for s in range(0, len(counts), 100):
        c = counts[s : s + 100][:, group]
        n = c @ v
        with np.errstate(all="ignore"):
            out.append(np.where(n >= fl.MIN_PIECES, np.log((c @ x) / n), np.nan))
    return np.vstack(out)


def alpha(lnm, tau):
    return np.gradient(lnm, np.log(tau), axis=1) / 2


def rule_fit(lnm, tau, floor, tau_max):
    """H and start lag of each row of lnm under the rule of the module docstring."""
    a = alpha(lnm, tau)
    ok = np.isfinite(a) & (tau >= floor)
    first = np.argmax(ok & (a < ALPHA_END), axis=1)
    found = (ok & (a < ALPHA_END)).any(1)
    start = np.where(found, tau[first], np.nan)
    m = np.isfinite(lnm) & (tau >= start[:, None]) & (tau <= tau_max)
    x = np.log(tau)
    n = m.sum(1)
    xm = (m * x).sum(1) / np.maximum(n, 1)
    ym = np.where(m, lnm, 0).sum(1) / np.maximum(n, 1)
    dx = np.where(m, x - xm[:, None], 0)
    with np.errstate(all="ignore"):
        h = (dx * np.where(m, lnm - ym[:, None], 0)).sum(1) / (dx**2).sum(1) / 2
    return np.where(n >= 3, h, np.nan), start


def band(x):
    return fl.interval(x[1:], axis=0)


def ci(x):
    lo, hi = fl.interval(x[1:])
    return float(x[0]), float(lo), float(hi)


def last_lag(lnm, tau):
    return tau[np.isfinite(lnm[0])].max()


def analyse_group(geo_rows, trees, mod_rows, dyn, route, v, rng):
    """Curves, H (rule and fixed windows) and the decomposition of one region and discipline.

    geo_rows, mod_rows: one row per crest path (trees: its tree); dyn, route: one row per flight."""
    _, g = np.unique(trees, return_inverse=True)
    ct, cf = draw_counts(g.max() + 1, N_BOOT, rng), draw_counts(len(dyn), N_BOOT, rng)
    tau = {"geo": fl.LAGS_K * fl.STEP_M / v, "model": fl.LAGS_T.astype(float), "dyn": fl.LAGS_T.astype(float), "route": fl.LAGS_K * fl.STEP_M / v}
    idx = np.arange(len(dyn))
    lnm = {"geo": log_means(geo_rows, g, ct), "model": log_means(mod_rows, g, ct), "dyn": log_means(dyn, idx, cf), "route": log_means(route, idx, cf)}
    floor, tau_max = FLOOR_M / v, min(last_lag(lnm[k], tau[k]) for k in CURVES)
    res = {"v": v, "tau_floor": floor, "tau_max": tau_max, "n_flights": int(np.isfinite(dyn[:, 0]).sum()), "n_paths": len(geo_rows),
           "curves": {}, "H": {}, "start": {}, "window": {}}
    h = {}
    for k in CURVES:
        a = alpha(lnm[k], tau[k])
        res["curves"][k] = {"tau": tau[k].tolist(), "lnm": lnm[k][0].tolist(), "lnm_band": band(lnm[k]).tolist(),
                            "alpha": a[0].tolist(), "alpha_band": band(a).tolist()}
        h[k], st = rule_fit(lnm[k], tau[k], floor, tau_max)
        res["H"][k], res["start"][k] = ci(h[k]), ci(st)
        res["window"][k] = {fl.fmt_window(lo, hi): ci(fl.slope_h(lnm[k], tau[k], lo / v, hi / v)) for lo, hi in fl.WINDOWS_M}
    res["decomposition"] = decomposition(h["geo"], h["model"], h["dyn"])
    res["decomposition_route"] = decomposition(h["geo"], h["model"], h["route"])
    return res


def decomposition(hg, hm, hr):
    """Three terms of 1 - H_real, each as a percentage of it, with bootstrap intervals."""
    tot = 1 - hr
    terms = {"geometry": 1 - hg, "intermittency": hg - hm, "other": hm - hr}
    return {"total": ci(tot), **{k: ci(100 * t / tot) for k, t in terms.items()}}


def main(cache, data):
    cache, data = Path(cache), Path(data)
    rng = np.random.default_rng(SEED)
    t = pd.read_csv(cache / "flights.csv", dtype={"flight": str})
    ok = t[t["status"] == "ok"].reset_index(drop=True)
    ok["disc"] = ok["discipline"].map({v: k for k, v in DISC_LONG.items()})
    rows = {k: np.load(cache / f"flights_rows_{k}.npz") for k in ("dyn", "route")}
    assert (rows["dyn"]["flight"] == ok["flight"].to_numpy()).all()
    out = {"phases": {}, "speed": {}, "groups": {}}
    # phases of the 1 Hz flights
    speed = pd.read_csv(cache / "glide_speed.csv", dtype={"flight_id": str})
    one_hz = ok[ok["dt_median_s"] == 1]
    for d in DISCS:
        runs = pd.read_parquet(Path(pa.DISK) / pa.FOLDERS[d] / "derived" / "segmentation/vilpellet/phase_segments.parquet")
        b = pa.blocks(runs)
        for r in REGIONS:
            sel = one_hz[(one_hz["disc"] == d) & (one_hz["region"] == r)]
            bb = b[b["flight_id"].isin(set(sel["flight"]))]
            sp = speed[(speed["discipline"] == DISC_LONG[d]) & speed["flight_id"].isin(set(sel["flight"]))]
            key = f"{r}/{d}"
            if sp.empty:
                continue
            out["speed"][key] = glide_speed(sp, rng)
            out["phases"][key] = {"n_flights_1hz": len(sel), "n_flights_with_phases": int(bb["flight_id"].nunique())}
            for kind in (pa.GLIDE, pa.PAUSE):
                x = bb[bb["kind"] == kind]
                if len(x) >= 30:
                    out["phases"][key][kind] = pa.fit_phase(x["duration_s"].to_numpy(), x["flight_id"].to_numpy(), seed=SEED)
                    out["phases"][key][kind]["survival"] = [a.tolist() for a in pa.survival(x["duration_s"].to_numpy())]
    # curves
    for r in REGIONS:
        paths, trees = load_paths(data, r)
        geo = np.array([fl.pooled([p], fl.LAGS_K) for p in paths])
        for d in DISCS:
            key = f"{r}/{d}"
            if key not in out["speed"]:
                continue
            ph, v = out["phases"][key], out["speed"][key]["v"]
            sel = (ok["region"] == r) & (ok["disc"] == d)
            if not (pa.GLIDE in ph and pa.PAUSE in ph) or sel.sum() < 10:
                continue
            glide = (ph[pa.GLIDE]["mu"], ph[pa.GLIDE]["tau0"], ph[pa.GLIDE]["x0"])
            pause = (ph[pa.PAUSE]["mu"], ph[pa.PAUSE]["tau0"], ph[pa.PAUSE]["x0"])
            mod = md.model_rows(paths, rng, v, glide, pause, N_REAL)
            try:
                out["groups"][key] = analyse_group(geo, trees, mod, rows["dyn"]["rows"][sel.to_numpy()], rows["route"]["rows"][sel.to_numpy()], v, rng)
            except (ValueError, IndexError) as e:  # too few flights for any lag with MIN_PIECES of them
                print(key, "not analysed:", e, flush=True)
    return out


def glide_speed(sp, rng):
    """Mean ground speed in glides, total length over total time, with a bootstrap over flights."""
    d, s = sp["dist_m"].to_numpy(), sp["time_s"].to_numpy(float)
    w = rng.multinomial(len(d), np.full(len(d), 1 / len(d)), size=N_BOOT)
    boot = (w @ d) / (w @ s)
    return {"v": float(d.sum() / s.sum()), "v_ci": [float(x) for x in np.percentile(boot, [2.5, 97.5])], "n_flights": len(d)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cache", default=None)
    ap.add_argument("--data", default=None)
    a = ap.parse_args()
    import run_flights

    cache = Path(a.cache) if a.cache else newest_cache()
    res = main(cache, a.data or run_flights.newest_data())
    (cache / "decomposition.json").write_text(json.dumps(res, indent=1, default=float))
