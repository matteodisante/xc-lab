"""Fission-fusion: the size distribution of groups from the rates at which they gain and lose pilots.

T(n) is the time spent in configurations of n pilots: the chance that a group seen at a random
instant has n pilots. If sizes change one pilot at a time, the flux from n to n + 1 balances the
flux back (a birth-death chain):

    T(n + 1) / T(n) = g(n) / l(n + 1),

with g(n) the rate at which a configuration of n gains pilots (join) and l(n) the rate at which it
loses them (leave, landing), per hour at size n (describe.py). Three predictions of T(n), each
normalised to the observed T over 2..N_MAX:

- measured: g and l of each size, 2..N_MAX (the last pooled);
- per capita: g(n) = a n and l(n) = mu n^(1 - gamma), every member joining others at the same
  rate and leaving at a rate that changes with size as n^-gamma; a, mu and gamma are fitted to the
  rates, not to T, and the chain runs to the largest observed size;
- log-series: the same with gamma = 0, T(n) proportional to x^n / n with x = a / mu.

And the form of Bonabeau et al. (1999), T(n) proportional to n^-alpha exp(-n / n_c), fitted to T.
Splits and merges are left out of the chain; their share of the changes is reported. Bands: 5-95%
of N_BOOT draws of the blocks of the note.

    uv run python fission.py [--run NAME]    # from this folder, about 30 s
"""

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from describe import events
from tamsd import LABELS, OUT, SEED, save

N_MAX = 8
N_BOOT = 200
MIN_HOURS = 10


def chain(g, l):
    """T(n) / T(2) for n = 2..len(g) + 1 from the gains g(n) and losses l(n + 1), n = 2.."""
    return np.r_[1.0, np.cumprod(np.asarray(g) / np.asarray(l))]


def per_capita(a, mu, gamma, n_top):
    """T(n) / T(2), n = 2..n_top, with g(n) = a n and l(n) = mu n^(1 - gamma)."""
    m = np.arange(2, n_top)
    return chain(a * m, mu * (m + 1.0) ** (1 - gamma))


def tables(confs, flights, episodes):
    ev = events(confs, flights)
    hours = (confs["t_end"] - confs["t_start"]).to_numpy() / 3600
    block = episodes.set_index("episode_id")["block"].reindex(confs["episode_id"]).to_numpy()
    return pd.DataFrame({"n": confs["n"].to_numpy(), "hours": hours, "outcome": ev["outcome"].to_numpy(), "block": block})


def rates(t, w=None):
    """Per size 2..N_MAX (pooled above): hours, mean size, gains, losses, splits and merges per hour;
    with `w`, the same for each row of block weights."""
    ids, k = np.unique(t["block"], return_inverse=True)
    j = np.clip(t["n"].to_numpy(), 2, N_MAX) - 2
    kinds = {"gain": ["join"], "loss": ["leave", "landing"], "split": ["split"], "merge": ["merge"]}
    sums = {}
    for name, outcomes in kinds.items():
        z = np.zeros((len(ids), N_MAX - 1))
        np.add.at(z, (k, j), t["outcome"].isin(outcomes).to_numpy().astype(float))
        sums[name] = z
    for name, x in (("hours", t["hours"].to_numpy()), ("nh", (t["n"] * t["hours"]).to_numpy())):
        z = np.zeros((len(ids), N_MAX - 1))
        np.add.at(z, (k, j), x)
        sums[name] = z
    w = np.ones((1, len(ids))) if w is None else w
    s = {name: w @ z for name, z in sums.items()}  # draws x sizes
    return {"hours": s["hours"], "n": s["nh"] / s["hours"],
            **{name: s[name] / s["hours"] for name in ("gain", "loss", "split", "merge")}}


def fit_capita(r):
    """a (per member per hour) and mu, gamma of l(n) = mu n^(1 - gamma), from the rates of one draw."""
    n, h = r["n"], r["hours"]
    a = (r["gain"] * h).sum() / (n * h).sum()
    (log_mu, gamma), _ = curve_fit(lambda x, lm, gm: lm - gm * x, np.log(n), np.log(r["loss"] / n), sigma=1 / np.sqrt(h))
    return a, np.exp(log_mu), gamma


def bonabeau(n, t):
    """alpha and n_c of T(n) proportional to n^-alpha exp(-n / n_c), least squares on log T."""
    (c, alpha, n_c), _ = curve_fit(lambda x, c, al, nc: c - al * np.log(x) - x / nc, n, np.log(t), p0=(0, 2, 10),
                                   bounds=([-np.inf, -5, 0.1], [np.inf, 10, 1e4]))
    return alpha, n_c


def fission(confs, flights, episodes, rng):
    t = tables(confs, flights, episodes)
    hours = t.groupby("n")["hours"].sum()
    hours = hours[hours >= MIN_HOURS]
    n_obs, T = hours.index.to_numpy(), hours.to_numpy() / hours.sum()
    n_top = int(n_obs.max()) + 1
    ids = np.unique(t["block"])
    w = rng.multinomial(len(ids), np.full(len(ids), 1 / len(ids)), size=N_BOOT).astype(float)
    point, draws = rates(t), rates(t, w)
    sizes = np.arange(2, N_MAX + 1)
    first = n_obs <= N_MAX

    def norm(pred, upto):
        return pred / pred[: upto].sum() * T[: upto].sum()

    measured = chain(point["gain"][0, :-1], point["loss"][0, 1:])
    measured_draws = np.array([chain(draws["gain"][b, :-1], draws["loss"][b, 1:]) for b in range(N_BOOT)])
    capita = {"point": fit_capita({k: v[0] for k, v in point.items()})}
    cap_draws = np.array([fit_capita({k: v[b] for k, v in draws.items()}) for b in range(N_BOOT)])
    a, mu, gamma = capita["point"]
    pc = per_capita(a, mu, gamma, n_top)
    pc_draws = np.array([per_capita(*c, n_top) for c in cap_draws])
    ls = per_capita(a, mu, 0.0, n_top)
    alpha, n_c = bonabeau(n_obs.astype(float), T)
    k = int(first.sum())
    changes = point["gain"] + point["loss"] + point["split"] + point["merge"]
    return {
        "n": n_obs.tolist(), "T": T.tolist(), "hours": hours.tolist(), "sizes": sizes.tolist(),
        "rates": {name: point[name][0].tolist() for name in ("gain", "loss", "split", "merge", "n")},
        "rates_band": {name: np.percentile(draws[name], [5, 95], axis=0).tolist() for name in ("gain", "loss")},
        "multi_share": ((point["split"] + point["merge"]) / changes)[0].tolist(),
        "measured": norm(measured, k).tolist(),
        "measured_band": np.percentile([norm(m, k) for m in measured_draws], [5, 95], axis=0).tolist(),
        "capita": {"a": a, "mu": mu, "gamma": gamma,
                   "a_band": np.percentile(cap_draws[:, 0], [5, 95]).tolist(),
                   "mu_band": np.percentile(cap_draws[:, 1], [5, 95]).tolist(),
                   "gamma_band": np.percentile(cap_draws[:, 2], [5, 95]).tolist()},
        "per_capita": norm(pc, k)[n_obs - 2].tolist(),
        "per_capita_band": np.percentile([norm(p, k)[n_obs - 2] for p in pc_draws], [5, 95], axis=0).tolist(),
        "log_series": norm(ls, k)[n_obs - 2].tolist(), "x": a / mu,
        "bonabeau": {"alpha": alpha, "n_c": n_c},
    }


def plot(r, path):
    """Left: per-capita gains and losses by size, with the fit; right: T(n) observed and predicted."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(10.5, 3.9))
    s = np.array(r["sizes"], float)
    n = np.array(r["rates"]["n"])
    for key, color, marker, label in (("gain", "#1baf7a", "^", "gain (join)"), ("loss", "#2a78d6", "v", "loss (leave, landing)")):
        v = np.array(r["rates"][key]) / n
        band = np.array(r["rates_band"][key]) / n
        a.fill_between(n, band[0], band[1], color=color, alpha=0.2, lw=0)
        a.plot(n, v, color=color, marker=marker, ms=5, lw=0, label=label + " per member")
    c = r["capita"]
    x = np.linspace(2, n.max(), 50)
    a.plot(x, np.full_like(x, c["a"]), color="#1baf7a", lw=1, ls="--")
    a.plot(x, c["mu"] * x ** -c["gamma"], color="#2a78d6", lw=1, ls="--",
           label=rf"$\mu n^{{-\gamma}}$, $\gamma$ = {c['gamma']:.2f}")
    a.set(xscale="log", yscale="log", xlabel="pilots (mean of each size; last: 8+)", ylabel="per member per hour")
    a.set_xticks([2, 3, 4, 5, 6, 8], labels=["2", "3", "4", "5", "6", "8"])
    a.minorticks_off()
    a.set_yticks([0.1, 0.2, 0.3, 0.5, 0.7], labels=["0.1", "0.2", "0.3", "0.5", "0.7"])
    a.legend(frameon=False, fontsize=7)
    no = np.array(r["n"])
    T = np.array(r["T"])
    b.plot(no, T, color="#0b0b0b", marker="o", ms=4, lw=0, label="observed")
    k = len(r["measured"])
    b.fill_between(no[:k], *np.array(r["measured_band"]), color="#eb6834", alpha=0.2, lw=0)
    b.plot(no[:k], r["measured"], color="#eb6834", lw=1.4, label="chain, measured rates")
    b.fill_between(no, *np.array(r["per_capita_band"]), color="#2a78d6", alpha=0.2, lw=0)
    b.plot(no, r["per_capita"], color="#2a78d6", lw=1.4, label=rf"chain, per capita ($\gamma$ = {c['gamma']:.2f})")
    b.plot(no, r["log_series"], color="#2a78d6", lw=1, ls=":", label=r"log-series ($\gamma$ = 0)")
    bo = r["bonabeau"]
    fitted = no ** -bo["alpha"] * np.exp(-no / bo["n_c"])
    b.plot(no, fitted / fitted.sum() * T.sum(), color="#8e5bd0", lw=1, ls="--",
           label=rf"$n^{{-\alpha}}e^{{-n/n_c}}$: $\alpha$ = {bo['alpha']:.1f}, $n_c$ = {bo['n_c']:.0f}")
    b.set(xscale="log", yscale="log", xlabel="pilots", ylabel="share of the time in configurations")
    b.set_xticks([2, 3, 5, 10, 20], labels=["2", "3", "5", "10", "20"])
    b.minorticks_off()
    b.legend(frameon=False, fontsize=7)
    save(fig, path)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    result = fission(pd.read_parquet(LABELS / "configurations.parquet"), pd.read_parquet(LABELS / "flights.parquet"),
                     pd.read_parquet(LABELS / "episodes.parquet"), np.random.default_rng(SEED))
    (OUT / "fission.json").write_text(json.dumps(result, indent=1))
    plot(result, OUT / "fission.pdf")
    print({k: result[k] for k in ("capita", "x", "bonabeau", "multi_share")}, flush=True)
    print("T", np.round(result["T"][:8], 4), "\nmeasured", np.round(result["measured"], 4),
          "\nper capita", np.round(result["per_capita"][:8], 4), "\nlog-series", np.round(result["log_series"][:8], 4))
