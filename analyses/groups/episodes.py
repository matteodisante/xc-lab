"""MSD of the largest episodes, each over the group time of its members, with H and H_loc.

Input: the pieces that tamsd.py measured (CACHE/pieces.npz), where a piece in a group carries
its episode, and the episodes table of ideas/group-identification. Largest means the most
distinct flights (n_flights). Conventions of tamsd.py: a member enters with its pieces in the
episode that last at least TAU_MAX / 0.8, with equal weight per member; the band resamples
the members of the episode.

    uv run python episodes.py [--run NAME]    # from this folder, after tamsd.py; writes out/<run>/
"""

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tamsd import CACHE, FIXES, LABELS, LAGS, NAMES, OUT, SEED, fit_h, local_h, per_flight, save

TOP, N_BOOT = 5, 1000
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#8e5bd0", "#d4a017"]


def episode_curves(measured, episodes, rng):
    """The TOP episodes with most flights, each with its members' MSD, H and H_loc; the
    members are pooled over disciplines only if one discipline has pieces in it."""
    out = []
    for e in episodes.sort_values(["n_flights", "t_end"], ascending=False).itertuples():
        for d, p in measured.items():
            mask = (p["population"] == "group") & (p["episode"] == e.episode_id)
            if not mask.any():
                continue
            flights, values = per_flight(p, mask)
            draws = values[rng.integers(0, len(flights), (N_BOOT, len(flights)))].mean(axis=1)
            mean = values.mean(axis=0)
            out.append({"episode": int(e.episode_id), "date": e.date, "discipline": d, "n_flights": int(e.n_flights),
                        "n_max": int(e.n_max), "duration_s": float(e.t_end - e.t_start), "members": len(flights),
                        "lags": LAGS.tolist(), "msd": mean.tolist(), "H": fit_h(LAGS, mean),
                        "H_band": np.percentile([fit_h(LAGS, x) for x in draws], [5, 95]).tolist(),
                        "H_loc": local_h(LAGS, mean).tolist()})
        if len({o["episode"] for o in out}) >= TOP:
            break
    return out


def plot(curves, reference, path):
    fig, axes = plt.subplots(2, 1, figsize=(6.5, 7), sharex=True)
    for d, style in (("para", "-"), ("hang", "--")):
        for pop, color in (("solo", "#555555"), ("group", "#000000")):
            r = reference.get(d, {}).get(pop)
            if r is not None and any(c["discipline"] == d for c in curves):
                axes[0].plot(r["lags"], r["msd"], color=color, ls=style, lw=1.5, label=f"all {pop}, {NAMES[d]}")
                axes[1].plot(r["lags"], r["H_loc"], color=color, ls=style, lw=1.5)
    for k, c in enumerate(curves):
        lo, hi = c["H_band"]
        label = (f"{c['date']}, {c['n_flights']} flights (max {c['n_max']}), {c['members']} in the MSD: "
                 f"H = {c['H']:.3f} [{lo:.3f}, {hi:.3f}]")
        axes[0].plot(c["lags"], c["msd"], color=COLORS[k % len(COLORS)], marker="o", ms=3, lw=1, label=label)
        axes[1].plot(c["lags"], c["H_loc"], color=COLORS[k % len(COLORS)], marker="o", ms=3, lw=1)
    axes[0].set(xscale="log", yscale="log", ylabel=r"MSD (m$^2$)")
    axes[0].legend(frameon=False, fontsize=6.5)
    for y in (0.5, 1):
        axes[1].axhline(y, color="#999999", lw=0.6, ls=":")
    axes[1].set(xscale="log", xlabel=r"$\tau$ (s)", ylabel=r"$H_{\rm loc}(\tau)$")
    save(fig, path)


if __name__ == "__main__":
    data = np.load(CACHE / "pieces.npz")
    measured = {d: {k.split("/")[1]: data[k] for k in data.files if k.startswith(d + "/")} for d in FIXES}
    episodes = pd.read_parquet(LABELS / "episodes.parquet")
    curves = episode_curves(measured, episodes, np.random.default_rng(SEED))
    reference = json.loads((OUT / "tamsd.json").read_text())
    (OUT / "episodes.json").write_text(json.dumps(curves, indent=1))
    plot(curves, reference, OUT / "episodes.pdf")
    for c in curves:
        print(f"episode {c['episode']} {c['date']} {c['discipline']}: {c['n_flights']} flights, "
              f"{c['members']} in the MSD, H = {c['H']:.3f} {np.round(c['H_band'], 3)}")
