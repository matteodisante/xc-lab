"""Glide ratio of experienced and beginner paragliders, in a group and with no contact.

Total straight-line distance over total height lost on the glides, the runs of
group-identification/performance.py kept for the flights of population.py at each level; bands as
in budget.py (5-95% of the bootstrap of the dependency blocks).

    uv run python glide.py [--run NAME]    # from this folder; writes out/<run>/glide.*
"""

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from population import CACHE, COLORS, LEVELS, OUT, POPULATIONS, SEED, keep, load
from budget import budget  # noqa: E402  (population puts group-identification on the path)

STATS = {"glide ratio": ("chord_m", "loss_m", 1, 1)}
FACE = {"experienced": None, "beginners": "white"}  # marker face


def plot(results, path):
    fig, ax = plt.subplots(figsize=(5, 3))
    rows = [(level, pop) for level in results for pop in POPULATIONS]
    for i, (level, pop) in enumerate(rows):
        r = results[level][pop]["glide ratio"]
        ax.errorbar(r["value"], i, xerr=[[r["value"] - r["band"][0]], [r["band"][1] - r["value"]]], color=COLORS[pop],
                    marker="o", mfc=FACE[level] or COLORS[pop], capsize=3)
    ax.set_yticks(range(len(rows)), [f"{level}, {POPULATIONS[pop]} ({results[level][pop]['flights']})"
                                     for level, pop in rows], fontsize=8)
    ax.set(xlabel="glide ratio", ylim=(len(rows) - 0.5, -0.5))
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"), dpi=200)


if __name__ == "__main__":
    summary, classes, alone_flights = load()
    terms = pd.read_parquet(CACHE / "performance_terms.parquet")
    block = summary.set_index("flight")["block"]
    results = {}
    for level in LEVELS:
        mask = keep(terms["flight"].to_numpy(), terms["population"].to_numpy(), classes, alone_flights, level)
        results[level] = budget(terms[mask], block, np.random.default_rng(SEED), STATS, POPULATIONS)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "glide.json").write_text(json.dumps(results, indent=1))
    plot(results, OUT / "glide.pdf")
    for level, populations in results.items():
        for pop, r in populations.items():
            print(level, pop, r["flights"], f"glide ratio = {r['glide ratio']['value']:.2f}",
                  np.round(r["glide ratio"]["band"], 2).tolist())
