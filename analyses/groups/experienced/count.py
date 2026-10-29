"""Experienced and beginner paragliders among group flights and flights with no contact.

Populations and levels as in population.py, read from the pieces of the run.

    uv run python count.py [--run NAME]    # from this folder; writes out/<run>/
"""

import json

import matplotlib.pyplot as plt
import numpy as np

from population import CACHE, COLORS, LEVELS, OUT, POPULATIONS, load, read_pieces


def count(p, classes, alone_flights):
    """Per population: its flights, those of each level, and those of each class of the levels."""
    out = {}
    for key in POPULATIONS:
        flights = np.unique(p["flight"][p["population"] == key])
        if key == "solo":
            flights = np.intersect1d(flights, alone_flights)
        wing = classes.reindex(flights)
        out[key] = {"flights": len(flights)}
        for level, names in LEVELS.items():
            label = wing.map(names)
            out[key][level] = int(label.notna().sum())
            out[key].update({c: int((label == c).sum()) for c in names.values()})
    return out


def draw(counts):
    """Bars by level, one per population."""
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    for j, key in enumerate(POPULATIONS):
        for i, level in enumerate(LEVELS):
            c = counts[key]
            x = i + (j - 0.5) * 0.38
            ax.bar(x, c[level], color=COLORS[key], width=0.34, label=POPULATIONS[key] if i == 0 else None)
            ax.text(x, c[level], f"{c[level]:,}\n{c[level] / c['flights']:.0%}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(range(len(LEVELS)), [f"{level} ({'/'.join(names.values())})" for level, names in LEVELS.items()])
    ax.set_ylabel("flights; share of the population")
    ax.set_ylim(0, 1.2 * max(c[level] for c in counts.values() for level in LEVELS))
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    summary, classes, alone_flights = load()
    counts = count(read_pieces(CACHE / "pieces.npz")["para"], classes, alone_flights)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "count.json").write_text(json.dumps(counts, indent=2))
    fig = draw(counts)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"count.{ext}", dpi=200)
    print(json.dumps(counts, indent=2))
