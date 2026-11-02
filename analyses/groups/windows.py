"""The effect g of prepost.py on the open flights against the length of the window: H_loc(1000 s) and the
time in climb, with D_in and D_out beside g for H_loc(1000 s). Reads the JSON of every `prepost.py --open
--window W` that exists; nothing is recomputed.

    uv run python windows.py --open
"""

import json

import matplotlib.pyplot as plt

import prepost
import tamsd
from tamsd import OUT, save

PANELS = (("H_loc(1000 s)", ("D_in", "D_out", "g"), r"$\Delta H_{\rm loc}(1000$ s$)$"),
          ("time in climb (%)", ("g",), "$\\Delta$ time in climb (points)"))


def read():
    """{window: differences} of every prepost_<window>.json of the run (not the samples)."""
    out = {}
    for path in OUT.glob("prepost_*.json"):
        name = path.stem.removeprefix("prepost_")
        if name.isdigit():
            out[int(name)] = json.loads(path.read_text())["differences"]
    return dict(sorted(out.items()))


def draw(windows, path):
    fig, axes = plt.subplots(1, len(PANELS), figsize=(6.6, 3.0), constrained_layout=True)
    for ax, (stat, effects, label) in zip(axes, PANELS):
        for k, effect in enumerate(effects):
            xs = [w * (1 + 0.025 * (k - (len(effects) - 1) / 2)) for w in windows]
            q = [d[effect][stat] for d in windows.values()]
            lo, hi = zip(*[v["band"] for v in q])
            val = [v["value"] for v in q]
            ax.errorbar(xs, val, yerr=[[v - a for v, a in zip(val, lo)], [b - v for v, b in zip(val, hi)]], fmt="o-", ms=4,
                        lw=1, elinewidth=1.6, capsize=2, color=prepost.FOREST[effect], label=effect.replace("_", "$_{\\rm ") + ("}$" if "_" in effect else ""))
        ax.axhline(0, color="#999999", lw=0.8, ls=":")
        ax.set(xscale="log", xlabel="window (s)", ylabel=label, xticks=list(windows), xticklabels=[str(w) for w in windows])
        ax.minorticks_off()
        ax.grid(color="#eeeeee", lw=0.5)
        if len(effects) > 1:
            ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Before, during and after a group, by the length of the window (5-95% bands)", fontsize=8)
    save(fig, path)


if __name__ == "__main__":
    windows = read()
    for w, d in windows.items():
        g = d["g"]
        print(w, g["events (MSD)"], {s: (round(g[s]["value"], 3), [round(x, 3) for x in g[s]["band"]]) for s, *_ in PANELS})
    draw(windows, OUT / "prepost_windows.pdf")
