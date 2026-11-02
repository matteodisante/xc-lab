"""The effect g of prepost.py on the open flights against the length of the window: left, g(tau) for each
window; right, D_in, D_out and g of H_loc(1000 s) against the window. Reads the JSON of every `prepost.py
--open --window W` that exists; nothing is recomputed.

    uv run python windows.py --open
"""

import json

import matplotlib.pyplot as plt

from tamsd import OUT, save

COLOR = {"D_in": "#d98200", "D_out": "#7a3b9e", "g": "#111111"}  # distinct, unlike the shades of prepost.FOREST
SHADES = ("#9ecae1", "#4292c6", "#2171b5", "#08306b")  # the windows, from the shortest to the longest
STAT = "H_loc(1000 s)"


def read():
    """{window: (differences, lags)} of every prepost_<window>.json of the run (not the samples)."""
    out = {}
    for path in OUT.glob("prepost_*.json"):
        name = path.stem.removeprefix("prepost_")
        if name.isdigit():
            p = json.loads(path.read_text())
            out[int(name)] = (p["differences"], p["lags"])
    return dict(sorted(out.items()))


def draw(windows, path):
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.0), constrained_layout=True)
    for (w, (d, lags)), color in zip(windows.items(), SHADES):
        c = d["g"]["dH_loc"]
        axes[0].fill_between(lags, *c["band"], color=color, alpha=0.15, lw=0)
        axes[0].plot(lags, c["value"], color=color, lw=1.4, label=f"{w} s")
    axes[0].axhline(0, color="#999999", lw=0.8, ls=":")
    axes[0].set(xscale="log", xlabel=r"$\tau$ (s)", ylabel=r"$g$ of $H_{\rm loc}(\tau)$")
    axes[0].legend(frameon=False, fontsize=7, title="window", title_fontsize=7)
    for k, effect in enumerate(("D_in", "D_out", "g")):
        xs = [w * (1 + 0.025 * (k - 1)) for w in windows]
        q = [d[effect][STAT] for d, _ in windows.values()]
        lo, hi = zip(*[v["band"] for v in q])
        val = [v["value"] for v in q]
        axes[1].errorbar(xs, val, yerr=[[v - a for v, a in zip(val, lo)], [b - v for v, b in zip(val, hi)]], fmt="o-", ms=4,
                         lw=1, elinewidth=1.6, capsize=2, color=COLOR[effect],
                         label=effect.replace("_", "$_{\\rm ") + ("}$" if "_" in effect else ""))
    axes[1].axhline(0, color="#999999", lw=0.8, ls=":")
    axes[1].set(xscale="log", xlabel="window (s)", ylabel=r"$\Delta H_{\rm loc}(1000$ s$)$", xticks=list(windows),
                xticklabels=[str(w) for w in windows])
    axes[1].minorticks_off()
    axes[1].legend(frameon=False, fontsize=7)
    for ax in axes:
        ax.grid(color="#eeeeee", lw=0.5)
    fig.suptitle("Before, during and after a group, by the length of the window (5-95% bands)", fontsize=8)
    save(fig, path)


if __name__ == "__main__":
    windows = read()
    for w, (d, lags) in windows.items():
        print(w, d["g"]["events (MSD)"], round(d["g"][STAT]["value"], 3))
    draw(windows, OUT / "prepost_windows.pdf")
