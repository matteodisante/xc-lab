"""Declared open flights fold back on themselves: why the comparisons keep the straight open ones.

Left: the opening of the declared open flights (closure.py: takeoff to landing over the path flown), all
of them and those of at least LONG_H hours, the ones that the longest lags rest on, against the threshold
openflights.MIN_OPENING. Right: H_loc(tau) of S, G and A on the declared open flights (longlags.py run with
--declared-open), with an inset of MSD.

    uv run python folding.py [--run NAME]    # after closure.py and longlags.py --declared-open; writes out/<run>/open_declared/folding.*
"""

import json

import matplotlib.pyplot as plt
import numpy as np

import closure
import curves
from openflights import MIN_OPENING
from tamsd import COLORS, OUT

LONG_H = 4  # hours: flights long enough for lags of hours
DECLARED = OUT / "open_declared"

if __name__ == "__main__":
    f = closure.flight_table()
    f = f[f["declared"] == "open"]
    hours = (f["t1"] - f["t0"]) / 3600
    groups = {"all": f["ratio"], f"{LONG_H} h or more": f.loc[hours >= LONG_H, "ratio"]}
    longlags = json.loads((DECLARED / "longlags.json").read_text())
    main = json.loads((DECLARED / "tamsd.json").read_text())["para"]
    pooled = longlags["panels"]["pooled"]
    result = {"min_opening": MIN_OPENING, "long_h": LONG_H,
              "flights": {k: int(len(v)) for k, v in groups.items()},
              "median opening": {k: round(float(v.median()), 2) for k, v in groups.items()},
              "share at least the minimum (%)": {k: round(100 * float((v >= MIN_OPENING).mean()), 1) for k, v in groups.items()},
              "straight open flights": int((f["ratio"] >= MIN_OPENING).sum()),
              "median opening, group flights": round(float(f.loc[f["group"], "ratio"].median()), 2),
              "median opening, never in a group": round(float(f.loc[~f["group"], "ratio"].median()), 2),
              "tau_max": pooled["tau_max"],
              "H_loc at tau_max": {k: round(c["H_loc"][-1], 3) for k, c in pooled["curves"].items()},
              "flights of the long series": {k: c["flights"] for k, c in pooled["curves"].items()}}
    (DECLARED / "folding.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))
    fig, (left, right) = plt.subplots(1, 2, figsize=(7.4, 3.0), constrained_layout=True)
    bins = np.linspace(0, 1, 41)
    for (k, v), color in zip(groups.items(), ("#555555", "#c24a1d")):
        left.hist(v.clip(0, 1), bins=bins, density=True, histtype="step", lw=1.3, color=color,
                  label=f"{k} ({len(v):,}), median {v.median():.2f}")
    left.axvline(MIN_OPENING, color="#1a7f5a", lw=1, ls="--")
    left.set(xlabel="takeoff to landing / path flown", ylabel="density", title="declared open flights")
    left.title.set_fontsize(8.5)
    left.legend(frameon=False, fontsize=6.5)
    label = {"solo": "S", "group": "G", "group_solo": "A"}
    drawn = []
    for k in ("solo", "group", "group_solo"):
        drawn.append(curves.series(main[k], f"{label[k]} ({main[k]['flights']:,})", COLORS[k]))
        drawn.append(curves.series(pooled["curves"][k], None, COLORS[k], "--"))
    curves.plot(right, drawn, ylim=(0.2, 1.0))
    right.legend(frameon=False, fontsize=6.5, loc="upper right")
    for ax in (left, right):
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.savefig(DECLARED / "folding.pdf", bbox_inches="tight")
    fig.savefig(DECLARED / "folding.png", dpi=200, bbox_inches="tight")
