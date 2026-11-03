"""The fitted H of S, A and G to several end lags, each on the smallest population that keeps the flights constant
along its lags: the pieces long enough for the end lag (as raw.py), so a longer fit has fewer flights. The bands are
those of raw.py (twoway.py), and the gaps of raw.GAPS share the draws.

    uv run python fitted_h.py --open    # after longlags.py; writes OUT/fitted_h_by_end.json
"""

import json

import numpy as np
import pandas as pd

import longlags
import raw
import twoway
from blocks import with_takeoffs
from tamsd import LABELS, OUT, SEED, ensemble, fit_h, read_flights

ENDS = (1000, 2760)  # s: end lags of the fit besides the longest lag of raw.py


def fitted(p, mask, block, rng, tau, counts):
    """Flights, and the fitted H of the point estimate (row 0) and of every draw, to the lag `tau`, on the pieces of
    `mask` long enough for `tau`."""
    keep = longlags.long_enough(p, mask, tau)
    lags = longlags.LONG_LAGS[longlags.LONG_LAGS <= tau]
    flights, k = np.unique(p["flight"][keep], return_inverse=True)
    sums, origins = np.zeros((len(flights), len(lags))), np.zeros((len(flights), len(lags)))
    np.add.at(sums, k, p["sums"][keep][:, :len(lags)])
    np.add.at(origins, k, p["origins"][keep][:, :len(lags)])
    mean, draws = ensemble(sums / origins, block.reindex(flights).to_numpy(), rng, counts)
    return len(flights), np.array([fit_h(lags, r) for r in np.vstack([mean, draws])])


if __name__ == "__main__":
    p = dict(np.load(longlags.PIECES))
    block, of_pilot, of_block = twoway.cells(with_takeoffs(read_flights()), pd.read_parquet(LABELS / "flight_summary.parquet"))
    rng = np.random.default_rng(SEED)
    counts = twoway.counts(of_pilot, of_block, rng)
    masks = longlags.populations(p)
    avail = {k: longlags.available(p, masks[k]) for k in longlags.POOLED}
    ends = [*ENDS, int(longlags.panels(avail)["longest"])]
    got = {e: {pop: fitted(p, masks[pop], block, rng, e, counts) for pop in raw.POPS} for e in ends}
    result = {"ends": ends,
              "flights": {str(e): {short: got[e][pop][0] for pop, short in raw.POPS.items()} for e in ends},
              "H": {short: {str(e): twoway.quote(got[e][pop][1]) for e in ends} for pop, short in raw.POPS.items()},
              "gaps": {gap: {str(e): twoway.quote(got[e][a][1] - got[e][b][1]) for e in ends} for gap, (a, b) in raw.GAPS.items()}}
    (OUT / "fitted_h_by_end.json").write_text(json.dumps(result, indent=1))
    print(json.dumps({"flights": result["flights"], "G - S": {e: [round(q["value"], 4), [round(x, 4) for x in q["band"]]] for e, q in result["gaps"]["G - S"].items()}}))
