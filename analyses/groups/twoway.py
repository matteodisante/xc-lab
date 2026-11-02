"""The bands of the comparisons (raw.py, matched.py, decomposition.py, prepost.py --open): a two-way
bootstrap of pilots and blocks.

A flight is tied to the other flights of its pilot and to those of its block (decompose.flight_table:
the flights joined by an episode of the grouping or by a site-day). Pilots and blocks are crossed, so
three one-way bootstraps of N_BOOT draws each resample with replacement the pilots, the blocks and the
cells (pilot x block); in a draw every unit of the data, group or solo, weighs the count of its pilot,
of its block or of its cell. With V_p, V_b, V_c the variances of an estimate over the draws of each
(Cameron, Gelbach and Miller 2011, J. Bus. Econ. Stat. 29, 238):

    V = max(V_p + V_b - V_c, V_p, V_b),    band = estimate -+ Z sqrt(V),  Z = 1.645 (a 90% band).

The draws are stacked in this order (pilots, blocks, cells), N_BOOT rows each; callers that keep the
point estimate put it in row 0, before them. coverage.py checks the band on the designs of the points.
"""

import numpy as np
import pandas as pd

import isolated
from tamsd import N_BOOT

Z = 1.6448536269514722  # the 95% quantile of the standard normal
CHUNK = 50  # draws made together


def cell_codes(pilot, block):
    """The cell code of units with pilot and block codes, and the pilot and block code of each cell."""
    pairs, cell = np.unique(np.column_stack([pilot, block]), axis=0, return_inverse=True)
    return cell.ravel(), pairs[:, 0], pairs[:, 1]


def cells(flights, summary, universe=None):
    """The cell (pilot x block) of every flight of `universe` (flight numbers, default those of `flights`,
    which carries lat0 and lon0): a Series of codes by flight, and the pilot and block code of each cell."""
    universe = flights["flight"].to_numpy() if universe is None else np.asarray(universe)
    pilot, block, _ = isolated.codes(flights, summary, universe)
    cell, pilot, block = cell_codes(pilot, block)
    return pd.Series(cell, index=universe), pilot, block


def draw(n, rng, size=N_BOOT):
    """`size` draws of n items with replacement: how many times each is drawn, (size, n) uint8."""
    return np.vstack([rng.multinomial(n, np.full(n, 1 / n), size=min(CHUNK, size - a)).astype(np.uint8)
                      for a in range(0, size, CHUNK)])


def counts(pilot, block, rng, size=N_BOOT):
    """The stacked draws as counts per cell (`pilot`, `block`: the codes of each cell): (3 size, cells)."""
    return np.vstack([draw(pilot.max() + 1, rng, size)[:, pilot], draw(block.max() + 1, rng, size)[:, block],
                      draw(len(pilot), rng, size)])


def weights(cell, stacked):
    """The weights of isolated.engine: for units with cell codes `cell` (rows) and columns (0 the point
    estimate, all 1, then the rows of `stacked`), and the number of columns."""
    def of(rows, cols):
        w = np.ones((len(rows), len(cols)))
        d = cols[cols > 0] - 1
        w[:, cols > 0] = stacked[d][:, cell[rows]].T
        return w

    return of, len(stacked) + 1


def variance(draws):
    """V of the stacked draws (rows), per column; NaN draws left out."""
    p, b, c = (np.nanvar(x, axis=0) for x in np.split(np.asarray(draws, float), 3))
    return np.maximum(p + b - c, np.maximum(p, b))


def band(value, draws):
    """The band of `value` from its stacked draws: [value - Z sd, value + Z sd]."""
    sd = np.sqrt(variance(draws))
    return [(value - Z * sd).tolist(), (value + Z * sd).tolist()]


def share_band(value, draws):
    """band() cut to [0, 1], for probabilities."""
    return np.clip(band(value, draws), 0, 1).tolist()


def quote(boot, years=None):
    """As isolated.quote: the point estimate (row 0 of `boot`), its band from the other rows and, when
    given, the minimum and maximum over the rows of `years`."""
    boot = np.asarray(boot, float)
    out = {"value": boot[0].tolist(), "band": band(boot[0], boot[1:])}
    if years is not None:
        out["years"] = [np.nanmin(years, axis=0).tolist(), np.nanmax(years, axis=0).tolist()]
    return out
