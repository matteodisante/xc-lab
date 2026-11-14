"""The bands: a two-way bootstrap of pilots and site-days (as analyses/groups/twoway.py), joint over disciplines.

A flight is tied to the other flights of its pilot and to those of its block, the flights whose takeoffs lie
within SITE_M of each other on the same day (chained). Pilots and blocks are crossed, so three one-way bootstraps
of N_BOOT draws each resample with replacement the pilots, the blocks and the cells (pilot x block); in a draw
every flight weighs the count of its pilot, of its block or of its cell. The draws are shared by every
population, so the draws of a difference between disciplines are paired. With V_p, V_b, V_c the variances of an
estimate over the draws of each (Cameron, Gelbach and Miller 2011, J. Bus. Econ. Stat. 29, 238):

    V = max(V_p + V_b - V_c, V_p, V_b),    band = estimate -+ Z sqrt(V),  Z = 1.645 (a 90% band).

The draws are stacked in this order (pilots, blocks, cells), N_BOOT rows each.
"""

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

N_BOOT, CHUNK = 1000, 50
Z = 1.6448536269514722  # the 95% quantile of the standard normal
SITE_M, EARTH_M, DAY_APART_M = 3000.0, 6_371_000.0, 1e8  # days are laid far apart so that no site spans two


def sites(lat, lon, date):
    """Block of each flight: takeoffs within SITE_M of each other on the same day, chained."""
    lat, lon = np.radians(np.asarray(lat, float)), np.radians(np.asarray(lon, float))
    day = pd.factorize(np.asarray(date))[0]
    xy = np.column_stack([EARTH_M * lon * np.cos(lat) + DAY_APART_M * day, EARTH_M * lat])
    pairs = cKDTree(xy).query_pairs(SITE_M, output_type="ndarray").reshape(-1, 2)
    graph = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(xy), len(xy)))
    return connected_components(graph, directed=False)[1]


def cell_codes(pilot, block):
    """Cell code of each flight (a pilot x block pair), and the pilot and block code of each cell."""
    pairs, cell = np.unique(np.column_stack([pilot, block]), axis=0, return_inverse=True)
    return cell.ravel(), pairs[:, 0], pairs[:, 1]


def draw(n, rng, size=N_BOOT):
    """`size` draws of n items with replacement: how many times each is drawn, (size, n) uint8."""
    return np.vstack([rng.multinomial(n, np.full(n, 1 / n), size=min(CHUNK, size - a)).astype(np.uint8)
                      for a in range(0, size, CHUNK)])


def counts(pilot, block, rng, size=N_BOOT):
    """The stacked draws as counts per cell (`pilot`, `block`: the codes of each cell): (3 size, cells)."""
    return np.vstack([draw(pilot.max() + 1, rng, size)[:, pilot], draw(block.max() + 1, rng, size)[:, block],
                      draw(len(pilot), rng, size)])


def mean_draws(values, cell, stacked):
    """Equal-weight mean over flights of each column of `values` (flights x columns, NaN = the flight does not
    count in that column), and its value in every stacked draw (rows of `stacked`, counts per cell code)."""
    ok = np.isfinite(values)
    with np.errstate(invalid="ignore", divide="ignore"):
        point = np.where(ok, values, 0).sum(axis=0) / ok.sum(axis=0)
    cells, k = np.unique(cell, return_inverse=True)
    s, n = np.zeros((len(cells), values.shape[1])), np.zeros((len(cells), values.shape[1]))
    np.add.at(s, k, np.where(ok, values, 0))
    np.add.at(n, k, ok)
    out = []
    with np.errstate(invalid="ignore", divide="ignore"):
        for a in range(0, len(stacked), CHUNK):
            c = stacked[a:a + CHUNK][:, cells].astype(np.float32)
            out.append((c @ s) / (c @ n))
    return point, np.vstack(out)


def variance(draws):
    """V of the stacked draws (rows), per column; NaN draws left out."""
    p, b, c = (np.nanvar(x, axis=0) for x in np.split(np.asarray(draws, float), 3))
    return np.maximum(p + b - c, np.maximum(p, b))


def band(value, draws):
    """The band of `value` from its stacked draws: [value - Z sd, value + Z sd]."""
    sd = np.sqrt(variance(draws))
    return [(value - Z * sd).tolist(), (value + Z * sd).tolist()]
