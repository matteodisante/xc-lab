"""The Kolmogorov-Smirnov threshold of a power-law tail, and the Hill exponent above it, for many weighted samples at
once.

Clauset, Shalizi and Newman (2009): the threshold xmin of a tail is the candidate that minimises the KS distance D
between the empirical survival of the durations from xmin up and the power law fitted there (Hill mu = n / sum
ln(x / xmin)). powerlaw.Fit does that on one sample, too slowly for a bootstrap. Here the durations are put on bins
(one per second below FINE_BELOW s, then a relative width BIN), the bootstrap weights of a draw are the counts of
the cells of the runs, so a draw is a histogram, and the scan over the candidates is one array operation over the
draws. The survival is read at the edges of the bins (D is the largest gap on either side of each edge), the
candidates are N_CANDIDATES edges on a log grid between two quantiles of the pooled durations, and a candidate
that leaves fewer than MIN_TAIL runs is not tried.

`thresholds` draws the common threshold of several populations (the largest of their own) afresh in every draw, and
returns the exponents above it.
"""

import numpy as np
from scipy.sparse import csr_matrix

FINE_BELOW = 200  # s: one bin per second up to here
BIN = 0.01  # relative width of the bins above
N_CANDIDATES = 80
QUANTILES = (50, 99.99)  # of the pooled durations: the range of the candidates
MIN_TAIL = 50  # runs above a candidate for its D to be taken
CHUNK = 50  # draws handled together


def edges_of(x):
    """The edges of the bins that hold the durations `x`, as an array of length bins + 1."""
    top = np.max(x)
    steps = int(np.ceil(np.log(max(top, FINE_BELOW) / FINE_BELOW) / np.log1p(BIN))) + 1
    return np.unique(np.r_[0, np.arange(1, FINE_BELOW), FINE_BELOW * (1 + BIN) ** np.arange(steps + 1)])


def candidates(x, edges):
    """The indices of the edges nearest to N_CANDIDATES log-spaced values between the QUANTILES of `x`."""
    lo, hi = np.percentile(x, QUANTILES)
    k = np.searchsorted(edges, np.geomspace(max(lo, 1), max(hi, 2), N_CANDIDATES))
    return np.unique(np.clip(k, 1, len(edges) - 2))


def histograms(x, cell, n_cells, edges):
    """Per cell, the runs and the sum of ln x of the durations `x` (cell code `cell`) in each bin: two sparse
    matrices (cells, bins)."""
    k = np.searchsorted(edges, x, "right") - 1
    shape = (n_cells, len(edges) - 1)
    return csr_matrix((np.ones(len(x)), (cell, k)), shape=shape), csr_matrix((np.log(x), (cell, k)), shape=shape)


def suffixes(weights, hists):
    """For the weights of the cells (rows, one per draw), the runs and the sum of ln x from each bin up: two arrays
    (draws, bins + 1), the last column 0."""
    out = []
    for m in hists:
        h = np.asarray((m.T @ weights.T.astype(float)).T)
        out.append(np.hstack([np.cumsum(h[:, ::-1], axis=1)[:, ::-1], np.zeros((len(h), 1))]))
    return out


def hill(n, logs, xmin):
    """Hill mu from the runs `n` and the sum of ln x `logs` above `xmin` (arrays over draws)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return n / (logs - n * np.log(xmin))


def distances(n, logs, edges, cand):
    """D at each candidate (index of an edge) for the suffix sums of suffixes(): array (draws, candidates), inf where
    fewer than MIN_TAIL runs lie above."""
    out = np.full((len(n), len(cand)), np.inf)
    with np.errstate(divide="ignore", invalid="ignore"):
        for k, j in enumerate(cand):
            nj = n[:, j]
            mu = hill(nj, logs[:, j], edges[j])[:, None]
            model = (edges[j:] / edges[j]) ** -mu  # the power law at every edge from the candidate up
            before = n[:, j:-1] / nj[:, None] - model[:, :-1]
            after = n[:, j + 1:] / nj[:, None] - model[:, 1:]
            d = np.maximum(np.abs(before).max(axis=1), np.abs(after).max(axis=1))
            out[:, k] = np.where(nj >= MIN_TAIL, d, np.inf)
    return out


def best(n, logs, edges, cand):
    """The index (into `cand`) of the candidate with the smallest D, per draw."""
    return np.argmin(distances(n, logs, edges, cand), axis=1)


def thresholds(samples, weights, n_cells):
    """Per draw, the exponents of the populations of `samples` above their common threshold.

    `samples`: {name: (x, cell)}, the durations of each population and the cell code of each; `weights`: the cell
    counts of the draws, (draws, cells), the first row of unit counts for the point estimate. Returns {"own":
    {name: xmin of the population alone (draws)}, "xmin": the largest of them (draws), "mu": {name: mu above it},
    "n": {name: runs above it}}, all arrays over the draws."""
    pooled = np.concatenate([x for x, _ in samples.values()])
    edges = edges_of(pooled)
    cand = candidates(pooled, edges)
    hists = {k: histograms(x, c, n_cells, edges) for k, (x, c) in samples.items()}
    out = {"own": {k: [] for k in samples}, "xmin": [], "mu": {k: [] for k in samples}, "n": {k: [] for k in samples}}
    for a in range(0, len(weights), CHUNK):
        sums = {k: suffixes(weights[a:a + CHUNK], h) for k, h in hists.items()}
        pick = {k: best(*s, edges, cand) for k, s in sums.items()}
        common = np.max([p for p in pick.values()], axis=0)
        j = cand[common]
        rows = np.arange(len(j))
        for k, (n, logs) in sums.items():
            out["own"][k].append(edges[cand[pick[k]]])
            out["mu"][k].append(hill(n[rows, j], logs[rows, j], edges[j]))
            out["n"][k].append(n[rows, j])
        out["xmin"].append(edges[j])
    return {"own": {k: np.concatenate(v) for k, v in out["own"].items()}, "xmin": np.concatenate(out["xmin"]),
            "mu": {k: np.concatenate(v) for k, v in out["mu"].items()}, "n": {k: np.concatenate(v) for k, v in out["n"].items()}}
