"""Crest edges to curves: flat bridges, pruning, strokes through junctions, splines resampled every STEP m.

Vertices are cell corners; the network is a graph with at most four edges per vertex.
"""

import numpy as np
from numba import njit
from scipy.interpolate import make_smoothing_spline
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from ridges import CELL, cells

STEP = 5.0  # m, resampling of the curves
BRIDGE_M = 1000.0  # flat stretches up to this length (in sides) are bridged
PRUNE_M = 100.0  # dead ends shorter than this are removed
MAX_DEFLECTION = 135.0  # deg, a stroke goes on if it turns by at most this
DIRECTION_M = 200.0  # distance from a junction at which the direction of a branch is measured
MIN_LENGTH = 1000.0  # m, shorter curves are dropped
SMOOTH_M = 100.0  # m, wavelength below which the smoothing spline removes the steps of the cell sides
EDGE_M = 100.0  # a curve reaches the edge of the data within this distance of a cell without data
BRIDGE_EDGES = cells(BRIDGE_M, CELL)  # 40 sides on the 25 m grid
PRUNE_EDGES = cells(PRUNE_M, CELL)  # 4 sides
EDGE_CELLS = cells(EDGE_M, CELL)  # 4 cells


def bridge(acc, flat, max_edges=BRIDGE_EDGES):
    """Accepted edges plus the flat-candidate stretches of at most `max_edges` sides that join the network
    at two or more vertices (spurs removed)."""
    if len(flat) == 0:
        return acc
    on_net = np.unique(acc)
    f = flat.astype(np.int64)
    vid, inv = np.unique(f, return_inverse=True)
    f = inv.reshape(-1, 2)
    keep = np.ones(len(f), bool)
    deg = np.bincount(f.ravel(), minlength=len(vid))
    touch = np.isin(vid, on_net, assume_unique=True)
    # remove spurs: edges at a vertex of degree one that is not on the network
    while True:
        spur = keep & (((deg[f[:, 0]] == 1) & ~touch[f[:, 0]]) | ((deg[f[:, 1]] == 1) & ~touch[f[:, 1]]))
        if not spur.any():
            break
        keep &= ~spur
        deg = np.bincount(f[keep].ravel(), minlength=len(vid))
    f = f[keep]
    if len(f) == 0:
        return acc
    n = len(vid)
    _, lab = connected_components(coo_matrix((np.ones(len(f)), (f[:, 0], f[:, 1])), shape=(n, n)), directed=False)
    size = np.bincount(lab[f[:, 0]], minlength=lab.max() + 1)
    ntouch = np.bincount(lab[touch], minlength=lab.max() + 1)
    ok = (size[lab[f[:, 0]]] <= max_edges) & (ntouch[lab[f[:, 0]]] >= 2)
    return np.r_[acc, vid[f[ok]].astype(acc.dtype)]


@njit(cache=True)
def _graph(a, b, n):
    nbr = np.full((n, 4), -1, np.int32)
    deg = np.zeros(n, np.int8)
    for e in range(a.size):
        nbr[a[e], deg[a[e]]] = b[e]
        deg[a[e]] += 1
        nbr[b[e], deg[b[e]]] = a[e]
        deg[b[e]] += 1
    return nbr, deg


@njit(cache=True)
def _cut(nbr, deg, u, v):
    for s in range(deg[u]):
        if nbr[u, s] == v:
            nbr[u, s] = nbr[u, deg[u] - 1]
            nbr[u, deg[u] - 1] = -1
            deg[u] -= 1
            return


@njit(cache=True)
def _prune(nbr, deg, max_edges):
    """Remove the branches from a dead end to a junction that have fewer than max_edges sides."""
    changed = True
    while changed:
        changed = False
        for v in range(deg.size):
            if deg[v] != 1:
                continue
            prev, cur, ln = v, nbr[v, 0], 1
            while deg[cur] == 2 and ln < max_edges:
                nx = nbr[cur, 0] if nbr[cur, 0] != prev else nbr[cur, 1]
                prev, cur, ln = cur, nx, ln + 1
            if deg[cur] >= 3 and ln < max_edges:
                # cut from the dead end along the chain
                prev, cur = v, nbr[v, 0]
                _cut(nbr, deg, v, cur)
                _cut(nbr, deg, cur, v)
                while deg[cur] == 1:
                    nx = nbr[cur, 0]
                    _cut(nbr, deg, cur, nx)
                    _cut(nbr, deg, nx, cur)
                    cur = nx
                changed = True


@njit(cache=True)
def _chains(nbr, deg):
    """Maximal paths whose inner vertices have two edges. Returns vertices, offsets, and the number of chains."""
    n = deg.size
    ne = 0
    for v in range(n):
        ne += deg[v]
    ne //= 2
    cv = np.empty(2 * ne + 8, np.int32)
    off = np.zeros(ne + 2, np.int64)
    used = np.zeros((n, 4), np.bool_)
    k, m = 0, 0
    for v in range(n):
        if deg[v] == 2 or deg[v] == 0:
            continue
        for s in range(deg[v]):
            if used[v, s]:
                continue
            used[v, s] = True
            cv[m] = v
            m += 1
            prev, cur = v, nbr[v, s]
            while True:
                cv[m] = cur
                m += 1
                if deg[cur] != 2:
                    for t in range(deg[cur]):
                        if nbr[cur, t] == prev:
                            used[cur, t] = True
                    break
                nx = nbr[cur, 0] if nbr[cur, 0] != prev else nbr[cur, 1]
                prev, cur = cur, nx
            k += 1
            off[k] = m
    return cv[:m], off[:k + 1]


@njit(cache=True)
def _loops(nbr, deg, seen):
    """Closed loops of degree-two vertices not on any chain."""
    cv = np.empty(2 * seen.size + 8, np.int32)
    off = np.zeros(seen.size // 3 + 2, np.int64)
    m = k = 0
    for v in range(deg.size):
        if deg[v] != 2 or seen[v]:
            continue
        prev, cur = nbr[v, 1], v
        start = v
        while True:
            seen[cur] = True
            cv[m] = cur
            m += 1
            nx = nbr[cur, 0] if nbr[cur, 0] != prev else nbr[cur, 1]
            prev, cur = cur, nx
            if cur == start:
                break
        cv[m] = start
        m += 1
        k += 1
        off[k] = m
    return cv[:m], off[:k + 1]


@njit(cache=True)
def _end_dirs(cv, off, xy, dist):
    """Node vertex and unit direction (node outwards, measured `dist` m along the chain) of every chain end."""
    nc = off.size - 1
    dirs = np.zeros((2 * nc, 2))
    node = np.empty(2 * nc, np.int64)
    for c in range(nc):
        lo, hi = off[c], off[c + 1]
        for side in range(2):
            i0 = lo if side == 0 else hi - 1
            step = 1 if side == 0 else -1
            node[2 * c + side] = cv[i0]
            j, acc = i0, 0.0
            while lo <= j + step < hi and acc < dist:
                p, q = cv[j], cv[j + step]
                acc += np.hypot(xy[p, 0] - xy[q, 0], xy[p, 1] - xy[q, 1])
                j += step
            d = xy[cv[j]] - xy[cv[i0]]
            nrm = np.hypot(d[0], d[1])
            dirs[2 * c + side] = d / nrm if nrm > 0 else d
    return node, dirs


@njit(cache=True)
def _pair(cv, off, xy, deg, dist, cos_min):
    """Pair the chain ends meeting at each junction by the straightest continuation. Returns, for the end
    (chain c, 0 = first vertex, 1 = last), the partner end 2 * chain + side, or -1."""
    node, dirs = _end_dirs(cv, off, xy, dist)
    partner = np.full(node.size, -1, np.int64)
    idx = np.argsort(node)
    i, n = 0, idx.size
    while i < n:
        j = i
        while j < n and node[idx[j]] == node[idx[i]]:
            j += 1
        g = idx[i:j]
        if deg[node[idx[i]]] >= 3:
            alive = np.ones(g.size, np.bool_)
            while alive.sum() >= 2:
                best, bx, by = 2.0, -1, -1
                for x in range(g.size):
                    for y in range(x + 1, g.size):
                        if alive[x] and alive[y]:
                            cs = dirs[g[x], 0] * dirs[g[y], 0] + dirs[g[x], 1] * dirs[g[y], 1]
                            if cs < best:
                                best, bx, by = cs, x, y
                if best > cos_min:  # the straightest pair still turns by more than the limit
                    break
                partner[g[bx]], partner[g[by]] = g[by], g[bx]
                alive[bx] = alive[by] = False
        i = j
    return partner


@njit(cache=True)
def _end_dirs(cv, off, xy, dist):
    """Node vertex and unit direction (node outwards, measured `dist` m along the chain) of every chain end."""
    nc = off.size - 1
    dirs = np.zeros((2 * nc, 2))
    node = np.empty(2 * nc, np.int64)
    for c in range(nc):
        lo, hi = off[c], off[c + 1]
        for side in range(2):
            i0 = lo if side == 0 else hi - 1
            step = 1 if side == 0 else -1
            node[2 * c + side] = cv[i0]
            j, acc = i0, 0.0
            while lo <= j + step < hi and acc < dist:
                p, q = cv[j], cv[j + step]
                acc += np.hypot(xy[p, 0] - xy[q, 0], xy[p, 1] - xy[q, 1])
                j += step
            d = xy[cv[j]] - xy[cv[i0]]
            nrm = np.hypot(d[0], d[1])
            dirs[2 * c + side] = d / nrm if nrm > 0 else d
    return node, dirs


@njit(cache=True)
def _strokes(cv, off, partner):
    """Vertex ids of the strokes (chains joined by `partner`) and their offsets."""
    nc = off.size - 1
    out = np.empty(cv.size + nc + 8, np.int32)
    so = np.zeros(nc + 1, np.int64)
    done = np.zeros(nc, np.bool_)
    m = k = 0
    for phase in range(2):
        for c0 in range(nc):
            if done[c0]:
                continue
            side = 0
            if phase == 0:  # open strokes start at an unpaired end
                if partner[2 * c0] == -1:
                    side = 0
                elif partner[2 * c0 + 1] == -1:
                    side = 1
                else:
                    continue
            c = c0
            first = True
            while True:
                done[c] = True
                lo, hi = off[c], off[c + 1]
                if side == 0:
                    for i in range(lo if first else lo + 1, hi):
                        out[m] = cv[i]
                        m += 1
                else:
                    for i in range(hi - 1 if first else hi - 2, lo - 1, -1):
                        out[m] = cv[i]
                        m += 1
                first = False
                nxt = partner[2 * c + (1 - side)]
                if nxt == -1 or done[nxt // 2]:
                    break
                c, side = nxt // 2, nxt % 2
            k += 1
            so[k] = m
    return out[:m], so[:k + 1]


def strokes(a, b, xy, prune_edges=PRUNE_EDGES, max_deflection=MAX_DEFLECTION, direction_m=DIRECTION_M):
    """Strokes of the network with edges (a, b) between vertices (rows of xy, metres).
    Returns the vertex ids of the strokes and their offsets."""
    nbr, deg = _graph(a.astype(np.int32), b.astype(np.int32), len(xy))
    _prune(nbr, deg, prune_edges)
    cv, off = _chains(nbr, deg)
    seen = np.zeros(len(xy), bool)
    seen[cv] = True
    lv, lo = _loops(nbr, deg, seen)
    if len(lo) > 1:
        cv, off = np.r_[cv, lv], np.r_[off, lo[1:] + off[-1]]
    partner = _pair(cv, off, xy, deg, direction_m, np.cos(np.radians(180.0 - max_deflection)))
    return _strokes(cv, off, partner)


def smooth_resample(p, step=STEP, smooth=SMOOTH_M):
    """Smoothing spline (penalty on the curvature, cut-off wavelength `smooth`) through the polyline p (n, 2),
    resampled every `step` m of arc length."""
    d = np.hypot(*np.diff(p, axis=0).T)
    t = np.r_[0.0, np.cumsum(d)]
    keep = np.r_[True, d > 0]
    p, t = p[keep], t[keep]
    if len(p) < 4:
        q = p
    else:
        lam = (smooth / (2 * np.pi)) ** 4
        u = np.arange(0.0, t[-1], step)
        u = np.r_[u, t[-1]]
        q = np.c_[make_smoothing_spline(t, p[:, 0], lam=lam)(u), make_smoothing_spline(t, p[:, 1], lam=lam)(u)]
    s = np.r_[0.0, np.cumsum(np.hypot(*np.diff(q, axis=0).T))]
    pts = np.arange(0.0, s[-1] + 1e-9, step)
    return np.c_[np.interp(pts, s, q[:, 0]), np.interp(pts, s, q[:, 1])].astype(np.float64)


@njit(cache=True)
def near_nodata(z, rows, cols, cells):
    """True for corners (rows, cols) within `cells` cells of a cell without data or of the array border."""
    h, w = z.shape
    out = np.zeros(rows.size, np.bool_)
    for k in range(rows.size):
        r, c = rows[k], cols[k]
        for rr in range(r - cells - 1, r + cells + 1):
            for cc in range(c - cells - 1, c + cells + 1):
                if rr < 0 or cc < 0 or rr >= h or cc >= w or np.isnan(z[rr, cc]):
                    out[k] = True
    return out
