"""Gap joins, trees and paths of the crest curves.

From every dead end of a curve a straight join of at most JOIN_M to another piece of the crest network is
looked for within CONE_DEG of the direction in which the curve runs out, never lower than HAND_MIN above the
streams in any DEM cell it crosses, snapped to a dead end of the other curve within SNAP_M, up to N_COMPONENTS other components tried per
dead end. Joins are made shortest first, a dead end in at most one, never inside a component: the curves
grow into trees. A tree's root is one end of its longest path; its paths run from the root to every other end.
"""

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial import cKDTree

STEP = 5.0
JOIN_M = 1000.0
CONE_DEG = 45.0
SNAP_M = 200.0
N_COMPONENTS = 3
RUNOUT_M = 200.0  # the direction a curve runs out in is measured over this length
HAND_MIN = 20.0  # m above the streams along a join
ATTACH_M = 50.0  # a junction this close to the end of a curve makes the end attached
CELL = 25.0


class Net:
    """Nodes (curve, sample index) and weighted edges: curve pieces, junction connectors, joins."""

    def __init__(self, xy, off, jc, jxy):
        self.xy, self.off = xy, off
        self.nc = len(off) - 1
        self.nodes = {}  # (curve, index) -> node id
        self.key = []
        self.edges = []  # (u, v, kind, curve, i0, i1, length); kind 0 piece, 1 connector, 2 join
        n = lambda c: off[c + 1] - off[c]
        near = int(ATTACH_M / STEP)
        for c in range(self.nc):
            self.node(c, 0)
            self.node(c, n(c) - 1)
        for (a, b), p in zip(jc, jxy):
            ia, ib = self.nearest(a, p), self.nearest(b, p)
            ia = self.snap_end(a, ia, near)
            ib = self.snap_end(b, ib, near)
            ua, ub = self.node(a, ia), self.node(b, ib)
            self.edges.append((ua, ub, 1, -1, 0, 0, float(np.hypot(*(self.pt(a, ia) - self.pt(b, ib))))))
        self.split_curves()

    def pt(self, c, i):
        return self.xy[self.off[c] + i]

    def nearest(self, c, p):
        a = self.xy[self.off[c]:self.off[c + 1]]
        return int(np.argmin(((a - p) ** 2).sum(axis=1)))

    def snap_end(self, c, i, near):
        n = self.off[c + 1] - self.off[c]
        return 0 if i <= near else (n - 1 if i >= n - 1 - near else i)

    def node(self, c, i):
        k = (int(c), int(i))
        if k not in self.nodes:
            self.nodes[k] = len(self.key)
            self.key.append(k)
        return self.nodes[k]

    def split_curves(self):
        """Piece edges between consecutive nodes of every curve (called again after joins add nodes)."""
        self.edges = [e for e in self.edges if e[2] != 0]
        by = {}
        for (c, i), u in self.nodes.items():
            by.setdefault(c, []).append((i, u))
        for c, lst in by.items():
            lst.sort()
            for (i0, u), (i1, v) in zip(lst[:-1], lst[1:]):
                self.edges.append((u, v, 0, c, i0, i1, (i1 - i0) * STEP))

    def arrays(self):
        e = self.edges
        u, v = np.array([x[0] for x in e], np.int64), np.array([x[1] for x in e], np.int64)
        w = np.array([x[6] for x in e])
        return u, v, w

    def labels(self):
        u, v, w = self.arrays()
        n = len(self.key)
        return connected_components(coo_matrix((np.ones(len(u)), (u, v)), shape=(n, n)), directed=False)[1]

    def degree(self):
        u, v, _ = self.arrays()
        return np.bincount(np.r_[u, v], minlength=len(self.key))


def resample_line(a, b, step=STEP):
    n = max(int(np.hypot(*(b - a)) // step), 1)
    t = np.linspace(0, 1, n + 1)[:, None]
    return a + t * (b - a)


def cells_crossed(a, b, x0, y0):
    """(rows, cols) of every DEM cell the segment a-b touches, corners included (cell (r, c) is the square of
    side CELL centred at (x0 + c CELL, y0 - r CELL))."""
    pa = np.array([(a[0] - x0) / CELL, (y0 - a[1]) / CELL])  # (col, row) in cell units
    pb = np.array([(b[0] - x0) / CELL, (y0 - b[1]) / CELL])
    d = pb - pa
    t = [0.0, 1.0]
    for k in range(2):
        if d[k] != 0:
            edges = np.arange(np.floor(min(pa[k], pb[k]) + 0.5) - 0.5, max(pa[k], pb[k]) + 1.0, 1.0)
            t += [x for x in (edges - pa[k]) / d[k] if 0 < x < 1]
    t = np.unique(np.round(np.array(t), 12))
    mid = (t[:-1] + t[1:]) / 2
    pts = pa + mid[:, None] * d
    cells = [np.rint(pts)]
    # a crossing through a corner also touches the cells diagonal to it
    for x in t[1:-1]:
        q = pa + x * d
        f = np.abs(q - np.floor(q) - 0.5)
        if (f < 1e-6).all():
            cells += [np.rint(q + [sx * 0.5, sy * 0.5]) for sx in (-1, 1) for sy in (-1, 1)]
    cells = np.unique(np.vstack(cells), axis=0).astype(int)
    return cells[:, 1], cells[:, 0]


def hand_ok(hand, x0, y0, a, b, hand_min=HAND_MIN):
    """True if every DEM cell the straight segment a-b crosses is at least hand_min above the streams
    (hand: int16 raster; stream cells are 0, cells draining to no valley are negative)."""
    r, c = cells_crossed(a, b, x0, y0)
    if (r < 0).any() or (c < 0).any() or (r >= hand.shape[0]).any() or (c >= hand.shape[1]).any():
        return False
    return bool((hand[r, c] >= hand_min).all())


def runout(net, c, end, length=RUNOUT_M):
    n = net.off[c + 1] - net.off[c]
    k = int(length / STEP)
    a, b = (net.pt(c, 0), net.pt(c, min(k, n - 1))) if end == 0 else (net.pt(c, n - 1), net.pt(c, max(n - 1 - k, 0)))
    d = a - b
    return d / max(np.hypot(*d), 1e-9)


def find_joins(net, hand, x0, y0, join_m=JOIN_M, cone_deg=CONE_DEG, snap_m=SNAP_M, n_comp=N_COMPONENTS):
    """Candidate joins (length, dead end node, target node or (curve, index), target is a dead end)."""
    lab = net.labels()
    cl = lab[[net.node(c, 0) for c in range(net.nc)]]
    deg = net.degree()
    ends = [u for u, (c, i) in enumerate(net.key) if deg[u] == 1 and i in (0, net.off[c + 1] - net.off[c] - 1)]
    end_set = set(ends)
    sub = np.arange(0, len(net.xy), 5)
    pts, owner = net.xy[sub], np.searchsorted(net.off, sub, side="right") - 1
    tree = cKDTree(pts)
    etree = cKDTree(np.array([net.pt(*net.key[u]) for u in ends])) if ends else None
    ends_arr = np.array(ends)
    cos_min = np.cos(np.radians(cone_deg))
    out = []
    for u in ends:
        c, i = net.key[u]
        end = 0 if i == 0 else 1
        p, d = net.pt(c, i), runout(net, c, end)
        idx = np.array(tree.query_ball_point(p, join_m), int)
        if len(idx) == 0:
            continue
        v = pts[idx] - p
        dist = np.hypot(v[:, 0], v[:, 1])
        ok = (dist > 0) & ((v @ d) >= cos_min * dist)
        comp = cl[owner[idx]]
        ok &= comp != lab[u]
        idx, dist, comp = idx[ok], dist[ok], comp[ok]
        taken = []
        for j in np.argsort(dist):
            if comp[j] in taken:
                continue
            q = pts[idx[j]]
            cj = int(owner[idx[j]])
            # snap to a dead end of the same component within snap_m
            target, is_end = None, False
            for cand in (etree.query_ball_point(q, snap_m) if etree is not None else []):
                w = int(ends_arr[cand])
                if lab[w] == comp[j] and w != u:
                    pw = net.pt(*net.key[w])
                    vv = pw - p
                    dd = np.hypot(*vv)
                    if 0 < dd <= join_m and vv @ d >= cos_min * dd and hand_ok(hand, x0, y0, p, pw):
                        if target is None or dd < target[0]:
                            target = (dd, w)
            if target is not None:
                out.append((target[0], u, target[1], True))
                taken.append(comp[j])
            else:
                m = net.off[cj] + np.arange(max(sub[idx[j]] - net.off[cj] - 5, 0), min(sub[idx[j]] - net.off[cj] + 6, net.off[cj + 1] - net.off[cj]))
                m = m[np.argsort(((net.xy[m] - p) ** 2).sum(axis=1))]
                for k in m[:3]:
                    pk = net.xy[k]
                    vv = pk - p
                    dd = np.hypot(*vv)
                    if 0 < dd <= join_m and vv @ d >= cos_min * dd and hand_ok(hand, x0, y0, p, pk):
                        out.append((dd, u, (cj, int(k - net.off[cj])), False))
                        taken.append(comp[j])
                        break
            if len(taken) >= n_comp:
                break
    return out, lab


def make_joins(net, hand, x0, y0, **kw):
    """Add the accepted joins to the network (shortest first); returns the number of joins."""
    cands, lab = find_joins(net, hand, x0, y0, **kw)
    cands.sort(key=lambda t: t[0])
    parent = np.arange(lab.max() + 1)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    used, joins = set(), []
    for dd, u, t, is_end in cands:
        if u in used:
            continue
        tn = t if is_end else None
        if is_end and tn in used:
            continue
        tc = lab[tn] if is_end else lab[net.node(t[0], 0)]
        ra, rb = find(lab[u]), find(tc)
        if ra == rb:
            continue
        parent[ra] = rb
        used.add(u)
        if is_end:
            used.add(tn)
        joins.append((u, t, is_end, dd))
    for u, t, is_end, dd in joins:
        v = t if is_end else net.node(t[0], t[1])
        net.edges.append((u, v, 2, -1, 0, 0, dd))
    net.split_curves()
    return len(joins)


def trees(net):
    """Yield one dict per tree with at least two ends: id, nodes, root (index in nodes), tips (indices in
    nodes, root excluded), dist and pred (from the root, over nodes). The root is one end of the longest path."""
    u, v, w = net.arrays()
    n = len(net.key)
    g = coo_matrix((w + 1e-6, (u, v)), shape=(n, n)).tocsr()
    g = g + g.T
    _, lab = connected_components(g, directed=False)
    deg = net.degree()
    for t in range(lab.max() + 1):
        nodes = np.nonzero(lab == t)[0]
        is_tip = deg[nodes] == 1
        if is_tip.sum() < 2:
            continue
        sub = g[nodes][:, nodes]
        far = lambda d: int(np.argmax(np.where(is_tip & np.isfinite(d), d, -1)))
        root = far(dijkstra(sub, indices=[int(np.argmax(is_tip))], directed=False)[0])
        dist, pred = dijkstra(sub, indices=[root], directed=False, return_predecessors=True)
        tips = np.nonzero(is_tip)[0]
        yield dict(id=t, nodes=nodes, root=root, tips=tips[tips != root], dist=dist[0], pred=pred[0])


def path_polyline(net, tr, tip, emap):
    """Polyline every STEP m from the root of tree `tr` to its tip (index in nodes), and the curves it follows."""
    seq, x = [], tip
    while x != tr["root"]:
        seq.append(x)
        x = tr["pred"][x]
    seq.append(tr["root"])
    pieces, used = [], set()
    for p, q in zip(seq[::-1][:-1], seq[::-1][1:]):
        pn, qn = int(tr["nodes"][p]), int(tr["nodes"][q])
        k, rev = emap[(pn, qn)]
        _, _, kind, c, i0, i1, _ = net.edges[k]
        if kind == 0:
            seg = net.xy[net.off[c] + i0:net.off[c] + i1 + 1]
            pieces.append(seg[::-1] if rev else seg)
            used.add(c)
        else:
            pieces.append(resample_line(net.pt(*net.key[pn]), net.pt(*net.key[qn])))
    return join_resample(pieces), used


def edge_map(net):
    emap = {}
    for k, e in enumerate(net.edges):
        emap[(e[0], e[1])] = (k, False)
        emap[(e[1], e[0])] = (k, True)
    return emap


def join_resample(pieces, step=STEP):
    """Concatenate the pieces of a path and resample every `step` m of arc length."""
    p = np.concatenate(pieces)
    s = np.r_[0.0, np.cumsum(np.hypot(*np.diff(p, axis=0).T))]
    m = np.arange(0.0, s[-1] + 1e-9, step)
    return np.c_[np.interp(m, s, p[:, 0]), np.interp(m, s, p[:, 1])]
