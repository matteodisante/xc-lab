"""Exact tiled (out-of-core) version of the routing and the crest edges of ridges.py (FLATS = "blm").

The grid is cut into square tiles of T cells; every array lives on disk (`Grid`, a .npy file read and written by
blocks) and a few tiles are in memory at a time. Each stage gives the arrays of ridges.py exactly:

fill   filled surface: priority flood per tile from its edge cells, minimum spill elevations between the
       watersheds of the edge cells (tile pairs and in-tile pairs), a flood on that graph from the true outlets
       (Barnes 2016), then each cell is raised to the level of its watershed.
flats  flats (cells without exit on the filled surface) that touch a tile edge: union over the edge cells.
dirs   drain directions by the code of ridges.py on the tile with a 2-cell halo; each flat that crosses a tile edge
       is gathered whole (bounding box) and resolved by the same code.
acc    flow accumulation in cell counts: local per tile, amounts crossing the tile edges summed on the graph of
       the exit cells, second local pass (Barnes 2017).
links  valley network (cells draining at least the valley area) built globally, links and their forest, then the
       link of each cell (lid) and the elevation of the valley cell it drains to (sz) pushed up the drainage per
       tile with the same exit graph; height above the streams (hand).
edges  crest sides and flat candidates per tile with a halo for the cross-section, in the order and format of
       ridges.crest_edges.

Link numbers differ from ridges.drainage (there they follow the visiting order); here they follow the grid index
of the first cell of the link. Everything else, and every pair (same link, upstream or downstream), is identical.
"""

import os
import time
from pathlib import Path

import numpy as np
from numba import njit
from scipy import ndimage as ndi
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

import curves
import ridges
from ridges import DC, DR, NODATA, OUTLET, UNSEEN

HAND_NONE = -32768  # int16 height above the streams where the cell drains to no valley (as in build.py)
FWD = ((0, 1), (1, 0), (1, 1), (1, -1))  # one orientation of each pair of neighbouring cells
INT32_MAX = np.iinfo(np.int32).max


# --- disk rasters --------------------------------------------------------------------------------------------

class Grid:
    """A 2-D .npy file read and written by blocks (pread/pwrite); reads outside the grid give `fill`."""

    def __init__(self, path, shape=None, dtype=None):
        path = Path(path)
        if shape is not None and not path.exists():
            tmp = path.with_name(path.stem + "_tmp.npy")
            np.lib.format.open_memmap(tmp, mode="w+", dtype=dtype, shape=tuple(shape)).flush()
            os.replace(tmp, path)
        with open(path, "rb") as f:
            major, _ = np.lib.format.read_magic(f)
            read = np.lib.format.read_array_header_1_0 if major == 1 else np.lib.format.read_array_header_2_0
            self.shape, _, self.dtype = read(f)
            self.off = f.tell()
        self.path = path
        self.fd = os.open(path, os.O_RDWR)

    def __del__(self):
        try:
            os.close(self.fd)
        except Exception:
            pass

    def _pread(self, n, off):
        buf = bytearray()
        while len(buf) < n:
            b = os.pread(self.fd, min(n - len(buf), 1 << 30), off + len(buf))
            if not b:
                raise IOError(f"short read in {self.path}")
            buf += b
        return buf

    def read(self, r0, r1, c0, c1, fill=0):
        H, W = self.shape
        sz = self.dtype.itemsize
        out = np.full((r1 - r0, c1 - c0), fill, self.dtype)
        ra, rb, ca, cb = max(r0, 0), min(r1, H), max(c0, 0), min(c1, W)
        if ra >= rb or ca >= cb:
            return out
        if ca == 0 and cb == W:
            a = np.frombuffer(self._pread((rb - ra) * W * sz, self.off + ra * W * sz), self.dtype)
            out[ra - r0:rb - r0, ca - c0:cb - c0] = a.reshape(rb - ra, W)
            return out
        for r in range(ra, rb):
            out[r - r0, ca - c0:cb - c0] = np.frombuffer(self._pread((cb - ca) * sz, self.off + (r * W + ca) * sz),
                                                          self.dtype)
        return out

    def write(self, r0, c0, a):
        W = self.shape[1]
        sz = self.dtype.itemsize
        a = np.ascontiguousarray(a, self.dtype)
        for k in range(a.shape[0]):
            os.pwrite(self.fd, a[k].tobytes(), self.off + ((r0 + k) * W + c0) * sz)

    def array(self):
        return self.read(0, self.shape[0], 0, self.shape[1])


class Run:
    """A tiled run: the folder of its arrays and tables, the grid shape, the tile size and the parameters."""

    def __init__(self, folder, shape, T, cell=ridges.CELL, valley_km2=ridges.VALLEY_KM2, xsec=ridges.XSEC_CELLS,
                 drop=ridges.XSEC_DROP, hand_min=ridges.HAND_MIN, out=None, name="tiled", keep=False, log=None):
        self.dir = Path(folder)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.out = Path(out) if out else self.dir
        self.out.mkdir(parents=True, exist_ok=True)
        self.shape, self.T = tuple(int(s) for s in shape), int(T)
        self.cell, self.xsec, self.drop, self.hand_min = cell, int(xsec), drop, hand_min
        self.min_cells = ridges.valley_cells(valley_km2, cell)
        self.halo = max(self.xsec + 1, curves.EDGE_CELLS + 1)
        self.name, self.keep = name, keep
        self.log = log or (lambda m: print(m, flush=True))
        self.t0 = time.time()
        self.z = self.grid("z", np.float32)

    def grid(self, name, dtype, path=None):
        return Grid(path or self.dir / f"{name}.npy", self.shape, dtype)

    def tiles(self):
        H, W = self.shape
        for i in range((H + self.T - 1) // self.T):
            for j in range((W + self.T - 1) // self.T):
                yield i, j, i * self.T, min(H, (i + 1) * self.T), j * self.T, min(W, (j + 1) * self.T)

    def discard(self, *names):
        """Deletes per-cell rasters that no later stage reads (all kept with keep=True)."""
        if not self.keep:
            for n in names:
                (self.dir / f"{n}.npy").unlink(missing_ok=True)

    def sdir(self, stage):
        d = self.dir / stage
        d.mkdir(exist_ok=True)
        return d

    def per_tile(self, stage, fn):
        """Runs fn(i, j, r0, r1, c0, c1) -> dict of arrays on every tile not done; the dict is the done marker."""
        d = self.sdir(stage)
        tiles = list(self.tiles())
        t = time.time()
        for n, (i, j, r0, r1, c0, c1) in enumerate(tiles):
            f = d / f"{i:04d}_{j:04d}.npz"
            if f.exists():
                continue
            tmp = d / f"{i:04d}_{j:04d}_tmp.npz"
            np.savez(tmp, **{**fn(i, j, r0, r1, c0, c1), "_done": np.zeros(1, np.int8)})
            os.replace(tmp, f)
            if (n + 1) % 10 == 0 or n + 1 == len(tiles):
                self.log(f"  {stage}: {n + 1}/{len(tiles)} tiles, {time.time() - t:.0f} s")

    def collect(self, stage):
        parts = [np.load(f) for f in sorted(self.sdir(stage).glob("[0-9]*[0-9].npz"))]
        return {k: np.concatenate([p[k] for p in parts]) for k in parts[0].files if k != "_done"}

    def gid(self, r0, c0, w, local):
        """Grid index (int64) of the cells of a tile given by their flat index in the tile."""
        return (r0 + local // w).astype(np.int64) * self.shape[1] + c0 + local % w


# --- kernels -------------------------------------------------------------------------------------------------

@njit(cache=True)
def tile_flood(z, seed):
    """Priority flood of a tile from its seeds. z float32 (h, w) with NaN; seed int32 (h, w): -1 for no seed, else
    the label of the seed. Returns f (level at which each cell is reached, at least its z) and lab (label of the seed
    that reaches it first, -1 for no data)."""
    h, w = z.shape
    f = z.copy()
    lab = np.full((h, w), -2, np.int32)
    hk, hi, hn = np.empty(1 << 16, np.float32), np.empty(1 << 16, np.int32), 0
    for r in range(h):
        for c in range(w):
            if np.isnan(z[r, c]):
                lab[r, c] = -1
            elif seed[r, c] >= 0:
                lab[r, c] = seed[r, c]
                hk, hi, hn = ridges._heap_push(hk, hi, hn, z[r, c], r * w + c)
    while hn > 0:
        key, i, hn = ridges._heap_pop(hk, hi, hn)
        r, c = i // w, i % w
        for k in range(8):
            rr, cc = r + DR[k], c + DC[k]
            if rr < 0 or cc < 0 or rr >= h or cc >= w or lab[rr, cc] != -2:
                continue
            f[rr, cc] = max(z[rr, cc], key)
            lab[rr, cc] = lab[r, c]
            hk, hi, hn = ridges._heap_push(hk, hi, hn, f[rr, cc], rr * w + cc)
    return f, lab


@njit(cache=True)
def minimax_flood(indptr, nbr, wt, src):
    """Level of every node of a graph: the smallest over the paths from src of the largest edge weight."""
    n = indptr.size - 1
    lev = np.full(n, np.inf, np.float32)
    lev[src] = -np.inf
    hk, hi, hn = np.empty(1 << 16, np.float32), np.empty(1 << 16, np.int32), 0
    hk, hi, hn = ridges._heap_push(hk, hi, hn, lev[src], src)
    while hn > 0:
        key, i, hn = ridges._heap_pop(hk, hi, hn)
        if key > lev[i]:
            continue
        for e in range(indptr[i], indptr[i + 1]):
            v = max(key, wt[e])
            if v < lev[nbr[e]]:
                lev[nbr[e]] = v
                hk, hi, hn = ridges._heap_push(hk, hi, hn, v, nbr[e])
    return lev


@njit(cache=True)
def tile_order(d):
    """Valid cells of a tile upstream first: every cell comes before the in-tile cell it drains into."""
    h, w = d.shape
    deg = np.zeros((h, w), np.int8)
    for r in range(h):
        for c in range(w):
            if d[r, c] >= 0:
                rr, cc = r + DR[d[r, c]], c + DC[d[r, c]]
                if 0 <= rr < h and 0 <= cc < w:
                    deg[rr, cc] += 1
    order = np.empty(h * w, np.int32)
    nq = 0
    for r in range(h):
        for c in range(w):
            if d[r, c] >= OUTLET and deg[r, c] == 0:
                order[nq] = r * w + c
                nq += 1
    qh = 0
    while qh < nq:
        i = order[qh]
        qh += 1
        r, c = i // w, i % w
        if d[r, c] >= 0:
            rr, cc = r + DR[d[r, c]], c + DC[d[r, c]]
            if 0 <= rr < h and 0 <= cc < w:
                deg[rr, cc] -= 1
                if deg[rr, cc] == 0:
                    order[nq] = rr * w + cc
                    nq += 1
    return order[:nq].copy()


@njit(cache=True)
def accumulate(d, order, eidx, eval_):
    """Cell counts down the in-tile drainage; cell eidx[k] starts with 1 + eval_[k] (amounts from other tiles)."""
    h, w = d.shape
    acc = np.ones((h, w), np.int64)
    a1 = acc.ravel()
    for k in range(eidx.size):
        a1[eidx[k]] += eval_[k]
    for m in range(order.size):
        i = order[m]
        r, c = i // w, i % w
        if d[r, c] >= 0:
            rr, cc = r + DR[d[r, c]], c + DC[d[r, c]]
            if 0 <= rr < h and 0 <= cc < w:
                acc[rr, cc] += acc[r, c]
    return acc


@njit(cache=True)
def exit_root(d, order):
    """Flat index of the cell where the in-tile drainage of each cell leaves the tile (-1: ends at an outlet)."""
    h, w = d.shape
    root = np.full((h, w), -1, np.int32)
    for m in range(order.size - 1, -1, -1):
        i = order[m]
        r, c = i // w, i % w
        if d[r, c] >= 0:
            rr, cc = r + DR[d[r, c]], c + DC[d[r, c]]
            if 0 <= rr < h and 0 <= cc < w:
                root[r, c] = root[rr, cc]
            else:
                root[r, c] = i
    return root


@njit(cache=True)
def exit_cells(d):
    """Flat index and direction of the cells that drain out of the tile."""
    h, w = d.shape
    n = 0
    for r in range(h):
        for c in range(w):
            if d[r, c] >= 0:
                rr, cc = r + DR[d[r, c]], c + DC[d[r, c]]
                if rr < 0 or cc < 0 or rr >= h or cc >= w:
                    n += 1
    ei, ek = np.empty(n, np.int64), np.empty(n, np.int64)
    n = 0
    for r in range(h):
        for c in range(w):
            if d[r, c] >= 0:
                rr, cc = r + DR[d[r, c]], c + DC[d[r, c]]
                if rr < 0 or cc < 0 or rr >= h or cc >= w:
                    ei[n], ek[n] = r * w + c, d[r, c]
                    n += 1
    return ei, ek


@njit(cache=True)
def sum_up(nxt, loc):
    """loc plus the sums of the children, nxt[e] being the parent of e (-1 for none); n_done < n means a cycle."""
    n = nxt.size
    acc = loc.copy()
    deg = np.zeros(n, np.int32)
    for e in range(n):
        if nxt[e] >= 0:
            deg[nxt[e]] += 1
    q = np.empty(n, np.int32)
    nq = 0
    for e in range(n):
        if deg[e] == 0:
            q[nq] = e
            nq += 1
    qh = 0
    while qh < nq:
        e = q[qh]
        qh += 1
        p = nxt[e]
        if p >= 0:
            acc[p] += acc[e]
            deg[p] -= 1
            if deg[p] == 0:
                q[nq] = p
                nq += 1
    return acc, nq


@njit(cache=True)
def chase(nxt, lid, sz):
    """Entries with lid < 0 take the lid and sz of the first resolved entry down the chain nxt."""
    n = nxt.size
    stack = np.empty(n, np.int32)
    for e in range(n):
        if lid[e] >= 0:
            continue
        sp, j = 0, e
        while lid[j] < 0:
            stack[sp] = j
            sp += 1
            j = nxt[j]
        for k in range(sp):
            lid[stack[k]], sz[stack[k]] = lid[j], sz[j]


@njit(cache=True)
def valley_links(tpos):
    """Links of a valley network given by the position of the downstream cell of each cell (-1 for none), the cells
    being sorted by grid index. A link starts at a cell with other than one inflow and continues downstream.
    Returns the link of each cell (1..nl, ordered by the first cell) and the downstream link of every link."""
    n = tpos.size
    nin = np.zeros(n, np.int32)
    par = np.full(n, -1, np.int32)
    for i in range(n):
        if tpos[i] >= 0:
            nin[tpos[i]] += 1
            par[tpos[i]] = i
    deg = nin.copy()
    q = np.empty(n, np.int32)
    nq = 0
    for i in range(n):
        if deg[i] == 0:
            q[nq] = i
            nq += 1
    lk = np.zeros(n, np.int32)
    start = np.empty(n, np.int64)
    nl, qh = 0, 0
    while qh < nq:
        i = q[qh]
        qh += 1
        if nin[i] == 1:
            lk[i] = lk[par[i]]
        else:
            start[nl] = i
            nl += 1
            lk[i] = nl
        t = tpos[i]
        if t >= 0:
            deg[t] -= 1
            if deg[t] == 0:
                q[nq] = t
                nq += 1
    if nq != n:
        raise RuntimeError("the valley network has a cycle")
    rank = np.argsort(start[:nl])
    new = np.empty(nl + 1, np.int32)
    for a in range(nl):
        new[rank[a] + 1] = a + 1
    for i in range(n):
        lk[i] = new[lk[i]]
    down = np.zeros(nl + 1, np.int32)
    for i in range(n):
        if tpos[i] >= 0 and lk[tpos[i]] != lk[i]:
            down[lk[i]] = lk[tpos[i]]
    return lk, down


@njit(cache=True)
def tile_chains(d, order, valley, lid, sz):
    """Fills lid and sz of the cells that are not valley cells with those of the first valley cell down their
    drainage; the cells that leave the tile before reaching one get lid -1. Returns the exit cell of each (-1: none)."""
    h, w = d.shape
    root = np.full((h, w), -1, np.int32)
    for m in range(order.size - 1, -1, -1):
        i = order[m]
        r, c = i // w, i % w
        if valley[r, c] or d[r, c] < 0:
            continue
        rr, cc = r + DR[d[r, c]], c + DC[d[r, c]]
        if 0 <= rr < h and 0 <= cc < w:
            lid[r, c], sz[r, c], root[r, c] = lid[rr, cc], sz[rr, cc], root[rr, cc]
        else:
            lid[r, c], root[r, c] = -1, i
    return root


# --- helpers -------------------------------------------------------------------------------------------------

def outlet_mask(zb):
    """Cells with data next to a cell without data (the outer ring of the block is not reliable)."""
    nan = np.isnan(zb)
    near = np.zeros_like(nan)
    hh, ww = zb.shape
    for dr, dc in zip(DR, DC):
        near[1:-1, 1:-1] |= nan[1 + dr:hh - 1 + dr, 1 + dc:ww - 1 + dc]
    return ~nan & near


def group_min(u, v, w):
    """Distinct pairs (u, v) with the smallest w."""
    o = np.lexsort((w, v, u))
    u, v, w = u[o], v[o], w[o]
    first = np.r_[u.size > 0, (u[1:] != u[:-1]) | (v[1:] != v[:-1])][:u.size]
    return u[first], v[first], w[first]


def edge_layer(h, w):
    m = np.zeros((h, w), bool)
    m[0], m[-1], m[:, 0], m[:, -1] = True, True, True, True
    return m


# --- A. fill -------------------------------------------------------------------------------------------------

def fill_pass1(run, i, j, r0, r1, c0, c1):
    h, w, W = r1 - r0, c1 - c0, run.shape[1]
    zb = run.z.read(r0 - 2, r1 + 2, c0 - 2, c1 + 2, np.nan)
    ob = outlet_mask(zb)
    out = ob[2:2 + h, 2:2 + w]
    zc = np.ascontiguousarray(zb[2:2 + h, 2:2 + w])
    carry = edge_layer(h, w) & ~np.isnan(zc) & ~out  # edge cells that are not true outlets: one watershed each
    seed = np.full((h, w), -1, np.int32)
    seed[out] = 0
    seed[carry] = np.arange(1, carry.sum() + 1)
    rr, cc = np.nonzero(carry)
    keys = (r0 + rr).astype(np.int64) * W + c0 + cc
    f, lab = tile_flood(zc, seed)
    run.grid("filled", np.float32).write(r0, c0, f)
    run.grid("fl_lab", np.int32).write(r0, c0, lab)
    km = np.r_[-1, keys]  # label -> key (-1: a true outlet)
    lo, hi, sp = [], [], []
    for dr, dc in FWD:
        sa = (slice(0, h - dr), slice(max(0, -dc), w - max(0, dc)))
        sb = (slice(dr, h), slice(max(0, dc), w - max(0, -dc)))
        la, lb = lab[sa], lab[sb]
        m = (la >= 0) & (lb >= 0) & (la != lb)
        lo.append(np.minimum(la, lb)[m])
        hi.append(np.maximum(la, lb)[m])
        sp.append(np.maximum(f[sa], f[sb])[m])
    lo, hi, sp = group_min(np.concatenate(lo), np.concatenate(hi), np.concatenate(sp))
    su, sv, sw = [km[lo]], [km[hi]], [sp]
    # pairs across the tile edge: the cell inside, its neighbour in the next tile
    zi = zb[2:2 + h, 2:2 + w]
    rows, cols = np.arange(h)[:, None], np.arange(w)[None, :]
    for dr, dc in FWD:
        rb, cb = rows + dr, cols + dc
        outside = ~((rb >= 0) & (rb < h) & (cb >= 0) & (cb < w))
        zn = zb[2 + dr:2 + dr + h, 2 + dc:2 + dc + w]
        m = outside & ~np.isnan(zi) & ~np.isnan(zn)
        ra, ca = np.nonzero(m)
        ka = np.where(ob[2 + ra, 2 + ca], -1, (r0 + ra).astype(np.int64) * W + c0 + ca)
        kb = np.where(ob[2 + ra + dr, 2 + ca + dc], -1, (r0 + ra + dr).astype(np.int64) * W + c0 + ca + dc)
        su.append(ka)
        sv.append(kb)
        sw.append(np.maximum(zi[ra, ca], zn[ra, ca]))
    return dict(keys=keys, su=np.concatenate(su), sv=np.concatenate(sv), sw=np.concatenate(sw))


def fill_graph(run):
    """Global level of every watershed of an edge cell (and -inf for the true outlets): `levels.npz`."""
    f = run.dir / "levels.npz"
    if f.exists():
        return
    t = run.collect("fill1")
    nodes = np.unique(np.r_[t["keys"], t["su"], t["sv"], -1])
    u, v, w = group_min(*(np.searchsorted(nodes, t[k]) for k in ("su", "sv")), t["sw"])
    a, b = np.r_[u, v], np.r_[v, u]
    o = np.argsort(a, kind="stable")
    indptr = np.r_[0, np.cumsum(np.bincount(a, minlength=nodes.size))]
    lev = minimax_flood(indptr, b[o].astype(np.int32), np.r_[w, w][o], 0)
    if not np.isfinite(lev[1:]).all():
        raise RuntimeError("a watershed is not connected to an outlet")
    np.savez(f, nodes=nodes, lev=lev)
    run.log(f"  fill: {nodes.size} watersheds, {u.size} spill pairs")


def fill_pass2(run, g, i, j, r0, r1, c0, c1):
    t = np.load(run.sdir("fill1") / f"{i:04d}_{j:04d}.npz")
    lv = np.r_[np.float32(-np.inf), g["lev"][np.searchsorted(g["nodes"], t["keys"])]]
    F = run.grid("filled", np.float32)  # raised in place: applying it twice changes nothing
    f = F.read(r0, r1, c0, c1)
    lab = run.grid("fl_lab", np.int32).read(r0, r1, c0, c1)
    out = np.where(lab >= 0, np.maximum(f, lv[np.maximum(lab, 0)]), np.nan).astype(np.float32)
    F.write(r0, c0, out)
    return {}


def fill(run):
    run.grid("fl_lab", np.int32), run.grid("filled", np.float32)
    run.per_tile("fill1", lambda *a: fill_pass1(run, *a))
    fill_graph(run)
    g = dict(np.load(run.dir / "levels.npz"))
    run.per_tile("fill2", lambda *a: fill_pass2(run, g, *a))
    (run.dir / "fl_lab.npy").unlink(missing_ok=True)


# --- B. flats and directions ---------------------------------------------------------------------------------

def _blocks(run, r0, r1, c0, c1, halo=2):
    z = run.z.read(r0 - halo, r1 + halo, c0 - halo, c1 + halo, np.nan)
    f = run.grid("filled", np.float32).read(r0 - halo, r1 + halo, c0 - halo, c1 + halo, np.nan)
    return z, f


def flats_tile(run, i, j, r0, r1, c0, c1):
    """Flats touching the tile edge: for each, its bounding box and size inside the tile; its edge cells."""
    h, w, W = r1 - r0, c1 - c0, run.shape[1]
    zb, fb = _blocks(run, r0, r1, c0, c1)
    d = ridges._exits(zb, fb.ravel()).reshape(zb.shape)
    lab, n = ndi.label(d[2:2 + h, 2:2 + w] == UNSEEN, structure=np.ones((3, 3)))
    er, ec = np.nonzero(edge_layer(h, w) & (lab > 0))
    comp, inv = np.unique(lab[er, ec], return_inverse=True)
    boxes = ndi.find_objects(lab, n)
    bbox = np.array([[r0 + boxes[k - 1][0].start, r0 + boxes[k - 1][0].stop,
                      c0 + boxes[k - 1][1].start, c0 + boxes[k - 1][1].stop] for k in comp], np.int64).reshape(-1, 4)
    size = np.bincount(lab.ravel(), minlength=n + 1)[comp]
    return dict(eidx=(r0 + er).astype(np.int64) * W + c0 + ec, enode=inv.astype(np.int64), bbox=bbox, size=size,
                ncomp=np.array([comp.size]))


def flats_global(run):
    """Flats crossing tile edges: `crossing.npz` with bounding box (r0, r1, c0, c1), cells and a cell of each."""
    f = run.dir / "crossing.npz"
    if f.exists():
        return
    W = run.shape[1]
    parts = [np.load(p) for p in sorted(run.sdir("flats").glob("[0-9]*[0-9].npz"))]
    base = np.r_[0, np.cumsum([p["ncomp"][0] for p in parts])]
    n = int(base[-1])
    eidx = np.concatenate([p["eidx"] for p in parts])
    enode = np.concatenate([p["enode"] + b for p, b in zip(parts, base)])
    bbox = np.concatenate([p["bbox"] for p in parts]).reshape(-1, 4)
    size = np.concatenate([p["size"] for p in parts])
    o = np.argsort(eidx)
    eidx, enode = eidx[o], enode[o]
    r, c = eidx // W, eidx % W
    ua, ub = [], []
    for dr, dc in FWD:
        ok = (r + dr < run.shape[0]) & (c + dc >= 0) & (c + dc < W)
        nb = (r + dr) * W + c + dc
        pos = np.searchsorted(eidx, nb)
        pos[pos == eidx.size] = 0
        hit = ok & (eidx[pos] == nb) & (enode[pos] != enode)
        ua.append(enode[hit])
        ub.append(enode[pos[hit]])
    ua, ub = np.concatenate(ua), np.concatenate(ub)
    _, grp = connected_components(coo_matrix((np.ones(ua.size), (ua, ub)), shape=(n, n)), directed=False)
    cnt = np.bincount(grp, minlength=grp.max() + 1 if n else 0)
    cross = np.flatnonzero(cnt[grp] > 1)
    box, cells, seed = np.empty((0, 4), np.int64), np.empty(0, np.int64), np.empty(0, np.int64)
    if cross.size:
        o = cross[np.argsort(grp[cross], kind="stable")]
        cut = np.r_[0, np.flatnonzero(np.diff(grp[o])) + 1]
        box = np.c_[np.minimum.reduceat(bbox[o, 0], cut), np.maximum.reduceat(bbox[o, 1], cut),
                    np.minimum.reduceat(bbox[o, 2], cut), np.maximum.reduceat(bbox[o, 3], cut)]
        cells = np.add.reduceat(size[o], cut)
        first = np.full(n, -1, np.int64)
        first[enode[::-1]] = eidx[::-1]
        seed = first[o[cut]]
    np.savez(f, bbox=box, cells=cells, seed=seed)
    run.log(f"  flats: {len(cells)} flats cross tile edges, largest {cells.max() if len(cells) else 0} cells")


def dirs_tile(run, i, j, r0, r1, c0, c1):
    h, w = r1 - r0, c1 - c0
    zb, fb = _blocks(run, r0, r1, c0, c1)
    f1 = fb.ravel()
    d = ridges._exits(zb, f1)
    ridges._resolve(f1, d, ridges._increments(f1, d, zb.shape[1]), zb.shape[1])
    run.grid("d", np.int8).write(r0, c0, d.reshape(zb.shape)[2:2 + h, 2:2 + w])
    return {}


def dirs_patch(run):
    """Resolves each flat that crosses a tile edge on its own bounding box and writes its cells."""
    mark = run.dir / "dirs_patched"
    if mark.exists():
        return
    g = np.load(run.dir / "crossing.npz")
    D, W = run.grid("d", np.int8), run.shape[1]
    for k, ((r0, r1, c0, c1), s) in enumerate(zip(g["bbox"], g["seed"])):
        zb, fb = _blocks(run, r0, r1, c0, c1)
        f1, w = fb.ravel(), zb.shape[1]
        d = ridges._exits(zb, f1)
        lab, _ = ndi.label(d.reshape(zb.shape) == UNSEEN, structure=np.ones((3, 3)))
        mine = lab == lab[s // W - r0 + 2, s % W - c0 + 2]
        ridges._resolve(f1, d, ridges._increments(f1, d, w), w)
        d = d.reshape(zb.shape)[2:-2, 2:-2]
        mine = mine[2:-2, 2:-2]
        D.write(r0, c0, np.where(mine, d, D.read(r0, r1, c0, c1)))
    mark.write_text(f"{len(g['bbox'])} flats\n")


def flats(run):
    run.per_tile("flats", lambda *a: flats_tile(run, *a))
    flats_global(run)


def dirs(run):
    run.grid("d", np.int8)
    flats(run)
    run.per_tile("dirs", lambda *a: dirs_tile(run, *a))
    dirs_patch(run)


# --- C. accumulation -----------------------------------------------------------------------------------------

def _tile_flow(run, r0, r1, c0, c1):
    d = run.grid("d", np.int8).read(r0, r1, c0, c1)
    if (d == UNSEEN).any():
        raise RuntimeError("cells without direction")
    order = tile_order(d)
    if order.size != (d != NODATA).sum():
        raise RuntimeError("directions have a cycle inside a tile")
    return d, order


def acc_pass1(run, i, j, r0, r1, c0, c1):
    h, w, W = r1 - r0, c1 - c0, run.shape[1]
    d, order = _tile_flow(run, r0, r1, c0, c1)
    acc = accumulate(d, order, np.empty(0, np.int64), np.empty(0, np.int64))
    root = exit_root(d, order)
    ei, ek = exit_cells(d)
    gi = run.gid(r0, c0, w, ei)
    tgt = gi + DR[ek] * W + DC[ek]
    pe = np.flatnonzero(edge_layer(h, w).ravel())
    pr = root.ravel()[pe].astype(np.int64)
    return dict(e_idx=gi, e_tgt=tgt, e_acc=acc.ravel()[ei], p_idx=run.gid(r0, c0, w, pe),
                p_root=np.where(pr >= 0, run.gid(r0, c0, w, np.maximum(pr, 0)), -1))


def exit_graph(t):
    """Order the exit cells by index; position of the exit that the drainage of each exit's target reaches."""
    o = np.argsort(t["e_idx"])
    e_idx, e_tgt, e_acc = t["e_idx"][o], t["e_tgt"][o], t["e_acc"][o] if "e_acc" in t else None
    po = np.argsort(t["p_idx"])
    p_idx = t["p_idx"][po]
    pos = np.searchsorted(p_idx, e_tgt)
    if not (p_idx[np.minimum(pos, p_idx.size - 1)] == e_tgt).all():
        raise RuntimeError("an exit leads to a cell not on a tile edge")
    return o, e_idx, e_tgt, e_acc, po, pos


def acc_graph(run):
    """Amount entering each cell from the tiles upstream: `incoming.npz` (cell index, cells)."""
    f = run.dir / "incoming.npz"
    if f.exists():
        return
    t = run.collect("acc1")
    o, e_idx, e_tgt, e_acc, po, pos = exit_graph(t)
    root = t["p_root"][po][pos]
    nxt = np.where(root >= 0, np.searchsorted(e_idx, np.maximum(root, 0)), -1)
    if not (e_idx[np.maximum(nxt, 0)][nxt >= 0] == root[nxt >= 0]).all():
        raise RuntimeError("an exit is not in the table")
    tot, nq = sum_up(nxt, e_acc)
    if nq != nxt.size:
        raise RuntimeError("the exits have a cycle")
    cell, inv = np.unique(e_tgt, return_inverse=True)
    np.savez(f, cell=cell, amt=np.bincount(inv, weights=tot.astype(np.float64), minlength=cell.size).astype(np.int64))
    run.log(f"  acc: {e_idx.size} exit cells")


def acc_pass2(run, g, i, j, r0, r1, c0, c1):
    w, W = c1 - c0, run.shape[1]
    d, order = _tile_flow(run, r0, r1, c0, c1)
    r, c = g["cell"] // W, g["cell"] % W
    m = (r >= r0) & (r < r1) & (c >= c0) & (c < c1)
    acc = accumulate(d, order, ((r[m] - r0) * w + c[m] - c0).astype(np.int64), g["amt"][m])
    run.grid("acc", np.int32).write(r0, c0, np.minimum(acc, INT32_MAX))
    return {}


def acc(run):
    run.grid("acc", np.int32)
    run.per_tile("acc1", lambda *a: acc_pass1(run, *a))
    acc_graph(run)
    g = dict(np.load(run.dir / "incoming.npz"))
    run.per_tile("acc2", lambda *a: acc_pass2(run, g, *a))


# --- D. valleys, links, height above the stream --------------------------------------------------------------

def valley_tile(run, i, j, r0, r1, c0, c1):
    w, W = c1 - c0, run.shape[1]
    d = run.grid("d", np.int8).read(r0, r1, c0, c1).ravel()
    v = np.flatnonzero(run.grid("acc", np.int32).read(r0, r1, c0, c1).ravel() >= run.min_cells)
    gi = run.gid(r0, c0, w, v)
    vz = run.grid("filled", np.float32).read(r0, r1, c0, c1).ravel()[v]
    dv = d[v].astype(np.int64)
    return dict(idx=gi, vz=vz, tgt=np.where(dv >= 0, gi + DR[np.maximum(dv, 0)] * W + DC[np.maximum(dv, 0)], -1))


def valley_global(run):
    """Valley network and its link forest: `valley.npz` (cell index, link, downstream link, tour times)."""
    f = run.dir / "valley.npz"
    if f.exists():
        return
    t = run.collect("valley")
    o = np.argsort(t["idx"])
    idx, tgt, vz = t["idx"][o], t["tgt"][o], t["vz"][o]
    tpos = np.where(tgt >= 0, np.searchsorted(idx, np.maximum(tgt, 0)), -1)
    if not (idx[np.maximum(tpos, 0)][tpos >= 0] == tgt[tpos >= 0]).all():
        raise RuntimeError("a valley cell drains into a cell that is not a valley cell")
    lk, down = valley_links(tpos)
    tin, tout = ridges._euler(down)
    np.savez(f, idx=idx, vz=vz, lk=lk, down=down, tin=tin, tout=tout)
    run.log(f"  links: {idx.size} valley cells, {down.size - 1} links")


def _tile_lid(run, r0, r1, c0, c1, v):
    """lid and sz of a tile from the links of its valley cells: those that reach a valley cell inside the tile are
    final, the others (lid -1) wait for the cell they leave the tile at (root)."""
    h, w, W = r1 - r0, c1 - c0, run.shape[1]
    d, order = _tile_flow(run, r0, r1, c0, c1)
    lo, hi = np.searchsorted(v["idx"], [r0 * W, r1 * W])
    cc = v["idx"][lo:hi] % W
    k = lo + np.flatnonzero((cc >= c0) & (cc < c1))
    vi = (v["idx"][k] // W - r0) * w + v["idx"][k] % W - c0
    valley = np.zeros((h, w), bool)
    valley.ravel()[vi] = True
    lid = np.zeros((h, w), np.int32)
    sz = np.full((h, w), np.nan, np.float32)
    lid.ravel()[vi], sz.ravel()[vi] = v["lk"][k], v["vz"][k]
    return d, lid, sz, tile_chains(d, order, valley, lid, sz)


def lid_pass1(run, v, i, j, r0, r1, c0, c1):
    h, w, W = r1 - r0, c1 - c0, run.shape[1]
    d, lid, sz, root = _tile_lid(run, r0, r1, c0, c1, v)
    ei, ek = exit_cells(d)
    ei = ei[lid.ravel()[ei] < 0]  # exits that are valley cells have their own link
    ek = d.ravel()[ei]
    gi = run.gid(r0, c0, w, ei)
    pe = np.flatnonzero(edge_layer(h, w).ravel())
    pr = root.ravel()[pe].astype(np.int64)
    return dict(e_idx=gi, e_tgt=gi + DR[ek] * W + DC[ek], p_idx=run.gid(r0, c0, w, pe), p_lid=lid.ravel()[pe],
                p_sz=sz.ravel()[pe], p_root=np.where(pr >= 0, run.gid(r0, c0, w, np.maximum(pr, 0)), -1))


def lid_graph(run):
    """lid and sz of the cells where the drainage of a tile leaves it: `exits.npz`."""
    f = run.dir / "exits.npz"
    if f.exists():
        return
    t = run.collect("lid1")
    o, e_idx, e_tgt, _, po, pos = exit_graph(t)
    p_root, p_lid, p_sz = (t[k][po][pos] for k in ("p_root", "p_lid", "p_sz"))
    nxt = np.where(p_root >= 0, np.searchsorted(e_idx, np.maximum(p_root, 0)), -1)
    lid, sz = np.where(p_root >= 0, -1, p_lid).astype(np.int32), p_sz.astype(np.float32)
    chase(nxt.astype(np.int32), lid, sz)
    np.savez(f, idx=e_idx, lid=lid, sz=sz)


def lid_pass2(run, v, g, i, j, r0, r1, c0, c1):
    w = c1 - c0
    d, lid, sz, root = _tile_lid(run, r0, r1, c0, c1, v)
    pend = np.flatnonzero(root.ravel() >= 0)
    if pend.size:
        k = np.searchsorted(g["idx"], run.gid(r0, c0, w, root.ravel()[pend].astype(np.int64)))
        lid.ravel()[pend], sz.ravel()[pend] = g["lid"][k], g["sz"][k]
    run.grid("lid", np.int32).write(r0, c0, lid)
    run.grid("sz", np.float32).write(r0, c0, sz)
    z = run.z.read(r0, r1, c0, c1)
    hand = np.where(lid > 0, np.clip(np.rint(z - sz), -32767, 32767), HAND_NONE)
    run.grid("hand", np.int16, run.out / f"hand_{run.name}.npy").write(r0, c0, hand)
    return {}


def links(run):
    run.per_tile("valley", lambda *a: valley_tile(run, *a))
    valley_global(run)
    run.discard("acc", "filled")  # the disk does not keep holes: make room before the next rasters
    run.grid("lid", np.int32), run.grid("sz", np.float32)
    run.grid("hand", np.int16, run.out / f"hand_{run.name}.npy")
    v = dict(np.load(run.dir / "valley.npz"))
    run.per_tile("lid1", lambda *a: lid_pass1(run, v, *a))
    lid_graph(run)
    g = dict(np.load(run.dir / "exits.npz"))
    run.per_tile("lid2", lambda *a: lid_pass2(run, v, g, *a))
    run.discard("d")


# --- E. crest edges ------------------------------------------------------------------------------------------

def edges_tile(run, i, j, r0, r1, c0, c1, v):
    H, W = run.shape
    hl = run.halo
    ro, co = r0 - hl, c0 - hl
    zb = run.z.read(ro, r1 + hl, co, c1 + hl, np.nan)
    lid = run.grid("lid", np.int32).read(ro, r1 + hl, co, c1 + hl, 0)
    sz = run.grid("sz", np.float32).read(ro, r1 + hl, co, c1 + hl, np.nan)
    wb = zb.shape[1]
    res = ridges.crest_edges(zb, lid.ravel(), sz.ravel(), v["tin"], v["tout"], run.xsec, run.drop, run.hand_min)
    out, verts = {}, []
    for name, e in zip(("a", "f"), res):
        e = e.astype(np.int64)
        rv, cv = e // (wb + 1) + ro, e % (wb + 1) + co  # corners (r, c) of both ends, in the grid
        own_r, own_c = rv[:, 1] - 1, cv[:, 1] - 1  # the cell (r, c) that owns the side
        m = (own_r >= r0) & (own_r < r1) & (own_c >= c0) & (own_c < c1)
        k = (cv[m, 0] != cv[m, 1]).astype(np.int64)  # 0 east side, 1 south side
        out[name + "_key"] = (own_r[m] * W + own_c[m]) * 2 + k
        ids = rv[m].astype(np.int64) * (W + 1) + cv[m]
        out[name + "_ids"] = ids
        verts.append(ids.ravel())
    vid = np.unique(np.concatenate(verts))
    near = curves.near_nodata(zb, vid // (W + 1) - ro, vid % (W + 1) - co, curves.EDGE_CELLS)
    return {**out, "vid": vid, "near": near, "cnt": np.array([len(out["a_key"]), len(out["f_key"])])}


def edges_merge(run, x0, y0):
    """edges_<name>.npz in the format of build.drain, the sides in the order of ridges.crest_edges. The sides of a
    tile row come before those of the next (the key starts with the row of the owner cell), so tile rows are sorted
    one at a time into arrays of the final size; the vertices are merged the same way (those on the edge between two
    tile rows are carried to the next one)."""
    H, W = run.shape
    files = sorted(run.sdir("edges").glob("[0-9]*[0-9].npz"))
    rows = {}
    for f in files:
        rows.setdefault(int(f.name[:4]), []).append(f)
    dt = np.int32 if (H + 1) * (W + 1) < 2**31 else np.int64
    cnt = np.array([np.load(f)["cnt"] for f in files]).sum(axis=0)
    res = [np.empty((n, 2), dt) for n in cnt]
    at = [0, 0]
    vids, nears = [], []
    carry = np.empty(0, np.int64), np.empty(0, bool)
    for i in sorted(rows):
        t = [np.load(f) for f in rows[i]]
        for k, n in enumerate(("a", "f")):
            key = np.concatenate([p[n + "_key"] for p in t])
            ids = np.concatenate([p[n + "_ids"] for p in t]).reshape(-1, 2)
            res[k][at[k]:at[k] + key.size] = ids[np.argsort(key)]
            at[k] += key.size
        v = np.concatenate([carry[0]] + [p["vid"] for p in t])
        nr = np.concatenate([carry[1]] + [p["near"] for p in t])
        v, first = np.unique(v, return_index=True)
        nr = nr[first]
        last = i == max(rows)
        out = np.ones(v.size, bool) if last else v // (W + 1) < min(H, (i + 1) * run.T)
        vids.append(v[out].astype(dt)), nears.append(nr[out])
        carry = v[~out], nr[~out]
    np.savez_compressed(run.out / f"edges_{run.name}.npz", acc=res[0], flat=res[1], vid=np.concatenate(vids),
                        near=np.concatenate(nears), shape=np.array(run.shape), x0=x0, y0=y0)
    run.log(f"  edges: {len(res[0])} crest sides, {len(res[1])} flat candidates")


def edges(run, x0, y0):
    v = dict(np.load(run.dir / "valley.npz"))
    run.per_tile("edges", lambda *a: edges_tile(run, *a, v))
    edges_merge(run, x0, y0)
