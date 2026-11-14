"""Flow routing and crest edges of a DEM.

Priority flood fills the depressions, every cell drains to its steepest lower neighbour (cells of a flat or a
filled depression to the cell that reached them), valleys are the cells draining at least VALLEY_KM2 and are
cut into links at the confluences. A crest edge is a cell side between two cells that drain into different
links, neither downstream of the other, with a cross-section and a height above the streams large enough.
Elevations are float32 with NaN for no data; cell (r, c) has corners (r, c) to (r + 1, c + 1).
"""

import numpy as np
from numba import njit

CELL = 25.0
VALLEY_KM2 = 0.5
XSEC_CELLS = 6  # cells from the crest to the ground of the cross-section
XSEC_DROP = 10.0  # m, ground this far below the crest on each side
HAND_MIN = 20.0  # m, crest above the stream on both sides
DR = np.array([0, 1, 1, 1, 0, -1, -1, -1], np.int64)
DC = np.array([1, 1, 0, -1, -1, -1, 0, 1], np.int64)
OUTLET, NODATA, UNSEEN, INHEAP = -1, -2, -3, -4  # codes of the direction array besides 0..7


def valley_cells(valley_km2=VALLEY_KM2):
    return int(round(valley_km2 * 1e6 / CELL**2))


@njit(cache=True)
def _heap_push(hk, hi, n, k, i):
    if n == hk.size:
        hk2, hi2 = np.empty(2 * n, np.float32), np.empty(2 * n, np.int32)
        hk2[:n], hi2[:n] = hk, hi
        hk, hi = hk2, hi2
    j = n
    while j > 0:
        p = (j - 1) >> 1
        if hk[p] <= k:
            break
        hk[j], hi[j] = hk[p], hi[p]
        j = p
    hk[j], hi[j] = k, i
    return hk, hi, n + 1


@njit(cache=True)
def _heap_pop(hk, hi, n):
    k0, i0 = hk[0], hi[0]
    n -= 1
    k, i = hk[n], hi[n]
    j = 0
    while True:
        ch = 2 * j + 1
        if ch >= n:
            break
        if ch + 1 < n and hk[ch + 1] < hk[ch]:
            ch += 1
        if hk[ch] >= k:
            break
        hk[j], hi[j] = hk[ch], hi[ch]
        j = ch
    if n > 0:
        hk[j], hi[j] = k, i
    return k0, i0, n


@njit(cache=True)
def flood(z):
    """Priority flood from the border and the data edge. Returns the filled surface (float32, flat),
    the drain direction (int8, flat) and the cells in the order they were reached (downstream first)."""
    h, w = z.shape
    zf = z.ravel()
    n_all = zf.size
    filled = zf.copy()
    d = np.full(n_all, UNSEEN, np.int8)
    n_valid = 0
    for i in range(n_all):
        if np.isnan(zf[i]):
            d[i] = NODATA
        else:
            n_valid += 1
    order = np.empty(n_valid, np.int32)
    hk, hi, hn = np.empty(1 << 20, np.float32), np.empty(1 << 20, np.int32), 0
    q, qh, qt = np.empty(1 << 20, np.int32), 0, 0
    for r in range(h):
        for c in range(w):
            i = r * w + c
            if d[i] == NODATA:
                continue
            edge = r == 0 or c == 0 or r == h - 1 or c == w - 1
            if not edge:
                for k in range(8):
                    if d[(r + DR[k]) * w + c + DC[k]] == NODATA:
                        edge = True
                        break
            if edge:
                d[i] = OUTLET
                hk, hi, hn = _heap_push(hk, hi, hn, filled[i], i)
    no = 0
    while True:
        if qh < qt:
            i = q[qh]
            qh += 1
            if qh == qt:
                qh = qt = 0
        elif hn > 0:
            key, i, hn = _heap_pop(hk, hi, hn)
            if d[i] == INHEAP:  # steepest strictly lower cell already reached
                r, c = i // w, i % w
                best, bs = OUTLET, 0.0
                for k in range(8):
                    rr, cc = r + DR[k], c + DC[k]
                    if rr < 0 or cc < 0 or rr >= h or cc >= w:
                        continue
                    j = rr * w + cc
                    if d[j] >= OUTLET and filled[j] < filled[i]:
                        s = (filled[i] - filled[j]) / (1.0 if k % 2 == 0 else 1.41421356)
                        if s > bs:
                            best, bs = k, s
                d[i] = best
        else:
            break
        order[no] = i
        no += 1
        key = filled[i]
        r, c = i // w, i % w
        for k in range(8):
            rr, cc = r + DR[k], c + DC[k]
            if rr < 0 or cc < 0 or rr >= h or cc >= w:
                continue
            j = rr * w + cc
            if d[j] != UNSEEN:
                continue
            if zf[j] <= key:
                filled[j] = key
                d[j] = (k + 4) % 8
                if qt == q.size:
                    q2 = np.empty(2 * qt, np.int32)
                    q2[:qt] = q
                    q = q2
                q[qt] = j
                qt += 1
            else:
                d[j] = INHEAP
                hk, hi, hn = _heap_push(hk, hi, hn, filled[j], j)
    return filled, d, order


@njit(cache=True)
def _links(filled, d, order, w, min_cells):
    """Accumulation and link labels. Returns lid (int32, flat: link of the valley each cell drains into,
    0 for none), sz (the cells' valley elevation, written over `filled`) and the downstream link of every link."""
    n = order.size
    acc = np.ones(filled.size, np.int32)
    for m in range(n - 1, -1, -1):
        i = order[m]
        if d[i] >= 0:
            acc[i + DR[d[i]] * w + DC[d[i]]] += acc[i]
    # acc becomes -1 on valley cells, 0 elsewhere, then minus the number of valley inflows
    for i in range(acc.size):
        acc[i] = -1 if acc[i] >= min_cells else 0
    for m in range(n):
        i = order[m]
        if acc[i] < 0 and d[i] >= 0:
            t = i + DR[d[i]] * w + DC[d[i]]
            if acc[t] < 0:
                acc[t] -= 1
    nl = 0
    for m in range(n - 1, -1, -1):  # upstream first
        i = order[m]
        if acc[i] < 0:
            nl += 1
            acc[i] = nl
        if acc[i] > 0 and d[i] >= 0:
            t = i + DR[d[i]] * w + DC[d[i]]
            if acc[t] == -2:
                acc[t] = acc[i]
    down = np.zeros(nl + 1, np.int32)
    for m in range(n):  # downstream first
        i = order[m]
        if acc[i] > 0:
            if d[i] >= 0:
                t = i + DR[d[i]] * w + DC[d[i]]
                if acc[t] > 0 and acc[t] != acc[i]:
                    down[acc[i]] = acc[t]
        else:
            if d[i] >= 0:
                t = i + DR[d[i]] * w + DC[d[i]]
                acc[i] = acc[t]
                filled[i] = filled[t]
            else:
                filled[i] = np.nan
    return acc, down


@njit(cache=True)
def _euler(down):
    """Entry and exit times of a depth-first tour of the link forest (children = upstream links)."""
    nl = down.size
    cnt = np.zeros(nl + 1, np.int32)
    for l in range(1, nl):
        cnt[down[l] + 1] += 1
    for l in range(nl):
        cnt[l + 1] += cnt[l]
    kids = np.empty(max(nl - 1, 1), np.int32)
    pos = cnt.copy()
    for l in range(1, nl):
        kids[pos[down[l]]] = l
        pos[down[l]] += 1
    tin, tout = np.zeros(nl, np.int32), np.zeros(nl, np.int32)
    stack, nxt = np.empty(nl, np.int32), np.empty(nl, np.int32)
    t = 0
    for root in range(1, nl):
        if down[root] != 0:
            continue
        sp = 0
        stack[0], nxt[0] = root, cnt[root]
        tin[root] = t
        t += 1
        while sp >= 0:
            l = stack[sp]
            if nxt[sp] < cnt[l + 1]:
                ch = kids[nxt[sp]]
                nxt[sp] += 1
                sp += 1
                stack[sp], nxt[sp] = ch, cnt[ch]
                tin[ch] = t
                t += 1
            else:
                tout[l] = t
                sp -= 1
    return tin, tout


def drainage(z, valley_km2=VALLEY_KM2):
    """(lid, sz, tin, tout): link of each cell (0 = none), elevation of the valley cell it drains to,
    and the tour times of the link forest. All flat arrays of z."""
    filled, d, order = flood(z)
    lid, down = _links(filled, d, order, z.shape[1], valley_cells(valley_km2))
    del d, order
    tin, tout = _euler(down)
    return lid, filled, tin, tout


@njit(cache=True)
def _scan(z, lid, sz, tin, tout, xsec, drop, hand_min, fill, ea, eb, fa, fb):
    h, w = z.shape
    zf = z.ravel()
    ne = nf = 0
    for r in range(h):
        for c in range(w):
            for k in range(2):  # east and south neighbours
                r2, c2 = (r, c + 1) if k == 0 else (r + 1, c)
                if r2 >= h or c2 >= w:
                    continue
                a, b = r * w + c, r2 * w + c2
                la, lb = lid[a], lid[b]
                if la <= 0 or lb <= 0 or la == lb:
                    continue
                if (tin[la] <= tin[lb] < tout[la]) or (tin[lb] <= tin[la] < tout[lb]):
                    continue
                if zf[a] - sz[a] < hand_min or zf[b] - sz[b] < hand_min:
                    continue
                dr, dc = r2 - r, c2 - c
                ok = True
                ra, ca = r - xsec * dr, c - xsec * dc
                if 0 <= ra < h and 0 <= ca < w and zf[a] - z[ra, ca] < drop:
                    ok = False
                rb, cb = r2 + xsec * dr, c2 + xsec * dc
                if 0 <= rb < h and 0 <= cb < w and zf[b] - z[rb, cb] < drop:
                    ok = False
                # corners of the shared side
                if k == 0:
                    v0, v1 = r * (w + 1) + c + 1, (r + 1) * (w + 1) + c + 1
                else:
                    v0, v1 = (r + 1) * (w + 1) + c, (r + 1) * (w + 1) + c + 1
                if ok:
                    if fill:
                        ea[ne], eb[ne] = v0, v1
                    ne += 1
                else:
                    if fill:
                        fa[nf], fb[nf] = v0, v1
                    nf += 1
    return ne, nf


def crest_edges(z, lid, sz, tin, tout, xsec=XSEC_CELLS, drop=XSEC_DROP, hand_min=HAND_MIN):
    """Crest sides as pairs of corner ids (r * (w + 1) + c): those passing every test, and those failing only
    the cross-section (candidates to bridge flat stretches)."""
    e0 = np.empty(0, np.int32)
    ne, nf = _scan(z, lid, sz, tin, tout, xsec, drop, hand_min, False, e0, e0, e0, e0)
    ea, eb, fa, fb = (np.empty(n, np.int32) for n in (ne, ne, nf, nf))
    _scan(z, lid, sz, tin, tout, xsec, drop, hand_min, True, ea, eb, fa, fb)
    return np.c_[ea, eb], np.c_[fa, fb]
