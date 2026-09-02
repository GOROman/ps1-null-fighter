"""Mesh optimisation for the PS1 converter.

decimate()  quadric-error-metric edge collapse on shared positions.  Each
            surviving triangle corner keeps its original UV so the old
            texture can still be sampled through it.
reatlas()   xatlas re-parametrisation: one contiguous UV chart set with
            padding, so vertices are shared across triangles and texels do
            not bleed between islands.
bake()      resample the original texture into the new atlas.

All functions are pure numpy (+ xatlas for reatlas) so they run on the build
host as part of the asset pipeline.
"""
import heapq

import numpy as np


# --------------------------------------------------------------------------
# quadric edge collapse
# --------------------------------------------------------------------------
def _face_normal(p0, p1, p2):
    n = np.cross(p1 - p0, p2 - p0)
    l = np.linalg.norm(n)
    return n / l if l > 0 else None


def decimate(P, T, target_tris, verbose=True, importance=None):
    """P: (n,3) float positions (modified copy returned), T: (m,3) int.
    importance: optional per-position weight (>1 protects a region, e.g. the
    face, so its edges collapse last).
    Returns (P2, T2, keep) where keep[i] is the index into T of the original
    triangle that survived as T2[i] (corner order preserved)."""
    P = P.astype(np.float64).copy()
    T = T.astype(np.int64).copy()
    n, m = len(P), len(T)
    alive = np.ones(m, dtype=bool)
    Q = np.zeros((n, 4, 4))
    for t in range(m):
        a, b, c = T[t]
        nrm = _face_normal(P[a], P[b], P[c])
        if nrm is None:
            alive[t] = False
            continue
        d = -np.dot(nrm, P[a])
        pl = np.append(nrm, d)
        K = np.outer(pl, pl)
        Q[a] += K
        Q[b] += K
        Q[c] += K
    if importance is not None:
        Q *= np.asarray(importance, dtype=np.float64)[:, None, None]
    # vertex -> set of triangle ids
    vt = [set() for _ in range(n)]
    for t in range(m):
        if alive[t]:
            for v in T[t]:
                vt[v].add(t)
    # boundary edges get a strong perpendicular-plane quadric
    edge_count = {}
    for t in range(m):
        if not alive[t]:
            continue
        a, b, c = T[t]
        for u, v in ((a, b), (b, c), (c, a)):
            edge_count[(min(u, v), max(u, v))] = edge_count.get((min(u, v), max(u, v)), 0) + 1
    for (u, v), cnt in edge_count.items():
        if cnt == 1:
            t = next(iter(vt[u] & vt[v]))
            a, b, c = T[t]
            fn = _face_normal(P[a], P[b], P[c])
            e = P[v] - P[u]
            nrm = np.cross(e, fn)
            l = np.linalg.norm(nrm)
            if l == 0:
                continue
            nrm /= l
            pl = np.append(nrm, -np.dot(nrm, P[u]))
            K = np.outer(pl, pl) * 1000.0
            Q[u] += K
            Q[v] += K

    version = np.zeros(n, dtype=np.int64)

    def edge_cost(u, v):
        q = Q[u] + Q[v]
        cands = [P[u], P[v], (P[u] + P[v]) * 0.5]
        A = q[:3, :3]
        if abs(np.linalg.det(A)) > 1e-12:
            try:
                x = np.linalg.solve(A, -q[:3, 3])
                # reject wild solutions far away from the edge
                if np.linalg.norm(x - cands[2]) < 2.0 * np.linalg.norm(P[u] - P[v]) + 1e-9:
                    cands.append(x)
            except np.linalg.LinAlgError:
                pass
        best, bc = None, None
        for c in cands:
            h = np.append(c, 1.0)
            cost = float(h @ q @ h)
            if best is None or cost < best:
                best, bc = cost, c
        return best, bc

    heap = []
    for (u, v) in edge_count:
        cost, pos = edge_cost(u, v)
        heapq.heappush(heap, (cost, u, v, version[u], version[v], pos))

    ntri = int(alive.sum())
    collapsed = 0
    while ntri > target_tris and heap:
        cost, u, v, vu, vv, pos = heapq.heappop(heap)
        if version[u] != vu or version[v] != vv or not vt[u] or not vt[v]:
            continue
        if not (vt[u] & vt[v]):
            continue
        # collapse u into v, moving v to pos.  Reject if any surviving face flips.
        shared = vt[u] & vt[v]
        affected = (vt[u] | vt[v]) - shared
        ok = True
        for t in affected:
            a, b, c = T[t]
            before = _face_normal(P[a], P[b], P[c])
            pa, pb, pc = (pos if x in (u, v) else P[x] for x in (a, b, c))
            after = _face_normal(pa, pb, pc)
            if before is None or after is None or np.dot(before, after) < 0.2:
                ok = False
                break
        if not ok:
            continue
        for t in shared:
            alive[t] = False
            for x in T[t]:
                vt[x].discard(t)
        ntri -= len(shared)
        for t in vt[u]:
            T[t][T[t] == u] = v
            vt[v].add(t)
        vt[u] = set()
        P[v] = pos
        Q[v] = Q[u] + Q[v]
        version[u] += 1
        version[v] += 1
        collapsed += 1
        # re-evaluate edges around v
        nbrs = set()
        for t in vt[v]:
            for x in T[t]:
                if x != v:
                    nbrs.add(int(x))
        for w in nbrs:
            c2, p2 = edge_cost(v, w)
            heapq.heappush(heap, (c2, v, w, version[v], version[w], p2))
    keep = np.where(alive)[0]
    if verbose:
        print("decimate: %d -> %d triangles (%d collapses)" % (m, len(keep), collapsed))
    return P, T[keep], keep


# --------------------------------------------------------------------------
# xatlas re-parametrisation
# --------------------------------------------------------------------------
def reatlas(P, T, resolution=256, padding=2, scale=None):
    """Returns (vmapping, new_tris, uvs) - vmapping maps new vertex -> old
    position index, uvs in [0,1].  Triangle order is preserved.
    scale: optional per-position factor; xatlas hands out atlas area by 3D
    surface area, so enlarging a region (the face) gives it more texels."""
    import xatlas
    Pin = P.astype(np.float64)
    if scale is not None:
        Pin = Pin * np.asarray(scale, dtype=np.float64)[:, None]
    area3d = sum(0.5 * np.linalg.norm(np.cross(Pin[b] - Pin[a], Pin[c] - Pin[a])) for a, b, c in T)

    def run(tpu):
        atlas = xatlas.Atlas()
        atlas.add_mesh(Pin.astype(np.float32), T.astype(np.uint32))
        co = xatlas.ChartOptions()
        po = xatlas.PackOptions()
        po.resolution = 0 if tpu else resolution
        po.texels_per_unit = float(tpu)
        po.padding = padding
        po.bilinear = True
        po.blockAlign = False
        po.bruteForce = True
        atlas.generate(chart_options=co, pack_options=po)
        return atlas

    # xatlas treats `resolution` as a hint and often overshoots; search for a
    # texel density whose atlas actually fits so the texture is not shrunk
    tpu = np.sqrt(resolution * resolution * 0.75 / area3d)
    best = None
    for _ in range(6):
        atlas = run(tpu)
        big = max(atlas.width, atlas.height)
        if big <= resolution:
            if best is None or big > max(best[0].width, best[0].height):
                best = (atlas, tpu)
            if big >= resolution * 0.95:
                break
            tpu *= min(1.15, (resolution / big) * 0.98)
        else:
            tpu *= (resolution / big) * 0.97
    if best is None:
        best = (atlas, tpu)
    atlas = best[0]
    vmapping, indices, uvs = atlas[0]
    uvs = np.asarray(uvs, dtype=np.float64)
    if uvs.max() > 1.5:   # some builds return pixel units
        uvs = uvs / np.array([atlas.width, atlas.height])
    # the atlas may be a little smaller than the texture: map it 1:1 into the
    # top-left corner so the texel density is what xatlas packed
    uvs = uvs * np.array([atlas.width, atlas.height]) / resolution
    indices = np.asarray(indices, dtype=np.int64)
    assert len(indices) == len(T), "xatlas changed the triangle count"
    print("reatlas: %d verts -> %d verts, %d charts, %dx%d, utilization %.0f%%" % (
        len(P), len(vmapping), atlas.chart_count, atlas.width, atlas.height, float(np.atleast_1d(atlas.utilization)[0]) * 100))
    return np.asarray(vmapping, dtype=np.int64), indices, uvs


# --------------------------------------------------------------------------
# texture bake
# --------------------------------------------------------------------------
def _sample_bilinear(img, u, v):
    """img: (H,W,3) float, u/v arrays in [0,1] (v = 0 at the top row)."""
    H, W = img.shape[:2]
    x = np.clip(u * W - 0.5, 0, W - 1)
    y = np.clip(v * H - 0.5, 0, H - 1)
    x0 = np.floor(x).astype(int)
    y0 = np.floor(y).astype(int)
    x1 = np.minimum(x0 + 1, W - 1)
    y1 = np.minimum(y0 + 1, H - 1)
    fx = (x - x0)[:, None]
    fy = (y - y0)[:, None]
    return (img[y0, x0] * (1 - fx) * (1 - fy) + img[y0, x1] * fx * (1 - fy) +
            img[y1, x0] * (1 - fx) * fy + img[y1, x1] * fx * fy)


def bake(new_uv_tris, old_uv_tris, src_img, size=256, dilate=4):
    """new_uv_tris / old_uv_tris: (m,3,2) arrays of UVs (u right, v down in
    [0,1]).  src_img: PIL image.  Returns a PIL RGB image of size x size."""
    from PIL import Image
    src = np.asarray(src_img.convert("RGB"), dtype=np.float64)
    out = np.zeros((size, size, 3))
    cov = np.zeros((size, size), dtype=bool)
    for tri_new, tri_old in zip(new_uv_tris, old_uv_tris):
        pts = tri_new * size
        x0 = max(int(np.floor(pts[:, 0].min())) - 1, 0)
        x1 = min(int(np.ceil(pts[:, 0].max())) + 1, size - 1)
        y0 = max(int(np.floor(pts[:, 1].min())) - 1, 0)
        y1 = min(int(np.ceil(pts[:, 1].max())) + 1, size - 1)
        if x1 < x0 or y1 < y0:
            continue
        xs, ys = np.meshgrid(np.arange(x0, x1 + 1) + 0.5, np.arange(y0, y1 + 1) + 0.5)
        xs = xs.ravel()
        ys = ys.ravel()
        (ax, ay), (bx, by), (cx, cy) = pts
        det = (bx - ax) * (cy - ay) - (cx - ax) * (by - ay)
        if abs(det) < 1e-12:
            continue
        l1 = ((bx - xs) * (cy - ys) - (cx - xs) * (by - ys)) / det
        l2 = ((cx - xs) * (ay - ys) - (ax - xs) * (cy - ys)) / det
        l3 = 1.0 - l1 - l2
        # slightly expanded coverage (half a texel) so edges are filled
        eps = -0.75 / max(np.sqrt(abs(det)), 1.0)
        inside = (l1 >= eps) & (l2 >= eps) & (l3 >= eps)
        if not inside.any():
            continue
        l1, l2, l3 = l1[inside], l2[inside], l3[inside]
        l1c, l2c, l3c = np.clip(l1, 0, 1), np.clip(l2, 0, 1), np.clip(l3, 0, 1)
        s = l1c + l2c + l3c
        l1c, l2c, l3c = l1c / s, l2c / s, l3c / s
        ou = l1c * tri_old[0, 0] + l2c * tri_old[1, 0] + l3c * tri_old[2, 0]
        ov = l1c * tri_old[0, 1] + l2c * tri_old[1, 1] + l3c * tri_old[2, 1]
        col = _sample_bilinear(src, ou, ov)
        px = xs[inside].astype(int)
        py = ys[inside].astype(int)
        # texels strictly inside win over expanded ones from neighbours
        strict = (l1 >= 0) & (l2 >= 0) & (l3 >= 0)
        write = strict | ~cov[py, px]
        out[py[write], px[write]] = col[write]
        cov[py[write], px[write]] = True
    # dilate: fill uncovered texels from covered neighbours
    for _ in range(dilate):
        if cov.all():
            break
        acc = np.zeros_like(out)
        cnt = np.zeros((size, size))
        for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)):
            sh = np.roll(np.roll(out, dy, 0), dx, 1)
            sc = np.roll(np.roll(cov, dy, 0), dx, 1)
            acc += sh * sc[:, :, None]
            cnt += sc
        fill = (~cov) & (cnt > 0)
        out[fill] = acc[fill] / cnt[fill][:, None]
        cov |= fill
    out[~cov] = (128, 128, 128)
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), "RGB")


# --------------------------------------------------------------------------
# hidden surface removal
# --------------------------------------------------------------------------
def remove_hidden(P, T, max_dist, verbose=True, occluders=None):
    """Drop triangles that are covered by another surface close in front of
    them (skirt under the jacket, body under clothes...): a ray from the
    centroid along the face normal hits another triangle within max_dist.
    Returns the boolean keep mask."""
    P = np.asarray(P, dtype=np.float64)
    T = np.asarray(T)
    A, B, C = P[T[:, 0]], P[T[:, 1]], P[T[:, 2]]
    N = np.cross(B - A, C - A)
    L = np.linalg.norm(N, axis=1)
    ok = L > 1e-12
    N[ok] /= L[ok][:, None]
    cen = (A + B + C) / 3.0
    E1, E2 = B - A, C - A
    keep = np.ones(len(T), dtype=bool)
    eps = max_dist * 0.02
    for i in range(len(T)):
        if not ok[i]:
            continue
        o = cen[i] + N[i] * eps
        d = N[i]
        # Moller-Trumbore against every other triangle
        pvec = np.cross(d, E2)
        det = np.einsum("ij,ij->i", E1, pvec)
        valid = np.abs(det) > 1e-12
        valid[i] = False
        if occluders is not None:
            valid &= occluders
        inv = np.zeros(len(T))
        inv[valid] = 1.0 / det[valid]
        tvec = o - A
        u = np.einsum("ij,ij->i", tvec, pvec) * inv
        qvec = np.cross(tvec, E1)
        v = np.einsum("j,ij->i", d, qvec) * inv
        t = np.einsum("ij,ij->i", E2, qvec) * inv
        # only a roughly parallel cover counts (a crossing surface does not)
        facing = (N @ N[i]) > 0.3
        hit = valid & facing & (u >= -1e-6) & (v >= -1e-6) & (u + v <= 1 + 1e-6) & (t > 0) & (t <= max_dist)
        if hit.any():
            keep[i] = False
    if verbose:
        print("remove_hidden: dropped %d of %d triangles" % (int((~keep).sum()), len(T)))
    return keep


# --------------------------------------------------------------------------
# triangles -> quad edges
# --------------------------------------------------------------------------
def quad_edges(P, T):
    """Greedily pair adjacent triangles into quads (most coplanar, squarest
    first) and return the edge list without the removed diagonals.
    T: (m,3) vertex indices (any numbering); edges are returned as (a, b)
    tuples of those indices."""
    P = np.asarray(P, dtype=np.float64)
    T = np.asarray(T)
    N = np.cross(P[T[:, 1]] - P[T[:, 0]], P[T[:, 2]] - P[T[:, 0]])
    L = np.linalg.norm(N, axis=1)
    N = N / np.where(L > 0, L, 1)[:, None]
    owners = {}
    for ti, t in enumerate(T):
        for k in range(3):
            a, b = int(t[k]), int(t[(k + 1) % 3])
            owners.setdefault((min(a, b), max(a, b)), []).append(ti)

    def quad_score(e, t0, t1):
        # opposite corners of the two triangles
        c0 = [int(x) for x in T[t0] if int(x) not in e][0]
        c1 = [int(x) for x in T[t1] if int(x) not in e][0]
        quad = [e[0], c0, e[1], c1]
        angles = []
        for i in range(4):
            a, b, c = P[quad[i - 1]], P[quad[i]], P[quad[(i + 1) % 4]]
            u, v = a - b, c - b
            lu, lv = np.linalg.norm(u), np.linalg.norm(v)
            if lu == 0 or lv == 0:
                return None
            angles.append(np.degrees(np.arccos(np.clip(np.dot(u, v) / (lu * lv), -1, 1))))
        if max(angles) > 178:            # would not be convex
            return None
        squareness = sum(abs(a - 90) for a in angles) / 4
        planarity = np.degrees(np.arccos(np.clip(np.dot(N[t0], N[t1]), -1, 1)))
        return squareness + 2 * planarity

    cands = []
    for e, ts in owners.items():
        if len(ts) == 2:
            sc = quad_score(e, ts[0], ts[1])
            if sc is not None:
                cands.append((sc, e, ts[0], ts[1]))
    cands.sort()
    paired = set()
    removed = set()
    for sc, e, t0, t1 in cands:
        if t0 in paired or t1 in paired:
            continue
        paired.update((t0, t1))
        removed.add(e)
    edges = [e for e in owners if e not in removed]
    print("quad_edges: %d triangles -> %d quads + %d triangles, %d edges" % (
        len(T), len(paired) // 2, len(T) - len(paired), len(edges)))
    return edges


# --------------------------------------------------------------------------
def draw_uv_check(img, uvs, tris, path, scale=3, highlight=None):
    """Write the atlas with the UV wireframe on top (highlight: per-triangle
    bool mask drawn in a second colour)."""
    from PIL import Image, ImageDraw
    size = img.size[0] * scale
    out = img.resize((size, size), Image.NEAREST).convert("RGB")
    d = ImageDraw.Draw(out)
    for i, t in enumerate(tris):
        pts = [(uvs[k][0] * size, uvs[k][1] * size) for k in t]
        col = (255, 255, 0) if highlight is not None and highlight[i] else (0, 255, 255)
        d.polygon(pts, outline=col)
    out.save(path)
    print("wrote %s" % path)


def vertex_normals(P, T):
    N = np.zeros_like(P)
    for a, b, c in T:
        n = np.cross(P[b] - P[a], P[c] - P[a])   # area weighted
        N[a] += n
        N[b] += n
        N[c] += n
    l = np.linalg.norm(N, axis=1)
    l[l == 0] = 1.0
    return N / l[:, None]
