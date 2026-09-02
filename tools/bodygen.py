"""Procedural body: rebuild every rigid part except the head as a clean
lathe (elliptical cross sections along the part's axis) whose profile is
measured from the original mesh, and texture it by projecting the original
texture onto the new surface (nearest triangle of the same part).

Coordinates: mesh local (FBX Y-up).  Parts are the auto rig bones
(autorig.BONES); the head keeps the original geometry.
"""
import math

import numpy as np

import autorig as ar

# atlas layout (256x256): torso parts in the left column, limbs in 64x64 cells
ATLAS = 256
LAYOUT = {                     # name: (u0, v0, w, h) in texels
    "root":       (0, 0, 128, 64),
    "spine":      (0, 64, 128, 64),
    "upperarm_L": (128, 0, 64, 64), "upperarm_R": (192, 0, 64, 64),
    "forearm_L":  (128, 64, 64, 64), "forearm_R":  (192, 64, 64, 64),
    "thigh_L":    (0, 128, 64, 64), "thigh_R":    (64, 128, 64, 64),
    "shin_L":     (0, 192, 64, 64), "shin_R":     (64, 192, 64, 64),
}
HAIR_RECT = (128, 144, 128, 112)     # xatlas-packed hair goes here
SKIN_RECT = (128, 128, 64, 16)       # solid skin colour (inner body layer)
UNDER_RECT = (192, 128, 64, 16)      # solid underwear colour
CLOTH_PARTS = ()                     # lathes that are cloth (cut by the monkey); the skirt only for now
INNER_SCALE = 0.86                   # inner body layer radius relative to the jacket
SEGMENTS = {"root": 10, "spine": 10}   # default 6
ROWS = {"root": 4, "spine": 5, "shin_L": 4, "shin_R": 4, "forearm_L": 4, "forearm_R": 4}   # default 3


def measure_profile(pts, rows, pad=0.02):
    """pts (n,3) of one part -> list of (y, cx, cz, rx, rz) for rows+1 rings
    from the bottom to the top of the part."""
    y0, y1 = pts[:, 1].min(), pts[:, 1].max()
    ys = np.linspace(y0, y1, rows + 1)
    band = (y1 - y0) / rows * 0.6 + 1e-9
    prof = []
    for y in ys:
        sl = pts[np.abs(pts[:, 1] - y) <= band]
        if len(sl) < 6:
            prof.append(None)
            continue
        cx, cz = np.median(sl[:, 0]), np.median(sl[:, 2])
        rx = np.percentile(np.abs(sl[:, 0] - cx), 92)
        rz = np.percentile(np.abs(sl[:, 2] - cz), 92)
        prof.append((y, cx, cz, max(rx, 1e-4), max(rz, 1e-4)))
    # fill gaps from neighbours
    for i in range(len(prof)):
        if prof[i] is None:
            j = next((k for k in list(range(i - 1, -1, -1)) + list(range(i + 1, len(prof))) if prof[k] is not None), None)
            prof[i] = (ys[i],) + prof[j][1:] if j is not None else (ys[i], 0, 0, 0.02, 0.02)
    return prof


def lathe(prof, segments, uv_rect, bone, taper_ends=(False, False), scale=1.0, flat_uv=False):
    """Rings through the profile.  Returns (positions, verts, tris, rings,
    quads) where verts = (pos_index, normal, uv[0..1 atlas]), tris index
    into verts, rings[row] lists vertex indices around, quads = (v0, v1,
    v2, v3, tri0, tri1) per cell.  flat_uv: every vertex maps to the
    centre of uv_rect (solid colour)."""
    u0, v0, w, h = uv_rect
    pos, verts, tris = [], [], []
    rings = []
    nrows = len(prof) - 1
    for r, (y, cx, cz, rx, rz) in enumerate(prof):
        f = r / nrows
        rx, rz = rx * scale, rz * scale
        if r == 0 and taper_ends[0]:
            rx, rz = rx * 0.6, rz * 0.6
        if r == nrows and taper_ends[1]:
            rx, rz = rx * 0.6, rz * 0.6
        ring = []
        for i in range(segments + 1):
            a = 2 * math.pi * (i % segments) / segments
            x, z = cx + rx * math.cos(a), cz + rz * math.sin(a)
            n = np.array([math.cos(a) / rx, 0.0, math.sin(a) / rz])
            n /= np.linalg.norm(n)
            pos.append(np.array([x, y, z]))
            if flat_uv:
                u, v = (u0 + w * 0.5) / ATLAS, (v0 + h * 0.5) / ATLAS
            else:
                u = (u0 + 0.5 + (w - 1) * i / segments) / ATLAS
                v = (v0 + 0.5 + (h - 1) * (1 - f)) / ATLAS      # top of the part at the top of the rect
            verts.append((len(pos) - 1, tuple(n), (u, 1.0 - v)))
            ring.append(len(verts) - 1)
        rings.append(ring)
    quads = []
    for r in range(nrows):
        for i in range(segments):
            a, b = rings[r][i], rings[r][i + 1]
            c, d = rings[r + 1][i], rings[r + 1][i + 1]
            # outward facing: (a, c, b) with y increasing upwards in FBX
            t0 = len(tris)
            tris += [(a, c, b), (b, c, d)]
            quads.append((a, b, d, c, t0, t0 + 1))
    return pos, verts, tris, rings, quads


# --------------------------------------------------------------------------
# projection bake
# --------------------------------------------------------------------------
def _closest_on_triangles(p, A, B, C):
    """closest points of p (3,) on triangles (m,3) -> (dist2 (m,), bary (m,3))
    (Ericson, vectorised)"""
    ab, ac, ap = B - A, C - A, p - A
    d1 = np.einsum("ij,ij->i", ab, ap)
    d2 = np.einsum("ij,ij->i", ac, ap)
    bp = p - B
    d3 = np.einsum("ij,ij->i", ab, bp)
    d4 = np.einsum("ij,ij->i", ac, bp)
    cp = p - C
    d5 = np.einsum("ij,ij->i", ab, cp)
    d6 = np.einsum("ij,ij->i", ac, cp)
    vc = d1 * d4 - d3 * d2
    vb = d5 * d2 - d1 * d6
    va = d3 * d6 - d5 * d4
    m = len(A)
    bary = np.zeros((m, 3))
    # vertex regions
    rA = (d1 <= 0) & (d2 <= 0)
    rB = (d3 >= 0) & (d4 <= d3)
    rC = (d6 >= 0) & (d5 <= d6)
    # edge regions
    eAB = (~rA) & (~rB) & (~rC) & (vc <= 0) & (d1 >= 0) & (d3 <= 0)
    eAC = (~rA) & (~rB) & (~rC) & (vb <= 0) & (d2 >= 0) & (d6 <= 0)
    eBC = (~rA) & (~rB) & (~rC) & (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0)
    inside = ~(rA | rB | rC | eAB | eAC | eBC)
    bary[rA] = (1, 0, 0)
    bary[rB] = (0, 1, 0)
    bary[rC] = (0, 0, 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = d1 / (d1 - d3)
        bary[eAB] = np.stack([1 - t, t, np.zeros_like(t)], 1)[eAB]
        t = d2 / (d2 - d6)
        bary[eAC] = np.stack([1 - t, np.zeros_like(t), t], 1)[eAC]
        t = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        bary[eBC] = np.stack([np.zeros_like(t), 1 - t, t], 1)[eBC]
        denom = va + vb + vc
        v = vb / denom
        w = vc / denom
        bary[inside] = np.stack([1 - v - w, v, w], 1)[inside]
    q = A * bary[:, :1] + B * bary[:, 1:2] + C * bary[:, 2:3]
    d = q - p
    return np.einsum("ij,ij->i", d, d), bary


def bake_part(img_out, rect, prof, segments, P, T, cuv, src, max_dist):
    """Fill rect of img_out by projecting the original part triangles (T,
    corner uvs cuv, image convention) onto the lathe surface."""
    from meshopt import _sample_bilinear
    u0, v0, w, h = rect
    if len(T) == 0:
        return
    A, B, C = P[T[:, 0]], P[T[:, 1]], P[T[:, 2]]
    nrows = len(prof) - 1
    ys = np.array([p[0] for p in prof])
    for ty in range(h):
        f = 1 - (ty + 0.5 - 0.5) / (h - 1)          # 1 = top ring
        f = min(max(f, 0.0), 1.0)
        # interpolate the profile
        pos = f * nrows
        r0 = min(int(pos), nrows - 1)
        t = pos - r0
        y, cx, cz, rx, rz = [(1 - t) * prof[r0][k] + t * prof[r0 + 1][k] for k in range(5)]
        for tx in range(w):
            a = 2 * math.pi * (tx + 0.5 - 0.5) / (w - 1)
            p = np.array([cx + rx * math.cos(a), y, cz + rz * math.sin(a)])
            d2, bary = _closest_on_triangles(p, A, B, C)
            i = int(np.argmin(d2))
            if d2[i] > max_dist * max_dist:
                col = (90, 90, 90)
            else:
                uv = cuv[i][0] * bary[i][0] + cuv[i][1] * bary[i][1] + cuv[i][2] * bary[i][2]
                col = _sample_bilinear(src, np.array([uv[0]]), np.array([uv[1]]))[0]
            img_out[v0 + ty, u0 + tx] = col


def generate_body(P, T, cuv, pos_bone, texture, verbose=True):
    """P/T/cuv/pos_bone: the split original mesh.  Returns
    (positions, verts, tris, image(256x256 uint8), parts_bones, keep) for
    every part except the head; verts reference the returned positions.
    keep: mask of original triangles to keep as they are (the hands)."""
    src = np.asarray(texture.convert("RGB"), dtype=np.float64)
    img = np.full((ATLAS, ATLAS, 3), 90, dtype=np.uint8)
    for rect, col in ((SKIN_RECT, (236, 196, 160)), (UNDER_RECT, (245, 245, 250))):
        u0, v0, w, h = rect
        img[v0:v0 + h, u0:u0 + w] = col
    all_pos, all_verts, all_tris, bones = [], [], [], []
    cloth = {"quads": [], "edges": [], "rings": {}}      # indices into the returned verts / tris
    tri_bone = pos_bone[T[:, 0]]
    ymin = P[:, 1].min()
    H = P[:, 1].max() - ymin
    wrist = ymin + H * ar.RIG["wrist_y"]
    keep = np.zeros(len(T), dtype=bool)
    for name, _ in ar.BONES:
        if name == "head" or name not in LAYOUT:
            continue
        b = ar.NAME[name]
        pts = P[pos_bone == b]
        Tb = T[tri_bone == b]
        if name.startswith("forearm"):
            # the hand (below the wrist) keeps the original mesh
            hand = (tri_bone == b) & (P[T][:, :, 1] < wrist + 0.005 * H).all(axis=1)
            keep |= hand
            pts = pts[pts[:, 1] >= wrist - 0.01 * H]
            Tb = T[(tri_bone == b) & ~hand]
        if len(pts) < 12 or len(Tb) == 0:
            if verbose:
                print("bodygen: %s has no geometry, skipped" % name)
            continue
        rows = ROWS.get(name, 3)
        segs = SEGMENTS.get(name, 6)
        prof = measure_profile(pts, rows)
        pos, verts, tris, rings, quads = lathe(prof, segs, LAYOUT[name], b)
        base = len(all_pos)
        vbase = len(all_verts)
        tbase = len(all_tris)
        all_pos += pos
        all_verts += [(i + base, n, uv) for (i, n, uv) in verts]
        all_tris += [(a + vbase, c + vbase, d + vbase) for (a, c, d) in tris]
        bones += [b] * len(pos)
        if name in CLOTH_PARTS:
            cloth["quads"] += [(q[0] + vbase, q[1] + vbase, q[2] + vbase, q[3] + vbase, q[4] + tbase, q[5] + tbase)
                               for q in quads]
            for ring in rings:
                cloth["edges"] += [(ring[i] + vbase, ring[i + 1] + vbase) for i in range(segs)]
            for r in range(rows):
                cloth["edges"] += [(rings[r][i] + vbase, rings[r + 1][i] + vbase) for i in range(segs)]
            cloth["rings"][name] = [[v + vbase for v in ring[:segs]] for ring in rings]
            # inner body layer (skin, underwear band on the upper torso)
            rect = UNDER_RECT if name == "spine" else SKIN_RECT
            iprof = [prof[0], prof[len(prof) // 2], prof[-1]]           # 2 rows are enough for a solid layer
            ipos, iverts, itris, _, _ = lathe(iprof, segs, rect, b, scale=INNER_SCALE, flat_uv=True)
            ibase, ivbase = len(all_pos), len(all_verts)
            all_pos += ipos
            all_verts += [(i + ibase, n, uv) for (i, n, uv) in iverts]
            all_tris += [(a + ivbase, c + ivbase, d + ivbase) for (a, c, d) in itris]
            bones += [b] * len(ipos)
        bake_part(img, LAYOUT[name], prof, segs, P, Tb, cuv[tri_bone == b], src, 0.08 * H)
        if verbose:
            print("bodygen: %-11s %3d verts %3d tris  profile rx %.3f..%.3f" % (
                name, len(pos), len(tris), min(p[3] for p in prof), max(p[3] for p in prof)))
    return np.array(all_pos), all_verts, all_tris, img, np.array(bones, dtype=np.int64), keep, cloth
