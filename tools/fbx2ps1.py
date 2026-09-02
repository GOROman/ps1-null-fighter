#!/usr/bin/env python3
"""Convert a skinned, animated FBX character into PlayStation 1 runtime data.
A plain static mesh (no bones / no animation) is accepted too and exported
as one root bone with a single-frame "static" animation.

Outputs
  model.bin  mesh (rigidly skinned, bone-local vertices), skeleton, sampled
             animations (quaternion per bone per frame)
  model.tim  256x256 8bpp CLUT texture

The binary layout mirrors src/model.h.  Coordinate system: FBX Z-up (right
handed) is rotated into PS1 space (X right, Y down, Z into the screen) with
1.0 FBX unit == UNIT_SCALE PS1 units.  Rotations are 4096 == 1.0 fixed point.

Usage
  fbx2ps1.py character.fbx --texture basecolor.jpg --out-bin model.bin --out-tim model.tim
  fbx2ps1.py character.fbx --preview idle 0 preview.png   # wireframe check
"""
import argparse
import math
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fbxparse import (FBX_TIME_PER_SEC, FbxFile, obj_id, obj_name,  # noqa: E402
                      obj_subclass, properties70, prop_scalar, prop_vec3)

UNIT_SCALE = 4096          # PS1 units per FBX unit (1 m character -> 4096, well inside int16)
FIX_ONE = 4096             # fixed point 1.0
SAMPLE_FPS = 15            # animation sample rate stored in the file
MAGIC = b"PS1M"

# FBX world -> PS1 (X right, Y down, Z into screen).  Both are proper
# rotations (det +1) so triangle winding is preserved.
C_ZUP_TO_PS1 = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64)   # (x, y, z) -> (x, -z, y)
C_YUP_TO_PS1 = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float64)  # (x, y, z) -> (x, -y, -z)
C_FBX_TO_PS1 = C_YUP_TO_PS1


# --------------------------------------------------------------------------
# small matrix helpers (column vectors, 4x4 float64)
# --------------------------------------------------------------------------
def mat_t(v):
    m = np.eye(4)
    m[:3, 3] = v
    return m


def mat_s(v):
    m = np.eye(4)
    m[0, 0], m[1, 1], m[2, 2] = v
    return m


def rot_axis(axis, deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    m = np.eye(4)
    if axis == 0:
        m[1, 1], m[1, 2], m[2, 1], m[2, 2] = c, -s, s, c
    elif axis == 1:
        m[0, 0], m[0, 2], m[2, 0], m[2, 2] = c, s, -s, c
    else:
        m[0, 0], m[0, 1], m[1, 0], m[1, 1] = c, -s, s, c
    return m


ROT_ORDERS = {
    0: (0, 1, 2),  # XYZ
    1: (0, 2, 1),  # XZY
    2: (1, 2, 0),  # YZX
    3: (1, 0, 2),  # YXZ
    4: (2, 0, 1),  # ZXY
    5: (2, 1, 0),  # ZYX
}


def mat_euler(deg3, order=0):
    """FBX Euler rotation.  order XYZ means X applied first, so the column
    vector matrix is Rz * Ry * Rx."""
    seq = ROT_ORDERS.get(order, ROT_ORDERS[0])
    m = np.eye(4)
    for axis in seq:  # first applied axis ends up rightmost
        m = rot_axis(axis, deg3[axis]) @ m
    return m


def mat_from_fbx_array(a):
    """FBX stores 4x4 matrices row-major for row vectors; transpose to
    column-vector convention."""
    return np.array(a, dtype=np.float64).reshape(4, 4).T


def mat_to_quat(m):
    """Rotation part of m (assumed orthonormal) -> quaternion (x, y, z, w)."""
    r = m[:3, :3]
    tr = r[0, 0] + r[1, 1] + r[2, 2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x = (r[2, 1] - r[1, 2]) / s
        y = (r[0, 2] - r[2, 0]) / s
        z = (r[1, 0] - r[0, 1]) / s
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = math.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2
        w = (r[2, 1] - r[1, 2]) / s
        x = 0.25 * s
        y = (r[0, 1] + r[1, 0]) / s
        z = (r[0, 2] + r[2, 0]) / s
    elif r[1, 1] > r[2, 2]:
        s = math.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2
        w = (r[0, 2] - r[2, 0]) / s
        x = (r[0, 1] + r[1, 0]) / s
        y = 0.25 * s
        z = (r[1, 2] + r[2, 1]) / s
    else:
        s = math.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2
        w = (r[1, 0] - r[0, 1]) / s
        x = (r[0, 2] + r[2, 0]) / s
        y = (r[1, 2] + r[2, 1]) / s
        z = 0.25 * s
    q = np.array([x, y, z, w])
    return q / np.linalg.norm(q)


def quat_to_mat3(q):
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def orthonormalize(m):
    """Strip scale/shear from the 3x3 part; return (rotation4x4, scale3)."""
    r = m[:3, :3].copy()
    scale = np.linalg.norm(r, axis=0)
    u, _, vt = np.linalg.svd(r)
    rot = u @ vt
    if np.linalg.det(rot) < 0:
        u[:, -1] *= -1
        rot = u @ vt
    out = np.eye(4)
    out[:3, :3] = rot
    out[:3, 3] = m[:3, 3]
    return out, scale


# --------------------------------------------------------------------------
# FBX scene extraction
# --------------------------------------------------------------------------
class ModelNode:
    def __init__(self, node):
        self.node = node
        self.id = obj_id(node)
        self.name = obj_name(node)
        self.subclass = obj_subclass(node)
        self.parent = None
        self.children = []
        p = properties70(node)
        self.props = p
        self.lcl_t = prop_vec3(p, "Lcl Translation")
        self.lcl_r = prop_vec3(p, "Lcl Rotation")
        self.lcl_s = prop_vec3(p, "Lcl Scaling", (1.0, 1.0, 1.0))
        self.pre_r = prop_vec3(p, "PreRotation")
        self.post_r = prop_vec3(p, "PostRotation")
        self.rot_order = int(prop_scalar(p, "RotationOrder", 0))
        self.rot_pivot = prop_vec3(p, "RotationPivot")
        self.sca_pivot = prop_vec3(p, "ScalingPivot")
        self.rot_offset = prop_vec3(p, "RotationOffset")
        self.sca_offset = prop_vec3(p, "ScalingOffset")
        self.geo_t = prop_vec3(p, "GeometricTranslation")
        self.geo_r = prop_vec3(p, "GeometricRotation")
        self.geo_s = prop_vec3(p, "GeometricScaling", (1.0, 1.0, 1.0))
        self.bone_index = -1

    def local_matrix(self, t=None, r=None, s=None):
        t = self.lcl_t if t is None else t
        r = self.lcl_r if r is None else r
        s = self.lcl_s if s is None else s
        # FBX SDK: T * Roff * Rp * Rpre * R * Rpost^-1 * Rp^-1 * Soff * Sp * S * Sp^-1
        m = mat_t(t) @ mat_t(self.rot_offset) @ mat_t(self.rot_pivot)
        m = m @ mat_euler(self.pre_r) @ mat_euler(r, self.rot_order)
        m = m @ np.linalg.inv(mat_euler(self.post_r))
        m = m @ mat_t(-np.array(self.rot_pivot)) @ mat_t(self.sca_offset) @ mat_t(self.sca_pivot)
        m = m @ mat_s(s) @ mat_t(-np.array(self.sca_pivot))
        return m

    def geometric_matrix(self):
        return mat_t(self.geo_t) @ mat_euler(self.geo_r) @ mat_s(self.geo_s)


class Curve:
    def __init__(self, node):
        self.times = np.array(node.first("KeyTime").props[0], dtype=np.float64) / FBX_TIME_PER_SEC
        self.values = np.array(node.first("KeyValueFloat").props[0], dtype=np.float64)

    def eval(self, t):
        return float(np.interp(t, self.times, self.values))

    @property
    def end(self):
        return float(self.times[-1]) if len(self.times) else 0.0


class CurveNode:
    """AnimationCurveNode: 3 channels (d|X, d|Y, d|Z) with defaults."""

    def __init__(self, fbx, node):
        p = properties70(node)
        self.default = [float(prop_scalar(p, "d|X", 0.0)), float(prop_scalar(p, "d|Y", 0.0)),
                        float(prop_scalar(p, "d|Z", 0.0))]
        self.curves = [None, None, None]
        for cn, prop in fbx.children_of(obj_id(node), "AnimationCurve"):
            ch = {"d|X": 0, "d|Y": 1, "d|Z": 2}.get(prop)
            if ch is not None:
                self.curves[ch] = Curve(cn)

    def eval(self, t):
        return tuple(self.curves[i].eval(t) if self.curves[i] else self.default[i] for i in range(3))

    @property
    def end(self):
        return max((c.end for c in self.curves if c), default=0.0)


class AnimStack:
    def __init__(self, fbx, node, models_by_id):
        self.name = obj_name(node).replace("preset:biped:", "").replace("AnimStack", "").strip(" :")
        p = properties70(node)
        self.start = float(prop_scalar(p, "LocalStart", 0)) / FBX_TIME_PER_SEC
        self.stop = float(prop_scalar(p, "LocalStop", 0)) / FBX_TIME_PER_SEC
        self.tracks = {}  # model id -> {"T": CurveNode, "R": ..., "S": ...}
        end = 0.0
        for layer, _ in fbx.children_of(obj_id(node), "AnimationLayer"):
            for cn, _ in fbx.children_of(obj_id(layer), "AnimationCurveNode"):
                curve_node = CurveNode(fbx, cn)
                end = max(end, curve_node.end)
                for model, prop in fbx.parents_of(obj_id(cn), "Model"):
                    key = {"Lcl Translation": "T", "Lcl Rotation": "R", "Lcl Scaling": "S"}.get(prop)
                    if key:
                        self.tracks.setdefault(obj_id(model), {})[key] = curve_node
        if self.stop <= self.start:
            self.start, self.stop = 0.0, end

    @property
    def duration(self):
        return self.stop - self.start

    def local_matrix(self, model, t):
        tr = self.tracks.get(model.id, {})
        lt = tr["T"].eval(t) if "T" in tr else None
        lr = tr["R"].eval(t) if "R" in tr else None
        ls = tr["S"].eval(t) if "S" in tr else None
        return model.local_matrix(lt, lr, ls)


class Scene:
    def __init__(self, path):
        self.fbx = FbxFile(path)
        fbx = self.fbx
        self.models = {}
        for n in fbx.by_type("Model"):
            self.models[obj_id(n)] = ModelNode(n)
        for m in self.models.values():
            for parent, _ in fbx.parents_of(m.id, "Model"):
                m.parent = self.models[obj_id(parent)]
                m.parent.children.append(m)
        self.roots = [m for m in self.models.values() if m.parent is None]

        # skeleton: LimbNodes in hierarchy (depth-first) order
        self.bones = []
        def walk(m):
            if m.subclass == "LimbNode":
                m.bone_index = len(self.bones)
                self.bones.append(m)
            for c in m.children:
                walk(c)
        for r in self.roots:
            walk(r)
        # A plain static mesh (no skeleton) is exported as a single identity
        # root bone with one 1-frame "static" animation so the runtime needs
        # no special case.
        self.static = not self.bones
        if self.static:
            print("no LimbNode bones found: exporting as a rigid static mesh")

        # mesh
        geoms = fbx.by_type("Geometry", "Mesh")
        if len(geoms) != 1:
            raise SystemExit("expected exactly one mesh, found %d" % len(geoms))
        self.geom = geoms[0]
        mesh_models = [self.models[obj_id(m)] for m, _ in fbx.parents_of(obj_id(self.geom), "Model")]
        self.mesh_model = mesh_models[0]

        # skin clusters
        self.clusters = {}  # bone index -> (indexes, weights, transform, transform_link)
        for skin in [d for d, _ in fbx.children_of(obj_id(self.geom), "Deformer")]:
            for cl, _ in fbx.children_of(obj_id(skin), "Deformer"):
                if obj_subclass(cl) != "Cluster":
                    continue
                bone_models = [self.models[obj_id(b)] for b, _ in fbx.children_of(obj_id(cl), "Model")]
                if not bone_models:
                    continue
                b = bone_models[0]
                idx = cl.first("Indexes")
                w = cl.first("Weights")
                tf = mat_from_fbx_array(cl.first("Transform").props[0])
                tl = mat_from_fbx_array(cl.first("TransformLink").props[0])
                self.clusters[b.bone_index] = (
                    np.array(idx.props[0] if idx else [], dtype=np.int64),
                    np.array(w.props[0] if w else [], dtype=np.float64), tf, tl)

        self.stacks = [AnimStack(fbx, s, self.models) for s in fbx.by_type("AnimationStack")]

        gs = properties70(fbx.root.first("GlobalSettings")) if fbx.root.first("GlobalSettings") else {}
        self.up_axis = int(prop_scalar(gs, "UpAxis", 1))          # 0=X 1=Y 2=Z
        self.up_sign = int(prop_scalar(gs, "UpAxisSign", 1))
        self.unit_scale = float(prop_scalar(gs, "UnitScaleFactor", 1.0))

    # ---- transforms --------------------------------------------------------
    def global_matrix(self, model, stack=None, t=0.0):
        m = np.eye(4)
        cur = model
        chain = []
        while cur is not None:
            chain.append(cur)
            cur = cur.parent
        for node in reversed(chain):
            lm = stack.local_matrix(node, t) if stack else node.local_matrix()
            m = m @ lm
        return m

    def bone_effective_local(self, bone, stack=None, t=0.0):
        """Local matrix of a bone relative to its nearest bone ancestor
        (non-bone ancestors such as the armature object are folded in)."""
        m = np.eye(4)
        cur = bone
        while True:
            lm = stack.local_matrix(cur, t) if stack else cur.local_matrix()
            m = lm @ m
            cur = cur.parent
            if cur is None or cur.subclass == "LimbNode":
                break
        return m

    def bone_parent_index(self, bone):
        cur = bone.parent
        while cur is not None and cur.subclass != "LimbNode":
            cur = cur.parent
        return cur.bone_index if cur is not None else -1


# --------------------------------------------------------------------------
# mesh extraction
# --------------------------------------------------------------------------
def layer_element(geom, name, data_key, index_key):
    le = geom.first(name)
    if le is None:
        return None
    mapping = le.first("MappingInformationType").props[0]
    ref = le.first("ReferenceInformationType").props[0]
    data = np.array(le.first(data_key).props[0], dtype=np.float64)
    index = None
    if ref == "IndexToDirect":
        idx_node = le.first(index_key)
        if idx_node is not None:
            index = np.array(idx_node.props[0], dtype=np.int64)
    return mapping, data, index


def build_mesh(scene):
    geom = scene.geom
    positions = np.array(geom.first("Vertices").props[0], dtype=np.float64).reshape(-1, 3)
    pvi = geom.first("PolygonVertexIndex").props[0]

    normals = layer_element(geom, "LayerElementNormal", "Normals", "NormalsIndex")
    uvs = layer_element(geom, "LayerElementUV", "UV", "UVIndex")

    def corner_attr(elem, corner, vert, comps):
        if elem is None:
            return None
        mapping, data, index = elem
        if mapping == "ByPolygonVertex":
            i = corner
        elif mapping in ("ByVertice", "ByVertex"):
            i = vert
        elif mapping == "AllSame":
            i = 0
        else:
            raise SystemExit("unsupported mapping %s" % mapping)
        if index is not None:
            i = int(index[i])
        return tuple(data[i * comps:(i + 1) * comps])

    # expand polygons -> triangles (fan) with unique (pos, uv, normal) vertices
    verts = []          # (pos_index, normal, uv)
    vmap = {}
    tris = []           # [(v0, v1, v2)]
    poly = []
    for corner, raw in enumerate(pvi):
        vi = ~raw if raw < 0 else raw
        n = corner_attr(normals, corner, vi, 3)
        uv = corner_attr(uvs, corner, vi, 2)
        key = (vi, None if n is None else tuple(round(x, 5) for x in n),
               None if uv is None else tuple(round(x, 5) for x in uv))
        idx = vmap.get(key)
        if idx is None:
            idx = len(verts)
            vmap[key] = idx
            verts.append((vi, n, uv))
        poly.append(idx)
        if raw < 0:
            for k in range(1, len(poly) - 1):
                tris.append((poly[0], poly[k], poly[k + 1]))
            poly = []
    return positions, verts, tris


def build_skin(scene, nverts_pos):
    """dominant bone per original position index + full weights."""
    weights = np.zeros((nverts_pos, max(1, len(scene.bones))))
    for b, (idx, w, _tf, _tl) in scene.clusters.items():
        for i, ww in zip(idx, w):
            weights[i, b] += ww
    dominant = np.argmax(weights, axis=1)
    unskinned = np.where(weights.sum(axis=1) <= 0)[0]
    if len(unskinned) and not scene.static:
        print("warning: %d vertices have no skin weights; assigned to bone 0" % len(unskinned))
        dominant[unskinned] = 0
    return dominant, weights


# --------------------------------------------------------------------------
# conversion
# --------------------------------------------------------------------------
def clamp16(v):
    return int(max(-32768, min(32767, round(v))))


def to_ps1_matrix(m):
    """conjugate a 4x4 FBX-space matrix into PS1 space."""
    c = np.eye(4)
    c[:3, :3] = C_FBX_TO_PS1
    return c @ m @ c.T


def optimize_mesh(positions, verts, tris, target_tris=0, reatlas=False, texture=None,
                  atlas_size=256, verbose=True, face_tex=False, face_min_y=0.78, dump_uv=None,
                  front=(0, 0, 1), hidden_dist=0.0, double_sided=None, gen_skirt=False):
    """Decimate to target_tris (0 = keep) and optionally re-unwrap the UVs
    with xatlas, baking the source texture into the new atlas.  With
    face_tex the triangles above the neck line get their own atlas/texture
    (tex id 1) so the eyes keep their detail.  Returns
    (positions, verts, tris, textures) where textures is a list of baked
    images (or None) indexed by the tris' tex id; verts/tris are in
    build_mesh format with an extra tex id per triangle."""
    from meshopt import (bake, decimate, reatlas as do_reatlas, vertex_normals, draw_uv_check, remove_hidden,
                         quad_edges)
    T = np.array([[verts[i][0] for i in t] for t in tris], dtype=np.int64)
    # corner UVs, converted to image convention (v = 0 at the top)
    cuv = np.array([[(verts[i][2][0], 1.0 - verts[i][2][1]) if verts[i][2] is not None else (0.0, 0.0)
                     for i in t] for t in tris])
    P = positions
    ymin, H = P[:, 1].min(), P[:, 1].max()

    def skirt_mask(T_, covered=None):
        """The visible skirt: every corner in the height band, near the body
        axis sideways (excludes the forearms) but outside the legs' radius.
        With `covered` (triangles that have another surface up to 6% of the
        height in front of them) the skirt top tucked under the jacket is
        taken too, up to SKIRT_WAIST + 0.02."""
        if not double_sided:
            return np.zeros(len(T_), dtype=bool)
        y0, y1 = double_sided[0], double_sided[1]
        xmax = double_sided[2] if len(double_sided) > 2 else 0.135
        dmin = double_sided[3] if len(double_sided) > 3 else 0.07
        cen = (P[T_].mean(axis=1) - [0, ymin, 0]) / (H - ymin)
        dd = np.hypot(cen[:, 0], cen[:, 2])
        vy = (P[T_][:, :, 1] - ymin) / (H - ymin)
        lateral = (np.abs(cen[:, 0]) < xmax) & (dd > dmin)
        # long slivers of the thighs whose centroid falls in the band must
        # not be taken: every corner has to be inside
        inside = (vy >= y0 - 0.02).all(axis=1) & (vy <= y1 + 0.02).all(axis=1)
        sel = inside & (cen[:, 1] >= y0) & (cen[:, 1] <= y1) & lateral
        if covered is not None:
            top = (vy >= y1 - 0.04).all(axis=1) & (vy <= SKIRT_WAIST + 0.02).all(axis=1)
            sel |= top & lateral & covered
        return sel

    skirt_dims = None
    if hidden_dist > 0:
        keep = remove_hidden(P, T, hidden_dist * (H - ymin), verbose=verbose)
        if gen_skirt and double_sided:
            # the original skirt is replaced by a generated one, so it must not
            # hide anything: find it (thighs under it are gone after pass 1),
            # then redo the removal with the skirt excluded as an occluder
            # occluders for the "under the jacket" test: not the hands/arms
            body = (np.abs(P[T][:, :, 0]) < 0.15 * (H - ymin)).all(axis=1)
            covered = ~remove_hidden(P, T, 0.06 * (H - ymin), verbose=False, occluders=body)
            sel = skirt_mask(T, covered)
            keep = remove_hidden(P, T, hidden_dist * (H - ymin), verbose=verbose, occluders=keep & ~sel)
            keep &= ~sel
            cs = P[T[sel]].mean(axis=1)
            y0, y1 = double_sided[0], double_sided[1]
            ylo, yhi = ymin + (H - ymin) * (y0 + 0.01), ymin + (H - ymin) * max(y1, SKIRT_WAIST)
            span = (H - ymin) * 0.05
            hem = cs[cs[:, 1] < ylo + span]
            top = cs[cs[:, 1] > yhi - span]
            if len(hem) < 3: hem = cs
            if len(top) < 3: top = cs
            rx0, rz0 = np.percentile(np.abs(hem[:, 0]), 95), np.percentile(np.abs(hem[:, 2]), 95)
            rx1, rz1 = np.percentile(np.abs(top[:, 0]), 95) * 0.9, np.percentile(np.abs(top[:, 2]), 95) * 0.9
            # the measured centroids sit inside the pleats and under-estimate
            # the radii: the waist is at least 80% of the hem, the hem flares
            # at least 30% beyond the waist
            rx1, rz1 = max(rx1, rx0 * 0.8), max(rz1, rz0 * 0.8)
            skirt_dims = {"y0": ylo, "y1": yhi, "rx1": rx1, "rz1": rz1,
                          "rx0": max(rx0, rx1 * 1.3), "rz0": max(rz0, rz1 * 1.3)}
            if verbose:
                print("generated skirt: replaces %d triangles, y %.3f..%.3f hem %.3fx%.3f waist %.3fx%.3f" % (
                    int(sel.sum()), ylo, yhi, skirt_dims["rx0"], skirt_dims["rz0"], skirt_dims["rx1"], skirt_dims["rz1"]))
        T, cuv = T[keep], cuv[keep]
    # face region (above the neck line, mesh Y-up): protected from decimation
    head = P[:, 1] >= ymin + (H - ymin) * face_min_y
    # the face proper: front half of the head (hair stays in the body atlas)
    fr = np.array(front, dtype=np.float64)
    centre = (P[head].min(axis=0) + P[head].max(axis=0)) * 0.5
    face = head & (((P - centre) @ fr) >= -0.01 * (H - ymin))
    importance = np.where(face, 10.0, 1.0) if face_tex else None
    if target_tris and target_tris < len(T):
        P, T, keep = decimate(P, T, target_tris, verbose=verbose, importance=importance)
        cuv = cuv[keep]
    N = vertex_normals(P, T)
    if reatlas:
        # triangle groups: 0 = body, 1 = face (all three corners above the neck)
        groups = [np.ones(len(T), dtype=bool)]
        if face_tex:
            f = face[T].all(axis=1)
            groups = [~f, f]
            if verbose:
                print("face texture: %d triangles" % int(f.sum()))
        # the skirt as its own object: centroid in the height band (fractions
        # of the height), close to the body axis sideways (excludes the
        # forearms) but not inside the legs' radius (the thighs under the
        # skirt are already gone after remove_hidden).  Skirt triangles are
        # made double sided and their quad edges are exported.
        ds = np.zeros(len(T), dtype=bool)
        if double_sided and not skirt_dims:
            ds = skirt_mask(T)
            if verbose:
                print("skirt (double sided): %d triangles" % int(ds.sum()))
        new_verts, new_tris, textures = [], [], []
        edges = []                         # (a, b) into new_verts: quad edges of the double sided region
        for gi, g in enumerate(groups):
            Tg = T[g]
            if len(Tg) == 0:
                textures.append(None)
                continue
            vmap, T2, uvs = do_reatlas(P, Tg, resolution=atlas_size)
            baked = bake(uvs[T2], cuv[g], texture, size=atlas_size) if texture is not None else None
            textures.append(baked)
            if dump_uv and baked is not None:
                path = dump_uv if gi == 0 else dump_uv.replace(".png", "_face.png")
                draw_uv_check(baked, uvs, T2, path)
            base = len(new_verts)
            new_verts += [(int(vmap[i]), tuple(N[vmap[i]]), (float(uvs[i][0]), 1.0 - float(uvs[i][1])))
                          for i in range(len(vmap))]
            dsg = ds[g]
            # double sided triangles carry a flag (the renderer flips the
            # winding of back faces) instead of duplicated geometry
            new_tris += [tuple(int(x) + base for x in t) + (gi, int(TRI_DOUBLE) if d else 0)
                         for t, d in zip(T2, dsg)]
            if dsg.any():
                edges += [(a + base, b + base) for a, b in quad_edges(P[vmap], T2[dsg])]
        skirt_pos_start = len(P)
        if skirt_dims:
            from sprite import tartan
            P, new_verts, new_tris, sk_edges = generate_skirt(P, new_verts, new_tris, skirt_dims, len(textures))
            edges += sk_edges
            textures.append(tartan(SKIRT_TEX_SIZE))
        return P, new_verts, new_tris, textures, edges, skirt_pos_start
    # no re-atlas: unique (position, uv) per corner
    vmap = {}
    new_verts, new_tris = [], []
    for t in range(len(T)):
        idx = []
        for c in range(3):
            key = (int(T[t][c]), round(float(cuv[t][c][0]), 5), round(float(cuv[t][c][1]), 5))
            i = vmap.get(key)
            if i is None:
                i = len(new_verts)
                vmap[key] = i
                new_verts.append((key[0], tuple(N[key[0]]), (cuv[t][c][0], 1.0 - cuv[t][c][1])))
            idx.append(i)
        new_tris.append(tuple(idx) + (0, 0))
    return P, new_verts, new_tris, [None], [], len(P)


TRI_DOUBLE = 1             # ModelTri.flags: double sided
SKIRT_WAIST = 0.53         # generated skirt reaches up to here (under the jacket)
SKIRT_TEX_SIZE = 128
SKIRT_TEX_V0 = 64          # texel row where the skirt texture starts in its page (below the sprites)
SKIRT_SEGMENTS = 24
SKIRT_ROWS = 3


def generate_skirt(P, new_verts, new_tris, dims, tex_id):
    """Elliptical, slightly pleated cone of SKIRT_SEGMENTS x SKIRT_ROWS quads
    between the measured hem and waist.  Double sided; returns the quad
    edge list (front side).  UVs tile the tartan every 4 segments."""
    N, R = SKIRT_SEGMENTS, SKIRT_ROWS
    period = 4                                  # segments per texture repeat
    pos = list(P)
    base = len(new_verts)
    ring_idx = []                               # [row][seg] -> new vertex index
    for r in range(R + 1):
        f = r / R                               # 0 = waist, 1 = hem
        y = dims["y1"] + (dims["y0"] - dims["y1"]) * f
        rx = dims["rx1"] + (dims["rx0"] - dims["rx1"]) * f
        rz = dims["rz1"] + (dims["rz0"] - dims["rz1"]) * f
        row = []
        for i in range(N + 1):                  # N+1: seam vertex duplicated for UVs
            k = i % N
            a = 2 * math.pi * k / N
            pleat = 1.0 - 0.06 * f * (k % 2)    # alternate segments tuck in towards the hem
            x, z = rx * pleat * math.cos(a), rz * pleat * math.sin(a)
            n = np.array([math.cos(a) / max(rx, 1e-6), 0.0, math.sin(a) / max(rz, 1e-6)])
            n = n / np.linalg.norm(n)
            pos.append(np.array([x, y, z]))
            u = ((i % period) * (SKIRT_TEX_SIZE // period)) if i % period or i == 0 else SKIRT_TEX_SIZE
            if i % period == 0 and i > 0:
                u = SKIRT_TEX_SIZE               # right edge of the repeat
            u = min(u, SKIRT_TEX_SIZE - 1)
            v = SKIRT_TEX_V0 + f * (SKIRT_TEX_SIZE - 1)
            new_verts.append((len(pos) - 1, tuple(n), (u / 255.0, 1.0 - v / 255.0)))
            row.append(len(new_verts) - 1)
        ring_idx.append(row)
    front_tris = []
    for r in range(R):
        for i in range(N):
            a, b = ring_idx[r][i], ring_idx[r][i + 1]
            c, d = ring_idx[r + 1][i], ring_idx[r + 1][i + 1]
            front_tris += [(a, b, c), (b, d, c)]
    for t in front_tris:
        new_tris.append(t + (tex_id, TRI_DOUBLE))
    edges = []
    for r in range(R + 1):
        for i in range(N):
            edges.append((ring_idx[r][i], ring_idx[r][i + 1]))
    for r in range(R):
        for i in range(N):
            edges.append((ring_idx[r][i], ring_idx[r + 1][i]))
    return np.array(pos), new_verts, new_tris, edges


def build_ik_table(bone_names, parents, bone_bind_global, inv_bind, bind_local, front_fbx, rig, verbose=True):
    """Find the limb chains / head / hip by bone name and express the
    effector offsets and directions in PS1 space (see ModelIK in model.h)."""
    c3 = C_FBX_TO_PS1
    lower = [n.lower() for n in bone_names]

    def find(*keys, side=None, exclude=("twist",)):
        for i, n in enumerate(lower):
            if any(e in n for e in exclude):
                continue
            if side and not (n.startswith(side.lower() + "_") or n.endswith("_" + side.lower())):
                continue
            if any(k in n for k in keys):
                return i
        return -1

    def joint_world(b):
        return bone_bind_global[b][:3, 3]

    front_ps1 = c3 @ np.array(front_fbx, dtype=np.float64)
    chains = []
    for side in ("L", "R"):
        for kind in ("arm", "leg"):
            if kind == "arm":
                up, lo, end = find("upperarm", "upper_arm", "shoulder", side=side), find("forearm", "lowerarm", side=side), find("hand", side=side)
                pole = -front_ps1                       # elbows bend backwards
            else:
                up, lo, end = find("thigh", "upleg", "upperleg", side=side), find("calf", "shin", "lowerleg", "leg", side=side), find("foot", side=side)
                pole = front_ps1                        # knees bend forwards
            if up < 0 or lo < 0:
                chains.append(None)
                continue
            if end >= 0:
                end_local = bind_local[end][:3, 3] * UNIT_SCALE   # already PS1 space (lower-bone local)
            else:
                eff = None
                if rig:
                    import autorig as ar
                    e = ar.EFFECTORS.get(bone_names[lo])
                    if e:
                        eff = np.array(e[1]) * rig["H"]
                if eff is None:
                    eff = joint_world(lo) + (joint_world(lo) - joint_world(up))   # extend by the upper length
                p_local = inv_bind[lo] @ np.append(eff, 1.0)
                end_local = c3 @ p_local[:3] * UNIT_SCALE
            chains.append({"upper": up, "lower": lo, "end": end, "end_local": end_local, "pole": pole})
    # order: arm_L, arm_R, leg_L, leg_R
    chains = [chains[0], chains[2], chains[1], chains[3]]
    head = find("head", exclude=("twist", "headtop", "hair"))
    # hip: lowest common ancestor of all chain roots
    def ancestors(b):
        out = []
        while b >= 0:
            out.append(b)
            b = parents[b]
        return out
    roots = [c["upper"] for c in chains if c]
    hip = 0
    if roots:
        common = set(ancestors(roots[0]))
        for r in roots[1:]:
            common &= set(ancestors(r))
        hip = max(common, key=lambda b: len(ancestors(b)))   # deepest common ancestor
    head_fwd = np.array([0, 0, -1.0])
    if head >= 0:
        rot_ps1 = to_ps1_matrix(bone_bind_global[head])[:3, :3]
        head_fwd = np.linalg.inv(rot_ps1) @ front_ps1
        head_fwd /= np.linalg.norm(head_fwd)
    if verbose:
        print("ik: hip=%s head=%s chains=%s" % (bone_names[hip], bone_names[head] if head >= 0 else None,
              [(bone_names[c["upper"]], bone_names[c["lower"]], bone_names[c["end"]] if c["end"] >= 0 else "-") if c else None for c in chains]))
    return {"hip": hip, "head": head, "head_fwd": head_fwd, "chains": chains}


def convert(scene, verbose=True, strip_root_motion=True, target_tris=0, reatlas=False, texture=None,
            autorig=False, front=(0, 0, 1), face_tex=False, dump_uv=None, hidden_dist=0.0,
            double_sided=None, gen_skirt=False):
    global C_FBX_TO_PS1
    if scene.up_axis == 2:
        C_FBX_TO_PS1 = C_ZUP_TO_PS1 * np.array([[1], [scene.up_sign], [1]])
    else:
        C_FBX_TO_PS1 = C_YUP_TO_PS1 * np.array([[1], [scene.up_sign], [1]])
    if verbose:
        print("world up axis: %s%s (UnitScaleFactor %g)" % ("+" if scene.up_sign > 0 else "-", "XYZ"[scene.up_axis], scene.unit_scale))
    positions, verts, tris = build_mesh(scene)
    textures = [None]
    edges = []
    skirt_pos_start = len(positions)
    if target_tris or reatlas or hidden_dist:
        positions, verts, tris, textures, edges, skirt_pos_start = optimize_mesh(positions, verts, tris, target_tris, reatlas,
                                                         texture, verbose=verbose, face_tex=face_tex,
                                                         dump_uv=dump_uv, front=front, hidden_dist=hidden_dist,
                                                         double_sided=double_sided, gen_skirt=gen_skirt)
        if verbose:
            print("optimized mesh: %d verts, %d tris" % (len(verts), len(tris)))
    else:
        tris = [tuple(t) + (0, 0) for t in tris]
    rig = None
    if autorig and scene.static:
        import autorig as ar
        mesh_bind0 = scene.global_matrix(scene.mesh_model) @ scene.mesh_model.geometric_matrix()
        Pw = np.array([(mesh_bind0 @ np.append(q, 1.0))[:3] for q in positions])
        H = float(Pw[:, 1].max())
        Tpos = np.array([[verts[i][0] for i in t[:3]] for t in tris], dtype=np.int64)
        rig = {"H": H, "joints": ar.joint_positions(H), "assign": ar.assign_bones(Pw, H, T=Tpos),
               "names": [n for n, _ in ar.BONES], "parents": [p for _, p in ar.BONES]}
        rig["assign"][skirt_pos_start:] = ar.NAME["root"]      # generated skirt hangs from the hips
        if verbose:
            counts = np.bincount(rig["assign"], minlength=len(ar.BONES))
            print("autorig: height %.3f, vertices per bone: %s" % (
                H, ", ".join("%s=%d" % (n, c) for n, c in zip(rig["names"], counts))))
    dominant, weights = build_skin(scene, len(positions))
    nb = max(1, len(scene.bones))
    bone_names = [b.name for b in scene.bones] or ["root"]
    if rig:
        nb = len(rig["names"])
        bone_names = rig["names"]
        dominant = rig["assign"]

    # bind pose: mesh transform + per-bone inverse bind (from TransformLink)
    mesh_bind = scene.global_matrix(scene.mesh_model) @ scene.mesh_model.geometric_matrix()
    tf_mesh = mesh_bind
    inv_bind = [None] * nb
    bone_bind_global = [None] * nb
    for b in range(nb):
        if rig:
            bone_bind_global[b] = mat_t(rig["joints"][b])
        elif b in scene.clusters:
            bone_bind_global[b] = scene.clusters[b][3]
        elif b < len(scene.bones):
            bone_bind_global[b] = scene.global_matrix(scene.bones[b])
        else:
            bone_bind_global[b] = np.eye(4)      # synthetic root of a static mesh
        inv_bind[b] = np.linalg.inv(bone_bind_global[b])
    # Cluster "Transform" convention check (informational): Blender writes
    # TL^-1 * MeshGlobal, the FBX SDK writes MeshGlobal.  Either way the
    # bone-local vertex is TL^-1 * MeshGlobal * v, which is what we use.
    if verbose and scene.clusters:
        b0 = next(iter(scene.clusters))
        tf = scene.clusters[b0][2]
        tl = scene.clusters[b0][3]
        if np.allclose(tf, np.linalg.inv(tl) @ mesh_bind, atol=1e-3):
            print("cluster Transform convention: Blender (TL^-1 * MeshGlobal)")
        elif np.allclose(tf, mesh_bind, atol=1e-3):
            print("cluster Transform convention: FBX SDK (MeshGlobal)")
        else:
            print("warning: cluster Transform matches neither convention; assuming vertices are in mesh-local space")

    # bind-pose local matrices derived from TransformLink (FBX space)
    parents = rig["parents"] if rig else ([scene.bone_parent_index(b) for b in scene.bones] or [-1])
    bind_local_fbx = []
    for b in range(nb):
        p = parents[b]
        if p >= 0:
            bind_local_fbx.append(inv_bind[p] @ bone_bind_global[b])
        else:
            bind_local_fbx.append(bone_bind_global[b])

    # ---- vertices: bone-local, PS1 space ----------------------------------
    c3 = C_FBX_TO_PS1
    out_verts = []  # dict per vertex
    for vi, n, uv in verts:
        p_world = tf_mesh @ np.append(positions[vi], 1.0)
        b = int(dominant[vi])
        p_local = inv_bind[b] @ p_world
        p = c3 @ p_local[:3] * UNIT_SCALE
        if n is not None:
            n_world = tf_mesh[:3, :3] @ np.array(n)
            n_local = inv_bind[b][:3, :3] @ n_world
            nn = c3 @ n_local
            nn = nn / (np.linalg.norm(nn) or 1.0)
        else:
            nn = np.array([0.0, -1.0, 0.0])
        if uv is not None:
            u = int(max(0, min(255, round(uv[0] * 255.0))))
            v = int(max(0, min(255, round((1.0 - uv[1]) * 255.0))))
        else:
            u = v = 0
        out_verts.append({"pos": p, "n": nn, "bone": b, "u": u, "v": v, "pos_index": vi})

    # ---- triangles: winding + face normal in face bone space --------------
    # bind-pose world-space positions for winding decision
    world_pos = [tf_mesh @ np.append(positions[v["pos_index"]], 1.0) for v in out_verts]
    world_pos = [c3 @ w[:3] for w in world_pos]
    out_tris = []
    for (a, b_, c, tex, tflags) in tris:
        pa, pb, pc = world_pos[a], world_pos[b_], world_pos[c]
        face_n = np.cross(pb - pa, pc - pa)
        # vertex normals in world space
        vn = np.zeros(3)
        for i in (a, b_, c):
            bw = bone_bind_global[out_verts[i]["bone"]][:3, :3]
            vn += c3 @ (bw @ (c3.T @ out_verts[i]["n"]))
        # GTE nclip > 0 for front faces when cross(v1-v0, v2-v0) points AWAY
        # from the viewer (see cube example); i.e. cross . outward_normal < 0
        if np.dot(face_n, vn) > 0:
            b_, c = c, b_
            pb, pc = pc, pb
            face_n = -face_n
        # outward face normal = -face_n (normalized)
        fnw = -face_n / (np.linalg.norm(face_n) or 1.0)
        # dominant bone for the face: most common among its vertices
        bones = [out_verts[i]["bone"] for i in (a, b_, c)]
        fb = max(set(bones), key=bones.count)
        fn_local = c3 @ (inv_bind[fb][:3, :3] @ (c3.T @ fnw))
        out_tris.append({"i": (a, b_, c), "bone": fb, "n": fn_local, "tex": tex, "flags": tflags})

    # ---- sort vertices by bone, triangles by bone --------------------------
    order = sorted(range(len(out_verts)), key=lambda i: out_verts[i]["bone"])
    remap = {old: new for new, old in enumerate(order)}
    out_verts = [out_verts[i] for i in order]
    for t in out_tris:
        t["i"] = tuple(remap[i] for i in t["i"])
    out_edges = [(remap[a], remap[b]) for a, b in edges]
    out_tris.sort(key=lambda t: t["bone"])

    bone_vert_range = [[0, 0] for _ in range(nb)]
    for i, v in enumerate(out_verts):
        r = bone_vert_range[v["bone"]]
        if r[1] == 0:
            r[0] = i
        r[1] += 1
    bone_tri_range = [[0, 0] for _ in range(nb)]
    for i, t in enumerate(out_tris):
        r = bone_tri_range[t["bone"]]
        if r[1] == 0:
            r[0] = i
        r[1] += 1

    # ---- skeleton ---------------------------------------------------------
    bind_local = []
    for b in range(nb):
        rot, scale = orthonormalize(to_ps1_matrix(bind_local_fbx[b]))
        if np.abs(scale - 1.0).max() > 0.02 and verbose:
            print("warning: bone %s has bind scale %s (ignored)" % (bone_names[b], scale))
        bind_local.append(rot)

    # ---- animations -------------------------------------------------------
    # first pass: sample local rotation/translation of every bone
    anims = []
    if rig:
        import autorig as ar
        stacks = [ar.SynthAnim(n, d, fn, bind_local_fbx, rig["H"]) for n, d, fn in ar.ANIMS]
    else:
        class _FbxAnim:
            def __init__(self, st):
                self.st, self.name, self.duration = st, st.name, st.duration
            def bone_local(self, b, t):
                if b < len(scene.bones) and scene.bones[b].id in self.st.tracks:
                    return scene.bone_effective_local(scene.bones[b], self.st, self.st.start + t)
                return bind_local_fbx[b]
        stacks = [_FbxAnim(st) for st in scene.stacks]
    for st in stacks:
        nframes = max(1, int(round(st.duration * SAMPLE_FPS)) + 1)
        quats = np.zeros((nframes, nb, 4))
        trans = np.zeros((nframes, nb, 3))
        prev_q = None
        for f in range(nframes):
            t = f / SAMPLE_FPS
            for b in range(nb):
                lm = to_ps1_matrix(st.bone_local(b, t))
                rot, _scale = orthonormalize(lm)
                q = mat_to_quat(rot)
                if prev_q is not None and np.dot(q, prev_q[b]) < 0:
                    q = -q
                quats[f, b] = q
                trans[f, b] = rot[:3, 3] * UNIT_SCALE
            prev_q = quats[f]
        anims.append({"name": st.name, "nframes": nframes, "q": quats, "t": trans})
        if verbose:
            print("anim %-16s %6.2fs %4d frames" % (st.name, st.duration, nframes))
    if not anims:
        # no animation: one frame holding the bind pose
        quats = np.zeros((1, nb, 4))
        trans = np.zeros((1, nb, 3))
        for b in range(nb):
            quats[0, b] = mat_to_quat(bind_local[b])
            trans[0, b] = bind_local[b][:3, 3] * UNIT_SCALE
        anims.append({"name": "static", "nframes": 1, "q": quats, "t": trans})

    # translation-animated bones: translation varies (within an anim or vs bind)
    trans_animated = np.zeros(nb, dtype=bool)
    for a in anims:
        var = np.abs(a["t"] - a["t"][0:1]).max(axis=0).max(axis=1)
        trans_animated |= var > UNIT_SCALE / 1000.0
        for b in range(nb):
            if np.abs(a["t"][0, b] - bind_local[b][:3, 3] * UNIT_SCALE).max() > UNIT_SCALE / 1000.0:
                trans_animated[b] = True
    if verbose:
        print("translation-animated bones: %s" % [bone_names[b] for b in range(nb) if trans_animated[b]])

    # second pass: remove horizontal root motion so the character stays put.
    # Done in PS1 world space (y = up/down); x/z of translation-animated bones
    # are pinned to their bind-pose world position.
    if strip_root_motion:
        bind_world = [None] * nb
        for b in range(nb):
            p = parents[b]
            bind_world[b] = (bind_world[p] @ bind_local[b]) if p >= 0 else bind_local[b]
        for a in anims:
            for f in range(a["nframes"]):
                world = [None] * nb
                for b in range(nb):
                    m = np.eye(4)
                    m[:3, :3] = quat_to_mat3(a["q"][f, b])
                    m[:3, 3] = a["t"][f, b] / UNIT_SCALE
                    p = parents[b]
                    pw = world[p] if p >= 0 else np.eye(4)
                    if trans_animated[b]:
                        wt = pw @ np.append(m[:3, 3], 1.0)
                        wt[0] = bind_world[b][0, 3]
                        wt[2] = bind_world[b][2, 3]
                        m[:3, 3] = (np.linalg.inv(pw) @ wt)[:3]
                        a["t"][f, b] = m[:3, 3] * UNIT_SCALE
                    world[b] = pw @ m

    ik = build_ik_table(bone_names, parents, bone_bind_global, inv_bind, bind_local, front, rig, verbose)

    return {
        "verts": out_verts, "tris": out_tris, "nbones": nb, "parents": parents, "ik": ik,
        "bind_local": bind_local, "bone_vert_range": bone_vert_range,
        "bone_tri_range": bone_tri_range, "anims": anims,
        "trans_animated": trans_animated, "bone_names": bone_names,
        "textures": textures, "rig": rig, "edges": out_edges,
    }


# --------------------------------------------------------------------------
# binary writer (must match src/model.h)
# --------------------------------------------------------------------------
def write_model_bin(model, path):
    verts, tris, nb = model["verts"], model["tris"], model["nbones"]
    anims = model["anims"]
    ta = model["trans_animated"]
    trans_slots = [b for b in range(nb) if ta[b]]

    vert_blob = b"".join(
        struct.pack("<hhhBBhhhBB",
                    clamp16(v["pos"][0]), clamp16(v["pos"][1]), clamp16(v["pos"][2]),
                    v["bone"], 0,
                    clamp16(v["n"][0] * FIX_ONE), clamp16(v["n"][1] * FIX_ONE), clamp16(v["n"][2] * FIX_ONE),
                    v["u"], v["v"]) for v in verts)
    tri_blob = b"".join(
        struct.pack("<HHHBBhhhBB", t["i"][0], t["i"][1], t["i"][2], t["bone"], t["tex"],
                    clamp16(t["n"][0] * FIX_ONE), clamp16(t["n"][1] * FIX_ONE), clamp16(t["n"][2] * FIX_ONE),
                    t.get("flags", 0), 0)
        for t in tris)
    bone_blob = b""
    for b in range(nb):
        bl = model["bind_local"][b]
        vr, tr = model["bone_vert_range"][b], model["bone_tri_range"][b]
        flags = 1 if ta[b] else 0
        slot = trans_slots.index(b) if ta[b] else 0xFF
        bone_blob += struct.pack("<bBBBHHHHhhh",
                                 model["parents"][b], flags, slot, 0,
                                 vr[0], vr[1], tr[0], tr[1],
                                 clamp16(bl[0, 3] * UNIT_SCALE), clamp16(bl[1, 3] * UNIT_SCALE), clamp16(bl[2, 3] * UNIT_SCALE))
    # every section starts 4-byte aligned (runtime structs are not packed)
    def pad4(b):
        return b + b"\0" * (-len(b) % 4)
    vert_blob, tri_blob, bone_blob = pad4(vert_blob), pad4(tri_blob), pad4(bone_blob)
    header_size = 40
    off_verts = header_size
    off_tris = off_verts + len(vert_blob)
    off_bones = off_tris + len(tri_blob)
    off_anims = off_bones + len(bone_blob)
    anim_table_size = 24 * len(anims)
    frame_size = nb * 8 + len(trans_slots) * 8
    anim_blob = b""
    frame_blobs = []
    data_off = off_anims + anim_table_size
    for a in anims:
        blob = b""
        for f in range(a["nframes"]):
            for b in range(nb):
                q = a["q"][f, b]
                blob += struct.pack("<hhhh", clamp16(q[0] * FIX_ONE), clamp16(q[1] * FIX_ONE),
                                    clamp16(q[2] * FIX_ONE), clamp16(q[3] * FIX_ONE))
            for b in trans_slots:
                t = a["t"][f, b]
                blob += struct.pack("<hhhh", clamp16(t[0]), clamp16(t[1]), clamp16(t[2]), 0)
        assert len(blob) == frame_size * a["nframes"]
        name = a["name"].encode("ascii", "replace")[:15]
        anim_blob += struct.pack("<16sHHI", name, a["nframes"], SAMPLE_FPS, data_off)
        frame_blobs.append(blob)
        data_off += len(blob)
    # IK table (after the animation data, 4-byte aligned)
    ik = model.get("ik")
    ik_blob = b""
    off_ik = 0
    if ik:
        frames_total = sum(map(len, frame_blobs))
        off_ik = data_off + (-frames_total % 4)
        pad_ik = b"\0" * (-frames_total % 4)
        hf = ik["head_fwd"]
        ik_blob = struct.pack("<bbbbhhhh", ik["hip"], ik["head"], 0, 0,
                              clamp16(hf[0] * FIX_ONE), clamp16(hf[1] * FIX_ONE), clamp16(hf[2] * FIX_ONE), 0)
        for c in ik["chains"]:
            if c is None:
                ik_blob += struct.pack("<bbbbhhhhhh", -1, -1, -1, 0, 0, 0, 0, 0, 0, 0)
            else:
                el, po = c["end_local"], c["pole"]
                ik_blob += struct.pack("<bbbbhhhhhh", c["upper"], c["lower"], c["end"], 0,
                                       clamp16(el[0]), clamp16(el[1]), clamp16(el[2]),
                                       clamp16(po[0] * FIX_ONE), clamp16(po[1] * FIX_ONE), clamp16(po[2] * FIX_ONE))
        assert len(ik_blob) == 12 + 16 * 4
        ik_blob = pad_ik + ik_blob
    # edge list (uint16 pairs) for the wire overlay / edge walker
    edges = model.get("edges") or []
    tail = header_size + len(vert_blob) + len(tri_blob) + len(bone_blob) + len(anim_blob) + sum(map(len, frame_blobs)) + len(ik_blob)
    edge_blob = b"\0" * (-tail % 4) + b"".join(struct.pack("<HH", a, b) for a, b in edges)
    off_edges = tail + (-tail % 4) if edges else 0
    header = struct.pack("<4sHHHHHHIIIIII", MAGIC, len(verts), len(tris), nb, len(anims), len(trans_slots),
                         len(edges), off_verts, off_tris, off_bones, off_anims, off_ik, off_edges)
    assert len(header) == header_size
    with open(path, "wb") as f:
        f.write(header + vert_blob + tri_blob + bone_blob + anim_blob + b"".join(frame_blobs) + ik_blob + edge_blob)
    total = tail + len(edge_blob)
    print("wrote %s: %d verts, %d tris, %d bones, %d anims, %d bytes (anim data %d bytes)" % (
        path, len(verts), len(tris), nb, len(anims), total, sum(map(len, frame_blobs))))


# --------------------------------------------------------------------------
# texture -> 8bpp TIM
# --------------------------------------------------------------------------
def write_tim(texture, out_path, size=256, vram_x=640, vram_y=0, clut_x=640, clut_y=256, slot=0):
    """slot 0: body texture, slot 1: face texture (next page), slot 2: skirt
    texture (128x128 in the sprite page below the monkey)."""
    if slot == 2:
        size, vram_x, vram_y, clut_y = SKIRT_TEX_SIZE, 896, SKIRT_TEX_V0, clut_y + 3
    else:
        vram_x += 128 * slot           # 8bpp: 256 texels == 128 VRAM words == one page
        clut_y += slot
    """texture: file path or PIL image."""
    from PIL import Image
    img = (Image.open(texture) if isinstance(texture, str) else texture).convert("RGB")
    if img.size != (size, size):
        img = img.resize((size, size), Image.LANCZOS)
    pal_img = img.quantize(colors=256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.FLOYDSTEINBERG)
    pal = (pal_img.getpalette() + [0] * (256 * 3))[:256 * 3]
    idx = pal_img.tobytes()
    clut = []
    for i in range(256):
        r, g, b = pal[i * 3:i * 3 + 3]
        c = (r >> 3) | ((g >> 3) << 5) | ((b >> 3) << 10)
        if c == 0:
            c = 0x8000  # opaque black (0x0000 would be transparent)
        clut.append(c)
    clut_blob = struct.pack("<%dH" % 256, *clut)
    pix_w = size // 2  # 8bpp: two texels per 16-bit VRAM word
    out = bytearray()
    out += struct.pack("<II", 0x10, 0x09)  # id, flags: 8bpp + CLUT
    out += struct.pack("<IHHHH", 12 + len(clut_blob), clut_x, clut_y, 256, 1) + clut_blob
    out += struct.pack("<IHHHH", 12 + len(idx), vram_x, vram_y, pix_w, size) + idx
    with open(out_path, "wb") as f:
        f.write(out)
    print("wrote %s: %dx%d 8bpp, %d bytes, vram (%d,%d) clut (%d,%d)" % (
        out_path, size, size, len(out), vram_x, vram_y, clut_x, clut_y))


# --------------------------------------------------------------------------
# preview: replicate the PS1 runtime pipeline with the quantized data
# --------------------------------------------------------------------------
def pose_matrices(model, anim_index, frame):
    nb = model["nbones"]
    a = model["anims"][anim_index]
    ta = model["trans_animated"]
    world = [None] * nb
    for b in range(nb):
        q = np.round(a["q"][frame, b] * FIX_ONE) / FIX_ONE
        q = q / np.linalg.norm(q)
        m = np.eye(4)
        m[:3, :3] = quat_to_mat3(q)
        if ta[b]:
            m[:3, 3] = np.round(a["t"][frame, b])
        else:
            m[:3, 3] = np.round(model["bind_local"][b][:3, 3] * UNIT_SCALE)
        p = model["parents"][b]
        world[b] = (world[p] @ m) if p >= 0 else m
    return world


def render_preview(model, anim_name, frame, out_path, size=512, shaded=True):
    from PIL import Image, ImageDraw
    names = [a["name"] for a in model["anims"]]
    if anim_name not in names:
        raise SystemExit("unknown anim %r; available: %s" % (anim_name, names))
    ai = names.index(anim_name)
    frame = min(frame, model["anims"][ai]["nframes"] - 1)
    world = pose_matrices(model, ai, frame)
    pts = []
    for v in model["verts"]:
        p = np.round(v["pos"])
        w = world[v["bone"]] @ np.append(p, 1.0)
        pts.append(w[:3])
    pts = np.array(pts)
    img = Image.new("RGB", (size * 2, size), (20, 20, 30))
    d = ImageDraw.Draw(img)

    def proj(p, view):
        # front view: x right, y down ; side view: z right, y down
        sx = p[0] if view == 0 else p[2]
        k = size / (2.4 * UNIT_SCALE)
        return (size * 0.5 + sx * k + view * size, size * 0.85 + p[1] * k)

    # shaded views (painter's algorithm) on the far right / bottom: front + side
    if shaded:
        img2 = Image.new("RGB", (size * 2, size), (20, 20, 30))
        d2 = ImageDraw.Draw(img2)
        light = np.array([-0.4, -0.6, -0.7]); light /= np.linalg.norm(light)
        for view in (0, 1):
            order = []
            for t in model["tris"]:
                a, b, c = (pts[i] for i in t["i"])
                depth = (a[2] + b[2] + c[2]) / 3 if view == 0 else -(a[0] + b[0] + c[0]) / 3
                order.append((depth, t))
            order.sort(key=lambda x: -x[0])          # far first (camera at -z / +x)
            for _, t in order:
                a, b, c = (pts[i] for i in t["i"])
                n = np.cross(b - a, c - a)
                l = np.linalg.norm(n)
                if l == 0:
                    continue
                n = -n / l
                # backface: outward normal points away from the camera
                cam = np.array([0, 0, -1.0]) if view == 0 else np.array([1.0, 0, 0])
                if np.dot(n, cam) <= 0:
                    continue
                sh = max(0.15, float(np.dot(n, light)) * 0.85 + 0.15)
                col = (int(200 * sh), int(180 * sh), int(170 * sh)) if t.get("tex", 0) == 0 else (int(240 * sh), int(200 * sh), int(120 * sh))
                d2.polygon([proj(a, view), proj(b, view), proj(c, view)], fill=col)
        d2.text((8, 8), "shaded  left: front  right: side (face texture = yellow)", fill=(255, 255, 255))
        img2.save(out_path.replace(".png", "_shaded.png"))
        print("wrote %s" % out_path.replace(".png", "_shaded.png"))

    for view in (0, 1):
        for t in model["tris"]:
            a, b, c = (pts[i] for i in t["i"])
            # backface test in front view (same sign convention as GTE nclip)
            if view == 0:
                pa, pb, pc = proj(a, 0), proj(b, 0), proj(c, 0)
                n = (pb[0] - pa[0]) * (pc[1] - pa[1]) - (pc[0] - pa[0]) * (pb[1] - pa[1])
                col = (90, 200, 120) if n > 0 else (70, 70, 90)
            else:
                col = (120, 160, 220)
            d.polygon([proj(a, view), proj(b, view), proj(c, view)], outline=col)
        for b in range(model["nbones"]):
            p = model["parents"][b]
            if p < 0:
                continue
            d.line([proj(world[p][:3, 3], view), proj(world[b][:3, 3], view)], fill=(255, 80, 80), width=2)
    d.text((8, 8), "%s frame %d/%d  left: front (green = front-facing)  right: side" % (
        anim_name, frame, model["anims"][ai]["nframes"]), fill=(255, 255, 255))
    img.save(out_path)
    print("wrote %s" % out_path)


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("fbx")
    ap.add_argument("--texture")
    ap.add_argument("--out-bin")
    ap.add_argument("--out-tim")
    ap.add_argument("--preview", nargs=3, metavar=("ANIM", "FRAME", "PNG"))
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--keep-root-motion", action="store_true", help="do not strip horizontal root motion")
    ap.add_argument("--target-tris", type=int, default=0, help="decimate the mesh down to this many triangles")
    ap.add_argument("--reatlas", action="store_true",
                    help="re-unwrap UVs with xatlas (shared vertices, no bleeding) and bake the texture")
    ap.add_argument("--dump-texture", help="write the (baked) 256x256 texture as PNG for inspection")
    ap.add_argument("--face-tex", action="store_true",
                    help="with --reatlas: give the face (above 0.78 of the height) its own 256x256 texture "
                         "(written to --out-face-tim) and protect it from decimation")
    ap.add_argument("--out-face-tim")
    ap.add_argument("--gen-skirt", action="store_true",
                    help="with --skirt: replace the selected skirt by a generated pleated cone with its own "
                         "tartan texture (--out-skirt-tim)")
    ap.add_argument("--out-skirt-tim")
    ap.add_argument("--skirt", dest="double_sided", type=float, nargs="+", metavar="F",
                    help="with --reatlas: Y0 Y1 [XMAX DMIN] (fractions of the height) select the skirt: "
                         "centroid height in Y0..Y1, |x| < XMAX (default 0.135), horizontal distance from "
                         "the body axis > DMIN (default 0.07).  Skirt triangles become double sided and "
                         "their quad edges are exported for the edge walker")
    ap.add_argument("--remove-hidden", type=float, default=0.0, metavar="DIST",
                    help="drop triangles covered by another surface within DIST x height in front of them "
                         "(e.g. 0.04: skirt under the jacket, body under clothes)")
    ap.add_argument("--dump-uv", help="with --reatlas: write the baked atlas with the UV wireframe (face in yellow)")
    ap.add_argument("--front", default="+z", choices=["+z", "-z", "+x", "-x"],
                    help="direction the character faces in FBX space (for IK pole vectors / head look-at)")
    ap.add_argument("--autorig", action="store_true",
                    help="un-rigged mesh: segment into body parts, add a biped skeleton and procedural animations")
    args = ap.parse_args()

    scene = Scene(args.fbx)
    if not args.quiet:
        print("FBX v%d: %d bones, %d anim stacks, mesh model %r" % (
            scene.fbx.version, len(scene.bones), len(scene.stacks), scene.mesh_model.name))
    src_tex = None
    if args.texture:
        from PIL import Image
        src_tex = Image.open(args.texture)
    model = convert(scene, verbose=not args.quiet, strip_root_motion=not args.keep_root_motion,
                    target_tris=args.target_tris, reatlas=args.reatlas, texture=src_tex,
                    autorig=args.autorig, face_tex=args.face_tex, dump_uv=args.dump_uv,
                    hidden_dist=args.remove_hidden, double_sided=args.double_sided, gen_skirt=args.gen_skirt,
                    front={"+z": (0, 0, 1), "-z": (0, 0, -1), "+x": (1, 0, 0), "-x": (-1, 0, 0)}[args.front])
    textures = model["textures"]
    tex = textures[0] or args.texture
    if args.out_bin:
        write_model_bin(model, args.out_bin)
    if args.out_tim:
        if not tex:
            raise SystemExit("--out-tim requires --texture")
        write_tim(tex, args.out_tim)
    if args.out_face_tim:
        face = textures[1] if len(textures) > 1 and textures[1] is not None else tex
        write_tim(face, args.out_face_tim, slot=1)
    if args.out_skirt_tim:
        if len(textures) > 2 and textures[2] is not None:
            write_tim(textures[2], args.out_skirt_tim, slot=2)
        else:
            raise SystemExit("--out-skirt-tim needs --gen-skirt")
    if args.dump_texture and tex:
        from PIL import Image
        (Image.open(tex) if isinstance(tex, str) else tex).convert("RGB").resize((256, 256)).save(args.dump_texture)
        if len(textures) > 1 and textures[1] is not None:
            textures[1].save(args.dump_texture.replace(".png", "_face.png"))
    if args.preview:
        render_preview(model, args.preview[0], int(args.preview[1]), args.preview[2])


if __name__ == "__main__":
    main()
