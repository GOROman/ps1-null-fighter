"""Automatic rigid rig for an un-rigged humanoid mesh.

The mesh is cut into body parts by simple geometric rules (heights and
side offsets as fractions of the model height, tuned for a standing
character with the arms hanging at the sides), a biped skeleton is placed at
the joints and a few procedural animations are generated.

Coordinate system: FBX Y-up, front = +Z, height 0..H.  Rotations are
returned as 4x4 local matrices (column vectors) about the bone's joint.
"""
import math

import numpy as np

# joint heights / offsets as fractions of the model height
RIG = {
    "hip_y":       0.47,   # crotch: legs below this
    "hip_x":       0.06,   # leg centre line
    "knee_y":      0.24,
    "shoulder_y":  0.74,
    "shoulder_x":  0.14,
    "arm_min_x":   0.125,  # |x| beyond this (between hip_y and elbow_y) = forearm/hand
    "arm_min_x_u": 0.09,   # |x| beyond this (between elbow_y and neck_y) = upper arm
    "hand_min_y":  0.30,   # arms/hands are never below this
    "elbow_y":     0.58,
    "chest_y":     0.60,   # spine joint
    "neck_y":      0.78,   # head above this
}

BONES = [  # name, parent
    ("root", -1),          # hips
    ("spine", 0),
    ("head", 1),
    ("upperarm_L", 1), ("forearm_L", 3),
    ("upperarm_R", 1), ("forearm_R", 5),
    ("thigh_L", 0), ("shin_L", 7),
    ("thigh_R", 0), ("shin_R", 9),
]
NAME = {n: i for i, (n, _) in enumerate(BONES)}

# effector positions (FBX space, fractions of H) of chains without an end bone
EFFECTORS = {
    "forearm_L": ("upperarm_L", (RIG["shoulder_x"], 0.42, 0)),
    "forearm_R": ("upperarm_R", (-RIG["shoulder_x"], 0.42, 0)),
    "shin_L": ("thigh_L", (RIG["hip_x"], 0.0, 0)),
    "shin_R": ("thigh_R", (-RIG["hip_x"], 0.0, 0)),
}


def joint_positions(H, R=RIG):
    """Bone joint positions (FBX space) for a model of height H."""
    j = {
        "root":  (0, R["hip_y"] * H, 0),
        "spine": (0, R["chest_y"] * H, 0),
        "head":  (0, R["neck_y"] * H, 0),
    }
    for side, sx in (("L", 1), ("R", -1)):
        j["upperarm_" + side] = (sx * R["shoulder_x"] * H, R["shoulder_y"] * H, 0)
        j["forearm_" + side] = (sx * R["shoulder_x"] * H, R["elbow_y"] * H, 0)
        j["thigh_" + side] = (sx * R["hip_x"] * H, R["hip_y"] * H, 0)
        j["shin_" + side] = (sx * R["hip_x"] * H, R["knee_y"] * H, 0)
    return [np.array(j[n], dtype=np.float64) for n, _ in BONES]


def assign_bones(P, H, R=RIG, T=None):
    """bone index per position (rigid segmentation).  Arms are tested first:
    the hands hang below the hip line, so a hip-first rule would glue them
    to the thighs.  With T (triangles) isolated mis-assignments are fixed by
    a neighbour majority vote."""
    out = np.zeros(len(P), dtype=np.int64)
    for i, (x, y, z) in enumerate(P):
        side = "L" if x >= 0 else "R"
        ax = abs(x)
        if R["hand_min_y"] * H <= y < (R["shoulder_y"] + 0.02) * H and (
                (y >= R["elbow_y"] * H and ax > R["arm_min_x_u"] * H) or
                (y < R["elbow_y"] * H and ax > R["arm_min_x"] * H)):
            out[i] = NAME["upperarm_" + side] if y >= R["elbow_y"] * H else NAME["forearm_" + side]
        elif y < R["hip_y"] * H:
            out[i] = NAME["thigh_" + side] if y >= R["knee_y"] * H else NAME["shin_" + side]
        elif y >= R["neck_y"] * H:
            out[i] = NAME["head"]
        elif y >= R["chest_y"] * H:
            out[i] = NAME["spine"]
        else:
            out[i] = NAME["root"]
    if T is not None:
        nbrs = [set() for _ in range(len(P))]
        for a, b, c in T:
            nbrs[a].update((b, c)); nbrs[b].update((a, c)); nbrs[c].update((a, b))
        for _ in range(3):
            changed = 0
            new = out.copy()
            for i in range(len(P)):
                if not nbrs[i]:
                    continue
                votes = np.bincount([out[j] for j in nbrs[i]], minlength=len(BONES))
                same = votes[out[i]]
                best = int(np.argmax(votes))
                # isolated vertex: fewer than a third of its neighbours agree
                if same * 3 < len(nbrs[i]) and best != out[i]:
                    new[i] = best
                    changed += 1
            out = new
            if not changed:
                break
    return out


# --------------------------------------------------------------------------
# procedural animation
# --------------------------------------------------------------------------
def rot(axis, deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    m = np.eye(4)
    if axis == "x":
        m[1, 1], m[1, 2], m[2, 1], m[2, 2] = c, -s, s, c
    elif axis == "y":
        m[0, 0], m[0, 2], m[2, 0], m[2, 2] = c, s, -s, c
    else:
        m[0, 0], m[0, 1], m[1, 0], m[1, 1] = c, -s, s, c
    return m


def trans(v):
    m = np.eye(4)
    m[:3, 3] = v
    return m


def _sin(t, period, phase=0.0):
    return math.sin(2 * math.pi * (t / period + phase))


def _pose_idle(t, H):
    b = _sin(t, 3.0)
    return {
        "spine": rot("x", 2 * b),
        "head": rot("x", -2 * b) @ rot("y", 4 * _sin(t, 6.0)),
        "upperarm_L": rot("z", 3 * b), "upperarm_R": rot("z", -3 * b),
        "root_t": (0, 0.004 * H * b, 0),
    }


def _pose_walk(t, H, period=1.0, swing=30.0, arm=25.0, bob=0.012, knee=45.0):
    ph = t / period
    # forward swing of the thigh = rotation about x moving the foot to +z
    l = -swing * math.sin(2 * math.pi * ph)
    r = -l
    # knee bends while the leg swings forward / lifts
    kl = knee * max(0.0, math.sin(2 * math.pi * (ph + 0.15)))
    kr = knee * max(0.0, math.sin(2 * math.pi * (ph + 0.65)))
    al = -arm * math.sin(2 * math.pi * ph)   # arms opposite to the same-side leg
    return {
        "thigh_L": rot("x", l), "thigh_R": rot("x", r),
        "shin_L": rot("x", kl), "shin_R": rot("x", kr),
        "upperarm_L": rot("x", -al), "upperarm_R": rot("x", al),
        "forearm_L": rot("x", -20), "forearm_R": rot("x", -20),
        "spine": rot("y", 6 * math.sin(2 * math.pi * ph)) @ rot("x", -3),
        "head": rot("y", -6 * math.sin(2 * math.pi * ph)),
        "root_t": (0, bob * H * abs(math.sin(2 * math.pi * ph)), 0),
    }


def _pose_run(t, H):
    return _pose_walk(t, H, period=0.6, swing=45.0, arm=45.0, bob=0.03, knee=70.0)


def _pose_wave(t, H):
    return {
        "upperarm_R": rot("z", 150) @ rot("x", 0),
        "forearm_R": rot("z", 30 * _sin(t, 0.8)),
        "head": rot("z", 8 * _sin(t, 1.6)),
        "spine": rot("z", 3 * _sin(t, 1.6)),
    }


def _pose_punch(t, H, period=0.6):
    p = (t / period) % 1.0
    # left arm punches forward on the first half, right on the second
    def punch(u):
        e = max(0.0, math.sin(math.pi * u)) if 0 <= u < 1 else 0.0
        return e
    el = punch(p * 2), punch((p - 0.5) * 2)
    out = {"spine": rot("y", 15 * (el[0] - el[1])) @ rot("x", -8)}
    for side, e, s in (("L", el[0], 1), ("R", el[1], -1)):
        out["upperarm_" + side] = rot("x", -90 * e - 20) @ rot("z", s * 20)
        out["forearm_" + side] = rot("x", -110 * (1 - e) - 10)
    out["thigh_L"] = rot("x", -10) @ rot("z", 10)
    out["thigh_R"] = rot("x", 10) @ rot("z", -10)
    out["shin_L"] = rot("x", 15)
    out["shin_R"] = rot("x", 15)
    return out


ANIMS = [
    ("idle", 3.0, _pose_idle),
    ("walk", 1.0, _pose_walk),
    ("run", 0.6, _pose_run),
    ("wave", 1.6, _pose_wave),
    ("punch", 1.2, _pose_punch),
]


class SynthAnim:
    """Mimics the AnimStack interface used by fbx2ps1.convert: name,
    duration, and bone_local(b, t) -> local 4x4 in FBX space relative to the
    parent bone (rotation about the joint, applied after the bind offset)."""

    def __init__(self, name, duration, pose_fn, bind_local, H):
        self.name = name
        self.duration = duration
        self.pose_fn = pose_fn
        self.bind_local = bind_local
        self.H = H

    def bone_local(self, b, t):
        pose = self.pose_fn(t, self.H)
        name = BONES[b][0]
        m = self.bind_local[b]
        if b == 0 and "root_t" in pose:
            m = m @ trans(pose["root_t"])
        r = pose.get(name)
        return m @ r if r is not None else m
