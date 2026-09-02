#!/usr/bin/env python3
"""Minimal FBX binary (v7.x) parser using only the Python standard library.

Exposes a node tree plus helpers to resolve the object graph (Connections,
Properties70).  Enough for meshes, skeletons, skin clusters and animation
curves exported by common tools.
"""
import struct
import zlib

FBX_TIME_PER_SEC = 46186158000


class Node:
    __slots__ = ("name", "props", "children")

    def __init__(self, name, props, children):
        self.name = name
        self.props = props
        self.children = children

    def find(self, name):
        return [c for c in self.children if c.name == name]

    def first(self, name, default=None):
        for c in self.children:
            if c.name == name:
                return c
        return default

    def __repr__(self):
        return "Node(%s, %r, %d children)" % (self.name, self.props[:3], len(self.children))


class FbxFile:
    def __init__(self, path):
        with open(path, "rb") as f:
            self.data = f.read()
        if not self.data.startswith(b"Kaydara FBX Binary"):
            raise ValueError("not an FBX binary file: %s" % path)
        self.version = struct.unpack_from("<I", self.data, 23)[0]
        self._big = self.version >= 7500
        self.root = self._parse_root()
        self.objects = {}      # id -> Node
        self.connections = []  # (kind, src, dst, prop)
        self._index_objects()

    # ---- low-level parsing -------------------------------------------------
    def _read_array(self, pos, fmt):
        n, enc, clen = struct.unpack_from("<III", self.data, pos)
        pos += 12
        raw = self.data[pos:pos + clen]
        pos += clen
        if enc:
            raw = zlib.decompress(raw)
        return list(struct.unpack("<%d%s" % (n, fmt), raw)), pos

    def _read_prop(self, pos):
        t = chr(self.data[pos])
        pos += 1
        if t in "YCIFDL":
            f = {"Y": "<h", "C": "<?", "I": "<i", "F": "<f", "D": "<d", "L": "<q"}[t]
            return struct.unpack_from(f, self.data, pos)[0], pos + struct.calcsize(f)
        if t in "SR":
            ln = struct.unpack_from("<I", self.data, pos)[0]
            pos += 4
            v = self.data[pos:pos + ln]
            return (v.decode("latin1") if t == "S" else v), pos + ln
        if t in "fdlib":
            return self._read_array(pos, {"f": "f", "d": "d", "l": "q", "i": "i", "b": "?"}[t])
        raise ValueError("unknown property type %r at %d" % (t, pos))

    def _read_node(self, pos):
        if self._big:
            end, nprops, _plen = struct.unpack_from("<QQQ", self.data, pos)
            pos += 24
        else:
            end, nprops, _plen = struct.unpack_from("<III", self.data, pos)
            pos += 12
        nlen = self.data[pos]
        pos += 1
        name = self.data[pos:pos + nlen].decode("latin1")
        pos += nlen
        if end == 0:
            return None, pos
        props = []
        for _ in range(nprops):
            v, pos = self._read_prop(pos)
            props.append(v)
        children = []
        while pos < end:
            n, pos = self._read_node(pos)
            if n is None:
                break
            children.append(n)
        return Node(name, props, children), end

    def _parse_root(self):
        pos = 27
        nodes = []
        while pos < len(self.data):
            n, pos = self._read_node(pos)
            if n is None:
                break
            nodes.append(n)
        return Node("", [], nodes)

    # ---- object graph ------------------------------------------------------
    def _index_objects(self):
        objs = self.root.first("Objects")
        for n in objs.children:
            if n.props and isinstance(n.props[0], int):
                self.objects[n.props[0]] = n
        conns = self.root.first("Connections")
        for c in conns.children:
            kind = c.props[0]
            src, dst = c.props[1], c.props[2]
            prop = c.props[3] if len(c.props) > 3 else None
            self.connections.append((kind, src, dst, prop))
        self._by_dst = {}
        self._by_src = {}
        for kind, src, dst, prop in self.connections:
            self._by_dst.setdefault(dst, []).append((kind, src, prop))
            self._by_src.setdefault(src, []).append((kind, dst, prop))

    def children_of(self, oid, node_name=None, kind=None):
        """Objects connected *to* oid (i.e. src objects whose dst is oid)."""
        out = []
        for k, src, prop in self._by_dst.get(oid, []):
            if kind and k != kind:
                continue
            n = self.objects.get(src)
            if n is None:
                continue
            if node_name and n.name != node_name:
                continue
            out.append((n, prop))
        return out

    def parents_of(self, oid, node_name=None):
        out = []
        for k, dst, prop in self._by_src.get(oid, []):
            n = self.objects.get(dst)
            if n is None:
                continue
            if node_name and n.name != node_name:
                continue
            out.append((n, prop))
        return out

    def by_type(self, node_name, subclass=None):
        out = []
        for n in self.objects.values():
            if n.name != node_name:
                continue
            if subclass is not None and (len(n.props) < 3 or n.props[2] != subclass):
                continue
            out.append(n)
        return out


def obj_id(node):
    return node.props[0]


def obj_name(node):
    return node.props[1].split("\x00")[0]


def obj_subclass(node):
    return node.props[2] if len(node.props) > 2 else ""


def properties70(node):
    """Return dict name -> list of values (from a Properties70 block)."""
    out = {}
    p70 = node.first("Properties70")
    if p70 is None:
        return out
    for p in p70.children:
        if p.name != "P":
            continue
        out[p.props[0]] = p.props[4:]
    return out


def prop_vec3(props, name, default=(0.0, 0.0, 0.0)):
    v = props.get(name)
    if v is None or len(v) < 3:
        return tuple(default)
    return (float(v[0]), float(v[1]), float(v[2]))


def prop_scalar(props, name, default=0):
    v = props.get(name)
    if not v:
        return default
    return v[0]
