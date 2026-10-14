"""Read and rewrite the skeletal mesh of a Combat Arms BR `.ltb` model (D3D model, header v9, file version 25).

Layout derived from the engine source (ENGINE/runtime/model/src/model_load.cpp and
render_a/src/sys/d3d/d3dmeshrendobj_skel.cpp) and checked against real files:

    20  LTB_Header (u8 type, pad, u16 version, u8, pad, 3 x u32)
    u32 file version (25), 15 x u32 allocation counts
    string command string, f32 visibility radius, u32 OBB count (0), u32 piece count
    per piece: string name, u32 nLODs, f32[nLODs] distances, 2 x u32 (unused)
      per LOD: u32 nTextures, i32[4] texture ids, i32 renderstyle, u8 priority, u32 object type,
               u32 object size, then the mesh object (below), u8 used-node count + bytes
    ... nodes, weight sets, child models, sockets, animations (kept as opaque bytes here)
    strings are u16 length + characters.

Mesh object (direct-render path, no matrix palettes):
    u32 vertCount, u32 polyCount, u32 maxBonesPerTri, u32 maxBonesPerVert, u8 reindexed,
    u32[4] stream flags, u8 useMatrixPalettes,
    vertices (stream 0) = 36 bytes each for flags 0x13: pos f32x3, blend f32 x (bonesPerTri-1), normal f32x3, uv f32x2
    u16 indices[polyCount*3]
    u32 boneSetCount, then boneSetCount x 12 bytes: u16 firstVert, u16 vertCount, u8 bones[4], u32 indexIntoIndexBuffer.

Only the mesh object is parsed; every other byte is carried through untouched, so unchanged files round-trip
byte for byte. Nothing here changes how the game loads files: it only reads and writes the same structures.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field

import numpy as np

HEADER = 20


@dataclass
class BoneSet:
    first_vert: int
    vert_count: int
    bones: bytes  # 4 bytes, 0xFF = unused
    index_end: int  # index into the index buffer (in indices, not triangles) where this set's triangles end


@dataclass
class SkelMesh:
    max_bones_per_tri: int
    max_bones_per_vert: int
    reindexed: int
    stream_flags: tuple
    use_palettes: int
    pos: np.ndarray       # N,3 float32
    blend: np.ndarray     # N,(bones_per_tri-1) float32
    normal: np.ndarray    # N,3
    uv: np.ndarray        # N,2
    tris: np.ndarray      # M,3 uint16
    bonesets: list[BoneSet] = field(default_factory=list)

    @property
    def n_verts(self):
        return len(self.pos)

    @property
    def n_tris(self):
        return len(self.tris)


@dataclass
class LtbFile:
    prefix: bytes       # everything before the mesh object's size field
    mesh_off: int       # offset of the mesh object size field
    obj_size: int       # original size field
    mesh: SkelMesh
    suffix: bytes       # everything after the mesh object
    alloc_off: int = 24  # allocation block offset


def _string(b, p):
    n, = struct.unpack_from('<H', b, p)
    return b[p + 2:p + 2 + n].decode('latin1'), p + 2 + n


def parse(data: bytes, piece: int = 0) -> LtbFile:
    if data[0] != 1 or struct.unpack_from('<H', data, 2)[0] != 9:
        raise ValueError('not a D3D model LTB v9')
    ver, = struct.unpack_from('<I', data, HEADER)
    if ver != 25:
        raise ValueError(f'file version {ver} not supported (only 25 was checked)')
    p = HEADER + 4 + 60
    _, p = _string(data, p)                    # command string
    p += 4                                     # visibility radius
    nobb, = struct.unpack_from('<I', data, p); p += 4
    if nobb:
        raise ValueError('OBBs not supported')
    npieces, = struct.unpack_from('<I', data, p); p += 4
    for i in range(npieces):
        name, p = _string(data, p)
        nlod, = struct.unpack_from('<I', data, p); p += 4 + 4 * nlod + 8
        if nlod != 1:
            raise ValueError('only single-LOD pieces are handled')
        p += 4 + 16 + 4 + 1 + 4                # nTextures, texture ids, renderstyle, priority, object type
        if i == piece:
            break
    off = p
    obj_size, = struct.unpack_from('<I', data, p); p += 4
    nv, npoly, mbt, mbv = struct.unpack_from('<4I', data, p); p += 16
    reidx = data[p]; p += 1
    flags = struct.unpack_from('<4I', data, p); p += 16
    palettes = data[p]; p += 1
    if palettes or flags[0] != 0x13 or flags[1:] != (0, 0, 0):
        raise ValueError(f'unsupported mesh variant: palettes={palettes} flags={flags}')
    nblend = mbt - 1
    vsize = 4 * (3 + nblend + 3 + 2)
    raw = np.frombuffer(data, np.uint8, nv * vsize, p).reshape(nv, vsize)
    f = raw.view(np.float32).reshape(nv, vsize // 4)
    pos, blend, normal, uv = f[:, 0:3], f[:, 3:3 + nblend], f[:, 3 + nblend:6 + nblend], f[:, 6 + nblend:8 + nblend]
    p += nv * vsize
    tris = np.frombuffer(data, np.uint16, npoly * 3, p).reshape(npoly, 3)
    p += npoly * 6
    nbs, = struct.unpack_from('<I', data, p); p += 4
    bs = []
    for _ in range(nbs):
        fv, vc = struct.unpack_from('<HH', data, p)
        bs.append(BoneSet(fv, vc, bytes(data[p + 4:p + 8]), struct.unpack_from('<I', data, p + 8)[0]))
        p += 12
    end = off + 4 + obj_size
    mesh = SkelMesh(mbt, mbv, reidx, flags, palettes, pos.copy(), blend.copy(), normal.copy(), uv.copy(), tris.copy(), bs)
    return LtbFile(bytes(data[:off]), off, obj_size, mesh, bytes(data[end:])), p - (off + 4), obj_size


def serialize_mesh(m: SkelMesh) -> bytes:
    nblend = m.max_bones_per_tri - 1
    out = bytearray()
    out += struct.pack('<4I', m.n_verts, m.n_tris, m.max_bones_per_tri, m.max_bones_per_vert)
    out += bytes([m.reindexed]) + struct.pack('<4I', *m.stream_flags) + bytes([m.use_palettes])
    v = np.concatenate([m.pos, m.blend.reshape(m.n_verts, nblend), m.normal, m.uv], axis=1).astype('<f4')
    out += v.tobytes()
    out += m.tris.astype('<u2').tobytes()
    out += struct.pack('<I', len(m.bonesets))
    for b in m.bonesets:
        out += struct.pack('<HH', b.first_vert, b.vert_count) + b.bones + struct.pack('<I', b.index_end)
    return bytes(out)


def serialize(f: LtbFile, patch_allocations: bool = True) -> bytes:
    body = serialize_mesh(f.mesh)
    data = bytearray(f.prefix) + struct.pack('<I', len(body)) + body + f.suffix
    return bytes(data)


# ---------------------------------------------------------------------------------------------------------------------
# General models: several pieces, each either a skinned mesh (object type 5, parsed above) or a rigid mesh
# (object type 4: one bone per piece, no bone sets). Verified on PV_SR_L96A1_SH (1 skinned piece) and
# PV_SR_L96A1_SUPER_MAGNUM_SH (3 rigid pieces): the computed object sizes match the files to the byte.
#
# Rigid mesh object: u32 size, u32 vertCount, u32 polyCount, u32 maxBonesPerTri, u32 maxBonesPerVert,
#   u32[4] stream flags, u32 boneEffector, vertices (flags 0x13 = pos f32x3, normal f32x3, uv f32x2 = 32 bytes),
#   u16 indices[polyCount*3].
# ---------------------------------------------------------------------------------------------------------------------


@dataclass
class RigidMesh:
    max_bones_per_tri: int
    max_bones_per_vert: int
    stream_flags: tuple
    bone_effector: int
    pos: np.ndarray
    normal: np.ndarray
    uv: np.ndarray
    tris: np.ndarray

    @property
    def n_verts(self):
        return len(self.pos)

    @property
    def n_tris(self):
        return len(self.tris)


@dataclass
class Model:
    segments: list   # opaque bytes between/around the meshes; len(segments) == len(meshes) + 1
    meshes: list     # SkelMesh or RigidMesh, one per (piece, LOD), in file order
    names: list
    lods: list       # LOD index of each mesh (0 = most detailed)


def _parse_rigid(data, p):
    nv, npoly, mbt, mbv = struct.unpack_from('<4I', data, p + 4)
    flags = struct.unpack_from('<4I', data, p + 20)
    be, = struct.unpack_from('<I', data, p + 36)
    if flags != (0x13, 0, 0, 0):
        raise ValueError(f'unsupported rigid vertex format {flags}')
    q = p + 40
    f = np.frombuffer(data, np.float32, nv * 8, q).reshape(nv, 8)
    q += nv * 32
    tris = np.frombuffer(data, np.uint16, npoly * 3, q).reshape(npoly, 3)
    return RigidMesh(mbt, mbv, flags, be, f[:, 0:3].copy(), f[:, 3:6].copy(), f[:, 6:8].copy(), tris.copy())


def _serialize_rigid(m: RigidMesh) -> bytes:
    body = struct.pack('<4I', m.n_verts, m.n_tris, m.max_bones_per_tri, m.max_bones_per_vert)
    body += struct.pack('<4I', *m.stream_flags) + struct.pack('<I', m.bone_effector)
    body += np.concatenate([m.pos, m.normal, m.uv], axis=1).astype('<f4').tobytes()
    body += m.tris.astype('<u2').tobytes()
    return body


def parse_model(data: bytes) -> Model:
    if data[0] != 1 or struct.unpack_from('<H', data, 2)[0] != 9:
        raise ValueError('not a D3D model LTB v9')
    if struct.unpack_from('<I', data, HEADER)[0] != 25:
        raise ValueError('only file version 25 was checked')
    p = HEADER + 4 + 60
    _, p = _string(data, p)
    p += 4
    if struct.unpack_from('<I', data, p)[0]:
        raise ValueError('OBBs not supported')
    p += 4
    npieces, = struct.unpack_from('<I', data, p); p += 4
    segs, meshes, names, lods, seg_start = [], [], [], [], 0
    for _ in range(npieces):
        name, p = _string(data, p)
        nlod, = struct.unpack_from('<I', data, p); p += 4 + 4 * nlod + 8
        for li in range(nlod):
            objtype, = struct.unpack_from('<I', data, p + 25); p += 29
            osz, = struct.unpack_from('<I', data, p)
            segs.append(bytes(data[seg_start:p]))
            if objtype == 5:
                meshes.append(_parse_skel_at(data, p)[0])
            elif objtype == 4:
                meshes.append(_parse_rigid(data, p))
            else:
                raise ValueError(f'unknown render object type {objtype}')
            names.append(name); lods.append(li)
            seg_start = p + 4 + osz
            p = seg_start
            p += 1 + data[p]          # used-node list (u8 count + bytes) stays inside the next opaque segment
    segs.append(bytes(data[seg_start:]))
    return Model(segs, meshes, names, lods)


def _parse_skel_at(data, p):
    """Parse a skinned mesh whose object-size field is at offset p."""
    # reuse parse() machinery by building the same checks inline
    osz, = struct.unpack_from('<I', data, p); q = p + 4
    nv, npoly, mbt, mbv = struct.unpack_from('<4I', data, q); q += 16
    reidx = data[q]; q += 1
    flags = struct.unpack_from('<4I', data, q); q += 16
    palettes = data[q]; q += 1
    if palettes or flags[0] != 0x13 or flags[1:] != (0, 0, 0):
        raise ValueError(f'unsupported skinned mesh variant: palettes={palettes} flags={flags}')
    nblend = mbt - 1
    vsize = 4 * (3 + nblend + 3 + 2)
    f = np.frombuffer(data, np.uint8, nv * vsize, q).reshape(nv, vsize).view(np.float32).reshape(nv, vsize // 4)
    pos, blend = f[:, 0:3], f[:, 3:3 + nblend]
    normal, uv = f[:, 3 + nblend:6 + nblend], f[:, 6 + nblend:8 + nblend]
    q += nv * vsize
    tris = np.frombuffer(data, np.uint16, npoly * 3, q).reshape(npoly, 3); q += npoly * 6
    nbs, = struct.unpack_from('<I', data, q); q += 4
    bs = []
    for _ in range(nbs):
        fv, vc = struct.unpack_from('<HH', data, q)
        bs.append(BoneSet(fv, vc, bytes(data[q + 4:q + 8]), struct.unpack_from('<I', data, q + 8)[0])); q += 12
    m = SkelMesh(mbt, mbv, reidx, flags, palettes, pos.copy(), blend.copy(), normal.copy(), uv.copy(), tris.copy(), bs)
    return m, q - (p + 4), osz


def serialize_model(model: Model, patch_tri_total: bool = True) -> bytes:
    out = bytearray()
    for i, m in enumerate(model.meshes):
        out += model.segments[i]
        body = serialize_mesh(m) if isinstance(m, SkelMesh) else _serialize_rigid(m)
        out += struct.pack('<I', len(body)) + body
    out += model.segments[-1]
    if patch_tri_total:  # allocation block, 6th count (only read by the engine, never used to size buffers)
        struct.pack_into('<I', out, 44, sum(m.n_tris for m in model.meshes))
    return bytes(out)
