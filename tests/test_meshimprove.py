import numpy as np

from ca_pipeline import meshimprove
from ca_pipeline.ltbmesh import BoneSet, SkelMesh


def _ring_mesh(n=8):
    """A short open tube with n sides (smooth normals) plus a flat plate, one bone set each."""
    pos, nrm, uv, tris = [], [], [], []
    for z in (0.0, 1.0):
        for i in range(n):
            a = 2 * np.pi * i / n
            pos.append((np.cos(a), np.sin(a), z)); nrm.append((np.cos(a), np.sin(a), 0)); uv.append((i / n, z))
    for i in range(n):
        j = (i + 1) % n
        tris += [(i, j, n + i), (j, n + j, n + i)]
    tube_v = len(pos)
    return SkelMesh(2, 1, 0, (0x13, 0, 0, 0), 0, np.array(pos, np.float32), np.zeros((tube_v, 1), np.float32),
                    np.array(nrm, np.float32), np.array(uv, np.float32), np.array(tris, np.uint16),
                    [BoneSet(0, tube_v, bytes([1, 255, 255, 255]), len(tris) * 3)])


def test_structure_is_valid_and_counts_grow():
    m = _ring_mesh()
    assert meshimprove.check(m) == []
    out, stats = meshimprove.improve(m)
    assert meshimprove.check(out) == []
    assert out.n_tris == 4 * m.n_tris and out.n_verts > m.n_verts


def test_original_vertices_unchanged_and_bounds_not_exceeded():
    m = _ring_mesh()
    out, _ = meshimprove.improve(m)
    assert np.array_equal(out.pos[: m.n_verts], m.pos)
    assert np.array_equal(out.normal[: m.n_verts], m.normal)
    # new points on a curved tube move outward at most to the true circle, never past it
    r = np.hypot(out.pos[:, 0], out.pos[:, 1])
    assert r.max() <= 1.0 + 1e-3


def test_hard_edges_stay_flat():
    # a box-like fold: two faces at 90 degrees must not be rounded
    pos = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 1]], np.float32)
    nrm = np.array([[0, 0, 1]] * 4 + [[0, -1, 0]] * 2, np.float32)
    tris = np.array([[0, 1, 2], [1, 3, 2], [0, 4, 1], [1, 4, 5]], np.uint16)
    m = SkelMesh(2, 1, 0, (0x13, 0, 0, 0), 0, pos, np.zeros((6, 1), np.float32), nrm, np.zeros((6, 2), np.float32),
                 tris, [BoneSet(0, 6, bytes([1, 255, 255, 255]), 12)])
    out, stats = meshimprove.improve(m)
    assert stats["max_disp"] < 1e-6
