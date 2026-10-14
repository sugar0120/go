"""Conservative geometry upgrade for a Combat Arms skeletal mesh (see ltbmesh.py).

One pass of 1->4 triangle subdivision where:
  * every original vertex keeps its exact position, normal, UV and blend weight (so rigging, animation and
    anything attached to the model line up exactly as before);
  * each new edge-midpoint is pulled onto the curve implied by the artist's own vertex normals
    (the cubic-Bezier midpoint used by PN triangles: mid = (a+b)/2 - (dot(b-a,na)*na + dot(a-b,nb)*nb)/8);
  * edges where the two adjacent faces differ by more than CREASE_DEG (hard edges), open edges and flat regions
    get a plain midpoint, so hard-surface detail is not rounded off;
  * UV seams stay seamless: the displacement is computed per welded (position-identical) edge, so both sides
    of a seam move identically while keeping their own UVs;
  * new vertices are placed inside the bone set (vertex range) of the triangle that created them, and the
    triangle groups of each bone set stay contiguous, which the direct-render path of the engine requires.
"""

from __future__ import annotations

import numpy as np

from .ltbmesh import BoneSet, SkelMesh

CREASE_DEG = 30.0   # edges whose two faces differ by more than this stay flat (hard edges)


def _unit(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, 1e-12)


def improve(m: SkelMesh) -> tuple[SkelMesh, dict]:
    pos, nrm, uv, blend = m.pos.astype(np.float64), m.normal.astype(np.float64), m.uv, m.blend
    # welded id per vertex (identical position -> same id)
    key = np.round(pos, 5)
    _, weld = np.unique(key, axis=0, return_inverse=True)
    weld = weld.reshape(-1)

    # 1. classify welded edges from the TRUE face angles (the artist's vertex normals are heavily smoothed,
    #    so they cannot tell a hard corner from a curve). Only edges shared by exactly two faces whose normals
    #    differ by less than CREASE_DEG are curved; each gets a displacement cap equal to the sagitta that the
    #    two faces' own angle implies, so nothing can bulge more than the geometry itself says.
    t_all = m.tris.astype(np.int64)
    fn = np.cross(pos[t_all[:, 1]] - pos[t_all[:, 0]], pos[t_all[:, 2]] - pos[t_all[:, 0]])
    fnl = np.linalg.norm(fn, axis=1)
    fn = fn / np.maximum(fnl[:, None], 1e-30)
    edge_faces: dict[tuple, list] = {}
    for fi, t in enumerate(t_all):
        if fnl[fi] < 1e-14:
            continue  # degenerate triangle: ignore for classification
        for i in range(3):
            wa, wb = int(weld[t[i]]), int(weld[t[(i + 1) % 3]])
            if wa != wb:
                edge_faces.setdefault((wa, wb) if wa < wb else (wb, wa), []).append(fi)
    smooth = {}
    cap_angle = {}
    for k, fl in edge_faces.items():
        if len(fl) == 2:
            ang = float(np.arccos(np.clip(np.dot(fn[fl[0]], fn[fl[1]]), -1, 1)))
            if ang < np.radians(CREASE_DEG):
                smooth[k] = True
                cap_angle[k] = ang
                continue
        smooth[k] = False
    nz = np.linalg.norm(nrm, axis=1) < 0.5
    if nz.any():  # a few vertices carry a zero normal: fall back to the average face normal there
        acc = np.zeros_like(nrm)
        for c in range(3):
            np.add.at(acc, t_all[:, c], fn)
        nrm = nrm.copy()
        nrm[nz] = _unit(acc[nz])

    stats = {"edges": len(smooth), "smooth_edges": int(sum(smooth.values())), "displaced": 0, "clamped": 0,
             "max_disp": 0.0}

    new_pos, new_nrm, new_uv, new_blend, new_tris, new_sets = [], [], [], [], [], []
    prev_end = 0
    for bs in m.bonesets:
        lo, hi = bs.first_vert, bs.first_vert + bs.vert_count
        tris = m.tris[prev_end // 3: bs.index_end // 3].astype(np.int64)
        prev_end = bs.index_end
        base = len(new_pos)  # first vertex of this set in the output
        new_pos.extend(pos[lo:hi]); new_nrm.extend(nrm[lo:hi]); new_uv.extend(uv[lo:hi]); new_blend.extend(blend[lo:hi])
        mids: dict[tuple, int] = {}

        def mid(a, b):
            k = (a, b) if a < b else (b, a)
            if k in mids:
                return mids[k]
            wa, wb = int(weld[a]), int(weld[b])
            pa, pb, na, nb = pos[a], pos[b], nrm[a], nrm[b]
            p = (pa + pb) / 2
            n = _unit((na + nb) / 2 + 1e-12)
            wk = (wa, wb) if wa < wb else (wb, wa)
            if wa != wb and smooth.get(wk, False):
                d = (np.dot(pb - pa, na) * na + np.dot(pa - pb, nb) * nb) / 8.0
                length = np.linalg.norm(pb - pa)
                cap = 0.5 * length * np.tan(cap_angle[wk] / 4.0)  # sagitta implied by the two faces' own angle
                dn = np.linalg.norm(d)
                if dn > cap:
                    d = d * (cap / dn) if dn > 0 else d
                    stats["clamped"] += 1
                p = p - d
                if dn > 1e-9:
                    stats["displaced"] += 1
                stats["max_disp"] = max(stats["max_disp"], float(np.linalg.norm(d)))
            idx = len(new_pos)
            new_pos.append(p); new_nrm.append(n); new_uv.append((uv[a] + uv[b]) / 2); new_blend.append((blend[a] + blend[b]) / 2)
            mids[k] = idx
            return idx

        for a, b, c in tris:
            ma, mb, mc = mid(a, b), mid(b, c), mid(c, a)
            a2, b2, c2 = a - lo + base, b - lo + base, c - lo + base
            new_tris += [(a2, ma, mc), (ma, b2, mb), (mc, mb, c2), (ma, mb, mc)]
        new_sets.append(BoneSet(base, len(new_pos) - base, bs.bones, len(new_tris) * 3))

    out = SkelMesh(
        m.max_bones_per_tri, m.max_bones_per_vert, m.reindexed, m.stream_flags, m.use_palettes,
        np.asarray(new_pos, np.float32), np.asarray(new_blend, np.float32).reshape(len(new_pos), -1),
        np.asarray(new_nrm, np.float32), np.asarray(new_uv, np.float32), np.asarray(new_tris, np.uint16), new_sets)
    assert out.n_verts < 65536, "exceeds 16-bit index range"
    return out, stats


def check(m: SkelMesh) -> list[str]:
    """Structural checks the engine's loader/renderer relies on. Returns a list of problems (empty = OK)."""
    bad = []
    if m.tris.size and int(m.tris.max()) >= m.n_verts:
        bad.append("index out of range")
    if not np.isfinite(m.pos).all() or not np.isfinite(m.normal).all():
        bad.append("non-finite values")
    prev, tot = 0, 0
    for i, s in enumerate(m.bonesets):
        t = m.tris[prev // 3: s.index_end // 3]
        lo, hi = s.first_vert, s.first_vert + s.vert_count
        if len(t) and not ((t >= lo) & (t < hi)).all():
            bad.append(f"bone set {i}: triangle uses a vertex outside its range")
        prev, tot = s.index_end, tot + s.vert_count
    if tot != m.n_verts:
        bad.append("bone set vertex counts do not sum to the vertex count")
    if prev != 3 * m.n_tris:
        bad.append("last bone set index_end != 3 * triangle count")
    nl = np.linalg.norm(m.normal, axis=1)
    if not ((np.abs(nl - 1.0) < 1e-2) | (nl < 1e-6)).all():
        bad.append("normals neither unit length nor zero")
    return bad
