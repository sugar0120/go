"""Automated checks for upgraded assets before they go back into REZ files."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .config import ENGINE_LIMITS, EngineLimits

# Bytes per pixel for the DXT/BC formats DTX supports; mips add ~1/3.
BYTES_PER_PIXEL = {"DXT1": 0.5, "DXT3": 1.0, "DXT5": 1.0, "BGRA32": 4.0}


def texture_bytes(width: int, height: int, fmt: str = "DXT5", mips: bool = True) -> int:
    size = width * height * BYTES_PER_PIXEL[fmt]
    return int(size * 4 / 3 if mips else size)


def pick_format(has_alpha: bool, alpha_cutout: bool = False) -> str:
    """DXT1 for opaque or 1-bit alpha, DXT5 for smooth alpha."""
    if not has_alpha or alpha_cutout:
        return "DXT1"
    return "DXT5"


@dataclass
class Issue:
    path: str
    check: str
    detail: str


def _is_pow2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def check_texture(original: Image.Image, upgraded: Image.Image, path: str = "",
                  role: str = "diffuse", limits: EngineLimits = ENGINE_LIMITS) -> list[Issue]:
    issues: list[Issue] = []
    w, h = upgraded.size
    if limits.require_power_of_two and not (_is_pow2(w) and _is_pow2(h)):
        issues.append(Issue(path, "power_of_two", f"{w}x{h}"))
    if max(w, h) > limits.max_texture_size:
        issues.append(Issue(path, "max_size", f"{w}x{h} > {limits.max_texture_size}"))
    ow, oh = original.size
    if not math.isclose(ow / oh, w / h, rel_tol=0.01):
        issues.append(Issue(path, "aspect_ratio", f"{ow}x{oh} -> {w}x{h}"))
    orig_alpha = original.mode in ("RGBA", "LA", "PA") or "transparency" in original.info
    if orig_alpha and upgraded.mode != "RGBA":
        issues.append(Issue(path, "alpha_lost", upgraded.mode))
    if role == "normal":
        v = np.asarray(upgraded.convert("RGB"), np.float32) / 127.5 - 1.0
        length = float(np.linalg.norm(v, axis=-1).mean())
        if abs(length - 1.0) > 0.05:
            issues.append(Issue(path, "normal_length", f"mean |n| = {length:.3f}"))
    if w <= ow and h <= oh:
        issues.append(Issue(path, "not_upscaled", f"{ow}x{oh} -> {w}x{h}"))
    return issues


@dataclass
class MeshStats:
    vertices: int
    faces: int
    triangles: int


def obj_stats(path: str | Path) -> MeshStats:
    """Count vertices/faces/triangles in a Wavefront OBJ (n-gons fan-triangulated)."""
    v = f = t = 0
    with open(path, errors="replace") as fh:
        for line in fh:
            if line.startswith("v "):
                v += 1
            elif line.startswith("f "):
                f += 1
                t += max(len(line.split()) - 3, 0)
    return MeshStats(v, f, t)


def check_mesh(original: MeshStats, upgraded: MeshStats, path: str = "",
               limits: EngineLimits = ENGINE_LIMITS) -> list[Issue]:
    if original.triangles and upgraded.triangles > original.triangles * limits.max_triangle_growth:
        return [Issue(path, "triangle_budget",
                      f"{original.triangles} -> {upgraded.triangles} tris "
                      f"(limit x{limits.max_triangle_growth})")]
    return []
