"""Synthetic degradation that mimics how Combat Arms textures were authored.

Training pairs are (low-res input, high-res target). For a model to learn to
fix *our* textures, the low-res side must look like them: downscaled, then
DXT-compressed (4x4 block colour quantisation, 565 endpoints). This module
reproduces that in numpy so pairs can be built anywhere, molab included.
"""

from __future__ import annotations

import numpy as np
from PIL import Image


def _blocks(arr: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    h, w, c = arr.shape
    ph, pw = (-h) % 4, (-w) % 4
    a = np.pad(arr, ((0, ph), (0, pw), (0, 0)), mode="edge")
    H, W = a.shape[:2]
    b = a.reshape(H // 4, 4, W // 4, 4, c).transpose(0, 2, 1, 3, 4).reshape(-1, 16, c)
    return b, (h, w, H, W)


def _unblocks(b: np.ndarray, shape: tuple[int, int, int, int]) -> np.ndarray:
    h, w, H, W = shape
    c = b.shape[-1]
    a = b.reshape(H // 4, W // 4, 4, 4, c).transpose(0, 2, 1, 3, 4).reshape(H, W, c)
    return a[:h, :w]


def _q565(c: np.ndarray) -> np.ndarray:
    levels = np.array([31.0, 63.0, 31.0], np.float32)
    return np.round(c * levels) / levels


def bc1_roundtrip(rgb: np.ndarray) -> np.ndarray:
    """Approximate DXT1/BC1 encode+decode of an HxWx3 float image."""
    b, shape = _blocks(rgb.astype(np.float32))
    lo, hi = b.min(axis=1), b.max(axis=1)
    inset = (hi - lo) / 16.0  # standard bounding-box inset, like simple real encoders
    c0, c1 = _q565(hi - inset), _q565(lo + inset)
    palette = np.stack([c0, c1, (2 * c0 + c1) / 3, (c0 + 2 * c1) / 3], axis=1)  # N,4,3
    d = ((b[:, :, None, :] - palette[:, None, :, :]) ** 2).sum(-1)  # N,16,4
    idx = d.argmin(-1)
    out = np.take_along_axis(palette, idx[..., None].repeat(3, -1), axis=1)
    return _unblocks(out, shape)


def bc3_alpha_roundtrip(alpha: np.ndarray) -> np.ndarray:
    """Approximate DXT5/BC3 alpha block encode+decode of an HxWx1 float image."""
    b, shape = _blocks(alpha.astype(np.float32))
    a0 = np.round(b.max(axis=1) * 255) / 255
    a1 = np.round(b.min(axis=1) * 255) / 255
    t = np.arange(8, dtype=np.float32) / 7.0
    palette = a1[:, None, :] + (a0 - a1)[:, None, :] * t[None, :, None]  # N,8,1
    idx = np.abs(b[:, :, None, 0] - palette[:, None, :, 0]).argmin(-1)
    out = np.take_along_axis(palette[..., 0], idx, axis=1)[..., None]
    return _unblocks(out, shape)


def degrade(hr: Image.Image, scale: int = 4, dxt: bool = True,
            rng: np.random.Generator | None = None) -> Image.Image:
    """Make the low-res training input for a high-res texture crop."""
    rng = rng or np.random.default_rng()
    w, h = hr.size
    # Mip generation in old tools was box/bilinear; vary it so the model generalises.
    kernel = [Image.BOX, Image.BILINEAR, Image.BICUBIC][rng.integers(3)]
    lr = hr.resize((w // scale, h // scale), kernel)
    if not dxt:
        return lr
    arr = np.asarray(lr, np.float32) / 255.0
    rgb = bc1_roundtrip(arr[..., :3])
    if arr.shape[2] == 4:
        rgb = np.concatenate([rgb, bc3_alpha_roundtrip(arr[..., 3:])], axis=2)
    a = (np.clip(rgb, 0, 1) * 255 + 0.5).astype(np.uint8)
    return Image.fromarray(a, "RGBA" if a.shape[2] == 4 else "RGB")


def random_crops(img: Image.Image, size: int, count: int,
                 rng: np.random.Generator | None = None) -> list[Image.Image]:
    rng = rng or np.random.default_rng()
    w, h = img.size
    if w < size or h < size:
        return []
    return [img.crop((x, y, x + size, y + size))
            for x, y in zip(rng.integers(0, w - size + 1, count), rng.integers(0, h - size + 1, count))]
