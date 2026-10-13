"""Texture processing: upscaling plus the pre/post steps game textures need.

Images move through the pipeline as float32 numpy arrays, HxWxC in [0, 1].
"""

from __future__ import annotations

import math
from typing import Protocol

import numpy as np
from PIL import Image

from .config import ENGINE_LIMITS, EngineLimits


class Upscaler(Protocol):
    name: str
    scale: int

    def __call__(self, rgb: np.ndarray) -> np.ndarray:
        """Upscale an HxWx3 float image by ``self.scale``."""


class LanczosUpscaler:
    """Plain resampling. Baseline for comparisons and fallback without a GPU."""

    def __init__(self, scale: int = 2):
        self.scale = scale
        self.name = f"lanczos-{scale}x"

    def __call__(self, rgb: np.ndarray) -> np.ndarray:
        h, w = rgb.shape[:2]
        channels = [
            np.asarray(
                Image.fromarray(rgb[..., c].astype(np.float32), mode="F").resize(
                    (w * self.scale, h * self.scale), Image.LANCZOS
                )
            )
            for c in range(rgb.shape[2])
        ]
        return np.clip(np.stack(channels, axis=-1), 0.0, 1.0)


class SpandrelUpscaler:
    """Any model spandrel can load: ESRGAN, Real-ESRGAN, SwinIR, HAT, DAT, DRCT...

    Large images are processed in overlapping tiles so huge transformer models
    fit in VRAM. Requires the ``ai`` extra (torch + spandrel).
    """

    def __init__(self, model_path: str, tile: int = 512, overlap: int = 32,
                 half: bool = True, device: str | None = None):
        import torch
        from spandrel import ModelLoader

        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ModelLoader().load_from_file(model_path).eval().to(self.device)
        self.half = half and self.device == "cuda" and self.model.supports_half
        if self.half:
            self.model.half()
        self.scale = self.model.scale
        self.name = f"{self.model.architecture.name}:{model_path.rsplit('/', 1)[-1]}"
        self.tile = tile
        self.overlap = overlap

    def _run(self, patch: np.ndarray) -> np.ndarray:
        torch = self._torch
        t = torch.from_numpy(np.ascontiguousarray(patch.transpose(2, 0, 1)))[None]
        t = t.to(self.device, torch.float16 if self.half else torch.float32)
        with torch.inference_mode():
            out = self.model(t)
        return out[0].float().clamp(0, 1).cpu().numpy().transpose(1, 2, 0)

    def __call__(self, rgb: np.ndarray) -> np.ndarray:
        h, w = rgb.shape[:2]
        s, tile, ov = self.scale, self.tile, self.overlap
        if h <= tile and w <= tile:
            return self._run(rgb)
        out = np.zeros((h * s, w * s, 3), np.float32)
        weight = np.zeros((h * s, w * s, 1), np.float32)
        step = tile - 2 * ov
        for y0 in range(0, h, step):
            for x0 in range(0, w, step):
                ys, xs = max(y0 - ov, 0), max(x0 - ov, 0)
                ye, xe = min(y0 + step + ov, h), min(x0 + step + ov, w)
                up = self._run(rgb[ys:ye, xs:xe])
                out[ys * s:ye * s, xs * s:xe * s] += up
                weight[ys * s:ye * s, xs * s:xe * s] += 1.0
        return out / weight


def to_float(img: Image.Image) -> np.ndarray:
    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGBA" if "transparency" in img.info or img.mode in ("LA", "PA") else "RGB")
    return np.asarray(img, dtype=np.float32) / 255.0


def to_image(arr: np.ndarray) -> Image.Image:
    a = (np.clip(arr, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    return Image.fromarray(a, "RGBA" if a.shape[2] == 4 else "RGB")


def wrap_pad(arr: np.ndarray, pad: int) -> np.ndarray:
    """Pad with the opposite edges so tiling textures upscale without seams."""
    return np.pad(arr, ((pad, pad), (pad, pad), (0, 0)), mode="wrap")


def renormalize_normal_map(rgb: np.ndarray) -> np.ndarray:
    """Upscalers treat normal maps as colours; restore unit-length vectors."""
    v = rgb * 2.0 - 1.0
    v[..., 2] = np.clip(v[..., 2], 0.0, None)  # tangent-space normals face outward
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    v = np.where(n > 1e-6, v / np.maximum(n, 1e-6), np.array([0.0, 0.0, 1.0], np.float32))
    return (v + 1.0) * 0.5


def looks_like_normal_map(rgb: np.ndarray) -> bool:
    """Heuristic for normal maps that lack a name suffix (bluish, centred R/G)."""
    m = rgb.reshape(-1, 3).mean(axis=0)
    return bool(m[2] > 0.7 and abs(m[0] - 0.5) < 0.12 and abs(m[1] - 0.5) < 0.12)


def fit_to_engine(img: Image.Image, limits: EngineLimits = ENGINE_LIMITS) -> Image.Image:
    """Clamp to the engine's max size and snap each side to a power of two."""
    w, h = img.size
    longest = max(w, h)
    if longest > limits.max_texture_size:
        k = limits.max_texture_size / longest
        w, h = max(1, round(w * k)), max(1, round(h * k))
    if limits.require_power_of_two:
        w, h = (2 ** round(math.log2(v)) for v in (w, h))
        w, h = min(w, limits.max_texture_size), min(h, limits.max_texture_size)
    return img if (w, h) == img.size else img.resize((w, h), Image.LANCZOS)


def process_texture(
    img: Image.Image,
    upscaler: Upscaler,
    role: str = "diffuse",
    tiling: bool = False,
    pad: int = 16,
    alpha_cutout: bool = False,
    target_scale: int | None = None,
    limits: EngineLimits = ENGINE_LIMITS,
) -> Image.Image:
    """Upscale one texture with the right handling for its role.

    - alpha is split off and upscaled on its own, then recombined
    - tiling textures are wrap-padded so edges stay seamless
    - normal maps are renormalised afterwards
    - with ``target_scale`` below the model's scale (e.g. a 4x model on a 2x
      category), the result is downsampled to that size
    - result is fitted to the engine limits (power of two, max size)
    """
    arr = to_float(img)
    rgb, alpha = arr[..., :3], (arr[..., 3:] if arr.shape[2] == 4 else None)
    if role == "normal" or (role == "diffuse" and looks_like_normal_map(rgb)):
        role = "normal"

    def up(x: np.ndarray) -> np.ndarray:
        if not tiling:
            return upscaler(x)
        s = upscaler.scale
        y = upscaler(wrap_pad(x, pad))
        return y[pad * s:-pad * s, pad * s:-pad * s]

    out = up(rgb)
    if role == "normal":
        out = renormalize_normal_map(out)
    if alpha is not None:
        a = up(np.repeat(alpha, 3, axis=2)).mean(axis=2, keepdims=True)
        if alpha_cutout:  # fences, foliage: keep hard edges for alpha testing
            a = (a >= 0.5).astype(np.float32)
        out = np.concatenate([out, a], axis=2)
    result = to_image(out)
    if target_scale and target_scale != upscaler.scale:
        h, w = rgb.shape[:2]
        result = result.resize((w * target_scale, h * target_scale), Image.LANCZOS)
    return fit_to_engine(result, limits)
