"""Texture looks for first-person weapon skins. The engine bakes a weapon's lighting into its diffuse texture, so
"glossier" and "more defined" have to be painted into the pixels. Everything works on luminance so hues stay put.

gloss(): hot specular highlights on the bright metal, a soft sheen bloom around them, local contrast for form,
         edge catchlights where a surface turns, a slight cool bias in the highlights (steel), deep blacks kept.
All strengths are explicit arguments so a look can be tuned from side-by-side crops.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter


def _blur(a: np.ndarray, radius: float) -> np.ndarray:
    im = Image.fromarray((np.clip(a, 0, 1) * 255 + 0.5).astype(np.uint8), "L").filter(ImageFilter.GaussianBlur(radius))
    return np.asarray(im).astype(np.float32) / 255.0


def gloss(img: Image.Image, scale: int = 2, lift: float = 0.88, spec: float = 0.75, sheen: float = 0.38,
          local: float = 0.9, edge: float = 1.3, cool: float = 0.012) -> Image.Image:
    """``img`` is the original-size RGB texture; returns it upscaled by ``scale`` with the gloss look applied."""
    big = img.convert("RGB").resize((img.width * scale, img.height * scale), Image.LANCZOS)
    k = scale / 2.0
    rgb = np.asarray(big).astype(np.float32) / 255.0
    y = rgb.mean(-1)                                       # luminance proxy; colours are scaled by the same ratio
    y0 = np.maximum(y, 0.035)                              # floor: do not blow up colour noise in near-black pixels
    # 1. open the shadows a little so the matte black parts show their shape
    y1 = np.power(y, lift)
    # 2. local contrast (form) and edge catchlights (defined edges)
    base = _blur(y1, 14 * k)
    y2 = y1 + local * 0.55 * (y1 - base)
    fine = _blur(y1, 2.2 * k)
    edges = np.clip(y1 - fine, 0, None)                    # only the bright side of an edge
    y2 = y2 + edge * 1.6 * edges
    # 3. gloss: hot highlights on the bright metal + soft sheen bloom around them
    hot = np.clip((y2 - 0.30) / 0.50, 0, 1) ** 2.0
    y3 = y2 + spec * hot * (1.0 - y2) * 0.9
    y3 = y3 + sheen * _blur(hot * (1.0 - y2), 7 * k)
    y3 = np.clip(y3, 0, 1)
    ratio = np.clip(y3 / y0, 0.0, 3.0)
    out = rgb * ratio[..., None]                           # keep each pixel's hue, change its brightness
    # 4. cool tint in the highlights only
    tint = (hot * cool)[..., None] * np.array([-0.4, 0.0, 1.0], np.float32)
    out = np.clip(out + tint, 0, 1)
    return Image.fromarray((out * 255 + 0.5).astype(np.uint8))
