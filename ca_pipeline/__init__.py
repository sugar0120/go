"""Asset upgrade pipeline for a private Combat Arms (2017 BR) rebuild.

The engine stays LithTech Jupiter EX, so every output must respect its limits:
power-of-two textures, a maximum texture size, DXT-compressible data and
polygon counts close to the originals.
"""

from .config import ENGINE_LIMITS, EngineLimits

__all__ = ["ENGINE_LIMITS", "EngineLimits"]
