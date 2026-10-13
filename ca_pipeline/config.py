"""Engine limits and per-category upscale policy.

The numbers here are conservative starting points. Verify them in the actual
client during the pilot (load an oversized texture/model and watch for
crashes, missing textures or frame drops) and adjust.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class EngineLimits:
    max_texture_size: int = 2048
    require_power_of_two: bool = True
    # How much a reworked model's triangle count may grow over the original.
    max_triangle_growth: float = 2.0
    # Total texture memory budget for one loaded map + loadouts, in MiB.
    vram_budget_mib: int = 512


ENGINE_LIMITS = EngineLimits()


@dataclass(frozen=True)
class CategoryPolicy:
    scale: int
    priority: int  # 1 = do first


# "70% plan" (PLAN.md): only first-person weapons get 4x; everything else 2x,
# which is most of the visible gain for a fraction of the time and VRAM.
# Lightmaps and skies are left as they are (scale 1).
CATEGORY_POLICY: dict[str, CategoryPolicy] = {
    "weapon": CategoryPolicy(scale=4, priority=1),
    "character": CategoryPolicy(scale=2, priority=2),
    "ui": CategoryPolicy(scale=2, priority=3),
    "prop": CategoryPolicy(scale=2, priority=4),
    "world": CategoryPolicy(scale=2, priority=5),
    "fx": CategoryPolicy(scale=2, priority=6),
    "sky": CategoryPolicy(scale=1, priority=7),
    "lightmap": CategoryPolicy(scale=1, priority=8),
    "other": CategoryPolicy(scale=2, priority=9),
}

# Path keywords used to guess a category. First match wins, so order matters.
CATEGORY_KEYWORDS: list[tuple[str, tuple[str, ...]]] = [
    ("lightmap", ("lightmap", "lm_")),
    ("sky", ("sky",)),
    ("ui", ("interface", "hud", "/ui/", "font", "icon", "menu")),
    ("weapon", ("weapon", "wpn", "/gun", "knife", "grenade", "pv_", "hands")),
    ("character", ("char", "body", "head", "skin", "player")),
    ("fx", ("fx", "sfx", "particle", "muzzle", "smoke", "spark")),
    ("prop", ("prop",)),
    ("world", ("world", "/map", "terrain", "wall", "floor", "tex")),
]

TEXTURE_EXTS = {".dtx", ".dds", ".png", ".tga", ".bmp", ".jpg", ".jpeg"}
MODEL_EXTS = {".ltb", ".obj", ".fbx", ".ase"}
# Formats Pillow can open directly; .dtx must be converted first (see README).
READABLE_IMAGE_EXTS = {".dds", ".png", ".tga", ".bmp", ".jpg", ".jpeg"}

ROLE_SUFFIXES: list[tuple[str, tuple[str, ...]]] = [
    ("normal", ("_n", "_nm", "_nrm", "_normal", "_bump")),
    ("specular", ("_s", "_spec", "_specular")),
    ("envmap", ("_env", "_cube", "_reflect")),
]
