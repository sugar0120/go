"""Asset inventory: one row per file, used to prioritise and batch the work."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image

from .config import (
    CATEGORY_KEYWORDS,
    CATEGORY_POLICY,
    MODEL_EXTS,
    READABLE_IMAGE_EXTS,
    ROLE_SUFFIXES,
    TEXTURE_EXTS,
)


@dataclass
class AssetRecord:
    path: str
    kind: str  # texture | model | other
    category: str
    role: str  # diffuse | normal | specular | envmap | "" for non-textures
    priority: int
    target_scale: int
    size_bytes: int
    width: int | None = None
    height: int | None = None
    mode: str | None = None
    has_alpha: bool | None = None
    note: str = ""


def guess_category(rel_path: str) -> str:
    p = "/" + rel_path.lower().replace("\\", "/")
    for category, keywords in CATEGORY_KEYWORDS:
        if any(k in p for k in keywords):
            return category
    return "other"


def guess_role(rel_path: str) -> str:
    stem = Path(rel_path).stem.lower()
    for role, suffixes in ROLE_SUFFIXES:
        if stem.endswith(suffixes):
            return role
    return "diffuse"


def _kind(ext: str) -> str:
    if ext in TEXTURE_EXTS:
        return "texture"
    if ext in MODEL_EXTS:
        return "model"
    return "other"


def scan(root: str | Path) -> list[AssetRecord]:
    """Walk ``root`` and describe every file, sorted by priority then path."""
    root = Path(root)
    records: list[AssetRecord] = []
    for f in sorted(root.rglob("*")):
        if not f.is_file() or f.name.startswith("."):
            continue
        rel = f.relative_to(root).as_posix()
        ext = f.suffix.lower()
        kind = _kind(ext)
        category = guess_category(rel)
        policy = CATEGORY_POLICY[category]
        rec = AssetRecord(
            path=rel,
            kind=kind,
            category=category,
            role=guess_role(rel) if kind == "texture" else "",
            priority=policy.priority,
            target_scale=policy.scale,
            size_bytes=f.stat().st_size,
        )
        if ext in READABLE_IMAGE_EXTS:
            try:
                with Image.open(f) as im:
                    rec.width, rec.height = im.size
                    rec.mode = im.mode
                    rec.has_alpha = im.mode in ("RGBA", "LA", "PA") or "transparency" in im.info
            except Exception as e:  # corrupt or unsupported variant
                rec.note = f"unreadable: {e}"
        elif ext == ".dtx":
            rec.note = "convert DTX to DDS/PNG first"
        records.append(rec)
    records.sort(key=lambda r: (r.priority, r.path))
    return records


def write_csv(records: list[AssetRecord], path: str | Path) -> None:
    fields = list(AssetRecord.__dataclass_fields__)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in records:
            w.writerow(asdict(r))
