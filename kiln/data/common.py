"""Names and helpers shared by load / info / export."""

from __future__ import annotations

import os
import re
import shutil
from pathlib import Path

LABEL_FIELD = "ground_truth"

SPLITS = ("train", "val", "test")
# Folder names that mean a split; "valid" (Roboflow) and "validation" are stored as "val".
SPLIT_ALIASES = {"train": "train", "val": "val", "valid": "val", "validation": "val", "test": "test"}

SCENE_PREFIX = "scene:"
FOLDER_PREFIX = "folder:"
GROUPS_FILE = "groups.json"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def is_image(p: Path) -> bool:
    return p.suffix.lower() in IMAGE_EXTS and p.is_file()


def split_of(rel: Path) -> str | None:
    """The split named by the deepest split-like folder in a path, e.g. data/val/images/x.jpg -> 'val'."""
    for part in reversed(rel.parts[:-1]):
        if part.lower() in SPLIT_ALIASES:
            return SPLIT_ALIASES[part.lower()]
    return None


def split_tags(tags: list[str]) -> list[str]:
    return [t for t in tags if t in SPLITS]


def iter_images(root: Path):
    """All image files under root, skipping hidden folders, in a stable order."""
    for p in sorted(root.rglob("*")):
        if any(part.startswith(".") for part in p.relative_to(root).parts):
            continue
        if is_image(p):
            yield p


def natural_key(p: Path | str):
    """Sort 'img2' before 'img10'."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", str(p))]


def clean_name(p: Path) -> str:
    """File name safe for any tooling: 'Fire (12).PNG' -> 'Fire_12.png'."""
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", p.stem).strip("_") or "image"
    return stem + p.suffix.lower()


def place_file(src: Path, dst: Path, link: bool) -> None:
    """Copy src to dst, or hard-link it when asked and possible (same filesystem)."""
    if link:
        try:
            os.link(src, dst)
            return
        except OSError:
            pass
    shutil.copy2(src, dst)
