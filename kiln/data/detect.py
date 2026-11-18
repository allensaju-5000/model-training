"""Work out what kind of dataset a folder holds.

Supported layouts (split folders may be named train / val / valid / test, at any depth):

    coco    one COCO json per split (Roboflow ``<split>/_annotations.coco.json``),
            or a single one for an unsplit set (FiftyOne ``labels.json`` + ``data/``)
    yolo    ``<split>/images`` + ``<split>/labels``, ``images/<split>`` + ``labels/<split>``,
            or plain ``images/`` + ``labels/``; class names from data.yaml / dataset.yaml
    images  anything else holding images (bare folder, class folders): loaded unlabelled
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from kiln import KilnError
from kiln.data.common import GROUPS_FILE, SPLIT_ALIASES, iter_images, split_of

MAX_DEPTH = 4


@dataclass
class Layout:
    format: str  # "coco" | "yolo" | "images"
    root: Path
    coco_files: list[Path] = field(default_factory=list)
    yolo_pairs: list[tuple[Path, Path]] = field(default_factory=list)  # (images dir, labels dir)
    classes: list[str] | None = None  # names declared by the dataset itself
    classes_from: str | None = None


def _shallow(root: Path, pattern: str):
    for p in sorted(root.rglob(pattern)):
        rel = p.relative_to(root)
        if len(rel.parts) <= MAX_DEPTH and not any(s.startswith(".") for s in rel.parts):
            yield p


def _looks_like_coco(p: Path) -> bool:
    if p.name == GROUPS_FILE:
        return False
    with p.open("rb") as f:
        head = f.read(1 << 20)
    return b'"images"' in head and (b'"annotations"' in head or b'"categories"' in head)


def _find_coco(root: Path) -> list[Path]:
    files = [p for p in _shallow(root, "*.json") if _looks_like_coco(p)]
    by_split: dict[str | None, list[Path]] = {}
    for p in files:
        split = split_of(p.relative_to(root)) or split_from_stem(p)
        by_split.setdefault(split, []).append(p)
    clashes = {s: ps for s, ps in by_split.items() if len(ps) > 1}
    if clashes:
        lines = [f"  {s or '(no split)'}: " + ", ".join(str(p.relative_to(root)) for p in ps) for s, ps in clashes.items()]
        raise KilnError(
            "Several COCO annotation files map to the same split; point `load` at a folder "
            "with one per split:\n" + "\n".join(lines)
        )
    if None in by_split and len(by_split) > 1:
        raise KilnError("Found COCO files both inside and outside split folders: " + ", ".join(map(str, files)))
    return files


def split_from_stem(p: Path) -> str | None:
    return SPLIT_ALIASES.get(p.stem.lower())


def _find_yolo(root: Path) -> list[tuple[Path, Path]]:
    pairs = []
    for labels in _shallow(root, "labels"):
        images = labels.parent / "images"
        if labels.is_dir() and images.is_dir() and any(labels.rglob("*.txt")):
            pairs.append((images, labels))
    return pairs


def _yaml_classes(root: Path) -> tuple[list[str] | None, str | None]:
    found: dict[tuple[str, ...], Path] = {}
    for p in [*_shallow(root, "*.yaml"), *_shallow(root, "*.yml")]:
        try:
            data = yaml.safe_load(p.read_text())
        except yaml.YAMLError:
            continue
        names = data.get("names") if isinstance(data, dict) else None
        if isinstance(names, dict):
            names = [names[k] for k in sorted(names, key=int)]
        if isinstance(names, list) and names:
            found.setdefault(tuple(str(n) for n in names), p)
    if len(found) > 1:
        detail = "\n".join(f"  {p.relative_to(root)}: {list(n)}" for n, p in found.items())
        raise KilnError(f"YAML files disagree on class names; pass --classes:\n{detail}")
    if found:
        names, p = next(iter(found.items()))
        return list(names), str(p.relative_to(root))
    return None, None


def coco_categories(data: dict) -> list[dict]:
    """Categories in id order, minus Roboflow's unused placeholder parent category."""
    cats = sorted(data.get("categories", []), key=lambda c: int(c["id"]))
    used = {int(a["category_id"]) for a in data.get("annotations", [])}
    parents = {c.get("supercategory") for c in cats}
    return [c for c in cats if not (c["name"] in parents and c.get("supercategory") != c["name"] and int(c["id"]) not in used)]


def _coco_classes(files: list[Path], root: Path) -> tuple[list[str] | None, str | None]:
    lists = {}
    for p in files:
        names = tuple(c["name"] for c in coco_categories(json.loads(p.read_text())))
        if names:
            lists.setdefault(names, p)
    if len(lists) > 1:
        detail = "\n".join(f"  {p.relative_to(root)}: {list(n)}" for n, p in lists.items())
        raise KilnError(f"COCO files disagree on categories; pass --classes:\n{detail}")
    if lists:
        names, p = next(iter(lists.items()))
        return list(names), str(p.relative_to(root))
    return None, None


def detect(root: Path) -> Layout:
    root = root.resolve()
    if not root.is_dir():
        raise KilnError(f"{root} is not a folder")

    coco = _find_coco(root)
    if coco:
        classes, src = _coco_classes(coco, root)
        return Layout("coco", root, coco_files=coco, classes=classes, classes_from=src)

    pairs = _find_yolo(root)
    if pairs:
        classes, src = _yaml_classes(root)
        return Layout("yolo", root, yolo_pairs=pairs, classes=classes, classes_from=src)

    if next(iter_images(root), None) is not None:
        return Layout("images", root)

    raise KilnError(f"No images found under {root}")
