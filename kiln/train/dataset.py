"""Dataset steps shared by every backend: find the format, settle the class names, write data.yaml."""

from __future__ import annotations

from pathlib import Path

import yaml

from kiln import KilnError
from kiln.config import load_config
from kiln.data.common import SPLIT_ALIASES, SPLITS, split_of
from kiln.data.detect import Layout, detect
from kiln.train.config import TrainConfig


def find_dataset(cfg: TrainConfig, formats: tuple[str, ...]) -> Layout:
    """Detect the dataset's format and check the backend can read it."""
    layout = detect(cfg.dataset)
    if layout.format == "images":
        raise KilnError(f"{cfg.dataset} has images but no labels; label it first (kiln label …)")
    if layout.format not in formats:
        raise KilnError(
            f"{cfg.dataset} is {layout.format}, but backend '{cfg.backend}' reads {' or '.join(formats)}. "
            f"Convert it with:  kiln data export <name> --format {formats[0]} --out <folder>"
        )
    return layout


def resolve_classes(cfg: TrainConfig, layout: Layout) -> list[str]:
    """Class names, in order of preference: the training config, kiln.toml, the dataset itself.

    If the dataset names its own classes, they must match: class ids in the labels follow that order.
    """
    if cfg.classes:
        classes, source = cfg.classes, str(cfg.path)
    elif (toml := load_config(cfg.dataset)).classes:
        classes, source = toml.classes, str(toml.path)
    elif layout.classes:
        return layout.classes
    else:
        raise KilnError(f"No class names for {cfg.dataset}: add `classes = [...]` to {cfg.path}")

    if layout.classes and layout.classes != classes:
        raise KilnError(
            f"Class names in {source} {classes} don't match the dataset's own "
            f"{layout.classes} (from {layout.classes_from}). Fix one of them."
        )
    return classes


def write_data_yaml(layout: Layout, classes: list[str]) -> Path:
    """Write <root>/data.yaml for a YOLO dataset, with split paths relative to the root.

    Rewritten every time, because datasets often ship one pointing at someone else's machine.
    """
    splits: dict[str, Path] = {}
    for images, _labels in layout.yolo_pairs:
        rel = images.relative_to(layout.root)
        split = split_of(rel / "x")
        if split:  # <split>/images
            splits.setdefault(split, rel)
            continue
        for sub in sorted(p for p in images.iterdir() if p.is_dir()):  # images/<split>
            if sub.name.lower() in SPLIT_ALIASES:
                splits.setdefault(SPLIT_ALIASES[sub.name.lower()], sub.relative_to(layout.root))

    missing = [s for s in ("train", "val") if s not in splits]
    if missing:
        raise KilnError(f"{layout.root}: no {' or '.join(missing)} split found (expected train / val / test folders)")

    data = {s: str(splits[s]) for s in SPLITS if s in splits}
    data |= {"nc": len(classes), "names": classes}
    path = layout.root / "data.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    return path
