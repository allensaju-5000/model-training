"""Optional per-dataset settings from ``kiln.toml``.

Looked up in the dataset folder first, then the current directory. Every key is
optional; CLI flags override the file.

    classes = ["smoke", "fire"]      # order = class ids
    ratios = [70, 15, 15]            # train / val / test, used by label-split
    guide = "guides/fire.md"         # labelling guide for label-split
    negative_folders = ["Safe_Fire"] # class folders whose images have nothing to box
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from kiln import KilnError

FILENAME = "kiln.toml"
_KEYS = {"classes", "ratios", "guide", "negative_folders"}


@dataclass
class Config:
    classes: list[str] | None = None
    ratios: tuple[int, int, int] = (70, 15, 15)
    guide: str | None = None
    negative_folders: list[str] = field(default_factory=list)
    path: Path | None = None  # where it was read from, if anywhere


def find_config(*dirs: Path) -> Path | None:
    for d in dirs:
        p = d / FILENAME
        if p.is_file():
            return p
    return None


def load_config(dataset_dir: Path | None = None) -> Config:
    dirs = [d for d in (dataset_dir, Path.cwd()) if d is not None]
    path = find_config(*dirs)
    if path is None:
        return Config()

    try:
        raw = tomllib.loads(path.read_text())
    except tomllib.TOMLDecodeError as e:
        raise KilnError(f"{path}: {e}") from e

    unknown = set(raw) - _KEYS
    if unknown:
        raise KilnError(f"{path}: unknown keys {sorted(unknown)} (allowed: {sorted(_KEYS)})")

    cfg = Config(path=path)
    if "classes" in raw:
        cfg.classes = parse_classes(raw["classes"], where=str(path))
    if "ratios" in raw:
        cfg.ratios = parse_ratios(raw["ratios"], where=str(path))
    if "guide" in raw:
        cfg.guide = str(raw["guide"])
    if "negative_folders" in raw:
        cfg.negative_folders = [str(x) for x in raw["negative_folders"]]
    return cfg


def parse_classes(value: object, where: str = "--classes") -> list[str]:
    if isinstance(value, str):
        value = [c.strip() for c in value.split(",")]
    if not isinstance(value, list) or not value or not all(isinstance(c, str) and c for c in value):
        raise KilnError(f"{where}: classes must be a non-empty list of names")
    if len(set(value)) != len(value):
        raise KilnError(f"{where}: duplicate class names in {value}")
    return list(value)


def parse_ratios(value: object, where: str = "--ratios") -> tuple[int, int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise KilnError(f"{where}: ratios must be three numbers (train, val, test)")
    nums = [int(v) for v in value]
    if any(n < 0 for n in nums) or sum(nums) != 100:
        raise KilnError(f"{where}: ratios must be non-negative and sum to 100, got {nums}")
    return nums[0], nums[1], nums[2]
