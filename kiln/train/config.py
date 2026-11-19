"""Training config: one TOML file per task, e.g. ``configs/fire.toml``.

    backend  = "rfdetr"                        # which training library (`kiln train backends` lists them)
    model    = "nano"                          # model name, as that backend calls it
    dataset  = "data/FIRE/fire-labelled-coco"  # YOLO or COCO folder with train / val / test
    classes  = ["smoke", "fire"]               # optional: else kiln.toml, else the dataset's own list
    epochs   = 50
    batch    = 2                               # or "auto" if the backend supports it
    imgsz    = 384                             # optional: else the model's default
    lr       = 1e-4
    patience = 10                              # stop after this many epochs without improvement
    workers  = 4
    output   = "runs/fire-rfdetr-nano"         # optional: else runs/<config name>-<backend>-<model>
    resume   = "runs/.../checkpoint.pth"       # optional: continue an earlier run
    mlflow   = false
    project  = "fire-detection"                # optional: experiment name for loggers

    [extra]                                    # passed as-is to the backend's train call
    grad_accum_steps = 8

Only backend, model and dataset are required. Relative paths are read from the current folder.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from kiln import KilnError
from kiln.config import parse_classes

# key -> allowed types
_FIELDS: dict[str, tuple[type, ...]] = {
    "backend": (str,),
    "model": (str,),
    "dataset": (str,),
    "classes": (list, str),
    "epochs": (int,),
    "batch": (int, str),
    "imgsz": (int,),
    "lr": (int, float),
    "patience": (int,),
    "workers": (int,),
    "output": (str,),
    "resume": (str,),
    "mlflow": (bool,),
    "project": (str,),
    "extra": (dict,),
}
_REQUIRED = ("backend", "model", "dataset")


@dataclass
class TrainConfig:
    backend: str
    model: str
    dataset: Path
    path: Path  # the config file itself
    classes: list[str] | None = None
    epochs: int = 50
    batch: int | str = 8
    imgsz: int | None = None
    lr: float = 1e-4
    patience: int = 10
    workers: int = 4
    output: Path | None = None
    resume: Path | None = None
    mlflow: bool = False
    project: str | None = None
    extra: dict = field(default_factory=dict)

    @property
    def name(self) -> str:
        """Short name for the task, from the file name: configs/fire.toml -> 'fire'."""
        return self.path.stem

    @property
    def output_dir(self) -> Path:
        return (self.output or Path("runs") / f"{self.name}-{self.backend}-{self.model}").resolve()


def load_train_config(path: Path) -> TrainConfig:
    path = Path(path)
    if not path.is_file():
        raise KilnError(f"No config file at {path}")
    try:
        raw = tomllib.loads(path.read_text())
    except tomllib.TOMLDecodeError as e:
        raise KilnError(f"{path}: {e}") from e

    unknown = set(raw) - set(_FIELDS)
    if unknown:
        raise KilnError(f"{path}: unknown keys {sorted(unknown)} (allowed: {sorted(_FIELDS)})")
    missing = [k for k in _REQUIRED if k not in raw]
    if missing:
        raise KilnError(f"{path}: missing required keys {missing}")
    for key, value in raw.items():
        allowed = _FIELDS[key]
        # TOML true/false are Python bools, which also count as ints; only accept them for bool keys.
        if not isinstance(value, allowed) or (isinstance(value, bool) and bool not in allowed):
            names = " or ".join(t.__name__ for t in allowed)
            raise KilnError(f"{path}: '{key}' must be {names}, got {value!r}")

    cfg = TrainConfig(backend=raw["backend"], model=raw["model"], dataset=Path(raw["dataset"]).resolve(), path=path)
    for key in ("epochs", "batch", "imgsz", "lr", "patience", "workers", "mlflow", "project", "extra"):
        if key in raw:
            setattr(cfg, key, raw[key])
    if "classes" in raw:
        cfg.classes = parse_classes(raw["classes"], where=f"{path}: classes")
    if "output" in raw:
        cfg.output = Path(raw["output"])
    if "resume" in raw:
        cfg.resume = Path(raw["resume"]).resolve()
    return cfg
