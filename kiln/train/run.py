"""The training steps, the same for every backend."""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

from kiln import KilnError
from kiln.train.backends import get_backend
from kiln.train.config import TrainConfig
from kiln.train.dataset import find_dataset, resolve_classes

CONFIG_COPY = "kiln-train.toml"
EXPORT_DIR = "export"


def train(cfg: TrainConfig) -> Path:
    """config → dataset → train. Returns the best weights file."""
    backend = get_backend(cfg.backend)
    backend.check_model(cfg.model)

    layout = find_dataset(cfg, backend.formats)
    cfg.classes = resolve_classes(cfg, layout)

    out_dir = cfg.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(cfg.path, out_dir / CONFIG_COPY)  # so every run records how it was made

    return backend.train(cfg, layout, out_dir)


def export(cfg: TrainConfig, weights: Path | None = None) -> Path:
    """Write the edge model folder <run>/export: model.onnx + labels.txt + manifest.json.

    The backend writes model.onnx; the rest is the same for every backend. The folder
    is rebuilt from scratch each time, so it only ever holds the files for one model.
    """
    backend = get_backend(cfg.backend)
    weights = (weights or backend.best_weights(cfg.output_dir)).resolve()
    if not weights.is_file():
        raise KilnError(f"No weights at {weights}. Train first, or pass --weights.")

    out_dir = cfg.output_dir / EXPORT_DIR
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    model = backend.export(cfg, weights, out_dir)

    digest = hashlib.sha256(model.onnx.read_bytes()).hexdigest()
    (out_dir / "labels.txt").write_text("\n".join(model.labels) + "\n")
    manifest = {
        "name": f"{cfg.name}-{cfg.backend}-{cfg.model}",
        "arch": model.arch,
        "num_classes": len(model.labels),
        "input": {"shape": [1, 3, model.input_size, model.input_size]},
        "labels": model.labels,
        "sha256": digest,
        "source": model.source,
        "licence": backend.licence,
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "training": {"dataset": cfg.dataset.name, **backend.metrics(weights.parent)},
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return out_dir
