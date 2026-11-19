"""What every backend has to provide. One subclass per training library."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from kiln import KilnError
from kiln.data.detect import Layout
from kiln.train.config import TrainConfig


@dataclass
class ExportedModel:
    """What a backend's export() hands back; kiln writes labels.txt and manifest.json from it."""

    onnx: Path            # the model.onnx it wrote
    arch: str             # output layout the edge reads: "rtdetr" (K, 6 rows) or "yolo" (raw head + NMS)
    labels: list[str]     # class names, in class-id order
    input_size: int       # square input side, in pixels
    source: str           # one line on what the model is and what its output looks like


class Backend:
    name: str = ""                 # what you write as `backend = "..."` in the config
    about: str = ""                # one line for `kiln train backends`
    licence: str = ""              # licence of the weights it produces, for manifest.json
    models: tuple[str, ...] = ()   # accepted `model = "..."` values
    formats: tuple[str, ...] = ()  # dataset formats it reads: "yolo", "coco"

    def check_model(self, model: str) -> None:
        if model not in self.models:
            raise KilnError(f"Backend '{self.name}' has no model '{model}'. Choose from: {', '.join(self.models)}")

    def train(self, cfg: TrainConfig, layout: Layout, out_dir: Path) -> Path:
        """Train, writing everything into out_dir. Return the best weights file."""
        raise NotImplementedError

    def best_weights(self, out_dir: Path) -> Path:
        """Where train() leaves the best weights inside out_dir."""
        raise NotImplementedError

    def export(self, cfg: TrainConfig, weights: Path, out_dir: Path) -> ExportedModel:
        """Write out_dir/model.onnx in a layout the edge can run."""
        raise NotImplementedError

    def metrics(self, run_dir: Path) -> dict:
        """Best val score and final test scores of a run, for manifest.json. Empty if unknown."""
        return {}
