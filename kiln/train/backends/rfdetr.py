"""RF-DETR (Roboflow). Reads COCO or YOLO folders directly. `rfdetr` and `torch` are imported lazily: they are slow."""

from __future__ import annotations

import copy
import csv
from pathlib import Path

from kiln import KilnError
from kiln.data.detect import Layout
from kiln.train.backends.base import Backend, ExportedModel
from kiln.train.config import TrainConfig
from kiln.train.dataset import write_data_yaml

_CLASSES = {"nano": "RFDETRNano", "small": "RFDETRSmall", "medium": "RFDETRMedium", "large": "RFDETRLarge"}


class RFDETRBackend(Backend):
    name = "rfdetr"
    about = "RF-DETR by Roboflow (transformer detector)"
    licence = "Apache-2.0 (RF-DETR)"
    models = tuple(_CLASSES)
    formats = ("coco", "yolo")

    def train(self, cfg: TrainConfig, layout: Layout, out_dir: Path) -> Path:
        import rfdetr

        if layout.format == "yolo":
            write_data_yaml(layout, cfg.classes)

        kwargs = dict(
            dataset_dir=str(layout.root),
            output_dir=str(out_dir),
            epochs=cfg.epochs,
            batch_size=cfg.batch,
            lr=cfg.lr,
            num_workers=cfg.workers,
            early_stopping=True,
            early_stopping_patience=cfg.patience,
            class_names=cfg.classes,
            run_test=True,
            progress_bar="tqdm",
            tensorboard=True,
            mlflow=cfg.mlflow,
            project=cfg.project or cfg.name,
            run=out_dir.name,
        )
        if cfg.imgsz:
            kwargs["resolution"] = cfg.imgsz
        if cfg.resume:
            kwargs["resume"] = str(cfg.resume)
        kwargs |= cfg.extra

        model = getattr(rfdetr, _CLASSES[cfg.model])()
        model.train(**kwargs)
        return self.best_weights(out_dir)

    def best_weights(self, out_dir: Path) -> Path:
        return out_dir / "checkpoint_best_total.pth"

    def export(self, cfg: TrainConfig, weights: Path, out_dir: Path) -> ExportedModel:
        """Export with an in-graph wrapper so the edge's `rtdetr` path runs it unchanged."""
        import torch
        from rfdetr.detr import RFDETR
        from rfdetr.models.backbone.dinov2 import DinoV2

        model = RFDETR.from_checkpoint(str(weights))
        labels = list(model.class_names)
        size = int(model.model_config.resolution)
        num_select = int(getattr(model.model_config, "num_select", None) or 300)

        net = copy.deepcopy(model.model.model).eval().cpu()
        for m in net.modules():  # freeze DINOv2 position embeddings to the export size (as rfdetr's exporter does)
            if isinstance(m, DinoV2):
                m.shape = (size, size)
                m.export()
        net.export()  # switch LWDETR to forward_export -> (boxes, logits)
        wrapper = _edge_wrapper(net, size, list(model.means), list(model.stds), num_select)

        onnx_path = out_dir / "model.onnx"
        dummy = torch.rand(1, 3, size, size)
        with torch.no_grad():
            n_slots = wrapper.net(dummy)[1].shape[-1]
            if n_slots != len(labels) + 1:
                raise KilnError(f"model has {n_slots} class slots, expected {len(labels)} classes + background")
            torch.onnx.export(wrapper, (dummy,), str(onnx_path), input_names=["images"], output_names=["output"],
                              opset_version=17, dynamo=False)

        return ExportedModel(
            onnx=onnx_path,
            arch="rtdetr",
            labels=labels,
            input_size=size,
            source=f"RF-DETR {cfg.model} fine-tune ({weights.name}); output (1, {num_select}, 6) "
                   "x1,y1,x2,y2 px, score, class; ImageNet normalisation in-graph",
        )

    def metrics(self, run_dir: Path) -> dict:
        """Best val epoch and final test metrics from RF-DETR's metrics.csv."""
        path = run_dir / "metrics.csv"
        if not path.is_file():
            return {}
        with path.open() as f:
            rows = list(csv.DictReader(f))
        val = [r for r in rows if r.get("val/mAP_50_95")]
        test = [r for r in rows if r.get("test/mAP_50_95")]
        out: dict = {}
        if val:
            best = max(val, key=lambda r: float(r["val/mAP_50_95"]))
            out["best_epoch"] = int(best["epoch"])
            out["val"] = {"mAP_50_95": round(float(best["val/mAP_50_95"]), 4)}
        if test:
            keys = ("mAP_50", "mAP_50_95", "precision", "recall", "F1")
            out["test"] = {k: round(float(test[-1][f"test/{k}"]), 4) for k in keys if test[-1].get(f"test/{k}")}
        return out


def _edge_wrapper(net, size: int, means: list[float], stds: list[float], num_select: int):
    """Wrap RF-DETR's export graph so it speaks the edge's ``rtdetr`` layout.

    In:  (1, 3, S, S) RGB in 0..1 (the edge's ``preprocess.to_input``); ImageNet
         normalisation happens inside the graph.
    Out: (1, K, 6) = x1, y1, x2, y2 (input pixels), score, class, which is what
         ``smartsight_edge.inference.postprocess.decode_output`` reads.

    Selection mirrors ``RFDETR.predict()``: per-class sigmoids with the trailing
    background slot dropped, then top-K over every (query, class) pair, so a box
    may appear once per class it scores on.
    """
    import torch
    from torch import nn

    class EdgeRTDETR(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = net
            self.register_buffer("mean", torch.tensor(means).view(1, 3, 1, 1))
            self.register_buffer("std", torch.tensor(stds).view(1, 3, 1, 1))

        def forward(self, x):
            boxes, logits = self.net((x - self.mean) / self.std)[:2]
            scores = logits[..., :-1].sigmoid()  # (B, Q, C); last slot = background
            num_classes = scores.shape[-1]
            top, idx = scores.flatten(1).topk(num_select, dim=1)
            query = torch.div(idx, num_classes, rounding_mode="floor")
            cls = idx - query * num_classes
            b = boxes.gather(1, query.unsqueeze(-1).expand(-1, -1, 4))
            cx, cy, w, h = b.unbind(-1)
            xyxy = torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1) * size
            return torch.cat([xyxy, top.unsqueeze(-1), cls.unsqueeze(-1).to(xyxy.dtype)], dim=-1)

    return EdgeRTDETR().eval()
