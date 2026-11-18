"""Fine-tune RF-DETR on D-FIRE (smoke + fire) and export the result to ONNX.

Edit the CONFIG block below, then:

    uv run python train.py           # train
    uv run python train.py export    # export OUTPUT_DIR's best checkpoint to ONNX

RF-DETR reads Ultralytics-style YOLO datasets directly, so D-FIRE needs no COCO
conversion. The script only writes a clean ``data.yaml`` into the dataset root,
because the one shipped with the dataset points at a Kaggle path.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# ---------------------------------------------------------------- CONFIG ----
MODEL_SIZE = "nano"              # "nano" | "small" | "medium" | "large"
DATASET_DIR = ROOT / "data" / "D-FIRE" / "data"   # holds train/ val/ test/ {images,labels}
OUTPUT_DIR = ROOT / "runs" / f"rfdetr-{MODEL_SIZE}"
CLASS_NAMES = ["smoke", "fire"]

EPOCHS = 1
BATCH_SIZE = 2                # int, or "auto" to fit the GPU
GRAD_ACCUM_STEPS = 8              # effective batch = BATCH_SIZE * GRAD_ACCUM_STEPS
LR = 1e-4
RESOLUTION = None                  # None = variant default (nano 384, small 512, medium 576)
NUM_WORKERS = 4
EARLY_STOPPING_PATIENCE = 10       # epochs without mAP improvement before stopping
RESUME_FROM = None                 # path to a checkpoint to resume, or None

MLFLOW = False                     # True logs to $MLFLOW_TRACKING_URI (e.g. ss-mlflow)

EXPORT_CHECKPOINT = OUTPUT_DIR / "checkpoint_best_total.pth"
EXPORT_DIR = OUTPUT_DIR / "export"
# ----------------------------------------------------------------------------

SPLITS = {"train": "train/images", "val": "val/images", "test": "test/images"}


def model_class(size: str):
    import rfdetr

    return {
        "nano": rfdetr.RFDETRNano,
        "small": rfdetr.RFDETRSmall,
        "medium": rfdetr.RFDETRMedium,
        "large": rfdetr.RFDETRLarge,
    }[size]


def prepare_dataset(dataset_dir: Path) -> None:
    """Check the YOLO split layout and write a data.yaml with paths relative to the root."""
    for rel in SPLITS.values():
        images = dataset_dir / rel
        labels = images.parent / "labels"
        if not images.is_dir() or not labels.is_dir():
            raise SystemExit(f"Missing {images} or {labels}; expected <root>/<split>/{{images,labels}}")

    lines = [f"{split}: {rel}" for split, rel in SPLITS.items()]
    lines += [f"nc: {len(CLASS_NAMES)}", f"names: {CLASS_NAMES}"]
    (dataset_dir / "data.yaml").write_text("\n".join(lines) + "\n")


def train() -> None:
    dataset_dir = Path(DATASET_DIR).resolve()
    output_dir = Path(OUTPUT_DIR).resolve()
    prepare_dataset(dataset_dir)

    kwargs = dict(
        dataset_dir=str(dataset_dir),
        output_dir=str(output_dir),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        grad_accum_steps=GRAD_ACCUM_STEPS,
        lr=LR,
        num_workers=NUM_WORKERS,
        early_stopping=True,
        early_stopping_patience=EARLY_STOPPING_PATIENCE,
        class_names=CLASS_NAMES,
        run_test=True,
        progress_bar="tqdm",
        tensorboard=True,
        mlflow=MLFLOW,
        project="fire-detection",
        run=output_dir.name,
    )
    if RESOLUTION:
        kwargs["resolution"] = RESOLUTION
    if RESUME_FROM:
        kwargs["resume"] = str(RESUME_FROM)

    model_class(MODEL_SIZE)().train(**kwargs)
    print(f"Done. Best checkpoint: {output_dir / 'checkpoint_best_total.pth'}")


def export() -> None:
    from rfdetr.detr import RFDETR

    model = RFDETR.from_checkpoint(str(EXPORT_CHECKPOINT))
    out = model.export(output_dir=str(EXPORT_DIR))
    print(f"Exported: {out}")


if __name__ == "__main__":
    export() if sys.argv[1:] == ["export"] else train()
