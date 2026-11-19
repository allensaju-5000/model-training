Trying out some model training

## Training

One config file per task in `configs/`, then:

    uv run kiln train run configs/fire.toml      # train
    uv run kiln train export configs/fire.toml   # edge model folder <output>/export: model.onnx, labels.txt, manifest.json
    uv run kiln train backends                   # what's supported

Supported backends (one file each in `kiln/train/backends/`):

- `rfdetr` — RF-DETR (nano, small, medium, large); COCO or YOLO datasets

To add one: write a `Backend` subclass in `kiln/train/backends/<name>.py` and add it to `BACKENDS` in `kiln/train/backends/__init__.py`.
