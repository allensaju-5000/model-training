"""`kiln data export`: write a reviewed FiftyOne dataset as train-ready folders.

Export does no splitting: every image must already carry exactly one of the
``train`` / ``val`` / ``test`` tags, be labelled (an empty label = negative), and
use only known classes. Anything else stops the export with a list of offenders.

    yolo   <out>/{train,val,test}/{images,labels}/ + data.yaml
    coco   <out>/{train,valid,test}/ images + _annotations.coco.json  (RF-DETR / Roboflow layout)

Both also get ``groups.json`` (scene tags) and a ``.kiln-export`` marker, which is
what lets ``--overwrite`` safely replace an earlier export.
"""

from __future__ import annotations

import json
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from kiln import KilnError
from kiln.data.common import GROUPS_FILE, LABEL_FIELD, SCENE_PREFIX, SPLITS, clean_name, place_file, split_tags

FORMATS = ("yolo", "coco")
MARKER = ".kiln-export"
SPLIT_DIRS = {"yolo": {"train": "train", "val": "val", "test": "test"},
              "coco": {"train": "train", "val": "valid", "test": "test"}}
MAX_EXAMPLES = 8


@dataclass
class Item:
    src: Path
    split: str
    boxes: list[tuple[int, list[float]]]  # class id, [x, y, w, h] relative, top-left, clamped
    size: tuple[int, int] | None
    scenes: list[str]
    dest: str = ""  # file name in the output split folder


@dataclass
class Result:
    out: Path
    classes: list[str] = field(default_factory=list)
    images: Counter = field(default_factory=Counter)
    negatives: Counter = field(default_factory=Counter)
    boxes: dict = field(default_factory=lambda: defaultdict(Counter))
    clamped: int = 0
    dropped: int = 0
    leaking_scenes: list[str] = field(default_factory=list)


def _clamp(box: list[float]) -> tuple[list[float] | None, bool]:
    x, y, w, h = box
    x0, y0, x1, y1 = max(0.0, x), max(0.0, y), min(1.0, x + w), min(1.0, y + h)
    if x1 - x0 <= 0 or y1 - y0 <= 0:
        return None, True
    out = [x0, y0, x1 - x0, y1 - y0]
    return out, any(abs(a - b) > 1e-9 for a, b in zip(out, box))


def _collect(ds, classes: list[str]) -> tuple[list[Item], int, int]:
    fields = ["filepath", "tags", f"{LABEL_FIELD}.detections.label",
              f"{LABEL_FIELD}.detections.bounding_box", "metadata.width", "metadata.height"]
    problems: dict[str, list[str]] = defaultdict(list)
    items, clamped, dropped = [], 0, 0
    index = {c: i for i, c in enumerate(classes)}

    for fp, tags, labels, bbs, w, h in zip(*ds.values(fields)):
        st = split_tags(tags)
        if not st:
            problems["no split tag (train/val/test)"].append(fp)
        elif len(st) > 1:
            problems["several split tags"].append(f"{fp}  {st}")
        if labels is None:
            problems["unlabelled (no ground_truth; give it boxes or an empty label)"].append(fp)
        else:
            unknown = sorted({l for l in labels if l not in index})
            if unknown:
                problems["labels not in the class list"].append(f"{fp}  {unknown}")
        if not Path(fp).is_file():
            problems["image file missing"].append(fp)
        if len(st) != 1 or labels is None:
            continue

        boxes = []
        for lbl, bb in zip(labels, bbs):
            if lbl not in index:
                continue
            box, changed = _clamp(bb)
            if box is None:
                dropped += 1
                continue
            clamped += changed
            boxes.append((index[lbl], box))
        scenes = [t[len(SCENE_PREFIX):] for t in tags if t.startswith(SCENE_PREFIX)]
        items.append(Item(Path(fp), st[0], boxes, (w, h) if w and h else None, scenes))

    present = {i.split for i in items}
    for s in SPLITS:
        if s not in present and not any(problems.values()):
            problems["empty splits (export needs train, val and test)"].append(s)

    if problems:
        lines = []
        for kind, offenders in problems.items():
            lines.append(f"{kind}: {len(offenders)}")
            lines += [f"    {o}" for o in offenders[:MAX_EXAMPLES]]
            if len(offenders) > MAX_EXAMPLES:
                lines.append(f"    … and {len(offenders) - MAX_EXAMPLES} more")
        raise KilnError(
            f"Dataset '{ds.name}' is not ready to export:\n" + "\n".join(lines)
            + "\nFix these in the FiftyOne app (tags / labels), then run export again. "
            "`kiln data info` shows the current state."
        )
    return items, clamped, dropped


def _prepare_out(out: Path, overwrite: bool) -> None:
    if out.exists() and any(out.iterdir()):
        if not overwrite:
            raise KilnError(f"{out} is not empty; use --overwrite to replace an earlier kiln export")
        if not (out / MARKER).is_file():
            raise KilnError(f"{out} was not created by kiln export (no {MARKER}); refusing to delete it")
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)


def _assign_names(items: list[Item]) -> None:
    taken: dict[str, set[str]] = defaultdict(set)
    for it in items:
        name = clean_name(it.src)
        stem, ext, n = Path(name).stem, Path(name).suffix, 1
        while name in taken[it.split]:
            n += 1
            name = f"{stem}_{n}{ext}"
        taken[it.split].add(name)
        it.dest = name


def _image_size(it: Item) -> tuple[int, int]:
    if it.size:
        return it.size
    from PIL import Image

    with Image.open(it.src) as im:
        return im.size


def _write_yolo(items: list[Item], out: Path, classes: list[str], link: bool) -> None:
    dirs = SPLIT_DIRS["yolo"]
    for s in SPLITS:
        (out / dirs[s] / "images").mkdir(parents=True, exist_ok=True)
        (out / dirs[s] / "labels").mkdir(parents=True, exist_ok=True)
    for it in items:
        d = out / dirs[it.split]
        place_file(it.src, d / "images" / it.dest, link)
        lines = [f"{c} {x + w / 2:.6f} {y + h / 2:.6f} {w:.6f} {h:.6f}" for c, (x, y, w, h) in it.boxes]
        (d / "labels" / it.dest).with_suffix(".txt").write_text("".join(l + "\n" for l in lines))
    yaml_lines = ["path: ."] + [f"{s}: {dirs[s]}/images" for s in SPLITS]
    yaml_lines += [f"nc: {len(classes)}", f"names: {json.dumps(classes)}"]
    (out / "data.yaml").write_text("\n".join(yaml_lines) + "\n")


def _write_coco(items: list[Item], out: Path, classes: list[str], link: bool) -> None:
    dirs = SPLIT_DIRS["coco"]
    categories = [{"id": i, "name": c, "supercategory": c} for i, c in enumerate(classes)]
    for s in SPLITS:
        d = out / dirs[s]
        d.mkdir(parents=True, exist_ok=True)
        images, anns = [], []
        for it in (i for i in items if i.split == s):
            place_file(it.src, d / it.dest, link)
            W, H = _image_size(it)
            images.append({"id": len(images) + 1, "file_name": it.dest, "width": W, "height": H})
            for c, (x, y, w, h) in it.boxes:
                px = [x * W, y * H, w * W, h * H]
                anns.append({"id": len(anns) + 1, "image_id": len(images), "category_id": c, "bbox": px,
                             "area": px[2] * px[3], "iscrowd": 0, "segmentation": []})
        data = {"info": {"description": "kiln export"}, "licenses": [], "categories": categories,
                "images": images, "annotations": anns}
        (d / "_annotations.coco.json").write_text(json.dumps(data))


def export(ds, out: Path, fmt: str = "yolo", classes: list[str] | None = None,
           overwrite: bool = False, link: bool = False) -> Result:
    if fmt not in FORMATS:
        raise KilnError(f"unknown format '{fmt}' (choose from {', '.join(FORMATS)})")
    classes = classes or list(ds.default_classes or [])
    if not classes:
        raise KilnError(f"Dataset '{ds.name}' has no class list; pass --classes (order = class ids)")

    items, clamped, dropped = _collect(ds, classes)
    out = out.resolve()
    _prepare_out(out, overwrite)
    _assign_names(items)
    (_write_yolo if fmt == "yolo" else _write_coco)(items, out, classes, link)

    groups: dict[str, list[str]] = defaultdict(list)
    dirs = SPLIT_DIRS[fmt]
    for it in items:
        rel = f"{dirs[it.split]}/images/{it.dest}" if fmt == "yolo" else f"{dirs[it.split]}/{it.dest}"
        for sc in it.scenes:
            groups[sc].append(rel)
    if groups:
        (out / GROUPS_FILE).write_text(json.dumps({"groups": groups}, indent=1) + "\n")
    (out / MARKER).write_text(json.dumps({
        "dataset": ds.name, "format": fmt, "classes": classes,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }, indent=1) + "\n")

    res = Result(out, classes, clamped=clamped, dropped=dropped)
    scene_splits: dict[str, set[str]] = defaultdict(set)
    for it in items:
        for sc in it.scenes:
            scene_splits[sc].add(it.split)
    res.leaking_scenes = sorted(sc for sc, sp in scene_splits.items() if len(sp) > 1)
    for it in items:
        res.images[it.split] += 1
        if not it.boxes:
            res.negatives[it.split] += 1
        for c, _ in it.boxes:
            res.boxes[it.split][classes[c]] += 1
    return res
