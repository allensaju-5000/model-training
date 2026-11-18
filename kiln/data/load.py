"""`kiln data load`: import a dataset folder into a persistent FiftyOne dataset.

Whatever the source format, the result looks the same:
  - boxes live in the ``ground_truth`` field (fo.Detections, label + box only);
    an image with an empty Detections is a labelled negative, a missing field means unlabelled
  - split folders become ``train`` / ``val`` / ``test`` sample tags
  - ``groups.json`` entries become ``scene:<name>`` tags
  - images in an unlabelled set get a ``folder:<path>`` tag as a hint (e.g. class folders)
  - ``dataset.default_classes`` holds the class order (= class ids on export)
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from kiln import KilnError
from kiln.config import load_config
from kiln.data.common import FOLDER_PREFIX, GROUPS_FILE, LABEL_FIELD, SCENE_PREFIX, iter_images, split_of
from kiln.data.detect import Layout, coco_categories, detect, split_from_stem


@dataclass
class Record:
    path: Path
    split: str | None
    boxes: list[tuple[str, list[float]]] | None  # None = unlabelled; box = [x, y, w, h] relative, top-left
    size: tuple[int, int] | None = None
    tags: list[str] = field(default_factory=list)


@dataclass
class Report:
    layout: Layout
    classes: list[str] | None
    classes_from: str | None
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------- readers ----


def _read_yolo(layout: Layout, classes: list[str] | None, report: Report) -> list[Record]:
    records, malformed, bad_ids = [], Counter(), Counter()
    for images, labels in layout.yolo_pairs:
        for img in iter_images(images):
            txt = labels / img.relative_to(images).with_suffix(".txt")
            boxes = []
            if txt.is_file():
                for line in txt.read_text().splitlines():
                    parts = line.split()
                    if not parts:
                        continue
                    try:
                        cid, vals = int(float(parts[0])), [float(v) for v in parts[1:]]
                    except ValueError:
                        malformed[txt.name] += 1
                        continue
                    if len(vals) == 4:
                        cx, cy, w, h = vals
                        box = [cx - w / 2, cy - h / 2, w, h]
                    elif len(vals) >= 6 and len(vals) % 2 == 0:  # segmentation polygon -> its bounding box
                        xs, ys = vals[0::2], vals[1::2]
                        box = [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)]
                    else:
                        malformed[txt.name] += 1
                        continue
                    if classes is not None and not 0 <= cid < len(classes):
                        bad_ids[cid] += 1
                        continue
                    boxes.append((classes[cid] if classes else str(cid), box))
            records.append(Record(img, split_of(img.relative_to(layout.root)), boxes))

    if bad_ids:
        raise KilnError(
            f"Label files use class ids {sorted(bad_ids)} but only {len(classes or [])} classes are known "
            f"({classes}). Fix the class list (--classes or kiln.toml)."
        )
    if malformed:
        report.warnings.append(f"skipped {sum(malformed.values())} malformed label lines in {len(malformed)} files")
    return records


def _resolve_coco_image(json_path: Path, root: Path, file_name: str) -> Path | None:
    base = json_path.parent
    for cand in (base / file_name, base / "data" / file_name, base / "images" / file_name, root / file_name):
        if cand.is_file():
            return cand
    return None


def _read_coco(layout: Layout, classes: list[str] | None, report: Report) -> list[Record]:
    records, missing = [], 0
    for jp in layout.coco_files:
        data = json.loads(jp.read_text())
        names = {int(c["id"]): c["name"] for c in data.get("categories", [])}
        if classes is not None:
            unknown = {n for n in (c["name"] for c in coco_categories(data)) if n not in classes}
            if unknown:
                raise KilnError(f"{jp.name}: categories {sorted(unknown)} are not in the class list {classes}")

        anns: dict[int, list[dict]] = {}
        for a in data.get("annotations", []):
            anns.setdefault(int(a["image_id"]), []).append(a)

        for im in data.get("images", []):
            path = _resolve_coco_image(jp, layout.root, im["file_name"])
            if path is None:
                missing += 1
                continue
            W, H = int(im["width"]), int(im["height"])
            boxes = []
            for a in anns.get(int(im["id"]), []):
                x, y, w, h = (float(v) for v in a["bbox"])
                boxes.append((names[int(a["category_id"])], [x / W, y / H, w / W, h / H]))
            split = split_of(jp.relative_to(layout.root)) or split_of(path.relative_to(layout.root))
            split = split or split_from_stem(jp)
            records.append(Record(path, split, boxes, size=(W, H)))

    if missing:
        report.warnings.append(f"{missing} images listed in the COCO json were not found on disk and were skipped")
    return records


def _read_images(layout: Layout) -> list[Record]:
    records = []
    for img in iter_images(layout.root):
        rel = img.relative_to(layout.root)
        tags = [FOLDER_PREFIX + rel.parent.as_posix()] if rel.parent != Path(".") else []
        records.append(Record(img, split_of(rel), None, tags=tags))
    return records


def _apply_groups(root: Path, records: list[Record], report: Report) -> None:
    gfile = root / GROUPS_FILE
    if not gfile.is_file():
        return
    data = json.loads(gfile.read_text())
    groups = data.get("groups", data) if isinstance(data, dict) else {}
    by_rel = {r.path.relative_to(root).as_posix(): r for r in records}
    unknown = 0
    for name, members in groups.items():
        for rel in members:
            r = by_rel.get(rel)
            if r is None:
                unknown += 1
            else:
                r.tags.append(SCENE_PREFIX + str(name))
    if unknown:
        report.warnings.append(f"{GROUPS_FILE}: {unknown} entries did not match an image and were ignored")


# ------------------------------------------------------------------ load ----


def read_source(path: Path, classes: list[str] | None = None) -> tuple[list[Record], Report]:
    """Detect a dataset folder and read every image (with its boxes, split and tags) from it."""
    layout = detect(path)
    cfg = load_config(layout.root)

    if classes:
        classes_from = "--classes"
    elif cfg.classes:
        classes, classes_from = cfg.classes, str(cfg.path)
    else:
        classes, classes_from = layout.classes, layout.classes_from
    report = Report(layout, classes, classes_from)

    if layout.format == "yolo":
        records = _read_yolo(layout, classes, report)
        if classes is None:
            ids = sorted({int(lbl) for r in records for lbl, _ in r.boxes or []})
            report.classes = [str(i) for i in range(ids[-1] + 1)] if ids else None
            report.classes_from = "class ids (no names found)"
            report.warnings.append("no class names found; using numeric ids. Pass --classes or add kiln.toml.")
    elif layout.format == "coco":
        records = _read_coco(layout, classes, report)
    else:
        records = _read_images(layout)

    if not records:
        raise KilnError(f"No images could be read from {layout.root}")
    _apply_groups(layout.root, records, report)
    return records, report


def load(path: Path, name: str, replace: bool = False, classes: list[str] | None = None):
    """Import ``path`` as FiftyOne dataset ``name``. Returns (dataset, Report)."""
    import fiftyone as fo

    records, report = read_source(path, classes)
    classes = report.classes
    layout = report.layout

    if fo.dataset_exists(name):
        if not replace:
            raise KilnError(f"FiftyOne dataset '{name}' already exists; use --replace to overwrite it")
        fo.delete_dataset(name)

    ds = fo.Dataset(name, persistent=True)
    ds.add_sample_field(LABEL_FIELD, fo.EmbeddedDocumentField, embedded_doc_type=fo.Detections)

    samples = []
    for r in records:
        tags = ([r.split] if r.split else []) + r.tags
        s = fo.Sample(filepath=str(r.path), tags=tags)
        if r.boxes is not None:
            s[LABEL_FIELD] = fo.Detections(
                detections=[fo.Detection(label=lbl, bounding_box=box) for lbl, box in r.boxes]
            )
        if r.size:
            s.metadata = fo.ImageMetadata(width=r.size[0], height=r.size[1])
        samples.append(s)
    ds.add_samples(samples, progress=True)

    if classes:
        ds.default_classes = classes
    ds.info["kiln"] = {"source": str(layout.root), "format": layout.format}
    ds.save()
    return ds, report

