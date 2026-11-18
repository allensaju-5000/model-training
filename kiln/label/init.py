"""`kiln label init`: create a labelling workspace for a source folder."""

from __future__ import annotations

from pathlib import Path

from kiln import KilnError
from kiln.config import load_config
from kiln.data.common import natural_key
from kiln.data.load import read_source
from kiln.label.workspace import BOXES_HEADER, MANIFEST, NEG, SCENES_HEADER, Entry, Img, Workspace, format_entry


def _is_negative_folder(folder: str, negative_folders: list[str]) -> bool:
    parts = folder.split("/")
    return any(folder == nf or folder.endswith("/" + nf) or nf in parts for nf in negative_folders)


def init(src: Path, work: Path, classes: list[str] | None = None, guide: Path | None = None,
         grouping: bool = True):
    """Returns (workspace, report, counts) where counts says how many lines were prefilled."""
    from PIL import Image

    work = work.resolve()
    if (work / MANIFEST).exists():
        raise KilnError(f"{work} is already a labelling workspace; continue in it, or delete it to start over")

    records, report = read_source(src, classes)
    root = report.layout.root
    cfg = load_config(root)
    classes = report.classes
    if not classes or report.classes_from == "class ids (no names found)":
        raise KilnError("No class list: pass --classes (e.g. --classes smoke,fire) or add `classes` to kiln.toml")

    if guide is None and cfg.guide:
        guide = (cfg.path.parent / cfg.guide) if cfg.path else Path(cfg.guide)
    if guide is not None and not guide.is_file():
        raise KilnError(f"labelling guide {guide} not found")

    def folder(r) -> str:
        p = r.path.relative_to(root).parent.as_posix()
        return "" if p == "." else p

    records.sort(key=lambda r: (natural_key(folder(r)), natural_key(r.path.name)))
    width = max(4, len(str(len(records))))

    images, prefill = {}, []
    n_boxes = n_neg = 0
    for k, r in enumerate(records, 1):
        i = str(k).zfill(width)
        if r.size:
            w, h = r.size
        else:
            with Image.open(r.path) as im:
                w, h = im.size
        images[i] = Img(i, r.path.relative_to(root).as_posix(), w, h, folder(r))

        if r.boxes:  # existing labels: start from them, in pixels
            boxes = [(c, x * w, y * h, (x + bw) * w, (y + bh) * h) for c, (x, y, bw, bh) in r.boxes]
            prefill.append(format_entry(i, Entry("boxes", boxes)))
            n_boxes += 1
        elif r.boxes is not None or _is_negative_folder(folder(r), cfg.negative_folders):
            prefill.append(format_entry(i, Entry(NEG)))
            n_neg += 1

    ws = Workspace(work, root, classes, str(guide.resolve()) if guide else None, images, grouping)
    ws.save()
    ws.boxes_path.write_text(BOXES_HEADER + "".join(l + "\n" for l in prefill))
    if grouping:
        ws.scenes_path.write_text(SCENES_HEADER)
    return ws, report, {"boxes": n_boxes, "neg": n_neg}
