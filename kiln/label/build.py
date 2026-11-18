"""`kiln label build`: turn a finished workspace into a YOLO dataset `kiln data load` can read.

    with ratios:  <out>/{train,val,test}/{images,labels}/ + data.yaml + groups.json
    without:      <out>/images/ + <out>/labels/ + data.yaml + groups.json

Every image must have a line in boxes.txt (boxes, NEG or SKIP) and, with grouping on, a
@scene; SKIP images are left out. Scenes are kept whole when splitting and written to
groups.json, which `kiln data load` turns into scene:<name> tags.
"""

from __future__ import annotations

import json
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from kiln import KilnError
from kiln.data.common import GROUPS_FILE, SPLITS, clean_name, place_file
from kiln.label.assign import assign
from kiln.label.workspace import NEG, SKIP, Workspace, compress_ids, is_done, scene_members, validate

MARKER = ".kiln-build"


@dataclass
class BuildResult:
    out: Path
    split: bool
    images: Counter = field(default_factory=Counter)  # split ("" if unsplit) -> images
    negatives: Counter = field(default_factory=Counter)
    boxes: dict = field(default_factory=lambda: defaultdict(Counter))
    strata: dict = field(default_factory=lambda: defaultdict(Counter))  # stratum -> split -> images
    skipped: int = 0
    groups: int = 0


def stratum(folder: str, kind: str, classes: list[str]) -> str:
    what = "NEG" if kind == NEG else "+".join(classes)
    return f"{folder or '.'} | {what}"


def build(ws: Workspace, out: Path, ratios: tuple[int, int, int] | None, seed: int = 0,
          overwrite: bool = False, link: bool = False) -> BuildResult:
    entries, _, problems = validate(ws)
    if problems:
        raise KilnError("boxes.txt / scenes.txt have problems; fix them first (kiln label check):\n  "
                        + "\n  ".join(problems[:20]))
    missing = [i for i in ws.images if not is_done(ws, entries.get(i))]
    if missing:
        what = "a line with a @scene" if ws.grouping else "a line"
        raise KilnError(f"{len(missing)} images have no {what} in boxes.txt yet: {compress_ids(missing)[:300]}")

    keep = [i for i in ws.images if entries[i].kind != SKIP]
    groups = scene_members(ws, entries)
    # every kept image belongs to exactly one unit: its scene, or itself when grouping is off
    units: dict[str, list[str]] = defaultdict(list)
    for i in keep:
        units[entries[i].scene if ws.grouping else f"_{i}"].append(i)

    def strat(i: str) -> str:
        e = entries[i]
        return stratum(ws.images[i].folder, e.kind, sorted({b[0] for b in e.boxes}))

    if ratios:
        split_of_unit = assign({u: Counter(strat(i) for i in ids) for u, ids in units.items()}, ratios, seed)
        split_of = {i: split_of_unit[u] for u, ids in units.items() for i in ids}
    else:
        split_of = {i: "" for i in keep}

    out = out.resolve()
    if out.exists() and any(out.iterdir()):
        if not overwrite:
            raise KilnError(f"{out} is not empty; use --overwrite to replace an earlier build")
        if not (out / MARKER).is_file():
            raise KilnError(f"{out} was not created by kiln label build (no {MARKER}); refusing to delete it")
        shutil.rmtree(out)

    res = BuildResult(out, split=bool(ratios), skipped=len(ws.images) - len(keep))
    taken: dict[str, set[str]] = defaultdict(set)
    rel_of: dict[str, str] = {}
    for i in keep:
        img, e, s = ws.images[i], entries[i], split_of[i]
        base = out / s if s else out
        (base / "images").mkdir(parents=True, exist_ok=True)
        (base / "labels").mkdir(parents=True, exist_ok=True)

        name = clean_name(Path(img.rel))
        stem, ext, n = Path(name).stem, Path(name).suffix, 1
        while name in taken[s]:
            n += 1
            name = f"{stem}_{n}{ext}"
        taken[s].add(name)

        place_file(ws.path_of(i), base / "images" / name, link)
        lines = []
        for cls, x0, y0, x1, y1 in e.boxes:
            cx, cy = (x0 + x1) / 2 / img.w, (y0 + y1) / 2 / img.h
            bw, bh = (x1 - x0) / img.w, (y1 - y0) / img.h
            lines.append(f"{ws.classes.index(cls)} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            res.boxes[s][cls] += 1
        (base / "labels" / name).with_suffix(".txt").write_text("".join(l + "\n" for l in lines))
        rel_of[i] = (f"{s}/" if s else "") + f"images/{name}"
        res.images[s] += 1
        res.negatives[s] += e.kind == NEG
        res.strata[strat(i)][s] += 1

    yaml = ["path: ."] + ([f"{s}: {s}/images" for s in SPLITS if res.images[s]] if ratios else ["train: images"])
    yaml += [f"nc: {len(ws.classes)}", f"names: {json.dumps(ws.classes)}"]
    (out / "data.yaml").write_text("\n".join(yaml) + "\n")

    scene_groups = {g: [rel_of[i] for i in ids if i in rel_of] for g, ids in groups.items()}
    scene_groups = {g: v for g, v in scene_groups.items() if len(v) > 1}
    if scene_groups:
        (out / GROUPS_FILE).write_text(json.dumps({"groups": scene_groups}, indent=1) + "\n")
    res.groups = len(scene_groups)
    (out / MARKER).write_text(json.dumps({"workspace": str(ws.dir), "ratios": ratios, "seed": seed}) + "\n")
    return res
