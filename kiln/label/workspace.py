"""The labelling workspace: plain files, so work can stop and resume anywhere.

    manifest.json   images (id -> path, size, folder), classes, guide, grouping; written by `init`
    boxes.txt       one line per image, in pixels of the original image, ending in its scene:
                        0007: fire 100 60 260 190; smoke 0 0 400 120 @lava_night
                        0008: NEG @forest_cam2   nothing to label (a negative)
                        0009: SKIP               leave the image out of the dataset
    scenes.txt      the scene catalogue, one line per scene (same place / camera / video):
                        lava_night: orange lava flow at night, hillside

Images in one scene always land in the same split. Scenes are decided while labelling:
each image either joins a scene already in scenes.txt or starts a new one. With grouping
off (`init --no-grouping`) lines carry no @scene and every image is split on its own.

Ids are short zero-padded numbers ("0007") so they are easy to read off a sheet.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from kiln import KilnError

MANIFEST = "manifest.json"
BOXES = "boxes.txt"
SCENES = "scenes.txt"
SHEETS = "sheets"
FEATURES = ".features.json"  # cache for kiln.label.match

NEG, SKIP = "NEG", "SKIP"
SCENE_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
_SCENE_TAIL = re.compile(r"\s*@(\S+)\s*$")

BOXES_HEADER = """\
# kiln boxes: one line per image, pixel coordinates of the original image (x0 y0 = top-left).
#   <id>: <class> x0 y0 x1 y1; <class> x0 y0 x1 y1 @<scene>
#   <id>: NEG @<scene>    nothing to label in this image (negative)
#   <id>: SKIP           leave this image out of the dataset
# <scene> is a name from scenes.txt (only when grouping is on; SKIP lines need none).
# '#' starts a comment. Check with: kiln label check --work <dir>
"""

SCENES_HEADER = """\
# kiln scenes: one line per scene (same place, camera, event or video).
#   <name>: <what it shows>
# An image belongs to a scene when its boxes.txt line ends in @<name>; a scene's images
# always share a split. Names: letters, digits, _ . -  (e.g. lava_night, forest_cam2).
"""


@dataclass
class Img:
    id: str
    rel: str  # path relative to the source folder
    w: int
    h: int
    folder: str = ""


@dataclass
class Workspace:
    dir: Path
    source: Path
    classes: list[str]
    guide: str | None
    images: dict[str, Img]  # id -> Img, in id order
    grouping: bool = True

    @property
    def boxes_path(self) -> Path:
        return self.dir / BOXES

    @property
    def scenes_path(self) -> Path:
        return self.dir / SCENES

    def path_of(self, i: str) -> Path:
        return self.source / self.images[i].rel

    @classmethod
    def open(cls, work: Path) -> "Workspace":
        mf = work / MANIFEST
        if not mf.is_file():
            raise KilnError(f"{work} is not a kiln labelling workspace (no {MANIFEST}); run `kiln label init` first")
        data = json.loads(mf.read_text())
        images = {d["id"]: Img(**d) for d in data["images"]}
        return cls(work.resolve(), Path(data["source"]), data["classes"], data.get("guide"), images,
                   data.get("grouping", True))

    def save(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        data = {"source": str(self.source), "classes": self.classes, "guide": self.guide,
                "grouping": self.grouping, "images": [vars(i) for i in self.images.values()]}
        (self.dir / MANIFEST).write_text(json.dumps(data, indent=1) + "\n")


# ---------------------------------------------------------------- ids ----


def expand_ids(spec: str, ws: Workspace) -> list[str]:
    """'0001-0004 0009,0012' -> ['0001', '0002', '0003', '0004', '0009', '0012']."""
    width = len(next(iter(ws.images)))
    out = []
    for tok in re.split(r"[\s,]+", spec.strip()):
        if not tok:
            continue
        m = re.fullmatch(r"(\d+)(?:-(\d+))?", tok)
        if not m:
            raise KilnError(f"bad id or range '{tok}' (use e.g. 0001 or 0001-0012)")
        a, b = int(m[1]), int(m[2] or m[1])
        for n in range(a, b + 1):
            i = str(n).zfill(width)
            if i not in ws.images:
                raise KilnError(f"no image with id {i}")
            out.append(i)
    return out


def compress_ids(ids: list[str]) -> str:
    """['0001', '0002', '0003', '0007'] -> '0001-0003 0007'."""
    nums = sorted(int(i) for i in ids)
    width = len(ids[0]) if ids else 4
    parts, start = [], None
    for k, n in enumerate(nums):
        if start is None:
            start = n
        if k + 1 == len(nums) or nums[k + 1] != n + 1:
            parts.append(str(start).zfill(width) if start == n else f"{str(start).zfill(width)}-{str(n).zfill(width)}")
            start = None
    return " ".join(parts)


# -------------------------------------------------------------- boxes ----


@dataclass
class Entry:
    kind: str  # "boxes" | NEG | SKIP
    boxes: list[tuple[str, float, float, float, float]] = field(default_factory=list)  # class, x0, y0, x1, y1 px
    scene: str | None = None
    line: int = 0


def parse_boxes(ws: Workspace, text: str | None = None) -> tuple[dict[str, Entry], list[str]]:
    """Parse boxes.txt. Returns (entries by id, problems). Boxes up to 3 px outside the image are clamped."""
    if text is None:
        text = ws.boxes_path.read_text() if ws.boxes_path.is_file() else ""
    entries: dict[str, Entry] = {}
    problems: list[str] = []
    width = len(next(iter(ws.images)))

    for n, raw in enumerate(text.splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if ":" not in line:
            problems.append(f"line {n}: expected '<id>: …', got {raw.strip()!r}")
            continue
        i, rest = (s.strip() for s in line.split(":", 1))
        i = i.zfill(width) if i.isdigit() else i
        if i not in ws.images:
            problems.append(f"line {n}: unknown image id {i!r}")
            continue
        if i in entries:
            problems.append(f"line {n}: image {i} already labelled on line {entries[i].line}")
            continue
        scene = None
        m = _SCENE_TAIL.search(rest)
        if m:
            scene, rest = m[1], rest[:m.start()].strip()
            if not SCENE_RE.fullmatch(scene):
                problems.append(f"line {n}: bad scene name {scene!r} (letters, digits, _ . - only)")
                continue
        if rest.upper() in (NEG, SKIP):
            entries[i] = Entry(rest.upper(), scene=scene, line=n)
            continue

        img, boxes, ok = ws.images[i], [], True
        for part in filter(None, (p.strip() for p in rest.split(";"))):
            toks = part.split()
            if len(toks) != 5:
                problems.append(f"line {n}: box {part!r} should be '<class> x0 y0 x1 y1'")
                ok = False
                continue
            cls = toks[0]
            if cls not in ws.classes:
                problems.append(f"line {n}: unknown class {cls!r} (classes: {', '.join(ws.classes)})")
                ok = False
                continue
            try:
                x0, y0, x1, y1 = (float(t) for t in toks[1:])
            except ValueError:
                problems.append(f"line {n}: coordinates in {part!r} are not numbers")
                ok = False
                continue
            if min(x0, y0) < -3 or x1 > img.w + 3 or y1 > img.h + 3:
                problems.append(f"line {n}: box {part!r} is outside the {img.w}x{img.h} image")
                ok = False
                continue
            x0, y0, x1, y1 = max(0, x0), max(0, y0), min(img.w, x1), min(img.h, y1)
            if x1 <= x0 or y1 <= y0:
                problems.append(f"line {n}: box {part!r} has x1<=x0 or y1<=y0")
                ok = False
                continue
            boxes.append((cls, x0, y0, x1, y1))
        if not boxes and ok:
            problems.append(f"line {n}: no boxes for {i}; write NEG if there is nothing to label")
        if ok and boxes:
            entries[i] = Entry("boxes", boxes, scene, line=n)
    return entries, problems


def format_entry(i: str, e: Entry) -> str:
    body = e.kind if e.kind != "boxes" else "; ".join(
        f"{c} {x0:.0f} {y0:.0f} {x1:.0f} {y1:.0f}" for c, x0, y0, x1, y1 in e.boxes)
    return f"{i}: {body}" + (f" @{e.scene}" if e.scene else "")


def is_done(ws: Workspace, e: Entry | None) -> bool:
    """Labelled, and (when grouping) placed in a scene. Prefilled lines without @scene still count as to do."""
    return e is not None and (e.kind == SKIP or not ws.grouping or e.scene is not None)


# ------------------------------------------------------------- scenes ----


def parse_scenes(ws: Workspace) -> tuple[dict[str, str], list[str]]:
    """scenes.txt -> (name -> description, problems)."""
    if not ws.scenes_path.is_file():
        return {}, []
    scenes: dict[str, str] = {}
    problems: list[str] = []
    for n, raw in enumerate(ws.scenes_path.read_text().splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        name, sep, desc = (s.strip() for s in line.partition(":"))
        if not sep or not SCENE_RE.fullmatch(name):
            problems.append(f"{SCENES} line {n}: expected '<name>: <what it shows>', got {raw.strip()!r}")
        elif name in scenes:
            problems.append(f"{SCENES} line {n}: scene {name!r} is listed twice")
        else:
            scenes[name] = desc
    return scenes, problems


def validate(ws: Workspace) -> tuple[dict[str, Entry], dict[str, str], list[str]]:
    """Parse boxes.txt and scenes.txt together. Returns (entries, scenes, problems)."""
    entries, problems = parse_boxes(ws)
    scenes, scene_problems = parse_scenes(ws)
    problems += scene_problems
    if ws.grouping:
        for e in entries.values():
            if e.scene and e.scene not in scenes:
                problems.append(f"line {e.line}: scene {e.scene!r} is not in {SCENES}; "
                                f"add a line '{e.scene}: <what it shows>' there")
    return entries, scenes, problems


def scene_members(ws: Workspace, entries: dict[str, Entry]) -> dict[str, list[str]]:
    """scene -> its kept image ids, in id order. Empty when grouping is off."""
    out: dict[str, list[str]] = {}
    if ws.grouping:
        for i in ws.images:
            e = entries.get(i)
            if e is not None and e.scene and e.kind != SKIP:
                out.setdefault(e.scene, []).append(i)
    return out


def merge_scenes(ws: Workspace, keep: str, others: list[str]) -> int:
    """Rename every @other in boxes.txt to @keep and drop the others from scenes.txt. Returns lines changed."""
    pat = re.compile(r"@(?:" + "|".join(map(re.escape, others)) + r")(?=[\s#]|$)", re.M)
    text, n = pat.subn("@" + keep, ws.boxes_path.read_text())
    ws.boxes_path.write_text(text)
    lines = ws.scenes_path.read_text().splitlines()
    kept = [l for l in lines if l.split("#", 1)[0].partition(":")[0].strip() not in others]
    ws.scenes_path.write_text("\n".join(kept) + "\n")
    return n
