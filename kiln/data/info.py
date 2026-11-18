"""`kiln data info`: what a FiftyOne dataset holds, per split."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field

from kiln.data.common import LABEL_FIELD, SCENE_PREFIX, SPLITS, split_tags

NO_SPLIT = "(no split)"
MULTI = "(several splits)"


@dataclass
class Summary:
    name: str
    classes: list[str]
    total: int = 0
    images: Counter = field(default_factory=Counter)  # split -> images
    unlabelled: Counter = field(default_factory=Counter)  # split -> images without a label field
    negatives: Counter = field(default_factory=Counter)  # split -> labelled images with no boxes
    boxes: dict = field(default_factory=lambda: defaultdict(Counter))  # split -> label -> boxes
    scenes: set = field(default_factory=set)
    scene_splits: dict = field(default_factory=lambda: defaultdict(set))  # scene -> splits it touches

    @property
    def rows(self) -> list[str]:
        order = [*SPLITS, NO_SPLIT, MULTI]
        return [s for s in order if self.images[s]]

    @property
    def leaking_scenes(self) -> list[str]:
        return sorted(s for s, sp in self.scene_splits.items() if len(sp - {NO_SPLIT}) > 1)


def summarize(ds) -> Summary:
    # labels is None for an unlabelled image and [] for a labelled negative
    tags_list, labels_list = ds.values(["tags", f"{LABEL_FIELD}.detections.label"])
    s = Summary(ds.name, list(ds.default_classes or []), total=len(tags_list))

    for tags, labels in zip(tags_list, labels_list):
        st = split_tags(tags)
        split = st[0] if len(st) == 1 else (NO_SPLIT if not st else MULTI)
        s.images[split] += 1
        if labels is None:
            s.unlabelled[split] += 1
        elif not labels:
            s.negatives[split] += 1
        for lbl in labels or []:
            s.boxes[split][lbl] += 1
        for t in tags:
            if t.startswith(SCENE_PREFIX):
                s.scenes.add(t)
                s.scene_splits[t].add(split)
    return s


def render(s: Summary) -> None:
    from rich.console import Console
    from rich.table import Table

    labels = list(s.classes) + sorted({l for c in s.boxes.values() for l in c} - set(s.classes))
    t = Table(title=f"{s.name}  ({s.total} images)", caption="class columns = box counts" if labels else None)
    for col in ["split", "images", "%", "negatives", "unlabelled", *labels]:
        t.add_column(col, justify="left" if col == "split" else "right")
    for split in s.rows:
        n = s.images[split]
        t.add_row(
            split, str(n), f"{100 * n / s.total:.1f}", str(s.negatives[split]), str(s.unlabelled[split]),
            *[str(s.boxes[split][l]) for l in labels],
        )
    con = Console()
    con.print(t)
    con.print(f"classes: {s.classes or '(none set)'}")
    unknown = [l for l in labels if l not in s.classes]
    if unknown:
        con.print(f"[yellow]labels not in the class list: {unknown}[/]")
    if s.scenes:
        con.print(f"scenes: {len(s.scenes)}")
        if s.leaking_scenes:
            con.print(f"[yellow]scenes spread over several splits: {', '.join(s.leaking_scenes[:10])}"
                      + (" …" if len(s.leaking_scenes) > 10 else "") + "[/]")
