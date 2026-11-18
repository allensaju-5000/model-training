"""`kiln label …`: helpers for labelling images by eye (used by the /label-split skill)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Annotated, Optional

import typer

from kiln import KilnError
from kiln.config import parse_classes, parse_ratios

app = typer.Typer(no_args_is_help=True, help="Label images into boxes and split them (workspace of plain files).")

WorkOpt = Annotated[Path, typer.Option("--work", "-w", help="Labelling workspace folder.")]


def _ws(work: Path):
    from kiln.label.workspace import Workspace

    return Workspace.open(work)


def _scene_names(spec: str, known) -> list[str]:
    """'all' or 'a,b c' -> scene names, all of which must exist."""
    if spec.strip().lower() == "all":
        return list(known)
    names = [t for t in re.split(r"[\s,]+", spec.strip()) if t]
    unknown = [n for n in names if n not in known]
    if unknown:
        raise KilnError(f"unknown scenes: {', '.join(unknown[:10])}")
    return names


@app.command()
def init(
    src: Annotated[Path, typer.Argument(help="Folder of images to label (any layout `kiln data load` reads).")],
    work: WorkOpt,
    classes: Annotated[Optional[str], typer.Option(help="Comma-separated class names; order = class ids.")] = None,
    guide: Annotated[Optional[Path], typer.Option(help="Labelling guide (markdown): what to box, edge cases.")] = None,
    grouping: Annotated[bool, typer.Option("--grouping/--no-grouping",
                                           help="Put every image in a scene while labelling (kept whole when splitting).")] = True,
):
    """Create a workspace: number the images, prefill existing labels and negative folders."""
    from kiln.label.init import init as do_init

    ws, report, counts = do_init(src, work, parse_classes(classes) if classes else None, guide, grouping)
    for w in report.warnings:
        typer.secho(f"warning: {w}", fg="yellow")
    folders = sorted({i.folder for i in ws.images.values()})
    typer.echo(f"Workspace {ws.dir}: {len(ws.images)} images from {ws.source}")
    typer.echo(f"Classes: {ws.classes}")
    typer.echo(f"Guide: {ws.guide or '(none; write one before labelling)'}")
    typer.echo(f"Folders: {', '.join(f or '.' for f in folders[:12])}{' …' if len(folders) > 12 else ''}")
    typer.echo(f"Prefilled boxes.txt: {counts['boxes']} with existing boxes, {counts['neg']} as NEG")
    typer.echo("Grouping: " + ("on (each line ends in @scene; catalogue in scenes.txt)" if grouping else "off"))


@app.command()
def sheet(
    work: WorkOpt,
    ids: Annotated[Optional[str], typer.Option(help="Image ids/ranges, e.g. '0001-0012 0020'.")] = None,
    scene: Annotated[Optional[str], typer.Option(help="Images of these scene(s): 'lava_night' or 'a,b'.")] = None,
    todo: Annotated[Optional[int], typer.Option(help="The next N images still to label (no line, or no @scene).")] = None,
    review_scenes: Annotated[Optional[str], typer.Option("--review-scenes", help="One row per scene: 'all' or scene names.")] = None,
    overlay: Annotated[bool, typer.Option("--overlay", help="Also mark unlabelled images (for checking).")] = False,
):
    """Render contact sheets into <work>/sheets/ and print their paths."""
    from functools import partial

    from kiln.label.match import candidates
    from kiln.label.sheets import label_sheets, scene_sheets
    from kiln.label.workspace import compress_ids, expand_ids, is_done, scene_members, validate

    ws = _ws(work)
    if sum(x is not None for x in (ids, scene, todo, review_scenes)) != 1:
        raise KilnError("pick exactly one of --ids, --scene, --todo, --review-scenes")
    if (scene or review_scenes) and not ws.grouping:
        raise KilnError("this workspace has grouping off (init --no-grouping): there are no scenes")
    entries, descs, _ = validate(ws)
    members = scene_members(ws, entries)

    if review_scenes:
        names = _scene_names(review_scenes, members)
        if not names:
            typer.echo("No scenes yet.")
            return
        label = "scenes" if review_scenes.strip().lower() == "all" else f"scenes-{names[0]}"
        for p in scene_sheets(ws, {n: members[n] for n in names}, label):
            typer.echo(str(p))
        for n in names:
            typer.echo(f"  @{n}: {len(members[n])} images  {descs.get(n, '')}")
        return

    if ids:
        sel, label = expand_ids(ids, ws), f"ids-{ids.split()[0].replace(',', '_')}"
    elif scene:
        names = _scene_names(scene, members)
        sel, label = [i for n in names for i in members[n]], f"scene-{names[0]}"
    else:
        sel = [i for i in ws.images if not is_done(ws, entries.get(i))][:todo]
        if not sel:
            typer.echo("Nothing left: every image is labelled" + (" and in a scene." if ws.grouping else "."))
            return
        label = f"todo-{sel[0]}"
    if overlay:
        label += "-overlay"
    scene_of = {i: e.scene for i, e in entries.items() if e.scene}
    cands_for = partial(candidates, ws, entries=entries, members=members) if ws.grouping and not overlay else None
    pages = label_sheets(ws, sel, label, entries, scene_of, overlay, cands_for)
    typer.echo(f"{len(sel)} images: {compress_ids(sel)}")
    for p, cands in pages:
        typer.echo(str(p))
        for s, ref in cands:
            typer.echo(f"  @{s} (ref {ref}): {descs.get(s, '')}")
    if ws.grouping and not overlay:
        typer.echo(f"{len(members)} scenes so far; all of them: {ws.scenes_path}")


@app.command()
def check(work: WorkOpt):
    """Validate boxes.txt and scenes.txt and show labelling progress."""
    from collections import Counter

    from kiln.label.workspace import compress_ids, is_done, scene_members, validate

    ws = _ws(work)
    entries, scenes, problems = validate(ws)
    kinds = Counter(e.kind for e in entries.values())
    nboxes = Counter(b[0] for e in entries.values() for b in e.boxes)
    todo = [i for i in ws.images if not is_done(ws, entries.get(i))]
    typer.echo(f"{len(ws.images)} images: {kinds['boxes']} boxed, {kinds['NEG']} NEG, {kinds['SKIP']} SKIP, "
               f"{len(todo)} to do")
    typer.echo("boxes: " + (", ".join(f"{c} {nboxes[c]}" for c in ws.classes)))
    if ws.grouping:
        members = scene_members(ws, entries)
        sizes = sorted((len(v) for v in members.values()), reverse=True)
        typer.echo(f"scenes: {len(members)} used" + (f" (largest {', '.join(map(str, sizes[:5]))})" if sizes else ""))
        unused = [s for s in scenes if s not in members]
        if unused:
            typer.secho(f"scenes in scenes.txt with no images: {', '.join(unused[:20])}", fg="yellow")
        ungrouped = [i for i in todo if i in entries]
        if ungrouped:
            typer.echo(f"labelled but no @scene yet: {compress_ids(ungrouped)[:200]}")
    if todo:
        typer.echo(f"to do: {compress_ids(todo)[:200]}")
    if problems:
        typer.secho(f"{len(problems)} problems:", fg="red")
        for p in problems[:30]:
            typer.echo(f"  {p}")
        raise typer.Exit(1)
    typer.secho("boxes.txt OK", fg="green")


@app.command()
def merge(
    work: WorkOpt,
    keep: Annotated[str, typer.Argument(help="Scene to keep.")],
    others: Annotated[list[str], typer.Argument(help="Scenes to fold into it (renamed in boxes.txt, removed from scenes.txt).")],
):
    """Merge scenes that turned out to be the same one."""
    from kiln.label.workspace import merge_scenes, parse_scenes

    ws = _ws(work)
    scenes, _ = parse_scenes(ws)
    others = [o.lstrip("@") for o in others]
    keep = keep.lstrip("@")
    unknown = [s for s in [keep, *others] if s not in scenes]
    if unknown:
        raise KilnError(f"not in scenes.txt: {', '.join(unknown)}")
    if keep in others:
        raise KilnError("the scene to keep is also listed to merge away")
    n = merge_scenes(ws, keep, others)
    typer.echo(f"Merged {', '.join(others)} into @{keep}: {n} lines in boxes.txt updated")


@app.command()
def build(
    work: WorkOpt,
    out: Annotated[Path, typer.Option("--out", "-o", help="Output dataset folder.")],
    ratios: Annotated[Optional[str], typer.Option(help="Split as 'train val test' percentages, e.g. '70 15 15'. Omit for no split.")] = None,
    seed: Annotated[int, typer.Option(help="Random seed for tie-breaks in the split.")] = 0,
    overwrite: Annotated[bool, typer.Option("--overwrite", help="Replace an earlier build in --out.")] = False,
    link: Annotated[bool, typer.Option("--link", help="Hard-link images instead of copying.")] = False,
):
    """Write the labelled (and optionally split) dataset as YOLO, ready for `kiln data load`."""
    from rich.console import Console
    from rich.table import Table

    from kiln.data.common import SPLITS
    from kiln.label.build import build as do_build

    ws = _ws(work)
    r = parse_ratios(re.split(r"[\s,/]+", ratios.strip())) if ratios else None
    res = do_build(ws, out, r, seed=seed, overwrite=overwrite, link=link)

    splits = [s for s in SPLITS if res.images[s]] if res.split else [""]
    total = sum(res.images.values())
    t = Table(title=f"{total} images ({res.skipped} skipped), {res.groups} multi-image scenes")
    t.add_column("stratum (folder | classes)")
    for s in splits:
        t.add_column(s or "images", justify="right")
    for k in sorted(res.strata):
        row = res.strata[k]
        n = sum(row.values())
        t.add_row(k, *[f"{row[s]} ({100 * row[s] / n:.0f}%)" if res.split else str(row[s]) for s in splits])
    t.add_row("[b]all[/]", *[f"[b]{res.images[s]} ({100 * res.images[s] / total:.1f}%)[/]" for s in splits])
    Console().print(t)
    for s in splits:
        per = ", ".join(f"{c} {res.boxes[s][c]}" for c in ws.classes)
        typer.echo(f"{s or 'all'}: {res.negatives[s]} negatives; boxes {per}")
    typer.echo(f"Built {res.out}\nNext: kiln data load {res.out} --name <name>  (opens the FiftyOne app)")

