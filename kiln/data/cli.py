"""`kiln data …` commands. FiftyOne is imported lazily: it is slow to import."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Optional

import typer

from kiln import KilnError
from kiln.config import parse_classes

app = typer.Typer(no_args_is_help=True, help="Load datasets into FiftyOne, inspect them, export them for training.")

ClassesOpt = Annotated[
    Optional[str], typer.Option("--classes", help="Comma-separated class names; order = class ids. Overrides kiln.toml / the dataset.")
]


def _load_ds(name: str):
    import fiftyone as fo

    if not fo.dataset_exists(name):
        raise KilnError(f"No FiftyOne dataset named '{name}'. Existing: {', '.join(fo.list_datasets()) or '(none)'}")
    return fo.load_dataset(name)


def _launch(ds, port: int) -> None:
    """Serve a dataset in the FiftyOne app until Ctrl+C."""
    import fiftyone as fo

    session = fo.launch_app(ds, port=port, auto=False)
    typer.echo(f"FiftyOne app: {session.url}  (Ctrl+C to stop)")
    session.wait(-1)


PortOpt = Annotated[int, typer.Option(help="App port.")]


@app.command()
def load(
    path: Annotated[Path, typer.Argument(help="Dataset folder (YOLO, COCO, class folders or bare images; split or not).")],
    name: Annotated[str, typer.Option("--name", "-n", help="FiftyOne dataset name.")],
    replace: Annotated[bool, typer.Option("--replace", help="Overwrite an existing FiftyOne dataset with this name.")] = False,
    classes: ClassesOpt = None,
    view: Annotated[bool, typer.Option("--view/--no-view", help="Open the FiftyOne app after loading.")] = True,
    port: PortOpt = 5151,
):
    """Import a dataset folder into FiftyOne (persistent), with split folders as tags, then open it."""
    from kiln.data.info import render, summarize
    from kiln.data.load import load as do_load

    ds, report = do_load(path, name, replace=replace, classes=parse_classes(classes) if classes else None)
    lay = report.layout
    typer.echo(f"Detected {lay.format} dataset at {lay.root}")
    typer.echo(f"Classes: {report.classes or '(none)'}" + (f"  [from {report.classes_from}]" if report.classes_from else ""))
    for w in report.warnings:
        typer.secho(f"warning: {w}", fg="yellow")
    render(summarize(ds))
    if view:
        _launch(ds, port)
    else:
        typer.echo(f"\nReview it:  kiln data view {name}")


@app.command()
def info(name: Annotated[str, typer.Argument(help="FiftyOne dataset name.")]):
    """Show images, negatives and boxes per split for a FiftyOne dataset."""
    from kiln.data.info import render, summarize

    render(summarize(_load_ds(name)))


@app.command()
def view(
    name: Annotated[str, typer.Argument(help="FiftyOne dataset name.")],
    port: PortOpt = 5151,
):
    """Open a dataset in the FiftyOne app (Ctrl+C to stop)."""
    _launch(_load_ds(name), port)


@app.command()
def export(
    name: Annotated[str, typer.Argument(help="FiftyOne dataset name.")],
    out: Annotated[Path, typer.Option("--out", "-o", help="Output folder.")],
    fmt: Annotated[str, typer.Option("--format", "-f", help="yolo or coco.")] = "yolo",
    classes: ClassesOpt = None,
    overwrite: Annotated[bool, typer.Option("--overwrite", help="Replace an earlier kiln export in --out.")] = False,
    link: Annotated[bool, typer.Option("--link", help="Hard-link images instead of copying (same disk only).")] = False,
):
    """Write train/val/test folders from split tags. Fails if any image lacks a split or labels."""
    from kiln.data.common import SPLITS
    from kiln.data.export import export as do_export

    res = do_export(_load_ds(name), out, fmt=fmt, classes=parse_classes(classes) if classes else None,
                    overwrite=overwrite, link=link)
    total = sum(res.images.values())
    for split in SPLITS:
        n = res.images[split]
        per = ", ".join(f"{c} {res.boxes[split][c]}" for c in res.classes)
        typer.echo(f"{split:5} {n:6} images ({100 * n / total:4.1f}%), {res.negatives[split]} negatives; {per}")
    if res.clamped or res.dropped:
        typer.secho(f"clamped {res.clamped} boxes to the image edge, dropped {res.dropped} with no area", fg="yellow")
    if res.leaking_scenes:
        typer.secho(f"warning: {len(res.leaking_scenes)} scenes span several splits (near-duplicate images can "
                    f"leak between train and test): {', '.join(res.leaking_scenes[:10])}", fg="yellow")
    typer.echo(f"Exported {total} images as {fmt} to {res.out}")
