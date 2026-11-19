"""`kiln train …` commands."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Optional

import typer

app = typer.Typer(no_args_is_help=True, help="Train detection models from a config file and export them to ONNX.")

ConfigArg = Annotated[Path, typer.Argument(help="Training config (TOML), e.g. configs/fire.toml.")]


@app.command()
def run(config: ConfigArg):
    """Train a model as described in the config."""
    from kiln.train.config import load_train_config
    from kiln.train.run import train

    cfg = load_train_config(config)
    typer.echo(f"Training {cfg.backend} / {cfg.model} on {cfg.dataset}\nOutput: {cfg.output_dir}")
    best = train(cfg)
    typer.echo(f"Done. Best weights: {best}\nExport:  kiln train export {config}")


@app.command()
def export(
    config: ConfigArg,
    weights: Annotated[Optional[Path], typer.Option("--weights", "-w", help="Weights file. Default: the run's best.")] = None,
):
    """Write the edge model folder <output>/export: model.onnx, labels.txt, manifest.json."""
    import json

    from kiln.train.config import load_train_config
    from kiln.train.run import export as do_export

    out = do_export(load_train_config(config), weights)
    m = json.loads((out / "manifest.json").read_text())
    typer.echo(f"Exported {m['name']} -> {out}  (model.onnx, labels.txt, manifest.json; "
               f"arch {m['arch']}, sha256 {m['sha256'][:12]})")


@app.command()
def backends():
    """List the training libraries kiln can use, and their models."""
    from kiln.train.backends import BACKENDS

    for name, cls in BACKENDS.items():
        typer.echo(f"{name:12} {cls.about}\n{'':12} models: {', '.join(cls.models)}   datasets: {', '.join(cls.formats)}")
