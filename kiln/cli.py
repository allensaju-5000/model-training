"""kiln command line. Command groups: `data`, `label`, `train`, `skill`; `mlflow` later."""

from __future__ import annotations

import typer

from kiln import KilnError
from kiln.data.cli import app as data_app
from kiln.label.cli import app as label_app
from kiln.skill_cli import app as skill_app
from kiln.train.cli import app as train_app

_app = typer.Typer(no_args_is_help=True, help="kiln: dataset and model workflow CLI.")
_app.add_typer(data_app, name="data")
_app.add_typer(label_app, name="label")
_app.add_typer(train_app, name="train")
_app.add_typer(skill_app, name="skill")


def app() -> None:
    try:
        _app()
    except KilnError as e:
        typer.secho(f"error: {e}", fg="red", err=True)
        raise SystemExit(1)
