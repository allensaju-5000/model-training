"""`kiln skill …`: AI-assisted workflows shipped with kiln as Claude Code skills.

Each skill is a folder under ``kiln/skills/<name>/`` holding a ``SKILL.md``. The
skill tells Claude how to drive the kiln CLI (e.g. ``/label-split`` labels and
splits a folder of images). ``kiln skill install`` copies them into a project's
``.claude/skills/`` so Claude Code picks them up as slash commands.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Annotated, Optional

import typer

from kiln import KilnError

SKILLS_DIR = Path(__file__).parent / "skills"

app = typer.Typer(no_args_is_help=True, help="AI-assisted workflows (Claude Code skills) bundled with kiln.")


def _available() -> dict[str, Path]:
    return {p.parent.name: p.parent for p in sorted(SKILLS_DIR.glob("*/SKILL.md"))}


def _description(skill_md: Path) -> str:
    for line in skill_md.read_text().splitlines():
        if line.startswith("description:"):
            return line.split(":", 1)[1].strip()
    return ""


def _pick(names: list[str] | None) -> dict[str, Path]:
    skills = _available()
    if not names:
        return skills
    unknown = [n for n in names if n not in skills]
    if unknown:
        raise KilnError(f"unknown skills: {', '.join(unknown)} (available: {', '.join(skills) or 'none'})")
    return {n: skills[n] for n in names}


@app.command("list")
def list_() -> None:
    """Show the skills bundled with kiln."""
    for name, folder in _available().items():
        typer.echo(f"/{name}: {_description(folder / 'SKILL.md')}")


@app.command()
def show(name: Annotated[str, typer.Argument(help="Skill name, e.g. label-split.")]) -> None:
    """Print a skill's instructions."""
    folder = _pick([name])[name]
    typer.echo((folder / "SKILL.md").read_text())


@app.command()
def install(
    names: Annotated[Optional[list[str]], typer.Argument(help="Skills to install (default: all).")] = None,
    project: Annotated[Path, typer.Option("--project", "-p", help="Project root; skills go in <project>/.claude/skills/.")] = Path("."),
    user: Annotated[bool, typer.Option("--user", help="Install for every project, into ~/.claude/skills/.")] = False,
    force: Annotated[bool, typer.Option("--force", help="Overwrite skills that are already installed.")] = False,
) -> None:
    """Install skills so Claude Code offers them as slash commands (e.g. /label-split)."""
    dest_root = (Path.home() if user else project).expanduser().resolve() / ".claude" / "skills"
    for name, src in _pick(names).items():
        dest = dest_root / name
        if dest.exists() and not force:
            typer.echo(f"skip /{name}: already at {dest} (use --force to update it)")
            continue
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(src, dest, ignore=shutil.ignore_patterns("__pycache__"))
        typer.echo(f"installed /{name} -> {dest}")
    typer.echo("Restart Claude Code (or start a new session) to pick up new skills.")
