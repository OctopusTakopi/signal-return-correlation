"""Path configuration for the signal-return-correlation project."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = Path(os.environ.get("ACB_RESULTS_DIR", ROOT / "results"))
FIGURES_DIR = Path(os.environ.get("ACB_FIGURES_DIR", ROOT / "figures"))


def rel(path) -> str:
    """A path expressed relative to the project root.

    Output files are announced by relative path so that a committed log or JSON
    artefact carries no trace of the machine that produced it.
    """
    path = Path(path)
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return path.name


def ensure_directories() -> None:
    for path in (RESULTS_DIR, FIGURES_DIR):
        path.mkdir(parents=True, exist_ok=True)
