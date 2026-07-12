"""Canonical paper paths. Never hardcode a paper directory anywhere else.

The live paper moves when the venue does. Everything that writes a figure, a
table, or a section reads PAPER_DIR from here, so retargeting a new venue is a
one-line change (or a PAPER_DIR env override).
"""

from __future__ import annotations

import os
from pathlib import Path

PAPER_DIR = Path(os.environ.get("PAPER_DIR", "papers/dmkd-2026"))
"""The live paper. Figures, tables and sections are written here."""

ARTIFACTS = PAPER_DIR / "artifacts"
"""Git-tracked figures/tables the paper \\includegraphics. Never write exploratory output here."""
