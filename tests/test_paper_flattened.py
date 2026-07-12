"""Anti-drift guard for the Springer single-file submission source.

Springer requires the manuscript as one ``.tex`` document, so ``main-submission.tex``
is generated from the modular ``main.tex`` + ``sections/*.tex`` by
``papers/dmkd-2026/flatten.py``. Nothing in LaTeX detects a stale copy: it compiles
happily to the *previous* revision of the paper, and the human uploads a PDF and a
source file that disagree. This test is the only thing that catches that.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

PAPER_DIR = Path(__file__).resolve().parents[1] / "papers" / "dmkd-2026"
FLATTEN_PY = PAPER_DIR / "flatten.py"
SUBMISSION = PAPER_DIR / "main-submission.tex"

pytestmark = pytest.mark.skipif(
    not FLATTEN_PY.is_file(), reason="paper sources not present in this checkout"
)


def _load_flatten():
    spec = importlib.util.spec_from_file_location("_dmkd_flatten", FLATTEN_PY)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_submission_source_is_not_stale() -> None:
    """Regenerating main-submission.tex must produce no diff."""
    flatten = _load_flatten()
    assert SUBMISSION.is_file(), f"{SUBMISSION} is missing; run: uv run python {FLATTEN_PY}"
    expected = flatten.render()
    actual = SUBMISSION.read_text(encoding="utf-8")
    assert actual == expected, (
        "papers/dmkd-2026/main-submission.tex is stale with respect to main.tex / "
        "sections/. Regenerate it: uv run python papers/dmkd-2026/flatten.py"
    )


def test_submission_source_has_no_input_directives() -> None:
    """The whole point of flattening: Springer forbids \\input in the submitted file."""
    text = SUBMISSION.read_text(encoding="utf-8")
    offenders = [
        line
        for line in text.splitlines()
        if line.lstrip().startswith("\\input{") and not line.lstrip().startswith("%")
    ]
    assert not offenders, f"flattened submission still contains \\input: {offenders}"
