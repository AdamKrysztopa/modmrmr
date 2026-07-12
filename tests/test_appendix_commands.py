"""Every reproduction command in the appendix must parse against its CLI.

The paper's appendix lists one command per study. A command that names a flag
its module does not accept -- or a study whose command is missing entirely --
is a reproducibility claim the repository cannot honour.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from analysis.paths import PAPER_DIR

APPENDIX = PAPER_DIR / "sections" / "99-appendix.tex"

pytestmark = pytest.mark.skipif(
    not APPENDIX.is_file(), reason="paper sources not present in this checkout"
)


def _commands() -> list[list[str]]:
    """Extract `uv run python -m ...` invocations, joining backslash continuations."""
    if not APPENDIX.is_file():
        return []
    text = APPENDIX.read_text().replace("\\\n", " ")
    out = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("uv run python -m "):
            out.append(line.split())
    return out


def test_appendix_lists_the_effect_sizes_command() -> None:
    """The omega-squared decomposition certifies Rule 1; it must be reproducible."""
    modules = {cmd[4] for cmd in _commands()}
    assert "analysis.effect_sizes" in modules


@pytest.mark.parametrize("cmd", _commands(), ids=lambda c: c[4])
def test_command_flags_are_accepted_by_its_module(cmd: list[str]) -> None:
    """Every --flag used must appear in the module's --help output."""
    module = cmd[4]
    help_text = subprocess.run(
        [sys.executable, "-m", module, "--help"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    for token in cmd[5:]:
        if token.startswith("--"):
            assert token in help_text, f"{module} does not accept {token}"


@pytest.mark.parametrize("cmd", _commands(), ids=lambda c: c[4])
def test_help_short_circuits_before_any_work(cmd: list[str]) -> None:
    """``--help`` must exit 0 and print a usage line -- never run the module's job.

    A module with an argparse CLI prints a line starting with "usage:" and exits
    immediately. A module that instead calls straight into ``main()`` from a bare
    ``if __name__ == "__main__":`` (no argparse) ignores ``--help`` entirely and
    runs its full job instead, printing its own progress output with no "usage:"
    line. That was exactly the bug: three appendix modules (``analysis.paper_figures``,
    ``analysis.paper_tables``, ``analysis.hardening_outputs``) had no argparse CLI, so
    ``test_command_flags_are_accepted_by_its_module`` above -- which only inspects
    stdout for flag substrings -- passed anyway even though ``--help`` had regenerated
    every figure/table and rewritten git-tracked files under the paper's
    ``artifacts/`` directory (``PAPER_DIR/artifacts``, see ``analysis.paths``).
    This test would have caught that: it fails on any module whose ``--help`` does
    not look like argparse's own output.
    """
    module = cmd[4]
    result = subprocess.run(
        [sys.executable, "-m", module, "--help"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, (
        f"{module} --help exited {result.returncode} (expected 0); it may be running "
        "its full job instead of short-circuiting"
    )
    assert "usage:" in result.stdout.lower(), (
        f"{module} --help printed no usage line -- it likely lacks an argparse CLI "
        "and ran its full job instead of exiting immediately"
    )
