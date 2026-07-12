"""Regression test: no ``mechanism`` runner may default ``--figures-dir`` into
``PAPER_DIR/artifacts`` (see ``analysis.paths``).

``PAPER_DIR/artifacts/`` is the git-tracked directory of curated figures the
paper's ``.tex`` sources actually ``\\includegraphics``/``\\input``. It must
hold exactly the tracked artifacts and nothing else. ``mechanism/run.py`` sets
the precedent every runner should follow: its ``--figures-dir`` defaults to
``results/figures`` (a gitignored, regenerable scratch directory), asserted in
``tests/mechanism/test_run.py``. ``run_factorial.py``, ``run_factorial_fast.py``,
and ``run_mi_comparison.py`` had diverged from that precedent and defaulted
straight into ``PAPER_DIR/artifacts``, so running the exact commands the
paper's appendix documents dumped 25 exploratory PNGs into the tracked
artifact directory.

This lives in its own module, separate from each runner's own
``test_run_*.py``, because it is a cross-cutting property over *every* runner
rather than a fact about any one of them. Modules are discovered by walking
``mechanism``'s ``run*`` submodules and introspecting each ``build_parser()``
for a ``figures_dir`` destination, rather than naming the four current runners
explicitly -- a fifth runner that adds ``--figures-dir`` is picked up
automatically; a runner that doesn't expose the flag (e.g.
``run_baseline_comparison``) is simply skipped, not asserted against.
"""

from __future__ import annotations

import importlib
import pkgutil
from pathlib import Path

import mechanism
from analysis.paths import ARTIFACTS

_PAPER_ARTIFACTS = ARTIFACTS.resolve()


def _run_modules_with_figures_dir_default() -> list[tuple[str, str]]:
    """(module name, --figures-dir default) for every mechanism.run* module that has one."""
    found: list[tuple[str, str]] = []
    for info in pkgutil.iter_modules(mechanism.__path__, prefix="mechanism."):
        leaf = info.name.rsplit(".", 1)[-1]
        if not leaf.startswith("run"):
            continue
        module = importlib.import_module(info.name)
        build_parser = getattr(module, "build_parser", None)
        if build_parser is None:
            continue
        ns = build_parser().parse_args([])
        if hasattr(ns, "figures_dir"):
            found.append((info.name, ns.figures_dir))
    return found


def test_figures_dir_default_never_resolves_inside_paper_artifacts():
    runners = _run_modules_with_figures_dir_default()

    # Guard the discovery itself: if this drops to zero, every assertion below
    # is vacuously true and the test silently stops testing anything.
    assert len(runners) >= 4, (
        f"expected at least 4 mechanism runners exposing --figures-dir, found {runners}; "
        "discovery may be broken"
    )

    for name, default in runners:
        resolved = Path(default).resolve()
        assert resolved != _PAPER_ARTIFACTS and _PAPER_ARTIFACTS not in resolved.parents, (
            f"{name}'s --figures-dir defaults to {default!r}, which resolves inside "
            f"{_PAPER_ARTIFACTS}; running that module with its documented, argument-free "
            "invocation would pollute the git-tracked paper artifact directory"
        )
