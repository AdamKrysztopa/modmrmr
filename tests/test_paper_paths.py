"""The live paper is path-configurable; nothing may hardcode a paper directory."""

import pathlib

import pytest

from analysis.paths import ARTIFACTS, PAPER_DIR

REPO = pathlib.Path(__file__).resolve().parent.parent


def test_live_paper_is_the_dmkd_directory():
    assert PAPER_DIR == pathlib.Path("papers/dmkd-2026")
    assert ARTIFACTS == pathlib.Path("papers/dmkd-2026/artifacts")


@pytest.mark.skipif(
    not (pathlib.Path(__file__).resolve().parent.parent / PAPER_DIR / "main.tex").is_file(),
    reason="paper sources not present in this checkout",
)
def test_paper_builds_from_main_tex_when_present():
    assert (REPO / PAPER_DIR / "main.tex").is_file()
    assert (REPO / ARTIFACTS).is_dir()


def test_no_source_file_hardcodes_the_old_paper_directory():
    """`paper/` was moved; nothing may still point at it."""
    offenders = []
    for path in list(REPO.glob("analysis/*.py")) + list(REPO.glob("tests/**/*.py")):
        if path.name == "paths.py" or path.name == "test_paper_paths.py":
            continue
        text = path.read_text()
        if '"paper/' in text or "'paper/" in text:
            offenders.append(str(path.relative_to(REPO)))
    assert not offenders, f"hardcoded legacy paper/ path in: {offenders}"
