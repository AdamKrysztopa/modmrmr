"""On-demand provisioning for the benchmark datasets that cannot be bundled.

Five scikit-feature microarray ``.mat`` files and two UCI archives are too large
to commit (and not ours to redistribute), so the registry in
:mod:`benchmarks.datasets` points at files under ``benchmarks/data/`` that this
module downloads and materialises on first use.

Provisioning is reproducible, not best-effort. Every source is content-addressed
by SHA-256 and the scikit-feature URLs are pinned to an upstream commit, so a
given dataset name always yields the same bytes or fails loudly — a silently
different upstream revision would otherwise change published benchmark numbers.

Downloads are verbose (per-megabyte progress on stderr) because the largest
archive is ~18 MB and a silent multi-second stall is indistinguishable from a
hang. Provision everything ahead of an offline run with::

    uv run python -m benchmarks.fetch
"""

from __future__ import annotations

import argparse
import hashlib
import io
import sys
import urllib.request
import zipfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from urllib.error import URLError

import pandas as pd

# Pinned so a re-tagged or force-pushed upstream cannot silently change the data
# underneath published results.
_SKFEATURE_COMMIT = "48cffad4e88ff4b9d2f1c7baffb314d1b3303792"
_SKFEATURE_BASE = (
    f"https://raw.githubusercontent.com/jundongl/scikit-feature/{_SKFEATURE_COMMIT}/skfeature/data"
)

_TIMEOUT = 120.0
_CHUNK = 1 << 20
_PROGRESS_STEP = 4 << 20


# --------------------------------------------------------------------------- #
# Materialisers — turn downloaded bytes into the file the loader expects
# --------------------------------------------------------------------------- #
def _write_verbatim(raw: bytes, dest: Path) -> None:
    dest.write_bytes(raw)


def _extract_member(member: str) -> Callable[[bytes, Path], None]:
    def _materialize(raw: bytes, dest: Path) -> None:
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            dest.write_bytes(archive.read(member))

    return _materialize


def _concat_headerless_csvs(raw: bytes, dest: Path) -> None:
    """Flatten BlogFeedback's 61 headerless CSVs into one labelled CSV.

    Upstream ships the data as one train file plus 60 daily test files, none of
    them carrying a header, with the target in the last column. The registry
    counts all of them (n=60021), so concatenate in sorted filename order for a
    deterministic row order and name the columns the loader expects.
    """
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        members = sorted(n for n in archive.namelist() if n.lower().endswith(".csv"))
        frames = [pd.read_csv(io.BytesIO(archive.read(m)), header=None) for m in members]

    df = pd.concat(frames, ignore_index=True)
    df.columns = [f"f{i}" for i in range(df.shape[1] - 1)] + ["target"]
    df.to_csv(dest, index=False)


# --------------------------------------------------------------------------- #
# Source registry
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RemoteFile:
    """One provisionable file: where it comes from and how to unpack it."""

    url: str
    sha256: str
    materialize: Callable[[bytes, Path], None] = field(default=_write_verbatim)


# Keyed by the local filename the loaders in benchmarks.datasets look for. Note
# the scikit-feature archive hyphenates two names that the registry spells with
# underscores, so the key and the URL deliberately differ.
REMOTE_FILES: dict[str, RemoteFile] = {
    "colon.mat": RemoteFile(
        url=f"{_SKFEATURE_BASE}/colon.mat",
        sha256="ffcdeba03eb67cec403fa1dc9f827c22a6e2c57786bf3e01dfe1b4b3e25e0a2f",
    ),
    "ALLAML.mat": RemoteFile(
        url=f"{_SKFEATURE_BASE}/ALLAML.mat",
        sha256="068afe0fe1021932e1e7d801294c267de8aeaa9d41899acfac024e72b0d44c1e",
    ),
    "lymphoma.mat": RemoteFile(
        url=f"{_SKFEATURE_BASE}/lymphoma.mat",
        sha256="bd834ed911d47ecf2e07625ed77617d514d89ee398fa2e7a787230f7cf5243f8",
    ),
    "Prostate_GE.mat": RemoteFile(
        url=f"{_SKFEATURE_BASE}/Prostate-GE.mat",
        sha256="050b598534dbc662f23e30f39b41354b61d02ab8567837bcf9b1091c1c636ab4",
    ),
    "SMK_CAN_187.mat": RemoteFile(
        url=f"{_SKFEATURE_BASE}/SMK-CAN-187.mat",
        sha256="96ff62a9dad001b4bb28a5c31f69e9c41fb1f42d02e4b39c31e4a57b46238aa7",
    ),
    "blogfeedback.csv": RemoteFile(
        url="https://archive.ics.uci.edu/static/public/304/blogfeedback.zip",
        sha256="1ba74e5ad920f7cd037502b2968581cc695146a226a73eff52fe8ad875ed4bcf",
        materialize=_concat_headerless_csvs,
    ),
    "ct_slices.csv": RemoteFile(
        url="https://archive.ics.uci.edu/static/public/206/"
        "relative+location+of+ct+slices+on+axial+axis.zip",
        sha256="9411f92082678169d2a01356f036708498c3ab507f6595919456bbb04a1d942a",
        materialize=_extract_member("slice_localization_data.csv"),
    ),
}


# --------------------------------------------------------------------------- #
# Download
# --------------------------------------------------------------------------- #
def _log(message: str) -> None:
    print(f"[benchmarks.fetch] {message}", file=sys.stderr, flush=True)


def _read_with_progress(response, label: str) -> bytes:
    header = response.headers.get("Content-Length")
    total = int(header) if header and header.isdigit() else 0
    chunks: list[bytes] = []
    got = 0
    next_mark = _PROGRESS_STEP
    while True:
        chunk = response.read(_CHUNK)
        if not chunk:
            break
        chunks.append(chunk)
        got += len(chunk)
        if got >= next_mark:
            pct = f" ({100 * got / total:.0f}%)" if total else ""
            _log(f"{label}: {got / 1e6:.1f} MB{pct}")
            next_mark += _PROGRESS_STEP
    return b"".join(chunks)


def _download(spec: RemoteFile, label: str) -> bytes:
    _log(f"{label}: downloading {spec.url}")
    try:
        with urllib.request.urlopen(spec.url, timeout=_TIMEOUT) as response:  # noqa: S310
            raw = _read_with_progress(response, label)
    except (URLError, OSError) as exc:
        raise RuntimeError(
            f"Could not download {label} from {spec.url} ({exc}). "
            f"Retry with network access, or provision the file manually."
        ) from exc

    digest = hashlib.sha256(raw).hexdigest()
    if digest != spec.sha256:
        raise RuntimeError(
            f"Checksum mismatch for {label} from {spec.url}: "
            f"expected {spec.sha256}, got {digest}. Upstream may have changed; "
            f"do not use this file for published results until it is reconciled."
        )
    return raw


def ensure_dataset_file(filename: str, data_dir: Path) -> Path:
    """Return the path to ``filename``, downloading it into ``data_dir`` if absent.

    Raises ``KeyError`` for a filename with no registered source, and
    ``RuntimeError`` if the download fails or its checksum does not match.
    """
    dest = data_dir / filename
    if dest.exists():
        return dest

    if filename not in REMOTE_FILES:
        raise KeyError(
            f"No download is registered for {filename!r}; expected it at {dest}. "
            f"Known: {sorted(REMOTE_FILES)}"
        )

    spec = REMOTE_FILES[filename]
    raw = _download(spec, filename)

    data_dir.mkdir(parents=True, exist_ok=True)
    # Materialise via a sibling temp file so an interrupted run cannot leave a
    # truncated file that later looks provisioned.
    partial = dest.with_name(dest.name + ".part")
    try:
        spec.materialize(raw, partial)
        partial.replace(dest)
    finally:
        partial.unlink(missing_ok=True)

    _log(f"{filename}: ready at {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
    return dest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m benchmarks.fetch",
        description="Provision the benchmark datasets that are not bundled with the repo.",
    )
    parser.add_argument(
        "files",
        nargs="*",
        choices=sorted(REMOTE_FILES),
        default=[],
        metavar="FILE",
        help="files to provision; default is all of them",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).parent / "data",
        help="destination directory (default: benchmarks/data/)",
    )
    args = parser.parse_args(argv)

    targets = args.files or sorted(REMOTE_FILES)
    _log(f"provisioning {len(targets)} file(s) into {args.data_dir}")
    for i, filename in enumerate(targets, start=1):
        _log(f"[{i}/{len(targets)}] {filename}")
        ensure_dataset_file(filename, args.data_dir)
    _log("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
