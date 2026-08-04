"""Provisioning tests for benchmarks.fetch.

These run offline: `ensure_dataset_file` goes through `urllib`, which serves
`file://` URLs, so the download path, checksum gate and atomic write are all
exercised end to end against fixtures instead of being mocked out.
"""

import hashlib
import io
import zipfile

import pandas as pd
import pytest

from benchmarks import fetch
from benchmarks.datasets import DATASETS, local_filename
from benchmarks.fetch import REMOTE_FILES, RemoteFile, ensure_dataset_file


def _file_url(path) -> str:
    return path.resolve().as_uri()


@pytest.fixture
def source(tmp_path):
    """A local file standing in for an upstream download."""
    payload = b"upstream-bytes"
    src = tmp_path / "upstream.bin"
    src.write_bytes(payload)
    return src, payload, hashlib.sha256(payload).hexdigest()


def test_downloads_verifies_and_caches(tmp_path, monkeypatch, source):
    src, payload, digest = source
    data_dir = tmp_path / "data"
    monkeypatch.setitem(REMOTE_FILES, "thing.bin", RemoteFile(_file_url(src), digest))

    path = ensure_dataset_file("thing.bin", data_dir)

    assert path == data_dir / "thing.bin"
    assert path.read_bytes() == payload
    # No temp file survives a successful run.
    assert list(data_dir.iterdir()) == [path]


def test_existing_file_is_not_redownloaded(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "thing.bin").write_bytes(b"already here")
    # An unreachable source: reaching the network at all would fail the test.
    monkeypatch.setitem(REMOTE_FILES, "thing.bin", RemoteFile("file:///nope", "0" * 64))

    assert ensure_dataset_file("thing.bin", data_dir).read_bytes() == b"already here"


def test_checksum_mismatch_refuses_and_leaves_nothing_behind(tmp_path, monkeypatch, source):
    src, _, _ = source
    data_dir = tmp_path / "data"
    monkeypatch.setitem(REMOTE_FILES, "thing.bin", RemoteFile(_file_url(src), "0" * 64))

    with pytest.raises(RuntimeError, match="Checksum mismatch"):
        ensure_dataset_file("thing.bin", data_dir)

    assert not (data_dir / "thing.bin").exists()


def test_unreachable_source_reports_actionably(tmp_path, monkeypatch):
    missing = tmp_path / "absent.bin"
    monkeypatch.setitem(REMOTE_FILES, "thing.bin", RemoteFile(_file_url(missing), "0" * 64))

    with pytest.raises(RuntimeError, match="Could not download thing.bin"):
        ensure_dataset_file("thing.bin", tmp_path / "data")


def test_unregistered_filename_raises_keyerror(tmp_path):
    with pytest.raises(KeyError, match="No download is registered"):
        ensure_dataset_file("not_a_dataset.csv", tmp_path)


# --------------------------------------------------------------------------- #
# Materialisers
# --------------------------------------------------------------------------- #
def _zip_bytes(members: dict[str, str]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as archive:
        for name, body in members.items():
            archive.writestr(name, body)
    return buf.getvalue()


def test_concat_headerless_csvs_labels_columns_and_preserves_sorted_order(tmp_path):
    # Two headerless shards, three columns, target last — as BlogFeedback ships.
    raw = _zip_bytes(
        {
            "blogData_test-b.csv": "3,4,30\n",
            "blogData_test-a.csv": "1,2,10\n5,6,50\n",
            "readme.txt": "ignored",
        }
    )
    dest = tmp_path / "blogfeedback.csv"

    fetch._concat_headerless_csvs(raw, dest)
    df = pd.read_csv(dest)

    assert list(df.columns) == ["f0", "f1", "target"]
    assert df.shape == (3, 3)
    # Sorted filename order: the "-a" shard's rows precede the "-b" shard's.
    assert df["target"].tolist() == [10, 50, 30]


def test_extract_member_pulls_the_named_csv(tmp_path):
    raw = _zip_bytes({"other.csv": "nope\n", "wanted.csv": "reference,value0\n1.5,2.5\n"})
    dest = tmp_path / "ct_slices.csv"

    fetch._extract_member("wanted.csv")(raw, dest)

    assert pd.read_csv(dest).to_dict("records") == [{"reference": 1.5, "value0": 2.5}]


# --------------------------------------------------------------------------- #
# Drift guard
# --------------------------------------------------------------------------- #
def test_every_file_backed_dataset_has_a_registered_download():
    """The bug this prevents: a registry entry naming a file nothing can supply.

    `colon` and `blog_feedback` shipped for months pointing at files that had no
    download path, so they could only ever raise. Any new file-backed dataset
    must come with a source.
    """
    file_backed = {name: local_filename(name) for name in DATASETS}
    file_backed = {n: f for n, f in file_backed.items() if f is not None}

    assert file_backed, "expected some datasets to be file-backed"
    unsourced = {n: f for n, f in file_backed.items() if f not in REMOTE_FILES}
    assert not unsourced, f"file-backed datasets with no download source: {unsourced}"


def test_every_registered_download_is_used_by_a_dataset():
    referenced = {local_filename(name) for name in DATASETS} - {None}
    assert set(REMOTE_FILES) == referenced


@pytest.mark.network
def test_pinned_source_still_matches_upstream(tmp_path):
    """Real download of the smallest registered source (~36 KB).

    Proves the pinned scikit-feature commit URL still resolves and its bytes
    still hash to what `REMOTE_FILES` claims. Deliberately only this one: the
    other six total ~37 MB, and pulling them nightly would trade real evidence
    for provider flakiness.
    """
    path = ensure_dataset_file("colon.mat", tmp_path)

    assert path.exists()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == REMOTE_FILES["colon.mat"].sha256


def test_remote_file_specs_are_well_formed():
    for filename, spec in REMOTE_FILES.items():
        assert spec.url.startswith("https://"), filename
        assert len(spec.sha256) == 64, filename
        assert callable(spec.materialize), filename
