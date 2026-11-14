import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from assemble_weights import AssembleError, assemble
from split_weights import CHUNK_BYTES, split

TOTAL = 200 * 1024 * 1024


@pytest.fixture(scope="module")
def sample(tmp_path_factory):
    folder = tmp_path_factory.mktemp("weights")
    source = folder / "pytorch_model.bin"
    block = bytes(range(256)) * 4096
    with open(source, "wb") as handle:
        written = 0
        while written < TOTAL:
            piece = block[: min(len(block), TOTAL - written)]
            handle.write(piece)
            written += len(piece)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    return {"path": source, "sha256": digest, "folder": folder}


def test_split_makes_expected_chunks(sample):
    out = sample["folder"] / "parts"
    manifest = split(sample["path"], out, quiet=True)
    expected = (TOTAL + CHUNK_BYTES - 1) // CHUNK_BYTES
    assert manifest["chunk_count"] == expected
    assert manifest["size"] == TOTAL
    assert manifest["full_sha256"] == sample["sha256"]
    assert len(manifest["chunk_sha256"]) == expected
    for index in range(expected):
        assert (out / f"part-{index:03d}.bin").is_file()


def test_manifest_round_trip(sample):
    out = sample["folder"] / "parts"
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["source"] == "pytorch_model.bin"
    assert manifest["chunk_bytes"] == CHUNK_BYTES
    assert sum(entry["bytes"] for entry in manifest["chunks"]) == manifest["size"]


def test_assemble_restores_bytes(sample):
    out = sample["folder"] / "parts"
    destination = sample["folder"] / "restored.bin"
    result = assemble(out, destination)
    assert result["bytes"] == TOTAL
    assert result["sha256"] == sample["sha256"]
    assert hashlib.sha256(destination.read_bytes()).hexdigest() == sample["sha256"]


def test_corrupted_chunk_rejected(sample, tmp_path):
    source_parts = sample["folder"] / "parts"
    broken = tmp_path / "broken"
    broken.mkdir()
    for path in source_parts.iterdir():
        (broken / path.name).write_bytes(path.read_bytes())
    victim = broken / "part-000.bin"
    data = bytearray(victim.read_bytes())
    data[100] ^= 0xFF
    victim.write_bytes(bytes(data))
    destination = tmp_path / "should_not_exist.bin"
    with pytest.raises(AssembleError):
        assemble(broken, destination)
    assert not destination.exists()


def test_missing_manifest_rejected(tmp_path):
    with pytest.raises(AssembleError):
        assemble(tmp_path, tmp_path / "out.bin")


def test_official_hash_mismatch_rejected(sample, tmp_path):
    out = sample["folder"] / "parts"
    with pytest.raises(AssembleError):
        assemble(out, tmp_path / "out.bin", official="0" * 64)
