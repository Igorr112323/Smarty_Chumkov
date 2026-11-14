import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from build_package import build

ROOT = Path(__file__).resolve().parent.parent


def test_package_is_reproducible_and_matches_record(tmp_path):
    target = build(tmp_path)
    data = json.loads((tmp_path / "package.json").read_text(encoding="utf-8"))
    assert data["under_limit"], data["bytes"]
    record = (ROOT / "package.sha256").read_text(encoding="utf-8").split()[0]
    assert data["sha256"] == record, "пакет в репозитории устарел: пересоберите python tools/build_package.py"
    assert target.exists()
