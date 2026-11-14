import json
from pathlib import Path

SPLITS_FILE = Path(__file__).with_name("splits.json")


def load_splits(path=None):
    target = Path(path) if path else SPLITS_FILE
    return json.loads(target.read_text(encoding="utf-8"))


def split_ids(name, path=None):
    data = load_splits(path)
    if name not in data:
        raise KeyError(f"нет сплита {name}")
    return list(data[name])


def ordered_pairs(pairs, ids):
    by_id = {}
    for pair in pairs:
        by_id[pair.get("pair_id")] = pair
    return [by_id[pair_id] for pair_id in ids if pair_id in by_id]
