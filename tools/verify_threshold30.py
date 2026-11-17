import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from spanver import decode
from spanver.evaluate import metrics
from spanver.snap import snapped
from tools.sweep_hyst import flags_with
from tools.sweep_ragtruth30 import compute, prepare
from tools.train_cv_feat import token_fpr
from tools.train_scheme13_a1 import read_pairs


def main():
    started = time.perf_counter()
    _, heldout = read_pairs()
    pairs = random.Random(30).sample(heldout, 50)
    ids = {pair["pair_id"] for pair in pairs}
    checks = []
    for model in ("bce", "focal"):
        cache = {}
        with (ROOT / f"artifacts/threshold30_{model}_scores.jsonl").open(encoding="utf-8") as handle:
            next(handle)
            for line in handle:
                item = json.loads(line)
                if item["pair_id"] in ids:
                    cache[item["pair_id"]] = item
        if len(cache) != len(pairs):
            raise ValueError("часть пар отсутствует в кэше")
        prepared = prepare(pairs, cache)
        for threshold in (0.25, 0.50, 0.75, 0.85):
            for hysteresis in (False, True):
                extend = threshold / 2 if hysteresis else threshold
                def predict(pair):
                    value = cache[pair["pair_id"]]
                    flags = flags_with(value["pieces"], threshold, extend, 0.0)
                    return snapped(decode.spans_from_flags(value["answer"], value["pieces"], flags, 2), value["answer"], "expand")
                original = metrics(pairs, predict)
                fast = compute(prepared, threshold, extend)
                for label, name in (("token", "token_level"), ("span", "span_level"), ("pair", "pair_level")):
                    for metric in ("precision", "recall", "f1"):
                        if fast[label][metric] != original[name][metric]:
                            raise ValueError(f"{model} {threshold}: {label} {metric} не совпадает")
                if fast["fpr_pair"] != original["fpr"] or fast["fpr_token"] != token_fpr(pairs, predict):
                    raise ValueError(f"{model} {threshold}: FPR не совпадает")
                checks.append({"model": model, "start": threshold, "extend": extend, "matches": True})
    report = {"pairs": len(pairs), "seed": 30, "comparisons": checks, "seconds": round(time.perf_counter() - started, 3)}
    (ROOT / "artifacts/threshold30_verification.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
