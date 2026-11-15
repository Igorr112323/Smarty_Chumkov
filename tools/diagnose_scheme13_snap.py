import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from spanver import decode, lexfeat
from spanver.evaluate import metrics
from spanver.snap import snapped
from tools.train_scheme13_a1 import read_pairs
from tools.train_scheme13_full import load_state


def main():
    import torch

    started = time.perf_counter()
    _, held = read_pairs()
    pairs = random.Random(28).sample(held, 50)
    tokenizer, model = lexfeat.load_feature_model(ROOT / "models/rubert-tiny2")
    model.base.config._attn_implementation = "eager"
    torch.set_num_threads(2)
    folder = ROOT / "artifacts/checkpoints/stage_a_full_e2"
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    state = load_state(folder, manifest["training_state_parts"], torch)
    model.load_state_dict(state["model"])
    del state
    model.eval()
    cache = lexfeat.batched_probabilities(tokenizer, model, pairs, top=4, batch=4)
    predictions = {}
    unique_predictions = {}
    raw_total = expanded_total = unique_total = 0
    examples = []
    for pair in pairs:
        value = cache.get(pair["pair_id"])
        if not value or max((item[2] for item in value["pieces"]), default=0.0) < 0.45:
            predictions[pair["pair_id"]] = []
            unique_predictions[pair["pair_id"]] = []
            continue
        flags = decode.flags_for(value["pieces"], 0.25, None)
        raw = decode.spans_from_flags(value["answer"], value["pieces"], flags, 2)
        expanded = snapped(raw, value["answer"], "expand")
        seen = set()
        unique = []
        for span in expanded:
            key = (span["start"], span["end"])
            if key not in seen:
                unique.append(span)
                seen.add(key)
        predictions[pair["pair_id"]] = expanded
        unique_predictions[pair["pair_id"]] = unique
        raw_total += len(raw)
        expanded_total += len(expanded)
        unique_total += len(unique)
        if len(expanded) - len(unique) > 2 and len(examples) < 5:
            examples.append({"pair_id": pair["pair_id"], "raw": len(raw), "expanded": len(expanded), "unique_expanded": len(unique), "example_expanded_bounds": [[item["start"], item["end"]] for item in expanded[:8]]})
    before = metrics(pairs, lambda pair: predictions[pair["pair_id"]])
    after = metrics(pairs, lambda pair: unique_predictions[pair["pair_id"]])
    report = {"pairs": len(pairs), "raw_spans": raw_total, "snapped_spans": expanded_total, "unique_snapped_spans": unique_total, "duplicate_snapped_spans": expanded_total - unique_total, "examples": examples, "raw_metrics_on_50_with_corrected_evaluator": before, "deduplicated_metrics_on_50_with_corrected_evaluator": after, "note": "checkpoint e2; RAGTruth held-out, случайные 50 пар; пороги не менялись; удаление дублей исследовано только для диагностики", "seconds": round(time.perf_counter() - started, 3)}
    (ROOT / "artifacts/diagnose_snap_28.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
