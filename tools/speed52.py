import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from spanver import decode, lexfeat
from spanver.evaluate import read_jsonl
from spanver.snap import snapped
from spanver.span_detector import choose_chunks
from tools.nested_cv35 import predict
from tools.speed_bench_batched import build_pairs

# Порог и gate берутся из манифеста весов, а не из констант, чтобы замер
# соответствовал той конфигурации, которая заявлена в отчёте.
MIN_CHARS = 2
CHUNKS_TOP = 4  # совпадает с tools/nested_cv35.probabilities(top=4)


def load_ensemble(count):
    import torch

    manifest = json.loads((ROOT / "weights/final/manifest.json").read_text(encoding="utf-8"))
    if manifest["variant"] != "v5" or not manifest["restore_verified"]:
        raise ValueError("ожидается проверенный ансамбль V5")
    entries = manifest["checkpoints"][:count]
    if len(entries) != count:
        raise ValueError("в манифесте меньше сидов, чем запрошено")
    from tools.predict_frozen38 import load_state

    models = []
    for entry in entries:
        tokenizer, model = lexfeat.load_feature_model(ROOT / "models/rubert-tiny2", seed=entry["seed"])
        model.base.config._attn_implementation = "eager"
        load_state(torch, model, ROOT / "weights/final" / f"seed_{entry['seed']}", entry)
        models.append((entry["seed"], tokenizer, model))
    return models, manifest


def merge(caches, pair):
    pieces = [caches[index][pair["pair_id"]]["pieces"] for index in range(len(caches))]
    # тот же порядок смещений, что в tools/nested_cv35.averaged
    return [[piece[0], piece[1], sum(item[position][2] for item in pieces) / len(pieces)]
            for position, piece in enumerate(pieces[0])]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lengths", default="838,1000,5000,20000")
    parser.add_argument("--seeds", default="1,3,5")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--out", default="artifacts/speed52.json")
    args = parser.parse_args()

    import torch
    torch.set_num_threads(4)
    lengths = [int(x) for x in args.lengths.split(",")]
    counts = [int(x) for x in args.seeds.split(",")]
    gold = read_jsonl(ROOT / "artifacts/ours33.jsonl")
    models, manifest = load_ensemble(max(counts))
    theta = manifest["threshold"]["start"]
    gate = manifest["gate"]
    rows = []
    for count in counts:
        for length in lengths:
            pairs = build_pairs(gold, length)
            chars = sum(len(p["document_text"]) for p in pairs)
            forwards = sum(len(choose_chunks(tokenizer, pair, top=CHUNKS_TOP)) for _, tokenizer, _ in models[:count] for pair in pairs)
            repeats = []
            for _ in range(args.repeats):
                started = time.perf_counter()
                caches = [lexfeat.batched_probabilities(tokenizer, model, pairs, top=CHUNKS_TOP, batch=16)
                          for _, tokenizer, model in models[:count]]
                for pair in pairs:
                    if any(pair["pair_id"] not in cache or not cache[pair["pair_id"]]["pieces"] for cache in caches):
                        continue
                    pieces = merge(caches, pair)
                    answer = pair["model_output"]
                    spans = []
                    if max((piece[2] for piece in pieces), default=0.0) >= gate:
                        spans = predict({"answer": answer, "pieces": pieces}, theta)
                    snapped(spans, answer, "expand")
                repeats.append((time.perf_counter() - started) * 1000.0 / (chars / 1000.0))
            rows.append({
                "models": count,
                "length": length,
                "documents": len(pairs),
                "doc_chars": chars,
                "answer_chars": sum(len(p["model_output"]) for p in pairs),
                "forward_passes": forwards,
                "forwards_per_pair": forwards / len(pairs),
                "ms_per_1000_doc_median": statistics.median(repeats),
                "ms_per_1000_doc_min": min(repeats),
                "ms_per_1000_doc_max": max(repeats),
                "samples": repeats,
            })
            print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
    report = {
        "status": "готово: измерение обученного ансамбля V5, включает признаки, все forward, усреднение, gate, гистерезис и клаузный expand",
        "threads": 4,
        "batch": 16,
        "top_chunks": CHUNKS_TOP,
        "theta": theta,
        "gate": gate,
        "torch": str(torch.__version__),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": __import__("os").cpu_count(),
        "limit_ms_per_1000": 30,
        "resolution": "ms_per_1000_doc = всё время / (сумма знаков документа / 1000); это верхняя оценка накладных расходов, потому что ответы не режутся",
        "rows": rows,
    }
    (ROOT / args.out).write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"отчёт: {args.out}", flush=True)


if __name__ == "__main__":
    main()
