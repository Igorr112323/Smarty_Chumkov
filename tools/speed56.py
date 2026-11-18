"""ПРОМПТ 56, критерий 2.7: скорость на новом входе.

Норма п.7 двусторонняя и проверяется здесь целиком:

  * мс на 1000 знаков документа <= 30;
  * прогон 2 000 000 знаков <= 60 с.

Отличие от `speed52.py`: тот мерил четыре заранее заданные длины и не собирал
сценарий «2 млн знаков за один прогон». Здесь длины берутся из фактического
распределения нового входа (медиана 3980 и хвост 20000), а 2 млн знаков —
отдельный сценарий: реальные пары повторяются до 2 000 000 знаков суммарно,
измеряется всё время прогона, а не нормированная величина.

Веса берутся из `weights/ablation51/ten/fold{N}` (те же 5 сидов на фолд).
Пороги — из манифеста фолда, то есть выбранные на внутренних фолдах.
"""
import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from spanver import lexfeat
from spanver.evaluate import read_jsonl
from spanver.snap import snapped
from spanver.span_detector import choose_chunks
from tools.nested_cv35 import predict
from tools.speed_bench_batched import build_pairs

CHUNKS_TOP = 4
LIMIT_MS_PER_1000 = 30.0
LIMIT_2M_SECONDS = 60.0
TWO_MILLION = 2_000_000


def load_fold(fold, count=5):
    import torch

    from tools.predict_frozen38 import load_state

    folder = ROOT / "weights/ablation51/ten" / f"fold{fold}"
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    entries = manifest["checkpoints"][:count]
    models = []
    for entry in entries:
        tokenizer, model = lexfeat.load_feature_model(ROOT / "models/rubert-tiny2", seed=entry["seed"])
        model.base.config._attn_implementation = "eager"
        load_state(torch, model, folder / f"seed_{entry['seed']}", entry)
        models.append((entry["seed"], tokenizer, model))
    return models, manifest


def merge(caches, pair):
    pieces = [cache[pair["pair_id"]]["pieces"] for cache in caches]
    return [[piece[0], piece[1], sum(item[position][2] for item in pieces) / len(pieces)]
            for position, piece in enumerate(pieces[0])]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", default="artifacts/ours56_docs.jsonl")
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--lengths", default="838,1000,3980,5000,20000")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--out", default="artifacts/speed56.json")
    args = parser.parse_args()

    import torch

    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    gold = read_jsonl(ROOT / args.pairs)
    models, manifest = load_fold(args.fold)
    theta = manifest["threshold"]["start"]
    gate = manifest["gate"]
    lengths = [int(item) for item in args.lengths.split(",")]

    rows = []
    for length in lengths:
        pairs = build_pairs(gold, length)
        chars = sum(len(pair["document_text"]) for pair in pairs)
        repeats = []
        for _ in range(args.repeats):
            started = time.perf_counter()
            caches = [lexfeat.batched_probabilities(tokenizer, model, pairs, top=CHUNKS_TOP, batch=16)
                      for _, tokenizer, model in models]
            for pair in pairs:
                pieces = merge(caches, pair)
                answer = pair["model_output"]
                spans = []
                if max((piece[2] for piece in pieces), default=0.0) >= gate:
                    spans = predict({"answer": answer, "pieces": pieces}, theta)
                snapped(spans, answer, "expand")
            repeats.append((time.perf_counter() - started) * 1000.0 / (chars / 1000.0))
        rows.append({
            "length": length,
            "documents": len(pairs),
            "doc_chars": chars,
            "answer_chars": sum(len(pair["model_output"]) for pair in pairs),
            "ms_per_1000_doc_median": statistics.median(repeats),
            "ms_per_1000_doc_min": min(repeats),
            "ms_per_1000_doc_max": max(repeats),
            "samples": repeats,
            "passes_limit": statistics.median(repeats) <= LIMIT_MS_PER_1000,
        })
        print(json.dumps(rows[-1], ensure_ascii=False), flush=True)

    # сценарий «2 млн знаков за один прогон»: реальные пары по кругу
    two_million = []
    for attempt in range(args.repeats):
        pairs, chars = [], 0
        index = 0
        while chars < TWO_MILLION:
            pair = gold[index % len(gold)]
            pairs.append(pair)
            chars += len(pair["document_text"])
            index += 1
        started = time.perf_counter()
        caches = [lexfeat.batched_probabilities(tokenizer, model, pairs, top=CHUNKS_TOP, batch=16)
                  for _, tokenizer, model in models]
        for position, pair in enumerate(pairs):
            key = pair["pair_id"]
            pieces = merge([cache[key] for cache in caches], pair)
            if max((piece[2] for piece in pieces), default=0.0) >= gate:
                snapped(predict({"answer": pair["model_output"], "pieces": pieces}, theta),
                        pair["model_output"], "expand")
        elapsed = time.perf_counter() - started
        two_million.append({"documents": len(pairs), "doc_chars": chars, "seconds": elapsed})
        print(json.dumps(two_million[-1], ensure_ascii=False), flush=True)

    seconds = statistics.median(item["seconds"] for item in two_million)
    report = {
        "status": "готово",
        "entry": "docs",
        "fold": args.fold,
        "models": len(models),
        "top_chunks": CHUNKS_TOP,
        "theta": theta,
        "gate": gate,
        "threads": 4,
        "torch": str(torch.__version__),
        "platform": platform.platform(),
        "limit_ms_per_1000": LIMIT_MS_PER_1000,
        "limit_2m_seconds": LIMIT_2M_SECONDS,
        "rows": rows,
        "two_million": {
            "runs": two_million,
            "median_seconds": seconds,
            "doc_chars_median": statistics.median(item["doc_chars"] for item in two_million),
            "passes_limit": seconds <= LIMIT_2M_SECONDS,
        },
        "acceptance": {
            "2.7_ms_per_1000_le_30": all(row["passes_limit"] for row in rows),
            "2.7_two_million_le_60s": seconds <= LIMIT_2M_SECONDS,
        },
        "acceptance_all": all(row["passes_limit"] for row in rows) and seconds <= LIMIT_2M_SECONDS,
        "test301": False,
    }
    report["acceptance_all"] = all(report["acceptance"].values())
    (ROOT / args.out).write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(report["acceptance"], ensure_ascii=False))


if __name__ == "__main__":
    main()
