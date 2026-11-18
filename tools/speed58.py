"""ПРОМПТ 58, стадия 3.3: скорость на медианной длине входа.

Норма п.7 двусторонняя, поэтому меряются обе стороны:
  * мс на 1000 знаков документа на медианной длине входа (`joined_reference` —
    838 знаков, это медиана, а не выбранная длина);
  * прогон 2 000 000 знаков за один проход.

Два режима, потому что они отвечают на разные вопросы:
  * `models=1` — тот, на котором норма понималась в `speed52` и который
    сопоставим с прежними замерами;
  * `models=5` — ансамбль, которым получены метрики 2.1–2.8 и стадии 3.
    В приёмку он не переводится: это была бы подмена нормы.
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

from spanver.evaluate import read_jsonl
from spanver.snap import snapped
from tools.nested_cv35 import predict
from tools.speed56 import CHUNKS_TOP, LIMIT_2M_SECONDS, LIMIT_MS_PER_1000, TWO_MILLION, load_fold

from spanver import lexfeat


def run_pass(models, pairs, theta, gate):
    caches = [lexfeat.batched_probabilities(tokenizer, model, pairs, top=CHUNKS_TOP, batch=16)
              for _, tokenizer, model in models]
    missing = [pair["pair_id"] for pair in pairs
               if any(pair["pair_id"] not in cache for cache in caches)]
    if missing:
        raise SystemExit(f"нет вероятностей для {len(missing)} пар, первая {missing[0]}")
    for pair in pairs:
        pieces = [[piece[0], piece[1], sum(cache[pair["pair_id"]]["pieces"][position][2]
                                           for cache in caches) / len(caches)]
                  for position, piece in enumerate(caches[0][pair["pair_id"]]["pieces"])]
        answer = pair["model_output"]
        spans = []
        if max((piece[2] for piece in pieces), default=0.0) >= gate:
            spans = predict({"answer": answer, "pieces": pieces}, theta)
        snapped(spans, answer, "expand")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", default="artifacts/ours33.jsonl")
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--counts", default="1,5")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--out", default="artifacts/final58_speed.json")
    args = parser.parse_args()

    import torch

    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    pairs = read_jsonl(ROOT / args.pairs)
    counts = [int(item) for item in args.counts.split(",")]
    models, manifest = load_fold(args.fold, max(counts))
    theta = manifest["threshold"]["start"]
    gate = manifest["gate"]
    lengths = sorted(len(pair["document_text"]) for pair in pairs)
    median_length = lengths[len(lengths) // 2]
    print(f"медиана длины документа: {median_length} знаков", flush=True)

    rows = []
    for count in counts:
        repeats = []
        for _ in range(args.repeats):
            started = time.perf_counter()
            run_pass(models[:count], pairs, theta, gate)
            chars = sum(len(pair["document_text"]) for pair in pairs)
            repeats.append((time.perf_counter() - started) * 1000.0 / (chars / 1000.0))
        rows.append({
            "models": count,
            "documents": len(pairs),
            "doc_chars": sum(len(pair["document_text"]) for pair in pairs),
            "median_doc_length": median_length,
            "ms_per_1000_doc_median": statistics.median(repeats),
            "ms_per_1000_doc_min": min(repeats),
            "ms_per_1000_doc_max": max(repeats),
            "samples": repeats,
            "passes_limit": statistics.median(repeats) <= LIMIT_MS_PER_1000,
        })
        print(json.dumps(rows[-1], ensure_ascii=False), flush=True)

    two_million = []
    usable = [pair for pair in pairs if pair.get("model_output") and pair.get("document_text")]
    for count in counts:
        runs = []
        for _ in range(args.repeats):
            batch, chars, index = [], 0, 0
            while chars < TWO_MILLION:
                base = usable[index % len(usable)]
                batch.append({"pair_id": str(index),
                              "model_output": base["model_output"],
                              "document_text": base["document_text"]})
                chars += len(base["document_text"])
                index += 1
            started = time.perf_counter()
            run_pass(models[:count], batch, theta, gate)
            runs.append({"documents": len(batch), "doc_chars": chars,
                         "seconds": time.perf_counter() - started})
        median = statistics.median(item["seconds"] for item in runs)
        two_million.append({"models": count, "runs": runs, "median_seconds": median,
                            "doc_chars": runs[0]["doc_chars"],
                            "passes_limit": median <= LIMIT_2M_SECONDS})
        print(json.dumps({k: v for k, v in two_million[-1].items() if k != "runs"},
                         ensure_ascii=False), flush=True)

    single = [row for row in rows if row["models"] == 1]
    single_two = [item for item in two_million if item["models"] == 1]
    report = {
        "status": "готово",
        "stage": "58 стадия 3",
        "entry": "joined_reference",
        "median_doc_length": median_length,
        "theta": theta,
        "gate": gate,
        "torch": str(torch.__version__),
        "platform": platform.platform(),
        "rows": rows,
        "two_million": two_million,
        "acceptance": {
            "2.7_ms_per_1000_le_30": bool(single) and all(row["passes_limit"] for row in single),
            "2.7_two_million_le_60s": bool(single_two) and single_two[0]["passes_limit"],
        },
        "note": ("приёмка по одному сиду, как норма понималась в speed52; "
                 "models=5 — цена ансамбля, которым получены метрики"),
        "test301": False,
    }
    report["acceptance_all"] = all(report["acceptance"].values())
    (ROOT / args.out).write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n",
                                 encoding="utf-8")
    print(json.dumps(report["acceptance"], ensure_ascii=False))


if __name__ == "__main__":
    main()
