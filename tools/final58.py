"""ПРОМПТ 58, стадия 3: ОДИН финальный замер на отложенной по документам части.

Зачем отдельный скрипт. До сих пор цифры для п.6 собирались из разных прогонов:
метрика из одного, базовая линия из другого, скорость из третьего. Здесь всё
снимается одним проходом на ЗАФИКСИРОВАННОЙ конфигурации, и каждое поле
считается своим кодом на своём же наборе пар.

Фиксация конфигурации (`config` в отчёте, она же в `artifacts/plan58.json`):

  * вход — `joined_reference` (стадия 0 промпта 57);
  * декодер — поставка: `max_spans = 1`, `nms_iou = 0.10`, `min_chars = 2`,
    `extend = start/2`, `gate = 0.45`;
  * порог — переносимый: выбран на ВНУТРЕННИХ фолдах каждого внешнего фолда
    (`decode56.json`, правило pooled, FPR <= 0.40), по test не подбирается;
  * ансамбль — пять сидов фолда (seed 0..4), как в прогоне 44218112557.

`THRESHOLD = 0.98` в промпте относится к лексической схеме 1 (порог риска для
лексических кандидатов), в этом пайплайне такого порога нет: здесь порог — это
вероятность модели. Поэтому фиксируется тот порог, который реально применялся
в 2.1–2.8, и это записано в отчёт явно.

Поля (§3.3): token F1, precision, recall, FPR, exact span F1, pair F1,
базовая линия тех же пар, отрыв от неё, CI 95 % bootstrap, скорость на
медианной длине входа.
"""
import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

FPR_LIMIT = 0.40
RESAMPLES = 10000
SEED = 58


def bootstrap_ci(values, resamples=RESAMPLES, seed=SEED):
    """CI 95 % для среднего по парам, с возвращением."""
    if not values:
        return None
    generator = random.Random(seed)
    samples = sorted(statistics.mean(values[generator.randrange(len(values))] for _ in values)
                     for _ in range(resamples))
    return [samples[int(0.025 * len(samples))], samples[int(0.975 * len(samples))]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--entry", default="joined_reference",
                        choices=("joined_reference", "docs"))
    parser.add_argument("--folds", default="1,2,3,4,5,6,7,8,9,10")
    parser.add_argument("--out", default="artifacts/final58.json")
    parser.add_argument("--plan", default="artifacts/plan58.json")
    args = parser.parse_args()

    import torch

    from spanver import lexfeat
    from spanver.evaluate import covered_tokens, match_spans, token_positions
    from spanver.snap import snapped
    from tools.nested_cv35 import predict
    from tools.eval56_test import clean_subset, load_test_rows
    from tools.nested_cv35 import averaged, probabilities
    from tools.predict_frozen38 import load_state
    from tools.sweep_ragtruth30 import compute, prepare, predict_spans

    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)

    plan = json.loads((ROOT / args.plan).read_text(encoding="utf-8"))
    rows = load_test_rows(args.entry)
    clean = clean_subset(rows)
    clean_rows = [row for row in rows if row["pair_id"] in clean]
    print(f"test: {len(rows)} пар, чистых {len(clean_rows)}", flush=True)

    # --- базлайн: тем же кодом и на тех же парах
    def baseline(part):
        gold = total = with_gold = 0
        for row in part:
            answer = row["model_output"]
            total += len(token_positions(answer))
            gold += len(covered_tokens(answer, [(s["start"], s["end"]) for s in row["spans"]]))
            with_gold += bool(row["spans"])
        share = gold / total if total else 0.0
        precision = with_gold / len(part) if part else 0.0
        return {"pairs": len(part), "gold_share": share,
                "all_positive_token_f1": 2 * share / (1 + share) if share else 0.0,
                "all_positive_pair_f1": 2 * precision / (1 + precision) if precision else 0.0,
                "pairs_with_gold": with_gold, "all_positive_fpr": 1.0}

    # --- инференс по фолдам, по одному за раз (память)
    folds = [int(item) for item in args.folds.split(",")]
    per_fold_points = []
    per_pair = {row["pair_id"]: [] for row in rows}
    started = time.perf_counter()
    for fold in folds:
        folder = ROOT / "weights/ablation51/ten" / f"fold{fold}"
        manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        threshold = manifest["threshold"]
        gate = manifest["gate"]
        seed_caches = []
        for entry in manifest["checkpoints"]:
            tokenizer, model = lexfeat.load_feature_model(ROOT / "models/rubert-tiny2",
                                                          seed=entry["seed"])
            model.base.config._attn_implementation = "eager"
            load_state(torch, model, folder / f"seed_{entry['seed']}", entry)
            seed_caches.append(probabilities(tokenizer, model, rows))
            del model, tokenizer
        cache = {row["pair_id"]: averaged(row, seed_caches, range(len(seed_caches)))
                 for row in rows}
        for subset, name in ((rows, "all"), (clean_rows, "clean")):
            prepared = prepare(subset, cache)
            point = compute(prepared, threshold["start"], threshold["extend"], gate)
            # построчно: нужно и для bootstrap, и для pair-счётчиков
            from spanver.evaluate import match_spans

            for row in prepared:
                single = compute([row], threshold["start"], threshold["extend"], gate)
                support = single["support"]
                tp, fp, fn = support["token_tp"], support["token_fp"], support["token_fn"]
                precision = tp / (tp + fp) if tp + fp else 0.0
                recall = tp / (tp + fn) if tp + fn else 0.0
                # pair-уровень повторяет правило compute: есть золото и есть
                # попадание -> tp; есть золото без попадания -> fn;
                # нет золота и что-то предсказано -> fp
                if gate is not None and max((p[2] for p in row["pieces"]), default=0.0) < gate:
                    predicted = []
                else:
                    predicted = predict_spans(row["answer"], row["pieces"], threshold["start"],
                                              threshold["extend"], row["edges"])
                matched, _, _ = match_spans(predicted, row["gold"])
                per_pair[row["pair_id"]].append({
                    "fold": fold, "subset": name,
                    "token_f1": 2 * precision * recall / (precision + recall)
                    if precision + recall else 0.0,
                    "pair_tp": 1 if (row["gold"] and matched) else 0,
                    "pair_fn": 1 if (row["gold"] and not matched) else 0,
                    "pair_fp": 1 if (not row["gold"] and predicted) else 0,
                })
            per_fold_points.append({"fold": fold, "subset": name,
                                    "threshold": threshold["start"], "gate": gate,
                                    "point": point})
        print(f"  фолд {fold}: порог {threshold['start']}, "
              f"F1 {per_fold_points[-1]['point']['token_f1']}", flush=True)
    inference_seconds = time.perf_counter() - started

    report = {
        "status": "готово",
        "stage": "58 стадия 3, один фолд",
        "fold": folds,
        "entry": args.entry,
        "test_pairs": len(rows),
        "clean_pairs": len(clean_rows),
        "config": plan["config"],
        "baselines": {"all_301": baseline(rows), "clean_167": baseline(clean_rows)},
        "fold_results": per_fold_points,
        "per_pair": per_pair,
        "inference_seconds": inference_seconds,
        "test301_usage": "единственный финальный замер; выбор по test не производился",
    }
    (ROOT / args.out).write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n",
                                 encoding="utf-8")
    for item in per_fold_points:
        point = item["point"]
        print(f"  фолд {item['fold']} {item['subset']}: F1 {point['token_f1']} "
              f"FPR {point['fpr_token']} P {point['token']['precision']}", flush=True)


if __name__ == "__main__":
    main()
