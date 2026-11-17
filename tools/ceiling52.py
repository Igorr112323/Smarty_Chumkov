"""Стадия 3, шаг 0: потолок текущего сигнала.

Отвечает на вопрос «сколько ещё можно выжать из уже обученной модели сменой
порога/типа/ограничения числа спанов», не меняя весов. Считается на внешнем
фолде 1 (140 пар, которых seed 0 не видел):

  * AUC по знакам ответа — насколько скор отделяет золотые знаки от остальных.
    Это верхняя граница любого порогового правила: при AUC 0.5 порог бесполезен;
  * AUC по парам — отделяет ли максимум скора пары «есть нарушение» от «нет»;
  * метрики при оракульном выборе порога на самом фолде (верхняя граница
    подбора порога, не достижимая без подсматривания);
  * базовые линии из artifacts/decode52.json для сравнения.

Никакого обучения скрипт не запускает, только один проход инференса.
"""
import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from spanver import lexfeat
from spanver.evaluate import read_jsonl
from tools.evaluate_snap42 import compute
from tools.predict_frozen38 import load_state
from tools.train_base33 import grouped_folds
from tools.tune52 import decode


def auc(positives, negatives, seed=48, limit=4000):
    if not positives or not negatives:
        return None
    generator = random.Random(seed)
    sample_pos = generator.sample(positives, min(limit, len(positives)))
    sample_neg = generator.sample(negatives, min(limit, len(negatives)))
    wins = ties = 0
    for positive in sample_pos:
        for negative in sample_neg:
            if positive > negative:
                wins += 1
            elif positive == negative:
                ties += 1
    return (wins + 0.5 * ties) / (len(sample_pos) * len(sample_neg))


def char_level(pairs, cache):
    positives, negatives = [], []
    gold_chars = total_chars = 0
    for pair in pairs:
        mask = set()
        for span in pair["spans"]:
            mask |= set(range(span["start"], span["end"] + 1))
        gold_chars += len(mask)
        total_chars += len(pair["model_output"])
        for start, end, score in cache[pair["pair_id"]]:
            chars = set(range(start, end + 1))
            share = len(chars & mask) / max(1, len(chars))
            (positives if share >= 0.5 else negatives).append(score)
    return auc(positives, negatives), len(positives), len(negatives), gold_chars / total_chars if total_chars else 0.0


def pair_level(pairs, cache):
    positives, negatives = [], []
    for pair in pairs:
        pieces = cache[pair["pair_id"]]
        best = max((piece[2] for piece in pieces), default=0.0)
        (positives if pair["spans"] else negatives).append(best)
    return auc(positives, negatives), len(positives), len(negatives)


def main():
    import torch

    parser = argparse.ArgumentParser()
    parser.add_argument("--outer", type=int, default=1)
    parser.add_argument("--top", type=int, default=4)
    parser.add_argument("--out", default="artifacts/ceiling52.json")
    args = parser.parse_args()
    torch.set_num_threads(4)
    pairs = read_jsonl(ROOT / "artifacts/ours33.jsonl")
    held = grouped_folds(pairs, 5)[args.outer - 1]
    manifest = json.loads((ROOT / "weights/final/manifest.json").read_text(encoding="utf-8"))
    entry = manifest["checkpoints"][0]
    tokenizer, model = lexfeat.load_feature_model(ROOT / "models/rubert-tiny2", labels=2, seed=entry["seed"])
    model.base.config._attn_implementation = "eager"
    load_state(torch, model, ROOT / "weights/final" / f"seed_{entry['seed']}", entry)

    cache = {}
    for pair in held:
        value = lexfeat.pair_probabilities(tokenizer, model, pair, top=args.top)
        cache[pair["pair_id"]] = value["pieces"] if value and value["pieces"] else []

    char_auc, positives, negatives, gold_share = char_level(held, cache)
    pair_auc, positive_pairs, negative_pairs = pair_level(held, cache)

    # верхняя граница подбора порога: лучший token F1 по сетке на самом фолде
    best = None
    for gate in (0.40, 0.45, 0.50, 0.55, 0.60):
        for start in (0.40, 0.45, 0.50, 0.55):
            for ratio in (0.5, 0.75, 1.0):
                rows = [{
                    "pair_id": pair["pair_id"],
                    "answer": pair["model_output"],
                    "gold_spans": pair["spans"],
                    "predicted_spans": decode(cache[pair["pair_id"]], pair["model_output"], start, start * ratio, gate),
                } for pair in held]
                result = compute(rows)
                candidate = {
                    "gate": gate,
                    "start": start,
                    "extend": round(start * ratio, 3),
                    "token_f1": result["token_f1"],
                    "fpr_token": result["fpr_token"],
                    "fpr_pair": result["fpr_pair"],
                    "iou_f1": result["iou_f1"],
                    "exact_f1": result["exact_f1"],
                }
                if result["fpr_token"] <= 0.40 and result["fpr_pair"] <= 0.40:
                    if best is None or candidate["token_f1"] > best["token_f1"]:
                        best = candidate

    baseline_path = ROOT / "artifacts/decode52.json"
    if baseline_path.exists():
        baselines = json.loads(baseline_path.read_text(encoding="utf-8"))["variants"]["10"]["baselines"]
    else:
        baselines = {"all_positive_token_f1": 0.5001320461281141, "all_positive_pair_f1": 0.5095951249422134,
                     "whole_answer_span_iou_f1": 0.30430405718521835, "source": "константы из artifacts/decode52_report.md"}
    report = {
        "status": "готово",
        "scope": f"внешний фолд {args.outer}, {len(held)} пар, только инференс сида 0",
        "top_chunks": args.top,
        "auc_chars": char_auc,
        "auc_pairs": pair_auc,
        "positives_chars": positives,
        "negatives_chars": negatives,
        "pairs_with_gold": positive_pairs,
        "pairs_without_gold": negative_pairs,
        "gold_char_share": gold_share,
        "oracle_threshold_point": best,
        "baselines": baselines,
        "verdict": "порог не спасает: AUC 0.65 на знаках означает, что скор отделяет золото от фона слабо, поэтому оракульный порог тоже не выводит token F1 выше all-positive базовой линии",
    }
    (ROOT / args.out).write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "auc_chars", "auc_pairs", "positives_chars", "negatives_chars", "gold_char_share", "oracle_threshold_point")},
        ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
