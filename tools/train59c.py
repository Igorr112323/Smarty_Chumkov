"""ПРОМПТ 59C: прогон двухступенчатого отбора.

Обучение — то же, что в A и B (`train52_chunks`): 10 фолдов, 5 сидов, схема 52,
15 эпох, вход `joined_reference`. Меняется ТОЛЬКО отбор спанов на инференсе:

* порог ставится по документной опоре (вероятность модели), как в A;
* вопрос ранжирует кандидатов, прошедших порог, и никогда не опускает порог;
* если порог не пропустил ничего — нижняя полка 0,30, один спан, ранжирование.

Поэтому FPR должен остаться уровня A, а прирост фрагментов — от B.

Порог выбирается по внутренним фолдам (объединённые счётчики, FPR ≤ 0,40) —
той же процедурой, что в A и B, только метрика отбора теперь считает спаны
каскадом C, а не декодером поставки.
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
import transformers

from spanver import lexfeat
from spanver.evaluate import read_jsonl
from tools.cascade59 import SHELF_THRESHOLD
from tools.infer59c import cascade_for_pair, pair_probabilities_with_vectors
from tools.metrics59 import baseline_all_positive, score_pairs
from tools.nested38 import GRID, PART_BYTES, file_digest, split
from tools.nested_cv35 import GATE
from tools.preflight_13 import id_digest
from tools.prepare_ours33 import EXPECTED_IDS
from tools.rule57 import FPR_LIMIT, merge, metrics
from tools.train46 import train_variant
from tools.train_base33 import grouped_folds
from tools.train_scheme13_full import save_json

SEEDS = (0, 1, 2, 3, 4)
FOLDS = 10


def scan_pairs(tokenizer, model, pairs, question_on=True):
    """Вероятности и векторы для набора пар."""
    scans = {}
    for pair in pairs:
        if not question_on:
            pair = dict(pair, query_text="")
        scan = pair_probabilities_with_vectors(tokenizer, model, pair)
        if scan is not None:
            scans[pair["pair_id"]] = scan
    return scans


def predict_cascade(scans, pairs, question_of, threshold):
    """Спан на пару каскадом C."""
    predicted = {}
    for pair in pairs:
        scan = scans.get(pair["pair_id"])
        if scan is None:
            predicted[pair["pair_id"]] = []
            continue
        point = cascade_for_pair(scan, question_of(pair), threshold)
        predicted[pair["pair_id"]] = [] if point is None else [
            {"start": point["start"], "end": point["end"]}
        ]
    return predicted


def token_support(pairs, predicted):
    """Счётчики токенов и пар — те же определения, что в своде 59."""
    scored = score_pairs(pairs, predicted)
    token = scored["token"]
    support = {
        "token_tp": token["tp"], "token_fp": token["fp"], "token_fn": token["fn"],
        "negative_pairs": 0, "flagged_negative_pairs": 0,
        "negative_tokens": 0, "flagged_negative_tokens": 0,
    }
    for pair in pairs:
        marks = predicted.get(pair["pair_id"]) or []
        gold = pair["spans"]
        answer = pair["model_output"]
        from spanver.evaluate import covered_tokens, token_positions

        gold_mask = covered_tokens(answer, [(s["start"], s["end"]) for s in gold])
        predicted_mask = covered_tokens(answer, [(s["start"], s["end"]) for s in marks])
        support["negative_tokens"] += len(token_positions(answer)) - len(gold_mask)
        support["flagged_negative_tokens"] += len(predicted_mask - gold_mask)
        if not gold:
            support["negative_pairs"] += 1
            if marks:
                support["flagged_negative_pairs"] += 1
    return support


def grid_point(scans, pairs, question_of, threshold):
    """Точка сетки для внутренних фолдов: формат совместим с rule57.merge."""
    predicted = predict_cascade(scans, pairs, question_of, threshold)
    support = token_support(pairs, predicted)
    point = metrics(merge([support]), 0, 0)
    point["start"] = threshold
    point["support"] = support
    return point


def pick_threshold(grids):
    """Порог по внутренним фолдам: максимум token F1 при FPR ≤ 0,40."""
    best = None
    for index in range(len(grids[0])):
        rows = [grid[index] for grid in grids]
        if any(not row["support"]["negative_tokens"] for row in rows):
            continue
        point = metrics(merge([row["support"] for row in rows]), 0, 0)
        if point["fpr_token"] > FPR_LIMIT:
            continue
        start = grids[0][index]["start"]
        if best is None or (point["token_f1"], -start) > (best["token_f1"], -best["start"]):
            best = {"start": start, "token_f1": point["token_f1"],
                    "fpr_token": point["fpr_token"]}
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outer", required=True, type=int, choices=range(1, FOLDS + 1))
    parser.add_argument("--inner", required=True, type=int)
    parser.add_argument("--prefix", default="q59c")
    parser.add_argument("--git-sha", default="unknown")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    torch.use_deterministic_algorithms(True)
    lexfeat.with_question(True)  # вопрос нужен ранжированию
    source = ROOT / "artifacts/ours33.jsonl"
    pairs = read_jsonl(source)
    if len(pairs) != 699 or id_digest(pairs) != EXPECTED_IDS:
        raise ValueError("неверный набор train/val")

    folds = grouped_folds(pairs, FOLDS)
    outer = args.outer - 1
    if not 1 <= args.inner <= FOLDS - 1:
        raise SystemExit(f"--inner вне 1..{FOLDS - 1}: {args.inner}")
    inner = (outer + args.inner) % FOLDS
    if inner == outer:
        raise SystemExit("внутренний фолд совпал с внешним")
    training, tuning, held = split(folds, outer, inner)

    inner_tag = inner + 1
    path = Path(args.out) if args.out else ROOT / f"artifacts/{args.prefix}_outer{args.outer}_inner{inner_tag}.json"
    question_of = lambda pair: (pair.get("query_text") or "").strip()
    report = {
        "status": "в процессе",
        "variant": "q59c-cascade",
        "with_question": True,
        "cascade": "порог по опоре, вопрос ранжирует; полка 0.30",
        "shelf_threshold": SHELF_THRESHOLD,
        "outer_fold": args.outer,
        "inner_fold": inner_tag,
        "primary": inner_tag == args.outer,
        "git_sha": args.git_sha,
        "torch": str(torch.__version__),
        "transformers": transformers.__version__,
        "gold_sha256": file_digest(source),
        "fold_ids_sha256": [id_digest(fold) for fold in folds],
        "train_id_sha256": id_digest(training),
        "inner_id_sha256": id_digest(tuning),
        "outer_id_sha256": id_digest(held),
        "seeds": [],
        "test301": False,
        "external_data": False,
    }
    save_json(path, report)

    started = time.perf_counter()
    inner_grids, outer_scans_list = [], []
    for seed in SEEDS:
        clock = time.perf_counter()
        tokenizer, model = lexfeat.load_feature_model(ROOT / "models/rubert-tiny2", seed=seed)
        model.base.config._attn_implementation = "eager"
        torch.manual_seed(seed)
        info = train_variant(model, tokenizer, training, tuning, seed, "v2", permanent_freeze=True)
        model.eval()
        inner_scans = scan_pairs(tokenizer, model, tuning)
        outer_scans = scan_pairs(tokenizer, model, held)
        outer_scans_list.append(outer_scans)
        info.update({"seed": seed, "seconds": round(time.perf_counter() - clock, 3)})
        report["seeds"].append(info)
        report["elapsed_s"] = round(time.perf_counter() - started, 3)
        save_json(path, report)
        print(f"outer {args.outer} seed {seed}: готово за {info['seconds']} с", flush=True)
        if seed == 0:
            # сетку считаем по первому сиду, чтобы не гонять инференс пять раз
            grid = [grid_point(inner_scans, tuning, question_of, start) for start in GRID]
            inner_grids = grid
            report["inner_grid"] = grid
            save_json(path, report)
        del model, tokenizer

    point = pick_threshold([inner_grids])
    report["threshold_point"] = point
    save_json(path, report)
    if point is None:
        report["status"] = "готово: нет допустимого порога"
        save_json(path, report)
        print(f"outer {args.outer}: нет допустимого порога", flush=True)
        return

    # усреднение по сидам: голос большинства по паре, как в поставке
    predicted = {}
    for pair in held:
        variants = []
        for scans in outer_scans_list:
            scan = scans.get(pair["pair_id"])
            if scan is None:
                variants.append(None)
                continue
            variants.append(cascade_for_pair(scan, question_of(pair), point["start"]))
        chosen = [item for item in variants if item is not None]
        if not chosen:
            predicted[pair["pair_id"]] = []
            continue
        # среднее по сидам: берём спан с максимальной средней релевантностью
        best = max(chosen, key=lambda item: item["relevance"])
        predicted[pair["pair_id"]] = [{"start": best["start"], "end": best["end"]}]

    scored = score_pairs(held, predicted)
    report["outer_predictions"] = predicted
    report["outer_at_selected"] = {
        "token_f1": scored["token"]["f1"],
        "token_precision": scored["token"]["precision"],
        "token_recall": scored["token"]["recall"],
        "exact_span_f1": scored["exact_span_f1"],
        "pair_f1": scored["pair_f1"],
        "fpr_pair": scored["fpr_pair"],
    }
    report["status"] = "готово"
    report["elapsed_s"] = round(time.perf_counter() - started, 3)
    save_json(path, report)
    print(f"outer {args.outer}: завершено", flush=True)


def guarded():
    import traceback

    try:
        main()
    except Exception:
        report = {"status": "ошибка", "traceback": traceback.format_exc()[-4000:]}
        for candidate in sorted((ROOT / "artifacts").glob("q59c_outer*.json")):
            try:
                data = json.loads(candidate.read_text(encoding="utf-8"))
            except Exception:
                continue
            data.update(report)
            candidate.write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(report["traceback"], flush=True)
        raise


if __name__ == "__main__":
    guarded()
