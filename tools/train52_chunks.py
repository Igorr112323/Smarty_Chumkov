"""ШАГ A: обучение с согласованным числом чанков.

Отличие от `tools/train51_ten.py` только одно: `spanver/lexfeat.build_table`
теперь кодирует все выбранные чанки пары (`TRAIN_TOP_CHUNKS`), а не первый.
Инференс и раньше считал по четырём чанкам, поэтому обучение и инференс
наконец видят один и тот же контекст.

Схема, пороги и оценка — те же, что в 51: пятифолдовая группировка по
документам, обучение на train минус внутренний фолд, порог выбирается по
внутреннему фолду с ограничениями FPR, внешний фолд не смотрится до конца.

Запуск только в Actions (локальная машина — 2 ядра, это часы на фолд).
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
from tools.nested38 import GRID, PART_BYTES, file_digest, split, target_inner
from tools.nested_cv35 import GATE, averaged, probabilities
from tools.preflight_13 import id_digest
from tools.prepare_ours33 import EXPECTED_IDS
from tools.sweep_ragtruth30 import compute, prepare
from tools.train46 import train_variant
from tools.train_base33 import grouped_folds
from tools.train_scheme13_full import save_json

SEEDS = (0, 1, 2, 3, 4)
FOLDS = 5


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outer", required=True, type=int, choices=range(1, FOLDS + 1))
    parser.add_argument("--git-sha", default="unknown")
    # ПРОМПТ 59: единственное различие между контролем A и опытом B.
    # По умолчанию выключено — поведение без флага совпадает с прежним.
    parser.add_argument("--with-question", choices=["true", "false"], default="false")
    parser.add_argument("--variant-tag", default="chunks")
    parser.add_argument("--out", default=None)
    parser.add_argument("--prefix", default="ablation52")
    # номер внутреннего фолда — смещение по циклу от внешнего, 1..4
    parser.add_argument("--inner", type=int, default=None)
    args = parser.parse_args()
    torch.use_deterministic_algorithms(True)
    with_question = args.with_question == "true"
    lexfeat.with_question(with_question)
    source = ROOT / "artifacts/ours33.jsonl"
    pairs = read_jsonl(source)
    if len(pairs) != 699 or id_digest(pairs) != EXPECTED_IDS:
        raise ValueError("неверный набор train/val")
    folds = grouped_folds(pairs, FOLDS)
    outer = args.outer - 1
    if getattr(args, "inner", None):
        # ПРОМПТ 59: 10 фолдов × 4 внутренних, как в схеме 56/51. Внутренние
        # смещены по циклу +1..+4, чтобы внешний фолд не попадал ни в train, ни
        # в tuning. В схеме 52 был один внутренний фолд ((outer+1) % 5).
        if not 1 <= args.inner <= FOLDS - 1:
            raise SystemExit(f"--inner вне 1..{FOLDS - 1}: {args.inner}")
        inner = (outer + args.inner) % FOLDS
    else:
        inner = target_inner(outer)
    training, tuning, held = split(folds, outer, inner)
    # Имя — по РЕАЛЬНОМУ номеру внутреннего фолда (inner + 1), а не по смещению
    # args.inner. Иначе для внешнего 6 при смещении 1 файл назывался бы
    # ..._inner1.json и затирал отчёт задания (6, 1). Задания (a,1,1) и (a,2,4)
    # оба дают inner+1 == 2, но смещения разные, поэтому затирания нет.
    inner_tag = inner + 1
    path = Path(args.out) if args.out else ROOT / f"artifacts/{args.prefix}_outer{args.outer}_inner{inner_tag}.json"
    report = {
        "status": "в процессе",
        "variant": args.variant_tag,
        "with_question": with_question,
        "outer_fold": args.outer,
        "inner_fold": inner + 1,
        "primary": (args.inner or (inner + 1)) == args.outer,
        "git_sha": args.git_sha,
        "torch": str(torch.__version__),
        "transformers": transformers.__version__,
        "gold_sha256": file_digest(source),
        "fold_ids_sha256": [id_digest(fold) for fold in folds],
        "train_id_sha256": id_digest(training),
        "inner_id_sha256": id_digest(tuning),
        "outer_id_sha256": id_digest(held),
        "train_top_chunks": lexfeat.TRAIN_TOP_CHUNKS,
        "question_input": lexfeat.QUESTION_INPUT,
        "question_pairs": sum(1 for row in pairs if (row.get("query_text") or "").strip()),
        "training_rows": None,
        "seeds": [],
        "test301": False,
        "external_data": False,
    }
    save_json(path, report)

    inner_caches, outer_caches = [], []
    started = time.perf_counter()
    for seed in SEEDS:
        clock = time.perf_counter()
        tokenizer, model = lexfeat.load_feature_model(ROOT / "models/rubert-tiny2", seed=seed)
        model.base.config._attn_implementation = "eager"
        torch.manual_seed(seed)
        info = train_variant(model, tokenizer, training, tuning, seed, "v2", permanent_freeze=True)
        if report["training_rows"] is None:
            report["training_rows"] = len(lexfeat.build_table(tokenizer, training))
            save_json(path, report)
        inner_caches.append(probabilities(tokenizer, model, tuning))
        outer_caches.append(probabilities(tokenizer, model, held))
        info.update({"seed": seed, "seconds": round(time.perf_counter() - clock, 3)})
        report["seeds"].append(info)
        report["elapsed_s"] = round(time.perf_counter() - started, 3)
        save_json(path, report)
        print(f"outer {args.outer} seed {seed}: готово за {info['seconds']} с", flush=True)
        del model, tokenizer

    inner_cache = {row["pair_id"]: averaged(row, inner_caches, range(len(SEEDS))) for row in tuning}
    prepared = prepare(tuning, inner_cache)
    # отчёт по каждому сиду сохраняется ДО агрегации: обучение идёт 13 мин на сид,
    # и терять пять сидов из-за ошибки в пост-обработке нельзя
    save_json(path, report)
    report["inner_grid"] = [compute(prepared, threshold, threshold / 2, GATE) for threshold in GRID]

    # то же правило отбора, что зафиксировано на стадии 1: объединённые счётчики
    # внутренних фолдов, ограничения FPR, максимум token F1
    rows = report["inner_grid"]
    # поля вложенные: sweep_ragtruth30.compute кладёт token/span/pair в подобъекты,
    # а fpr_* — в корень. Первая версия читала row["token_f1"] и падала.
    allowed = [row for row in rows if row["fpr_token"] <= 0.40 and row["fpr_pair"] <= 0.40]
    report["threshold_point"] = (max(allowed, key=lambda row: row["token_f1_exact"]) if allowed else None)

    outer_cache = {row["pair_id"]: averaged(row, outer_caches, range(len(SEEDS))) for row in held}
    report["outer_predictions"] = {key: value["pieces"] for key, value in outer_cache.items()}
    if report["threshold_point"] is not None:
        threshold = report["threshold_point"]["start"]
        import json as json_module

        payload = {key: json_module.loads(json_module.dumps(value)) for key, value in outer_cache.items()}
        report["outer_at_selected"] = compute(prepare(held, payload), threshold, threshold / 2, GATE)
    report["status"] = "готово"
    report["elapsed_s"] = round(time.perf_counter() - started, 3)
    save_json(path, report)
    print(f"outer {args.outer}: завершено", flush=True)


def guarded():
    """Пишет трейсбек в отчёт: логи Actions в песочнице недоступны."""
    import traceback

    try:
        main()
    except Exception:
        report = {"status": "ошибка", "traceback": traceback.format_exc()[-4000:]}
        for candidate in sorted((ROOT / "artifacts").glob("ablation52_outer*.json")):
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
