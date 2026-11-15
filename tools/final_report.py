import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"


def read(name):
    path = ARTIFACTS / name
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", default="artifacts/report.json")
    args = parser.parse_args(argv)

    target = Path(args.report)
    data = json.loads(target.read_text(encoding="utf-8")) if target.exists() else {}
    base5 = read("cv_report_base5.json")
    hard5 = read("cv_report_hard5.json")
    feat5 = read("cv_report_feat5.json")
    mixed = read("mixed_w.json")
    speed = read("speed_report.json")
    hard = read("hard_report.json")
    frozen = read("frozen_config.json")
    test = read("test_frozen.json")

    data["honest_eval"] = {
        "one_shot_test_301": {
            "token_f1": test.get("token_f1"),
            "span_f1": test.get("span_f1"),
            "pair_f1": test.get("pair_f1"),
            "fpr_pair": test.get("fpr_pair"),
            "fpr_token": test.get("fpr_token"),
            "config": "frozen_config.json, чекпоинт negatives/epoch_1 (утрачен при перезапуске песочницы)",
            "note": "единственный прогон test, повторов нет",
        },
        "cv_5fold_699": {
            "base": base5.get("summary", {}),
            "hard_negatives": hard5.get("summary", {}),
            "lexical_features": feat5.get("summary", {}),
            "folds": 5,
            "epochs_per_fold": 1,
            "note": "обучение с нуля от models/rubert-tiny2 на каждом фолде, RuFacts-негативы пропорционально",
        },
        "lexical_on_699": {
            "span_f1": 0.3002,
            "token_f1": 0.5121,
            "pair_f1": 0.374,
            "fpr_pair": 0.3783,
            "fpr_token": 0.6448,
            "type_accuracy": 0.7158,
        },
        "lexical_on_test_301": {"span_f1": 0.2835, "token_f1": 0.3923, "pair_f1": 0.3547, "fpr": 0.3527},
    }
    data["fpr_budget"] = {
        "limit": 0.40,
        "one_shot_test": {"fpr_pair": test.get("fpr_pair"), "used_share": round((test.get("fpr_pair") or 0) / 0.40, 4)},
        "cv_base": {"fpr_pair": base5.get("summary", {}).get("fpr_pair", {}).get("mean")},
        "cv_hard": {"fpr_pair": hard5.get("summary", {}).get("fpr_pair", {}).get("mean")},
        "cv_feat": {"fpr_pair": feat5.get("summary", {}).get("fpr_pair", {}).get("mean")},
        "ensemble_b_val150": {"fpr_pair": 0.0769, "note": "значение прошлого промта, val150"},
    }
    data["speed"] = {
        "rows": speed.get("rows", []),
        "norm_ms_per_1000_doc_chars": 30,
        "checkpoint": speed.get("checkpoint"),
    }
    data["mixed_mode"] = {
        "weights": mixed.get("weights"),
        "thresholds": mixed.get("thresholds"),
        "rows": mixed.get("rows", []),
        "chosen": mixed.get("chosen", []),
        "best_span": mixed.get("best_span"),
        "best_token": mixed.get("best_token"),
        "detail": mixed.get("detail"),
        "target": 0.2835,
        "verdict": "отвергнуто: span F1 0.1779 ниже 0.2835 и ниже чистой лексики 0.3002 на тех же 699 парах",
    }
    data["hard_negatives"] = hard
    data["cv_reports"] = {
        "base": {"summary": base5.get("summary"), "rows": base5.get("rows"), "elapsed_s": base5.get("elapsed_s")},
        "hard": {"summary": hard5.get("summary"), "rows": hard5.get("rows"), "elapsed_s": hard5.get("elapsed_s")},
        "feat": {"summary": feat5.get("summary"), "rows": feat5.get("rows"), "elapsed_s": feat5.get("elapsed_s")},
    }
    data["frozen_config_sha256"] = sha(ARTIFACTS / "frozen_config.json") if (ARTIFACTS / "frozen_config.json").exists() else None
    data["rejected_hypotheses"] = [
        "схема 1: NMS_IOU и MAX_SPANS не влияют — после порога риска 0,98 у лексики ≤ 1 кандидата на пару (36 комбинаций дают 0,2595 span F1 на val150)",
        "схема 3 shrink: token F1 0,5354 → 0,4433 на val150",
        "схема 3 на лексическом бэкенде: expand/shrink/nearest не меняют span F1",
        "ступени A/B/C на RuFacts: span F1 0,0044",
        "RuFacts-негативы поверх v5_last: span F1 0,0569 → 0,0367 после 1-й эпохи",
        "ансамбли A/C/D на val150: FPR_pair 0,48…0,49 выше лимита 0,40",
        "смешанный режим w-скор (промт 21, шаг 1): лучший span F1 0,1779 при FPR_pair 0,3783 на 699 парах, ниже 0,2835 и ниже чистой лексики 0,3002",
        "порог токена 0,20 как оптимум: расширенная сетка 0,05…0,30 дала внутренний оптимум 0,25 (val150 0,5751)",
        "порог токена 0,25 на честном test: token F1 0,4217 при FPR_pair 0,3140 — гейт 0,50 не подтверждён",
    ]
    target.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"отчёт обновлён: {target}", flush=True)
    for name in ("honest_eval", "fpr_budget", "speed", "mixed_mode"):
        print(f"{name}: ok", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
