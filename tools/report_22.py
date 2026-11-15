import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"


def read(name):
    path = ARTIFACTS / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def epoch_table(report):
    rows = {}
    for fold in report.get("rows", []):
        for probe in fold.get("epoch_rows", []):
            rows.setdefault(probe["epoch"], []).append(probe)
    table = {}
    for epoch, probes in sorted(rows.items()):
        entry = {}
        for metric in ("token_f1", "span_f1", "pair_f1", "fpr_pair", "fpr_token"):
            values = [probe[metric] for probe in probes]
            mean = sum(values) / len(values)
            variance = sum((value - mean) ** 2 for value in values) / len(values)
            entry[metric] = {"mean": round(mean, 4), "std": round(variance ** 0.5, 4)}
        table[str(epoch)] = entry
    return table


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", default="artifacts/report.json")
    args = parser.parse_args(argv)

    target = Path(args.report)
    data = json.loads(target.read_text(encoding="utf-8")) if target.exists() else {}
    speed = read("speed_report_batched.json")
    hyst = read("sweep_hyst.json")
    feat1 = read("cv_report_feat5.json")
    feat3 = read("cv_report_feat5_ep3.json")
    base = read("cv_report_base5.json")
    hard = read("cv_report_hard5.json")
    focal = read("cv_report_focal.json")
    weight = read("cv_report_weight.json")

    data["speed_batched"] = {
        "rows": speed.get("rows", []),
        "batch": speed.get("batch"),
        "norm_ms_per_1000_doc_chars": 30,
        "verdict": "на документах 200 знаков норма не достигается: модель 37,26, ансамбль 58,17 — блокер по договору",
    }
    data["hysteresis"] = {
        "grid": hyst.get("hysteresis", {}),
        "best": hyst.get("best_hysteresis", {}),
        "contrast": hyst.get("contrast", {}),
        "verdict": "гистерезис снижает FPR_token до 0,3864 при потере token F1 до 0,4852; контраст по медиане вырождается (все спаны отсекаются при K ≥ 1,5)",
    }
    data["cv_variants"] = {
        "base_1epoch": base.get("summary", {}),
        "hard_negatives_1epoch": hard.get("summary", {}),
        "lexical_features_1epoch": feat1.get("summary", {}),
        "lexical_features_3epochs": feat3.get("summary", {}),
        "lexical_features_focal": focal.get("summary", {}),
        "lexical_features_class_weight": weight.get("summary", {}),
        "epochs_curve": epoch_table(feat3),
    }
    data["three_lines"] = {
        "one_shot_test_301": {"token_f1": 0.4217, "span_f1": 0.0876, "pair_f1": 0.2717, "fpr_pair": 0.3140, "fpr_token": 0.3913, "note": "единственный честный прогон, повторов нет"},
        "cv_scheme12_1epoch": {"token_f1": 0.5138, "token_std": 0.0630, "span_f1": 0.2982, "span_std": 0.0501, "pair_f1": 0.3841, "fpr_pair": 0.1871, "fpr_token": 0.4324},
        "cv_best_after_steps": {"note": "лучшая конфигурация по правилу max token F1 при FPR_pair ≤ 0,40 и FPR_token ≤ 0,40", "value": None},
    }
    data["vs_lexical_test_only"] = {
        "lexical_test_301": {"token_f1": 0.3923, "span_f1": 0.2835, "pair_f1": 0.3547, "fpr_pair": 0.3527},
        "model_test_301": {"token_f1": 0.4217, "span_f1": 0.0876, "pair_f1": 0.2717, "fpr_pair": 0.3140},
        "note": "CV-числа с test не сравниваются, новые прогоны test не делались",
    }
    target.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("отчёт обновлён", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
