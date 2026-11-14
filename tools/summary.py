import argparse
import json
from pathlib import Path


def row(name, value):
    return f"| {name} | {value} |"


def build(report):
    test = report.get("test") or {}
    span = test.get("span_level") or {}
    token = test.get("token_level") or {}
    pair = test.get("pair_level") or {}
    threshold = report.get("threshold") or {}
    stress = report.get("stress") or {}
    calibration = report.get("calibration") or {}
    model = report.get("model") or {}
    head = report.get("head") or {}
    package = report.get("package") or {}
    lines = [
        "## Метрики договора",
        "",
        "| показатель | значение |",
        "|---|---|",
        row("правило TP", "IoU ≥ 0,5 без условия типа"),
        row("test, пар", test.get("n_pairs")),
        row("span precision", span.get("precision")),
        row("span recall", span.get("recall")),
        row("span F1", span.get("f1")),
        row("token precision", token.get("precision")),
        row("token recall", token.get("recall")),
        row("token F1", token.get("f1")),
        row("pair precision", pair.get("precision")),
        row("pair recall", pair.get("recall")),
        row("pair F1", pair.get("f1")),
        row("FPR", test.get("fpr")),
        row("точность типа", test.get("type_accuracy")),
        row("порог (выбран на val)", threshold.get("value")),
        row("F1 на val при пороге", threshold.get("val_f1")),
        row("обучающих пар в головке", head.get("train_pairs")),
        row("модель", model.get("name")),
        row("веса, sha256", (model.get("weights_sha256") or "")[:16]),
        row("калибровка ai_share, max|Δ|", calibration.get("max_abs_err")),
        row("калибровка, leave-one-out", calibration.get("loo_max_abs_err")),
        row("стресс, знаков", stress.get("stress_chars")),
        row("стресс, медиана мс", stress.get("median_elapsed_ms")),
        row("стресс, мс на 1000", stress.get("ms_per_1000")),
        row("пакет, байт", package.get("bytes")),
        row("пакет, sha256", package.get("sha256")),
        row("seed", report.get("seed")),
    ]
    verdict = "достигнут" if (span.get("f1") or 0) >= 0.50 else "не достигнут"
    lines += ["", f"**F1 против 0,50: {verdict}** (факт {span.get('f1')}, FPR {test.get('fpr')}).", ""]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", default="artifacts/model_eval_report.json")
    parser.add_argument("--out", default="")
    args = parser.parse_args(argv)
    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    text = build(report)
    print(text)
    if args.out:
        target = Path(args.out)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
