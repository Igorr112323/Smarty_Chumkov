import argparse
import hashlib
import json
from pathlib import Path

MODULE_REPORT = "artifacts/module_report.json"
MODEL_REPORT = "artifacts/model_eval_report.json"
TARGET = "artifacts/compliance.json"
DEFAULT_BACKEND = "lexical"
NAMES = {
    1: "локальный модуль без сети и GPU, ядро на stdlib",
    2: "формат ответа: spans, ai_share, meta, семь типов",
    3: "пять признаков на токен, включая внимание",
    4: "доля участия ИИ 0–1 и калибровка шагом 0,25",
    5: "P/R/F1/FPR по трём уровням, IoU ≥ 0,5",
    6: "F1 ≥ 0,50 при FPR ≤ 0,40 на test",
    7: "скорость: 50k, 2M, мс на 1000 знаков",
    8: "CLI, HTTP и локальный веб",
    9: "пакет ≤ 0,6 МБ и веса отдельным файлом",
}
DONE = "выполнено"
PARTIAL = "частично"
FAILED = "не выполнено"
UNKNOWN = "нет данных"


def read(path):
    target = Path(path)
    if not target.exists():
        return None, None
    raw = target.read_bytes()
    return json.loads(raw.decode("utf-8")), hashlib.sha256(raw).hexdigest()


def item(number, status, value):
    return {"item": number, "name": NAMES[number], "status": status, "value": value}


def first_item(module):
    contract = (module or {}).get("contract") or {}
    if not contract:
        return item(1, UNKNOWN, "нет поля contract в module_report.json")
    clean = not contract.get("network") and not contract.get("gpu") and contract.get("stdlib_only")
    value = (
        f"сеть: {'нет' if not contract.get('network') else 'есть'}, "
        f"GPU: {'нет' if not contract.get('gpu') else 'есть'}, "
        f"ядро на stdlib: {'да' if contract.get('stdlib_only') else 'нет'}"
    )
    return item(1, DONE if clean else FAILED, value)


def second_item(module):
    contract = (module or {}).get("contract") or {}
    answer = contract.get("answer") or []
    span = contract.get("span") or []
    types = contract.get("types") or []
    if not answer and not span:
        return item(2, UNKNOWN, "нет поля contract в module_report.json")
    complete = len(answer) == 3 and len(span) == 5
    value = f"ответ: {', '.join(answer)}; фрагмент: {', '.join(span)}; типов: {len(types)}"
    return item(2, DONE if complete else FAILED, value)


def third_item(module, model):
    token_features = ((model or {}).get("features") or {}).get("token_level") or []
    head_features = ((model or {}).get("features") or {}).get("head") or []
    lexical = ((module or {}).get("module") or {}).get("five_token_features") or []
    if not token_features:
        return item(3, UNKNOWN, "нет поля features.token_level в model_eval_report.json")
    required = ("doc_attention_share", "attention_entropy", "cos_proximity")
    covered = [name for name in required if name in token_features]
    value = (
        f"признаков на токен: {len(token_features)} ({', '.join(token_features)}); "
        f"обязательных из договора: {len(covered)} из {len(required)}; "
        f"в головке чисел: {len(head_features)}; базовые лексические: {', '.join(lexical)}"
    )
    status = DONE if len(token_features) == 5 and len(covered) == 3 else PARTIAL
    return item(3, status, value)


def fourth_item(module, model):
    lexical = (module or {}).get("ai_share_calib") or {}
    heavy = (model or {}).get("calibration") or {}
    lexical_points = lexical.get("points") or []
    heavy_points = heavy.get("points") or []
    if not lexical_points and not heavy_points:
        return item(4, UNKNOWN, "нет полей ai_share_calib и calibration в отчётах")
    lexical_error = lexical.get("max_abs_err")
    heavy_error = heavy.get("max_abs_err")
    value = (
        f"лексический: точек {len(lexical_points)}, max|Δ| {lexical_error}, "
        f"leave-one-out {lexical.get('loo_max_abs_err')}; "
        f"модельный: точек {len(heavy_points)}, max|Δ| {heavy_error}, "
        f"leave-one-out {heavy.get('loo_max_abs_err')}"
    )
    if lexical.get("max_abs_err") == 0.0 and len(lexical_points) >= 5:
        status = DONE if (heavy_error is not None and heavy_error <= 0.25) else PARTIAL
    else:
        status = PARTIAL
    return item(4, status, value)


def fifth_item(module, model):
    lexical = ((module or {}).get("eval_test") or {})
    lexical_span = lexical.get("span_level") or {}
    test = (model or {}).get("test") or {}
    span = test.get("span_level") or {}
    token = test.get("token_level") or {}
    pair = test.get("pair_level") or {}
    if not span or not token or not pair:
        return item(5, UNKNOWN, "нет полей test.span_level / token_level / pair_level")
    value = (
        f"модельный: span F1 {span.get('f1')}, token F1 {token.get('f1')}, pair F1 {pair.get('f1')}, FPR {test.get('fpr')}; "
        f"лексический: span F1 {lexical_span.get('f1')}, FPR {lexical.get('fpr')}; "
        f"правило {test.get('rule')}, пар {test.get('n_pairs')}"
    )
    return item(5, DONE if test.get("rule") else PARTIAL, value)


def sixth_item(module, model):
    test = (model or {}).get("test") or {}
    verdict = test.get("contract_pass") or {}
    span = test.get("span_level") or {}
    lexical = ((module or {}).get("eval_test") or {})
    lexical_span = lexical.get("span_level") or {}
    lexical_verdict = lexical.get("contract_pass") or {}
    if not verdict and not lexical_verdict:
        return item(6, UNKNOWN, "нет поля contract_pass в отчётах")
    passed = []
    if lexical_verdict.get("f1_ge_0.50") and lexical_verdict.get("fpr_le_0.40"):
        passed.append("лексический")
    if verdict.get("f1_ge_0.50") and verdict.get("fpr_le_0.40"):
        passed.append("модельный")
    value = (
        f"модельный F1 {span.get('f1')} при FPR {test.get('fpr')}; "
        f"лексический F1 {lexical_span.get('f1')} при FPR {lexical.get('fpr')}"
    )
    return item(6, DONE if passed else FAILED, value)


def seventh_item(module, model):
    speed = (module or {}).get("speed") or {}
    stress = (module or {}).get("stress") or {}
    heavy = (model or {}).get("stress") or {}
    if not speed and not stress and not heavy:
        return item(7, UNKNOWN, "нет полей speed и stress в отчётах")
    flags = [speed.get("under_1s"), stress.get("under_60s"), stress.get("under_30ms_per_1000")]
    value = (
        f"50k {speed.get('median_elapsed_ms')} мс (≤ 1 с: {speed.get('under_1s')}); "
        f"2M {stress.get('median_elapsed_ms')} мс, {stress.get('ms_per_1000')} мс на 1000 (≤ 60 с: {stress.get('under_60s')}); "
        f"модельный 2M {heavy.get('median_elapsed_ms')} мс, {heavy.get('ms_per_1000')} мс на 1000 (≤ 60 с: {heavy.get('under_60s')})"
    )
    heavy_ok = heavy.get("under_60s") if heavy else None
    if all(flags):
        status = DONE if heavy_ok is not False else PARTIAL
    else:
        status = FAILED
    return item(7, status, value)


def eighth_item(module):
    contract = (module or {}).get("contract") or {}
    interfaces = contract.get("interfaces") or []
    if not interfaces:
        return item(8, UNKNOWN, "нет поля contract.interfaces в module_report.json")
    return item(8, DONE if len(interfaces) == 3 else PARTIAL, f"интерфейсы: {', '.join(interfaces)}")


def ninth_item(module, model):
    package = (module or {}).get("package") or {}
    weights = ((model or {}).get("model") or {}).get("weights_bytes")
    if not package.get("built"):
        return item(9, UNKNOWN, "пакет не собран, нет поля package в module_report.json")
    value = (
        f"пакет {package.get('package')} {package.get('bytes')} байт (лимит соблюдён: {package.get('under_limit')}), "
        f"sha256 {package.get('sha256')}; веса отдельным файлом: {weights} байт"
    )
    status = DONE if package.get("under_limit") and weights else PARTIAL
    return item(9, status, value)


def build(module_path=MODULE_REPORT, model_path=MODEL_REPORT):
    module, module_sha = read(module_path)
    model, model_sha = read(model_path)
    items = [
        first_item(module),
        second_item(module),
        third_item(module, model),
        fourth_item(module, model),
        fifth_item(module, model),
        sixth_item(module, model),
        seventh_item(module, model),
        eighth_item(module),
        ninth_item(module, model),
    ]
    summary = {
        DONE: sum(1 for entry in items if entry["status"] == DONE),
        PARTIAL: sum(1 for entry in items if entry["status"] == PARTIAL),
        FAILED: sum(1 for entry in items if entry["status"] == FAILED),
        UNKNOWN: sum(1 for entry in items if entry["status"] == UNKNOWN),
        "всего": len(items),
    }
    return {
        "backend_default": DEFAULT_BACKEND,
        "sources": {
            str(module_path): module_sha,
            str(model_path): model_sha,
        },
        "items": items,
        "summary": summary,
    }


def markdown(data):
    lines = [
        "## Соответствие договору",
        "",
        f"Бэкенд по умолчанию: **{data['backend_default']}**.",
        "",
        "| пункт | требование | статус | факт |",
        "|---|---|---|---|",
    ]
    for entry in data["items"]:
        lines.append(f"| {entry['item']} | {entry['name']} | {entry['status']} | {entry['value']} |")
    summary = data["summary"]
    lines += [
        "",
        f"Итог: выполнено {summary[DONE]}, частично {summary[PARTIAL]}, "
        f"не выполнено {summary[FAILED]}, нет данных {summary[UNKNOWN]} из {summary['всего']}.",
        "",
    ]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--module", default=MODULE_REPORT)
    parser.add_argument("--model", default=MODEL_REPORT)
    parser.add_argument("--out", default=TARGET)
    parser.add_argument("--markdown", action="store_true")
    args = parser.parse_args(argv)
    data = build(args.module, args.model)
    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.markdown:
        print(markdown(data))
    else:
        print(json.dumps(data, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
