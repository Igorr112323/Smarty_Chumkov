"""V14 шаг 5: реальные замеры скорости на CPU + попытка квантования.

Замеряется три вещи:

  A. Рантайм пакета (spanver, чистый Python): verify() на 1k / 50k / 2M знаков
     документа — те же точки, что в договоре (0.03 с/1k, 50k ≤ 1 с, 2M ≤ 60 с).

  B. Инференс span-модели на кандидатах: текущее представление (список словарей,
     как в ranker_weights_v3.json) против плоских массивов (int16-индексы,
     float32-пороги) и против квантованных порогов (сетка 1/256). Сравнивается
     скорость и совпадение предсказаний.

  C. Попытка загрузить веса модели «режима с весами» (RuBERT) — фиксируется
     блокер: в песочнице нет `pytorch_model.bin`, HF Hub недоступен, зеркал на
     GitHub не найдено. Квантование (int8/int4/ONNX) применять не к чему, пока
     весов нет; вместо этого меряется оптимизация представления деревьев.

Пишет artifacts/speed_v14.json.
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import numpy as np

from spanver.api import verify
from spanver.local_ranker import predict_trees

TARGETS = {"per_1000_chars": 0.03, "50k": 1.0, "2M": 60.0}


def build_document(length):
    pool = ("Компания была основана в 1902 году в Москве и выросла за столетие. "
            "Она выпускает оборудование для промышленности и поставляет его в 40 стран. ")
    document = (pool * (length // len(pool) + 1))[:length]
    answer = ("Компания основана в 1902 году в Санкт-Петербурге, а выручка выросла на 12 "
              "процентов. Компания поставляет оборудование в 55 стран.")
    return answer, document


def lexical_speed():
    rows = []
    for length, label in ((1000, "1k"), (50000, "50k"), (2000000, "2M")):
        answer, document = build_document(length)
        started = time.perf_counter()
        result = verify(answer, document)
        seconds = time.perf_counter() - started
        rows.append({
            "chars": len(document) + len(answer),
            "label": label,
            "seconds": round(seconds, 4),
            "ms_per_1000_chars": round(seconds * 1000.0 / max(1.0, (len(document) + len(answer)) / 1000.0), 3),
            "spans": len(result["spans"]),
            "ai_share": result["ai_share"],
        })
    checks = {
        "per_1000_chars_le_0.03": rows[0]["ms_per_1000_chars"] / 1000.0 <= TARGETS["per_1000_chars"],
        "50k_le_1s": rows[1]["seconds"] <= TARGETS["50k"],
        "2M_le_60s": rows[2]["seconds"] <= TARGETS["2M"],
    }
    return rows, checks


def flatten_model(model):
    """Плоское представление деревьев: int16-дети/признаки, float32-пороги/листья."""
    feature, threshold, left, right, value, offsets = [], [], [], [], [], [0]
    for tree in model["trees"]:
        base = len(feature)
        for node in tree:
            if node["leaf"]:
                feature.append(-1)
                threshold.append(0.0)
                left.append(-1)
                right.append(-1)
                value.append(float(node["v"]))
            else:
                feature.append(int(node["f"]))
                threshold.append(float(node["t"]))
                left.append(base + int(node["l"]))
                right.append(base + int(node["r"]))
                value.append(0.0)
        offsets.append(len(feature))
    # плоские python-списки: индексация списков быстрее numpy-скаляров
    return {
        "feature": feature,
        "threshold": threshold,
        "left": left,
        "right": right,
        "value": value,
        "offsets": offsets,
        "tree_starts": offsets[:-1],
        "baseline": float(model["baseline"]),
    }


def flat_predict_logit(flat, values):
    total = flat["baseline"]
    feature, threshold = flat["feature"], flat["threshold"]
    left, right, value = flat["left"], flat["right"], flat["value"]
    for index in flat["tree_starts"]:
        while True:
            name = feature[index]
            if name < 0:
                total += value[index]
                break
            current = values[name]
            # как sklearn: NaN идёт влево
            index = left[index] if (current != current) or current <= threshold[index] \
                else right[index]
    return total


def flat_predict(flat, values):
    logit = flat_predict_logit(flat, values)
    if logit > 35:
        return 1.0
    if logit < -35:
        return 0.0
    return 1.0 / (1.0 + np.exp(-logit))


def quantize_thresholds(model, grid=256.0):
    result = {"baseline": model["baseline"], "trees": []}
    for tree in model["trees"]:
        nodes = []
        for node in tree:
            item = dict(node)
            if not node["leaf"]:
                item["t"] = round(round(node["t"] * grid) / grid, 8)
            nodes.append(item)
        result["trees"].append(nodes)
    return result


def model_speed(iterations=12):
    cache = ROOT / "artifacts/cache_v14/local_real.npz"
    model_path = ROOT / "artifacts/local_hgb_v14.json"
    if not cache.exists() or not model_path.exists():
        return {"status": "нет кэша признаков или модели"}
    data = np.load(cache, allow_pickle=True)
    X = data["X"][:600]
    model = json.loads(model_path.read_text(encoding="utf-8"))
    flat = flatten_model(model)
    quantized = quantize_thresholds(model)

    def bench(fn):
        started = time.perf_counter()
        for _ in range(iterations):
            for row in X:
                fn(row)
        elapsed = time.perf_counter() - started
        return elapsed / (iterations * len(X))

    dict_seconds = bench(lambda row: predict_trees(model, list(row)))
    flat_seconds = bench(lambda row: flat_predict(flat, row))
    quant_seconds = bench(lambda row: predict_trees(quantized, list(row)))

    sample = X[:: max(1, len(X) // 64)]
    base = np.asarray([predict_trees(model, list(row)) for row in sample])
    quant = np.asarray([predict_trees(quantized, list(row)) for row in sample])
    fast = np.asarray([flat_predict(flat, row) for row in sample])
    return {
        "status": "ok",
        "rows": int(len(X)),
        "candidates_per_second": {
            "list_of_dicts": round(1.0 / dict_seconds),
            "flat_int16_float32": round(1.0 / flat_seconds),
            "quantized_thresholds_1_256": round(1.0 / quant_seconds),
        },
        "us_per_candidate": {
            "list_of_dicts": round(dict_seconds * 1e6, 2),
            "flat_int16_float32": round(flat_seconds * 1e6, 2),
            "quantized_thresholds_1_256": round(quant_seconds * 1e6, 2),
        },
        "speedup_flat_vs_dict": round(dict_seconds / flat_seconds, 2) if flat_seconds else None,
        "max_abs_delta_quantized_vs_float": float(np.max(np.abs(base - quant))),
        "max_abs_delta_flat_vs_float": float(np.max(np.abs(base - fast))),
        "flat_bytes": int(sum(sys.getsizeof(array) for array in
                              (flat["feature"], flat["threshold"], flat["left"],
                               flat["right"], flat["value"], flat["offsets"]))),
        "json_bytes": len(json.dumps(model)),
        "note": "int8/int4/ONNX применимы только к весам трансформера; весов нет — "
                "квантуем то, что реально исполняется в пакете (деревья)",
    }


def transformer_probe():
    """Проверка «режима с весами»: есть ли чем инициализировать RuBERT."""
    model_dir = ROOT / "models/rubert-tiny2"
    files = sorted(item.name for item in model_dir.iterdir()) if model_dir.exists() else []
    weight_files = [name for name in files
                    if name.endswith((".bin", ".safetensors", ".pt", ".onnx"))]
    probe = {
        "model_dir": str(model_dir.relative_to(ROOT)),
        "files": files,
        "weight_files": weight_files,
        "torch_cuda": None,
        "load_attempt": None,
    }
    try:
        import torch
        probe["torch_cuda"] = bool(torch.cuda.is_available())
    except Exception as error:  # pragma: no cover
        probe["torch_cuda"] = f"torch недоступен: {error}"
    if weight_files:
        try:
            from transformers import AutoModel
            model = AutoModel.from_pretrained(str(model_dir))
            probe["load_attempt"] = f"ok, параметров {sum(p.numel() for p in model.parameters())}"
        except Exception as error:
            probe["load_attempt"] = f"ошибка: {type(error).__name__}: {error}"
    else:
        probe["load_attempt"] = ("невозможно: в каталоге только config.json и токенизатор "
                                 "(2,8 MB), файла весов нет; huggingface.co недоступен "
                                 "(разрешены github/codeload/api.github/npm/pypi), зеркал с "
                                 "pytorch_model.bin на GitHub не найдено")
    probe["quantization"] = ("int8/int4/ONNX-квантование не выполнимо: нет весов, "
                             "квантовать нечего")
    return probe


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", default="artifacts/speed_v14.json")
    args = parser.parse_args(argv)

    rows, checks = lexical_speed()
    report = {
        "rule": "замеры на этой машине (CPU, 1 поток), без кэширования между вызовами; "
                "договор: 0.03 с/1k знаков, 50k ≤ 1 с, 2M ≤ 60 с",
        "runtime": "spanver (чистый Python, без torch/sklearn)",
        "lexical": {"rows": rows, "targets": TARGETS, "checks": checks,
                    "meets_all": all(checks.values())},
        "model_inference": model_speed(),
        "weights_mode": transformer_probe(),
    }
    (ROOT / args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                    encoding="utf-8")
    for row in rows:
        print(row, flush=True)
    print("checks", checks, flush=True)
    print("model", json.dumps(report["model_inference"], ensure_ascii=False)[:400], flush=True)


if __name__ == "__main__":
    main()
