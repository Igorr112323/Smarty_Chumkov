"""ШАГ D: int8 + ONNX на один сид.

Что делает:
 1. Экспортирует FeatureModel (rubert-tiny2 + 5 признаков) в ONNX;
 2. Квантизует динамически в int8 (onnxruntime.quantization, QUInt8 для весов);
 3. Сверяет вероятности с torch на тех же парах — модель после квантизации
    обязана остаться той же моделью, иначе замер скорости бессмыслен;
 4. Меряет полный конвейер (признаки + forward + gate + гистерезис + expand)
    теми же длинами, что стадия 0, и печатает мс на 1000 знаков документа.

Сравнение идёт с теми же числами, что в artifacts/speed52.json (один сид):
838 знаков 23.335126, 1000 знаков 32.116921, 5000 знаков 28.167185.
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

from spanver import lexfeat
from spanver.evaluate import read_jsonl
from spanver.snap import snapped
from spanver.span_detector import choose_chunks
from tools.nested_cv35 import predict
from tools.speed_bench_batched import build_pairs

TOP = 4
GATE = 0.45
THETA = 0.50
MIN_CHARS = 2


def load_trained():
    """Модель V5 seed 0 с весами: экспорт и torch-эталон должны быть одной моделью."""
    import torch

    from tools.predict_frozen38 import load_state

    manifest = json.loads((ROOT / "weights/final/manifest.json").read_text(encoding="utf-8"))
    entry = manifest["checkpoints"][0]
    tokenizer, model = lexfeat.load_feature_model(ROOT / "models/rubert-tiny2", labels=2, seed=entry["seed"])
    model.base.config._attn_implementation = "eager"
    load_state(torch, model, ROOT / "weights/final" / f"seed_{entry['seed']}", entry)
    return tokenizer, model


def export(path, tokenizer=None, model=None):
    """Экспортирует ПЕРЕДАННУЮ модель.

    Раньше функция загружала свежий rubert-tiny2 и экспортировала его: без весов
    V5. Из-за этого ONNX-путь давал token F1 0.3296 против 0.4541 у torch, и
    разница была списана на int8. Теперь модель обязательна.
    """
    import torch

    if model is None:
        raise ValueError("экспортировать нужно обученную модель, а не свежую")
    model.base.config._attn_implementation = "eager"
    model.eval()
    sample = tokenizer("документ", "ответ", return_tensors="pt", truncation="only_first", max_length=64)
    width = sample["input_ids"].shape[1]
    dummy_ids = torch.ones((1, width), dtype=torch.long)
    dummy_mask = torch.ones((1, width), dtype=torch.long)
    dummy_features = torch.zeros((1, width, lexfeat.FEATURE_COUNT), dtype=torch.float)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        (dummy_ids, dummy_mask, dummy_features),
        str(target),
        input_names=["input_ids", "attention_mask", "features"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "length"},
            "attention_mask": {0: "batch", 1: "length"},
            "features": {0: "batch", 1: "length"},
            "logits": {0: "batch", 1: "length"},
        },
        opset_version=17,
        do_constant_folding=True,
    )
    return tokenizer, model, target


def quantize(source, target):
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(str(source), str(target), weight_type=QuantType.QUInt8)


def make_runner(path):
    import numpy as np
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])

    def run(ids, mask, features):
        values = session.run(None, {
            "input_ids": np.asarray(ids, dtype=np.int64),
            "attention_mask": np.asarray(mask, dtype=np.int64),
            "features": np.asarray(features, dtype=np.float32),
        })
        logits = values[0]
        shifted = logits - logits.max(axis=-1, keepdims=True)
        exp = np.exp(shifted)
        return exp / exp.sum(axis=-1, keepdims=True)

    return run


def batched_probabilities(tokenizer, pair, run, top=TOP, max_length=512):
    """Повторяет lexfeat.pair_probabilities, но forward идёт через ONNX."""
    import numpy as np

    from spanver.span_detector import encode_pair

    answer = pair.get("model_output") or ""
    if not answer:
        return None
    rows = lexfeat.answer_features(answer, pair.get("document_text") or "")
    chunks = choose_chunks(tokenizer, pair, top=top, max_length=max_length)
    encoded = [item for item in (encode_pair(tokenizer, chunk["text"], answer, max_length) for chunk in chunks) if item]
    if not encoded:
        return None
    width = max(len(item["input_ids"]) for item in encoded)
    pad = tokenizer.pad_token_id
    ids = [item["input_ids"] + [pad] * (width - len(item["input_ids"])) for item in encoded]
    mask = [[1] * len(item["input_ids"]) + [0] * (width - len(item["input_ids"])) for item in encoded]
    features = []
    for item in encoded:
        values = [[0.0] * lexfeat.FEATURE_COUNT for _ in range(width)]
        for index, start, end in item["answer_tokens"]:
            values[index] = lexfeat.aligned(rows, start, end + 1)
        features.append(values)
    probabilities = run(ids, mask, features)[:, :, 1]
    best = {}
    for position, item in enumerate(encoded):
        for index, start, end in item["answer_tokens"]:
            value = float(probabilities[position][index])
            previous = best.get(start)
            if previous is None or value > previous[2]:
                best[start] = (start, end, value)
    return {"answer": answer, "pieces": [best[key] for key in sorted(best)]}


def main():
    import numpy as np

    parser = argparse.ArgumentParser()
    parser.add_argument("--lengths", default="838,1000,5000")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--out", default="artifacts/quant52.json")
    args = parser.parse_args()
    lengths = [int(x) for x in args.lengths.split(",")]
    gold = read_jsonl(ROOT / "artifacts/ours33.jsonl")
    sample = build_pairs(gold, 838)

    tokenizer, model = load_trained()
    export("artifacts/onnx52/model.onnx", tokenizer, model)
    quantize("artifacts/onnx52/model.onnx", "artifacts/onnx52/model_int8.onnx")
    runner = make_runner("artifacts/onnx52/model_int8.onnx")

    # сверка вероятностей с torch: без неё замер скорости ничего не значит
    import torch
    torch.set_num_threads(2)
    deltas = []
    for pair in sample[:5]:
        torch_value = lexfeat.pair_probabilities(tokenizer, model, pair, top=TOP)
        onnx_value = batched_probabilities(tokenizer, pair, runner, top=TOP)
        if not torch_value or not onnx_value:
            continue
        for left, right in zip(torch_value["pieces"], onnx_value["pieces"]):
            deltas.append(abs(left[2] - right[2]))
    max_delta = max(deltas) if deltas else None
    mean_delta = sum(deltas) / len(deltas) if deltas else None

    rows = []
    for length in lengths:
        pairs = build_pairs(gold, length)
        chars = sum(len(p["document_text"]) for p in pairs)
        repeats = []
        for _ in range(args.repeats):
            started = time.perf_counter()
            for pair in pairs:
                value = batched_probabilities(tokenizer, pair, runner, top=TOP)
                pieces = value["pieces"] if value else []
                if not pieces:
                    continue
                answer = pair["model_output"]
                spans = []
                if max(piece[2] for piece in pieces) >= GATE:
                    spans = predict({"answer": answer, "pieces": pieces}, THETA)
                snapped(spans, answer, "expand")
            repeats.append((time.perf_counter() - started) * 1000.0 / (chars / 1000.0))
        rows.append({
            "length": length,
            "documents": len(pairs),
            "doc_chars": chars,
            "ms_per_1000_doc_median": statistics.median(repeats),
            "ms_per_1000_doc_min": min(repeats),
            "ms_per_1000_doc_max": max(repeats),
            "samples": repeats,
        })
        print(json.dumps(rows[-1], ensure_ascii=False), flush=True)

    report = {
        "status": "готово: полный конвейер на int8/ONNX, один сид",
        "model": "rubert-tiny2 + 5 признаков, seed 0, QUInt8 dynamic",
        "top_chunks": TOP,
        "gate": GATE,
        "theta": THETA,
        "threads": 2,
        "torch_threads_baseline": 4,
        "platform": platform.platform(),
        "cpu_count": __import__("os").cpu_count(),
        "probability_max_abs_delta_vs_torch": max_delta,
        "probability_mean_abs_delta_vs_torch": mean_delta,
        "baseline_one_seed_ms_per_1000": {"838": 23.335126, "1000": 32.116921, "5000": 28.167185},
        "limit_ms_per_1000": 30,
        "limit_838": 25,
        "rows": rows,
    }
    (ROOT / args.out).write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"max_delta": max_delta, "mean_delta": mean_delta}, ensure_ascii=False))


if __name__ == "__main__":
    main()
