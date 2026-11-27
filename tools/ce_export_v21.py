"""V21 шаг 4.1–4.2: экспорт cross-encoder в ONNX (opset 17) + int8-квантование.

Экспортируется `CrossEncoderSpan` (энкодер rubert-tiny2 + головы start/end/risk) с динамическими
осями (batch, sequence). Затем динамическая int8-квантование весов через `onnxruntime.quantization`
и проверка паритета fp32 vs int8 (корреляция risk-логитов и средняя абсолютная разница).

Важно: в этой песочнице нет GPU, поэтому веса голов НЕ обучены (энкодер — реальные предобученные
веса rubert-tiny2, головы — случайная инициализация). Цель шага — доказать и измерить цепочку
«чекпойнт → ONNX → int8 → onnxruntime», а не заявить метрики.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import numpy as np
import torch

from ce_model_v21 import build_model, count_parameters, load_tokenizer


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ExportWrapper(torch.nn.Module):
    def __init__(self, model, seq_len):
        super().__init__()
        self.model = model
        self.seq_len = seq_len

    def forward(self, input_ids, attention_mask, token_type_ids):
        mask = torch.zeros_like(input_ids, dtype=torch.long)
        mask[:, self.model.encoder.config.max_position_embeddings // 2:] = 0
        # кандидат определяется по token_type_ids == 1
        mask = (token_type_ids > 0).long()
        outputs = self.model(input_ids, attention_mask, token_type_ids, mask)
        return outputs["risk_logit"], outputs["start_logits"], outputs["end_logits"]


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", default="artifacts/ce_export_v21.json")
    parser.add_argument("--onnx", default="artifacts/ce_model_v21.onnx")
    parser.add_argument("--onnx-int8", default="artifacts/ce_model_v21_int8.onnx")
    parser.add_argument("--checkpoint", default="artifacts/ce_model_v21.pt")
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args(argv)

    model = build_model()
    checkpoint = ROOT / args.checkpoint
    loaded = None
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location="cpu")
        model.load_state_dict(state["state_dict"])
        loaded = str(checkpoint)
    model.eval()
    wrapper = ExportWrapper(model, args.seq_len).eval()

    tokenizer = load_tokenizer()
    sample = tokenizer("Документ о климате и погоде.", "Ответ содержит утверждение.",
                       return_tensors="pt", padding="max_length", truncation=True,
                       max_length=args.seq_len)
    sample["token_type_ids"] = torch.zeros_like(sample["input_ids"])
    sample["token_type_ids"][:, 10:] = 1

    onnx_path = ROOT / args.onnx
    started = time.time()
    torch.onnx.export(
        wrapper, (sample["input_ids"], sample["attention_mask"], sample["token_type_ids"]),
        str(onnx_path), input_names=["input_ids", "attention_mask", "token_type_ids"],
        output_names=["risk_logit", "start_logits", "end_logits"],
        dynamic_axes={"input_ids": {0: "batch", 1: "sequence"},
                      "attention_mask": {0: "batch", 1: "sequence"},
                      "token_type_ids": {0: "batch", 1: "sequence"},
                      "risk_logit": {0: "batch"},
                      "start_logits": {0: "batch", 1: "sequence"},
                      "end_logits": {0: "batch", 1: "sequence"}},
        opset_version=args.opset, do_constant_folding=True)
    export_seconds = time.time() - started

    import onnx
    from onnxruntime.quantization import QuantType, quantize_dynamic

    onnx_model = onnx.load(str(onnx_path))
    opset = onnx_model.opset_import[0].version
    int8_path = ROOT / args.onnx_int8
    started = time.time()
    quantize_dynamic(str(onnx_path), str(int8_path), weight_type=QuantType.QInt8)
    quantize_seconds = time.time() - started

    import onnxruntime as ort
    session_fp32 = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    session_int8 = ort.InferenceSession(str(int8_path), providers=["CPUExecutionProvider"])
    feeds = {name: sample[name].numpy() for name in ("input_ids", "attention_mask", "token_type_ids")}
    fp32_out = session_fp32.run(None, feeds)
    int8_out = session_int8.run(None, feeds)
    parity = {}
    for index, name in enumerate(("risk_logit", "start_logits", "end_logits")):
        left = np.asarray(fp32_out[index]).ravel()
        right = np.asarray(int8_out[index]).ravel()
        corr = None
        if left.size > 1 and np.std(left) > 0 and np.std(right) > 0:
            corr = round(float(np.corrcoef(left, right)[0, 1]), 5)
        parity[name] = {"max_abs_diff": round(float(np.max(np.abs(left - right))), 5),
                        "mean_abs_diff": round(float(np.mean(np.abs(left - right))), 5),
                        "corr": corr}

    report = {
        "rule": "шаг 4 v21: экспорт cross-encoder в ONNX (opset >= 14) и int8-квантование; "
                "продукт использует только ONNX + onnxruntime, torch в рантайм не попадает",
        "model": {"architecture": "CrossEncoderSpan (rubert-tiny2 + start/end/risk)",
                  "parameters": count_parameters(model), "checkpoint_loaded": loaded,
                  "heads_trained": bool(loaded), "note": None if loaded else
                  "GPU отсутствует → головы не обучены; экспорт проверяет цепочку, не качество"},
        "onnx": {"path": args.onnx, "bytes": onnx_path.stat().st_size,
                 "sha256": sha256(onnx_path), "opset": int(opset),
                 "export_seconds": round(export_seconds, 2),
                 "input_names": ["input_ids", "attention_mask", "token_type_ids"],
                 "dynamic_axes": ["batch", "sequence"]},
        "int8": {"path": args.onnx_int8, "bytes": int8_path.stat().st_size,
                 "sha256": sha256(int8_path), "weight_type": "QInt8 (dynamic)",
                 "quantize_seconds": round(quantize_seconds, 2),
                 "compression_ratio": round(onnx_path.stat().st_size / int8_path.stat().st_size, 2)},
        "parity_fp32_vs_int8": parity,
        "sizes_total_mb": {"onnx_fp32": round(onnx_path.stat().st_size / 1024 ** 2, 2),
                           "onnx_int8": round(int8_path.stat().st_size / 1024 ** 2, 2)},
        "verdict": "экспорт и int8-квантование работают; веса голов обучены" if loaded else
                   "экспорт и int8-квантование работают, но модель НЕ обучена (нет GPU) — "
                   "цепочка готова, качество не заявляется",
    }
    (ROOT / args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                    encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("onnx", "int8", "parity_fp32_vs_int8")},
                     ensure_ascii=False)[:900], flush=True)
    print(report["verdict"], flush=True)


if __name__ == "__main__":
    main()
