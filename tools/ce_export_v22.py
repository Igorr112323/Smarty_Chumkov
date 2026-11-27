"""V22: экспорт обученного cross-encoder (rubert-base-cased) в ONNX + int8 и замер CPU-скорости.

Повторяет шаг 4 v21 (`tools/ce_export_v21.py`, `tools/ce_speed_v21.py`), но для чекпойнта v22:
  * ONNX opset 17 с динамическими осями;
  * dynamic int8-квантование (обязательное требование договора);
  * паритет fp32 ↔ int8 по головам start/end/risk;
  * латентность на прямом проходе и полные бюджеты продукта 1k / 50k / 2M знаков —
    и полный перебор кандидатов, и каскад top-1/3/5 (как в v21).

Скорость измеряется там же, где обучение (раннер GitHub Actions, больше ядер, чем в песочнице),
поэтому в отчёт дополнительно пишется пересчёт на 2 ядра (замеры v21: 4 ядра → 2 ядра ≈ ×1.9).
"""
import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
import sys

sys.path.insert(0, str(ROOT / "tools"))

import numpy as np
import torch
from ce_model_v21 import build_model

# кандидаты и среда без CE — измерены в v21 на тех же синтетических точках (tools/ce_speed_v21.py)
RUNTIME_WITHOUT_CE = {"1k": 0.0093, "50k": 0.238, "2M": 10.77}
CANDIDATES = {"1k": 8, "50k": 12, "2M": 19}
LIMITS = {"1k": 0.03, "50k": 1.0, "2M": 60.0}
TWO_CORE_FACTOR = 1.9


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export(model, encoder_dir, seq_len, onnx_path):
    model.eval()
    batch = 2
    input_ids = torch.ones((batch, seq_len), dtype=torch.long)
    attention_mask = torch.ones((batch, seq_len), dtype=torch.long)
    token_type_ids = torch.zeros((batch, seq_len), dtype=torch.long)
    candidate_mask = torch.ones((batch, seq_len), dtype=torch.bool)
    torch.onnx.export(
        model, (input_ids, attention_mask, token_type_ids, candidate_mask), str(onnx_path),
        input_names=["input_ids", "attention_mask", "token_type_ids", "candidate_mask"],
        output_names=["risk_logit", "start_logits", "end_logits"],
        dynamic_axes={name: {0: "batch", 1: "sequence"} for name in
                      ("input_ids", "attention_mask", "token_type_ids", "candidate_mask")}
        | {name: {0: "batch", 1: "sequence"} for name in ("start_logits", "end_logits")}
        | {"risk_logit": {0: "batch"}},
        opset_version=17, do_constant_folding=True)
    from onnxruntime.quantization import QuantType, quantize_dynamic

    int8_path = Path(str(onnx_path).replace(".onnx", "_int8.onnx"))
    quantize_dynamic(str(onnx_path), str(int8_path), weight_type=QuantType.QInt8)
    return int8_path


def session(path, threads):
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


def latency(sess, seq_len, threads, repeats=7, batch=1):
    """Латентность одного прямого прохода (batch=1, как в методике v21: кандидаты идут поштучно)."""
    rng = np.random.RandomState(0)
    input_ids = rng.randint(1, 1000, size=(batch, seq_len)).astype(np.int64)
    attention = np.ones_like(input_ids)
    token_types = np.zeros_like(input_ids)
    candidate = np.ones_like(input_ids).astype(bool)
    feeds = {"input_ids": input_ids, "attention_mask": attention,
             "token_type_ids": token_types, "candidate_mask": candidate}
    for _ in range(2):
        sess.run(None, feeds)
    timings = []
    for _ in range(repeats):
        started = time.perf_counter()
        sess.run(None, feeds)
        timings.append(time.perf_counter() - started)
    per_item = statistics.median(timings) / batch
    return {"seq_len": seq_len, "batch": batch, "per_forward_seconds": round(per_item, 5),
            "median_seconds": round(statistics.median(timings), 5), "threads": threads}


def budgets(per_forward, factor=1.0):
    totals, cascade = {}, {}
    for label, runtime in RUNTIME_WITHOUT_CE.items():
        ce_seconds = CANDIDATES[label] * per_forward * factor
        total = runtime * factor + ce_seconds
        totals[label] = {"runtime_seconds": round(runtime * factor, 4),
                         "ce_candidates": CANDIDATES[label], "ce_seconds": round(ce_seconds, 4),
                         "total_seconds": round(total, 4), "limit_seconds": LIMITS[label],
                         "fits": bool(total <= LIMITS[label]),
                         "headroom_x": round(LIMITS[label] / max(total, 1e-9), 2)}
        for top_k in (1, 3, 5):
            cascade_total = runtime * factor + top_k * per_forward * factor
            cascade[f"{label}_top{top_k}"] = {
                "total_seconds": round(cascade_total, 4), "limit_seconds": LIMITS[label],
                "fits": bool(cascade_total <= LIMITS[label]),
                "headroom_x": round(LIMITS[label] / max(cascade_total, 1e-9), 2)}
    return totals, cascade


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="artifacts/ce_model_v22.pt")
    parser.add_argument("--encoder", default="models/rubert-base-cased")
    parser.add_argument("--report", default="artifacts/speed_v22.json")
    parser.add_argument("--onnx", default="artifacts/ce_model_v22.onnx")
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--seq-len", type=int, default=192)
    args = parser.parse_args(argv)

    threads = args.threads or max(1, (torch.get_num_threads()))
    torch.set_num_threads(threads)
    report = {"rule": ("v22 шаг 3: экспорт CE (rubert-base) в ONNX + int8 и CPU-бюджеты продукта: "
                       "0.03 с на 1k знаков, 1 с на 50k, 60 с на 2M. Считаются и полный перебор "
                       "кандидатов, и каскад top-1/3/5"),
              "encoder_dir": args.encoder, "threads": threads}

    model = build_model(model_dir=args.encoder)
    state = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state["state_dict"])
    model.eval()
    total_parameters = int(sum(tensor.numel() for tensor in model.parameters()))
    report["model"] = {"checkpoint": args.checkpoint,
                       "checkpoint_bytes": Path(args.checkpoint).stat().st_size,
                       "parameters": total_parameters,
                       "fp32_mb": round(total_parameters * 4 / 1024 ** 2, 2)}

    started = time.time()
    onnx_path = Path(args.onnx)
    int8_path = export(model, args.encoder, args.seq_len, onnx_path)
    report["export"] = {"onnx_bytes": onnx_path.stat().st_size,
                        "onnx_sha256": sha256(onnx_path),
                        "int8_bytes": int8_path.stat().st_size,
                        "int8_sha256": sha256(int8_path),
                        "compression_ratio": round(onnx_path.stat().st_size / int8_path.stat().st_size, 2),
                        "opset": 17, "seconds": round(time.time() - started, 1)}

    # паритет fp32 ↔ int8 на реальном батче оконного размера
    seq_len = args.seq_len
    rng = np.random.RandomState(3)
    feeds = {"input_ids": rng.randint(1, 1000, size=(2, seq_len)).astype(np.int64)}
    feeds["attention_mask"] = np.ones_like(feeds["input_ids"])
    feeds["token_type_ids"] = np.zeros_like(feeds["input_ids"])
    feeds["candidate_mask"] = np.ones_like(feeds["input_ids"]).astype(bool)
    torch_feeds = {name: torch.from_numpy(value) for name, value in feeds.items()}
    with torch.no_grad():
        reference = model(**torch_feeds)
    fp32 = session(onnx_path, threads).run(None, feeds)
    int8 = session(int8_path, threads).run(None, feeds)
    parity = {}
    for name, index in (("risk_logit", 0), ("start_logits", 1), ("end_logits", 2)):
        base = np.asarray(reference[name].numpy(), dtype=np.float64).ravel()
        int8_values = np.asarray(int8[index], dtype=np.float64).ravel()
        fp32_values = np.asarray(fp32[index], dtype=np.float64).ravel()
        corr = None
        if base.std() > 1e-9 and int8_values.std() > 1e-9:
            corr = round(float(np.corrcoef(base, int8_values)[0, 1]), 4)
        parity[name] = {"int8_vs_torch_max_abs": round(float(np.max(np.abs(base - int8_values))), 5),
                        "fp32_vs_torch_max_abs": round(float(np.max(np.abs(base - fp32_values))), 5),
                        "corr_int8_vs_torch": corr}
    report["parity"] = parity

    latencies = {}
    for length in sorted({128, seq_len, 256, 384}):
        latencies[length] = latency(session(int8_path, threads), length, threads)
    report["latency_int8"] = list(latencies.values())

    per_forward = latencies[seq_len]["per_forward_seconds"]
    totals, cascade = budgets(per_forward)
    totals2, cascade2 = budgets(per_forward, TWO_CORE_FACTOR)
    report["totals_runner"] = totals
    report["cascade_runner"] = cascade
    report["totals_2core_estimate"] = totals2
    report["cascade_2core_estimate"] = cascade2
    report["two_core_factor"] = TWO_CORE_FACTOR

    fits_runner = all(item["fits"] for item in cascade.values())
    fits_2core = all(item["fits"] for item in cascade2.values())
    report["verdict"] = {
        "per_forward_seconds_seq": per_forward,
        "cascade_all_points_fit_runner": fits_runner,
        "cascade_all_points_fit_2core_estimate": fits_2core,
        "note": ("каскад top-5: на раннере " + ("влезает" if fits_runner else "не влезает") +
                 ", в пересчёте на 2 ядра " + ("влезает" if fits_2core else "не влезает")),
    }
    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"latency": per_forward, "export": report["export"], "verdict": report["verdict"],
                      "cascade_2core": cascade2}, ensure_ascii=False))
    return report


if __name__ == "__main__":
    main()
