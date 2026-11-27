"""V21 шаг 4.3: CPU-скорость cross-encoder (int8 ONNX) против лимитов договора §4.1.

Меряется честно и по частям:
  1. штатный рантайм `spanver.module.verify` на 1k / 50k / 2M знаков (без изменений, контроль);
  2. извлечение кандидатов (`candidate_spans`) для ответов разной длины;
  3. латентность int8-ONNX cross-encoder (batch 1) на длинах окна 128 / 192 / 256 / 384;
  4. итог: `runtime + n_кандидатов × латентность` против лимитов 0.03 с / 1 с / 60 с.

Документ в cross-encoder идёт префиксом (200 токенов) и кэшируется на пару, поэтому цена считается
по числу кандидатов ответа, а не по длине документа.
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import numpy as np

from spanver.module import candidate_spans, verify

SENTENCE = ("Согласно отчёту ведомства, рост промышленного производства в регионе составил "
            "три целых две десятых процента за квартал. ")
LIMITS = {"1k": 0.03, "50k": 1.0, "2M": 60.0}
POINTS = {"1k": 1_000, "50k": 50_000, "2M": 2_000_000}
ANSWER_CHARS = {"1k": 160, "50k": 320, "2M": 520}


def make_document(chars):
    text = (SENTENCE * (chars // len(SENTENCE) + 1))[:chars]
    return text


def make_answer(chars, document):
    """Ответ из документных предложений + одно «вставленное» утверждение (как в реальном кейсе)."""
    sentences = [part.strip() for part in document.split(".") if len(part.strip()) > 40][:6]
    answer = ". ".join(sentences) + ". "
    addition = ("В термосфере Юпитера происходит довольно много интересных процессов, "
                "именно здесь планета теряет излучением значительную часть своего тепла.")
    answer = (answer + addition)
    return answer[:chars] if chars < len(answer) else answer


def time_verify(answer, document, repeats=2):
    values = []
    verify(answer, document)  # прогрев
    for _ in range(repeats):
        started = time.perf_counter()
        verify(answer, document)
        values.append(time.perf_counter() - started)
    return min(values)


def bench_session(session, seq_len, repeats=5, batch=1):
    input_ids = np.ones((batch, seq_len), dtype=np.int64)
    attention = np.ones((batch, seq_len), dtype=np.int64)
    types = np.zeros((batch, seq_len), dtype=np.int64)
    types[:, seq_len // 2:] = 1
    feeds = {"input_ids": input_ids, "attention_mask": attention, "token_type_ids": types}
    session.run(None, feeds)  # прогрев
    values = []
    for _ in range(repeats):
        started = time.perf_counter()
        session.run(None, feeds)
        values.append(time.perf_counter() - started)
    return {"seq_len": seq_len, "batch": batch, "median_seconds": round(statistics.median(values), 4),
            "min_seconds": round(min(values), 4)}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", default="artifacts/speed_v21.json")
    parser.add_argument("--onnx-int8", default="artifacts/ce_model_v21_int8.onnx")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--doc-tokens", type=int, default=200)
    args = parser.parse_args(argv)

    import onnxruntime as ort
    options = ort.SessionOptions()
    options.intra_op_num_threads = args.threads
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(ROOT / args.onnx_int8), sess_options=options,
                                   providers=["CPUExecutionProvider"])

    latency = [bench_session(session, seq_len) for seq_len in (128, 192, 256, 384)]
    latency_batch = [bench_session(session, seq_len, batch=8) for seq_len in (192, 384)]
    per_candidate = {row["seq_len"]: row["median_seconds"] for row in latency}

    runtime, candidates, totals = {}, {}, {}
    for label, chars in POINTS.items():
        document = make_document(chars)
        answer = make_answer(ANSWER_CHARS[label], document)
        spans = list(candidate_spans(answer))
        counts = {"candidates": len(spans), "answer_chars": len(answer),
                  "document_chars": len(document)}
        candidates[label] = counts
        runtime[label] = round(time_verify(answer, document), 4)
        seq_len = 128 if counts["candidates"] <= 12 else (192 if counts["candidates"] <= 20 else 256)
        ce_seconds = counts["candidates"] * per_candidate[seq_len]
        total = runtime[label] + ce_seconds
        totals[label] = {"runtime_seconds": runtime[label],
                         "ce_candidates": counts["candidates"],
                         "ce_seq_len": seq_len,
                         "ce_seconds": round(ce_seconds, 4),
                         "total_seconds": round(total, 4),
                         "limit_seconds": LIMITS[label],
                         "fits": bool(total <= LIMITS[label]),
                         "headroom_x": round(LIMITS[label] / max(total, 1e-9), 2)}

    cascade = {}
    for label in POINTS:
        for top_k in (1, 3, 5):
            ce_seconds = top_k * per_candidate[128 if top_k <= 5 else 192]
            total = runtime[label] + ce_seconds
            cascade[f"{label}_top{top_k}"] = {
                "total_seconds": round(total, 4), "limit_seconds": LIMITS[label],
                "fits": bool(total <= LIMITS[label]),
                "headroom_x": round(LIMITS[label] / max(total, 1e-9), 2)}

    long_answer = make_document(50_000)
    long_spans = list(candidate_spans(long_answer))
    stress = {"answer_chars": len(long_answer), "candidates": len(long_spans),
              "ce_seconds": round(len(long_spans) * per_candidate[256], 2),
              "limit_seconds": LIMITS["50k"],
              "fits": bool(len(long_spans) * per_candidate[256] <= LIMITS["50k"])}

    report = {
        "rule": "шаг 4.3 v21: CPU-скорость int8 cross-encoder против лимитов §4.1 (0.03 с/1k, "
                "50k <= 1 с, 2M <= 60 с) на 2 ядрах",
        "setup": {"threads": args.threads, "provider": "CPUExecutionProvider",
                  "onnx_int8": args.onnx_int8,
                  "bytes": (ROOT / args.onnx_int8).stat().st_size,
                  "doc_tokens": args.doc_tokens,
                  "note": "документ подаётся префиксом фиксированной длины, кандидат — целиком; "
                          "цена считается по числу кандидатов ответа"},
        "latency_per_forward": latency,
        "latency_batch8": latency_batch,
        "candidates": candidates,
        "runtime_without_ce": runtime,
        "cascade_top_k": cascade,
        "stress_long_answer_50k": stress,
        "totals": totals,
        "verdict": {
            "stress_note": "при ответе на 50k знаков число кандидатов растёт до сотен, и cross-encoder "
                           "перестаёт вписываться в лимит даже int8",
            "all_points_fit": all(item["fits"] for item in totals.values()),
            "per_point": {label: item["fits"] for label, item in totals.items()},
            "note": "оценка оптимистична для качества: документ урезан до 200 токенов, поэтому "
                    "cross-encoder видит только префикс; при полном документе цена выросла бы в "
                    "десятки раз и лимиты §4.1 не выполнялись бы",
        },
    }
    (ROOT / args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                    encoding="utf-8")
    print(json.dumps({"latency": latency, "totals": totals,
                      "all_fit": report["verdict"]["all_points_fit"]}, ensure_ascii=False)[:1400],
          flush=True)


if __name__ == "__main__":
    main()
