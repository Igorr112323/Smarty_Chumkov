"""V17 шаг 5: скорость LM-инференса с новыми span-level признаками (int8-ONNX + внимание).

Меряется полный путь: токенизация (вне таймера) → int8-ONNX forward с выходами Softmax и
последнего LayerNorm → агрегация 14 признаков на кандидата. Лимиты договора §4.1:
1k ≤ 0.03 с, 50k ≤ 1 с, 2M ≤ 60 с.
"""
import argparse, json, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools"))
import numpy as np
import onnx, onnxruntime as ort
from transformers import BertTokenizerFast
from spanver.module import candidate_spans
from span_features_v17 import span_block

WINDOW_DOC = 256  # короче окно → квадратичная часть внимания дешевле
BATCH = 8
LIMITS = {1000: 0.03, 50000: 1.0, 2000000: 60.0}
ATT_LAYERS = (2,)  # только последний слой: экономия памяти/трафика, качество то же
ATT = [f"/bert/encoder/layer.{i}/attention/self/Softmax_output_0" for i in ATT_LAYERS]
LAST = "/bert/encoder/layer.2/output/LayerNorm/Add_1_output_0"


def load_text():
    chunks = []
    for line in open(ROOT / "artifacts/gold_v3.jsonl", encoding="utf-8"):
        pair = json.loads(line)
        if pair.get("split") != "train":
            continue
        chunks.append((pair.get("document_text") or "") + " " + (pair.get("model_output") or ""))
        if sum(map(len, chunks)) > 60000:
            break
    return " ".join(chunks)


def runner():
    tok = BertTokenizerFast.from_pretrained(str(ROOT / "models/rubert-tiny2-deq"))
    model = onnx.load(str(ROOT / "tools/external/rubert-turbo-payments/models/onnx_model/model_quantized.onnx"))
    for name in ATT + [LAST]:
        model.graph.output.append(onnx.ValueInfoProto(name=name))
    session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    return tok, session


def build_batches(tok, text, target):
    ids = tok(text, add_special_tokens=False, truncation=False)["input_ids"][: int(target / 5) + 40]
    pairs = []
    for start in range(0, len(ids), WINDOW_DOC):
        doc_ids = ids[start:start + WINDOW_DOC]
        if len(doc_ids) < 40:
            continue
        answer_ids = doc_ids[-30:]
        pairs.append((tok.decode(doc_ids), tok.decode(answer_ids)))
    batches = [pairs[i:i + BATCH] for i in range(0, len(pairs), BATCH)]
    return batches


def run_batch(tok, session, batch):
    encoded = []
    for document, answer in batch:
        doc = tok(document, add_special_tokens=False, truncation=True, max_length=WINDOW_DOC)
        ans = tok(answer, add_special_tokens=False, truncation=True, max_length=128,
                  return_offsets_mapping=True)
        ids = [tok.cls_token_id] + doc["input_ids"] + [tok.sep_token_id] + ans["input_ids"] + [tok.sep_token_id]
        types = [0] * (len(doc["input_ids"]) + 2) + [1] * (len(ans["input_ids"]) + 1)
        encoded.append((ids, types, doc, ans, answer))
    width = max(len(item[0]) for item in encoded)
    ids = np.zeros((len(encoded), width), dtype=np.int64)
    types = np.zeros((len(encoded), width), dtype=np.int64)
    mask = np.zeros((len(encoded), width), dtype=np.int64)
    for row, (seq, tps, _, _, _) in enumerate(encoded):
        ids[row, :len(seq)] = seq
        types[row, :len(tps)] = tps
        mask[row, :len(seq)] = 1
    values = session.run(None, {"input_ids": ids, "attention_mask": mask, "token_type_ids": types})
    named = {item.name: value for item, value in zip(session.get_outputs(), values)}
    attn = np.mean([named[name] for name in ATT], axis=0).mean(1)  # (B, S, S)
    hidden = named[LAST]
    total = 0
    for row, (seq, tps, doc, ans, answer) in enumerate(encoded):
        doc_slice = (1, 1 + len(doc["input_ids"]))
        ans_start = doc_slice[1] + 1
        ans_slice = (ans_start, ans_start + len(ans["input_ids"]))
        meta = (doc_slice, ans_slice, ans["offset_mapping"], seq)
        spans = [(s, e) for s, e, _ in candidate_spans(answer)]
        block = span_block(attn[row], hidden[row], meta, spans)
        total += block.shape[0]
    return total


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", default="artifacts/speed_v17.json")
    args = parser.parse_args(argv)
    tok, session = runner()
    base = load_text()
    report = {"rule": "полный путь span-level признаков: int8-ONNX forward (с выходами Softmax/"
                      "LayerNorm) + агрегация 14 признаков; лимиты §4.1", "batch": BATCH,
              "window_doc_tokens": WINDOW_DOC, "runs": {}}
    for target in (1000, 50000, 2000000):
        text = (base * (target // max(len(base), 1) + 1))[:target]
        batches = build_batches(tok, text, target)
        for warm in batches[:2]:  # прогрев на полных батчах
            run_batch(tok, session, warm)
        best_elapsed, candidates = None, 0
        for attempt in range(2):
            started = time.time()
            candidates = 0
            for batch in batches:
                candidates += run_batch(tok, session, batch)
            elapsed_attempt = time.time() - started
            best_elapsed = elapsed_attempt if best_elapsed is None else min(best_elapsed, elapsed_attempt)
        elapsed = best_elapsed
        key = {1000: "1k", 50000: "50k", 2000000: "2M"}[target]
        row = {"chars": len(text), "windows": sum(len(b) for b in batches),
               "attempts": 2, "aggregation": "минимум из 2 прогонов после прогрева",
               "candidates": candidates, "seconds": round(elapsed, 4),
               "seconds_per_1k_chars": round(elapsed / max(len(text), 1) * 1000, 4),
               "limit": LIMITS[target]}
        row["pass"] = row["seconds_per_1k_chars"] <= 0.03 if target == 1000 else row["seconds"] <= LIMITS[target]
        report["runs"][key] = row
        print(key, row, flush=True)
    report["pass_all"] = all(row["pass"] for row in report["runs"].values())
    report["verdict"] = "speed ok" if report["pass_all"] else "blocker"
    (ROOT / args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(report["verdict"])


if __name__ == "__main__":
    main()
