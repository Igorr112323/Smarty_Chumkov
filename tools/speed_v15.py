"""V15 шаг 4: скорость LM-инференса на CPU (договор §4.1: 0.03 с/1k, 50k ≤ 1 с, 2M ≤ 60 с).

Меряется только энкодер (без классификатора — он чужой), в двух реализациях:

  * `torch_fp32` — та, что использовалась для признаков (transformers, float32);
  * `onnx_int8`  — исходный `model_quantized.onnx` через onnxruntime с добавленными выходами
    Softmax (вероятности внимания) и последнего LayerNorm (скрытые состояния). Так проверяем,
    спасает ли int8-квантование бюджет скорости.

Документ режется на окна по 510 wordpiece (batch 8), как в `tools/lm_features_v15.py`.
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

SIZES = [(1_000, 1_000), (50_000, 50_000), (2_000_000, 2_000_000)]
LIMITS = {1_000: 0.03, 50_000: 1.0, 2_000_000: 60.0}
WINDOW_TOKENS = 510
BATCH = 8
ATTENTION_TENSORS = [
    "/bert/encoder/layer.0/attention/self/Softmax_output_0",
    "/bert/encoder/layer.1/attention/self/Softmax_output_0",
    "/bert/encoder/layer.2/attention/self/Softmax_output_0",
]
LAST_HIDDEN = "/bert/encoder/layer.2/output/LayerNorm/Add_1_output_0"


def russian_text():
    path = ROOT / "artifacts/gold_v3.jsonl"
    buffer = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            pair = json.loads(line)
            if pair.get("split") != "train":
                continue
            buffer.append((pair.get("document_text") or "") + "\n" + (pair.get("model_output") or ""))
            if sum(len(chunk) for chunk in buffer) > 60_000:
                break
    return "\n".join(buffer)


class TorchRunner:
    name = "torch_fp32"

    def __init__(self):
        import torch
        from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast
        self.torch = torch
        self.tokenizer = BertTokenizerFast.from_pretrained(str(ROOT / "models/rubert-tiny2-deq"))
        config = json.loads((ROOT / "models/rubert-tiny2-deq/config.json").read_text(encoding="utf-8"))
        self.model = BertForSequenceClassification(BertConfig(**config, num_labels=9))
        self.model.load_state_dict(
            torch.load(ROOT / "models/rubert-tiny2-deq/pytorch_model.bin", map_location="cpu"),
            strict=True)
        self.model.eval()

    def run(self, texts):
        torch = self.torch
        batch = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True,
                               max_length=512)
        with torch.no_grad():
            out = self.model.bert(**batch, output_attentions=True, output_hidden_states=True)
        return out.attentions[0].numpy(), out.hidden_states[-1].numpy()


class OnnxRunner:
    name = "onnx_int8"

    def __init__(self):
        import onnx
        import onnxruntime as ort
        from transformers import BertTokenizerFast
        self.tokenizer = BertTokenizerFast.from_pretrained(str(ROOT / "models/rubert-tiny2-deq"))
        source = ROOT / "tools/external/rubert-turbo-payments/models/onnx_model/model_quantized.onnx"
        model = onnx.load(str(source))
        for name in ATTENTION_TENSORS + [LAST_HIDDEN]:
            model.graph.output.append(onnx.ValueInfoProto(name=name))
        self.session = ort.InferenceSession(model.SerializeToString(),
                                            providers=["CPUExecutionProvider"])
        self.attention_names = ATTENTION_TENSORS

    def run(self, texts):
        batch = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True,
                               max_length=512)
        feeds = {key: batch[key].numpy() for key in
                 ("input_ids", "attention_mask", "token_type_ids")}
        values = self.session.run(None, feeds)
        return values[-4:-1], values[-1]


def measure(runner, tokenizer, base_text, target_chars):
    """Готовим окна заранее (токенизация вне таймера), время — только форварды."""
    text = (base_text * (target_chars // max(len(base_text), 1) + 1))[:target_chars]
    tokenize_started = time.time()
    ids = tokenizer(text, add_special_tokens=False, truncation=False)["input_ids"]
    tokenize_seconds = time.time() - tokenize_started
    windows_ids = [ids[index:index + WINDOW_TOKENS]
                   for index in range(0, len(ids), WINDOW_TOKENS)]
    texts = [tokenizer.decode(row) for row in windows_ids]
    runner.run(texts[:2])  # прогрев
    batches = [texts[index:index + BATCH] for index in range(0, len(texts), BATCH)]
    started = time.time()
    for batch in batches:
        runner.run(batch)
    elapsed = time.time() - started
    return {"chars": len(text), "tokens": len(ids), "windows": len(windows_ids),
            "seconds": round(elapsed, 4),
            "seconds_per_1k_chars": round(elapsed / max(len(text), 1) * 1000, 4),
            "tokenize_seconds": round(tokenize_seconds, 3)}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", default="artifacts/speed_v15.json")
    args = parser.parse_args(argv)

    text = russian_text()
    report = {"rule": "договор §4.1: LM-инференс 1k ≤ 0.03 с, 50k ≤ 1 с, 2M ≤ 60 с; возм.: "
                      "ONNX int8 / rubert-tiny вместо base / чанк-батч",
              "hardware": "CPU x86-64 без GPU, RAM < 8 ГБ в песочнице",
              "window_tokens": WINDOW_TOKENS, "batch": BATCH,
              "limits": {"1k": 0.03, "50k": 1.0, "2M": 60.0},
              "runners": {}}

    for cls in (TorchRunner, OnnxRunner):
        runner = cls()
        entry = {}
        for _, target in SIZES:
            row = measure(runner, runner.tokenizer, text, target)
            key = {1_000: "1k", 50_000: "50k", 2_000_000: "2M"}[target]
            row["limit_seconds"] = LIMITS[target]
            row["pass"] = (row["seconds_per_1k_chars"] <= 0.03) if target == 1_000 else (row["seconds"] <= LIMITS[target])
            entry[key] = row
            print(runner.name, key, row, flush=True)
        entry["pass_all"] = all(entry[key]["pass"] for key in ("1k", "50k", "2M"))
        report["runners"][runner.name] = entry

    best = min(report["runners"].items(), key=lambda item: item[1]["2M"]["seconds_per_1k_chars"])
    report["best"] = {"runner": best[0], "2M_seconds": best[1]["2M"]["seconds"],
                      "2M_limit": 60.0, "pass_all": best[1]["pass_all"]}
    report["verdict"] = ("speed ok" if best[1]["pass_all"] else
                         f"blocker: даже лучший вариант ({best[0]}) не укладывает 2M в 60 с "
                         f"({best[1]['2M']['seconds']} с)")
    (ROOT / args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                    encoding="utf-8")
    print(json.dumps(report["best"], ensure_ascii=False), flush=True)
    print(report["verdict"], flush=True)


if __name__ == "__main__":
    main()
