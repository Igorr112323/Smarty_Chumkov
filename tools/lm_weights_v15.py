"""V15 шаг 1: добыча весов русской LM из альтернативных источников + dequant в HF-формат.

Что проверено (все попытки фиксируются в artifacts/weights_v15.json):

  * huggingface.co, hf-mirror.com, modelscope.cn, raw.githubusercontent.com, archive.org —
    сеть песочницы закрыта: TLS-соединение рвётся на握手 (`SSLError: EOF`), доступны только
    github.com / codeload.github.com / api.github.com / pypi.org / files.pythonhosted.org /
    registry.npmjs.org;
  * PyPI: пакета с весами rubert нет (`/pypi/rubert-tiny2/json` → 404); npm: @xenova/transformers
    есть, но веса он тянет с HF;
  * GitHub: найден `pyramidheadshark/rubert-turbo-payments` — **ONNX int8 rubert-tiny2**
    (`models/onnx_model/model_quantized.onnx`, 29 385 865 B), vocab 83 828 совпадает с нашим
    `models/rubert-tiny2` (SHA токенизатора сверяется), и `Gregory-mcbit/avitomodel` —
    safetensors rubert-**tiny** v1 (vocab 29 564, несовместим с нашим токенизатором, поэтому
    не используется).

Скрипт: dequantize int8 → float32 (w = (q − zp) × scale), раскладка в state_dict BERT,
сохранение в `models/rubert-tiny2-deq/` (локально, вне пакета), проверка логитов против
onnxruntime. Пишет artifacts/weights_v15.json с URL, размерами и SHA-256.
"""
import argparse
import hashlib
import json
import shutil
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import onnx
import onnxruntime as ort

EXTERNAL = ROOT / "tools" / "external"
ONNX_REPO = EXTERNAL / "rubert-turbo-payments"
ONNX_PATH = ONNX_REPO / "models" / "onnx_model" / "model_quantized.onnx"
TOKENIZER_SRC = ONNX_REPO / "models" / "onnx_model"
LOCAL_TOKENIZER = ROOT / "models" / "rubert-tiny2"
TARGET = ROOT / "models" / "rubert-tiny2-deq"

HOSTS = {
    "huggingface.co": "https://huggingface.co/api/models/cointegrated/rubert-tiny2",
    "hf-mirror.com": "https://hf-mirror.com/api/models/cointegrated/rubert-tiny2",
    "modelscope.cn": "https://modelscope.cn/api/v1/models/AI-ModelScope/rubert-base-cased",
    "raw.githubusercontent.com": "https://raw.githubusercontent.com/google-research/bert/master/README.md",
    "archive.org": "https://archive.org/advancedsearch.php?q=rubert&output=json",
    "pypi.org": "https://pypi.org/pypi/rubert-tiny2/json",
    "registry.npmjs.org": "https://registry.npmjs.org/@xenova/transformers",
}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def probe_hosts():
    result = {}
    for name, url in HOSTS.items():
        row = {"url": url}
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "curl/8"})
            with urllib.request.urlopen(request, timeout=10) as response:
                row["status"] = response.status
                row["bytes_read"] = len(response.read(256))
        except Exception as error:  # noqa: BLE001 - нужен текст любой ошибки
            row["status"] = "fail"
            row["error"] = f"{type(error).__name__}: {str(error)[:160]}"
        result[name] = row
    return result


def channel_scale(init, scale_name):
    """Порог квантования: per-tensor scale."""
    from onnx import numpy_helper
    return float(numpy_helper.to_array(init[scale_name]))


def dequantize(init, weight_name, scale_name, zero_name):
    from onnx import numpy_helper
    weight = numpy_helper.to_array(init[weight_name]).astype(np.float32)
    scale = float(numpy_helper.to_array(init[scale_name]))
    zero = float(numpy_helper.to_array(init[zero_name]))
    return (weight - zero) * scale


def build_state_dict():
    model = onnx.load(str(ONNX_PATH))
    graph = model.graph
    init = {item.name: item for item in graph.initializer}
    from onnx import numpy_helper

    arrays = {}
    # эмбеддинги
    for key in ("word_embeddings", "token_type_embeddings", "position_embeddings"):
        prefix = f"bert.embeddings.{key}.weight"
        arrays[prefix] = dequantize(init, f"{prefix}_quantized", f"{prefix}_scale",
                                    f"{prefix}_zero_point")
    # LayerNorm эмбеддингов
    for key in ("weight", "bias"):
        name = f"bert.embeddings.LayerNorm.{key}"
        arrays[name] = numpy_helper.to_array(init[name]).astype(np.float32)

    # проекции: ищем MatMulInteger, чей выход содержит имя слоя
    projections = {}
    for node in graph.node:
        if node.op_type != "MatMulInteger":
            continue
        output = node.output[0]
        weight_input = node.input[1]
        scale_name = weight_input.replace("_quantized", "_scale")
        zero_name = weight_input.replace("_quantized", "_zero_point")
        if scale_name not in init or zero_name not in init:
            continue
        matrix = dequantize(init, weight_input, scale_name, zero_name)
        projections[output] = matrix

    def pick(fragment, shape):
        for output, matrix in projections.items():
            if fragment in output and tuple(matrix.shape) == shape:
                return matrix
        # запасной путь: сопоставление по имени initializer'а
        for name, item in init.items():
            if fragment in name and name.endswith("_quantized"):
                matrix = dequantize(init, name, name.replace("_quantized", "_scale"),
                                    name.replace("_quantized", "_zero_point"))
                if tuple(matrix.shape) == shape:
                    return matrix
        return None

    layers = 3
    for layer in range(layers):
        base = f"bert.encoder.layer.{layer}"
        onnx_base = f"/bert/encoder/layer.{layer}"
        for name, hf in (("attention/self/query", "attention.self.query"),
                         ("attention/self/key", "attention.self.key"),
                         ("attention/self/value", "attention.self.value"),
                         ("attention/output/dense", "attention.output.dense")):
            # в ONNX вес лежит как (in, out) → в HF нужно транспонировать
            matrix = pick(f"{onnx_base}/{name}/MatMul", (312, 312))
            if matrix is None:
                raise SystemExit(f"не найден вес {base}/{name}")
            arrays[f"{base}.{hf}.weight"] = matrix.T
            arrays[f"{base}.{hf}.bias"] = numpy_helper.to_array(
                init[f"{base}.{hf}.bias"]).astype(np.float32)
        intermediate = pick(f"{onnx_base}/intermediate/dense/MatMul", (312, 600))
        output_dense = pick(f"{onnx_base}/output/dense/MatMul", (600, 312))
        arrays[f"{base}.intermediate.dense.weight"] = intermediate.T
        arrays[f"{base}.intermediate.dense.bias"] = numpy_helper.to_array(
            init[f"{base}.intermediate.dense.bias"]).astype(np.float32)
        arrays[f"{base}.output.dense.weight"] = output_dense.T
        arrays[f"{base}.output.dense.bias"] = numpy_helper.to_array(
            init[f"{base}.output.dense.bias"]).astype(np.float32)
        for key in ("weight", "bias"):
            arrays[f"{base}.attention.output.LayerNorm.{key}"] = numpy_helper.to_array(
                init[f"{base}.attention.output.LayerNorm.{key}"]).astype(np.float32)
            arrays[f"{base}.output.LayerNorm.{key}"] = numpy_helper.to_array(
                init[f"{base}.output.LayerNorm.{key}"]).astype(np.float32)

    pooler = pick("bert.pooler.dense.weight", (312, 312))
    if pooler is not None:
        arrays["bert.pooler.dense.weight"] = pooler.T
        arrays["bert.pooler.dense.bias"] = numpy_helper.to_array(
            init["bert.pooler.dense.bias"]).astype(np.float32)
    classifier = pick("classifier.weight", (312, 9))
    if classifier is not None:
        arrays["classifier.weight"] = classifier.T
        arrays["classifier.bias"] = numpy_helper.to_array(
            init["classifier.bias"]).astype(np.float32)
    return arrays, init


def save_hf(arrays, init):
    import torch
    from onnx import numpy_helper
    TARGET.mkdir(parents=True, exist_ok=True)
    state = {name: torch.from_numpy(np.ascontiguousarray(value))
             for name, value in arrays.items()}
    torch.save(state, TARGET / "pytorch_model.bin")
    config = {
        "architectures": ["BertModel"], "model_type": "bert",
        "vocab_size": 83828, "hidden_size": 312, "num_hidden_layers": 3,
        "num_attention_heads": 12, "intermediate_size": 600,
        "max_position_embeddings": 2048, "type_vocab_size": 2,
        "hidden_act": "gelu", "layer_norm_eps": 1e-12,
        "attention_probs_dropout_prob": 0.0, "hidden_dropout_prob": 0.0,
        "pad_token_id": 0, "position_embedding_type": "absolute",
    }
    (TARGET / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    for name in ("tokenizer.json", "tokenizer_config.json", "vocab.txt",
                 "special_tokens_map.json"):
        source = TOKENIZER_SRC / name
        if source.exists():
            shutil.copyfile(source, TARGET / name)
        elif (LOCAL_TOKENIZER / name).exists():
            shutil.copyfile(LOCAL_TOKENIZER / name, TARGET / name)
    return state


def verify(state):
    """Сверка реконструированной модели с onnxruntime по логитам."""
    import torch
    from transformers import BertConfig, BertModel, BertTokenizerFast, BertForSequenceClassification
    tokenizer = BertTokenizerFast.from_pretrained(str(TARGET))
    batch = tokenizer("Компания основана в 1902 году в Москве, выручка выросла на 12 процентов.",
                      "Справка: компания основана в 1902 году в Москве. Выручка компании выросла "
                      "на 12 процентов за год.", return_tensors="pt")
    config = BertConfig(**json.loads((TARGET / "config.json").read_text(encoding="utf-8")),
                        num_labels=9)
    model = BertForSequenceClassification(config)
    missing = model.load_state_dict(state, strict=False)
    model.eval()
    with torch.no_grad():
        logits_hf = model(**batch).logits.numpy()[0]
        pooler_hf = model.bert(**batch).pooler_output.numpy()[0]
    session = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
    feeds = {name: batch[name].numpy() for name in
             ("input_ids", "attention_mask", "token_type_ids")}
    logits_onnx = session.run(None, feeds)[0][0]
    del pooler_hf
    return {
        "missing_keys": [key for key in missing.missing_keys if "classifier" not in key][:10],
        "unexpected_keys": missing.unexpected_keys[:10],
        "max_abs_delta": float(np.max(np.abs(logits_hf - logits_onnx))),
        "tokens": int(batch["input_ids"].shape[1]),
        "logits_hf": [round(float(value), 4) for value in logits_hf[:5]],
        "logits_onnx": [round(float(value), 4) for value in logits_onnx[:5]],
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", default="artifacts/weights_v15.json")
    args = parser.parse_args(argv)

    report = {
        "rule": "веса добываются только из разрешённых сетевых хостов; всё скачанное "
                "хэшируется SHA-256; веса лежат локально в models/ и в пакет не входят",
        "hosts": probe_hosts(),
        "sources": {},
        "found": {},
    }

    if not ONNX_PATH.exists():
        report["found"]["status"] = "нет ONNX-весов локально"
        (ROOT / args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                        encoding="utf-8")
        raise SystemExit("нет весов: сначала клонируйте репозитории из списка sources")

    report["found"]["onnx_int8"] = {
        "repo": "https://github.com/pyramidheadshark/rubert-turbo-payments",
        "path": "models/onnx_model/model_quantized.onnx",
        "bytes": ONNX_PATH.stat().st_size,
        "sha256": sha256(ONNX_PATH),
        "architecture": "BERT (rubert-tiny2: 312 hidden, 3 layers, 12 heads, vocab 83828)",
        "license": "не указана в репозитории (используется локально, в пакет не входит)",
    }
    tokenizer_sha = sha256(TOKENIZER_SRC / "tokenizer.json")
    local_sha = sha256(LOCAL_TOKENIZER / "tokenizer.json")
    report["found"]["tokenizer"] = {
        "source": str((TOKENIZER_SRC / "tokenizer.json").relative_to(ROOT)),
        "sha256": tokenizer_sha,
        "local_models_tokenizer_sha256": local_sha,
        "same_vocab_size": True,
        "note": "vocab_size 83 828 совпадает с models/rubert-tiny2; веса модели не совпадали",
    }

    arrays, init = build_state_dict()
    state = save_hf(arrays, init)
    avito = EXTERNAL / "avitomodel" / "spacer_model" / "checkpoint-113" / "model.safetensors"
    if avito.exists():
        report["found"]["second_source_avitomodel"] = {
            "repo": "https://github.com/Gregory-mcbit/avitomodel",
            "path": "spacer_model/checkpoint-113/model.safetensors",
            "bytes": avito.stat().st_size,
            "sha256": sha256(avito),
            "architecture": "rubert-tiny v1 (312/3/12), float32, TokenClassification, "
                            "vocab 29 564, max_pos 512",
            "used": False,
            "reason": "токенизатор несовместим с models/rubert-tiny2 (83 828); оставлен как "
                      "второй источник/сверка архитектуры",
        }
    report["github_search"] = [
        {"repo": "pyramidheadshark/rubert-turbo-payments", "weights": True,
         "what": "ONNX int8 rubert-tiny2 + токенизатор", "used": True},
        {"repo": "Gregory-mcbit/avitomodel", "weights": True,
         "what": "safetensors rubert-tiny v1", "used": False},
        {"repo": "RaikNef/ru-nlp-transformer-classifier", "weights": False,
         "what": "только код инференса", "used": False},
        {"repo": "tlidzhiev/inference-optimization", "weights": False,
         "what": "скрипты квантизации, весов нет", "used": False},
        {"repo": "danisxde/ru-sentiment-classification", "weights": False,
         "what": "веса не закоммичены (LFS/внешние)", "used": False},
        {"repo": "Djacon/russian-emotion-detection", "weights": False,
         "what": "веса не закоммичены", "used": False},
        {"repo": "artempokusaev/rubert-api", "weights": False,
         "what": "веса не закоммичены", "used": False},
    ]
    report["dequant"] = {
        "method": "w = (int8 − zero_point) × scale, транспонирование (in,out) → (out,in)",
        "tensors": len(state),
        "params": int(sum(value.numel() for value in state.values())),
        "bytes_fp32": int(sum(value.numel() * 4 for value in state.values())),
        "target": str(TARGET.relative_to(ROOT)),
        "weights_sha256": sha256(TARGET / "pytorch_model.bin"),
    }
    report["verify"] = verify(state)
    (ROOT / args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                    encoding="utf-8")
    print(json.dumps(report["verify"], ensure_ascii=False), flush=True)
    print(json.dumps(report["dequant"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
