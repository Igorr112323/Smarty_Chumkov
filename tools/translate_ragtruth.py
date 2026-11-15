import argparse
import json
import random
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from spanver.ragtruth import chunks_from, context_text, doc_id, load_jsonl, rebuild, select_pairs, src_sha1, spans_from_labels

MAX_RPS = 5.0
BACKOFF = (1, 2, 4, 8, 16)
TIMEOUT = 20
GOOGLE_URL = "https://translate.googleapis.com/translate_a/single"
DEEPL_URL = "https://api-free.deepl.com/v2/translate"
YANDEX_URL = "https://translate.api.cloud.yandex.net/translate/v2/translate"


class NetworkBlocked(RuntimeError):
    pass


class EchoTranslator:
    name = "echo"

    def translate(self, text, source="en", target="ru"):
        return text


class GoogleTranslator:
    name = "google"

    def __init__(self):
        self.last = 0.0

    def _call(self, text):
        query = urllib.parse.urlencode({"client": "gtx", "sl": "en", "tl": "ru", "dt": "t", "q": text})
        request = urllib.request.Request(f"{GOOGLE_URL}?{query}", headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return "".join(part[0] for part in payload[0] if part and part[0])

    def translate(self, text, source="en", target="ru"):
        delay = 1.0 / MAX_RPS
        wait = delay - (time.time() - self.last)
        if wait > 0:
            time.sleep(wait)
        for attempt, pause in enumerate((0,) + BACKOFF):
            if pause:
                time.sleep(pause)
            try:
                value = self._call(text)
                self.last = time.time()
                return value
            except urllib.error.HTTPError as error:
                if error.code == 429:
                    continue
                raise
            except urllib.error.URLError as error:
                raise NetworkBlocked(f"сеть недоступна: {error.reason}") from error
        raise RuntimeError("исчерпаны попытки перевода")


class DeepLTranslator:
    name = "deepl"

    def __init__(self, key):
        self.key = key

    def translate(self, text, source="en", target="ru"):
        data = urllib.parse.urlencode({"text": text, "source_lang": "EN", "target_lang": "RU"}).encode("utf-8")
        request = urllib.request.Request(
            DEEPL_URL, data=data, headers={"Authorization": f"DeepL-Auth-Key {self.key}", "Content-Type": "application/x-www-form-urlencoded"}
        )
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return payload["translations"][0]["text"]


class YandexTranslator:
    name = "yandex"

    def __init__(self, key, folder):
        self.key = key
        self.folder = folder

    def translate(self, text, source="en", target="ru"):
        body = json.dumps({"sourceLanguageCode": "en", "targetLanguageCode": "ru", "texts": [text], "folderId": self.folder}).encode("utf-8")
        request = urllib.request.Request(
            YANDEX_URL, data=body, headers={"Authorization": f"Api-Key {self.key}", "Content-Type": "application/json"}
        )
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return payload["translations"][0]["text"]


class NllbTranslator:
    name = "nllb"

    def __init__(self, path):
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        torch.set_num_threads(4)
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(str(path), local_files_only=True)
        self.model.eval()
        self.bos = self.tokenizer.convert_tokens_to_ids("rus_Cyrl")

    def translate(self, text, source="en", target="ru"):
        self.tokenizer.src_lang = "eng_Latn"
        batch = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with self.torch.no_grad():
            output = self.model.generate(**batch, forced_bos_token_id=self.bos, num_beams=1, max_length=512)
        return self.tokenizer.batch_decode(output, skip_special_tokens=True)[0]


class Cache:
    def __init__(self, path):
        self.path = Path(path)
        self.items = {}
        if self.path.exists():
            with open(self.path, encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    item = json.loads(line)
                    self.items[item["src_sha1"]] = item["dst"]
        self.handle = open(self.path, "a", encoding="utf-8")

    def key(self, source, text):
        return src_sha1(f"{source}|{text}")

    def get(self, key):
        return self.items.get(key)

    def put(self, key, text, value):
        self.items[key] = value
        self.handle.write(json.dumps({"src_sha1": key, "src": text, "dst": value}, ensure_ascii=False) + "\n")
        self.handle.flush()

    def close(self):
        self.handle.close()


def translate(translator, cache, text, stats):
    if not text:
        return ""
    key = cache.key(translator.name, text)
    cached = cache.get(key)
    if cached is not None:
        stats["cache_hits"] += 1
        return cached
    value = translator.translate(text)
    cache.put(key, text, value)
    stats["translated"] += 1
    stats["chars"] += len(text)
    return value


def back_translate_similarity(translator, cache, texts, sample, seed=0, reverse=None):
    import difflib

    rng = random.Random(seed)
    picked = rng.sample(texts, min(sample, len(texts)))
    scores = []
    for text in picked:
        forward = translate(translator, cache, text, {"cache_hits": 0, "translated": 0, "chars": 0})
        back = reverse.translate(forward) if reverse else forward
        scores.append(difflib.SequenceMatcher(None, text, back).ratio())
    scores.sort()
    middle = scores[len(scores) // 2] if scores else 0.0
    return {"samples": len(scores), "median": round(middle, 4), "min": round(scores[0], 4) if scores else 0.0, "max": round(scores[-1], 4) if scores else 0.0}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--responses", default="Dataset/RAGTruth_en/response.jsonl")
    parser.add_argument("--sources", default="Dataset/RAGTruth_en/source_info.jsonl")
    parser.add_argument("--out", default="artifacts/ragtruth_ru.jsonl")
    parser.add_argument("--negatives-out", default="artifacts/ragtruth_ru_negative.jsonl")
    parser.add_argument("--cache", default="artifacts/translate_cache.jsonl")
    parser.add_argument("--report", default="artifacts/ragtruth_report.json")
    parser.add_argument("--translator", default="google", choices=("google", "deepl", "yandex", "nllb", "echo"))
    parser.add_argument("--nllb-path", default="models/nllb-200-distilled-600M")
    parser.add_argument("--limit", type=int, default=6000)
    parser.add_argument("--negatives", type=int, default=1500)
    parser.add_argument("--qc", type=int, default=100)
    parser.add_argument("--examples", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)

    started = time.perf_counter()
    rows = load_jsonl(Path(args.responses))
    sources = {str(item.get("source_id")): item for item in load_jsonl(Path(args.sources))}
    positives, negatives = select_pairs(rows, args.limit, args.negatives, args.seed)
    unique_sources = {}
    for row, _ in positives:
        source = sources.get(str(row.get("source_id"))) or {}
        context = context_text(source)
        unique_sources[row.get("source_id")] = context
    response_chars = sum(len(row.get("response") or "") for row, _ in positives)
    negative_chars = sum(len(row.get("response") or "") for row in negatives)
    source_chars = sum(len(text) for text in unique_sources.values())
    plan = {
        "positives": len(positives),
        "negatives": len(negatives),
        "spans": sum(len(spans) for _, spans in positives),
        "response_chars": response_chars,
        "negative_chars": negative_chars,
        "unique_sources": len(unique_sources),
        "source_chars": source_chars,
        "total_chars": response_chars + negative_chars + source_chars,
    }
    print(json.dumps(plan, ensure_ascii=False), flush=True)

    if args.self_test:
        args.translator = "echo"
        args.out = "artifacts/ragtruth_ru_selftest.jsonl"
        args.negatives_out = "artifacts/ragtruth_ru_negative_selftest.jsonl"
        args.cache = "artifacts/translate_cache_selftest.jsonl"

    if args.translator == "echo":
        translator = EchoTranslator()
    elif args.translator == "google":
        translator = GoogleTranslator()
    elif args.translator == "deepl":
        translator = DeepLTranslator(need_env("DEEPL_KEY"))
    elif args.translator == "yandex":
        translator = YandexTranslator(need_env("YANDEX_KEY"), need_env("YANDEX_FOLDER"))
    else:
        translator = NllbTranslator(Path(args.nllb_path))

    cache = Cache(args.cache)
    stats = {"cache_hits": 0, "translated": 0, "chars": 0}
    report = {"plan": plan, "translator": translator.name, "self_test": bool(args.self_test), "started": time.strftime("%Y-%m-%dT%H:%M:%S")}
    try:
        documents = {}
        for key, context in unique_sources.items():
            documents[key] = translate(translator, cache, context, stats)
        out_rows = []
        out_negatives = []
        for row, spans in positives:
            chunks = chunks_from(row.get("response") or "", spans)
            translated = [translate(translator, cache, chunk["text"], stats) for chunk in chunks]
            text, new_spans = rebuild(chunks, translated)
            context = documents.get(row.get("source_id"), "")
            out_rows.append(
                {
                    "pair_id": f"ragtruth-{row.get('id')}",
                    "doc_id": doc_id(unique_sources.get(row.get("source_id")) or ""),
                    "document_text": context,
                    "model_output": text,
                    "spans": new_spans,
                    "label_type": (spans[0]["label_type"] if spans else ""),
                    "source": "ragtruth_ru" if not args.self_test else "ragtruth_ru_selftest",
                    "source_id": row.get("source_id"),
                }
            )
        for row in negatives:
            text = translate(translator, cache, row.get("response") or "", stats)
            out_negatives.append(
                {
                    "pair_id": f"ragtruth-{row.get('id')}",
                    "doc_id": doc_id(unique_sources.get(row.get("source_id")) or ""),
                    "document_text": documents.get(row.get("source_id"), ""),
                    "model_output": text,
                    "spans": [],
                    "label_type": "",
                    "source": "ragtruth_ru" if not args.self_test else "ragtruth_ru_selftest",
                    "source_id": row.get("source_id"),
                }
            )
    except NetworkBlocked as error:
        report.update({"error": str(error), "status": "сеть недоступна", "stats": stats})
        report["elapsed_s"] = round(time.perf_counter() - started, 1)
        cache.close()
        Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"ОСТАНОВ: {error}", flush=True)
        print(f"переведено до останова: {stats['translated']} кусков, кэш сохранён", flush=True)
        return 3
    cache.close()

    with open(args.out, "w", encoding="utf-8") as handle:
        for row in out_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with open(args.negatives_out, "w", encoding="utf-8") as handle:
        for row in out_negatives:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    out_of_bounds = 0
    empty = 0
    equal_prefix = 0
    ratios = []
    for row, origin in zip(out_rows, positives):
        text = row["model_output"]
        original_text = origin[0].get("response") or ""
        for index, span in enumerate(row["spans"]):
            if span["start"] < 0 or span["end"] >= len(text) or span["start"] > span["end"]:
                out_of_bounds += 1
            if not span["span_text"].strip():
                empty += 1
            source_span = origin[1][index] if index < len(origin[1]) else None
            if source_span:
                source_length = source_span["end"] - source_span["start"] + 1
                if source_length:
                    ratios.append((span["end"] - span["start"] + 1) / source_length)
    ratios.sort()
    examples = []
    for row in out_rows[: args.examples]:
        examples.append(
            {
                "pair_id": row["pair_id"],
                "model_output": row["model_output"][:400],
                "spans": [{"start": span["start"], "end": span["end"], "text": span["span_text"], "type": span["label_type"]} for span in row["spans"]],
            }
        )
    report.update(
        {
            "status": "готово",
            "output": args.out,
            "negatives_output": args.negatives_out,
            "written": len(out_rows),
            "written_negatives": len(out_negatives),
            "stats": stats,
            "qc": {
                "bounds_violations": out_of_bounds,
                "empty_span_text": empty,
                "length_ratio_median": round(ratios[len(ratios) // 2], 4) if ratios else None,
                "length_ratio_samples": len(ratios),
                "span_count_written": sum(len(row["spans"]) for row in out_rows),
            },
            "examples": examples,
            "elapsed_s": round(time.perf_counter() - started, 1),
        }
    )
    if args.translator == "echo":
        report["qc"]["note"] = "самопроверка: переводчик echo, метрики качества перевода неприменимы, проверены только координаты"
    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("status", "written", "written_negatives", "qc", "elapsed_s")}, ensure_ascii=False, indent=2), flush=True)
    return 0


def need_env(name):
    import os

    value = os.environ.get(name)
    if not value:
        raise SystemExit(f"нет переменной окружения {name}")
    return value


if __name__ == "__main__":
    raise SystemExit(main())
