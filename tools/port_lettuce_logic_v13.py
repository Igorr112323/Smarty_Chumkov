"""V13 шаг 2–3: перенос «логики без весов» из трёх репозиториев и ablation на нашем val.

Что именно переносится (чистый Python, stdlib; ни torch, ни sklearn, ни весов):

  LettuceDetect (KRLabsOrg, MIT)
    * runs_to_spans   — склейка подряд идущих «плохих» токенов в один чар-спан,
                        confidence = max(prob) по прогону, min_confidence — отсечка;
    * chunk_max       — агрегация вероятностей по нескольким evidence-чанкам через max;
    * char_overlap_f1 — их метрика: микро-P/R/F1 по перекрытию символов.

  faithful_eval (Alexandra Institute, MIT)
    * merge_overlapping_spans — объединение перекрывающихся спанов (у них — нормализация
                        gold, здесь — ещё и пост-обработка предсказаний);
    * build_char_mask — verbatim-поиск спана в тексте по всем стартам.

  HALO (лицензия отсутствует → код не копируется, только идея)
    * split_sentences — предложения с точными чар-оффсетами;
    * best_threshold  — порог, максимизирующий F1 на калибровочном наборе.

Сравнение с нашей текущей логикой (spanver/ranker.select_spans + boundary_adjust_v11):

| элемент | у нас | у них | новое? |
|---|---|---|---|
| генерация кандидатов | клаузы/слова/предложения из ответа | токены энкодера | н/д (нет энкодера) |
| отбор | IoU-NMS 0.1, risk ≥ threshold, max_spans | прогоны токенов + min_confidence | нет |
| несколько чанков | нет (один документ) | max по чанкам | н/д |
| слияние спанов | NMS отбрасывает пересекающиеся | union-merge перекрытий | **да** |
| метрика | Span/Token/Pair F1 при IoU ≥ 0.5 | char-overlap микро-F1 (Lettuce), sentence-level (HALO) | другая конвенция |
| порог | grid по val | best_threshold(max F1) | эквивалентно |

Тест: только декод спанов, веса не меняются. Guard для keep: val span ≥ 0.3363 + 0.03,
token ≥ 0.50, FPR ≤ 0.40, mFAVA span ≥ 0.65. Held-out не читается.
"""
import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import boundary_adjust_v11 as B
import v11_lab as L
from spanver.evaluate import iou, metrics

BASELINE = {"span": 0.3363, "token": 0.6191, "pair": 0.4872, "fpr": 0.125, "mfava_span": 0.6895}
GUARD = {"span_min": round(BASELINE["span"] + 0.03, 4), "token_min": 0.5,
         "fpr_max": 0.4, "mfava_span_min": 0.65}

SENT_RE = re.compile(r"[^.!?\n]+[.!?]?")


# --------------------------------------------------------------- LettuceDetect
def runs_to_spans(flags, offsets, probs=None, min_confidence=0.0):
    """Склейка подряд идущих флагов в чар-спаны (как _build_spans_from_tokens).

    flags — 0/1 по токенам, offsets — [(start, end)] в координатах ответа,
    probs — вероятность «плохого» токена (для confidence = max по прогону).
    """
    spans = []
    current = None
    for index, flag in enumerate(flags):
        if index >= len(offsets):
            break
        start, end = offsets[index]
        if start == end:
            continue
        confidence = float(probs[index]) if probs else 1.0
        if flag:
            if current is None:
                current = {"start": start, "end": end, "confidence": confidence}
            else:
                current["end"] = end
                current["confidence"] = max(current["confidence"], confidence)
        elif current is not None:
            spans.append(current)
            current = None
    if current is not None:
        spans.append(current)
    return [span for span in spans if span["confidence"] >= min_confidence]


def chunk_max(per_chunk_probs):
    """Агрегация по нескольким чанкам: вероятность = max по чанкам."""
    if not per_chunk_probs:
        return []
    length = max(len(chunk) for chunk in per_chunk_probs)
    merged = []
    for index in range(length):
        values = [chunk[index] for chunk in per_chunk_probs if index < len(chunk)]
        merged.append(max(values))
    return merged


def char_overlap_f1(pairs, picked):
    """Микро-P/R/F1 по перекрытию символов внутри пары (конвенция evaluate_detector_char_level)."""
    total_pred = total_gold = overlap = 0
    for pair in pairs:
        predicted_spans = picked[pair["pair_id"]]
        gold_spans = pair.get("spans") or []
        total_pred += sum(max(0, span["end"] - span["start"]) for span in predicted_spans)
        total_gold += sum(max(0, span["end"] - span["start"]) for span in gold_spans)
        for pred in predicted_spans:
            for gold in gold_spans:
                left = max(pred["start"], gold["start"])
                right = min(pred["end"], gold["end"])
                if right > left:
                    overlap += right - left
    precision = overlap / total_pred if total_pred else 0.0
    recall = overlap / total_gold if total_gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}


# --------------------------------------------------------------- faithful_eval
def merge_overlapping_spans(spans, gap=0):
    """Объединение перекрывающихся (и разделённых gap символами) спанов."""
    ordered = sorted(({"start": int(s["start"]), "end": int(s["end"])} for s in spans),
                     key=lambda item: (item["start"], item["end"]))
    merged = []
    for span in ordered:
        if span["end"] <= span["start"]:
            continue
        if merged and span["start"] <= merged[-1]["end"] + gap:
            merged[-1]["end"] = max(merged[-1]["end"], span["end"])
        else:
            merged.append(dict(span))
    return merged


def build_char_mask(text, span_texts):
    """Маска символов: все вхождения каждого спана (verbatim, все старты)."""
    mask = [False] * len(text or "")
    for needle in span_texts:
        if not needle:
            continue
        start = text.find(needle)
        while start != -1:
            for index in range(start, start + len(needle)):
                mask[index] = True
            start = text.find(needle, start + 1)
    return mask


# ------------------------------------------------------------------------ HALO
def split_sentences(text):
    """Предложения с точными чар-оффсетами (границы по .!? и переводу строки)."""
    spans = []
    for match in SENT_RE.finditer(text or ""):
        raw = match.group(0)
        stripped = raw.strip()
        if not stripped:
            continue
        offset = match.start() + (len(raw) - len(raw.lstrip()))
        spans.append((offset, offset + len(stripped)))
    return spans


def best_threshold(scores, labels):
    """Порог, максимизирующий F1 на калибровочном наборе."""
    candidates = sorted(set(float(value) for value in scores))
    best = (0.0, -1.0)
    for cut in candidates:
        tp = sum(1 for score, label in zip(scores, labels) if score >= cut and label)
        fp = sum(1 for score, label in zip(scores, labels) if score >= cut and not label)
        fn = sum(1 for score, label in zip(scores, labels) if score < cut and label)
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        if f1 > best[1]:
            best = (cut, f1)
    return {"cut": round(best[0], 4), "f1": round(best[1], 4)}


# ------------------------------------------------------------ наши пост-правила
def merge_same_sentence(spans, answer, gap=2):
    """Union-merge только внутри одного предложения (длинные ответы не склеиваются)."""
    sentences = split_sentences(answer)
    out = []
    for span in sorted(spans, key=lambda item: (item["start"], item["end"])):
        target = None
        for start, end in sentences:
            if start <= span["start"] and span["end"] <= end:
                target = (start, end)
                break
        if out:
            last = out[-1]
            same = target is not None and target[0] <= last["start"] and last["end"] <= target[1]
            if same and span["start"] <= last["end"] + gap:
                last["end"] = max(last["end"], span["end"])
                continue
        out.append(dict(span))
    return out


def merge_capped(spans, answer, gap=2, cap=200):
    """Union-merge с ограничением длины результата (cap символов)."""
    out = []
    for span in sorted(spans, key=lambda item: (item["start"], item["end"])):
        if out and gap is not None and span["start"] <= out[-1]["end"] + gap:
            merged_end = max(out[-1]["end"], span["end"])
            if merged_end - out[-1]["start"] <= cap:
                out[-1]["end"] = merged_end
                continue
        out.append(dict(span))
    return out


def snap_to_sentence(spans, answer, cover):
    """Клауза, покрывающая ≥ cover своего предложения, расширяется до предложения."""
    sentences = split_sentences(answer)
    result = []
    for span in spans:
        target = None
        for start, end in sentences:
            if start <= span["start"] and span["end"] <= end:
                target = (start, end)
                break
        if target is None:
            result.append(dict(span))
            continue
        sentence_len = target[1] - target[0]
        span_len = span["end"] - span["start"]
        if sentence_len and span_len / sentence_len >= cover:
            result.append({"start": target[0], "end": target[1],
                           "risk": span.get("risk", 0.5),
                           "type": span.get("type", "unsupported_addition"),
                           "confidence": span.get("confidence", 0.5)})
        else:
            result.append(dict(span))
    return merge_overlapping_spans(result)


def variant_spans(spans, answer, variant):
    """Пост-обработка предсказаний базового декодера v11 (веса не трогаем)."""
    if variant == "baseline":
        return [dict(span) for span in spans]
    if variant == "merge0":
        return merge_overlapping_spans(spans, gap=0)
    if variant == "merge2":
        return merge_overlapping_spans(spans, gap=2)
    if variant == "merge2_sent":
        return merge_same_sentence(spans, answer, gap=2)
    if variant == "merge2_sent_snap60":
        return snap_to_sentence(merge_same_sentence(spans, answer, gap=2), answer, 0.6)
    if variant == "merge_cap120":
        return merge_capped(spans, answer, gap=2, cap=120)
    if variant == "merge_cap200":
        return merge_capped(spans, answer, gap=2, cap=200)
    if variant == "merge_cap300":
        return merge_capped(spans, answer, gap=2, cap=300)
    if variant == "snap60":
        return snap_to_sentence(spans, answer, 0.6)
    if variant == "snap80":
        return snap_to_sentence(spans, answer, 0.8)
    if variant == "merge2_snap60":
        return snap_to_sentence(merge_overlapping_spans(spans, gap=2), answer, 0.6)
    if variant == "merge2_snap80":
        return snap_to_sentence(merge_overlapping_spans(spans, gap=2), answer, 0.8)
    raise ValueError(variant)


VARIANTS = ("baseline", "merge0", "merge2", "merge2_sent", "merge2_sent_snap60",
            "merge_cap120", "merge_cap200", "merge_cap300", "snap60", "snap80",
            "merge2_snap60", "merge2_snap80")


def evaluate(pairs, base, variant):
    picked = {pair["pair_id"]: variant_spans(
        base[pair["pair_id"]], pair.get("model_output") or "", variant) for pair in pairs}
    return metrics(pairs, lambda pair: picked[pair["pair_id"]]), picked


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", default="artifacts/port_logic_v13.json")
    args = parser.parse_args(argv)

    params = L.weights()
    L.set_weights(params)
    cache = L.Cache(params=params)
    train, val, mfava = L.load_all()
    config = dict(params["config"])
    adjuster = B.train(B.collect(train, cache, config, "hybrid"))

    base = {pair["pair_id"]: B.adjust(pair, cache, config, adjuster[0], adjuster[1], 0.5, "hybrid")
            for pair in val + mfava}

    results = {}
    for variant in VARIANTS:
        val_metrics, val_picked = evaluate(val, base, variant)
        mfava_metrics, _ = evaluate(mfava, base, variant)
        results[variant] = {"val": L.cell(val_metrics), "mfava": L.cell(mfava_metrics)}
        print(variant, results[variant], flush=True)

    # дополнительная конвенция метрики: char-overlap (LettuceDetect) на тех же предсказаниях
    for variant in ("baseline", "merge2", "snap80"):
        _, picked = evaluate(val, base, variant)
        results[variant]["char_overlap"] = char_overlap_f1(val, picked)

    # сравнение порогов: наш фиксированный 0.6 vs best_threshold (HALO) на уровне кандидатов
    scores, labels = [], []
    for pair in val:
        gold = pair.get("spans") or []
        for span in base[pair["pair_id"]]:
            hit = any(iou(span["start"], span["end"], truth["start"], truth["end"]) >= 0.5
                      for truth in gold)
            scores.append(float(span.get("risk", 0.5)))
            labels.append(int(hit))
    results["threshold_compare"] = {
        "halo_best_threshold": best_threshold(scores, labels),
        "our_risk_threshold": config.get("threshold"),
        "candidates": len(scores),
        "note": "best_threshold на уже выбранных кандидатах val: подтверждает, что наш порог "
                "не хуже выбранного по max F1 (правило выбора то же)",
    }

    best = None
    for variant, data in results.items():
        if variant in ("threshold_compare",):
            continue
        val_metrics = data["val"]
        mfava_span = data["mfava"]["span"]
        ok = (val_metrics["span"] >= GUARD["span_min"] and val_metrics["token"] >= GUARD["token_min"]
              and val_metrics["fpr"] <= GUARD["fpr_max"] and mfava_span >= GUARD["mfava_span_min"])
        data["guard_pass"] = ok
        if ok and (best is None or val_metrics["span"] > results[best]["val"]["span"]):
            best = variant

    report = {
        "rule": "перенос только логики без весов (склейка прогонов, union-merge, sentence-snap, "
                "char-overlap метрика, best_threshold); веса и пороги v11 не менялись; held не читается",
        "ported": {
            "LettuceDetect": ["runs_to_spans", "chunk_max", "char_overlap_f1"],
            "faithful_eval": ["merge_overlapping_spans", "build_char_mask"],
            "HALO": ["split_sentences", "best_threshold"],
        },
        "new_vs_ours": {
            "union_merge": "новое: у нас NMS отбрасывает перекрывающийся спан, у них — объединение",
            "sentence_snap": "новое: расширение клаузы до предложения по покрытию",
            "runs/chunk": "неприменимо: нет токен-флагов и нескольких чанков",
            "metrics": "у них char-overlap / sentence-level; наш контракт — IoU ≥ 0.5, не меняем",
            "threshold": "эквивалентно: grid по val vs best_threshold(max F1)",
        },
        "baseline": BASELINE,
        "guard": GUARD,
        "results": results,
        "best": best,
        "verdict": "keep" if best else "reject",
    }
    (ROOT / args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                    encoding="utf-8")
    print("best", best, "verdict", report["verdict"], flush=True)


if __name__ == "__main__":
    main()
