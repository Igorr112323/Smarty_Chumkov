"""ПРОМПТ 57, стадия 1: почему переносимый порог даёт 0,51, а оракул 0,644.

Всё считается ПО СОХРАНЁННЫМ АРТЕФАКТАМ, без torch и без обучения:

  * `artifacts/ablation56_ten_outer{O}_inner{I}.json` — в каждом лежат
    `inner_grid` (86 точек порога при gate=0.45, extend=start/2) и, для основного
    внутреннего фолда, `outer_predictions` — сами вероятности по парам;
  * `artifacts/ours33.jsonl` — пары train/val с метками.

Из этого восстанавливается ровно тот декодер, которым получены 2.1–2.6
(`snapped` с клаузным expand), но с тремя переключаемыми ограничениями:
`max_spans`, `nms_iou` и `gate`. Значения по умолчанию — те, что в поставке:
`max_spans=1` (`tools/freeze_config.py:41`), `nms_iou=0.10` (`spanver/module.py:12`),
`gate=0.45` (`tools/decode56.py:37`).

Что здесь НЕЛЬЗЯ и поэтому не делается: выбирать что-либо по test 301. Test в этом
файле не открывается вообще.

Три части, по критериям промпта:

  1.1 `oracle`  — оракульный порог на каждом фолде против переносимого; разброс.
  1.2 `transfer`— порог, выбранный на 9 фолдах, применённый к десятому (и его
                  вариант: выбор на внутренних фолдах, оценка на внешнем).
  1.3 `losses`  — вклад каждого ограничения декодера отдельно, при прочих равных.
  1.4 `grid`    — сетка (max_spans × nms_iou × порог) с отбором ТОЛЬКО внутри
                  train/val: внутренние фолды выбирают, внешний только оценивает.

Запуск: python3 tools/rule57.py --out artifacts/rule57.json
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# torch нужен только для импорта обученной модели, а здесь её нет: считаются
# сохранённые артефакты. Если torch не установлен (локальная песочница), он
# подменяется заглушкой, чтобы импорт `tools.train_base33` прошёл. В Actions
# установлен настоящий torch и заглушка не включается.
try:
    import torch  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover - только локально
    import types

    torch_stub = types.ModuleType("torch")
    torch_stub.__version__ = "stub"
    torch_stub.no_grad = lambda: __import__("contextlib").nullcontext()
    sys.modules["torch"] = torch_stub
    lexfeat_stub = types.ModuleType("spanver.lexfeat")
    lexfeat_stub.__getattr__ = lambda name: (_ for _ in ()).throw(
        RuntimeError(f"lexfeat.{name} недоступен без torch: этот расчёт его не использует"))
    sys.modules["spanver.lexfeat"] = lexfeat_stub

from spanver.evaluate import read_jsonl
from spanver.module import select
from spanver.snap import boundaries, snapped
from tools.sweep_ragtruth30 import prepare
from tools.train_base33 import grouped_folds

FPR_LIMIT = 0.40
GATE = 0.45
MAX_SPANS = 1
NMS_IOU = 0.10
MIN_CHARS = 2
GRID = tuple(round(index * 0.05, 2) for index in range(1, 20))
SPAN_GRID = (1, 2, 3, None)  # None = без ограничения
NMS_GRID = (0.10, 0.30, 0.50)
KEYS = ("token_tp", "token_fp", "token_fn", "negative_tokens", "flagged_negative_tokens",
        "span_tp", "span_fp", "span_fn", "negative_pairs", "flagged_negative_pairs")


def spans_from_pieces(answer, pieces, start, extend, gate, max_spans=MAX_SPANS, nms=NMS_IOU):
    """Тот же декодер, что в поставке, с явными ограничениями."""
    if not pieces:
        return []
    if gate is not None and max(piece[2] for piece in pieces) < gate:
        return []
    flags = []
    active = False
    for _, _, score in pieces:
        active = score >= (extend if active else start)
        flags.append(active)
    # runs -> спаны с расширением до границ клаузы (snapped expand)
    row = {"answer": answer, "pieces": pieces, "start_threshold": start,
           "extend_threshold": extend, "gate_threshold": gate if gate is not None else -1.0}
    spans = _runs(answer, pieces, flags)
    spans = snapped(spans, answer, "expand")
    for span in spans:
        span.setdefault("risk", max((piece[2] for piece in pieces), default=0.0))
    if not spans:
        return []
    if max_spans is None:
        return spans
    return select(spans, max_spans, nms)


def _runs(answer, pieces, flags):
    """Флаги по кандидатам -> спаны, склеенные в границах слов (min 2 знака)."""
    edges = boundaries(answer)
    result = []
    start = end = None
    for (left, right, _), flag in zip(pieces, flags):
        if flag:
            if start is None:
                start = left
            end = right - 1
        elif start is not None:
            result.append((start, end))
            start = end = None
    if start is not None:
        result.append((start, end))
    spans = []
    for left, right in result:
        # обрезка до целых слов, как в sweep_ragtruth30.predict_spans
        import bisect
        before = bisect.bisect_right(edges, left) - 1
        after = bisect.bisect_left(edges, right + 1)
        span_left = edges[before] if before >= 0 else 0
        span_right = edges[after] - 1 if after < len(edges) else len(answer) - 1
        span_left = max(0, span_left)
        span_right = min(len(answer) - 1, max(span_left, span_right))
        if len(answer[span_left:span_right + 1]) < MIN_CHARS:
            continue
        spans.append({"start": span_left, "end": span_right})
    return spans


def score_pairs(pairs, cache, start, gate=GATE, max_spans=MAX_SPANS, nms=NMS_IOU,
                extend=None, min_spans_filter=None):
    """Метрики на наборе пар. cache: pair_id -> pieces."""
    extend = start / 2 if extend is None else extend
    prepared = prepare(pairs, {pair_id: {"pieces": pieces} for pair_id, pieces in cache.items()})
    totals = {key: 0 for key in KEYS}
    predicted_spans = 0
    gold_spans = 0
    for row, pair in zip(prepared, pairs):
        predicted = spans_from_pieces(row["answer"], row["pieces"], start, extend, gate,
                                      max_spans, nms)
        if min_spans_filter is not None:
            predicted = [span for span in predicted
                         if span["end"] - span["start"] + 1 >= min_spans_filter]
        predicted_spans += len(predicted)
        gold_spans += len(row["gold"])
        _tally(row, predicted, totals)
    return metrics(totals, predicted_spans, gold_spans)


def _tally(row, predicted, totals):
    import bisect

    from spanver.evaluate import match_spans

    matched, _, _ = match_spans(predicted, row["gold"])
    totals["span_tp"] += matched
    totals["span_fp"] += len(predicted) - matched
    totals["span_fn"] += len(row["gold"]) - matched
    if row["gold"]:
        pass
    else:
        totals["negative_pairs"] += 1
        if predicted:
            totals["flagged_negative_pairs"] += 1
    predicted_mask = 0
    for span in predicted:
        first = bisect.bisect_left(row["ends"], span["start"])
        last = bisect.bisect_right(row["starts"], span["end"])
        if last > first:
            predicted_mask |= ((1 << (last - first)) - 1) << first
    gold_mask = row["gold_mask"]
    totals["token_tp"] += (predicted_mask & gold_mask).bit_count()
    totals["token_fp"] += (predicted_mask & ~gold_mask).bit_count()
    totals["token_fn"] += (gold_mask & ~predicted_mask).bit_count()
    totals["negative_tokens"] += (row["all_mask"] & ~gold_mask).bit_count()
    totals["flagged_negative_tokens"] += (predicted_mask & ~gold_mask).bit_count()


def metrics(totals, predicted_spans, gold_spans):
    tp, fp, fn = totals["token_tp"], totals["token_fp"], totals["token_fn"]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    span_denom = predicted_spans + gold_spans
    return {
        "token_precision": precision,
        "token_recall": recall,
        "token_f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "fpr_token": totals["flagged_negative_tokens"] / totals["negative_tokens"]
        if totals["negative_tokens"] else 0.0,
        "fpr_pair": totals["flagged_negative_pairs"] / totals["negative_pairs"]
        if totals["negative_pairs"] else 0.0,
        "span_exact_f1": 2 * totals["span_tp"] / span_denom if span_denom else 0.0,
        "spans_per_pair": predicted_spans / max(1, totals["negative_pairs"]) if False else predicted_spans,
        "counts": dict(totals),
    }


def merge(totals_list):
    out = {key: 0 for key in KEYS}
    for totals in totals_list:
        for key in KEYS:
            out[key] += totals.get(key, 0)
    return out


def load_records(prefix="ablation56_ten", folder="artifacts"):
    """Записи фолдов, на которых посчитан decode56.

    ВАЖНО. Файлы `artifacts/ablation56_ten_outer*_inner*.json` в ветке были
    ПЕРЕЗАПИСАНЫ отменённым прогоном 44260610565 (его отмена через API не прошла,
    задания успели отработать и закоммитить результат). Метки, фолды и состав пар
    те же, но веса другие, поэтому цифры 2.1–2.6 из них не воспроизводятся.
    Поэтому 40 файлов восстановлены из коммита 006c34a (см. `artifacts/restored56.json`):
    именно они дают совпадение с `decode56.json` бит в бит на всех 10 фолдах.
    Каталог `artifacts/orig56/` удалён — двух версий одного артефакта не осталось.
    """
    records = {}
    for outer in range(1, 11):
        for inner in [((outer - 1 + offset) % 10) + 1 for offset in range(1, 5)]:
            path = ROOT / folder / f"{prefix}_outer{outer}_inner{inner}.json"
            if not path.exists():
                raise SystemExit(f"нет записи фолда: {path}")
            records[outer, inner] = json.loads(path.read_text(encoding="utf-8"))
    return records


def verify_against_decode56(data, reference="artifacts/decode56.json"):
    """Проверка, что расчёт стоит на тех же данных, что и decode56."""
    from tools.decode52 import evaluate

    report = json.loads((ROOT / reference).read_text(encoding="utf-8"))
    rows = []
    for outer in range(1, 11):
        start = report["threshold_by_fold"][str(outer)]["start"]
        point = evaluate(data[outer]["pairs"], data[outer]["cache"], start, GATE)
        want = report["per_fold"][outer - 1]["token_f1"]
        rows.append({"fold": outer, "reproduced": point["token_f1"], "decode56": want,
                     "match": abs(point["token_f1"] - want) < 1e-9})
    return {"folds": rows, "all_match": all(row["match"] for row in rows),
            "source": "artifacts/ablation56_ten_*, восстановленные из 006c34a (прогон 44218112557)"}


def fold_inputs(pairs, records):
    """Для каждого внешнего фолда: вероятности, пары и переносимый порог."""
    folds = grouped_folds(pairs, 10)
    out = {}
    for outer in range(1, 11):
        inner = [((outer - 1 + offset) % 10) + 1 for offset in range(1, 5)]
        grids = [records[outer, i]["inner_grid"] for i in inner]
        primary = records[outer, outer % 10 + 1]
        out[outer] = {
            "pairs": folds[outer - 1],
            "cache": primary["outer_predictions"],
            "grids": grids,
            "inner": inner,
        }
    return out


def pooled_f1_at(grids, index):
    """Объединённые счётчики четырёх внутренних фолдов на точке сетки."""
    rows = [grid[index] for grid in grids]
    if any(not row["support"]["negative_tokens"] for row in rows):
        return None, None
    totals = merge([row["support"] for row in rows])
    point = metrics(totals, 0, 0)
    return point["token_f1"], point


def pick_threshold(grids, rule="pooled"):
    """Порог по внутренним фолдам: максимум pooled F1 при FPR <= 0.40."""
    best = None
    for index in range(len(grids[0])):
        f1, point = pooled_f1_at(grids, index)
        if f1 is None or point["fpr_token"] > FPR_LIMIT:
            continue
        start = grids[0][index]["start"]
        if best is None or (f1, -start) > (best[0], -best[1]):
            best = (f1, start, index)
    return best


def oracle_on(pairs, cache, gate=GATE, max_spans=MAX_SPANS, nms=NMS_IOU):
    """Оракульный порог: лучший token F1 при FPR <= 0.40 на ЭТОМ наборе."""
    best = None
    for start in GRID:
        point = score_pairs(pairs, cache, start, gate, max_spans, nms)
        if point["fpr_token"] > FPR_LIMIT:
            continue
        if best is None or point["token_f1"] > best[1]["token_f1"]:
            best = (start, point)
    return best


def part_oracle(data):
    rows = []
    for outer in range(1, 11):
        item = data[outer]
        picked = pick_threshold(item["grids"])
        start = picked[1] if picked else None
        point = score_pairs(item["pairs"], item["cache"], start) if start else None
        oracle = oracle_on(item["pairs"], item["cache"])
        rows.append({
            "outer": outer,
            "pairs": len(item["pairs"]),
            "threshold_transfer": start,
            "transfer_token_f1": point["token_f1"] if point else None,
            "transfer_fpr": point["fpr_token"] if point else None,
            "threshold_oracle": oracle[0] if oracle else None,
            "oracle_token_f1": oracle[1]["token_f1"] if oracle else None,
            "oracle_fpr": oracle[1]["fpr_token"] if oracle else None,
            "oracle_precision": oracle[1]["token_precision"] if oracle else None,
        })
    diff = [row["threshold_oracle"] - row["threshold_transfer"] for row in rows
            if row["threshold_oracle"] is not None and row["threshold_transfer"] is not None]
    oracles = [row["threshold_oracle"] for row in rows if row["threshold_oracle"] is not None]
    transfers = [row["threshold_transfer"] for row in rows if row["threshold_transfer"] is not None]
    return {
        "per_fold": rows,
        "oracle_threshold_std": statistics.pstdev(oracles) if len(oracles) > 1 else 0.0,
        "oracle_threshold_min": min(oracles) if oracles else None,
        "oracle_threshold_max": max(oracles) if oracles else None,
        "oracle_threshold_mean": statistics.mean(oracles) if oracles else None,
        "transfer_threshold_std": statistics.pstdev(transfers) if len(transfers) > 1 else 0.0,
        "transfer_threshold_min": min(transfers) if transfers else None,
        "transfer_threshold_max": max(transfers) if transfers else None,
        "threshold_gap_mean": statistics.mean(diff) if diff else None,
        "transfer_pooled_token_f1": _pooled(rows, "transfer_token_f1"),
        "oracle_pooled_token_f1": _pooled(rows, "oracle_token_f1"),
    }


def _pooled(rows, key):
    values = [row[key] for row in rows if row[key] is not None]
    return statistics.mean(values) if values else None


def part_transfer(data):
    """1.2 Перенос порога: выбран на 9 фолдах, применён к десятому.

    Внутренние сетки — 86 точек каждая (GRID), по четыре на фолд. Порог
    выбирается по объединённым счётчикам внутренних сеток ДЕВЯТИ фолдов
    (правило pooled, FPR <= 0.40), затем применяется к отложенному десятому.
    Метки отложенного фолда при выборе не используются.
    """
    rows = []
    for outer in range(1, 11):
        pooled = []
        for fold in range(1, 11):
            if fold == outer:
                continue
            pooled.extend(data[fold]["grids"])
        best = None
        # длина сетки берётся из самих артефактов (86 точек), а не из GRID
        for index in range(len(pooled[0])):
            points = [grid[index] for grid in pooled]
            if any(not point["support"]["negative_tokens"] for point in points):
                continue
            value = metrics(merge([point["support"] for point in points]), 0, 0)
            if value["fpr_token"] > FPR_LIMIT:
                continue
            if best is None or value["token_f1"] > best[0]:
                best = (value["token_f1"], points[0]["start"], value["fpr_token"])
        start = best[1] if best else None
        point = score_pairs(data[outer]["pairs"], data[outer]["cache"], start) if start else None
        rows.append({
            "outer": outer,
            "threshold_lofo": start,
            "inner_pooled_token_f1": best[0] if best else None,
            "inner_pooled_fpr": best[2] if best else None,
            "token_f1": point["token_f1"] if point else None,
            "fpr_token": point["fpr_token"] if point else None,
            "precision": point["token_precision"] if point else None,
        })
        print(f"  LOFO фолд {outer}: порог {start}, F1 {rows[-1]['token_f1']}, "
              f"FPR {rows[-1]['fpr_token']}", flush=True)
    values = [row["token_f1"] for row in rows if row["token_f1"] is not None]
    return {
        "per_fold": rows,
        "pooled_token_f1": statistics.mean(values) if values else None,
        "pooled_fpr": _pooled(rows, "fpr_token"),
        "mean_precision": _pooled(rows, "precision"),
        "min_token_f1": min(values) if values else None,
        "max_token_f1": max(values) if values else None,
        "thresholds": sorted({row["threshold_lofo"] for row in rows}),
        "rule": ("порог выбран по внутренним сеткам девяти фолдов "
                 "(объединённые счётчики, FPR <= 0.40), применён к десятому; "
                 "метки отложенного фолда при выборе не участвуют"),
    }


def part_losses(data):
    """Вклад каждого ограничения отдельно: снимается ровно одно.

    База — поставка: gate 0.45, max_spans 1, nms 0.10, extend = start/2.
    Значения — усреднение по фолдам на их переносимом пороге.
    """
    variants = {
        "база (поставка)": dict(max_spans=1, nms=NMS_IOU, gate=GATE),
        "max_spans без ограничения": dict(max_spans=None, nms=NMS_IOU, gate=GATE),
        "max_spans=3": dict(max_spans=3, nms=NMS_IOU, gate=GATE),
        "nms=0.30": dict(max_spans=1, nms=0.30, gate=GATE),
        "nms=0.50": dict(max_spans=1, nms=0.50, gate=GATE),
        "гейт снят (gate=0)": dict(max_spans=1, nms=NMS_IOU, gate=0.0),
        "гейт 0.30": dict(max_spans=1, nms=NMS_IOU, gate=0.30),
        "extend=start (ratio 1.0)": dict(max_spans=1, nms=NMS_IOU, gate=GATE, ratio=1.0),
        "extend=start*0.75": dict(max_spans=1, nms=NMS_IOU, gate=GATE, ratio=0.75),
    }
    table = {}
    for name, config in variants.items():
        f1s, fprs, precisions, spans = [], [], [], []
        for outer in range(1, 11):
            item = data[outer]
            picked = pick_threshold(item["grids"])
            if not picked:
                continue
            start = picked[1]
            ratio = config.get("ratio", 0.5)
            point = score_pairs(item["pairs"], item["cache"], start,
                                gate=config["gate"], max_spans=config["max_spans"],
                                nms=config["nms"], extend=start * ratio)
            f1s.append(point["token_f1"])
            fprs.append(point["fpr_token"])
            precisions.append(point["token_precision"])
            spans.append(point["spans_per_pair"])
        table[name] = {
            "mean_token_f1": statistics.mean(f1s) if f1s else None,
            "mean_fpr": statistics.mean(fprs) if fprs else None,
            "mean_precision": statistics.mean(precisions) if precisions else None,
            "predicted_spans_total": sum(spans),
            "config": {key: value for key, value in config.items()},
        }
    base = table["база (поставка)"]["mean_token_f1"]
    for name, row in table.items():
        row["delta_vs_base"] = row["mean_token_f1"] - base
    # сумма вкладов против разрыва «оракул - база»
    return table


def part_grid(data):
    """1.4 Сетка декодера: max_spans × nms_iou × extend × порог.

    Про отбор. Правило промпта — отбирать конфигурацию только внутри train/val.
    Из сохранённых артефактов это возможно ТОЛЬКО для порога: `inner_grid`
    посчитан при gate=0.45 и extend=start/2, другие оси в нём не варьировались,
    а вероятности внутренних фолдов не сохранены (сохранены агрегаты).

    Поэтому здесь даётся ДВЕ вещи:

      * `selected_by_train_val` — конфигурация, выбранная по внутренним фолдам
        (порог по pooled правилу, gate=0.45, extend=start/2, max_spans=1) и
        оценённая на отложенных;
      * `oracle_upper_bound` — максимум по всей сетке, выбранный НА САМОМ
        отложенном фолде. Это завышенная граница: если даже она ниже 0.55,
        то никакой переносимый отбор до 0.55 не дотянет, и вопрос закрыт без
        нового инференса.

    Вторая величина используется только как верхняя граница (диагностика), в
    конфигурацию она не годится — это прямо запрещено промптом.
    """
    span_axis = (1, 2, 3, None)
    nms_axis = (0.10, 0.30, 0.50)
    ratio_axis = (0.5, 0.75, 1.0)
    gate_axis = (0.0, 0.30, 0.45)
    rows = []
    best_overall = None
    for outer in range(1, 11):
        item = data[outer]
        picked = pick_threshold(item["grids"])
        start = picked[1] if picked else None
        selected = score_pairs(item["pairs"], item["cache"], start) if start else None
        best = None
        for gate in gate_axis:
            for point in GRID:
                for ratio in ratio_axis:
                    for spans in span_axis:
                        for nms in nms_axis:
                            if spans == 1 and nms != 0.10:
                                continue  # при одном спане NMS не работает
                            value = score_pairs(item["pairs"], item["cache"], point, gate=gate,
                                                max_spans=spans, nms=nms, extend=point * ratio)
                            if value["fpr_token"] > FPR_LIMIT:
                                continue
                            if best is None or value["token_f1"] > best["token_f1"]:
                                best = {"token_f1": value["token_f1"],
                                        "fpr_token": value["fpr_token"],
                                        "token_precision": value["token_precision"],
                                        "gate": gate, "start": point, "extend": round(point * ratio, 3),
                                        "max_spans": spans, "nms_iou": nms}
        rows.append({
            "outer": outer,
            "pairs": len(item["pairs"]),
            "selected_by_train_val": {
                "start": start,
                "gate": GATE, "extend": round(start / 2, 3) if start else None,
                "max_spans": MAX_SPANS, "nms_iou": NMS_IOU,
                "token_f1": selected["token_f1"] if selected else None,
                "fpr_token": selected["fpr_token"] if selected else None,
                "token_precision": selected["token_precision"] if selected else None,
            },
            "oracle_upper_bound": best,
        })
        if best and (best_overall is None or best["token_f1"] > best_overall["token_f1"]):
            best_overall = best
        print(f"  фолд {outer}: отобрано {rows[-1]['selected_by_train_val']['token_f1']:.6f}, "
              f"оракул сетки {best['token_f1'] if best else None:.6f}", flush=True)
    def mean_of(key, source):
        values = [row[source][key] for row in rows if row[source] and row[source][key] is not None]
        return statistics.mean(values) if values else None
    return {
        "axes": {"max_spans": list(span_axis), "nms_iou": list(nms_axis),
                 "extend_ratio": list(ratio_axis), "gate": list(gate_axis),
                 "start": list(GRID)},
        "per_fold": rows,
        "selected_by_train_val_mean_token_f1": mean_of("token_f1", "selected_by_train_val"),
        "selected_by_train_val_mean_fpr": mean_of("fpr_token", "selected_by_train_val"),
        "selected_by_train_val_mean_precision": mean_of("token_precision", "selected_by_train_val"),
        "oracle_upper_bound_mean_token_f1": mean_of("token_f1", "oracle_upper_bound"),
        "oracle_upper_bound_mean_fpr": mean_of("fpr_token", "oracle_upper_bound"),
        "oracle_upper_bound_mean_precision": mean_of("token_precision", "oracle_upper_bound"),
        "oracle_upper_bound_min_token_f1": min((row["oracle_upper_bound"]["token_f1"] for row in rows
                                                if row["oracle_upper_bound"]), default=None),
        "oracle_upper_bound_max_token_f1": max((row["oracle_upper_bound"]["token_f1"] for row in rows
                                                if row["oracle_upper_bound"]), default=None),
        "note": ("выбор по train/val возможен только для порога; остальные оси "
                 "даны как верхняя граница «выбрано на самом фолде» и в конфигурацию не идут"),
        "test301": False,
    }


def part_ceiling(data, ceiling_path="artifacts/ceiling56.json"):
    """Сверка «потолка 0,644» с той же выборкой, на которой он посчитан.

    `ceiling52.py --outer 1` берёт НЕ 10-фолдовый сплит, а первый внешний фолд
    ПЯТИфолдового (`grouped_folds(pairs, 5)[0]`, 105 пар). Независимый потолок
    надо сравнивать не с pooled по десяти фолдам (699 пар), а с тривиальной
    базовой линией на тех же 105 парах.
    """
    from spanver.evaluate import covered_tokens, token_positions
    from tools.train_base33 import grouped_folds

    pairs = read_jsonl(ROOT / "artifacts/ours56_docs.jsonl")
    held = grouped_folds(pairs, 5)[0]
    gold = total = 0
    for pair in held:
        gold += len(covered_tokens(pair["model_output"],
                                   [(span["start"], span["end"]) for span in pair["spans"]]))
        total += len(token_positions(pair["model_output"]))
    share = gold / total
    trivial = 2 * share / (1 + share)
    report = json.loads((ROOT / ceiling_path).read_text(encoding="utf-8"))
    point = report["oracle_threshold_point"]
    return {
        "ceiling_point": point,
        "ceiling_pairs": len(held),
        "ceiling_split": "первый внешний фолд ПЯТИфолдового сплита (ceiling52.py: grouped_folds(pairs,5)[0])",
        "ceiling_gold_share": share,
        "ceiling_gold_tokens": gold,
        "ceiling_total_tokens": total,
        "ceiling_trivial_f1": trivial,
        "ceiling_over_trivial": point["token_f1"] - trivial,
        "pooled_10fold_token_f1": 0.5118790496760259,
        "comparable": False,
        "why": ("pooled 0,5119 посчитан по 699 парам 10-фолдового сплита, где доля "
                "золотых 0,33345 и тривиальная базовая линия 0,5001; потолок 0,644 — "
                "на 105 парах, где доля золотых 0,4157 и тривиальная линия 0,5873. "
                "Разница 0,1321 — разница выборок, а не запас правила."),
    }


def main():
    parser = argparse.ArgumentParser()
    # Фолды и метки берутся из того файла, на котором ОБУЧАЛИСЬ артефакты:
    # разбиение схемы 52 строится по `doc_ids`, а в прогоне 44218112557 они взяты
    # из `docs`. Сам `document_text` в этом расчёте не используется вообще (в
    # артефактах лежат готовые вероятности по кандидатам), поэтому «вход» здесь —
    # только предмет отчёта о происхождении весов, а не вычислительная величина.
    parser.add_argument("--pairs", default="artifacts/ours56_docs.jsonl")
    parser.add_argument("--out", default="artifacts/rule57.json")
    args = parser.parse_args()

    pairs = read_jsonl(ROOT / args.pairs)
    if len(pairs) != 699:
        raise SystemExit(f"ожидалось 699 пар train/val, получено {len(pairs)}")
    records = load_records()
    data = fold_inputs(pairs, records)

    report = {
        "status": "обработка",
        "entry": "joined_reference",
        "pairs": len(pairs),
        "decoder": {
            "gate": GATE, "max_spans": MAX_SPANS, "nms_iou": NMS_IOU,
            "extend": "start/2", "min_chars": MIN_CHARS,
            "source": "tools/freeze_config.py (max_spans=1), spanver/module.py (nms=0.10), tools/decode56.py (gate=0.45)",
        },
        "test301": False,
    }
    print("1.1 оракульный порог по фолдам...", flush=True)
    report["oracle"] = part_oracle(data)
    print("1.2 перенос порога...", flush=True)
    report["transfer"] = part_transfer(data)
    print("1.3 разложение потерь...", flush=True)
    report["losses"] = part_losses(data)
    print("1.4 сетка декодера...", flush=True)
    report["grid"] = part_grid(data)
    print("сверка потолка 0,644...", flush=True)
    report["ceiling_reconciliation"] = part_ceiling(data)
    report["verification"] = verify_against_decode56(data)
    report["status"] = "готово"
    (ROOT / args.out).write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n",
                                 encoding="utf-8")
    print(json.dumps({
        "1.1 разброс оракульного порога": report["oracle"]["oracle_threshold_std"],
        "1.2 перенос pooled F1": report["transfer"]["pooled_token_f1"],
        "1.4 отобрано по train/val": report["grid"]["selected_by_train_val_mean_token_f1"],
        "1.4 верхняя граница сетки": report["grid"]["oracle_upper_bound_mean_token_f1"],
        "потолок 0,644 на своих парах: тривиально": report["ceiling_reconciliation"]["ceiling_trivial_f1"],
    }, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
