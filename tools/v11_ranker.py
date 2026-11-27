"""V11 блок 2: признаки границ/document support и сборка listwise-групп.

Один и тот же код считает признаки для обучения (позитивы + сконструированные
негативы) и для инференса (все кандидаты пары), поэтому утечки «train видит одно,
decode другое» нет. Признаки не используют gold: только ответ, документ и
baseline-скор v3.
"""
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import v11_lab as L
from spanver import ranker
from spanver.evaluate import iou
from spanver.module import candidate_spans, document_context

FEATURES = (
    "base_risk", "base_rank", "base_rel", "is_top1", "is_whole",
    "log_chars", "log_words", "rel_len", "position",
    "token_cov", "cos", "sentence_cov", "verbatim", "numbers_missing", "names_missing",
    "left_ok", "right_ok", "is_clause", "sentence_aligned",
    "cov_rank", "is_min_cov", "cov_gap",
    "answer_cov", "answer_cos", "min_clause_cov", "mean_clause_cov", "n_clauses", "n_low_cov",
    "gate", "log_answer_len", "n_sentences_answer", "numbers_in_answer", "capital_share",
    "deficit", "iou_top1", "support_rank",
)
N_FEATURES = len(FEATURES)


def pair_context(pair, params):
    """Ответ, документ, clause-сетка и baseline-скоринг всех кандидатов."""
    answer = pair.get("model_output") or ""
    document = pair.get("document_text") or ""
    context = document_context(document)
    rows = []
    for start, end, text in candidate_spans(answer):
        vector = ranker.features(text, start, end, answer, document, context[0], context[2])
        risk = ranker._logit(vector, params["cand"], ranker.FEATURE_COUNT)
        rows.append({"start": start, "end": end, "text": text, "features": vector, "risk": risk})
    risks = [row["risk"] for row in rows] or [0.0]
    gate = ranker._logit(
        ranker.pair_features([{"features": row["features"]} for row in rows], risks, answer),
        params["pair"], ranker.PAIR_FEATURE_COUNT,
    )
    clauses = L.clause_spans(answer)
    clause_covs = []
    for start, end, text in clauses:
        clause_covs.append(L.support_features(text, document)["token_cov"])
    whole = L.support_features(answer, document)
    ordered = sorted(range(len(rows)), key=lambda index: (-rows[index]["risk"], rows[index]["start"]))
    rank_of = {index: position + 1 for position, index in enumerate(ordered)}
    return {
        "pair": pair, "params": params, "answer": answer, "document": document, "rows": rows, "gate": gate,
        "clauses": clauses, "clause_covs": clause_covs, "whole": whole,
        "n_rows": max(1, len(rows)), "rank_of": rank_of,
        "top1": rows[ordered[0]] if rows else None,
        "top1_risk": rows[ordered[0]]["risk"] if rows else 0.0,
        "min_clause_cov": min(clause_covs) if clause_covs else 0.0,
        "mean_clause_cov": sum(clause_covs) / len(clause_covs) if clause_covs else 0.0,
        "n_low_cov": sum(1 for value in clause_covs if value < 0.5),
    }


def base_risk(ctx, start, end, text):
    """Baseline-риск для произвольной границы: берём готовый кандидат либо считаем."""
    for row in ctx["rows"]:
        if row["start"] == start and row["end"] == end:
            return row["risk"], True
    context = document_context(ctx["document"])
    vector = ranker.features(text, start, end, ctx["answer"], ctx["document"], context[0], context[2])
    params = ctx["params"]
    return ranker._logit(vector, params["cand"], ranker.FEATURE_COUNT), False


def vector(ctx, start, end, text):
    answer = ctx["answer"]
    support = L.support_features(text, ctx["document"])
    risk, known = base_risk(ctx, start, end, text)
    rank = ctx["rank_of"].get(
        next((index for index, row in enumerate(ctx["rows"])
              if row["start"] == start and row["end"] == end), -1), None)
    rank_value = (rank / ctx["n_rows"]) if rank else 0.5
    clause_hit = any(c[0] == start and c[1] == end for c in ctx["clauses"])
    sentence_aligned = 0.0
    if start == 0 or re.search(r"[.!?\n]\s*$", answer[:start]):
        if end >= len(answer) - 1 or re.match(r"\s*[.!?]", answer[end + 1:]):
            sentence_aligned = 1.0
    left_ok, right_ok = L.clause_boundaries(answer, start, end)
    words = L.WORD_RE.findall(text)
    capitals = sum(1 for word in words if word[:1].isupper())
    support_score = 0.6 * support["token_cov"] + 0.4 * support["cos"]
    clause_scores = [0.6 * cov + 0.4 * 0.5 for cov in ctx["clause_covs"]]
    support_rank = (sum(1 for value in clause_scores if value < support_score) / max(1, len(clause_scores)))
    top1 = ctx["top1"]
    iou_top1 = iou(start, end, top1["start"], top1["end"]) if top1 else 0.0
    values = [
        risk,
        rank_value,
        risk - ctx["top1_risk"],
        float(bool(top1 and top1["start"] == start and top1["end"] == end)),
        float(start == 0 and end == len(answer) - 1),
        math.log1p(len(text)),
        math.log1p(len(words)),
        len(text) / max(1, len(answer)),
        start / max(1, len(answer)),
        support["token_cov"],
        support["cos"],
        support["sentence_cov"],
        support["verbatim"],
        support["numbers_missing"],
        support["names_missing"],
        left_ok,
        right_ok,
        float(clause_hit),
        sentence_aligned,
        rank_value,
        float(abs(support["token_cov"] - ctx["min_clause_cov"]) < 1e-9),
        ctx["mean_clause_cov"] - support["token_cov"],
        ctx["whole"]["token_cov"],
        ctx["whole"]["cos"],
        ctx["min_clause_cov"],
        ctx["mean_clause_cov"],
        float(len(ctx["clauses"])),
        float(ctx["n_low_cov"]),
        ctx["gate"],
        math.log1p(len(answer)),
        float(len(L.SENT_RE.findall(answer))),
        float(len(re.findall(r"\d", answer))),
        capitals / max(1, len(words)),
        1.0 - support_score,
        iou_top1,
        support_rank,
    ]
    del known
    return values


def snap_to_words(answer, start, end, step):
    """Сдвиг границы на `step` слов внутри ответа (для hard negatives)."""
    if step < 0:
        left = answer[:start].rstrip()
        if not left:
            return None
        words = list(re.finditer(r"\S+", left))
        if not words:
            return None
        new_start = words[max(0, len(words) - abs(step))].start()
        return (new_start, end)
    right = answer[end + 1:].lstrip()
    if not right:
        return None
    words = list(re.finditer(r"\S+", right))
    if not words:
        return None
    offset = len(answer) - len(right)
    position = words[min(len(words) - 1, step - 1)]
    return (start, offset + position.end() - 1)


def neighbour_clauses(ctx, start, end):
    """Соседние clause слева/справа и расширение/сжатие на один clause."""
    clauses = ctx["clauses"]
    found = []
    for index, clause in enumerate(clauses):
        if clause[0] == start and clause[1] == end:
            if index > 0:
                found.append(("prev_clause", clauses[index - 1][0], clauses[index - 1][1]))
            if index + 1 < len(clauses):
                found.append(("next_clause", clauses[index + 1][0], clauses[index + 1][1]))
            break
    for index, clause in enumerate(clauses):
        if clause[0] <= start and clause[1] >= end and (clause[0] < start or clause[1] > end):
            found.append(("expand_clause", clause[0], clause[1]))
            break
    for index, clause in enumerate(clauses):
        if start <= clause[0] and end >= clause[1] and (clause[0] > start or clause[1] < end):
            found.append(("shrink_clause", clause[0], clause[1]))
            if index + 1 < len(clauses) and clauses[index + 1][1] <= end:
                found.append(("shrink_clause2", clauses[index + 1][0], clauses[index + 1][1]))
            break
    return found


def make_group(pair, ctx, label_spans, rng=None, max_negatives=12):
    """Позитивы = gold-границы, негативы = сдвиги/расширения/соседи/топ-1 baseline."""
    answer = ctx["answer"]
    positives = []
    for gold in label_spans:
        best, best_iou = None, 0.0
        for row in ctx["rows"]:
            value = iou(row["start"], row["end"], gold["start"], gold["end"])
            if value > best_iou:
                best, best_iou = row, value
        if best is not None and best_iou >= 0.5:
            positives.append((best["start"], best["end"], best["text"], 1))
        else:
            text = answer[gold["start"]:gold["end"] + 1]
            positives.append((gold["start"], gold["end"], text, 1))
    seen = {(item[0], item[1]) for item in positives}
    negatives = []

    def add(start, end, tag):
        del tag
        if start is None or end is None or start < 0 or end >= len(answer) or start >= end:
            return
        if (start, end) in seen:
            return
        if any(iou(start, end, positive[0], positive[1]) >= 0.5 for positive in positives):
            return
        seen.add((start, end))
        negatives.append((start, end, answer[start:end + 1], 0))

    for start, end, _, _ in list(positives):
        for step in (-3, -2, -1, 1, 2, 3):
            shift = snap_to_words(answer, start, end, step)
            if shift:
                add(shift[0], shift[1], f"shift{step}")
        for tag, left, right in neighbour_clauses(ctx, start, end):
            add(left, right, tag)
    top1 = ctx["top1"]
    if top1:
        add(top1["start"], top1["end"], "top1")
    if len(positives) > 1:
        for start, end, _, _ in list(positives):
            add(*snap_to_words(answer, start, end, 1) or (None, None), "pos_shift")
    if rng is not None and len(negatives) < max_negatives:
        pool = list(ctx["rows"])
        rng.shuffle(pool)
        for row in pool:
            if len(negatives) >= max_negatives:
                break
            add(row["start"], row["end"], "random")
    if len(negatives) > max_negatives:
        step = len(negatives) / max_negatives
        negatives = [negatives[int(index * step)] for index in range(max_negatives)]
    group = []
    for start, end, text, label in positives + negatives:
        group.append({"start": start, "end": end, "text": text, "label": label,
                      "vector": vector(ctx, start, end, text)})
    return group


# ------------------------------------------------------------------ экспорт/инференс

def export_lightgbm(booster):
    """Деревья LightGBM в чистый JSON (leaf_value уже включает shrinkage)."""
    dump = booster.dump_model()
    trees = []
    for info in dump["tree_info"]:
        flattened = []
        stack = [(info["tree_structure"], 0)]
        pending = {id(info["tree_structure"]): 0}
        nodes = []

        def walk(node):
            index = len(nodes)
            nodes.append(None)
            if "leaf_value" in node and "split_feature" not in node:
                nodes[index] = {"leaf": True, "v": float(node["leaf_value"])}
            else:
                feature = int(node["split_feature"])
                threshold = float(node["threshold"])
                left = walk(node["left_child"])
                right = walk(node["right_child"])
                nodes[index] = {"leaf": False, "f": feature, "t": threshold,
                                "l": left, "r": right, "v": 0.0}
            return index

        del stack, pending
        walk(info["tree_structure"])
        trees.append(nodes)
        flattened.append(len(nodes))
    return {"trees": trees, "baseline": 0.0, "n_features": dump["max_feature_idx"] + 1,
            "objective": "lambdarank", "nodes_per_tree": flattened}


def predict_raw(model, values):
    total = float(model.get("baseline", 0.0))
    for tree in model["trees"]:
        index = 0
        while True:
            node = tree[index]
            if node["leaf"]:
                total += node["v"]
                break
            index = node["l"] if values[node["f"]] <= node["t"] else node["r"]
    return total
