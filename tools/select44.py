import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from spanver.decode44 import predict44
from tools.evaluate_snap42 import compute as span_metrics
from tools.fpr_bound43 import counts as fpr_counts
from tools.sweep_ragtruth30 import compute, prepare

GRID = tuple(number / 100 for number in range(10, 96))
DELTAS = (0.0, 0.02, 0.04, 0.06)
LIMIT_NUMERATOR = 19
LIMIT_DENOMINATOR = 50


def admissible(negative_pairs, flagged_pairs, negative_tokens, flagged_tokens):
    return bool(negative_pairs and negative_tokens and flagged_pairs * LIMIT_DENOMINATOR <= negative_pairs * LIMIT_NUMERATOR and flagged_tokens * LIMIT_DENOMINATOR <= negative_tokens * LIMIT_NUMERATOR)


def blend(seed_rows):
    if not seed_rows:
        raise ValueError("нет сидов")
    maps = [{row["pair_id"]: row for row in rows} for rows in seed_rows]
    expected = set(maps[0])
    if any(set(mapping) != expected for mapping in maps[1:]):
        raise ValueError("у сидов разные пары")
    pairs = []
    cache = {}
    for row in seed_rows[0]:
        key = row["pair_id"]
        entries = [mapping[key] for mapping in maps]
        answer = row["answer"]
        gold = row["gold_spans"]
        offsets = [(piece[0], piece[1]) for piece in row["pieces"]]
        if any(entry["answer"] != answer or entry["gold_spans"] != gold or [(piece[0], piece[1]) for piece in entry["pieces"]] != offsets for entry in entries):
            raise ValueError("у сидов различаются gold или координаты")
        pieces = [[left, right, sum(entry["pieces"][index][2] for entry in entries) / len(entries)] for index, (left, right) in enumerate(offsets)]
        pairs.append({"pair_id": key, "model_output": answer, "spans": gold})
        cache[key] = {"answer": answer, "pieces": pieces}
    return pairs, cache


def base_curve(inner_sets):
    if len(inner_sets) != 4:
        raise ValueError("нужны четыре внутренних фолда")
    prepared = [prepare(pairs, cache) for pairs, cache in inner_sets]
    curve = []
    for start in GRID:
        values = [compute(group, start, start / 2, 0.45) for group in prepared]
        valid = all(admissible(row["support"]["negative_pairs"], row["support"]["flagged_negative_pairs"], row["support"]["negative_tokens"], row["support"]["flagged_negative_tokens"]) for row in values)
        curve.append({"start": start, "extend": start / 2, "mean_token_f1": statistics.mean(row["token_f1_exact"] for row in values), "max_fpr_pair": max(row["fpr_pair"] for row in values), "max_fpr_token": max(row["fpr_token"] for row in values), "eligible": valid, "inner": values})
    eligible = [row for row in curve if row["eligible"]]
    return (max(eligible, key=lambda row: (row["mean_token_f1"], -row["start"])) if eligible else None), curve


def prediction_rows(pairs, cache, start, extend, snap):
    return [{"pair_id": row["pair_id"], "answer": row["model_output"], "gold_spans": row["spans"], "predicted_spans": predict44(cache[row["pair_id"]], start, extend, snap)} for row in pairs]


def compensation_curve(inner_sets, start, extend):
    if len(inner_sets) != 4:
        raise ValueError("нужны четыре внутренних фолда")
    base = [fpr_counts(prediction_rows(pairs, cache, start, extend, False)) for pairs, cache in inner_sets]
    output = []
    for delta in DELTAS:
        folds = []
        eligible = True
        for index, (pairs, cache) in enumerate(inner_sets):
            rows = prediction_rows(pairs, cache, start + delta, extend, True)
            scores = span_metrics(rows)
            fpr = fpr_counts(rows)
            permitted = admissible(fpr["negative_pairs"], fpr["flagged_negative_pairs"], fpr["negative_tokens"], fpr["flagged_negative_tokens"])
            neutral = fpr["flagged_negative_tokens"] <= base[index]["flagged_negative_tokens"]
            eligible = eligible and permitted
            folds.append({"inner_index": index + 1, "exact_f1": scores["exact_f1"], "overlap_f1": scores["overlap_f1"], "token_f1": scores["token_f1"], "fpr_pair": fpr["fpr_pair"], "fpr_token": fpr["fpr_token"], "flagged_negative_tokens": fpr["flagged_negative_tokens"], "base_flagged_negative_tokens": base[index]["flagged_negative_tokens"], "fpr_neutral": neutral, "eligible": permitted})
        output.append({"delta": delta, "start": start + delta, "extend": extend, "mean_exact_f1": statistics.mean(row["exact_f1"] for row in folds), "mean_overlap_f1": statistics.mean(row["overlap_f1"] for row in folds), "mean_fpr_token": statistics.mean(row["fpr_token"] for row in folds), "max_fpr_token": max(row["fpr_token"] for row in folds), "max_fpr_pair": max(row["fpr_pair"] for row in folds), "eligible": eligible, "inner": folds})
    options = [row for row in output if row["eligible"]]
    return (max(options, key=lambda row: (row["mean_exact_f1"], -row["delta"])) if options else None), output
