from tools.collect_nested38 import GRID, select_worst
from tools.nested38 import split, target_inner
from tools.train_base33 import grouped_folds


def test_four_inner_folds_never_train_on_outer_or_inner():
    pairs = [{"pair_id": str(index), "doc_ids": [str(index // 2)]} for index in range(50)]
    folds = grouped_folds(pairs, 5)
    for outer in range(5):
        for inner in range(5):
            if inner == outer:
                continue
            train, held_inner, held_outer = split(folds, outer, inner)
            sets = [{row["pair_id"] for row in group} for group in (train, held_inner, held_outer)]
            assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
            assert sum(map(len, sets)) == len(pairs)
        assert target_inner(outer) != outer


def test_threshold_requires_fpr_limit_in_every_inner_fold():
    def grid(fold):
        result = []
        for start in GRID:
            good = start in (0.5, 0.51)
            pair_fp = 2 if good else 3
            token_fp = 4 if good else 5
            if fold == 2 and start == 0.5:
                token_fp = 5
            score = 0.8 if start == 0.5 else 0.7
            result.append({"start": start, "extend": start / 2, "token_f1_exact": score, "fpr_pair": pair_fp / 5, "fpr_token": token_fp / 10, "support": {"negative_pairs": 5, "flagged_negative_pairs": pair_fp, "negative_tokens": 10, "flagged_negative_tokens": token_fp}})
        return result
    result = select_worst([grid(i) for i in range(4)])
    assert result["start"] == 0.51
    assert result["max_inner_fpr_pair"] == 0.4
    assert result["max_inner_fpr_token"] == 0.4
    rows = [grid(i) for i in range(4)]
    for row in rows[0]:
        row["support"]["flagged_negative_tokens"] = 5
    assert select_worst(rows) is None
