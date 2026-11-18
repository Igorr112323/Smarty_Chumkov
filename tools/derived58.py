"""ПРОМПТ 58, стадия 2.3: пересчёт прошлых цифр как отрыва от базлайна.

Каждая цифра из отчётов 52/56/57 пересчитывается вместе с базлайном ИМЕННО своего
набора. Набор — это не «тест» или «валидация» вообще, а конкретный список пар,
поэтому базлайн считается по тому же списку тем же кодом
(`covered_tokens` + `token_positions`).

Наборы:
  * pooled 2.1 (699 пар, 10 фолдов)              — базлайн 0,5001320461281141;
  * test 301 в целом;
  * 167 чистых test-пар (список из критерия 1.7 промпта 56);
  * фолд 1 пятифолдового сплита (те 105 пар, где мерялся «потолок 2.5»);
  * LOFO-оценка из 57 (те же 699 пар);
  * оракул сетки 57 (те же 699 пар, среднее по фолдам).

Результат — `artifacts/derived58.json`, из него собирается таблица в `review58.md`.
"""
import ast
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from spanver.answer_json import parse_answer
from spanver.evaluate import covered_tokens, read_jsonl, token_positions
from spanver.io_csv import read_gold
from spanver.locate import locate
from spanver.splits import load_splits
from tools.torchless57 import stub_if_absent

stub_if_absent()

from tools.train_base33 import grouped_folds  # noqa: E402


def baseline(pairs, gold_of):
    """gold_of(pair) -> список интервалов; baseline по ответам этих пар."""
    gold = total = 0
    with_gold = 0
    for pair in pairs:
        answer = pair["model_output"]
        total += len(token_positions(answer))
        gold += len(covered_tokens(answer, gold_of(pair)))
        if gold_of(pair):
            with_gold += 1
    share = gold / total if total else 0.0
    return {"pairs": len(pairs), "gold_share": share,
            "all_positive_token_f1": 2 * share / (1 + share) if share else 0.0,
            "pairs_with_gold": with_gold}


def test_pairs():
    """301 тестовая пара: метки из CSV, вход не нужен."""
    splits = load_splits()
    test_ids = set(splits["test"])
    rows = {}
    for filename, prefix in (("sberquad-rag.csv", "sber"), ("ruscibench-rag.csv", "rusci")):
        for number, row in enumerate(read_gold(ROOT / "Dataset/1_rushallu_rag_gold" / filename)):
            pair_id = f"{prefix}{number:06d}"
            if pair_id not in test_ids:
                continue
            output = row.get("model_output") or ""
            raw = parse_answer(row.get("answer"), pair_id)
            spans = []
            for item, found in zip(raw, locate(raw, output)):
                spans.append({"start": found["start"], "end": found["end"]})
            rows[pair_id] = {"pair_id": pair_id, "model_output": output, "spans": spans}
    return rows


def clean_ids(rows):
    """167 чистых пар: ни один doc_id пары не встречается в train+val."""
    splits = load_splits()
    train_val = set(splits["train"]) | set(splits["val"])
    docs_of = {}
    for filename, prefix in (("sberquad-rag.csv", "sber"), ("ruscibench-rag.csv", "rusci")):
        for number, row in enumerate(read_gold(ROOT / "Dataset/1_rushallu_rag_gold" / filename)):
            pair_id = f"{prefix}{number:06d}"
            text = str(row.get("docs") or "").strip()
            value = ast.literal_eval(text) if text.startswith("[") else []
            docs_of[pair_id] = {str(item.get("doc_id")) for item in value if isinstance(item, dict)}
    train_docs = set().union(*[docs_of[p] for p in train_val if p in docs_of])
    return [pair_id for pair_id in rows if not (docs_of.get(pair_id, set()) & train_docs)]


def main():
    report = {"status": "готово", "stage": "58 стадия 2.3", "sets": {}}

    # --- 699 пар train/val (pooled 2.1, LOFO 57, оракул сетки 57)
    train_val = read_jsonl(ROOT / "artifacts/ours56_docs.jsonl")
    base = baseline(train_val, lambda p: [(s["start"], s["end"]) for s in p["spans"]])
    report["sets"]["train_val_699"] = {
        "description": "pooled 2.1 по 10 фолдам, LOFO 1.2 и оракул сетки 1.4 промпта 57",
        **base,
        "figures": {
            "pooled_token_f1_2_1": 0.5118790496760259,
            "lofo_token_f1_1_2": 0.5084861158391286,
            "grid_selected_1_4": 0.5058005779543769,
            "grid_oracle_upper_1_4": 0.5094646870856996,
        },
    }

    # --- 105 пар: фолд 1 пятифолдового сплита (там мерялся «потолок 2.5»)
    five = grouped_folds(train_val, 5)[0]
    base5 = baseline(five, lambda p: [(s["start"], s["end"]) for s in p["spans"]])
    report["sets"]["five_fold_1_105"] = {
        "description": "внешний фолд 1 пятифолдового сплита — набор, на котором измерен «потолок 2.5» (0,644) и «потолок 52» (0,4791)",
        **base5,
        "figures": {
            "ceiling_2_5_token_f1": 0.644,
            "ceiling_52_token_f1": 0.4791,
        },
    }

    # --- test 301 и 167 чистых
    rows = test_pairs()
    ids = sorted(rows)
    whole = [rows[pair_id] for pair_id in ids]
    base_test = baseline(whole, lambda p: [(s["start"], s["end"]) for s in p["spans"]])
    report["sets"]["test_301"] = {
        "description": "все 301 пара test",
        **base_test,
        "figures": {
            "docs_token_f1_2_8": 0.4131520690242379,
            "joined_token_f1_2_8": 0.436824592915403,
        },
    }
    clean = clean_ids(rows)
    clean_rows = [rows[pair_id] for pair_id in clean]
    base_clean = baseline(clean_rows, lambda p: [(s["start"], s["end"]) for s in p["spans"]])
    report["sets"]["test_clean_167"] = {
        "description": "167 чистых пар test: их doc_id не встречаются в train+val",
        **base_clean,
        "figures": {
            "docs_token_f1_2_8": 0.4422872066252019,
            "joined_token_f1_2_8": 0.46766536554488874,
        },
    }

    # --- отрывы
    for name, item in report["sets"].items():
        base_value = item["all_positive_token_f1"]
        item["margins"] = {key: value - base_value for key, value in item["figures"].items()}
        item["margin_note"] = "отрыв = метрика минус all-positive token F1 того же набора"

    (ROOT / "artifacts/derived58.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    for name, item in report["sets"].items():
        print(f"{name}: n={item['pairs']} базлайн {item['all_positive_token_f1']:.6f} "
              f"(доля {item['gold_share']:.6f})")
        for key, value in item["figures"].items():
            print(f"    {key}: {value} → отрыв {item['margins'][key]:+.6f}")


if __name__ == "__main__":
    main()
