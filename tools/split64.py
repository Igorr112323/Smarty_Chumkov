"""Промпт 64: документные компоненты и воспроизводимый сплит всей ручной разметки.

Не меняет splits.json и не запускает обучение или инференс. Применение:
python tools/split64.py --write; python tools/split64.py --check.
"""
import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from spanver.build import build_gold
from spanver.pairs import doc_ids
from tools.preflight_13 import components, our_keys, id_digest

OUTPUT = ROOT / 'spanver/splits1000.json'


def gold_rows():
    rows = build_gold(ROOT / 'Dataset', {})
    ids = [row['pair_id'] for row in rows]
    if len(rows) != 1000 or len(set(ids)) != 1000 or sum(len(r['spans']) for r in rows) != 423:
        raise ValueError('Ожидались 1000 уникальных пар и 423 спана')
    for row in rows:
        if not row['document_text'] or row['doc_ids'] != doc_ids(row['document_text']):
            raise ValueError('Неверные doc_ids или пустой joined_reference: ' + row['pair_id'])
        for span in row['spans']:
            if not isinstance(span['start'], int) or not isinstance(span['end'], int) or not 0 <= span['start'] <= span['end'] < len(row['model_output']):
                raise ValueError('Некорректные границы разметки: ' + row['pair_id'])
    return rows


def gold_digest(rows):
    # Каноническое содержимое всего набора, включая метки и исходные тексты;
    # старое поле split не участвует в новом сплите.
    clean = [{k: v for k, v in row.items() if k != 'split'} for row in rows]
    encoded = ''.join(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(',', ':')) + '\n' for row in clean)
    return hashlib.sha256(encoded.encode('utf-8')).hexdigest()


def assign(rows):
    groups = components(rows, our_keys)
    if len(groups) != 943 or max(map(len, groups)) != 3:
        raise ValueError('Неожиданные характеристики документных компонент')
    # Размещаем самые размеченные компоненты первыми, выбираем фолд с
    # минимальным числом спанов, при равенстве — с меньшим числом пар.
    # Сид фиксирует только порядок равных по весу компонент.
    random.Random(0).shuffle(groups)
    groups.sort(key=lambda g: -sum(len(r['spans']) for r in g))
    folds = [[] for _ in range(10)]
    counts = [0] * 10
    for group in groups:
        weight = sum(len(r['spans']) for r in group)
        idx = min(range(10), key=lambda i: ((counts[i], len(folds[i]), i) if weight else (len(folds[i]), counts[i], i)))
        folds[idx].extend(group)
        counts[idx] += weight
    return folds


def validate(rows, folds):
    if len(folds) != 10:
        raise ValueError('Не 10 фолдов')
    all_ids = [r['pair_id'] for fold in folds for r in fold]
    if len(all_ids) != 1000 or len(set(all_ids)) != 1000 or set(all_ids) != {r['pair_id'] for r in rows}:
        raise ValueError('Пары потеряны или повторены')
    overlaps = []
    for i, held in enumerate(folds):
        held_docs = {v for r in held for v in r['doc_ids']}
        train_docs = {v for j, fold in enumerate(folds) if j != i for r in fold for v in r['doc_ids']}
        overlap = held_docs & train_docs
        overlaps.append(len(overlap))
        if overlap:
            raise ValueError(f'Пересечение doc_ids в фолде {i+1}: {sorted(overlap)}')
    return overlaps


def make_record(rows):
    folds = assign(rows)
    overlaps = validate(rows, folds)
    return {'schema': 'prompt64-doc-split-v1',
            'source': 'Dataset/1_rushallu_rag_gold; doc_ids из joined_reference',
            'assignment': 'компоненты; seed 0; положительные: минимум (спаны, пары); нулевые: минимум (пары, спаны)',
            'gold_sha256': gold_digest(rows), 'gold_sha256_format': 'JSONL без split, UTF-8, sort_keys, separators без пробелов',
            'pairs': len(rows), 'spans': sum(len(r['spans']) for r in rows),
            'components': 943, 'max_component_pairs': 3,
            'fold_ids_sha256': [id_digest(fold) for fold in folds],
            'folds': [{'fold': i + 1, 'pairs': len(fold), 'spans': sum(len(r['spans']) for r in fold),
                       'doc_overlap': overlaps[i], 'pair_ids': sorted(r['pair_id'] for r in fold)}
                      for i, fold in enumerate(folds)]}


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--write', action='store_true')
    group.add_argument('--check', action='store_true')
    args = parser.parse_args()
    expected = make_record(gold_rows())
    if args.write:
        if OUTPUT.exists():
            raise SystemExit('Отказ: splits1000.json уже существует; не перезаписывать')
        OUTPUT.write_text(json.dumps(expected, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    elif json.loads(OUTPUT.read_text(encoding='utf-8')) != expected:
        raise SystemExit('Зафиксированный сплит или исходные данные изменились')
    print('gold_sha256:', expected['gold_sha256'])
    print('компоненты:', expected['components'], 'максимум:', expected['max_component_pairs'])
    for row, digest in zip(expected['folds'], expected['fold_ids_sha256']):
        print(f"{row['fold']:2}: {row['pairs']:3} пар, {row['spans']:2} спанов; пересечение {row['doc_overlap']}; {digest}")


if __name__ == '__main__':
    main()
