"""Промпт 64: адаптер входа/фолдов для НЕИЗМЕНЁННОЙ схемы 52.

10 внешних фолдов; inner=(outer+1)%10; 5 сидов; модель обучается на 8/10.
Только GitHub Actions: обучение и вычисление предсказаний.
"""
import argparse
import json
import platform
import statistics
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def load_data():
    from tools.split64 import gold_rows, make_record
    rows = gold_rows()
    recorded = json.loads((ROOT / 'spanver/splits1000.json').read_text(encoding='utf-8'))
    if recorded != make_record(rows):
        raise ValueError('Зафиксированный сплит не совпадает с входными данными')
    by_id = {r['pair_id']: r for r in rows}
    return rows, [[by_id[i] for i in f['pair_ids']] for f in recorded['folds']], recorded


def measure_speed(tokenizer, model, pairs):
    from tools.nested_cv35 import probabilities, predict
    import torch
    median = statistics.median(len(r['document_text']) for r in pairs)
    row = min(pairs, key=lambda r: (abs(len(r['document_text']) - median), r['pair_id']))
    def run():
        value = probabilities(tokenizer, model, [row])[row['pair_id']]
        predict(value, .45)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(4)
    run()
    times = []
    for _ in range(5):
        start = time.perf_counter(); run(); times.append(time.perf_counter() - start)
    torch.set_num_threads(previous_threads)
    return {'models': 1, 'seed': 0, 'length_median_corpus': median,
            'length_sample': len(row['document_text']), 'pair_id': row['pair_id'],
            'samples_ms_per_1000': [1000 * t / (len(row['document_text']) / 1000) for t in times],
            'ms_per_1000': statistics.median(times) * 1000 / (len(row['document_text']) / 1000),
            'threads': 4, 'platform': platform.platform(), 'note': 'внутреннее сравнение, 1 сид; длина документа'}


def run(outer, commit, path):
    import torch
    from spanver import lexfeat
    from tools.train46 import train_variant
    from tools.train52_chunks import SEEDS, GRID, GATE
    from tools.nested38 import split
    from tools.nested_cv35 import averaged, probabilities
    from tools.preflight_13 import id_digest
    from tools.sweep_ragtruth30 import prepare, compute

    torch.use_deterministic_algorithms(True)
    lexfeat.with_question(False)
    pairs, folds, manifest = load_data()
    inner = outer % 10
    training, tuning, held = split(folds, outer - 1, inner)
    report = {'status': 'в процессе', 'outer_fold': outer, 'inner_fold': inner + 1,
              'git_sha': commit, 'gold_sha256': manifest['gold_sha256'],
              'fold_ids_sha256': manifest['fold_ids_sha256'],
              'train_id_sha256': id_digest(training), 'inner_id_sha256': id_digest(tuning),
              'outer_id_sha256': id_digest(held),
              'train_pairs': len(training), 'inner_pairs': len(tuning), 'outer_pairs': len(held),
              'configuration': 'train52_chunks: v2 permanent_freeze, 15 epochs max, 5 seeds, без вопроса',
              'old_test_included': True, 'separate_test_measurement': False,
              'seeds': []}
    def save():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    save()
    inner_caches, outer_caches = [], []
    start = time.perf_counter()
    for seed in SEEDS:
        clock = time.perf_counter()
        tokenizer, model = lexfeat.load_feature_model(ROOT / 'models/rubert-tiny2', seed=seed)
        model.base.config._attn_implementation = 'eager'
        torch.manual_seed(seed)
        info = train_variant(model, tokenizer, training, tuning, seed, 'v2', permanent_freeze=True)
        inner_caches.append(probabilities(tokenizer, model, tuning))
        outer_caches.append(probabilities(tokenizer, model, held))
        if outer == 1 and seed == 0:
            report['speed'] = measure_speed(tokenizer, model, pairs)
        report['seeds'].append({'seed': seed, 'seconds': round(time.perf_counter()-clock, 2), **info})
        save()
        del model, tokenizer
    inner_cache = {p['pair_id']: averaged(p, inner_caches, range(len(SEEDS))) for p in tuning}
    prepared = prepare(tuning, inner_cache)
    grid = [compute(prepared, threshold, threshold / 2, GATE) for threshold in GRID]
    # Точное правило train52_chunks.py, включая оба FPR и tie в порядке сетки.
    allowed = [r for r in grid if r['fpr_token'] <= .40 and r['fpr_pair'] <= .40]
    selected = max(allowed, key=lambda r: r['token_f1_exact']) if allowed else None
    report['inner_grid'] = grid
    report['threshold_point'] = selected
    report['outer_predictions'] = {p['pair_id']: averaged(p, outer_caches, range(len(SEEDS)))['pieces'] for p in held}
    report['elapsed_s'] = round(time.perf_counter()-start, 2)
    if selected:
        from tools.nested_cv35 import predict
        from tools.metrics64 import measure
        predictions = {p['pair_id']: predict({'answer': p['model_output'], 'pieces': report['outer_predictions'][p['pair_id']]}, selected['start']) for p in held}
        report['metrics'] = measure(held, predictions)
        report['metrics']['threshold'] = selected['start']
    report['status'] = 'готово' if selected else 'нет допустимого порога'
    save()
    if not selected:
        raise ValueError('Нет допустимого порога на внутреннем фолде')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--outer', type=int, required=True, choices=range(1, 11))
    parser.add_argument('--git-sha', required=True)
    args = parser.parse_args()
    path = ROOT / f'artifacts/p64_fold{args.outer:02}.json'
    try:
        run(args.outer, args.git_sha, path)
    except Exception:
        report = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {'outer_fold': args.outer}
        report.update(status='ошибка', error=traceback.format_exc())
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
        raise


if __name__ == '__main__':
    main()
