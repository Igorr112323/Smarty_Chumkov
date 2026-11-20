"""Прогон согласованной итерации 63: 20 эпох, разморозка после третьей.

Сохраняет вероятности обоих фолдов для калибровки ансамбля на внутреннем фолде.
Отчёты старых конфигураций отделены префиксом. Только запуск в Actions.
"""
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tools import train63 as T

T.UNFREEZE_AFTER = 3
predict_original = T.predict_pairs
train_original = T.train_model
save_original = T.save_json
predictions = []


def predict(*args, **kwargs):
    result = predict_original(*args, **kwargs)
    predictions.append(result)
    return result


def train(*args, **kwargs):
    history = train_original(*args, **kwargs)
    if any(row.get('diverged') for row in history):
        raise RuntimeError('Обучение разошлось; отчёт нельзя использовать')
    return history


def save(path, report):
    if len(predictions) >= 2:
        report['inner_candidates'] = predictions[0]
        report['outer_candidates'] = predictions[1]
    save_original(path, report)


T.predict_pairs, T.train_model, T.save_json = predict, train, save

if __name__ == '__main__':
    try:
        T.main()
    except Exception:
        # Сохраняем ошибку под тем же именем, что ожидает заливщик.
        def arg(name, default=None):
            return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default
        outer = int(arg('--outer', '0'))
        inner = (outer - 1 + int(arg('--inner', '0'))) % T.FOLDS + 1
        seed = int(arg('--seed', '0'))
        prefix = arg('--prefix', 'seg63c')
        path = Path(arg('--out') or ROOT / f'artifacts/{prefix}_outer{outer}_inner{inner}_seed{seed}.json')
        report = {'status': 'падение', 'outer_fold': outer, 'inner_fold': inner,
                  'seed': seed, 'error': traceback.format_exc()}
        save_original(path, report)
        raise
