"""Чистый Python-вывод для локального каскада (без numpy/scikit-learn).

Веса экспортируютсяtraining-скриптом `tools/train_local_ranker.py` в
`spanver/ranker_weights.json`. Структура дерева повторяет представление
sklearn HistGradientBoostingClassifier: сравнение идёт с `num_threshold`,
листовое значение уже включает learning rate.
"""

import math


def sigmoid(value):
    if value >= 0:
        if value > 35:
            return 1.0
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    if value < -35:
        return 0.0
    z = math.exp(value)
    return z / (1.0 + z)


def logistic(features, params, expected=None):
    """Логистическая модель с StandardScaler, совместимая с sklearn."""
    if expected is not None and len(features) != expected:
        raise ValueError(f"local_ranker: ожидается {expected} признаков, получено {len(features)}")
    total = params["b"]
    for value, weight, mean, scale in zip(features, params["w"], params["mu"], params["sd"]):
        total += weight * ((value - mean) / scale if scale else 0.0)
    return sigmoid(total)


def predict_trees(model, features):
    """Вероятность положительного класса на boosting деревьях."""
    total = float(model["baseline"])
    for tree in model["trees"]:
        index = 0
        while True:
            node = tree[index]
            if node["leaf"]:
                total += node["v"]
                break
            value = features[node["f"]]
            # sklearn: unknown values идут влево; в признаках NaN NaN не бывает.
            index = node["l"] if value <= node["t"] else node["r"]
    return sigmoid(total)
