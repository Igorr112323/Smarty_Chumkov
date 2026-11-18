"""Заглушка torch для локального расчёта по сохранённым артефактам.

В песочнице агента torch не устанавливается, а `tools.train_base33` тянет его
через `spanver.lexfeat`. Расчёты стадий 1–2 промпта 58 модель не используют:
считаются сохранённые вероятности и метки. Поэтому при отсутствии torch
подставляется заглушка, а обращения к lexfeat падают с внятным сообщением.
В Actions стоит настоящий torch и заглушка не включается.
"""
import sys
import types


def stub_if_absent():
    try:
        import torch  # noqa: F401
        return False
    except ModuleNotFoundError:
        pass
    torch_stub = types.ModuleType("torch")
    torch_stub.__version__ = "stub"
    sys.modules["torch"] = torch_stub
    lexfeat_stub = types.ModuleType("spanver.lexfeat")

    def _denied(name):
        raise RuntimeError(
            f"lexfeat.{name} недоступен без torch: этот расчёт его не использует")

    lexfeat_stub.__getattr__ = _denied
    sys.modules["spanver.lexfeat"] = lexfeat_stub
    return True
