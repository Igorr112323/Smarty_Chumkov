from .decode import hysteresis_flags, spans_from_flags
from .sentence_snap import snap_sentences
from .snap import snapped

GATE = 0.45


def predict44(value, start, extend, snap=False):
    if max((piece[2] for piece in value["pieces"]), default=0.0) < GATE:
        return []
    flags = hysteresis_flags(value["pieces"], start, extend)
    result = snapped(spans_from_flags(value["answer"], value["pieces"], flags, 2), value["answer"], "expand")
    return snap_sentences(result, value["answer"], "B3") if snap else result
