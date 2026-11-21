"""Метрики промпта 64 без изменения декодера схемы 52."""
from spanver.evaluate import covered_tokens, match_spans, token_positions
from tools.metrics59 import prf


def measure(pairs, predictions):
    tp = fp = fn = negatives = 0
    span_pred = span_gold = exact_tp = soft_tp = 0
    pair_tp = pair_fp = pair_fn = neg_pairs = marked_neg = marked = 0
    for pair in pairs:
        pred = predictions[pair['pair_id']]
        gold = pair['spans']
        text = pair['model_output']
        g = covered_tokens(text, [(s['start'], s['end']) for s in gold])
        p = covered_tokens(text, [(s['start'], s['end']) for s in pred])
        tp += len(g & p); fp += len(p - g); fn += len(g - p)
        negatives += len(token_positions(text)) - len(g)
        span_pred += len(pred); span_gold += len(gold)
        etp = match_spans(pred, gold, 1.0)[0]
        exact_tp += etp
        stp = match_spans(pred, gold, .5)[0]
        soft_tp += stp
        marked += bool(pred)
        # Совпадает с pair_level схемы 52: пара засчитывается при IoU >= 0.5.
        if gold:
            if stp: pair_tp += 1
            else: pair_fn += 1
        else:
            neg_pairs += 1
            if pred: pair_fp += 1; marked_neg += 1
    return {'pairs': len(pairs), 'token': prf(tp, fp, fn), 'fpr_token': fp / negatives if negatives else 0,
            'fpr_pair': marked_neg / neg_pairs if neg_pairs else 0,
            'span_f1_soft': 2 * soft_tp / (span_pred + span_gold) if span_pred + span_gold else 0,
            'span_f1_exact': 2 * exact_tp / (span_pred + span_gold) if span_pred + span_gold else 0,
            'pair_f1': prf(pair_tp, pair_fp, pair_fn)['f1'], 'predicted_pairs': marked,
            'gold_spans': span_gold, 'predicted_spans': span_pred}
