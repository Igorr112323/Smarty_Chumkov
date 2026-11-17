"""Projection and deterministic equal-architecture probability averaging."""
from bisect import bisect_left


def blend(primary, secondary):
    if primary['answer'] != secondary['answer']:
        raise ValueError('разные ответы')
    a, b = primary['pieces'], secondary['pieces']
    result = []
    for left, right, value in a:
        if right <= left:
            raise ValueError('пустой субтокен')
        overlap_sum = coverage = 0
        for start, end, score in b:
            overlap = max(0, min(right, end) - max(left, start))
            if overlap:
                overlap_sum += overlap * score
                coverage += overlap
        if coverage > right-left:
            raise ValueError('перекрывающиеся субтокены второй модели')
        # Missing second-model characters contribute zero; no gold or threshold used.
        secondary_score = overlap_sum / (right - left)
        result.append([left, right, (value + secondary_score) / 2])
    return {'answer': primary['answer'], 'pieces': result}


def fit_isotonic(scores_labels):
    if not scores_labels:
        raise ValueError('нет данных калибровки')
    ordered = sorted(scores_labels)
    grouped = []
    for score, label in ordered:
        if not 0 <= score <= 1 or label not in (0, 1):
            raise ValueError('неверная пара score/label')
        if grouped and grouped[-1][1] == score:
            grouped[-1][2] += label
            grouped[-1][3] += 1
        else:
            grouped.append([score, score, label, 1])
        while len(grouped) > 1 and grouped[-2][2] / grouped[-2][3] > grouped[-1][2] / grouped[-1][3]:
            latter = grouped.pop()
            grouped[-1][1] = latter[1]
            grouped[-1][2] += latter[2]
            grouped[-1][3] += latter[3]
    return [{'min': a, 'max': b, 'value': total / count, 'count': count} for a,b,total,count in grouped]


def calibrate(value, blocks):
    # Piecewise-linear interpolation between block boundaries, with clipping.
    points = []
    for block in blocks:
        points.append((block['min'], block['value']))
        if block['max'] != block['min']:
            points.append((block['max'], block['value']))
    if not points:
        raise ValueError('пустая изотоническая модель')
    xs = [x for x,_ in points]
    index = bisect_left(xs, value)
    if index == 0:
        return points[0][1]
    if index == len(points):
        return points[-1][1]
    x0,y0 = points[index-1]; x1,y1 = points[index]
    return y0 + (y1-y0) * (value-x0) / (x1-x0) if x1 != x0 else y1
