"""Success-class diagnostics at a fixed, untuned probability threshold."""
import numpy as np


def divide_or_none(numerator, denominator):
    return float(numerator / denominator) if denominator > 0 else None


def classification_metrics(probabilities, outcomes, weights, threshold=.5):
    p = np.asarray(probabilities, dtype=float); y = np.asarray(outcomes); w = np.asarray(weights, dtype=float)
    if p.ndim != 1 or y.shape != p.shape or w.shape != p.shape or not len(p):
        raise ValueError('Aligned one-dimensional predictions, outcomes and weights required')
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)) or not np.isin(y, [0, 1]).all():
        raise ValueError('Invalid probability or binary outcome')
    if not np.isfinite(w).all() or np.any(w < 0) or w.sum() <= 0 or not 0 <= threshold <= 1:
        raise ValueError('Invalid weights or threshold')
    w = w / w.sum(); predicted = p >= threshold; success = y == 1
    masks = dict(tp=predicted & success, fp=predicted & ~success,
                 fn=~predicted & success, tn=~predicted & ~success)
    counts = {name: int(mask.sum()) for name, mask in masks.items()}
    masses = {name + '_weight': float(w[mask].sum()) for name, mask in masks.items()}
    tp, fp, fn = [masses[k + '_weight'] for k in ['tp', 'fp', 'fn']]
    return dict(classification_threshold=float(threshold), positive_class='correct', **counts, **masses,
                precision=divide_or_none(tp, tp + fp), recall=divide_or_none(tp, tp + fn),
                f1=divide_or_none(2 * tp, 2 * tp + fp + fn), predicted_positive_rate=tp + fp)


def mean_defined(rows, key):
    values = [row[key] for row in rows if row[key] is not None]
    return float(np.mean(values)) if values else None


def classification_summary(rows):
    tp = sum(row['tp_weight'] for row in rows); fp = sum(row['fp_weight'] for row in rows)
    fn = sum(row['fn_weight'] for row in rows)
    result = dict(classification_threshold=.5, positive_class='correct')
    for name in ['precision', 'recall', 'f1']:
        result[name] = mean_defined(rows, name)
        result[name + '_defined_pairs'] = sum(row[name] is not None for row in rows)
    result.update(precision_equal_pair_pooled=divide_or_none(tp, tp + fp),
                  recall_equal_pair_pooled=divide_or_none(tp, tp + fn),
                  f1_equal_pair_pooled=divide_or_none(2 * tp, 2 * tp + fp + fn))
    result.update({name + '_raw_count': sum(row[name] for row in rows) for name in ['tp', 'fp', 'fn', 'tn']})
    return result
