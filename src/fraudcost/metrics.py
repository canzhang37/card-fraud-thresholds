"""Ranking, calibration and cost metrics."""
import numpy as np


def average_precision(y, score, w=None):
    """Average precision, ties as one threshold."""
    y = np.asarray(y, float)
    w = np.ones_like(y) if w is None else np.asarray(w, float)
    order = np.argsort(-np.asarray(score), kind="mergesort")
    s, y, w = np.asarray(score)[order], y[order], w[order]
    tp = np.cumsum(w * y)
    seen = np.cumsum(w)
    last = np.r_[np.flatnonzero(np.diff(s) != 0), len(s) - 1]
    tp, seen = tp[last], seen[last]
    if tp[-1] == 0:
        return np.nan
    gained = np.diff(np.r_[0.0, tp])
    return float(np.sum(gained * tp / seen) / tp[-1])


def top_share_mask(score, rate, w=None):
    """Flag the top `rate` share of transactions."""
    w = np.ones(len(score)) if w is None else np.asarray(w, float)
    order = np.argsort(-np.asarray(score), kind="mergesort")
    keep = np.cumsum(w[order]) <= rate * w.sum()
    mask = np.zeros(len(score), bool)
    mask[order[keep]] = True
    return mask


def recall_at(y, score, amount, rate, w=None):
    """Share of frauds and fraud dollars caught."""
    w = np.ones(len(y)) if w is None else np.asarray(w, float)
    m = top_share_mask(score, rate, w)
    fraud = w * y
    return fraud[m].sum() / fraud.sum(), (fraud * amount)[m].sum() / (fraud * amount).sum()


def cost(y, flag, amount, c, w=None):
    """Cost c per flag plus every missed fraud."""
    w = np.ones(len(y)) if w is None else np.asarray(w, float)
    return float(np.sum(w * (c * flag + (1 - flag) * y * amount)))


def f1_threshold(y, score):
    """Score cut-off with the best F1."""
    order = np.argsort(-score, kind="mergesort")
    s, yy = score[order], y[order]
    tp = np.cumsum(yy)
    k = np.arange(1, len(s) + 1)
    last = np.r_[np.flatnonzero(np.diff(s) != 0), len(s) - 1]
    f1 = 2 * tp[last] / (k[last] + yy.sum())
    return float(s[last][np.argmax(f1)])


def cost_threshold(y, score, amount, c):
    """Single score cut-off with the lowest total cost."""
    order = np.argsort(-score, kind="mergesort")
    s, yy, a = score[order], y[order], amount[order]
    missed = (yy * a).sum() - np.cumsum(yy * a)
    total = c * np.arange(1, len(s) + 1) + missed
    last = np.r_[np.flatnonzero(np.diff(s) != 0), len(s) - 1]
    best = last[np.argmin(total[last])]
    if (yy * a).sum() < total[best]:  # flagging nothing is cheaper
        return np.inf
    return float(s[best])


def cluster_weights(groups, n_boot, seed):
    """Bootstrap row weights, whole groups drawn."""
    codes, inverse = np.unique(groups, return_inverse=True)
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(len(codes), np.full(len(codes), 1 / len(codes)), size=n_boot)
    return counts.astype(np.float32)[:, inverse]  # float32 keeps 1,000 x 90,000 rows near 360 MB
