"""
AUROC with a 95% confidence interval via DeLong's method.
Adapted from https://github.com/yandexdataschool/roc_comparison (no Torch dependency).

Scores may be raw logits or probabilities: AUROC and the DeLong variance are
rank-based, so any monotonic transform of the score gives identical results.

    auc, (lo, hi) = delong_ci(y_true, y_score)
"""
import numpy as np
import scipy.stats


def compute_midrank(x):
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1)
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T + 1
    return T2


def fastDeLong(predictions_sorted_transposed, label_1_count):
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty([k, m], dtype=float)
    ty = np.empty([k, n], dtype=float)
    tz = np.empty([k, m + n], dtype=float)
    for r in range(k):
        tx[r, :] = compute_midrank(positive_examples[r, :])
        ty[r, :] = compute_midrank(negative_examples[r, :])
        tz[r, :] = compute_midrank(predictions_sorted_transposed[r, :])
    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov


def delong_ci(y_true, y_score, alpha=0.95):
    """
    AUROC and its confidence interval via DeLong's method.

    y_true : array of 0/1 labels (both classes must be present).
    y_score: positive-class score per sample (logit or probability).

    Returns (auc, (lo, hi)).
    """
    y_true = np.asarray(y_true).ravel()
    if not np.array_equal(np.unique(y_true), np.array([0, 1])):
        raise ValueError(
            "y_true must contain both classes coded 0/1; got unique values "
            f"{np.unique(y_true)}. AUROC is undefined for a single-class set."
        )
    y_score = np.asarray(y_score, dtype=float).ravel()
    if len(y_score) != len(y_true):
        raise ValueError(f"y_score length {len(y_score)} != y_true length {len(y_true)}")

    # Sort so positives come first. mergesort = stable, for reproducibility.
    order = np.argsort(-y_true, kind="mergesort")
    label_1_count = int(y_true.sum())

    aucs, delongcov = fastDeLong(y_score[np.newaxis, order], label_1_count)
    auc = float(aucs[0])
    auc_std = float(np.sqrt(np.asarray(delongcov).ravel()[0]))

    lower_upper_q = np.abs(np.array([0, 1]) - (1 - alpha) / 2)
    ci = scipy.stats.norm.ppf(lower_upper_q, loc=auc, scale=auc_std)
    return auc, tuple(np.clip(ci, 0, 1))
