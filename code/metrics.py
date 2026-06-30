import numpy as np
from sklearn.metrics import cohen_kappa_score

SCORE_MIN = 1
SCORE_MAX = 6
LABEL_OFFSET = 1
DEFAULT_THRESHOLDS = [1.5, 2.5, 3.5, 4.5, 5.5]


def quadratic_weighted_kappa(y_true, y_pred) -> float:
    """Competition metric: quadratic weighted Cohen kappa."""
    return cohen_kappa_score(y_true, y_pred, weights="quadratic")


def score_to_label(score: int) -> int:
    """Convert essay score 1..6 to class label 0..5."""
    return int(score) - LABEL_OFFSET


def label_to_score(label: int) -> int:
    """Convert class label 0..5 back to essay score 1..6."""
    return int(label) + LABEL_OFFSET


def apply_thresholds(values, thresholds) -> np.ndarray:
    """Apply regression cut points to raw model scores."""
    values = np.asarray(values, dtype=float)
    thresholds = np.asarray(thresholds, dtype=float)
    return (np.digitize(values, thresholds) + SCORE_MIN).clip(SCORE_MIN, SCORE_MAX).astype(int)


def tune_thresholds(raw_values, true_scores) -> tuple[list[float], float]:
    """Tune regression cut points on validation or OOF predictions."""
    raw_values = np.asarray(raw_values, dtype=float)
    true_scores = np.asarray(true_scores, dtype=int)
    thresholds = np.asarray(DEFAULT_THRESHOLDS, dtype=float)
    best_qwk = quadratic_weighted_kappa(true_scores, apply_thresholds(raw_values, thresholds))

    for span, step in [(0.75, 0.05), (0.30, 0.02), (0.12, 0.01)]:
        improved = True
        while improved:
            improved = False
            for idx in range(len(thresholds)):
                lower = thresholds[idx - 1] + 0.05 if idx > 0 else SCORE_MIN - 0.5
                upper = thresholds[idx + 1] - 0.05 if idx < len(thresholds) - 1 else SCORE_MAX + 0.5
                candidates = np.arange(
                    max(lower, thresholds[idx] - span),
                    min(upper, thresholds[idx] + span) + 1e-9,
                    step,
                )
                for candidate in candidates:
                    trial = thresholds.copy()
                    trial[idx] = candidate
                    qwk = quadratic_weighted_kappa(true_scores, apply_thresholds(raw_values, trial))
                    if qwk > best_qwk:
                        thresholds = trial
                        best_qwk = qwk
                        improved = True

    return thresholds.round(4).tolist(), float(best_qwk)


def logits_to_scores(logits, task: str, thresholds=None) -> list[int]:
    """Turn model logits into integer essay scores."""
    if task == "regression":
        values = logits.squeeze(-1).detach().cpu().numpy()
        if thresholds is not None:
            return apply_thresholds(values, thresholds).tolist()
        return np.rint(values).clip(SCORE_MIN, SCORE_MAX).astype(int).tolist()

    labels = logits.argmax(dim=-1).detach().cpu().tolist()
    return [label_to_score(label) for label in labels]


def labels_to_scores(labels, task: str) -> list[int]:
    """Turn batch labels back into integer essay scores."""
    if task == "regression":
        return labels.detach().cpu().round().clamp(SCORE_MIN, SCORE_MAX).long().tolist()
    return [label_to_score(label) for label in labels.detach().cpu().tolist()]
