"""Cost-based selection of the OK/KO decision threshold.

On a production line a missed defect (false negative) usually costs far more than
rejecting a good part (false positive). Instead of the threshold maximising F1,
we pick the one minimising the cost: cost_fn * FN + cost_fp * FP.

With defect_rate=None the cost counts errors on the validation set, which implicitly
assumes the line has the same share of defects as that set. Passing defect_rate (the
expected share of defects on the line, strictly between 0 and 1) gives the expected
cost per part instead: cost_fn * defect_rate * FN_rate + cost_fp * (1 - defect_rate) * FP_rate.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass
class OperatingPoint:
    threshold: float
    recall: float  # share of defective parts caught
    false_reject_rate: float  # share of good parts rejected
    false_negatives: int
    false_positives: int
    cost: float  # error count cost if defect_rate is None, else expected cost per part


def _check_inputs(
    scores: np.ndarray, labels: np.ndarray, defect_rate: float | None
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Convert to arrays, validate, return (scores, labels, n_pos, n_neg)."""
    if defect_rate is not None and (defect_rate <= 0.0 or defect_rate >= 1.0):
        raise ValueError("defect_rate must be None or in (0, 1)")
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    if scores.shape != labels.shape or scores.ndim != 1:
        raise ValueError("scores and labels must be 1-D arrays of the same length")
    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        raise ValueError("need both good and defective samples")
    return scores, labels, n_pos, n_neg


def evaluate_threshold(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    cost_fn: float = 10.0,
    cost_fp: float = 1.0,
    defect_rate: float | None = None,
) -> OperatingPoint:
    """Evaluate a single threshold. A part is flagged KO when score >= threshold.

    labels: 1 = defective, 0 = good."""
    scores, labels, n_pos, n_neg = _check_inputs(scores, labels, defect_rate)
    flagged = scores >= threshold
    fn = int(np.sum(~flagged & (labels == 1)))
    fp = int(np.sum(flagged & (labels == 0)))
    fp_rate = fp / n_neg
    fn_rate = fn / n_pos
    fn_cost = cost_fn * fn
    fp_cost = cost_fp * fp
    if defect_rate is not None:
        # Adjust the cost to account for the expected defect rate in production.
        fn_cost = cost_fn * fn_rate * defect_rate
        fp_cost = cost_fp * fp_rate * (1 - defect_rate)
    return OperatingPoint(
        threshold=float(threshold),
        recall=1 - fn_rate,
        false_reject_rate=fp_rate,
        false_negatives=fn,
        false_positives=fp,
        cost=fn_cost + fp_cost,
    )


def operating_points(
    scores: np.ndarray,
    labels: np.ndarray,
    cost_fn: float = 10.0,
    cost_fp: float = 1.0,
    defect_rate: float | None = None,
) -> list[OperatingPoint]:
    """Evaluate every candidate threshold. A part is flagged KO when score >= threshold.

    labels: 1 = defective, 0 = good.
    """
    scores, labels, _, _ = _check_inputs(scores, labels, defect_rate)
    # Candidates: midpoints between unique scores, plus -inf and +inf to flag everything or nothing.
    sorted_scores = np.concatenate([[-np.inf], np.unique(scores), [np.inf]])
    candidates = (sorted_scores[:-1] + sorted_scores[1:]) * 0.5
    points = []
    for t in candidates:
        points.append(
            evaluate_threshold(
                scores,
                labels,
                threshold=t,
                cost_fn=cost_fn,
                cost_fp=cost_fp,
                defect_rate=defect_rate,
            )
        )
    return points


def select_threshold(
    scores: np.ndarray,
    labels: np.ndarray,
    cost_fn: float = 10.0,
    cost_fp: float = 1.0,
    defect_rate: float | None = None,
) -> OperatingPoint:
    """Lowest-cost operating point; ties broken by the lowest false reject rate."""
    points = operating_points(scores, labels, cost_fn, cost_fp, defect_rate)
    return min(points, key=lambda p: (p.cost, p.false_reject_rate))


def save_threshold(
    point: OperatingPoint, path: Path, cost_fn: float, cost_fp: float, defect_rate: float | None
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {**asdict(point), "cost_fn": cost_fn, "cost_fp": cost_fp, "defect_rate": defect_rate}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_threshold(path: Path) -> float:
    return float(json.loads(path.read_text(encoding="utf-8"))["threshold"])
