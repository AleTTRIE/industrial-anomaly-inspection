"""Cost-based selection of the OK/KO decision threshold.

On a production line a missed defect (false negative) usually costs far more than
rejecting a good part (false positive). Instead of the threshold maximising F1,
we pick the one minimising the expected cost: cost_fn * FN + cost_fp * FP.
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
    cost: float


def operating_points(
    scores: np.ndarray, labels: np.ndarray, cost_fn: float = 10.0, cost_fp: float = 1.0
) -> list[OperatingPoint]:
    """Evaluate every candidate threshold. A part is flagged KO when score >= threshold.

    labels: 1 = defective, 0 = good.
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    if scores.shape != labels.shape or scores.ndim != 1:
        raise ValueError("scores and labels must be 1-D arrays of the same length")
    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        raise ValueError("need both good and defective samples")

    # Candidates: midpoints between unique scores, plus -inf and +inf to flag everything or nothing.
    a = np.concatenate([[-np.inf], np.unique(scores), [np.inf]])
    candidates = (a[:-1] + a[1:]) * 0.5  # midpoints between unique scores
    points = []
    for t in candidates:
        flagged = scores >= t
        fn = int(np.sum(~flagged & (labels == 1)))
        fp = int(np.sum(flagged & (labels == 0)))
        points.append(
            OperatingPoint(
                threshold=float(t),
                recall=(n_pos - fn) / n_pos,
                false_reject_rate=fp / n_neg,
                false_negatives=fn,
                false_positives=fp,
                cost=cost_fn * fn + cost_fp * fp,
            )
        )
    return points


def select_threshold(
    scores: np.ndarray, labels: np.ndarray, cost_fn: float = 10.0, cost_fp: float = 1.0
) -> OperatingPoint:
    """Lowest-cost operating point; ties broken by the lowest false reject rate."""
    points = operating_points(scores, labels, cost_fn, cost_fp)
    return min(points, key=lambda p: (p.cost, p.false_reject_rate))


def save_threshold(point: OperatingPoint, path: Path, cost_fn: float, cost_fp: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {**asdict(point), "cost_fn": cost_fn, "cost_fp": cost_fp}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_threshold(path: Path) -> float:
    return float(json.loads(path.read_text(encoding="utf-8"))["threshold"])
