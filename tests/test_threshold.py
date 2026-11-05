import json

import numpy as np
import pytest

from inspection.threshold import (
    evaluate_threshold,
    load_threshold,
    operating_points,
    save_threshold,
    select_threshold,
)


def test_perfect_separation_catches_all_defects_without_rejects():
    scores = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
    labels = np.array([0, 0, 0, 1, 1])
    point = select_threshold(scores, labels)
    assert point.recall == 1.0
    assert point.false_reject_rate == 0.0
    assert point.cost == 0.0


def test_high_miss_cost_lowers_threshold():
    # One defect (0.25) hides among good parts.
    scores = np.array([0.1, 0.2, 0.3, 0.4, 0.25, 0.9])
    labels = np.array([0, 0, 0, 0, 1, 1])
    cautious = select_threshold(scores, labels, cost_fn=10.0, cost_fp=1.0)
    lenient = select_threshold(scores, labels, cost_fn=1.0, cost_fp=10.0)
    assert cautious.recall == 1.0
    assert cautious.threshold < lenient.threshold
    assert lenient.false_reject_rate == 0.0
    assert cautious.cost == 2.0


def test_operating_points_include_flag_nothing():
    points = operating_points(np.array([0.1, 0.9]), np.array([0, 1]))
    assert points[-1].threshold == float("inf")
    assert points[-1].recall == 0.0


def test_requires_both_classes():
    with pytest.raises(ValueError):
        select_threshold(np.array([0.1, 0.2]), np.array([0, 0]))


def test_save_and_load_roundtrip(tmp_path):
    point = select_threshold(np.array([0.1, 0.9]), np.array([0, 1]))
    path = tmp_path / "threshold.json"
    save_threshold(point, path, cost_fn=10.0, cost_fp=1.0, defect_rate=0.01)
    assert load_threshold(path) == point.threshold
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["defect_rate"] == 0.01

    save_threshold(point, path, cost_fn=10.0, cost_fp=1.0, defect_rate=None)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["defect_rate"] is None


def test_threshold_between_max_good_and_min_defective():
    scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    labels = np.array([0, 0, 1, 1, 1])
    point = select_threshold(scores, labels)
    assert point.threshold > max(scores[labels == 0]) and point.threshold < min(scores[labels == 1])


def test_rare_defects_raise_threshold():
    scores = np.array([0.1, 0.2, 0.3, 0.4, 0.25, 0.9])
    labels = np.array([0, 0, 0, 0, 1, 1])
    balanced = select_threshold(scores, labels, cost_fn=10.0, cost_fp=1.0, defect_rate=0.5)
    rare = select_threshold(scores, labels, cost_fn=10.0, cost_fp=1.0, defect_rate=0.01)
    assert rare.threshold > balanced.threshold
    assert rare.recall == 0.5
    assert balanced.recall == 1.0


def test_default_cost_counts_errors():
    scores = np.array([0.1, 0.2, 0.3, 0.4, 0.25, 0.9])
    labels = np.array([0, 0, 0, 0, 1, 1])
    default = select_threshold(scores, labels, cost_fn=10.0, cost_fp=1.0)
    assert default.cost == 2.0


def test_rejects_invalid_defect_rate():
    scores = np.array([0.1, 0.9])
    labels = np.array([0, 1])
    with pytest.raises(ValueError):
        select_threshold(scores, labels, defect_rate=0.0)
    with pytest.raises(ValueError):
        select_threshold(scores, labels, defect_rate=1.0)


def test_evaluate_threshold_exact_counts():
    scores = np.array([0.1, 0.3, 0.6, 0.8, 0.2, 0.7, 0.9])
    labels = np.array([0, 0, 0, 0, 1, 1, 1])
    cost_fn = 10.0
    cost_fp = 1.0
    point = evaluate_threshold(
        scores, labels, threshold=0.5, cost_fn=cost_fn, cost_fp=cost_fp, defect_rate=None
    )
    assert point.false_negatives == 1
    assert point.false_positives == 2
    assert point.recall == pytest.approx((3 - 1) / 3)
    assert point.false_reject_rate == 2.0 / 4.0
    assert point.cost == cost_fn * 1.0 + cost_fp * 2.0


def test_score_equal_to_threshold_is_flagged():
    scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.9])
    labels = np.array([0, 0, 0, 0, 1, 1])
    point = evaluate_threshold(
        scores, labels, threshold=0.4, cost_fn=10.0, cost_fp=1.0, defect_rate=0.1
    )
    assert point.false_reject_rate == 1.0 / 4.0


def test_evaluate_threshold_accepts_lists():
    from_lists = evaluate_threshold([0.1, 0.9], [0, 1], threshold=0.5)
    from_arrays = evaluate_threshold(np.array([0.1, 0.9]), np.array([0, 1]), threshold=0.5)
    assert from_lists == from_arrays
