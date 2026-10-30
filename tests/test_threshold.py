import numpy as np
import pytest

from inspection.threshold import load_threshold, operating_points, save_threshold, select_threshold


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
    save_threshold(point, path, cost_fn=10.0, cost_fp=1.0)
    assert load_threshold(path) == point.threshold


def test_threshold_between_max_good_and_min_defective():
    scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    labels = np.array([0, 0, 1, 1, 1])
    point = select_threshold(scores, labels)
    assert point.threshold > max(scores[labels == 0]) and point.threshold < min(scores[labels == 1])
