import collections

import pytest

from inspection.evaluate import load_scores, save_scores, split_results, summarize
from inspection.threshold import OperatingPoint


def test_save_and_load(tmp_path):
    results = [("image1.png", "good", 0, 0.5), ("image2.png", "defect", 1, 0.8)]
    path = tmp_path / "scores.json"
    save_scores(results, path, "modelX", "categoryY", 42)
    results_loaded = load_scores(path)
    assert results == results_loaded


def test_logic_split_results():
    results = [(f"good_{i}.png", "good", 0, 0.1 + i * 0.01) for i in range(10)] + [
        (f"scratch_{i}.png", "scratch", 1, 0.6 + i * 0.01) for i in range(10)
    ]
    validation, test = split_results(results, test_size=0.5, seed=42)
    assert set(validation) & set(test) == set()
    assert set(validation) | set(test) == set(results)

    assert collections.Counter(r[1] for r in validation) == {"good": 5, "scratch": 5}
    assert collections.Counter(r[1] for r in test) == {"good": 5, "scratch": 5}


def make_point(threshold, recall, false_reject_rate):
    return OperatingPoint(
        threshold=threshold,
        recall=recall,
        false_reject_rate=false_reject_rate,
        false_negatives=0,
        false_positives=0,
        cost_fn=10.0,
        cost_fp=1.0,
        defect_rate=None,
        cost=0.0,
    )


def test_coherent_summarize():

    measured_points = [
        make_point(threshold=30.0, recall=0.5, false_reject_rate=0.25),
        make_point(threshold=32.0, recall=0.75, false_reject_rate=0.0),
        make_point(threshold=34.0, recall=1.0, false_reject_rate=0.5),
    ]

    summary = summarize(measured_points)
    assert summary["threshold"]["min"] == 30.0
    assert summary["threshold"]["max"] == 34.0
    assert summary["threshold"]["mean"] == pytest.approx(32.0)
    assert summary["threshold"]["std"] == pytest.approx(2.0)

    assert summary["recall"]["min"] == 0.5
    assert summary["recall"]["max"] == 1.0
    assert summary["recall"]["mean"] == pytest.approx(0.75)
    assert summary["recall"]["std"] == pytest.approx(0.25)

    assert summary["false_reject_rate"]["min"] == 0.0
    assert summary["false_reject_rate"]["max"] == 0.5
    assert summary["false_reject_rate"]["mean"] == pytest.approx(0.25)
    assert summary["false_reject_rate"]["std"] == pytest.approx(0.25)
