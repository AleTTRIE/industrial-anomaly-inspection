"""Measure the cost-based threshold rule on held-out data, then choose the final threshold.

Raw test scores are cached in results/scores/. Performance is measured over N stratified
validation/test splits of the test set (threshold chosen on one half, measured on the other)
and written to reports/. The final threshold is then chosen on all test images and saved to
models/threshold.json for the API.

Usage:
    python -m inspection.evaluate --model patchcore --category screw --cost-fn 100
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from inspection.config import (
    CATEGORIES,
    DATA_ROOT,
    MODELS,
    REPORTS_DIR,
    RESULTS_DIR,
    THRESHOLD_PATH,
)
from inspection.threshold import evaluate_threshold, save_threshold, select_threshold


def collect_scores(model_name, category, seed):
    from anomalib.data import MVTecAD
    from anomalib.engine import Engine
    from anomalib.models import Padim, Patchcore
    from anomalib.post_processing import PostProcessor
    from lightning import seed_everything

    seed_everything(seed)

    # 1. Load the dataset
    datamodule = MVTecAD(root=DATA_ROOT, category=category, num_workers=0)

    # 2. Create the model
    post_processor = PostProcessor(enable_normalization=False)
    if model_name == "patchcore":
        model = Patchcore(post_processor=post_processor)
    elif model_name == "padim":
        model = Padim(post_processor=post_processor)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    # 3. Create the engine
    engine = Engine()

    # 4. Train and predict
    engine.fit(model=model, datamodule=datamodule)
    predictions = engine.predict(model=model, datamodule=datamodule)

    results = []
    for batch in predictions:
        gt_label_list = batch.gt_label.int().tolist()
        pred_score_list = batch.pred_score.tolist()
        for path, gt_label, score in zip(
            batch.image_path, gt_label_list, pred_score_list, strict=True
        ):
            type_defaut = Path(path).parent.name
            results.append((path, type_defaut, gt_label, score))
    return results


def scores_path(model_name, category, seed):
    return RESULTS_DIR / "scores" / f"{model_name}_{category}_{seed}_raw.json"


def save_scores(results, path, model_name, category, seed) -> None:
    from importlib.metadata import PackageNotFoundError, version

    try:
        anomalib_version = version("anomalib")
    except PackageNotFoundError:
        anomalib_version = None

    path.parent.mkdir(parents=True, exist_ok=True)
    dict_to_save = {
        "model": model_name,
        "category": category,
        "seed": seed,
        "normalize": False,
        "anomalib_version": anomalib_version,
        "scores": [],
    }
    for image_path, defect_type, gt_label, image_score in results:
        dict_to_save["scores"].append(
            {
                "path": image_path,
                "defect_type": defect_type,
                "gt_label": gt_label,
                "score": image_score,
            }
        )
    path.write_text(json.dumps(dict_to_save, indent=2), encoding="utf-8")


def load_scores(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    results = []
    for item in data["scores"]:
        results.append((item["path"], item["defect_type"], item["gt_label"], item["score"]))
    return results


def load_or_collect_scores(model_name, category, seed):
    path = scores_path(model_name, category, seed)
    if path.exists():
        results = load_scores(path)
        print(f"Loaded {len(results)} scores for {model_name} on {category}")
    else:
        results = collect_scores(model_name, category, seed)
        save_scores(results, path, model_name, category, seed)
        print(f"Collected and saved {len(results)} scores for {model_name} on {category}")
    return results


def split_results(results, test_size=0.5, seed=42):
    """Split results into validation and test sets, stratified by defect type."""
    defect_types = [r[1] for r in results]

    validation, test = train_test_split(
        results,
        test_size=test_size,
        random_state=seed,
        stratify=defect_types,
    )

    return validation, test


def scores_and_labels(results):
    """Return scores and labels as numpy arrays."""
    scores = np.array([r[3] for r in results])
    labels = np.array([r[2] for r in results])
    return scores, labels


def evaluate_split(results, split_seed, cost_fn, cost_fp, defect_rate):
    validation, test = split_results(results, test_size=0.5, seed=split_seed)
    val_scores, val_labels = scores_and_labels(validation)
    test_scores, test_labels = scores_and_labels(test)

    chosen = select_threshold(
        val_scores, val_labels, cost_fn=cost_fn, cost_fp=cost_fp, defect_rate=defect_rate
    )
    measured = evaluate_threshold(
        test_scores,
        test_labels,
        threshold=chosen.threshold,
        cost_fn=cost_fn,
        cost_fp=cost_fp,
        defect_rate=defect_rate,
    )
    return chosen, measured


def summarize(measured_points):
    """Summarize measured points by calculating min, max, mean and std for each metric."""
    metrics = ["threshold", "recall", "false_reject_rate", "cost"]
    summary = {}
    for metric in metrics:
        values = [getattr(point, metric) for point in measured_points]
        summary[metric] = {
            "min": np.min(values),
            "max": np.max(values),
            "mean": np.mean(values),
            "std": np.std(values, ddof=1),  # Sample standard deviation
        }
    return summary


def parse_args():
    parser = argparse.ArgumentParser(
        description="Collect test-set scores and choose the decision threshold."
    )
    parser.add_argument("--model", choices=MODELS, required=True, help="Model name")
    parser.add_argument("--category", choices=CATEGORIES, required=True, help="Category name")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--cost-fn", type=float, default=10.0, help="Cost of false negatives")
    parser.add_argument("--cost-fp", type=float, default=1.0, help="Cost of false positives")
    parser.add_argument(
        "--defect-rate",
        type=float,
        default=None,
        help="Expected share of defects on the line, strictly between 0 and 1."
        " Omit for a count-based cost.",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=20,
        help="Number of validation/test splits used to measure performance",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print(args)
    results = load_or_collect_scores(args.model, args.category, args.seed)
    measured = [
        evaluate_split(results, split_seed, args.cost_fn, args.cost_fp, args.defect_rate)[1]
        for split_seed in range(args.n_splits)
    ]
    summary = summarize(measured)

    rate = "none" if args.defect_rate is None else f"{args.defect_rate:g}"
    name = f"{args.model}_{args.category}_s{args.seed}"
    name +=f"_fn{args.cost_fn:g}_fp{args.cost_fp:g}_p{rate}.json"
    report_path = REPORTS_DIR / name

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_data = {
        "model": args.model,
        "category": args.category,
        "seed": args.seed,
        "cost_fn": args.cost_fn,
        "cost_fp": args.cost_fp,
        "defect_rate": args.defect_rate,
        "n_splits": args.n_splits,
        "summary": summary,
    }
    report_path.write_text(json.dumps(report_data, indent=2), encoding="utf-8")
    print(f"Summary written to {report_path}")

    final_threshold = select_threshold(
        *scores_and_labels(results),
        cost_fn=args.cost_fn,
        cost_fp=args.cost_fp,
        defect_rate=args.defect_rate,
    )
    save_threshold(final_threshold, THRESHOLD_PATH)
    print(f"Final threshold is {final_threshold.threshold:g}, saved to {THRESHOLD_PATH}")

    for metric, stat in summary.items():
        print(
            f">>  {metric}: mean={stat['mean']:.3f}, std={stat['std']:.3f}",
            f", min={stat['min']:.3f}, max={stat['max']:.3f}",
        )


if __name__ == "__main__":
    main()
