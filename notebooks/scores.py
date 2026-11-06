import collections
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from inspection.config import DATA_ROOT, RESULTS_DIR
from inspection.threshold import evaluate_threshold, select_threshold


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


def save_scores(results, path, model_name, category, seed):
    import anomalib

    path.parent.mkdir(parents=True, exist_ok=True)
    dict_to_save = {
        "model": model_name,
        "category": category,
        "seed": seed,
        "normalize": False,
        "anomalib_version": anomalib.__version__,
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


def score(results):
    """Return min max mean for each type of defect in the results."""
    # Group results by defect type
    grouped = collections.defaultdict(list)
    for _, defect_type, _, score in results:
        grouped[defect_type].append(score)

    # Calculate statistics for each defect type
    stats = {}
    for defect_type, scores in grouped.items():
        stats[defect_type] = {
            "min": min(scores),
            "max": max(scores),
            "mean": sum(scores) / len(scores),
        }
    return stats


def split_results(results, test_size=0.5, seed=42):
    """Split results into validation and test sets, stratified by defect type."""
    defect_types = [r[1] for r in results]

    validation, test = train_test_split(
        results,
        test_size=test_size,
        random_state=seed,
        stratify=defect_types,
    )

    # The two sets should be disjoint and cover all results
    assert set(validation) & set(test) == set()
    assert set(validation) | set(test) == set(results)

    return validation, test


def present_results(results):
    print(f"Total results: {len(results)}")
    print(collections.Counter([r[1] for r in results]))  # Count of ground truth labels
    print("Score statistics:")
    for defect_type, stat in score(results).items():
        print(
            f"  {defect_type}:",
            f" min={stat['min']:.3f},",
            f" max={stat['max']:.3f},",
            f" mean={stat['mean']:.3f}",
        )
    print()


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
            "std": np.std(values),
        }
    return summary


def main():
    model = "patchcore"  # or "padim"
    category = "screw"  # or another MVTec AD category
    seed = 42
    results = load_or_collect_scores(model, category, seed)
    print(
        f"Using {len(results)} scores for",
        f"model '{model}' on category '{category}' with seed {seed}.",
    )
    present_results(results)

    for cost_fn, cost_fp in [(10.0, 1.0), (100.0, 1.0)]:
        for defect_rate in [None, 0.01]:
            measured = [
                evaluate_split(results, split_seed, cost_fn, cost_fp, defect_rate)[1]
                for split_seed in range(20)
            ]
            print(f"cost_fn={cost_fn}, cost_fp={cost_fp}, defect_rate={defect_rate}:")
            summary = summarize(measured)
            for metric, stat in summary.items():
                print(
                    f">>  {metric}: mean={stat['mean']:.3f}, std={stat['std']:.3f}",
                    f", min={stat['min']:.3f}, max={stat['max']:.3f}",
                )


if __name__ == "__main__":
    main()
