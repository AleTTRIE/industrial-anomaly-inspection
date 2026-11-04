import collections
from pathlib import Path

from sklearn.model_selection import train_test_split

from inspection.config import DATA_ROOT


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


def main():
    model = "patchcore"  # or "padim"
    category = "screw"  # or another MVTec AD category
    seed = 42
    results = collect_scores(model, category, seed)
    print(
        f"Collected {len(results)} scores for",
        f"model '{model}' on category '{category}' with seed {seed}.",
    )

    present_results(results)

    split_seed = 42
    validation, test = split_results(results, test_size=0.5, seed=split_seed)
    print("\nValidation set:")
    present_results(validation)
    print("\nTest set:")
    present_results(test)

    # The two sets should be disjoint and cover all results
    assert set(validation) & set(test) == set()
    assert set(validation) | set(test) == set(results)


if __name__ == "__main__":
    main()
