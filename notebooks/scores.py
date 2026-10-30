import collections
from pathlib import Path

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


def main():
    model = "patchcore"  # or "padim"
    category = "bottle"  # or another MVTec AD category
    seed = 42
    results = collect_scores(model, category, seed)
    print(
        f"Collected {len(results)} scores for",
        f"model '{model}' on category '{category}' with seed {seed}.",
    )
    print(collections.Counter([r[1] for r in results]))  # Count of ground truth labels
    print("Score statistics:")
    for defect_type, stat in score(results).items():
        print(
            f"  {defect_type}:",
            f" min={stat['min']:.3f},",
            f" max={stat['max']:.3f},",
            f" mean={stat['mean']:.3f}",
        )


if __name__ == "__main__":
    main()
