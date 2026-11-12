"""Train and test an anomalib model on one MVTec AD category.

Fits the model on the category's "good" training images, runs the anomalib test
loop on the test split, and writes the metrics (image and pixel AUROC, F1) to
results/{model}_{category}_{seed}_results.json.

Usage: python -m inspection.train --model patchcore --category screw --seed 42

anomalib is imported inside train() because CI does not install it.
"""

import argparse
import json

from inspection.config import CATEGORIES, DATA_ROOT, MODELS, RESULTS_DIR


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and test anomalib models on MVTec AD categories."
    )
    parser.add_argument(
        "--model", choices=MODELS, required=True, help="Name of the model to train."
    )
    parser.add_argument(
        "--category", choices=CATEGORIES, required=True, help="MVTec AD category to train on."
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    return parser.parse_args()


def train(model_name, category, seed):
    from anomalib.data import MVTecAD
    from anomalib.engine import Engine
    from anomalib.models import Padim, Patchcore
    from lightning import seed_everything

    seed_everything(seed)

    # 1. Load the dataset
    datamodule = MVTecAD(root=DATA_ROOT, category=category, num_workers=0)

    # 2. Create the model
    if model_name == "patchcore":
        model = Patchcore()
    elif model_name == "padim":
        model = Padim()
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    # 3. Create the engine
    engine = Engine()

    # 4. Train and evaluate
    engine.fit(model=model, datamodule=datamodule)
    results = engine.test(model=model, datamodule=datamodule)
    return results[0]


def main():
    args = parse_args()
    print(args)
    results = train(args.model, args.category, args.seed)
    results["model"] = args.model
    results["category"] = args.category
    results["seed"] = args.seed
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / f"{args.model}_{args.category}_{args.seed}_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
