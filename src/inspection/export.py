"""Export a trained anomalib model to ONNX, with raw (unnormalized) scores.

Usage: python -m inspection.export --model patchcore --category screw --seed 42
"""

import argparse

import numpy as np
from PIL import Image

from inspection.config import (
    CATEGORIES,
    DATA_ROOT,
    EXPORT_DIR,
    IMAGE_SIZE,
    MODEL_PATH,
    MODELS,
    THRESHOLD_PATH,
)
from inspection.evaluate import build_model, load_scores, scores_path
from inspection.inference import preprocess
from inspection.threshold import load_threshold


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and export anomalib models on MVTec AD categories."
    )
    parser.add_argument(
        "--model", choices=MODELS, required=True, help="Name of the model to export."
    )
    parser.add_argument(
        "--category", choices=CATEGORIES, required=True, help="MVTec AD category to train on."
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument(
        "--skip-export",
        action="store_true",
        help="Reuse MODEL_PATH instead of training and exporting.",
    )
    return parser.parse_args()


def export(model_name, category, seed):
    import onnx
    from anomalib.data import MVTecAD
    from anomalib.engine import Engine
    from lightning import seed_everything

    seed_everything(seed)

    # 1. Load the dataset
    datamodule = MVTecAD(root=DATA_ROOT, category=category, num_workers=0)

    # 2. Create the model
    model = build_model(model_name)

    # 3. Create the engine
    engine = Engine()

    # 4. Train and export
    engine.fit(model=model, datamodule=datamodule)
    onnx_path = engine.export(
        model=model,
        export_type="onnx",
        export_root=EXPORT_DIR,
        onnx_kwargs={"opset_version": 18, "dynamic_shapes": None},
        input_size=IMAGE_SIZE,
    )
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    # 5. Merge graph + external weights into one file
    loaded_model = onnx.load(onnx_path)
    onnx.save(loaded_model, MODEL_PATH)
    return MODEL_PATH


def check_onnx(onnx_path):
    import onnxruntime as ort

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    print("Inputs:")
    for input_onnx in session.get_inputs():
        print(f"  {input_onnx.name}: {input_onnx.shape}, {input_onnx.type}")
    print("Outputs:")
    for output_onnx in session.get_outputs():
        print(f"  {output_onnx.name}: {output_onnx.shape}, {output_onnx.type}")

    print("Run on random input:")
    input_data = np.random.rand(1, 3, 256, 256).astype(np.float32)
    output = session.run(None, {session.get_inputs()[0].name: input_data})
    outputs = dict(zip([o.name for o in session.get_outputs()], output, strict=True))
    score = outputs["pred_score"]
    print(f"  pred_score = {score[0]:.4f}")


def compare_with_cache(onnx_path, model_name, category, seed):
    import onnxruntime as ort

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    reference = load_scores(scores_path(model_name, category, seed))
    threshold = load_threshold(THRESHOLD_PATH)

    ref_scores = []
    onnx_scores = []
    for path, _, _, ref_score in reference:
        image = preprocess(Image.open(path), size=IMAGE_SIZE)
        onnx_score = session.run(["pred_score"], {input_name: image})[0][0]
        ref_scores.append(ref_score)
        onnx_scores.append(onnx_score)

    ref_scores = np.array(ref_scores)
    onnx_scores = np.array(onnx_scores)

    max_abs_diff = np.max(np.abs(onnx_scores - ref_scores))
    max_rel_diff = np.max(np.abs(onnx_scores - ref_scores) / np.abs(ref_scores))
    mean_signed_diff = np.mean(onnx_scores - ref_scores)
    num_mismatches = np.sum((onnx_scores >= threshold) != (ref_scores >= threshold))

    diff = np.abs(onnx_scores - ref_scores)
    idx = np.argmax(diff)  # argmax of the absolute difference

    print(f"Max absolute difference: {max_abs_diff:.6f}")
    print(f"Mean signed difference: {mean_signed_diff:+.6f}")
    print(f"Max relative difference: {max_rel_diff:.6f}")
    print(
        f"Worst image: {reference[idx][0]}  diff={diff[idx]:.6f}  cache={ref_scores[idx]:.6f}",
        f"onnx={onnx_scores[idx]:.6f}",
        sep="",
    )
    print(f"Number of mismatches with threshold {threshold}: {num_mismatches}")


def main():
    args = parse_args()
    onnx_path = MODEL_PATH
    if not args.skip_export:
        onnx_path = export(args.model, args.category, args.seed)
    print(onnx_path)
    check_onnx(onnx_path)
    compare_with_cache(onnx_path, args.model, args.category, args.seed)


if __name__ == "__main__":
    main()
