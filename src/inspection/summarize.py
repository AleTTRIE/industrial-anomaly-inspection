import json
import statistics
from pathlib import Path
from typing import Any

from inspection.config import CATEGORIES_TYPES, RESULTS_DIR


def extract_results_json(results_dir: Path = RESULTS_DIR) -> list[dict[str, Any]]:
    """Read JSON files from the project's results directory."""
    return [
        json.loads(path.read_text(encoding="utf-8")) for path in sorted(results_dir.glob("*.json"))
    ]


def group_by_run(results: list[dict[str, Any]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    """
    Group results by model and category.
    returns a dictionary where keys are tuples of (model, category) and values are lists of results.
    """
    grouped_results = {}
    for result in results:
        key = (result["model"], result["category"])
        if key not in grouped_results:
            grouped_results[key] = []
        grouped_results[key].append(result)
    return grouped_results


def summarize(
    groups: dict[tuple[str, str], list[dict[str, Any]]],
) -> dict[tuple[str, str], dict[str, float]]:
    """
    Summarize results by calculating the mean and the standard deviation for each group.
    Returns a dictionary where keys are tuples of (model, category) and values are dictionaries
    with the mean and the standard deviation of image_AUROC and pixel_AUROC.
    """
    summary = {}
    for key, results in groups.items():
        image_auroc = [result["image_AUROC"] for result in results]
        pixel_auroc = [result["pixel_AUROC"] for result in results]
        mean_image_auroc = statistics.mean(image_auroc)
        mean_pixel_auroc = statistics.mean(pixel_auroc)
        std_image_auroc = statistics.stdev(image_auroc) if len(image_auroc) > 1 else 0.0
        std_pixel_auroc = statistics.stdev(pixel_auroc) if len(pixel_auroc) > 1 else 0.0
        summary[key] = {
            "mean_image_AUROC": mean_image_auroc,
            "mean_pixel_AUROC": mean_pixel_auroc,
            "std_image_AUROC": std_image_auroc,
            "std_pixel_AUROC": std_pixel_auroc,
        }
    return summary


def format_score(mean, std):
    """Format the mean and standard deviation as a string with three decimal places."""
    return f"{mean:.3f} ± {std:.3f}"


def main():
    """
    Main function to extract, group, summarize results and print them in a Markdown table format.
    """

    results = extract_results_json()
    grouped_results = group_by_run(results)
    # Sort by category, then by model
    summary = sorted(summarize(grouped_results).items(), key=lambda x: (x[0][1], x[0][0]))

    # Print the header of the Markdown table
    print("| Model | Category | Type | Mean Image AUROC ± Std | Mean Pixel AUROC ± Std |")
    print("|-------|----------|------|------------------------|------------------------|")

    # Print each row of the table
    for (model, category), stats in summary:
        type_ = CATEGORIES_TYPES[category]
        image = format_score(stats["mean_image_AUROC"], stats["std_image_AUROC"])
        pixel = format_score(stats["mean_pixel_AUROC"], stats["std_pixel_AUROC"])
        print(f"| {model} | {category} | {type_} | {image} | {pixel} |")


if __name__ == "__main__":
    main()
