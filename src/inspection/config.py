"""Paths and settings, overridable through environment variables."""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# Datasets live outside the repo (MVTec AD is ~5 GB), next to it by default.
DATA_ROOT = Path(os.environ.get("DATA_ROOT", REPO_ROOT.parent / "datasets" / "MVTecAD"))

# Training runs, checkpoints, exported models.
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", REPO_ROOT / "results"))

# Figures and tables referenced by the README.
REPORTS_DIR = REPO_ROOT / "reports"

# Serving: exported ONNX model + chosen decision threshold.
MODEL_PATH = Path(os.environ.get("MODEL_PATH", REPO_ROOT / "models" / "model.onnx"))
THRESHOLD_PATH = Path(os.environ.get("THRESHOLD_PATH", REPO_ROOT / "models" / "threshold.json"))

# Starting categories: at least one texture and one object.
CATEGORIES_TYPES = {
    "bottle": "object",
    "screw": "object",
    "carpet": "texture",
    "metal_nut": "object",
}
CATEGORIES = list(CATEGORIES_TYPES.keys())
MODELS = ["patchcore", "padim"]
