# Industrial anomaly inspection

Unsupervised visual inspection on [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad): the model learns from good parts only and flags anything that deviates.

The focus is not on chasing state-of-the-art AUROC (most methods are above 98% on MVTec AD) but on what a production line needs:

- **a decision threshold justified by costs**: a missed defect costs more than a false reject,
- **CPU latency** compatible with a line rate,
- **a deployable service**: Dockerized API, deployed on Google Cloud Run.

> Status: work in progress. See [docs/roadmap.md](docs/roadmap.md).

<!-- TODO: demo GIF + live demo link -->

## Results

<!-- TODO(week 1): image / pixel AUROC per category and model -->

| Category | Type | PatchCore AUROC (img / px) | PaDiM AUROC (img / px) |
|---|---|---|---|
| bottle | object | | |
| screw | object | | |
| carpet | texture | | |
| metal_nut | object | | |

## Choosing the decision threshold

<!-- TODO(week 2): recall vs false-reject-rate curve, chosen operating point, sensitivity to the cost ratio -->

The threshold minimises `cost_fn * missed_defects + cost_fp * false_rejects` (`src/inspection/threshold.py`), instead of maximising F1, which treats both errors as equal.

## CPU latency

<!-- TODO(week 2): median / p95 ms per image, PyTorch vs ONNX Runtime vs OpenVINO, CPU model -->

## Limitations

<!-- TODO: what fails and why (heatmaps of failure cases), what the benchmark does not say about a real line -->

## Usage

```bash
# Training / evaluation (local, heavy dependencies)
pip install -e ".[train,dev]"
python -m inspection.train --model patchcore --category bottle --export-onnx
python -m inspection.evaluate --model patchcore --category bottle --cost-fn 10

# API
pip install -e ".[serve,dev]"
uvicorn inspection.api:app --reload
curl -F "file=@image.png" http://localhost:8000/predict

# Docker
docker compose up --build
```

Data location is set by `DATA_ROOT` (see `src/inspection/config.py`).

## Project layout

```
src/inspection/
  config.py      paths and settings (env-overridable)
  train.py       anomalib training + ONNX export
  evaluate.py    test scores -> cost-based threshold
  threshold.py   operating points and threshold selection
  benchmark.py   CPU latency benchmark
  inference.py   ONNX Runtime inference (no PyTorch)
  api.py         FastAPI service
tests/           unit tests (run in CI)
docs/            roadmap, Cloud Run deployment
```

## Licence

Code: MIT. MVTec AD is distributed under CC BY-NC-SA 4.0 (non-commercial use).
