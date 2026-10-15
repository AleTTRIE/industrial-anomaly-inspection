# Roadmap

## Week 1: baseline
- [ ] Install training deps in a venv: `pip install -e ".[train,dev]"` (check the torch CPU/GPU wheel)
- [ ] Download MVTec AD to `DATA_ROOT` (default: ../datasets/MVTecAD, next to the repo), ~5 GB
- [ ] Check `train.py` imports against the installed anomalib version
- [ ] PatchCore + PaDiM on bottle, screw, carpet, metal_nut: image and pixel AUROC
- [ ] Results table in the README

## Week 2: industrial angle
- [ ] Hold-out split for threshold selection (no leak from the test set)
- [ ] Recall vs false-reject-rate curve, cost-based threshold, sensitivity to cost_fn/cost_fp
- [ ] Latency benchmark: PyTorch vs ONNX Runtime vs OpenVINO (median, p95, CPU model)
- [ ] Heatmap figures: good detections and failure cases
- [ ] Put the GitHub link back on the CV once the README is clean

## Week 3: serving
- [ ] Export the best model to ONNX, check output names, fix `inference.py` TODOs
- [ ] Heatmap returned by the API
- [ ] Docker image tested locally (`docker compose up`)
- [ ] Deploy to Cloud Run, live demo link in the README

## Later (only if offers ask for it)
- [ ] CD on tags from GitHub Actions
- [ ] MLflow experiment tracking
- [ ] DVC for data versioning
- [ ] Drift monitoring (Evidently) on degraded images
