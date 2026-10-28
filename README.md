# Industrial anomaly inspection

Unsupervised visual inspection on [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad): the model learns from good parts only and flags anything that deviates.

The focus is not on chasing state-of-the-art AUROC (most methods are above 98% on MVTec AD) but on what a production line needs:

- **a decision threshold justified by costs**: a missed defect costs more than a false reject,
- **CPU latency** compatible with a line rate,
- **a deployable service**: Dockerized API, deployed on Google Cloud Run.

> Status: work in progress. See [docs/roadmap.md](docs/roadmap.md).

<!-- TODO: demo GIF + live demo link -->

## Results

| Model | Category | Type | Mean Image AUROC ± Std | Mean Pixel AUROC ± Std |
|-------|----------|------|------------------------|------------------------|
| padim | bottle | object | 0.997 ± 0.002 | 0.979 ± 0.001 |
| patchcore | bottle | object | 1.000 ± 0.000 | 0.986 ± 0.000 |
| padim | carpet | texture | 0.969 ± 0.013 | 0.989 ± 0.001 |
| patchcore | carpet | texture | 0.987 ± 0.001 | 0.991 ± 0.000 |
| padim | metal_nut | object | 0.956 ± 0.011 | 0.953 ± 0.008 |
| patchcore | metal_nut | object | 0.998 ± 0.001 | 0.987 ± 0.000 |
| padim | screw | object | 0.792 ± 0.038 | 0.977 ± 0.002 |
| patchcore | screw | object | 0.964 ± 0.003 | 0.989 ± 0.000 |

Runs use anomalib 2.7.0 with its default settings and 5 fixed seeds (9, 24, 42, 54, 85); the table shows the mean ± standard deviation over seeds.
Both models use anomalib defaults, which differ: PatchCore uses a WideResNet-50 backbone, PaDiM a ResNet-18.

F1 is not reported: anomalib picks its threshold on the test images themselves, so the resulting F1 is biased and optimistic. A threshold chosen on held-out data comes in week 2.

To reproduce the runs (PowerShell):
```powershell
foreach ($m in "patchcore","padim") {
    foreach ($s in 9, 24, 42, 54, 85) {
        foreach ($c in "bottle","screw","carpet","metal_nut") {
            python -m inspection.train --model $m --category $c --seed $s
        }
    }
}
```

To reproduce the table (PowerShell):
```powershell
python -m inspection.summarize
```

- PatchCore, which compares each patch to its whole memory bank, outperforms PaDiM on objects whose position varies (screw, metal_nut). (part of the gap may come from the backbone)
- For textures like carpet, PatchCore is ahead and more stable on image AUROC, PaDiM's score varies more. As for pixel, both models handle the texture well (carpet, no background): the pattern repeats over the whole image, so the position in the image barely matters.
- PaDiM trains about 3 times faster: PatchCore has to build its memory bank and then a coreset, which takes time, while PaDiM has no coreset to build. (4 categories x 5 seeds : 15min for PaDiM vs 45min for PatchCore with a RTX 3060)
- Changing the seed moves the score from 0 to 0.003 for PatchCore and 0.002 to 0.038 for PaDiM, so:
  - bottle (1.000 ± 0.000 vs 0.997 ± 0.002): the gap (0.003) is about the size of the noise, so neither model can be called better;
  - screw (0.964 ± 0.003 vs 0.792 ± 0.038): the gap (0.17) is more than 4 times the noise, so the difference is real.
- Changing the seed has different impacts on the models, for PaDiM, the impact can be up to 10 times larger than for PatchCore. So an unstable model has to be re-validated after every retraining.

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
python -m inspection.train --model patchcore --category bottle
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
