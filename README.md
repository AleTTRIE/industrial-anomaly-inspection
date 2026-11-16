# Industrial anomaly inspection

![CI](https://github.com/AleTTRIE/industrial-anomaly-inspection/actions/workflows/ci.yml/badge.svg)

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

F1 is not reported: anomalib picks its threshold on the test images themselves, so the resulting F1 is biased and optimistic. See [Choosing the decision threshold](#choosing-the-decision-threshold) for a threshold chosen on held-out data.

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

The threshold minimises the expected cost per part (`src/inspection/threshold.py`), instead of maximising F1, which treats both errors as equal:

`cost = cost_fn * p * FN_rate + cost_fp * (1 - p) * FP_rate`

- `cost_fn`: cost of a missed defect,
- `cost_fp`: cost of a false reject,
- `FN_rate`: share of defective parts that are missed (1 - recall),
- `FP_rate`: share of good parts that are rejected,
- `p`: expected share of defective parts on the line (`--defect-rate`). Without it, the cost counts errors on the evaluation images.

The threshold is chosen on a validation half and measured on the other half, over 20 stratified validation/test splits of the screw test set (41 good parts, 119 defective). The threshold written to `models/threshold.json` is then chosen on all 160 images.

PatchCore on screw, seed 42, `cost_fp = 1`. Mean (min–max) on the test halves of 20 splits:

| cost_fn | defect_rate | Threshold | Recall | False reject rate |
|---------|-------------|-----------|--------|-------------------|
| 10  | none | 29.73 (29.46–30.38) | 0.987 (0.949–1.000) | 27.5% (19.0–45.0%) |
| 100 | none | 29.73 (29.46–30.38) | 0.987 (0.949–1.000) | 27.5% (19.0–45.0%) |
| 10  | 0.01 | 34.50 (31.98–35.97) | 0.737 (0.550–0.898) | 6.6% (0.0–25.0%) |
| 100 | 0.01 | 31.96 (29.94–33.14) | 0.875 (0.797–0.983) | 14.1% (0.0–28.6%) |

- **`none` is misleading**: it assumes the proportion of the test images, about 3 defective parts for 1 good part. Defects dominate, so the threshold drops and rejects 27.5% of good parts. `cost_fn = 10` is already enough to reject any threshold that misses a defect, so `cost_fn = 100` gives the same threshold.
- **`p = 0.01` makes a big difference**: the threshold goes up, but with `cost_fn = 10` we miss 26% of defects. A missed defect weighs 0.01 × 10 = 0.1 against 0.99 × 1 = 0.99 for a false reject, so missing a defect is cheaper. With `cost_fn = 100`, the weights are 1 and 0.99: both errors count almost equally, and the threshold settles on a compromise.
- **Small samples**: each test half has about 20 good parts, so one false reject is worth 5%. That is why the false reject rate ranges from 0% to 28.6% between splits: the measurement is imprecise, not only the model.
- **Conclusion**: with 12.5% missed defects and 14% false rejects, PatchCore is not ready for a screw line. Two things are missing: more good parts in the evaluation set, to measure the false reject rate precisely, and a better model, since the scores of good and defective screws overlap.

To reproduce (scores are cached in `results/scores/` after the first run, which trains the model):
```powershell
python -m inspection.evaluate --model patchcore --category screw --cost-fn 100 --defect-rate 0.01
The summary over the splits goes to reports/, the final threshold to models/threshold.json.

## CPU latency

<!-- TODO(week 2): median / p95 ms per image, PyTorch vs ONNX Runtime vs OpenVINO, CPU model -->

## Limitations

<!-- TODO: what fails and why (heatmaps of failure cases), what the benchmark does not say about a real line -->

## Usage

```bash
# Training / evaluation (local, heavy dependencies)
pip install -e ".[train,dev]"
python -m inspection.train --model patchcore --category bottle
python -m inspection.evaluate --model patchcore --category bottle --cost-fn 100 --defect-rate 0.01

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
  train.py       anomalib training and test metrics
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
