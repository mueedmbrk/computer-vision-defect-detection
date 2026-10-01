# Final Submission Report

## Project
Visual Defect Detection — Normal vs Defective

## Objective
Automatically classify a product image as Normal or Defective and expose the trained model through an HTTP API.

## Dataset
- Source: public PerceptiLabs AC Piston Checks dataset, used because no client dataset was supplied
- Dataset source: https://github.com/PerceptiLabs/AC-Piston-Checks
- Normal images are kept as `normal`; defect_1 and defect_2 are combined into `defective`
- For a client handoff, replace this substitute with the supplied client images and rerun the pipeline
- Total images: **fill from `prepare_dataset.py` output**
- Normal: **fill**
- Defective: **fill**
- Invalid images removed: **fill**
- Exact duplicates removed: **fill**

## Split
- Training: 70%
- Validation: 15%
- Test: 15%
- Method: stratified split
- Group-aware split: use if product/batch/session IDs are available

## Model
EfficientNet-B0 with ImageNet transfer learning.

## Training
- Input size: 224 × 224
- Optimizer: AdamW
- Learning rate: 2e-4
- Weight decay: 1e-4
- Augmentation: rotation, flips where valid, mild color changes and limited blur
- Imbalance strategy: WeightedRandomSampler when required
- Best checkpoint selection: validation F1
- Seed: 42

## Test Results

Fill these from `artifacts/evaluation/metrics.json`:

| Metric | Result |
|---|---:|
| Accuracy | |
| Defective Precision | |
| Defective Recall | |
| Defective F1 | |
| Macro Precision | |
| Macro Recall | |
| Macro F1 | |
| False Positives | |
| False Negatives | |

## Confusion Matrix

The generated file is:

```text
artifacts/evaluation/confusion_matrix.png
```

## Error Analysis

### False positives
Review:

```text
artifacts/evaluation/errors/false_positives/
```

Write the observed pattern here.

### False negatives
Review:

```text
artifacts/evaluation/errors/false_negatives/
```

Write the observed pattern here.

## API

Endpoint:

```text
POST /predict
```

Response contains:

- predicted class;
- confidence;
- class probabilities;
- latency;
- model version.

## Deployment

The service is packaged with Docker.

## Known limitations

The classifier predicts only whether the complete image is normal or defective. It does not localize the defect.

Performance should be revalidated if the camera, lighting, product type or production environment changes.
