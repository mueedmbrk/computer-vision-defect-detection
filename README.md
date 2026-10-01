# Visual Defect Detection — Technical Assessment

## Overview

This project solves a simple manufacturing inspection problem: given an image of a product, decide whether it is **Normal** or **Defective**.

I approached the task as an image-classification problem because the requirement is to make an image-level decision. I kept the solution deliberately practical: a pretrained EfficientNet-B0 model, a reproducible training pipeline, proper evaluation, error analysis, and a small FastAPI service for inference.

The project is structured so the training code and inference service can be run separately.

---

## Project structure

```text
defect_detection_pro/
│
├── app/
│   └── main.py                 # FastAPI inference service
│
├── src/
│   ├── config.py               # Model/config settings
│   ├── dataset.py              # Dataset + transforms
│   ├── model.py                # EfficientNet-B0
│   └── utils.py                # Reproducibility/logging helpers
│
├── scripts/
│   ├── prepare_dataset.py      # Validate, inspect and split images
│   ├── train.py                # Model training
│   └── evaluate.py             # Test metrics + error analysis
│
├── tests/
│   └── test_api.py
│
├── docs/
│   └── architecture.svg
│
├── data/
│   └── raw/
│       ├── normal/
│       └── defective/
│
├── artifacts/
│   └── .gitkeep
│
├── Dockerfile
├── requirements.txt
├── README.md
├── INTERVIEW_NOTES.md
└── SUBMISSION_CHECKLIST.md
```

---

# Dataset used for the assessment demo

The original client dataset was not supplied with the assessment, so I used a
public manufacturing dataset as a reproducible substitute:

**PerceptiLabs — AC Piston Checks**

Source:
https://github.com/PerceptiLabs/AC-Piston-Checks

The repository contains images of normal air-conditioner pistons and two defect
categories. The source describes defects such as broken/out-of-shape parts and
oily, greasy or rusty stains. For this assessment, the two defect folders are
combined into the single required `Defective` class. This makes the dataset
closely aligned with the requested manufacturing Normal-vs-Defective task.

The repository also credits the underlying dataset to a Kaggle dataset.

Download it automatically:

```bash
python scripts/download_dataset.py
```

The downloader creates:

```text
data/raw/
├── normal/
└── defective/
```

This is explicitly a **public substitute dataset**, not the client's private
dataset. If the interviewer supplies the original dataset, I would replace
`data/raw/` and rerun the same pipeline.

# 1. Dataset preparation

The expected dataset is:

```text
data/raw/
├── normal/
│   ├── image_001.jpg
│   ├── image_002.jpg
│   └── ...
└── defective/
    ├── image_101.jpg
    ├── image_102.jpg
    └── ...
```

Before training, I validate the images and collect basic information such as:

- image dimensions;
- image mode;
- class;
- file hash;
- readable/corrupted status.

Exact duplicate files are removed from the manifest. This is useful because duplicate images can make a model look better during evaluation without actually improving generalization.

Run:

```bash
python scripts/prepare_dataset.py --input data/raw --output data/processed --seed 42
```

The script creates:

```text
data/processed/
├── train.csv
├── val.csv
└── test.csv
```

The split is:

- 70% training
- 15% validation
- 15% test

and is stratified by class.

### Important data-leakage consideration

If the dataset contains multiple images of the same physical product, batch, or production sequence, a normal random split can leak very similar images into different sets.

For that situation I would change the split to a **group-aware split**, using the product/batch/session identifier. I would inspect the supplied dataset for this before final reporting.

---

# 2. Preprocessing and augmentation

The model receives 224 × 224 RGB images.

For training I use:

- resize;
- random horizontal flip;
- limited vertical flip;
- small rotation;
- mild brightness/contrast/saturation changes;
- occasional light Gaussian blur;
- ImageNet normalization.

Validation and test images use only deterministic resize and normalization.

I intentionally keep augmentation moderate. In manufacturing, an augmentation is useful only if it represents a variation that can realistically happen on the production line. An aggressive transformation that removes or creates a defect would make the training data misleading.

---

# 3. Model selection

## EfficientNet-B0 + transfer learning

I selected **EfficientNet-B0** with ImageNet pretrained weights.

The main reasons were:

1. The assessment dataset is relatively small, so training a deep network from scratch would be unnecessary.
2. Pretrained features give the model a useful starting point.
3. EfficientNet-B0 is relatively lightweight compared with larger modern architectures.
4. It is practical for CPU inference and can also use GPU acceleration.
5. It gives a reasonable balance between accuracy, model size and inference time.

I replace the original classification head with a two-class output:

```text
0 → Normal
1 → Defective
```

I would consider a larger model only if the error analysis showed that the model capacity was actually the limiting factor.

---

# 4. Class imbalance

I check the class distribution before training.

If one class is substantially smaller, the training loader uses a `WeightedRandomSampler`. This gives minority-class examples a higher chance of being selected during training.

I do **not** oversample the validation or test sets.

This matters because the evaluation set should represent the real data distribution rather than an artificially balanced dataset.

---

# 5. Training

Install the environment:

### Windows

```powershell
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Linux/macOS

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Train:

```bash
python scripts/train.py \
  --train-csv data/processed/train.csv \
  --val-csv data/processed/val.csv \
  --output artifacts \
  --epochs 12 \
  --batch-size 16 \
  --seed 42
```

Windows PowerShell:

```powershell
python scripts/train.py `
  --train-csv data/processed/train.csv `
  --val-csv data/processed/val.csv `
  --output artifacts `
  --epochs 12 `
  --batch-size 16 `
  --seed 42
```

The best model is selected using validation F1 and saved as:

```text
artifacts/best_model.pt
```

The training configuration and history are also saved so the run can be reproduced.

---

# 6. Evaluation

After training:

```bash
python scripts/evaluate.py \
  --test-csv data/processed/test.csv \
  --checkpoint artifacts/best_model.pt \
  --output artifacts/evaluation
```

The evaluation produces:

```text
artifacts/evaluation/
├── metrics.json
├── classification_report.txt
├── confusion_matrix.csv
├── confusion_matrix.png
├── predictions.csv
└── errors/
    ├── false_positives/
    └── false_negatives/
```

The reported metrics are:

- Accuracy
- Precision
- Recall
- F1-score
- Macro precision/recall/F1
- Confusion matrix

I do not use accuracy as the only metric because a class-imbalanced defect dataset can make accuracy misleading.

---

# 7. False positives and false negatives

The error analysis separates the two important failure types.

### False positive

```text
Actual:    Normal
Predicted: Defective
```

This can result in a good product being rejected or sent for manual inspection.

### False negative

```text
Actual:    Defective
Predicted: Normal
```

This is particularly important in this application because a defective product may pass inspection.

The evaluation script automatically copies these images into separate folders. I would review them visually and look for patterns such as:

- very small defects;
- reflections or lighting changes;
- unusual orientation;
- background differences;
- low-quality images;
- defect types with too few examples;
- ambiguous labels.

I would only claim a particular cause after looking at the actual error images.

---

# 8. API

The model is exposed through FastAPI.

Start the service:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Health check:

```text
GET /health
```

Prediction:

```text
POST /predict
```

Example:

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "accept: application/json" \
  -F "file=@sample.jpg"
```

Example response:

```json
{
  "predicted_class": "Defective",
  "confidence": 0.9731,
  "probabilities": {
    "Normal": 0.0269,
    "Defective": 0.9731
  },
  "latency_ms": 42.17,
  "model_version": "efficientnet_b0_v1"
}
```

The API validates:

- file type;
- upload size;
- image decoding;
- model availability.

It also logs the prediction and inference latency.

---

# 9. Docker

Build:

```bash
docker build -t defect-detector-api .
```

Run:

```bash
docker run --rm -p 8000:8000 \
  -v "$(pwd)/artifacts:/app/artifacts:ro" \
  defect-detector-api
```

On Windows PowerShell:

```powershell
docker run --rm -p 8000:8000 `
  -v "${PWD}/artifacts:/app/artifacts:ro" `
  defect-detector-api
```

The image expects the trained checkpoint at:

```text
artifacts/best_model.pt
```

For a real deployment, I would store the model artifact in an artifact/model registry rather than rebuilding the container every time the model changes.

---

# 10. Testing

Run:

```bash
pytest -q
```

The current tests cover the API health endpoint and unsupported file types.

For a production rollout I would add:

- valid-image inference tests;
- corrupted-image tests;
- oversized-upload tests;
- model-loading tests;
- latency smoke tests.

---

# 11. Accuracy, latency and trade-offs

There are three practical trade-offs I considered.

### Accuracy

A larger model could potentially improve accuracy, but only if the data and error analysis justify the additional complexity.

### Latency

EfficientNet-B0 is intentionally small enough to make CPU inference practical. The API reports the actual inference request latency so it can be benchmarked on the deployment machine.

### Complexity

I chose a conventional PyTorch model and FastAPI service instead of introducing unnecessary infrastructure. That makes the solution easier to debug and hand over to another engineer.

---

# 12. Production considerations

Before using this system to make automatic production decisions, I would validate it on representative images from the real production line.

I would monitor:

- request volume;
- API errors;
- inference latency;
- prediction distribution;
- confidence distribution;
- class distribution changes;
- performance on periodically labeled production samples.

I would also monitor for changes caused by:

- a new camera;
- different lighting;
- a new product variant;
- camera repositioning;
- image compression;
- changes in the manufacturing process.

These changes can create data drift even when the software itself has not changed.

---

# 13. Threshold selection

The initial implementation uses the class with the highest predicted probability.

For production, I would not automatically assume that 0.50 is the correct operating threshold.

If missing a defect is significantly more expensive than sending a normal item for manual review, I would tune the defective-class threshold on validation data according to the actual business cost.

The final threshold should then be locked and evaluated on the untouched test set.

---

# 14. Limitations

This is an image-level classifier. It answers:

> Is this product normal or defective?

It does not answer:

> Where exactly is the defect?

If defect localization is required, I would move to object detection or segmentation and collect bounding-box or mask annotations.

Other limitations:

- small datasets can give uncertain performance estimates;
- unseen defect types may be missed;
- lighting/camera changes can affect performance;
- classification confidence should not be treated as a guarantee of correctness.

---

# 15. What I would do next with more time

If this were moving from assessment to production, my next steps would be:

1. Review every false positive and false negative.
2. Check for product/batch leakage.
3. Add difficult negative examples.
4. Validate on a separate production-day dataset.
5. Tune the operating threshold based on business costs.
6. Calibrate model probabilities.
7. Benchmark CPU and GPU latency.
8. Export to ONNX if the target environment benefits from it.
9. Add monitoring and model-version tracking.
10. Consider localization if operators need to see the defect position.

---

# 16. Final assessment results

The repository deliberately does **not** contain invented performance numbers.

Once the supplied assessment dataset is placed under `data/raw/`, run the preparation, training and evaluation commands above. The resulting `artifacts/evaluation/metrics.json`, confusion matrix and prediction file are the numbers and evidence I would submit.

This keeps the reported results tied to the actual assessment dataset rather than using example or fabricated metrics.

---

## Short explanation for the interviewer

> I treated the problem as binary image classification because the requested output is an image-level Normal/Defective decision. I used EfficientNet-B0 with ImageNet transfer learning because the dataset is small and I wanted a good balance between model quality and inference cost. I validated the images, removed exact duplicates, used a stratified train/validation/test split, and kept augmentation restricted to the training set. For imbalance, I used weighted sampling rather than changing the evaluation distribution. I evaluated precision, recall, F1 and the confusion matrix, then separated false positives and false negatives for visual inspection. Finally, I wrapped the trained model in FastAPI with input validation, logging, latency measurement and Docker packaging. The main production concern I would address next is validating the model against real production-line variation and choosing the operating threshold based on the cost of missed defects versus unnecessary manual inspections.
