# Submission Checklist

## Included in the repository
- [x] Dataset validation and preparation script
- [x] Duplicate detection
- [x] Stratified train/validation/test split
- [x] Training preprocessing and augmentation
- [x] Class imbalance handling
- [x] Transfer-learning model
- [x] Validation-based checkpoint selection
- [x] Precision / recall / F1 / accuracy evaluation
- [x] Confusion matrix
- [x] False-positive and false-negative extraction
- [x] FastAPI inference endpoint
- [x] Input validation and structured errors
- [x] Logging
- [x] Inference latency
- [x] Health endpoint
- [x] Docker setup
- [x] Automated tests
- [x] Architecture diagram
- [x] GitHub CI
- [x] Setup instructions
- [x] Interview notes
- [x] Final report template

## Before sending to the interviewer
- [x] Public assessment substitute downloader (`scripts/download_dataset.py`)
- [x] Public Normal/Defective dataset source documented
- [ ] Replace substitute dataset with client data if/when supplied
- [ ] Run dataset preparation
- [ ] Inspect class counts and image quality
- [ ] Check whether product/batch/session grouping is needed
- [ ] Train the model
- [ ] Evaluate the untouched test set
- [ ] Review false positives and false negatives
- [ ] Fill `docs/final_submission_report.md`
- [ ] Start the API and test a real image
- [ ] Build and test the Docker image
- [ ] Record the 2–3 minute demo
- [ ] Push the final repository to GitHub

## Do not put in GitHub
- API keys or passwords
- private client data unless explicitly permitted
- `.venv`
- raw confidential images
- unnecessary cache files

## Interview questions to be ready for
1. Why classification instead of detection?
2. Why transfer learning?
3. Why EfficientNet-B0?
4. How did you handle class imbalance?
5. What is a false negative here?
6. How did you prevent data leakage?
7. Why are precision, recall and F1 useful?
8. How would you tune the threshold?
9. How would you reduce inference latency?
10. What would you monitor after deployment?
