import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import argparse
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader

from src.dataset import DefectDataset, build_eval_transform
from src.model import load_checkpoint
from src.utils import save_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-csv", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("artifacts/evaluation"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "errors" / "false_positives").mkdir(parents=True, exist_ok=True)
    (args.output / "errors" / "false_negatives").mkdir(parents=True, exist_ok=True)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_checkpoint(args.checkpoint, device=device)
    model.to(device).eval()

    dataset = DefectDataset(args.test_csv, transform=build_eval_transform())
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    rows = []
    with torch.inference_mode():
        for images, labels, paths in loader:
            logits = model(images.to(device))
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = probs.argmax(axis=1)

            for i, path in enumerate(paths):
                true_label = int(labels[i])
                pred_label = int(preds[i])
                row = {
                    "path": path,
                    "true_label": true_label,
                    "predicted_label": pred_label,
                    "confidence": float(probs[i, pred_label]),
                    "normal_probability": float(probs[i, 0]),
                    "defective_probability": float(probs[i, 1]),
                    "error_type": (
                        "false_positive" if true_label == 0 and pred_label == 1
                        else "false_negative" if true_label == 1 and pred_label == 0
                        else ""
                    ),
                }
                rows.append(row)

    pred_df = pd.DataFrame(rows)
    y_true = pred_df["true_label"].to_numpy()
    y_pred = pred_df["predicted_label"].to_numpy()

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[1], average="binary", zero_division=0
    )
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "defective_precision": float(precision),
        "defective_recall": float(recall),
        "defective_f1": float(f1),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "test_samples": int(len(pred_df)),
        "false_positives": int(((y_true == 0) & (y_pred == 1)).sum()),
        "false_negatives": int(((y_true == 1) & (y_pred == 0)).sum()),
        "checkpoint": str(args.checkpoint),
        "model": checkpoint.get("model_name", "unknown"),
    }

    save_json(metrics, args.output / "metrics.json")
    pred_df.to_csv(args.output / "predictions.csv", index=False)

    report = classification_report(
        y_true, y_pred, target_names=["Normal", "Defective"], zero_division=0
    )
    (args.output / "classification_report.txt").write_text(report, encoding="utf-8")

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    pd.DataFrame(
        cm,
        index=["Actual Normal", "Actual Defective"],
        columns=["Pred Normal", "Pred Defective"],
    ).to_csv(args.output / "confusion_matrix.csv")

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm)
    ax.set_xticks([0, 1], ["Normal", "Defective"])
    ax.set_yticks([0, 1], ["Normal", "Defective"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(args.output / "confusion_matrix.png", dpi=180)
    plt.close(fig)

    for _, row in pred_df[pred_df["error_type"] != ""].iterrows():
        src = Path(row["path"])
        prefix = f"{row['confidence']:.3f}_{src.name}"
        if row["error_type"] == "false_positive":
            dst = args.output / "errors" / "false_positives" / prefix
        else:
            dst = args.output / "errors" / "false_negatives" / prefix
        try:
            shutil.copy2(src, dst)
        except OSError:
            pass

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
