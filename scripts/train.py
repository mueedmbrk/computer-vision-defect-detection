import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.dataset import DefectDataset, build_eval_transform, build_train_transform
from src.model import build_model
from src.utils import save_json, seed_everything, setup_logging


def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train(train)
    total_loss = 0.0
    all_true, all_pred = [], []

    for images, labels, _ in loader:
        images, labels = images.to(device), labels.to(device)

        with torch.set_grad_enabled(train):
            logits = model(images)
            loss = criterion(logits, labels)

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        all_true.extend(labels.detach().cpu().numpy())
        all_pred.extend(preds.detach().cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    f1 = f1_score(all_true, all_pred, average="binary", zero_division=0)
    return avg_loss, f1


def make_loader(csv_path, transform, batch_size, train=False):
    dataset = DefectDataset(csv_path, transform=transform)

    if train:
        df = pd.read_csv(csv_path)
        counts = df["label"].value_counts().sort_index()
        class_weights = {int(label): len(df) / (2 * count) for label, count in counts.items()}
        sample_weights = df["label"].map(class_weights).astype(float).to_numpy()
        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=len(sample_weights),
            replacement=True,
        )
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=0)

    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", required=True, type=Path)
    parser.add_argument("--val-csv", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    seed_everything(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(args.output / "logs", "training")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device=%s", device)

    train_loader = make_loader(
        args.train_csv,
        build_train_transform(),
        args.batch_size,
        train=True,
    )
    val_loader = make_loader(
        args.val_csv,
        build_eval_transform(),
        args.batch_size,
        train=False,
    )

    model = build_model(pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )

    best_f1 = -1.0
    history = []

    for epoch in range(1, args.epochs + 1):
        train_loss, train_f1 = run_epoch(
            model, train_loader, criterion, optimizer, device, train=True
        )
        val_loss, val_f1 = run_epoch(
            model, val_loader, criterion, optimizer, device, train=False
        )
        scheduler.step(val_f1)

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_f1": train_f1,
            "val_loss": val_loss,
            "val_f1": val_f1,
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(record)

        logger.info(
            "Epoch %02d | train_loss=%.4f train_f1=%.4f | val_loss=%.4f val_f1=%.4f",
            epoch, train_loss, train_f1, val_loss, val_f1
        )

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_name": "efficientnet_b0_v1",
                    "class_names": ["Normal", "Defective"],
                    "image_size": 224,
                    "best_val_f1": best_f1,
                    "seed": args.seed,
                },
                args.output / "best_model.pt",
            )
            logger.info("Saved new best checkpoint.")

    save_json(history, args.output / "training_history.json")
    save_json(
        {
            "seed": args.seed,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "device": device,
            "model": "EfficientNet-B0 transfer learning",
            "imbalance_strategy": "WeightedRandomSampler",
            "best_validation_f1": best_f1,
        },
        args.output / "training_config.json",
    )

    print(json.dumps(history[-1], indent=2))


if __name__ == "__main__":
    main()
