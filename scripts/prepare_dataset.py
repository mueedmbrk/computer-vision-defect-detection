import argparse
import hashlib
from pathlib import Path

import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
LABELS = {"normal": 0, "defective": 1}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect(input_dir: Path):
    rows = []
    seen_hashes = set()

    for folder_name, label in LABELS.items():
        folder = input_dir / folder_name
        if not folder.exists():
            raise FileNotFoundError(f"Missing class directory: {folder}")

        for path in sorted(folder.rglob("*")):
            if path.suffix.lower() not in EXTENSIONS:
                continue

            try:
                file_hash = sha256_file(path)
                if file_hash in seen_hashes:
                    continue

                with Image.open(path) as image:
                    image.verify()

                with Image.open(path) as image:
                    width, height = image.size
                    mode = image.mode

                seen_hashes.add(file_hash)
                rows.append({
                    "path": str(path.resolve()),
                    "label": label,
                    "class_name": folder_name.title(),
                    "width": width,
                    "height": height,
                    "mode": mode,
                    "sha256": file_hash,
                })
            except Exception as exc:
                print(f"[WARN] Skipping invalid image {path}: {exc}")

    if not rows:
        raise RuntimeError("No valid images found.")

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = collect(args.input)

    if df["label"].nunique() != 2:
        raise ValueError("Both Normal and Defective classes are required.")

    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        stratify=df["label"],
        random_state=args.seed,
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        stratify=temp_df["label"],
        random_state=args.seed,
    )

    args.output.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(args.output / "train.csv", index=False)
    val_df.to_csv(args.output / "val.csv", index=False)
    test_df.to_csv(args.output / "test.csv", index=False)

    summary = (
        df.groupby("class_name")
        .agg(
            total=("path", "count"),
            width_mean=("width", "mean"),
            height_mean=("height", "mean"),
        )
        .round(2)
    )
    print("\nDataset summary:\n", summary)
    print(f"\nTotal valid unique images: {len(df)}")
    print(f"Train: {len(train_df)} | Validation: {len(val_df)} | Test: {len(test_df)}")
    print(f"Manifests written to: {args.output}")


if __name__ == "__main__":
    main()
