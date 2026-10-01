from pathlib import Path
from typing import Sequence

import pandas as pd
import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from torchvision import transforms

ImageFile.LOAD_TRUNCATED_IMAGES = False

CLASS_TO_INDEX = {"Normal": 0, "Defective": 1}
INDEX_TO_CLASS = {0: "Normal", 1: "Defective"}


def build_train_transform(image_size: int = 224):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.15),
        transforms.RandomRotation(degrees=12),
        transforms.ColorJitter(brightness=0.18, contrast=0.18, saturation=0.12),
        transforms.RandomApply(
            [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.2))],
            p=0.10,
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def build_eval_transform(image_size: int = 224):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


class DefectDataset(Dataset):
    def __init__(self, manifest: str | Path | pd.DataFrame, transform=None):
        self.df = (
            pd.read_csv(manifest)
            if isinstance(manifest, (str, Path))
            else manifest.copy()
        )
        if not {"path", "label"}.issubset(self.df.columns):
            raise ValueError("Manifest must contain 'path' and 'label' columns.")
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index: int):
        row = self.df.iloc[index]
        path = Path(row["path"])
        label = int(row["label"])

        try:
            with Image.open(path) as image:
                image = image.convert("RGB")
        except Exception as exc:
            raise RuntimeError(f"Unable to read image: {path}") from exc

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long), str(path)
