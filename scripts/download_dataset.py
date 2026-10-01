"""
Download a public manufacturing defect dataset for this assessment.

Source:
https://github.com/PerceptiLabs/AC-Piston-Checks

The source repository contains:
    data/normal/
    data/defect_1/
    data/defect_2/

For this assessment, defect_1 and defect_2 are combined into one
"defective" class, while normal remains "normal".

This is a public substitute because the client dataset was not supplied.
Check the source repository/Kaggle attribution and terms before redistribution.
"""

from __future__ import annotations

import argparse
import io
import shutil
import urllib.request
import zipfile
from pathlib import Path

REPO_ZIP = "https://github.com/PerceptiLabs/AC-Piston-Checks/archive/refs/heads/main.zip"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def download(url: str) -> bytes:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "defect-detection-assessment/1.0"},
    )
    with urllib.request.urlopen(req, timeout=90) as response:
        return response.read()


def copy_images(source_dir: Path, destination_dir: Path, prefix: str = "") -> int:
    destination_dir.mkdir(parents=True, exist_ok=True)
    count = 0

    if not source_dir.exists():
        return 0

    for path in source_dir.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        target = destination_dir / f"{prefix}{path.name}"
        if target.exists():
            stem, suffix = target.stem, target.suffix
            i = 1
            while target.exists():
                target = destination_dir / f"{stem}_{i}{suffix}"
                i += 1

        shutil.copy2(path, target)
        count += 1

    return count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/raw")
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    print("Downloading public AC Piston manufacturing dataset...")
    archive = download(REPO_ZIP)

    temp = output / "_download"
    temp.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(io.BytesIO(archive)) as zf:
        zf.extractall(temp)

    roots = [p for p in temp.iterdir() if p.is_dir()]
    if not roots:
        raise RuntimeError("Could not find extracted repository.")

    repo = roots[0]
    data = repo / "data"

    normal = copy_images(data / "normal", output / "normal")
    defect_1 = copy_images(data / "defect_1", output / "defective", "defect1_")
    defect_2 = copy_images(data / "defect_2", output / "defective", "defect2_")

    shutil.rmtree(temp, ignore_errors=True)

    print()
    print(f"Normal images:       {normal}")
    print(f"Defect type 1:       {defect_1}")
    print(f"Defect type 2:       {defect_2}")
    print(f"Total defective:     {defect_1 + defect_2}")
    print()
    print("Dataset ready under data/raw/")


if __name__ == "__main__":
    main()
