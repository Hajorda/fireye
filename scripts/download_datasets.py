"""
Download all four source datasets into data/raw/.

Run from the repo root:
    python scripts/download_datasets.py --root data/raw

On Google Colab, this script is called from 01_dataset_prep.ipynb after
mounting Drive and setting up credentials.

Dataset sources:
  1. D-Fire       — Kaggle: sayedgamal99/smoke-fire-detection-yolo
  2. Pyro-SDIS    — HuggingFace: pyronear/pyro-sdis
  3. AI for Mankind — GitHub: aiformankind/wildfire-smoke-dataset
  4. Catargiu 2024  — GitHub: CostiCatargiu/NEWFireSmokeDataset_YoloModels
"""

import argparse
import os
import shutil
import subprocess
import zipfile
from pathlib import Path

from tqdm import tqdm


def download_dfire(raw_root: Path) -> None:
    """Download D-Fire via Kaggle API."""
    dest = raw_root / "dfire"
    dest.mkdir(parents=True, exist_ok=True)
    print("\n[1/4] Downloading D-Fire from Kaggle...")
    subprocess.run(
        [
            "kaggle", "datasets", "download",
            "sayedgamal99/smoke-fire-detection-yolo",
            "--path", str(dest),
            "--unzip",
        ],
        check=True,
    )
    print(f"  D-Fire saved to {dest}")


def download_pyro_sdis(raw_root: Path) -> None:
    """
    Download Pyro-SDIS from HuggingFace and serialize to YOLO format on disk.

    The HF dataset stores each example as:
      - image: PIL Image
      - annotation: YOLO-format string (one line per bbox, class x y w h)
    """
    from datasets import load_dataset
    from PIL import Image as PILImage

    dest = raw_root / "pyro_sdis"
    images_dir = dest / "images"
    labels_dir = dest / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    print("\n[2/4] Downloading Pyro-SDIS from HuggingFace...")
    ds = load_dataset("pyronear/pyro-sdis", split="train", trust_remote_code=True)

    for i, example in enumerate(tqdm(ds, desc="  Serializing Pyro-SDIS")):
        stem = f"pyro_{i:06d}"
        img_path = images_dir / f"{stem}.jpg"
        lbl_path = labels_dir / f"{stem}.txt"

        img = example["image"]
        if not isinstance(img, PILImage.Image):
            img = PILImage.fromarray(img)
        img.save(str(img_path), "JPEG", quality=95)

        annotation = example.get("annotation", "") or ""
        lbl_path.write_text(annotation.strip())

    print(f"  Pyro-SDIS saved to {dest} ({i+1} images)")


def download_aiformankind(raw_root: Path, max_images: int = 50_000) -> None:
    """
    Clone AI for Mankind wildfire smoke dataset.
    Samples up to max_images to stay within Drive space limits.
    """
    import random

    dest = raw_root / "aiformankind"
    dest.mkdir(parents=True, exist_ok=True)

    repo_url = "https://github.com/aiformankind/wildfire-smoke-dataset.git"
    clone_dir = dest / "repo"

    print(f"\n[3/4] Cloning AI for Mankind dataset (shallow)...")
    if clone_dir.exists():
        print("  Already cloned, skipping.")
    else:
        subprocess.run(
            ["git", "clone", "--depth=1", repo_url, str(clone_dir)],
            check=True,
        )

    # Collect all image paths from the repo
    image_exts = {".jpg", ".jpeg", ".png"}
    all_images = [
        p for p in clone_dir.rglob("*")
        if p.suffix.lower() in image_exts
    ]
    print(f"  Found {len(all_images)} images in repo.")

    if len(all_images) > max_images:
        random.seed(42)
        all_images = random.sample(all_images, max_images)
        print(f"  Sampled {max_images} images (Drive space limit).")

    images_dir = dest / "images"
    labels_dir = dest / "labels"
    images_dir.mkdir(exist_ok=True)
    labels_dir.mkdir(exist_ok=True)

    for img_path in tqdm(all_images, desc="  Copying AI for Mankind images"):
        shutil.copy2(img_path, images_dir / img_path.name)
        # Look for paired label file (same stem, .txt)
        lbl_src = img_path.with_suffix(".txt")
        lbl_dst = labels_dir / img_path.with_suffix(".txt").name
        if lbl_src.exists():
            shutil.copy2(lbl_src, lbl_dst)
        else:
            # Create empty label (hard negative)
            lbl_dst.write_text("")

    print(f"  AI for Mankind saved to {dest}")


def download_catargiu(raw_root: Path) -> None:
    """Clone Catargiu 2024 FireAndSmoke dataset."""
    dest = raw_root / "catargiu"
    dest.mkdir(parents=True, exist_ok=True)

    repo_url = "https://github.com/CostiCatargiu/NEWFireSmokeDataset_YoloModels.git"
    clone_dir = dest / "repo"

    print(f"\n[4/4] Cloning Catargiu FireAndSmoke dataset...")
    if clone_dir.exists():
        print("  Already cloned, skipping.")
    else:
        subprocess.run(
            ["git", "clone", "--depth=1", repo_url, str(clone_dir)],
            check=True,
        )
    print(f"  Catargiu dataset saved to {dest}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download wildfire datasets")
    parser.add_argument("--root", default="data/raw", help="Raw data root directory")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["dfire", "pyro_sdis", "aiformankind", "catargiu"],
        choices=["dfire", "pyro_sdis", "aiformankind", "catargiu"],
        help="Which datasets to download",
    )
    parser.add_argument(
        "--aiformankind-max",
        type=int,
        default=50_000,
        help="Max images to sample from AI for Mankind (default 50k for Drive)",
    )
    args = parser.parse_args()

    raw_root = Path(args.root)
    raw_root.mkdir(parents=True, exist_ok=True)

    if "dfire" in args.datasets:
        download_dfire(raw_root)
    if "pyro_sdis" in args.datasets:
        download_pyro_sdis(raw_root)
    if "aiformankind" in args.datasets:
        download_aiformankind(raw_root, max_images=args.aiformankind_max)
    if "catargiu" in args.datasets:
        download_catargiu(raw_root)

    print("\nAll downloads complete.")


if __name__ == "__main__":
    main()
