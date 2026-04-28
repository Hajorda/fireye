"""
Upload the merged YOLO dataset to HuggingFace Hub as a structured dataset.

Creates repo: hajorda/fireye-wildfire-detection (or value of HF_DATASET_REPO env var)

Each example contains:
  - image: PIL Image
  - image_id: string
  - split: "train" | "validation" | "test"
  - source: which original dataset the image came from
  - annotations: list of {class_id, class_name, x_center, y_center, width, height}
  - has_fire: bool
  - has_smoke: bool

Run from repo root (requires HF_TOKEN env var):
    python scripts/push_to_hub.py --merged-root data/merged
"""

import argparse
import os
from pathlib import Path

from datasets import Dataset, DatasetDict, Features, Image, Sequence, Value
from huggingface_hub import HfApi
from PIL import Image as PILImage
from tqdm import tqdm

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
CLASS_NAMES = {0: "fire", 1: "smoke"}


FEATURES = Features(
    {
        "image": Image(),
        "image_id": Value("string"),
        "split": Value("string"),
        "source": Value("string"),
        "annotations": [
            {
                "class_id": Value("int32"),
                "class_name": Value("string"),
                "x_center": Value("float32"),
                "y_center": Value("float32"),
                "width": Value("float32"),
                "height": Value("float32"),
            }
        ],
        "has_fire": Value("bool"),
        "has_smoke": Value("bool"),
    }
)


def load_yolo_labels(lbl_path: Path) -> list[dict]:
    annotations = []
    if not lbl_path.exists():
        return annotations
    for line in lbl_path.read_text().strip().splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls_id = int(parts[0])
        annotations.append(
            {
                "class_id": cls_id,
                "class_name": CLASS_NAMES.get(cls_id, str(cls_id)),
                "x_center": float(parts[1]),
                "y_center": float(parts[2]),
                "width": float(parts[3]),
                "height": float(parts[4]),
            }
        )
    return annotations


def build_examples(img_dir: Path, lbl_dir: Path, split_name: str) -> list[dict]:
    examples = []
    image_paths = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)

    for img_path in tqdm(image_paths, desc=f"  Building {split_name}"):
        lbl_path = lbl_dir / f"{img_path.stem}.txt"
        annotations = load_yolo_labels(lbl_path)

        # Extract source from filename prefix (e.g., "dfire_img_0001.jpg" → "dfire")
        parts = img_path.stem.split("_", 1)
        source = parts[0] if len(parts) > 1 else "unknown"

        class_ids = {a["class_id"] for a in annotations}

        try:
            pil_img = PILImage.open(img_path).convert("RGB")
        except Exception:
            continue

        examples.append(
            {
                "image": pil_img,
                "image_id": img_path.stem,
                "split": split_name,
                "source": source,
                "annotations": annotations,
                "has_fire": 0 in class_ids,
                "has_smoke": 1 in class_ids,
            }
        )

    return examples


def main() -> None:
    parser = argparse.ArgumentParser(description="Push dataset to HuggingFace Hub")
    parser.add_argument("--merged-root", default="data/merged")
    parser.add_argument(
        "--repo-id",
        default=os.environ.get("HF_DATASET_REPO", "hajorda/fireye-wildfire-detection"),
    )
    parser.add_argument("--shard-size", default="500MB")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise EnvironmentError("HF_TOKEN environment variable is not set.")

    merged_root = Path(args.merged_root)

    # Build HF DatasetDict
    hf_split_map = {"train": "train", "val": "validation", "test": "test"}
    dataset_splits = {}

    for local_split, hf_split in hf_split_map.items():
        img_dir = merged_root / "images" / local_split
        lbl_dir = merged_root / "labels" / local_split
        if not img_dir.exists():
            print(f"  Skipping {local_split} (not found)")
            continue

        print(f"\nBuilding {hf_split} split...")
        examples = build_examples(img_dir, lbl_dir, hf_split)
        dataset_splits[hf_split] = Dataset.from_list(examples, features=FEATURES)
        print(f"  {len(examples)} examples")

    if not dataset_splits:
        raise RuntimeError("No splits found. Run merge_datasets.py first.")

    dataset = DatasetDict(dataset_splits)

    print(f"\nPushing to HuggingFace Hub: {args.repo_id}")
    dataset.push_to_hub(
        args.repo_id,
        token=token,
        max_shard_size=args.shard_size,
    )
    print(f"\nDataset uploaded successfully: https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
