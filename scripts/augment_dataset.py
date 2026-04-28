"""
Offline augmentation of the training split.

For each original training image, generates N augmented variants (default: 2).
Underrepresented classes (fire-only images) get extra variants (default: 4).
Augmented files are written into the same data/merged/images/train and labels/train
directories with an _aug{N} suffix so the originals are preserved.

The augmentation pipeline is defined in configs/augment_config.py.

Run from repo root:
    python scripts/augment_dataset.py --merged-root data/merged --multiplier 2
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

# Allow importing configs/ from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs.augment_config import get_train_transforms


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
FIRE_CLASS = 0
SMOKE_CLASS = 1


def load_yolo_labels(lbl_path: Path) -> tuple[list[int], list[list[float]]]:
    """Return (class_ids, bboxes) from a YOLO .txt label file."""
    class_ids, bboxes = [], []
    if not lbl_path.exists():
        return class_ids, bboxes
    for line in lbl_path.read_text().strip().splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls = int(parts[0])
        bbox = [float(x) for x in parts[1:5]]
        class_ids.append(cls)
        bboxes.append(bbox)
    return class_ids, bboxes


def save_yolo_labels(lbl_path: Path, class_ids: list[int], bboxes: list[list[float]]) -> None:
    lines = [f"{cls} {' '.join(f'{v:.6f}' for v in bbox)}" for cls, bbox in zip(class_ids, bboxes)]
    lbl_path.write_text("\n".join(lines) + ("\n" if lines else ""))


def has_fire_only(class_ids: list[int]) -> bool:
    return FIRE_CLASS in class_ids and SMOKE_CLASS not in class_ids


def augment_image(
    img_path: Path,
    lbl_path: Path,
    output_img_dir: Path,
    output_lbl_dir: Path,
    n_variants: int,
    transform,
) -> int:
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        return 0
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    class_ids, bboxes = load_yolo_labels(lbl_path)

    generated = 0
    for i in range(n_variants):
        try:
            result = transform(
                image=img_rgb,
                bboxes=bboxes,
                class_labels=class_ids,
            )
        except Exception:
            continue

        aug_img = result["image"]
        aug_bboxes = result["bboxes"]
        aug_cls = result["class_labels"]

        # Clamp bbox values to [0, 1] to handle float precision edge cases
        aug_bboxes = [
            [min(max(v, 0.0), 1.0) for v in bb]
            for bb in aug_bboxes
        ]

        stem = img_path.stem
        out_stem = f"{stem}_aug{i}"
        out_img = output_img_dir / f"{out_stem}{img_path.suffix}"
        out_lbl = output_lbl_dir / f"{out_stem}.txt"

        cv2.imwrite(str(out_img), cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR))
        save_yolo_labels(out_lbl, list(aug_cls), list(aug_bboxes))
        generated += 1

    return generated


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline training data augmentation")
    parser.add_argument("--merged-root", default="data/merged")
    parser.add_argument(
        "--multiplier",
        type=int,
        default=2,
        help="Augmented variants per image (default 2)",
    )
    parser.add_argument(
        "--fire-multiplier",
        type=int,
        default=4,
        help="Extra variants for fire-only images (default 4)",
    )
    args = parser.parse_args()

    merged_root = Path(args.merged_root)
    img_dir = merged_root / "images" / "train"
    lbl_dir = merged_root / "labels" / "train"

    if not img_dir.exists():
        print(f"Training images not found at {img_dir}. Run merge_datasets.py first.")
        return

    transform = get_train_transforms()

    image_paths = [p for p in img_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS]
    total_generated = 0

    for img_path in tqdm(image_paths, desc="Augmenting train split"):
        lbl_path = lbl_dir / f"{img_path.stem}.txt"
        class_ids, _ = load_yolo_labels(lbl_path)

        n = args.fire_multiplier if has_fire_only(class_ids) else args.multiplier

        total_generated += augment_image(
            img_path, lbl_path,
            img_dir, lbl_dir,
            n_variants=n,
            transform=transform,
        )

    print(f"\nAugmentation complete — {total_generated} new images added to train split")
    train_total = len(list(img_dir.iterdir()))
    print(f"Train split now has {train_total} images (originals + augmented)")


if __name__ == "__main__":
    main()
