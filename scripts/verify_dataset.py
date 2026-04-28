"""
Sanity-check the merged dataset before uploading to HuggingFace.

Checks:
  1. Every image has a paired label file
  2. All class IDs are 0 (fire) or 1 (smoke) only
  3. All bbox coordinates are in [0, 1]
  4. Prints class distribution per split
  5. Renders a 5×5 sample grid with bounding boxes (saved to data/merged/sample_grid.jpg)

Run from repo root:
    python scripts/verify_dataset.py --merged-root data/merged
"""

import argparse
import random
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
CLASS_COLORS = {0: (0, 0, 255), 1: (255, 140, 0)}  # fire=red, smoke=orange (BGR)
CLASS_NAMES = {0: "fire", 1: "smoke"}


def load_yolo_labels(lbl_path: Path) -> list[tuple[int, float, float, float, float]]:
    if not lbl_path.exists():
        return []
    rows = []
    for line in lbl_path.read_text().strip().splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        rows.append((int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])))
    return rows


def draw_bboxes(img_bgr: np.ndarray, labels: list) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    out = img_bgr.copy()
    for cls_id, xc, yc, bw, bh in labels:
        x1 = int((xc - bw / 2) * w)
        y1 = int((yc - bh / 2) * h)
        x2 = int((xc + bw / 2) * w)
        y2 = int((yc + bh / 2) * h)
        color = CLASS_COLORS.get(cls_id, (200, 200, 200))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(out, CLASS_NAMES.get(cls_id, str(cls_id)),
                    (x1, max(y1 - 5, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return out


def verify_split(
    split_name: str,
    img_dir: Path,
    lbl_dir: Path,
) -> dict:
    stats = {
        "total": 0,
        "missing_label": 0,
        "bad_class_id": 0,
        "bad_bbox": 0,
        "has_fire": 0,
        "has_smoke": 0,
        "has_both": 0,
        "empty": 0,
    }

    image_paths = [p for p in img_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS]
    stats["total"] = len(image_paths)

    for img_path in image_paths:
        lbl_path = lbl_dir / f"{img_path.stem}.txt"

        if not lbl_path.exists():
            stats["missing_label"] += 1
            continue

        labels = load_yolo_labels(lbl_path)

        if not labels:
            stats["empty"] += 1
            continue

        classes_in_img = set()
        for cls_id, xc, yc, bw, bh in labels:
            if cls_id not in (0, 1):
                stats["bad_class_id"] += 1
            classes_in_img.add(cls_id)
            for v in (xc, yc, bw, bh):
                if not (0.0 <= v <= 1.0):
                    stats["bad_bbox"] += 1
                    break

        if 0 in classes_in_img:
            stats["has_fire"] += 1
        if 1 in classes_in_img:
            stats["has_smoke"] += 1
        if 0 in classes_in_img and 1 in classes_in_img:
            stats["has_both"] += 1

    return stats


def make_sample_grid(
    img_dir: Path,
    lbl_dir: Path,
    out_path: Path,
    grid_size: int = 5,
    cell_size: int = 200,
    seed: int = 42,
) -> None:
    image_paths = [p for p in img_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS]
    random.seed(seed)
    sample = random.sample(image_paths, min(grid_size * grid_size, len(image_paths)))

    cells = []
    for img_path in sample:
        img = cv2.imread(str(img_path))
        if img is None:
            img = np.zeros((cell_size, cell_size, 3), dtype=np.uint8)
        labels = load_yolo_labels(lbl_dir / f"{img_path.stem}.txt")
        img = draw_bboxes(img, labels)
        img = cv2.resize(img, (cell_size, cell_size))
        cells.append(img)

    # Pad to full grid
    while len(cells) < grid_size * grid_size:
        cells.append(np.zeros((cell_size, cell_size, 3), dtype=np.uint8))

    rows = []
    for r in range(grid_size):
        row = np.hstack(cells[r * grid_size : (r + 1) * grid_size])
        rows.append(row)
    grid = np.vstack(rows)
    cv2.imwrite(str(out_path), grid)
    print(f"  Sample grid saved to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify merged dataset integrity")
    parser.add_argument("--merged-root", default="data/merged")
    parser.add_argument("--grid-size", type=int, default=5)
    args = parser.parse_args()

    merged_root = Path(args.merged_root)
    errors_found = False

    for split in ("train", "val", "test"):
        img_dir = merged_root / "images" / split
        lbl_dir = merged_root / "labels" / split

        if not img_dir.exists():
            print(f"[{split}] Directory not found, skipping.")
            continue

        print(f"\n[{split}]")
        stats = verify_split(split, img_dir, lbl_dir)

        print(f"  Total images    : {stats['total']}")
        print(f"  Missing labels  : {stats['missing_label']}")
        print(f"  Bad class IDs   : {stats['bad_class_id']}")
        print(f"  Bad bbox coords : {stats['bad_bbox']}")
        print(f"  Empty (neg)     : {stats['empty']}")
        print(f"  Has fire        : {stats['has_fire']}")
        print(f"  Has smoke       : {stats['has_smoke']}")
        print(f"  Has both        : {stats['has_both']}")

        if stats["bad_class_id"] > 0 or stats["bad_bbox"] > 0:
            print(f"  WARNING: Errors found in {split} split!")
            errors_found = True

    # Visual sample grid from training set
    train_img_dir = merged_root / "images" / "train"
    train_lbl_dir = merged_root / "labels" / "train"
    if train_img_dir.exists():
        print("\nGenerating sample grid...")
        make_sample_grid(
            train_img_dir, train_lbl_dir,
            out_path=merged_root / "sample_grid.jpg",
            grid_size=args.grid_size,
        )

    if errors_found:
        print("\nVerification FAILED — fix errors before uploading.")
        raise SystemExit(1)
    else:
        print("\nVerification PASSED — dataset looks clean.")


if __name__ == "__main__":
    main()
