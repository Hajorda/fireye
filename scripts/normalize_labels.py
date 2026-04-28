"""
Normalize YOLO label class IDs across all sources to the unified convention:
    fire = 0
    smoke = 1

Source-specific remaps:
    D-Fire:        smoke=0, fire=1  →  swap: {0→1, 1→0}
    Pyro-SDIS:     smoke=0 only    →  {0→1}
    AI for Mankind: smoke=0 only   →  {0→1}
    Catargiu 2024: fire=0, smoke=1 →  identity (no change)

Run from repo root:
    python scripts/normalize_labels.py --raw-root data/raw --processed-root data/processed
"""

import argparse
import shutil
from pathlib import Path

from tqdm import tqdm

# Per-source class ID remapping tables
REMAP_TABLES = {
    "dfire":        {0: 1, 1: 0},   # smoke↔fire swap
    "pyro_sdis":    {0: 1},          # smoke→1
    "aiformankind": {0: 1},          # smoke→1
    "catargiu":     {},              # already fire=0, smoke=1
}

VALID_CLASS_IDS = {0, 1}


def remap_label_file(src: Path, dst: Path, remap: dict) -> int:
    """
    Read a YOLO .txt label file, remap class IDs, write to dst.
    Returns number of annotations written.
    """
    lines = src.read_text().strip().splitlines() if src.exists() else []
    out_lines = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls_id = int(parts[0])
        new_cls = remap.get(cls_id, cls_id)
        if new_cls not in VALID_CLASS_IDS:
            continue  # drop unknown classes
        out_lines.append(f"{new_cls} {' '.join(parts[1:])}")

    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("\n".join(out_lines) + ("\n" if out_lines else ""))
    return len(out_lines)


def normalize_source(
    source_name: str,
    raw_root: Path,
    processed_root: Path,
) -> None:
    remap = REMAP_TABLES[source_name]
    src_base = raw_root / source_name
    dst_base = processed_root / source_name

    # Locate images and labels directories
    # Different sources may have flat or nested layouts — search broadly
    image_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    image_paths = [
        p for p in src_base.rglob("*")
        if p.suffix.lower() in image_exts
    ]

    if not image_paths:
        print(f"  [{source_name}] No images found in {src_base}, skipping.")
        return

    (dst_base / "images").mkdir(parents=True, exist_ok=True)
    (dst_base / "labels").mkdir(parents=True, exist_ok=True)

    copied = 0
    total_annotations = 0

    for img_path in tqdm(image_paths, desc=f"  {source_name}"):
        stem = img_path.stem
        # Pair label: same directory, same stem, .txt extension
        lbl_src = img_path.with_suffix(".txt")
        if not lbl_src.exists():
            # Try labels/ sibling folder
            lbl_src = img_path.parent.parent / "labels" / f"{stem}.txt"

        dst_img = dst_base / "images" / img_path.name
        dst_lbl = dst_base / "labels" / f"{stem}.txt"

        shutil.copy2(img_path, dst_img)
        n = remap_label_file(lbl_src, dst_lbl, remap)
        total_annotations += n
        copied += 1

    print(
        f"  [{source_name}] {copied} images, {total_annotations} annotations "
        f"(remap: {remap or 'identity'})"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize label class IDs")
    parser.add_argument("--raw-root", default="data/raw")
    parser.add_argument("--processed-root", default="data/processed")
    parser.add_argument(
        "--sources",
        nargs="+",
        default=list(REMAP_TABLES.keys()),
        choices=list(REMAP_TABLES.keys()),
    )
    args = parser.parse_args()

    raw_root = Path(args.raw_root)
    processed_root = Path(args.processed_root)

    for source in args.sources:
        normalize_source(source, raw_root, processed_root)

    print("\nLabel normalization complete.")


if __name__ == "__main__":
    main()
