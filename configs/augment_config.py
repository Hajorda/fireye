"""
Albumentations pipeline for offline augmentation of the training split.
Imported by scripts/augment_dataset.py.
"""

import albumentations as A


def get_train_transforms() -> A.Compose:
    return A.Compose(
        [
            # Lighting / sensor variation
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7),
            A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=30, val_shift_limit=20, p=0.5),
            A.RandomGamma(gamma_limit=(70, 130), p=0.4),
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.3),
            A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.3),
            A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.2),
            A.ToGray(p=0.05),

            # Atmospheric degradation — simulate early smoke / haze / fog
            A.RandomFog(fog_coef_range=(0.1, 0.4), alpha_coef=0.1, p=0.3),
            A.GaussianBlur(blur_limit=(3, 7), p=0.2),
            A.MotionBlur(blur_limit=7, p=0.15),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),

            # Geometric — fixed cameras but fire/smoke scenes have perspective variation
            A.HorizontalFlip(p=0.5),
            # flipud=0: sky must stay up — never flip vertically
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.2,
                rotate_limit=15,
                border_mode=0,
                p=0.5,
            ),
            A.Perspective(scale=(0.05, 0.1), p=0.2),

            # Occlusion robustness — smoke often partially blocks the fire
            A.CoarseDropout(
                num_holes_range=(1, 8),
                hole_height_range=(16, 32),
                hole_width_range=(16, 32),
                p=0.2,
            ),
        ],
        bbox_params=A.BboxParams(
            format="yolo",
            label_fields=["class_labels"],
            min_visibility=0.3,  # drop bbox if < 30% visible after transform
        ),
    )
