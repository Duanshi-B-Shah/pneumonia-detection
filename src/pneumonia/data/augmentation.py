"""Image augmentation pipelines for training and inference."""
from __future__ import annotations

from torchvision import transforms

from pneumonia.utils.config import AugmentationConfig

# ImageNet normalization constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transforms(image_size: int, aug_cfg: AugmentationConfig) -> transforms.Compose:
    """Build training augmentation pipeline.

    Applies random geometric and photometric transforms followed by
    resizing, tensor conversion, and ImageNet normalization.
    """
    transform_list = [
        transforms.Resize((image_size, image_size)),
    ]

    if aug_cfg.horizontal_flip:
        transform_list.append(transforms.RandomHorizontalFlip(p=0.5))

    if aug_cfg.rotation_degrees > 0:
        transform_list.append(
            transforms.RandomRotation(degrees=aug_cfg.rotation_degrees)
        )

    if aug_cfg.brightness > 0 or aug_cfg.contrast > 0:
        transform_list.append(
            transforms.ColorJitter(
                brightness=aug_cfg.brightness,
                contrast=aug_cfg.contrast,
            )
        )

    if aug_cfg.affine_translate or aug_cfg.affine_scale:
        transform_list.append(
            transforms.RandomAffine(
                degrees=0,
                translate=tuple(aug_cfg.affine_translate) if aug_cfg.affine_translate else None,
                scale=tuple(aug_cfg.affine_scale) if aug_cfg.affine_scale else None,
            )
        )

    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    return transforms.Compose(transform_list)


def get_val_transforms(image_size: int) -> transforms.Compose:
    """Build validation/test transform pipeline (no augmentation)."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_inference_transforms(image_size: int) -> transforms.Compose:
    """Build inference transform pipeline (identical to validation)."""
    return get_val_transforms(image_size)


def denormalize(tensor):
    """Reverse ImageNet normalization for visualization."""
    import torch

    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return tensor * std + mean
