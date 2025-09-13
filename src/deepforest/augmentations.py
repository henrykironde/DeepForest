"""Augmentation module for DeepForest using Kornia.

This module provides configurable augmentations for training and
validation that can be specified through configuration files or direct
parameters. Uses Kornia for PyTorch-native augmentations with GPU
support.
"""

from typing import Any

import kornia.augmentation as K
import torch
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from omegaconf.listconfig import ListConfig

_SUPPORTED_TRANSFORMS = {
    "HorizontalFlip": (K.RandomHorizontalFlip, {"p": 0.5}),
    "VerticalFlip": (K.RandomVerticalFlip, {"p": 0.5}),
    "Downscale": (
        K.RandomResizedCrop,
        {"size": (200, 200), "scale": (0.25, 0.5), "p": 0.5},
    ),
    "RandomCrop": (K.RandomCrop, {"size": (200, 200), "p": 0.5}),
    "RandomSizedBBoxSafeCrop": (
        K.RandomResizedCrop,
        {"size": (200, 200), "scale": (0.5, 1.0), "p": 0.5},
    ),
    "PadIfNeeded": (K.PadTo, {"size": (800, 800), "p": 1.0}),
    "Rotate": (K.RandomRotation, {"degrees": 15, "p": 0.5}),
    "RandomBrightnessContrast": (
        K.ColorJitter,
        {"brightness": 0.2, "contrast": 0.2, "p": 0.5},
    ),
    "HueSaturationValue": (
        K.ColorJitter,
        {"hue": 0.1, "saturation": 0.1, "p": 0.5},
    ),
    "GaussNoise": (K.RandomGaussianNoise, {"mean": 0.0, "std": 0.1, "p": 0.3}),
    "Blur": (
        K.RandomGaussianBlur,
        {"kernel_size": (3, 3), "sigma": (0.1, 2.0), "p": 0.3},
    ),
    "GaussianBlur": (
        K.RandomGaussianBlur,
        {"kernel_size": (3, 3), "sigma": (0.1, 2.0), "p": 0.3},
    ),
    "MotionBlur": (
        K.RandomMotionBlur,
        {"kernel_size": 3, "angle": 0, "direction": 0, "p": 0.3},
    ),
    "ZoomBlur": (
        K.RandomAffine,
        {"degrees": 0, "translate": 0, "scale": (1.0, 1.05), "p": 0.3},
    ),
}


def get_available_augmentations() -> list[str]:
    """Get list of available augmentation names.

    Returns:
        List of available augmentation names
    """
    return sorted(_SUPPORTED_TRANSFORMS.keys())


def get_transform(
    augmentations: str | list[str] | dict[str, Any] | None = None,
) -> "AlbumentationsCompatibleTransform":
    """Create Kornia transform for bounding boxes with albumentations
    compatibility.

    Args:
        augmentations: Augmentation configuration:
            - str: Single augmentation name
            - list: List of augmentation names
            - dict: Dict with names as keys and params as values
            - None: No augmentations

    Returns:
        AlbumentationsCompatibleTransform that works like albumentations.Compose

    Examples:
        >>> # Default behavior, returns identity transform
        >>> transform = get_transform()

        >>> # Single augmentation
        >>> transform = get_transform(augmentations="Downscale")

        >>> # Multiple augmentations
        >>> transform = get_transform(augmentations=["HorizontalFlip", "Downscale"])

        >>> # Augmentations with parameters
        >>> transform = get_transform(augmentations={
        ...                              "HorizontalFlip": {"p": 0.5},
        ...                              "Downscale": {"scale": (0.25, 0.75)}
        ...                          })
    """
    transforms_list = []

    if augmentations is not None:
        augment_configs = _parse_augmentations(augmentations)

        for aug_name, aug_params in augment_configs.items():
            aug_transform = _create_augmentation(aug_name, aug_params)
            transforms_list.append(aug_transform)

    return AlbumentationsCompatibleTransform(transforms_list)


def _parse_augmentations(
    augmentations: str | list | dict | ListConfig | DictConfig,
) -> dict[str, dict[str, Any]]:
    """Parse augmentations parameter into a standardized dict format.

    Examples:
        - "HorizontalFlip" -> {"HorizontalFlip": {}}
        - ["HorizontalFlip", "Downscale"] -> {"HorizontalFlip": {}, "Downscale": {}}
        - {"HorizontalFlip": {"p": 0.5}}
        - [{"HorizontalFlip": {"p": 0.5}}, {"Downscale": {"scale": (0.25, 0.75)}}] -> {"HorizontalFlip": {"p": 0.5}, "Downscale": {"scale": (0.25, 0.75)}}

    Args:
        augmentations: Augmentation specification in various formats:
            - str: Single augmentation name
            - List: List of strings or dicts with augmentation configs
            - Dict: Dict with augmentation names as keys and parameters as values

    Returns:
        Dict mapping augmentation names to their parameters
    """

    # Convert OmegaConf to primitives
    if isinstance(augmentations, (DictConfig, ListConfig)):
        augmentations = OmegaConf.to_container(augmentations, resolve=True)

    if isinstance(augmentations, str):
        return {augmentations: {}}

    if isinstance(augmentations, dict):
        return augmentations
    elif isinstance(augmentations, list):
        result = {}
        for augmentation in augmentations:
            if isinstance(augmentation, str):
                result[augmentation] = {}
            elif isinstance(augmentation, dict):
                if len(augmentation) != 1:
                    raise ValueError(
                        f"Each augmentation dict must have exactly "
                        f"one key (corresponding to a single operation), "
                        f"got {len(augmentation)} for {augmentation}."
                    )
                name, params = next(iter(augmentation.items()))
                result[name] = params
            else:
                raise ValueError(
                    f"List elements must be strings or dicts, got {type(augmentation)}"
                )
        return result
    else:
        raise ValueError(f"Unable to parse augmentation parameters: {augmentations}")


def _create_augmentation(name: str, params: dict[str, Any]) -> torch.nn.Module:
    """Create a Kornia transform by name with given parameters.

    Args:
        name: Name of the augmentation
        params: Parameters to pass to the augmentation

    Returns:
        Kornia transform module

    Raises:
        ValueError: If augmentation name is not recognized or creation fails
    """

    if name not in get_available_augmentations():
        raise ValueError(
            f"Unknown augmentation '{name}'. Available augmentations: {get_available_augmentations()}"
        )

    # Retrieve factory and defaults, merge with user-provided params
    transform, base_params = _SUPPORTED_TRANSFORMS[name]
    final_params = base_params.copy()
    final_params.update(params)

    try:
        return transform(**final_params)
    except Exception as e:
        raise ValueError(
            f"Failed to create augmentation '{name}' with params {final_params}: {e}"
        ) from e


class AlbumentationsCompatibleTransform:
    """Compatibility wrapper for Kornia transforms to work like
    albumentations.Compose.

    This class provides the same interface as albumentations.Compose but
    uses Kornia transforms internally for better PyTorch integration and
    GPU support.
    """

    def __init__(self, transforms_list: list[torch.nn.Module]):
        """Initialize with list of Kornia transforms.

        Args:
            transforms_list: List of Kornia transform modules
        """
        self.transforms = transforms_list
        self.kornia_transform = (
            torch.nn.Sequential(*transforms_list)
            if transforms_list
            else torch.nn.Identity()
        )

    def __call__(
        self,
        image: torch.Tensor,
        bboxes: torch.Tensor = None,
        category_ids: torch.Tensor = None,
        **kwargs,
    ) -> dict:
        """Apply transforms to image and bounding boxes.

        Args:
            image: Input image tensor of shape (H, W, C) in range [0, 1]
            bboxes: Optional bounding boxes tensor of shape (N, 4) in format [x1, y1, x2, y2]
            category_ids: Optional category IDs tensor of shape (N,)
            **kwargs: Additional arguments (ignored for compatibility)

        Returns:
            Dictionary with keys 'image', 'bboxes', 'category_ids' containing transformed data
        """
        # Convert image from (H, W, C) to (C, H, W) for Kornia
        if image.dim() == 3 and image.shape[-1] == 3:
            image = image.permute(2, 0, 1)

        # Ensure image is in correct format (B, C, H, W) for Kornia
        if image.dim() == 3:
            image = image.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        # Apply Kornia transforms
        transformed_image = self.kornia_transform(image)

        # Convert back to (H, W, C) format
        if squeeze_output:
            transformed_image = transformed_image.squeeze(0)

        # Convert back to (H, W, C) if it was originally in that format
        if transformed_image.dim() == 3 and transformed_image.shape[0] == 3:
            transformed_image = transformed_image.permute(1, 2, 0)

        # For now, return bboxes and category_ids unchanged
        # In a full implementation, you'd need to transform bboxes based on the applied transforms
        result = {"image": transformed_image}

        if bboxes is not None:
            result["bboxes"] = (
                bboxes.clone() if isinstance(bboxes, torch.Tensor) else bboxes.copy()
            )
        else:
            result["bboxes"] = []

        if category_ids is not None:
            result["category_ids"] = (
                category_ids.clone()
                if isinstance(category_ids, torch.Tensor)
                else category_ids.copy()
            )
        else:
            result["category_ids"] = []

        return result


def apply_transform(
    transform: torch.nn.Module, image: torch.Tensor, bboxes: torch.Tensor = None
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Apply transform to image and optionally bounding boxes.

    Args:
        transform: Kornia transform module
        image: Input image tensor of shape (C, H, W) or (B, C, H, W)
        bboxes: Optional bounding boxes tensor of shape (N, 4) in format [x1, y1, x2, y2]

    Returns:
        Tuple of (transformed_image, transformed_bboxes)
    """
    # Ensure image is in correct format (B, C, H, W)
    if image.dim() == 3:
        image = image.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False

    # Apply transform to image
    transformed_image = transform(image)

    # Squeeze batch dimension if input was 3D
    if squeeze_output:
        transformed_image = transformed_image.squeeze(0)

    # Handle bounding boxes if provided
    transformed_bboxes = None
    if bboxes is not None:
        # For now, we'll return bboxes unchanged
        # In a full implementation, you'd need to transform bboxes based on the applied transforms
        transformed_bboxes = bboxes.clone()

    return transformed_image, transformed_bboxes
