"""
Image augmentation pipelines for patch-level WSI data.

Two pipelines:
  - train_transforms: heavy augmentation suitable for histopathology
  - val_transforms:   deterministic resize + normalize only
"""

import torchvision.transforms as T
import torchvision.transforms.functional as TF


# ImageNet statistics work well for ImageNet-pretrained backbones even on H&E patches.
# Some works use H&E-specific means/stds, but the difference is minor.
_MEAN = (0.485, 0.456, 0.406)
_STD  = (0.229, 0.224, 0.225)


def train_transforms(image_size: int = 224) -> T.Compose:
    """Augmentation pipeline used during training.

    Applies stain-agnostic augmentations relevant for H&E pathology:
      - random resized crop  (scale invariance)
      - horizontal / vertical flip  (no preferred orientation)
      - color jitter  (stain variation simulation)
      - random grayscale  (stain-independence)
      - Gaussian blur  (focus variation)
      - normalization
    """
    return T.Compose([
        T.RandomResizedCrop(image_size, scale=(0.7, 1.0), ratio=(0.9, 1.1)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.RandomApply([
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05)
        ], p=0.8),
        T.RandomGrayscale(p=0.1),
        T.RandomApply([T.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.3),
        T.ToTensor(),
        T.Normalize(mean=_MEAN, std=_STD),
    ])


def val_transforms(image_size: int = 224) -> T.Compose:
    """Deterministic pipeline used for validation and testing."""
    return T.Compose([
        T.Resize(int(image_size * 1.1)),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize(mean=_MEAN, std=_STD),
    ])
