from albumentations.core.composition import BboxParams, Compose

from albumentations.augmentations.transforms import *
from albumentations import *

# (
#     LongestMaxSize,
#     PadIfNeeded,
#     RandomSizedBBoxSafeCrop,
#     ShiftScaleRotate,
#     RandomRotate90,
#     HorizontalFlip,
#     RandomBrightnessContrast,
#     RandomGamma,
#     HueSaturationValue,
#     MotionBlur,
#     JpegCompression,
#     Normalize,
# )


def get_transforms(config, mode: str = "train") -> Compose:
    """
    Composes albumentations transforms.
    Returns the full list of transforms when mode is "train".
    mode should be one of "train", "val".
    """
    # compose validation transforms
    if mode == "val":
        transforms = Compose(
            [],
            bbox_params=BboxParams(
                format="pascal_voc",
                min_area=0.0,
                min_visibility=0.0,
                label_fields=["category_id"],
            ),
        )
    # compose train transforms
    # TODO: make transformation parameters configurable from yml
    elif mode == "train":
        transforms = Compose(
            [
                # LongestMaxSize(),
                # PadIfNeeded(min_height=768, min_width=768, border_mode=0, p=1),
                # RandomSizedBBoxSafeCrop(),

                ShiftScaleRotate(p=0.8),
                HorizontalFlip(),
                VerticalFlip(),

                Sharpen(),
                RandomRotate90(),
                RandomBrightnessContrast(),
                MotionBlur(p=0.1),

                Downscale(),
                Superpixels(),
                RandomContrast()

                # RandomGamma(),
                # HueSaturationValue(),
                # JpegCompression(),
            ],
            bbox_params=BboxParams(
                format="pascal_voc",
                min_area=0.0,
                min_visibility=0.0,
                label_fields=["category_id"],
            ),
        )
    return transforms
