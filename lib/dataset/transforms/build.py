from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from . import transforms as T


def build_transforms(cfg, is_train=True):
    assert is_train is True, 'Please only use build_transforms for training.'
    assert isinstance(cfg.DATASET.OUTPUT_SIZE, (list, tuple)), 'DATASET.OUTPUT_SIZE should be list or tuple'
    if is_train:
        max_rotation = cfg.DATASET.MAX_ROTATION
        min_scale = cfg.DATASET.MIN_SCALE
        max_scale = cfg.DATASET.MAX_SCALE
        max_translate = cfg.DATASET.MAX_TRANSLATE
        input_size = cfg.DATASET.INPUT_SIZE
        output_size = cfg.DATASET.OUTPUT_SIZE
        flip = cfg.DATASET.FLIP
        scale_type = cfg.DATASET.SCALE_TYPE
    else:
        scale_type = cfg.DATASET.SCALE_TYPE
        max_rotation = 0
        min_scale = 1
        max_scale = 1
        max_translate = 0
        input_size = 512
        output_size = [128]
        flip = 0

    imagenet_means = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    transforms_list = [
                        T.RandomAffineTransform(
                            input_size,
                            output_size,
                            max_rotation,
                            min_scale,
                            max_scale,
                            scale_type,
                            max_translate,
                            perspective=cfg.DATASET.PERSPECTIVE
                        ),
                        T.RandomHorizontalFlip(output_size, flip),
                        T.ToTensor(),
                        T.RandomGaussianNoise(min_var=0.0, max_var=0.0005),
                        T.ColorJitterPerso(),
                        T.GaussianBlurPerso(),
                        T.Normalize(mean=imagenet_means, std=imagenet_std)
                      ]

    transforms = T.Compose(transforms_list)

    return transforms
