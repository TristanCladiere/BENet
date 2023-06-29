from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from yacs.config import CfgNode as CN

_C = CN()

_C.OUTPUT_DIR = ''
_C.LOG_DIR = ''
_C.DATA_DIR = ''
_C.GPUS = (0,)
_C.WORKERS = 4
_C.AUTO_RESUME = False
_C.PIN_MEMORY = True
_C.RANK = 0
_C.VERBOSE = True
_C.DIST_BACKEND = 'nccl'  # gloo for Windows / nccl for Linux
_C.MULTIPROCESSING_DISTRIBUTED = True

# Cudnn related params
_C.CUDNN = CN()
_C.CUDNN.BENCHMARK = True
_C.CUDNN.DETERMINISTIC = False
_C.CUDNN.ENABLED = True


# common params for NETWORK
_C.MODEL = CN()
_C.MODEL.NAME = 'BENet'
_C.MODEL.INIT_WEIGHTS = True
_C.MODEL.PRETRAINED = ''
_C.MODEL.NUM_CAT_EMOTIONS = 26
_C.MODEL.NUM_CONT_EMOTIONS = 3
_C.MODEL.EXTRA = CN(new_allowed=True)
_C.MODEL.SYNC_BN = False
_C.MODEL.HEAD_EXPANSION = 4

_C.LOSS = CN()
_C.LOSS.WITH_HM_LOSS = (True, True)
_C.LOSS.HM_LOSS_FACTOR = (1.0, 1.0)
_C.LOSS.WITH_HW_LOSS = (True, True)
_C.LOSS.HW_LOSS_FACTOR = (0.1, 0.1)
_C.LOSS.WITH_BU_CAT_LOSS = (True, True)
_C.LOSS.BU_CAT_LOSS_FACTOR = (1.0, 1.0)
_C.LOSS.WITH_BU_CONT_LOSS = (True, True)
_C.LOSS.BU_CONT_LOSS_FACTOR = (1.0, 1.0)
_C.LOSS.WITH_PC_CAT_LOSS = (True,)
_C.LOSS.PC_CAT_LOSS_FACTOR = (1.0,)
_C.LOSS.WITH_PC_CONT_LOSS = (True,)
_C.LOSS.PC_CONT_LOSS_FACTOR = (1.0,)
_C.LOSS.WITH_CONTEXT_CAT_LOSS = (True,)
_C.LOSS.CONTEXT_CAT_LOSS_FACTOR = (1.0,)
_C.LOSS.WITH_CONTEXT_CONT_LOSS = (True,)
_C.LOSS.CONTEXT_CONT_LOSS_FACTOR = (1.0,)
_C.LOSS.WITH_FUSION_CAT_LOSS = (True,)
_C.LOSS.FUSION_CAT_LOSS_FACTOR = (1.0,)
_C.LOSS.WITH_FUSION_CONT_LOSS = (True,)
_C.LOSS.FUSION_CONT_LOSS_FACTOR = (1.0,)

# DATASET related params
_C.DATASET = CN()
_C.DATASET.ROOT = ''
_C.DATASET.DATASET = 'mixed_center_emotion'
_C.DATASET.DATASET_TEST = 'mixed'
_C.DATASET.WITH_VAL = True
_C.DATASET.VAL_STEP = 10
_C.DATASET.NUM_CAT_EMOTIONS = 26
_C.DATASET.NUM_CONT_EMOTIONS = 3
_C.DATASET.MAX_NUM_PEOPLE = 15
_C.DATASET.TRAIN = 'emotic_train_tdpc'
_C.DATASET.VAL = 'emotic_val_tdpc'
_C.DATASET.TEST = 'emotic_test_tdpc'
_C.DATASET.MIXED_WITH = ''
_C.DATASET.NUM_CAT_MIXED = 0


# training data augmentation
_C.DATASET.PERSPECTIVE = True
_C.DATASET.MAX_ROTATION = 5
_C.DATASET.MIN_SCALE = 0.75
_C.DATASET.MAX_SCALE = 1.5
_C.DATASET.SCALE_TYPE = 'short'
_C.DATASET.MAX_TRANSLATE = 40
_C.DATASET.INPUT_SIZE = 512
_C.DATASET.OUTPUT_SIZE = [128, 256]
_C.DATASET.FLIP = 0.5
_C.DATASET.PRE_TRANSFORMS = []

# heatmap generator (default is OUTPUT_SIZE/64)
_C.DATASET.SIGMA = -1
_C.DATASET.BASE_SIZE = 256.0
_C.DATASET.BASE_SIGMA = 2.0
_C.DATASET.INT_SIGMA = False

# train
_C.TRAIN = CN()

_C.TRAIN.LR_FACTOR = 0.1
_C.TRAIN.LR_STEP = [90, 110]
_C.TRAIN.LR = 0.001

_C.TRAIN.OPTIMIZER = 'adam'
_C.TRAIN.MOMENTUM = 0.9
_C.TRAIN.WD = 0.0001
_C.TRAIN.NESTEROV = False
_C.TRAIN.GAMMA1 = 0.99
_C.TRAIN.GAMMA2 = 0.0

_C.TRAIN.BEGIN_EPOCH = 0
_C.TRAIN.END_EPOCH = 140

_C.TRAIN.RESUME = False
_C.TRAIN.CHECKPOINT = ''

_C.TRAIN.IMAGES_PER_GPU = 32
_C.TRAIN.SHUFFLE = True

# testing
_C.TEST = CN()

# size of images for each device
_C.TEST.IMAGES_PER_GPU = 32
# Test Model Epoch
_C.TEST.FLIP_TEST = False
_C.TEST.ADJUST = True
_C.TEST.REFINE = True

# detection
_C.TEST.DETECTION_THRESHOLD = 0.2
_C.TEST.MODEL_FILE = ''
_C.TEST.NMS_KERNEL = 3
_C.TEST.NMS_PADDING = 1
_C.TEST.PROJECT2IMAGE = False


def update_config_mt(cfg, args):
    cfg.defrost()
    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)

    if not os.path.exists(cfg.DATASET.ROOT):
        cfg.DATASET.ROOT = os.path.join(
            cfg.DATA_DIR, cfg.DATASET.ROOT
        )

    cfg.MODEL.PRETRAINED = os.path.join(
        cfg.DATA_DIR, cfg.MODEL.PRETRAINED
    )

    if cfg.TEST.MODEL_FILE:
        cfg.TEST.MODEL_FILE = os.path.join(
            cfg.DATA_DIR, cfg.TEST.MODEL_FILE
        )

    if not isinstance(cfg.DATASET.OUTPUT_SIZE, (list, tuple)):
        cfg.DATASET.OUTPUT_SIZE = [cfg.DATASET.OUTPUT_SIZE]

    cfg.freeze()


if __name__ == '__main__':
    import sys
    with open(sys.argv[1], 'w') as f:
        print(_C, file=f)
