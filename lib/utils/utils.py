from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import time
from pathlib import Path

import torch
import torch.optim as optim
import numpy as np


def setup_logger(final_output_dir, rank, phase):
    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}_{}_rank{}.log'.format(phase, time_str, rank)
    final_log_file = os.path.join(final_output_dir, log_file)
    head = '%(asctime)-15s %(message)s'
    # logging.basicConfig(format=head)
    logging.basicConfig(filename=str(final_log_file),
                        format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    return logger, time_str


def create_logger(cfg, cfg_name, phase='train'):
    root_output_dir = Path(cfg.OUTPUT_DIR)
    # set up logger
    if not root_output_dir.exists() and cfg.RANK == 0:
        print('=> creating {}'.format(root_output_dir))
        root_output_dir.mkdir(parents=True, exist_ok=True)
    else:
        while not root_output_dir.exists():
            print('=> wait for {} created'.format(root_output_dir))
            time.sleep(30)

    dataset = cfg.DATASET.DATASET
    dataset = dataset.replace(':', '_')
    model = cfg.MODEL.NAME
    cfg_name = os.path.basename(cfg_name).split('.')[0]

    final_output_dir = root_output_dir / dataset / model / cfg_name

    if cfg.RANK == 0:
        print('=> creating {}'.format(final_output_dir))
        final_output_dir.mkdir(parents=True, exist_ok=True)
    else:
        while not final_output_dir.exists():
            print('=> wait for {} created'.format(final_output_dir))
            time.sleep(5)

    logger, time_str = setup_logger(final_output_dir, cfg.RANK, phase)

    tensorboard_log_dir = Path(cfg.LOG_DIR) / dataset / model / \
        (cfg_name + '_' + time_str)

    print('=> creating {}'.format(tensorboard_log_dir))
    tensorboard_log_dir.mkdir(parents=True, exist_ok=True)

    return logger, str(final_output_dir), str(tensorboard_log_dir)


def get_optimizer(cfg, model):
    optimizer = None
    if cfg.TRAIN.OPTIMIZER == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=cfg.TRAIN.LR,
            momentum=cfg.TRAIN.MOMENTUM,
            weight_decay=cfg.TRAIN.WD,
            nesterov=cfg.TRAIN.NESTEROV
        )
    elif cfg.TRAIN.OPTIMIZER == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=cfg.TRAIN.LR
        )

    return optimizer


def save_checkpoint(states, is_best, output_dir, filename='checkpoint.pth.tar'):
    torch.save(states, os.path.join(output_dir, filename))

    if is_best:
        torch.save(
            states['best_state_dict'],
            os.path.join(output_dir, 'model_best.pth.tar')
        )


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0


class EmotionConverter:
    def __init__(self, dataset_name, other_name=''):

        name = dataset_name.split("_")[0]

        all_cats = {"EMOTIC": ['Affection', 'Anger', 'Annoyance', 'Anticipation', 'Aversion', 'Confidence',
                               'Disapproval', 'Disconnection', 'Disquietment', 'Doubt/Confusion', 'Embarrassment',
                               'Engagement', 'Esteem', 'Excitement', 'Fatigue', 'Fear', 'Happiness', 'Pain', 'Peace',
                               'Pleasure', 'Sadness', 'Sensitivity', 'Suffering', 'Surprise', 'Sympathy', 'Yearning'],
                    "CAERS": ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise'],
                    "HECO": ['Anger', 'Disgust', 'Excitement', 'Fear', 'Happiness', 'Peace', 'Sadness', 'Surprise']}

        cat = all_cats[name]
        ind = [i for i in range(len(cat))]

        cat_to_ind = dict()
        ind_to_cat = dict()
        cat_to_ind[name] = dict(zip(cat, ind))
        ind_to_cat[name] = dict(zip(ind, cat))

        if other_name:
            cat = all_cats[other_name]
            ind = [i for i in range(len(cat))]
            cat_to_ind[other_name] = dict(zip(cat, ind))
            ind_to_cat[other_name] = dict(zip(ind, cat))

        self.cat_to_ind = cat_to_ind
        self.ind_to_cat = ind_to_cat

    def one_hot_encode(self, anno, dataset_name):
        list_index = []
        one_hot = np.zeros(len(self.cat_to_ind[dataset_name]))
        for emo in anno:
            if isinstance(emo, tuple):
                emo = emo[0]
            list_index.append(self.cat_to_ind[dataset_name][emo])


        one_hot[list_index] = 1.
        assert len(anno) == np.count_nonzero(one_hot)
        return one_hot

    def get_cat_emotions(self, anno, dataset_name):
        if 'annotations_categories' not in anno[0].keys():
            return np.array([], dtype=np.float32)
        else:
            num_people = len(anno)
            cat_emotions = np.zeros((num_people, len(self.cat_to_ind[dataset_name])), dtype=np.float32)
            for i, obj in enumerate(anno):
                cat_emotions[i] = self.one_hot_encode(obj['annotations_categories'], dataset_name)
            return cat_emotions

    @staticmethod
    def get_cont_emotions(anno):
        if 'annotations_continuous' not in anno[0].keys():
            return np.array([], dtype=np.float32)

        else:
            num_people = len(anno)
            cont_emotions = np.zeros((num_people, 3), dtype=np.float32)
            for i, obj in enumerate(anno):
                valence = obj['annotations_continuous']['valence']
                arousal = obj['annotations_continuous']['arousal']
                dominance = obj['annotations_continuous']['dominance']
                cont_emotions[i] = (float(valence)/10 if valence != 'None' else None,
                                    float(arousal)/10 if arousal != 'None' else None,
                                    float(dominance)/10 if dominance != 'None' else None)  # from 0-10 to 0-1
            return cont_emotions
