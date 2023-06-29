from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import numpy as np

from .MixedDataset import MixedDataset
from utils.utils import EmotionConverter

logger = logging.getLogger(__name__)


class MixedCenterEmotion(MixedDataset):
    def __init__(self,
                 cfg,
                 dataset_name,
                 remove_images_without_annotations,
                 bb_generator,
                 bu_cat_generator,
                 bu_cont_generator,
                 transforms=None,
                 pre_transforms=None):
        super().__init__(cfg.DATASET.ROOT,
                         dataset_name,
                         transform=pre_transforms)
        self.cfg = cfg
        self.num_scales = self._init_check(bb_generator, bu_cat_generator, bu_cont_generator)
        self.max_num_people = cfg.DATASET.MAX_NUM_PEOPLE
        self.base_sigma = cfg.DATASET.BASE_SIGMA
        self.base_size = cfg.DATASET.BASE_SIZE
        self.int_sigma = cfg.DATASET.INT_SIGMA

        if remove_images_without_annotations:
            self.ids = [
                img_id
                for img_id in self.ids
                if len(self.coco.getAnnIds(imgIds=img_id, iscrowd=None)) > 0
            ]

        self.transforms = transforms
        self.bb_generator = bb_generator
        self.bu_cat_generator = bu_cat_generator
        self.bu_cont_generator = bu_cont_generator
        self.converter = EmotionConverter(dataset_name, other_name=cfg.DATASET.MIXED_WITH)

    def __getitem__(self, idx):
        # when all_anns is not None, it corresponds to all the person annotated on the current image, and anno
        # is usefull for the bottom up part (some subjects may have been removed with RandomMaskedSubjects)
        pre_transformed, anno, all_anns, path, database = super().__getitem__(idx)

        img_type = pre_transformed["head"]
        img = pre_transformed["image"]
        from_mixed_database = database == self.cfg.DATASET.MIXED_WITH
        if img_type in ["det", "bu"]:
            if img_type == "det":
                anno = all_anns

            bbox = self.get_bbox(anno)
            bbox_list = [bbox.copy() for _ in range(self.num_scales)]

            if self.transforms:
                img, bbox_list = self.transforms(img, img_type, bbox_list)

            if img_type == "bu":
                cat_emotions = self.converter.get_cat_emotions(anno, database)
                cont_emotions = self.converter.get_cont_emotions(anno)
                cat_list = [cat_emotions.copy() for _ in range(self.num_scales)]
                cont_list = [cont_emotions.copy() for _ in range(self.num_scales)]

                for scale_id in range(self.num_scales):
                    id_bis = scale_id+self.num_scales if from_mixed_database else scale_id
                    cat_emo_t = self.bu_cat_generator[id_bis](bbox_list[scale_id], cat_list[scale_id])
                    cat_list[scale_id] = cat_emo_t.astype(np.int32)

                for scale_id in range(self.num_scales):
                    cont_emo_t = self.bu_cont_generator[scale_id](bbox_list[scale_id], cont_list[scale_id])
                    cont_list[scale_id] = cont_emo_t.astype(np.float32)

            else:
                hm_list = []
                hw_list = []
                for scale_id in range(self.num_scales):
                    hm, hw = self.bb_generator[scale_id](bbox_list[scale_id])
                    hm_list.append(hm)
                    hw_list.append(hw)

        elif img_type in ["context", "pc", "fusion"]:
            bbox_list = []
            if img_type == "context":
                cat_emotions = self.converter.get_cat_emotions(all_anns, database)
                cat_list = cat_emotions.max(axis=0)[np.newaxis]
                cont_emotions = self.converter.get_cont_emotions(all_anns)
                cont_list = cont_emotions.mean(axis=0)[np.newaxis]
            elif img_type in ["pc", "fusion"]:
                if img_type == "fusion":
                    bbox = self.get_bbox(all_anns)
                    bbox_list = [bbox.copy() for _ in range(self.num_scales)]
                cat_emotions = self.converter.get_cat_emotions([all_anns[0]], database)
                cat_list = cat_emotions
                cont_emotions = self.converter.get_cont_emotions([all_anns[0]])
                cont_list = cont_emotions

            if self.transforms:
                img, bbox_list = self.transforms(img, img_type, bbox_list)
                if img_type == "fusion":
                    centers = []
                    for bb in bbox_list:
                        x = ((bb[:, 0] + bb[:, 2]) / 2).astype(int).tolist()
                        y = ((bb[:, 1] + bb[:, 3]) / 2).astype(int).tolist()
                        centers.append(zip(x, y))

        img = {"image": img, "head": img_type}
        if img_type == "fusion":
            img["centers"] = centers
        if img_type == "det":
            bb_list = {"hm": hm_list, "hw": hw_list}
            cat_list = cont_list = None
        else:
            bb_list = None

        return img, bb_list, cat_list, cont_list, int(from_mixed_database)

    @staticmethod
    def get_bbox(anno):
        num_people = len(anno)
        bbox = np.zeros((num_people, 4))

        for i, obj in enumerate(anno):
            bbox[i] = obj['bbox']

        return bbox

    def _init_check(self, bb_generator, cat_emo_generator, cont_emo_generator):
        assert isinstance(bb_generator, (list, tuple)), 'bb_generator should be a list or tuple'
        assert isinstance(cat_emo_generator, (list, tuple)), 'cat_emo_generator should be a list or tuple'
        assert isinstance(cont_emo_generator, (list, tuple)), 'cont_emo_generator should be a list or tuple'

        return len(bb_generator)
