from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import defaultdict
from collections import OrderedDict
import logging
import os
import os.path
import json_tricks as json
import numpy as np
from torch.utils.data import Dataset

from pycocotools.cocoeval_center_emotions import COCOeval_center_emotions
from utils.utils import EmotionConverter
from sklearn.metrics import average_precision_score
from PIL.Image import open as imread

logger = logging.getLogger(__name__)


class MixedDataset(Dataset):
    def __init__(self, root, dataset, transform=None):
        from pycocotools.coco import COCO
        self.name = 'mixed'
        self.root = root
        self.dataset = dataset
        if "test" in dataset:
            self.coco_bu_mt = COCO(f'new_annotations/EMOTIC_test_x1y1x2y2.json')
            self.ids_bu_mt = list(self.coco_bu_mt.imgs.keys())
        self.coco = COCO(f'new_annotations/{self.dataset}.json')
        self.ids = list(self.coco.imgs.keys())

        self.transform = transform

        cats = [cat['name']
                for cat in self.coco.loadCats(self.coco.getCatIds())]
        self.classes = ['__background__'] + cats
        logger.info('=> classes: {}'.format(self.classes))
        self.num_classes = len(self.classes)
        self._class_to_ind = dict(zip(self.classes, range(self.num_classes)))
        self._class_to_coco_ind = dict(zip(cats, self.coco.getCatIds()))
        self._coco_ind_to_class_ind = dict(
            [
                (self._class_to_coco_ind[cls], self._class_to_ind[cls])
                for cls in self.classes[1:]
            ]
        )
        self.converter = EmotionConverter(dataset)
        self.num_cat_emo = 0
        self.num_cont_emo = 0

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)
        img_infos = coco.loadImgs(img_id)[0]

        database = img_infos["database"]
        if database == "HECO":
            folder = os.path.join(self.root, img_infos["folder"])
        elif database == 'EMOTIC':
            folder = os.path.join(self.root, 'EMOTIC/PAMI/emotic', img_infos['folder'])
        else:
            raise ValueError(f"Database '{database}' not handled")

        file_name = coco.loadImgs(img_id)[0]['file_name']
        path = os.path.join(folder, file_name)
        img = imread(path).convert('RGB')

        all_targets = []
        for i in range(len(target[0]["bbox"])):
            all_targets.append({"bbox": target[0]["bbox"][i],
                                "annotations_categories": target[0]["annotations_categories"][i],
                                "annotations_continuous": target[0]["annotations_continuous"][i]})

        transformed, target, all_targets = self.transform(img, target, all_targets)

        return transformed, target, all_targets, path, database

    def __len__(self):
        return len(self.ids)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

    def evaluate(self, cfg, preds, results_without_det, scores, output_dir, temp=False, state=""):
        res_folder = os.path.join(output_dir, 'results')
        if not os.path.exists(res_folder):
            os.makedirs(res_folder)
        if temp:
            res_file = os.path.join(res_folder, f'temp_{self.dataset}_results.json')
            res_without_det_file = os.path.join(res_folder, f'temp_{self.dataset}_results_without_det.json')
        else:
            res_file = os.path.join(res_folder, f'{self.dataset}_results_{state}.json')
            res_without_det_file = os.path.join(res_folder, f'{self.dataset}_results_without_det_{state}.json')

        self.num_cat_emo = cfg.MODEL.NUM_CAT_EMOTIONS
        self.num_cont_emo = cfg.MODEL.NUM_CONT_EMOTIONS

        # preds is a list of: image x person x (bbox), or it is an empty list if we don't use the detection feature
        # bbox: 1 * (x1, y1, x2, y2, score, bu_cat, bu_cont)

        bboxs = defaultdict(list)
        for idx, _bbox in enumerate(preds):
            img_id = self.ids_bu_mt[idx]
            file_name = self.coco_bu_mt.loadImgs(img_id)[0]['file_name']

            for idx_bbox, bbox in enumerate(_bbox):
                area = (bbox[0, 2] - bbox[0, 0]) * (bbox[0, 3] - bbox[0, 1])

                bboxs[file_name[:-4]].append(
                    {
                        'bbox': bbox[0],
                        'score': scores[idx][idx_bbox],
                        'image': file_name[:-4],
                        'image_id': img_id,
                        'area': area
                    }
                )

        # rescoring and oks nms
        oks_nmsed_bboxs = []
        for img in bboxs.keys():
            img_bboxs = bboxs[img]
            keep = []
            if len(keep) == 0:
                oks_nmsed_bboxs.append(img_bboxs)
            else:
                oks_nmsed_bboxs.append([img_bboxs[_keep] for _keep in keep])

        self._write_coco_bbox_results(
            oks_nmsed_bboxs, results_without_det, res_file, res_without_det_file
        )

        if 'EMOTIC_test' in self.dataset:
            info_str, results_emo = self._do_python_bbox_eval(res_file, res_without_det_file)
            if info_str is not None:
                name_value = OrderedDict(info_str)
            else:
                name_value = None

            return name_value, results_emo
        else:
            return None, None

    def _write_coco_bbox_results(self, bboxs, results_without_det, res_file, res_without_det_file):
        if bboxs:
            data_pack = [
                {
                    'cat_id': self._class_to_coco_ind[cls],
                    'cls_ind': cls_ind,
                    'cls': cls,
                    'ann_type': 'bbox',
                    'bboxs': bboxs,
                }
                for cls_ind, cls in enumerate(self.classes) if not cls == '__background__'
            ]

            results = self._coco_bbox_results_one_category_kernel(data_pack[0])
            logger.info('=> Writing results json to %s' % res_file)
            with open(res_file, 'w') as f:
                json.dump(results, f, sort_keys=True, indent=4)
            try:
                json.load(open(res_file))
            except Exception:
                content = []
                with open(res_file, 'r') as f:
                    for line in f:
                        content.append(line)
                content[-1] = ']'
                with open(res_file, 'w') as f:
                    for c in content:
                        f.write(c)
        if results_without_det:
            logger.info('=> Writing results json to %s' % res_without_det_file)
            with open(res_without_det_file, 'w') as f:
                json.dump(results_without_det, f, sort_keys=True, indent=4)
            try:
                json.load(open(res_without_det_file))
            except Exception:
                content = []
                with open(res_without_det_file, 'r') as f:
                    for line in f:
                        content.append(line)
                content[-1] = ']'
                with open(res_without_det_file, 'w') as f:
                    for c in content:
                        f.write(c)

    def _coco_bbox_results_one_category_kernel(self, data_pack):
        cat_id = data_pack['cat_id']
        bboxs = data_pack['bboxs']
        cat_results = []

        for img_bbox in bboxs:
            if len(img_bbox) == 0:
                continue

            for k in range(len(img_bbox)):
                cat_results.append({
                    'image': img_bbox[k]['image'],
                    'image_id': img_bbox[k]['image_id'],
                    'emotions': img_bbox[k]['bbox'][5:].tolist(),
                    'category_id': cat_id,
                    'score': img_bbox[k]['score'],
                    'bbox': img_bbox[k]['bbox'][:4].tolist()
                })

        return cat_results

    def _do_python_bbox_eval(self, res_file, res_without_det):
        coco_eval = None
        if os.path.isfile(res_file):
            coco_dt = self.coco_bu_mt.loadRes(res_file)
            coco_eval = COCOeval_center_emotions(self.coco_bu_mt, coco_dt, 'bbox')

            coco_eval.params.useSegm = None
            coco_eval.evaluate()

        if os.path.isfile(res_without_det):
            emos_without_det = json.load(open(res_without_det))
            bu_cat_wo = emos_without_det["bu"]["cat"]
            bu_cont_wo = emos_without_det["bu"]["cont"]
            nb_anns = bu_cat_wo.shape[0]
        else:
            nb_anns = 1

        if self.num_cat_emo > 0 or self.num_cont_emo > 0:
            nb_thrs = 10
            ap_emo = None
            ap_emo_only_detected = None
            ap_emo_mixed = None
            vad = None
            vad_only_detected = None
            vad_mixed = None

            if self.num_cont_emo > 0:
                all_continuous_preds = dict()
                all_continuous_mixed_preds = dict()
                only_detected_continuous_preds = dict()
                only_detected_continuous_gts = dict()
                all_continuous_gts = np.array([]).reshape(0, self.num_cont_emo)
                for idx_thr in range(nb_thrs):
                    all_continuous_preds[idx_thr] = np.array([]).reshape(0, self.num_cont_emo)
                    all_continuous_mixed_preds[idx_thr] = np.array([]).reshape(0, self.num_cont_emo)
                    only_detected_continuous_preds[idx_thr] = np.array([]).reshape(0, self.num_cont_emo)
                    only_detected_continuous_gts[idx_thr] = np.array([]).reshape(0, self.num_cont_emo)

            if self.num_cat_emo > 0:
                all_categorical_preds = dict()
                all_categorical_mixed_preds = dict()
                only_detected_categorical_preds = dict()
                only_detected_categorical_gts = dict()
                all_categorical_gts = np.array([]).reshape(0, self.num_cat_emo)
                for idx_thr in range(nb_thrs):
                    all_categorical_preds[idx_thr] = np.array([]).reshape(0, self.num_cat_emo)
                    all_categorical_mixed_preds[idx_thr] = np.array([]).reshape(0, self.num_cat_emo)
                    only_detected_categorical_preds[idx_thr] = np.array([]).reshape(0, self.num_cat_emo)
                    only_detected_categorical_gts[idx_thr] = np.array([]).reshape(0, self.num_cat_emo)

            det_recall = []
            if coco_eval is not None:
                if self.num_cat_emo > 0:
                    ap_emo = np.zeros((nb_thrs, self.num_cat_emo))
                    ap_emo_mixed = np.zeros((nb_thrs, self.num_cat_emo))
                    ap_emo_only_detected = np.zeros((nb_thrs, self.num_cat_emo))
                if self.num_cont_emo > 0:
                    vad = np.zeros((nb_thrs, self.num_cont_emo))
                    vad_mixed = np.zeros((nb_thrs, self.num_cont_emo))
                    vad_only_detected = np.zeros((nb_thrs, self.num_cont_emo))
                for evalImg in coco_eval.evalImgs[0:len(coco_eval.params.imgIds)]:
                    for i, gtIgnore in enumerate(evalImg['gtIgnore']):
                        if gtIgnore:
                            continue
                        else:
                            gt_id = evalImg['gtIds'][i]
                            if self.num_cat_emo > 0:
                                oneHotGtCatEmo = self.converter.one_hot_encode(evalImg['gtCatEmo'][i], "EMOTIC")
                                all_categorical_gts = np.vstack((all_categorical_gts, oneHotGtCatEmo))

                            if self.num_cont_emo > 0:
                                cont = False
                                gtContEmo = self.converter.get_cont_emotions(
                                    [{"annotations_continuous": evalImg['gtContEmo'][i]}])[0]
                                if not np.any(np.isnan(gtContEmo)):
                                    all_continuous_gts = np.vstack((all_continuous_gts, gtContEmo))
                                    cont = True

                            if not evalImg['dtEmo']:
                                for idx_thr in range(nb_thrs):
                                    if self.num_cat_emo > 0:
                                        all_categorical_preds[idx_thr] = np.vstack((all_categorical_preds[idx_thr],
                                                                                    np.zeros_like(oneHotGtCatEmo)))

                                        all_categorical_mixed_preds[idx_thr] = np.vstack(
                                            (all_categorical_mixed_preds[idx_thr],
                                             bu_cat_wo[gt_id, self.num_cat_emo:])
                                        )

                                    if self.num_cont_emo > 0 and cont:
                                        all_continuous_preds[idx_thr] = np.vstack((all_continuous_preds[idx_thr],
                                                                                   np.zeros_like(gtContEmo)))

                                        all_continuous_mixed_preds[idx_thr] = np.vstack(
                                            (all_continuous_mixed_preds[idx_thr],
                                             bu_cont_wo[gt_id, self.num_cont_emo:])
                                        )

                            else:
                                gtMatches = evalImg['gtMatches'][:, i]
                                for idx_thr, match in enumerate(gtMatches):
                                    pos = np.where(evalImg['dtIds'] == match)[0]
                                    if pos.size > 0:
                                        pos = pos.item()
                                        dt = np.array(evalImg['dtEmo'][pos]).reshape(1, -1)

                                        if self.num_cat_emo > 0:
                                            dtCat = dt[:, :self.num_cat_emo].mean(axis=0)
                                            all_categorical_preds[idx_thr] = np.vstack((all_categorical_preds[idx_thr],
                                                                                        dtCat))

                                            all_categorical_mixed_preds[idx_thr] = np.vstack(
                                                (all_categorical_mixed_preds[idx_thr], dtCat))

                                            only_detected_categorical_preds[idx_thr] = np.vstack(
                                                (only_detected_categorical_preds[idx_thr], dtCat)
                                            )

                                            only_detected_categorical_gts[idx_thr] = np.vstack(
                                                (only_detected_categorical_gts[idx_thr], oneHotGtCatEmo)
                                            )

                                        if self.num_cont_emo > 0 and cont:
                                            dtCont = dt[:, self.num_cat_emo:].mean(axis=0)
                                            all_continuous_preds[idx_thr] = np.vstack((all_continuous_preds[idx_thr],
                                                                                       dtCont))

                                            all_continuous_mixed_preds[idx_thr] = np.vstack(
                                                (all_continuous_mixed_preds[idx_thr], dtCont))

                                            only_detected_continuous_preds[idx_thr] = np.vstack(
                                                (only_detected_continuous_preds[idx_thr], dtCont)
                                            )

                                            only_detected_continuous_gts[idx_thr] = np.vstack(
                                                (only_detected_continuous_gts[idx_thr], gtContEmo)
                                            )

                                    else:
                                        if self.num_cat_emo > 0:
                                            all_categorical_preds[idx_thr] = np.vstack((all_categorical_preds[idx_thr],
                                                                                        np.zeros_like(oneHotGtCatEmo)))

                                            all_categorical_mixed_preds[idx_thr] = np.vstack(
                                                (all_categorical_mixed_preds[idx_thr],
                                                 bu_cat_wo[gt_id, self.num_cat_emo:])
                                            )
                                        if self.num_cont_emo > 0 and cont:
                                            all_continuous_preds[idx_thr] = np.vstack((all_continuous_preds[idx_thr],
                                                                                       np.zeros_like(gtContEmo)))

                                            all_continuous_mixed_preds[idx_thr] = np.vstack(
                                                (all_continuous_mixed_preds[idx_thr],
                                                 bu_cont_wo[gt_id, self.num_cont_emo:])
                                            )

                for idx_thr in range(nb_thrs):
                    if self.num_cat_emo > 0:
                        assert all_categorical_preds[idx_thr].shape == all_categorical_gts.shape, \
                            fr'CatPred[{idx_thr}].shape = {all_categorical_preds[idx_thr].shape} =\=' \
                            fr'CatGT.shape = {all_categorical_gts.shape}'

                        assert only_detected_categorical_preds[idx_thr].shape == only_detected_categorical_gts[
                            idx_thr].shape, \
                            fr'OnlyDetCatPred[{idx_thr}].shape = {only_detected_categorical_preds[idx_thr].shape} =\=' \
                            fr'OnlyDetCatGT.shape = {only_detected_categorical_gts[idx_thr].shape}'
                        det_recall.append(only_detected_categorical_preds[idx_thr].shape[0] / nb_anns * 100)
                        assert all_categorical_mixed_preds[idx_thr].shape == all_categorical_gts.shape, \
                            fr'CatMixedPred[{idx_thr}].shape = {all_categorical_mixed_preds[idx_thr].shape} =\=' \
                            fr'CatGT.shape = {all_categorical_gts.shape}'

                        for i in range(self.num_cat_emo):
                            ap_emo[idx_thr, i] = average_precision_score(all_categorical_gts[:, i],
                                                                         all_categorical_preds[idx_thr][:, i]) * 100

                            if only_detected_categorical_gts[idx_thr].any():
                                ap_emo_only_detected[idx_thr, i] = average_precision_score(
                                    only_detected_categorical_gts[idx_thr][:, i],
                                    only_detected_categorical_preds[idx_thr][:, i]
                                ) * 100
                            else:
                                ap_emo_only_detected[idx_thr, i] = 0

                            ap_emo_mixed[idx_thr, i] = average_precision_score(
                                all_categorical_gts[:, i], all_categorical_mixed_preds[idx_thr][:, i]) * 100

                    if self.num_cont_emo > 0:
                        assert all_continuous_preds[idx_thr].shape == all_continuous_gts.shape, \
                            fr'ContPred[{idx_thr}].shape = {all_continuous_preds[idx_thr].shape} =\=' \
                            fr'ContGT.shape = {all_continuous_gts.shape}'

                        assert only_detected_continuous_preds[idx_thr].shape == only_detected_continuous_gts[
                            idx_thr].shape, \
                            fr'OnlyDetContPred[{idx_thr}].shape = {only_detected_continuous_preds[idx_thr].shape} =\=' \
                            fr'OnlyDetContGT.shape = {only_detected_continuous_gts[idx_thr].shape}'

                        assert all_continuous_mixed_preds[idx_thr].shape == all_continuous_gts.shape, \
                            fr'ContMixedPred[{idx_thr}].shape = {all_continuous_mixed_preds[idx_thr].shape} =\=' \
                            fr'ContGT.shape = {all_continuous_gts.shape}'

                        for i in range(self.num_cont_emo):
                            vad[idx_thr, i] = np.mean(
                                np.abs(all_continuous_preds[idx_thr][:, i] - all_continuous_gts[:, i]))

                            if only_detected_continuous_gts[idx_thr].any():
                                vad_only_detected[idx_thr, i] = np.mean(
                                    np.abs(only_detected_continuous_preds[idx_thr][:, i] -
                                           only_detected_continuous_gts[idx_thr][:, i])
                                )
                            else:
                                vad_only_detected[idx_thr, i] = 1

                            vad_mixed[idx_thr, i] = np.mean(
                                np.abs(all_continuous_mixed_preds[idx_thr][:, i] - all_continuous_gts[:, i])
                            )

            ap_emo_wo_det = np.zeros(self.num_cat_emo)
            for i in range(self.num_cat_emo):
                ap_emo_wo_det[i] = average_precision_score(bu_cat_wo[:, i],
                                                           bu_cat_wo[:, i + self.num_cat_emo]) * 100
            if self.num_cont_emo > 0:
                vad_wo_det = np.zeros(self.num_cont_emo)
                for i in range(self.num_cont_emo):
                    vad_wo_det[i] = np.mean(
                        np.abs(bu_cont_wo[:, i] - bu_cont_wo[:, i + self.num_cont_emo])
                    )
            else:
                vad_wo_det = None

            ind2thr = dict()
            for i, thr in enumerate([' 0.50', ' 0.55', ' 0.60', ' 0.65', ' 0.70', ' 0.75', ' 0.80', ' 0.85', ' 0.90',
                                     ' 0.95']):
                ind2thr[i] = thr

            ind2cat = self.converter.ind_to_cat
            results_emo = {"bu_cat_all_dets": ap_emo,
                           "det_recall": det_recall,
                           "bu_cat_only_det": ap_emo_only_detected,
                           "bu_cat_wo_det": ap_emo_wo_det,
                           "bu_cat_mixed": ap_emo_mixed,
                           "bu_cont_all_dets": vad,
                           "bu_cont_only_det": vad_only_detected,
                           "bu_cont_wo_det": vad_wo_det,
                           "bu_cont_mixed": vad_mixed,
                           "ind2thr": ind2thr,
                           "ind2cat": ind2cat
                           }

            # Test each modality separately
            for mod in ["pc", "context", "fusion"]:
                val = emos_without_det[mod]

                if self.num_cat_emo > 0:
                    temp = np.zeros(self.num_cat_emo)
                    for i in range(self.num_cat_emo):
                        temp[i] = average_precision_score(val["cat"][:, i],
                                                          val["cat"][:, i + self.num_cat_emo]) * 100
                    results_emo[f"{mod}_cat"] = temp

                if self.num_cont_emo > 0:
                    temp = np.zeros(self.num_cont_emo)
                    for i in range(self.num_cont_emo):
                        temp[i] = np.mean(np.abs(val["cont"][:, i] - val["cont"][:, i + self.num_cont_emo]))
                    results_emo[f"{mod}_cont"] = temp

            # Average main modality with other modalities:
            for mod in ["bu", "context_rep"]:
                val = emos_without_det[mod]
                if self.num_cat_emo > 0:
                    temp = np.zeros(self.num_cat_emo)
                    for i in range(self.num_cat_emo):
                        temp[i] = average_precision_score(
                            emos_without_det["pc"]["cat"][:, i],  # GT
                            np.mean(
                                (val["cat"][:, i + self.num_cat_emo],
                                 emos_without_det["pc"]["cat"][:, i + self.num_cat_emo]),
                                axis=0  # Averaged preds
                            )
                        ) * 100
                    results_emo[f"pc-{mod if mod == 'bu' else 'context'}_cat"] = temp

                if self.num_cont_emo > 0:
                    temp = np.zeros(self.num_cont_emo)
                    for i in range(self.num_cont_emo):
                        temp[i] = np.mean(
                            np.abs(emos_without_det["pc"]["cont"][:, i] -
                                   np.mean(
                                       (val["cont"][:, i + self.num_cont_emo],
                                        emos_without_det["pc"]["cont"][:, i + self.num_cont_emo]), axis=0
                                   )
                                   )
                        )
                    results_emo[f"pc-{mod if mod == 'bu' else 'context'}_cont"] = temp

            if self.num_cat_emo > 0:
                temp = np.zeros(self.num_cat_emo)
                for i in range(self.num_cat_emo):
                    temp[i] = average_precision_score(
                        emos_without_det["bu"]["cat"][:, i],  # GT
                        np.mean(
                            (emos_without_det["bu"]["cat"][:, i + self.num_cat_emo],
                             emos_without_det["context_rep"]["cat"][:, i + self.num_cat_emo]),
                            axis=0  # Averaged preds
                        )
                    ) * 100
                results_emo[f"bu-context_cat"] = temp

            if self.num_cont_emo > 0:
                temp = np.zeros(self.num_cont_emo)
                for i in range(self.num_cont_emo):
                    temp[i] = np.mean(
                        np.abs(emos_without_det["bu"]["cont"][:, i] -
                               np.mean(
                                   (emos_without_det["bu"]["cont"][:, i + self.num_cont_emo],
                                    emos_without_det["context_rep"]["cont"][:, i + self.num_cont_emo]), axis=0
                               )
                               )
                    )
                results_emo[f"bu-context_cont"] = temp

            # Average all modalities
            if self.num_cat_emo > 0:
                temp = np.zeros(self.num_cat_emo)
                cat_mean = np.mean((emos_without_det["bu"]["cat"][:, self.num_cat_emo:],
                                    emos_without_det["pc"]["cat"][:, self.num_cat_emo:],
                                    emos_without_det["context_rep"]["cat"][:, self.num_cat_emo:]), axis=0)

                for i in range(self.num_cat_emo):
                    temp[i] = average_precision_score(
                        emos_without_det["pc"]["cat"][:, i],  # GT
                        cat_mean[:, i]
                    ) * 100
                results_emo[f"pc-bu-context_cat"] = temp

            if self.num_cont_emo > 0:
                temp = np.zeros(self.num_cont_emo)
                cont_mean = np.mean((emos_without_det["bu"]["cont"][:, self.num_cont_emo:],
                                     emos_without_det["pc"]["cont"][:, self.num_cont_emo:],
                                     emos_without_det["context_rep"]["cont"][:, self.num_cont_emo:]), axis=0)

                for i in range(self.num_cont_emo):
                    temp[i] = np.mean(
                        np.abs(emos_without_det["pc"]["cont"][:, i] - cont_mean[:, i])
                    )
                results_emo[f"pc-bu-context_cont"] = temp

        if coco_eval is not None:
            coco_eval.accumulate()
            coco_eval.summarize()
            stats_names = ['AP', 'AP .5', 'AP .75', 'AP (S)', 'AP (M)', 'AP (L)', 'AR-1', 'AR-10', 'AR-100', 'AR (S)',
                           'AR (M)', 'AR (L)']

            info_str = []
            for ind, name in enumerate(stats_names):
                info_str.append((name, coco_eval.stats[ind]))
        else:
            info_str = None

        return info_str, results_emo
