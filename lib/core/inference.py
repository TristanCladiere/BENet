from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import numpy as np


def get_outputs(cfg, model_cfg, model, inputs, with_flip=False, project2image=False, size_projected=None, mt=True):
    nb_cat = model_cfg.NUM_CAT_EMOTIONS
    nb_cont = model_cfg.NUM_CONT_EMOTIONS
    nb_res = len(cfg.DATASET.OUTPUT_SIZE)
    with torch.no_grad():
        outputs = model(inputs)

        hm = outputs["hm"]
        hw = outputs["hw"]
        bu = outputs["bu"]
        pc = outputs["pc"]
        context = outputs["context"]
        fusion = outputs["fusion"]

        results = {"hm": [0, 0] if with_flip else [0],
                   "hw": [0, 0] if with_flip else [0],
                   "bu": {"cat": [0, 0] if with_flip else [0],
                          "cont": [0, 0] if with_flip else [0]},
                   "pc": {"cat": [0, 0] if with_flip else [0],
                          "cont": [0, 0] if with_flip else [0]},
                   "context": {"cat": [0, 0] if with_flip else [0],
                               "cont": [0, 0] if with_flip else [0]},
                   "fusion": {"cat": [0, 0] if with_flip else [0],
                              "cont": [0, 0] if with_flip else [0]}}

        for i in range(nb_res):
            if nb_res > 1 and i != nb_res - 1:
                hm[i] = torch.nn.functional.interpolate(
                    hm[i],
                    size=(hm[-1].size(2), hm[-1].size(3)),
                    mode='bilinear',
                    align_corners=False
                )

                hw[i] = torch.nn.functional.interpolate(
                    hw[i],
                    size=(hw[-1].size(2), hw[-1].size(3)),
                    mode='bilinear',
                    align_corners=False
                ) * (2 ** (nb_res - 1 - i))

                bu[i] = torch.nn.functional.interpolate(
                    bu[i],
                    size=(bu[-1].size(2), bu[-1].size(3)),
                    mode='bilinear',
                    align_corners=False
                )

            results["hm"][0] += torch.sigmoid(hm[i])
            results["hw"][0] += hw[i]

            if nb_cat > 0:
                results["bu"]["cat"][0] += torch.sigmoid(bu[i][:, :nb_cat])

            if nb_cont > 0:
                results["bu"]["cont"][0] += torch.sigmoid(bu[i][:, nb_cat:nb_cat+nb_cont])

        results["hm"][0] = results["hm"][0] / nb_res
        results["hw"][0] = results["hw"][0] / nb_res
        if mt:
            pc = torch.sigmoid(pc[0])
            context = torch.sigmoid(context[0])
            fusion = torch.sigmoid(fusion[0])

        if nb_cat > 0:
            results["bu"]["cat"][0] = results["bu"]["cat"][0] / nb_res
            if mt:
                results["pc"]["cat"][0] = pc[:nb_cat]
                results["context"]["cat"][0] = context[:nb_cat]
                results["fusion"]["cat"][0] = fusion[:nb_cat]

        if nb_cont > 0:
            results["bu"]["cont"][0] = results["bu"]["cont"][0] / nb_res
            if mt:
                results["pc"]["cont"][0] = pc[nb_cat:nb_cat+nb_cont]
                results["context"]["cont"][0] = context[nb_cat:nb_cat+nb_cont]
                results["fusion"]["cont"][0] = fusion[nb_cat:nb_cat+nb_cont]

        if with_flip:
            inputs["images"] = torch.flip(inputs["images"], [3])
            if mt:
                for i, res in enumerate(cfg.DATASET.OUTPUT_SIZE):
                    for j, (x, y) in enumerate(inputs["centers"][0][i]):
                        inputs["centers"][0][i][j] = (res - x - 1, y)

            flipped_outputs = model(inputs)
            hm = flipped_outputs["hm"]
            hw = flipped_outputs["hw"]
            bu = flipped_outputs["bu"]
            pc = flipped_outputs["pc"]
            context = flipped_outputs["context"]
            fusion = flipped_outputs["fusion"]

            for i in range(nb_res):
                if nb_res > 1 and i != nb_res - 1:
                    hm[i] = torch.nn.functional.interpolate(
                        hm[i],
                        size=(hm[-1].size(2), hm[-1].size(3)),
                        mode='bilinear',
                        align_corners=False
                    )

                    hw[i] = torch.nn.functional.interpolate(
                        hw[i],
                        size=(hw[-1].size(2), hw[-1].size(3)),
                        mode='bilinear',
                        align_corners=False
                    ) * (2 ** (nb_res - 1 - i))

                    bu[i] = torch.nn.functional.interpolate(
                        bu[i],
                        size=(bu[-1].size(2), bu[-1].size(3)),
                        mode='bilinear',
                        align_corners=False
                    )

                results["hm"][1] += torch.sigmoid(torch.flip(hm[i], [3]))
                results["hw"][1] += torch.flip(hw[i], [3])
                bu[i] = torch.flip(bu[i], [3])
                if nb_cat > 0:
                    results["bu"]["cat"][1] += torch.sigmoid(bu[i][:, :nb_cat])

                if nb_cont > 0:
                    results["bu"]["cont"][1] += torch.sigmoid(bu[i][:, nb_cat:nb_cat + nb_cont])

            results["hm"][1] = results["hm"][1] / nb_res
            results["hw"][1] = results["hw"][1] / nb_res
            if mt:
                pc = torch.sigmoid(pc[0])
                context = torch.sigmoid(context[0])
                fusion = torch.sigmoid(fusion[0])

            if nb_cat > 0:
                results["bu"]["cat"][1] = results["bu"]["cat"][1] / nb_res
                if mt:
                    results["pc"]["cat"][1] = pc[:nb_cat]
                    results["context"]["cat"][1] = context[:nb_cat]
                    results["fusion"]["cat"][1] = fusion[:nb_cat]

            if nb_cont > 0:
                results["bu"]["cont"][1] = results["bu"]["cont"][0] / nb_res
                if mt:
                    results["pc"]["cont"][1] = pc[nb_cat:nb_cat + nb_cont]
                    results["context"]["cont"][1] = context[nb_cat:nb_cat + nb_cont]
                    results["fusion"]["cont"][1] = fusion[nb_cat:nb_cat + nb_cont]

        if project2image and size_projected:
            results["hm"] = [
                torch.nn.functional.interpolate(
                    hm,
                    size=(size_projected[1], size_projected[0]),
                    mode='bilinear',
                    align_corners=False
                )
                for hm in results["hm"]
            ]

            results["hw"] = [
                torch.nn.functional.interpolate(
                    hw,
                    size=(size_projected[1], size_projected[0]),
                    mode='bilinear',
                    align_corners=False
                ) * (min(size_projected)/max(cfg.DATASET.OUTPUT_SIZE))
                for hw in results["hw"]
            ]

            if nb_cat > 0:
                results["bu"]["cat"] = [
                    torch.nn.functional.interpolate(
                        bu_cat,
                        size=(size_projected[1], size_projected[0]),
                        mode='bilinear',
                        align_corners=False
                    )
                    for bu_cat in results["bu"]["cat"]
                ]

            if nb_cont > 0:
                results["bu"]["cont"] = [
                    torch.nn.functional.interpolate(
                        bu_cont,
                        size=(size_projected[1], size_projected[0]),
                        mode='bilinear',
                        align_corners=False
                    )
                    for bu_cont in results["bu"]["cont"]
                ]

        final_hm = (results["hm"][0] + results["hm"][1]) / 2.0 if with_flip else results["hm"][0]
        final_hw = (results["hw"][0] + results["hw"][1]) / 2.0 if with_flip else results["hw"][0]

        if nb_cat > 0:
            final_bu_cat = (results["bu"]["cat"][0] + results["bu"]["cat"][1]) / 2.0 \
                if with_flip else results["bu"]["cat"][0]
            if mt:
                final_pc_cat = (results["pc"]["cat"][0] + results["pc"]["cat"][1]) / 2.0 \
                    if with_flip else results["pc"]["cat"][0]
                final_context_cat = (results["context"]["cat"][0] + results["context"]["cat"][1]) / 2.0 \
                    if with_flip else results["context"]["cat"][0]
                final_fusion_cat = (results["fusion"]["cat"][0] + results["fusion"]["cat"][1]) / 2.0 \
                    if with_flip else results["fusion"]["cat"][0]
            else:
                final_pc_cat = final_context_cat = final_fusion_cat = None
        else:
            final_bu_cat = final_pc_cat = final_context_cat = final_fusion_cat = None

        if nb_cont > 0:
            final_bu_cont = (results["bu"]["cont"][0] + results["bu"]["cont"][1]) / 2.0 \
                if with_flip else results["bu"]["cont"][0]
            if mt:
                final_pc_cont = (results["pc"]["cont"][0] + results["pc"]["cont"][1]) / 2.0 \
                    if with_flip else results["pc"]["cont"][0]
                final_context_cont = (results["context"]["cont"][0] + results["context"]["cont"][1]) / 2.0 \
                    if with_flip else results["context"]["cont"][0]
                final_fusion_cont = (results["fusion"]["cont"][0] + results["fusion"]["cont"][1]) / 2.0 \
                    if with_flip else results["fusion"]["cont"][0]
            else:
                final_pc_cont = final_context_cont = final_fusion_cont = None

        else:
            final_bu_cont = final_pc_cont = final_context_cont = final_fusion_cont = None

    return final_hm, final_hw, final_bu_cat, final_bu_cont, final_pc_cat, final_pc_cont, final_context_cat, \
        final_context_cont, final_fusion_cat, final_fusion_cont


class Params(object):
    def __init__(self, cfg):
        self.num_points = 1
        self.max_num_people = cfg.DATASET.MAX_NUM_PEOPLE
        self.detection_threshold = cfg.TEST.DETECTION_THRESHOLD


class Decoder(object):
    def __init__(self, cfg):
        self.params = Params(cfg)
        self.pool = torch.nn.MaxPool2d(
            cfg.TEST.NMS_KERNEL, 1, cfg.TEST.NMS_PADDING
        )

    def nms(self, hm):
        maxm = self.pool(hm)
        maxm = torch.eq(maxm, hm).float()
        hm = hm * maxm
        return hm

    def top_k(self, hm, hw, bu_cat, bu_cont):
        hm = self.nms(hm)
        num_images = hm.size(0)
        num_points = hm.size(1)
        h = hm.size(2)
        w = hm.size(3)
        hm = hm.view(num_images, num_points, -1)
        val_k, ind = hm.topk(self.params.max_num_people, dim=2)

        hw = hw.view(num_images, hw.size(1), w * h)
        hw = torch.unsqueeze(hw, dim=1).expand(-1, num_points, -1, -1)
        hw_k = torch.zeros(num_images, num_points, self.params.max_num_people, hw.size(2))
        for i in range(hw.size(2)):
            hw_k[:, :, :, i] = torch.gather(hw[:, :, i, :], 2, ind)
        hw_k = hw_k.cpu().numpy()

        if bu_cat is not None:
            bu_cat = bu_cat.view(num_images, bu_cat.size(1), w*h)
            bu_cat = torch.unsqueeze(bu_cat, dim=1).expand(-1, num_points, -1, -1)
            bu_cat_k = torch.zeros(num_images, num_points, self.params.max_num_people, bu_cat.size(2))
            for i in range(bu_cat.size(2)):
                bu_cat_k[:, :, :, i] = torch.gather(bu_cat[:, :, i, :], 2, ind)

            bu_cat_k = bu_cat_k.cpu().numpy()
        else:
            bu_cat_k = None

        if bu_cont is not None:
            bu_cont = bu_cont.view(num_images, bu_cont.size(1), w*h)
            bu_cont = torch.unsqueeze(bu_cont, dim=1).expand(-1, num_points, -1, -1)
            bu_cont_k = torch.zeros(num_images, num_points, self.params.max_num_people, bu_cont.size(2))
            for i in range(bu_cont.size(2)):
                bu_cont_k[:, :, :, i] = torch.gather(bu_cont[:, :, i, :], 2, ind)

            bu_cont_k = bu_cont_k.cpu().numpy()
        else:
            bu_cont_k = None

        x = ind % w
        y = (ind / w).long()

        ind_k = torch.stack((x, y), dim=3)

        ans = {
            'loc_k': ind_k.cpu().numpy(),
            'val_k': val_k.cpu().numpy(),
            'hw_k': hw_k,
            'bu_cat_k': bu_cat_k,
            'bu_cont_k': bu_cont_k
        }

        return ans

    def adjust(self, loc_k, hm):

        for people_id, people in enumerate(loc_k):
            for point_id, point in enumerate(people):
                y, x = point[0:2]
                xx, yy = int(x), int(y)

                tmp = hm[0][point_id]
                if tmp[xx, min(yy+1, tmp.shape[1]-1)] > tmp[xx, max(yy-1, 0)]:
                    y += 0.25
                else:
                    y -= 0.25

                if tmp[min(xx+1, tmp.shape[0]-1), yy] > tmp[max(0, xx-1), yy]:
                    x += 0.25
                else:
                    x -= 0.25
                loc_k[people_id, point_id, 0:2] = (y+0.5, x+0.5)

        return loc_k

    def decode(self, hm, hw, bu_cat, bu_cont, adjust=True):

        temp_ans = self.top_k(hm, hw, bu_cat, bu_cont)
        loc_k = np.squeeze(temp_ans['loc_k'], axis=0).transpose(1, 0, 2)
        val_k = np.squeeze(temp_ans['val_k'], axis=0).transpose(1, 0)
        mask = val_k > self.params.detection_threshold
        if True in mask:
            if adjust:
                loc_k = self.adjust(loc_k[mask, None], hm)
            bbox = np.concatenate((loc_k, loc_k), axis=2)

            hw_k = np.squeeze(temp_ans['hw_k'], axis=0).transpose(1, 0, 2)
            hw_k = hw_k[mask, None]
            bbox[..., 0] = bbox[..., 0] - hw_k[..., 1] / 2
            bbox[..., 1] = bbox[..., 1] - hw_k[..., 0] / 2
            bbox[..., 2] = bbox[..., 2] + hw_k[..., 1] / 2
            bbox[..., 3] = bbox[..., 3] + hw_k[..., 0] / 2
            _ans = np.concatenate((bbox, val_k[mask, None, None]), axis=2).astype(np.float32)

            if temp_ans['bu_cat_k'] is not None:
                bu_cat_k = np.squeeze(temp_ans['bu_cat_k'], axis=0).transpose(1, 0, 2)
                ans = np.concatenate((_ans, bu_cat_k[mask, None]), axis=2).astype(np.float32)

            if temp_ans['bu_cont_k'] is not None:
                bu_cont_k = np.squeeze(temp_ans['bu_cont_k'], axis=0).transpose(1, 0, 2)
                ans = np.concatenate((ans, bu_cont_k[mask, None]), axis=2).astype(np.float32)

            ans = [ans]

        else:
            ans = [np.array([]).astype(np.float32)]

        scores = [i[:, 4].item() for i in ans[0]]

        return ans, scores
