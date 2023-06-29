from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class EmoLoss(nn.Module):
    def __init__(self, loss):
        super().__init__()
        self.loss = loss

    def singleImgLoss(self, single_pred, single_gt):  # [nb_emo*res²] ; [max_people, nb_emo, 3]
        pred_all_people = []
        gt_all_people = []

        # Iterate through all the annotated people in the image
        for one_person in single_gt:  # [26, 3]
            if torch.any(torch.isnan(one_person[:, 0])):
                continue
            tmp_pred = []
            tmp_gt = []

            for emo in one_person:  # [3,]
                if emo[2] > 0:
                    tmp_pred.append(single_pred[int(emo[1])])
                    tmp_gt.append(emo[0].to(torch.float32))
            if len(tmp_pred) == 0:
                continue

            pred_all_people.append(torch.stack(tmp_pred))
            gt_all_people.append(torch.stack(tmp_gt))

        num_people = len(gt_all_people)
        if num_people == 0:
            return None
        else:
            return self.loss(torch.stack(pred_all_people), torch.stack(gt_all_people))

    def forward(self, pred, gt):
        tot_loss = []
        batch_size = pred.size(0)
        for i in range(batch_size):
            if gt[i].any():
                loss = self.singleImgLoss(pred[i], gt[i])
                if loss is not None:
                    tot_loss.append(loss)

        if len(tot_loss) == 0:
            return None
        else:
            return torch.stack(tot_loss)


class FocalLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, gt):
        tot_loss = []
        batch_size = pred.size(0)
        for i in range(batch_size):
            _gt = gt[i]
            if not torch.isnan(_gt).any():
                eps = 1e-6
                loss = 0
                _pred = torch.sigmoid(pred[i])

                pos_inds = _gt.eq(1).float()
                neg_inds = _gt.lt(1).float()

                neg_weights = torch.pow(1 - _gt, 4)

                pos_loss = torch.log(_pred + eps) * torch.pow(1 - _pred, 2) * pos_inds
                neg_loss = torch.log((1 - _pred) + eps) * torch.pow(_pred, 2) * neg_weights * neg_inds
                num_pos = pos_inds.sum()
                pos_loss = pos_loss.sum()
                neg_loss = neg_loss.sum()

                if num_pos == 0:
                    loss = loss - neg_loss
                else:
                    loss = loss - (pos_loss + neg_loss) / num_pos
                tot_loss.append(loss)

        if len(tot_loss) == 0:
            return None
        else:
            return torch.stack(tot_loss)


class RegL1Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def singleImgLoss(self, pred, gt):  # pred.shape = (2*res²) gt.shape = (max_num_people, 2, 3)
        pred_all_people = []
        gt_all_people = []

        # Iterate through all the annotated people in the image
        for one_person in gt:
            tmp_pred = []
            tmp_gt = []

            for dim in one_person:  # [3,]
                if dim[2] > 0:
                    tmp_pred.append(pred[int(dim[1])])
                    tmp_gt.append(dim[0].to(torch.float32))
            if len(tmp_pred) == 0:
                continue

            pred_all_people.append(torch.stack(tmp_pred))
            gt_all_people.append(torch.stack(tmp_gt))

        num_people = len(gt_all_people)
        if num_people == 0:
            return None
        else:
            return F.l1_loss(torch.stack(pred_all_people), torch.stack(gt_all_people), reduction="sum") / num_people

    def forward(self, preds, gts):
        tot_loss = []
        batch_size = preds.size(0)
        for i in range(batch_size):
            if gts[i].any():
                loss = self.singleImgLoss(preds[i], gts[i])
                if loss is not None:
                    tot_loss.append(loss)

        if len(tot_loss) == 0:
            return None
        else:
            return torch.stack(tot_loss)


class FocalTagLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, gt):
        eps = 1e-6
        tot_loss = []

        pos_inds = gt.eq(1).type(torch.float64)
        neg_inds = gt.lt(1).type(torch.float64)

        loss = 0
        pos_loss = torch.log(pred+eps) * torch.pow(1 - pred, 2) * pos_inds
        neg_loss = torch.log((1 - pred)+eps) * torch.pow(pred, 2) * neg_inds

        num_pos = pos_inds.float().sum(1)
        pos_loss = pos_loss.sum(1)
        neg_loss = neg_loss.sum(1)

        for i in range(num_pos.numel()):
            if num_pos[i] == 0:
                temp = loss - neg_loss[i]
            else:
                temp = loss - (pos_loss[i] + neg_loss[i])
            tot_loss.append(temp)

        return torch.stack(tot_loss).mean()


class MultiLossFactory(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.num_cat = cfg.MODEL.NUM_CAT_EMOTIONS
        self.num_cont = cfg.MODEL.NUM_CONT_EMOTIONS
        self.num_cat_mixed = cfg.DATASET.NUM_CAT_MIXED

        self.hm_loss = nn.ModuleList(
                [FocalLoss() if with_this_loss else None for with_this_loss in cfg.LOSS.WITH_HM_LOSS])
        self.hm_loss_factor = cfg.LOSS.HM_LOSS_FACTOR

        self.hw_loss = nn.ModuleList(
                [RegL1Loss() if with_this_loss else None for with_this_loss in cfg.LOSS.WITH_HW_LOSS])
        self.hw_loss_factor = cfg.LOSS.HW_LOSS_FACTOR

        self.bu_cat_loss = nn.ModuleList(
            [EmoLoss(FocalTagLoss()) if with_this_loss else None for with_this_loss in cfg.LOSS.WITH_BU_CAT_LOSS])
        self.bu_cat_loss_factor = cfg.LOSS.BU_CAT_LOSS_FACTOR

        self.bu_cont_loss = nn.ModuleList(
                [EmoLoss(nn.MSELoss()) if with_this_loss else None for with_this_loss in cfg.LOSS.WITH_BU_CONT_LOSS])
        self.bu_cont_loss_factor = cfg.LOSS.BU_CONT_LOSS_FACTOR

        self.pc_cat_loss = nn.ModuleList(
            [FocalTagLoss() if with_this_loss else None for with_this_loss in cfg.LOSS.WITH_PC_CAT_LOSS])
        self.pc_cat_loss_factor = cfg.LOSS.PC_CAT_LOSS_FACTOR

        self.pc_cont_loss = nn.ModuleList(
                [nn.MSELoss() if with_this_loss else None for with_this_loss in cfg.LOSS.WITH_PC_CONT_LOSS])
        self.pc_cont_loss_factor = cfg.LOSS.PC_CONT_LOSS_FACTOR

        self.context_cat_loss = nn.ModuleList(
            [FocalTagLoss() if with_this_loss else None for with_this_loss in cfg.LOSS.WITH_CONTEXT_CAT_LOSS])
        self.context_cat_loss_factor = cfg.LOSS.CONTEXT_CAT_LOSS_FACTOR

        self.context_cont_loss = nn.ModuleList(
                [nn.MSELoss() if with_this_loss else None for with_this_loss in cfg.LOSS.WITH_CONTEXT_CONT_LOSS])
        self.context_cont_loss_factor = cfg.LOSS.CONTEXT_CONT_LOSS_FACTOR

        self.fusion_cat_loss = nn.ModuleList(
            [FocalTagLoss() if with_this_loss else None for with_this_loss in cfg.LOSS.WITH_FUSION_CAT_LOSS])
        self.fusion_cat_loss_factor = cfg.LOSS.FUSION_CAT_LOSS_FACTOR

        self.fusion_cont_loss = nn.ModuleList(
                [nn.MSELoss() if with_this_loss else None for with_this_loss in cfg.LOSS.WITH_FUSION_CONT_LOSS])
        self.fusion_cont_loss_factor = cfg.LOSS.FUSION_CONT_LOSS_FACTOR

    def forward(self, outputs, custom_batch):

        # Preds
        hm = outputs["hm"]
        hw = outputs["hw"]
        bu = outputs["bu"]
        bu_mixed = outputs["bu_mixed"]
        pc = outputs["pc"]
        pc_mixed = outputs["pc_mixed"]
        context = outputs["context"]
        context_mixed = outputs["context_mixed"]
        fusion = outputs["fusion"]
        fusion_mixed = outputs["fusion_mixed"]

        hm_losses = []
        hw_losses = []
        bu_cat_losses = []
        bu_cont_losses = []
        pc_cat_losses = []
        pc_cont_losses = []
        context_cat_losses = []
        context_cont_losses = []
        fusion_cat_losses = []
        fusion_cont_losses = []

        if hm is not None:
            hm_gt = list(map(lambda x: x.cuda(non_blocking=True), custom_batch["hm"]))
            for idx in range(len(hm)):
                if self.hm_loss[idx]:
                    hm_loss = self.hm_loss[idx](hm[idx], hm_gt[idx])
                    if hm_loss is not None:
                        hm_loss *= self.hm_loss_factor[idx]
                        hm_losses.append(hm_loss)
                    else:
                        hm_losses.append(None)
                else:
                    hm_losses.append(None)

        if hw is not None:
            hw_gt = list(map(lambda x: x.cuda(non_blocking=True), custom_batch["hw"]))
            for idx in range(len(hw)):
                if self.hw_loss[idx]:
                    batch_size = hw[idx].size()[0]
                    hw[idx] = hw[idx].contiguous().view(batch_size, -1)
                    hw_loss = self.hw_loss[idx](hw[idx], hw_gt[idx])
                    if hw_loss is not None:
                        hw_loss *= self.hw_loss_factor[idx]
                        hw_losses.append(hw_loss)
                    else:
                        hw_losses.append(None)
                else:
                    hw_losses.append(None)

        for key, val in {"main": bu, "mixed": bu_mixed}.items():
            if val is not None:
                num_cat = self.num_cat if key == "main" else self.num_cat_mixed
                if num_cat > 0:
                    bu_cat_gt = list(map(lambda x: x.cuda(non_blocking=True), custom_batch["bu"]["cat_emo"][key]))
                if self.num_cont > 0:
                    bu_cont_gt = list(map(lambda x: x.cuda(non_blocking=True), custom_batch["bu"]["cont_emo"][key]))
                for idx in range(len(val)):
                    if self.bu_cat_loss[idx]:
                        bu_cat_pred = val[idx][:, :num_cat]
                        bu_cat_pred = torch.sigmoid(bu_cat_pred)
                        batch_size = bu_cat_pred.size()[0]
                        bu_cat_pred = bu_cat_pred.contiguous().view(batch_size, -1)
                        bu_cat_loss = self.bu_cat_loss[idx](bu_cat_pred, bu_cat_gt[idx])
                        if bu_cat_loss is not None:
                            bu_cat_loss = bu_cat_loss * self.bu_cat_loss_factor[idx]
                            bu_cat_losses.append(bu_cat_loss)
                        else:
                            bu_cat_losses.append(None)
                    else:
                        bu_cat_losses.append(None)

                    if self.bu_cont_loss[idx]:
                        bu_cont_pred = val[idx][:, num_cat:num_cat + self.num_cont]
                        bu_cont_pred = torch.sigmoid(bu_cont_pred)
                        batch_size = bu_cont_pred.size()[0]
                        bu_cont_pred = bu_cont_pred.contiguous().view(batch_size, -1)
                        bu_cont_loss = self.bu_cont_loss[idx](bu_cont_pred, bu_cont_gt[idx])
                        if bu_cont_loss is not None:
                            bu_cont_loss = bu_cont_loss * self.bu_cont_loss_factor[idx]
                            bu_cont_losses.append(bu_cont_loss)
                        else:
                            bu_cont_losses.append(None)
                    else:
                        bu_cont_losses.append(None)

        for key, val in {"main": pc, "mixed": pc_mixed}.items():
            if val is not None:
                num_cat = self.num_cat if key == "main" else self.num_cat_mixed
                if self.pc_cat_loss[0]:
                    pc_cat_gt = custom_batch["pc"]["cat_emo"][key].cuda(non_blocking=True)
                    batch_size = pc_cat_gt.size()[0]
                    pc_cat_loss_temp = []
                    pc_cat_pred = torch.sigmoid(val[:, :num_cat])
                    for i in range(batch_size):
                        if pc_cat_gt[i].any():
                            pc_cat_loss_temp.append(self.pc_cat_loss[0](
                                torch.unsqueeze(pc_cat_pred[i], dim=0), pc_cat_gt[i])
                            )
                    if len(pc_cat_loss_temp) > 0:
                        pc_cat_losses.append(torch.stack(pc_cat_loss_temp) * self.pc_cat_loss_factor[0])
                    else:
                        pc_cat_losses.append(None)
                else:
                    pc_cat_losses.append(None)

                if self.pc_cont_loss[0]:
                    pc_cont_gt = custom_batch["pc"]["cont_emo"][key].cuda(non_blocking=True)
                    batch_size = pc_cont_gt.size()[0]
                    pc_cont_loss_temp = []
                    pc_cont_pred = torch.sigmoid(val[:, num_cat:num_cat + self.num_cont])
                    for i in range(batch_size):
                        if pc_cont_gt[i][0].any() and not torch.any(torch.isnan(pc_cont_gt[i][0])):
                            pc_cont_loss_temp.append(self.pc_cont_loss[0](
                                pc_cont_pred[i], pc_cont_gt[i][0])
                            )
                    if len(pc_cont_loss_temp) > 0:
                        pc_cont_losses.append(torch.stack(pc_cont_loss_temp) * self.pc_cont_loss_factor[0])
                    else:
                        pc_cont_losses.append(None)
                else:
                    pc_cont_losses.append(None)

        for key, val in {"main": context, "mixed": context_mixed}.items():
            if val is not None:
                num_cat = self.num_cat if key == "main" else self.num_cat_mixed
                if self.context_cat_loss[0]:
                    context_cat_gt = custom_batch["context"]["cat_emo"][key].cuda(non_blocking=True)
                    batch_size = context_cat_gt.size()[0]
                    context_cat_loss_temp = []
                    context_cat_pred = torch.sigmoid(val[:, :num_cat])
                    for i in range(batch_size):
                        if context_cat_gt[i].any():
                            context_cat_loss_temp.append(self.context_cat_loss[0](
                                torch.unsqueeze(context_cat_pred[i], dim=0), context_cat_gt[i])
                            )
                    if len(context_cat_loss_temp) > 0:
                        context_cat_losses.append(torch.stack(context_cat_loss_temp) * self.context_cat_loss_factor[0])
                    else:
                        context_cat_losses.append(None)
                else:
                    context_cat_losses.append(None)

                if self.context_cont_loss[0]:
                    context_cont_gt = custom_batch["context"]["cont_emo"][key].cuda(non_blocking=True)
                    batch_size = context_cont_gt.size()[0]
                    context_cont_loss_temp = []
                    context_cont_pred = torch.sigmoid(val[:, num_cat:num_cat+self.num_cont])
                    for i in range(batch_size):
                        if context_cont_gt[i][0].any() and not torch.any(torch.isnan(context_cont_gt[i][0])):
                            context_cont_loss_temp.append(self.context_cont_loss[0](
                                context_cont_pred[i], context_cont_gt[i][0])
                            )
                    if len(context_cont_loss_temp) > 0:
                        context_cont_losses.append(torch.stack(context_cont_loss_temp) * self.context_cont_loss_factor[0])
                    else:
                        context_cont_losses.append(None)
                else:
                    context_cont_losses.append(None)

        for key, val in {"main": fusion, "mixed": fusion_mixed}.items():
            if val is not None:
                num_cat = self.num_cat if key == "main" else self.num_cat_mixed
                if self.fusion_cat_loss[0]:
                    fusion_cat_gt = custom_batch["fusion"]["cat_emo"][key].cuda(non_blocking=True)
                    batch_size = fusion_cat_gt.size()[0]
                    fusion_cat_loss_temp = []
                    fusion_cat_pred = torch.sigmoid(val[:, :num_cat])
                    for i in range(batch_size):
                        if fusion_cat_gt[i].any():
                            fusion_cat_loss_temp.append(self.fusion_cat_loss[0](
                                torch.unsqueeze(fusion_cat_pred[i], dim=0), fusion_cat_gt[i])
                            )
                    if len(fusion_cat_loss_temp) > 0:
                        fusion_cat_losses.append(torch.stack(fusion_cat_loss_temp) * self.fusion_cat_loss_factor[0])
                    else:
                        fusion_cat_losses.append(None)
                else:
                    fusion_cat_losses.append(None)

                if self.fusion_cont_loss[0]:
                    fusion_cont_gt = custom_batch["fusion"]["cont_emo"][key].cuda(non_blocking=True)
                    batch_size = fusion_cont_gt.size()[0]
                    fusion_cont_loss_temp = []
                    fusion_cont_pred = torch.sigmoid(val[:, num_cat:num_cat+self.num_cont])
                    for i in range(batch_size):
                        if fusion_cont_gt[i].any():
                            fusion_cont_loss_temp.append(self.fusion_cont_loss[0](
                                torch.unsqueeze(fusion_cont_pred[i], dim=0), fusion_cont_gt[i])
                            )
                    if len(fusion_cont_loss_temp) > 0:
                        fusion_cont_losses.append(torch.stack(fusion_cont_loss_temp) * self.fusion_cont_loss_factor[0])
                    else:
                        fusion_cont_losses.append(None)
                else:
                    fusion_cont_losses.append(None)

        return hm_losses, hw_losses, bu_cat_losses, bu_cont_losses, pc_cat_losses, pc_cont_losses, \
            context_cat_losses, context_cont_losses, fusion_cat_losses, fusion_cont_losses
