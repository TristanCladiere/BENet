from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F


BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class AttentionBlock(nn.Module):
    def __init__(self, input_channels, ratio=2):
        super(AttentionBlock, self).__init__()
        self.input_channels = input_channels
        self.ratio = ratio
        self.mlp = nn.Sequential(nn.Linear(input_channels, int(input_channels/ratio)),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(int(input_channels / ratio), input_channels))
        self.AvgPool2D = nn.AdaptiveAvgPool2d((1, 1))
        self.MaxPool2D = nn.AdaptiveMaxPool2d((1, 1))
        self.sigmoid = nn.Sigmoid()
        self.conv2d = nn.Conv2d(2, 1, kernel_size=1, bias=False)

    def forward(self, x):
        # Channel attention
        avg_pool = self.mlp(self.AvgPool2D(x).squeeze())
        max_pool = self.mlp(self.MaxPool2D(x).squeeze())
        ca = self.sigmoid(avg_pool + max_pool)
        temp = x * ca[..., None, None]

        # Spatial attention
        s_max = temp.max(dim=1).values[:, None]
        s_mean = temp.mean(dim=1)[:, None]
        sa = self.sigmoid(self.conv2d(torch.concat((s_max, s_mean), 1)))
        out = temp * sa

        return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.attention_block = AttentionBlock(planes)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.attention_block(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion,
                                  momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.attention_block = AttentionBlock(planes * self.expansion)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.attention_block(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class HeadBlock(nn.Module):

    def __init__(self, inplanes, planes, stride=1, downsample=None, expansion=4):
        super(HeadBlock, self).__init__()
        self.expansion = expansion
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion,
                                  momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.attention_block = AttentionBlock(planes)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.attention_block(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class FusionHead(nn.Module):
    def __init__(self, nb_emo):
        super(FusionHead, self).__init__()
        self.nb_emo = nb_emo
        self.mlp = nn.Sequential(nn.Linear(3*nb_emo, int(1.5*nb_emo)),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(int(1.5*nb_emo), nb_emo))

    def forward(self, out_bu, out_pc, out_context, centers):
        bu = 0
        for i_res, res in enumerate(out_bu):
            preds_one_res = []
            for i_img, img in enumerate(res):
                preds_all_subjects = []
                for xc, yc in centers[i_img][i_res]:
                    preds_all_subjects.append(img[:, yc, xc])
                preds_all_subjects = torch.stack(preds_all_subjects).mean(0)
                preds_one_res.append(preds_all_subjects)
            bu += torch.stack(preds_one_res)
        bu = bu / len(out_bu)
        x = torch.hstack([bu, out_pc, out_context])
        out_fusion = self.mlp(x)

        return out_fusion


class HighResolutionModule(nn.Module):
    def __init__(self, num_branches, blocks, num_blocks, num_inchannels, num_channels, fuse_method):
        super(HighResolutionModule, self).__init__()
        self._check_branches(num_branches, num_blocks, num_inchannels, num_channels)

        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches

        self.branches = self._make_branches(num_branches, blocks, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(True)

    def _check_branches(self, num_branches, num_blocks, num_inchannels, num_channels):
        if num_branches != len(num_blocks):
            error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(
                num_branches, len(num_blocks))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(
                num_branches, len(num_channels))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_inchannels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_INCHANNELS({})'.format(
                num_branches, len(num_inchannels))
            logger.error(error_msg)
            raise ValueError(error_msg)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels, stride=1):
        downsample = None
        if stride != 1 or \
           self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.num_inchannels[branch_index],
                          num_channels[branch_index] * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(num_channels[branch_index] * block.expansion,
                               momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.num_inchannels[branch_index],
                            num_channels[branch_index], stride, downsample))
        self.num_inchannels[branch_index] = \
            num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(block(self.num_inchannels[branch_index], num_channels[branch_index]))

        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []

        for i in range(num_branches):
            branches.append(
                self._make_one_branch(i, block, num_blocks, num_channels))

        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        num_final_channels = self.num_inchannels
        fuse_layers = []
        for i in range(num_branches):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(nn.Sequential(
                        nn.Conv2d(num_final_channels[j],
                                  num_final_channels[i],
                                  1,
                                  1,
                                  0,
                                  bias=False),
                        nn.BatchNorm2d(num_final_channels[i]),
                        nn.Upsample(scale_factor=2**(j-i), mode='nearest')))
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i-j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_final_channels[i]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_final_channels[j],
                                          num_outchannels_conv3x3,
                                          3, 2, 1, bias=False),
                                nn.BatchNorm2d(num_outchannels_conv3x3)))
                        else:
                            num_outchannels_conv3x3 = num_final_channels[j]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_final_channels[j],
                                          num_outchannels_conv3x3,
                                          3, 2, 1, bias=False),
                                nn.BatchNorm2d(num_outchannels_conv3x3),
                                nn.ReLU(True)))
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []

        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))

        return x_fuse


blocks_dict = {
    'BASIC': BasicBlock,
    'BOTTLENECK': Bottleneck,
}


class BENet(nn.Module):

    def __init__(self, cfg, model):
        self.cfg = cfg
        self.inplanes = 64
        extra = model.EXTRA
        super(BENet, self).__init__()

        nb_cat_emotions = model.NUM_CAT_EMOTIONS
        nb_cont_emotions = model.NUM_CONT_EMOTIONS
        self.nb_emotions = nb_cat_emotions + nb_cont_emotions
        self.nb_emotions_mixed = cfg.DATASET.NUM_CAT_MIXED + nb_cont_emotions if cfg.DATASET.NUM_CAT_MIXED else 0

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(Bottleneck, 64, 4)

        self.stage2_cfg = extra['STAGE2']
        num_channels = self.stage2_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage2_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition1 = self._make_transition_layer([256], num_channels)
        self.stage2, pre_stage_channels = self._make_stage(
            self.stage2_cfg, num_channels)

        self.stage3_cfg = extra['STAGE3']
        num_channels = self.stage3_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage3_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition2 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage3, pre_stage_channels = self._make_stage(
            self.stage3_cfg, num_channels)

        self.stage4_cfg = extra['STAGE4']
        num_channels = self.stage4_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage4_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition3 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage4, pre_stage_channels = self._make_stage(
            self.stage4_cfg, num_channels)

        self.num_deconvs = extra.DECONV.NUM_DECONVS
        self.deconv_config = model.EXTRA.DECONV
        self.final_kernel = extra.FINAL_CONV_KERNEL
        self.head_expansion = model.HEAD_EXPANSION
        self.loss_config = cfg.LOSS
        self.pretrained_layers = extra['PRETRAINED_LAYERS']

        # BU Head
        self.bu_final_layers = self._make_final_layers(pre_stage_channels[0], self.nb_emotions)
        self.bu_deconv_layers = self._make_deconv_layers(pre_stage_channels[0], self.nb_emotions)

        # Det Head
        self.hm_final_layers = self._make_final_layers(pre_stage_channels[0], 1)
        self.hw_final_layers = self._make_final_layers(pre_stage_channels[0], 2)
        self.hm_deconv_layers = self._make_deconv_layers(pre_stage_channels[0], 1)
        self.hw_deconv_layers = self._make_deconv_layers(pre_stage_channels[0], 2)

        # PC Head
        self.pc_incre_modules, self.pc_downsamp_modules, self.pc_final_layer = self._make_td_head(pre_stage_channels)
        self.pc_classifier = nn.Sequential(nn.Linear(128, 64),
                                           nn.ReLU(inplace=True),
                                           nn.Linear(64, self.nb_emotions))

        # Context Head
        self.incre_modules, self.downsamp_modules, self.final_layer = self._make_td_head(pre_stage_channels)
        self.classifier = nn.Sequential(nn.Linear(128, 64),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(64, self.nb_emotions))

        # Fusion Head
        self.fusion = FusionHead(self.nb_emotions)

        # Final layers if we used a mixed dataset
        if self.nb_emotions_mixed:
            self.bu_mixed = nn.ModuleList([
                nn.Conv2d(
                    in_channels=self.nb_emotions,
                    out_channels=self.nb_emotions_mixed,
                    kernel_size=self.final_kernel,
                    stride=1,
                    padding=1 if self.final_kernel == 3 else 0),
                nn.Conv2d(
                    in_channels=self.nb_emotions,
                    out_channels=self.nb_emotions_mixed,
                    kernel_size=self.final_kernel,
                    stride=1,
                    padding=1 if self.final_kernel == 3 else 0)
            ])

            self.pc_mixed = nn.Linear(self.nb_emotions, self.nb_emotions_mixed)
            self.context_mixed = nn.Linear(self.nb_emotions, self.nb_emotions_mixed)
            self.fusion_mixed = nn.Linear(self.nb_emotions, self.nb_emotions_mixed)

    def _make_td_head(self, pre_stage_channels):
        head_block = HeadBlock
        head_channels = [32, 64, 128, 256]
        incre_modules = []
        for i, channels in enumerate(pre_stage_channels):
            incre_module = self._make_head_layer(head_block,
                                                 channels,
                                                 head_channels[i],
                                                 1,
                                                 stride=1)
            incre_modules.append(incre_module)
        incre_modules = nn.ModuleList(incre_modules)

        # downsampling modules
        downsamp_modules = []
        for i in range(len(pre_stage_channels) - 1):
            in_channels = head_channels[i] * self.head_expansion
            out_channels = head_channels[i + 1] * self.head_expansion

            downsamp_module = nn.Sequential(
                nn.Conv2d(in_channels=in_channels,
                          out_channels=out_channels,
                          kernel_size=3,
                          stride=2,
                          padding=1),
                nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True)
            )

            downsamp_modules.append(downsamp_module)
        downsamp_modules = nn.ModuleList(downsamp_modules)

        final_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=head_channels[3] * self.head_expansion,
                out_channels=128,
                kernel_size=1,
                stride=1,
                padding=0
            ),
            nn.BatchNorm2d(128, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True)
        )

        return incre_modules, downsamp_modules, final_layer

    def _make_final_layers(self, input_channels, output_channels):
        layers = []
        for _ in range(self.deconv_config.NUM_BASIC_BLOCKS):
            layers.append(nn.Sequential(
                BasicBlock(input_channels, input_channels),
            ))

        layers.append(nn.Sequential(
            nn.Conv2d(
                in_channels=input_channels,
                out_channels=int((input_channels+output_channels)/2),
                kernel_size=3,
                stride=1,
                padding=1),
            nn.BatchNorm2d(int((input_channels+output_channels)/2), momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=int((input_channels+output_channels)/2),
                out_channels=output_channels,
                kernel_size=self.final_kernel,
                stride=1,
                padding=1 if self.final_kernel == 3 else 0)
        ))
        final_layers = [nn.Sequential(*layers)]

        for i in range(self.deconv_config.NUM_DECONVS):
            input_channels = self.deconv_config.NUM_CHANNELS[i]
            final_layers.append(nn.Sequential(
                nn.Conv2d(
                    in_channels=input_channels,
                    out_channels=int((input_channels+output_channels)/2),
                    kernel_size=3,
                    stride=1,
                    padding=1),
                nn.BatchNorm2d(int((input_channels+output_channels)/2), momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    in_channels=int((input_channels+output_channels)/2),
                    out_channels=output_channels,
                    kernel_size=self.final_kernel,
                    stride=1,
                    padding=1 if self.final_kernel == 3 else 0)
            ))

        return nn.ModuleList(final_layers)

    def _make_deconv_layers(self, input_channels, concat_channels):

        deconv_layers = []
        for i in range(self.num_deconvs):
            if self.deconv_config.CAT_OUTPUT[i]:
                input_channels += concat_channels
            output_channels = self.deconv_config.NUM_CHANNELS[i]
            deconv_kernel, padding, output_padding = self._get_deconv_cfg(self.deconv_config.KERNEL_SIZE[i])

            layers = [nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels=input_channels,
                    out_channels=output_channels,
                    kernel_size=deconv_kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=False),
                nn.BatchNorm2d(output_channels, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True)
            )]
            for _ in range(self.deconv_config.NUM_BASIC_BLOCKS):
                layers.append(nn.Sequential(
                    BasicBlock(output_channels, output_channels),
                ))
            deconv_layers.append(nn.Sequential(*layers))
            input_channels = output_channels

        return nn.ModuleList(deconv_layers)

    def _get_deconv_cfg(self, deconv_kernel):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding

    def _make_transition_layer(
            self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(nn.Sequential(
                        nn.Conv2d(num_channels_pre_layer[i],
                                  num_channels_cur_layer[i],
                                  3,
                                  1,
                                  1,
                                  bias=False),
                        nn.BatchNorm2d(num_channels_cur_layer[i]),
                        nn.ReLU(inplace=True)))
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i+1-num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] \
                        if j == i-num_branches_pre else inchannels
                    conv3x3s.append(nn.Sequential(
                        nn.Conv2d(
                            inchannels, outchannels, 3, 2, 1, bias=False),
                        nn.BatchNorm2d(outchannels),
                        nn.ReLU(inplace=True)))
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_head_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * self.head_expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * self.head_expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * self.head_expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample, self.head_expansion))
        inplanes = planes * self.head_expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes, expansion=self.head_expansion))

        return nn.Sequential(*layers)

    def _make_stage(self, layer_config, num_inchannels):
        num_modules = layer_config['NUM_MODULES']
        num_branches = layer_config['NUM_BRANCHES']
        num_blocks = layer_config['NUM_BLOCKS']
        num_channels = layer_config['NUM_CHANNELS']
        block = blocks_dict[layer_config['BLOCK']]
        fuse_method = layer_config['FUSE_METHOD']

        modules = []
        for i in range(num_modules):
            modules.append(
                HighResolutionModule(
                    num_branches,
                    block,
                    num_blocks,
                    num_inchannels,
                    num_channels,
                    fuse_method)
            )
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels

    def forward(self, inputs):
        x = inputs["images"]
        indexes = inputs["indexes"]
        from_mixed_dataset = inputs["from_mixed_dataset"]

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)

        x_list = []
        for i in range(self.stage2_cfg['NUM_BRANCHES']):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)

        x_list = []
        for i in range(self.stage3_cfg['NUM_BRANCHES']):
            if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)

        x_list = []
        for i in range(self.stage4_cfg['NUM_BRANCHES']):
            if self.transition3[i] is not None:
                x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage4(x_list)

        hm = None
        hw = None
        bu = None
        bu_mixed = None
        fusion_bu = None
        pc = None
        pc_mixed = None
        fusion_pc = None
        context = None
        context_mixed = None
        fusion_context = None
        final_fusion = None
        final_fusion_mixed = None

        # Det Head
        if indexes["global"]["det"]:
            hm = []
            hw = []
            x_det = y_list[0][indexes["global"]["det"]]
            y_hm = self.hm_final_layers[0](x_det)
            y_hw = self.hw_final_layers[0](x_det)
            hm.append(y_hm)
            hw.append(y_hw)
            for i in range(self.num_deconvs):
                if self.deconv_config.CAT_OUTPUT[i]:
                    x_hm = torch.cat((x_det, y_hm), 1)
                    x_hw = torch.cat((x_det, y_hw), 1)

                x_hm = self.hm_deconv_layers[i](x_hm)
                x_hw = self.hw_deconv_layers[i](x_hw)
                y_hm = self.hm_final_layers[i + 1](x_hm)
                y_hw = self.hw_final_layers[i + 1](x_hw)
                hm.append(y_hm)
                hw.append(y_hw)

        # BU Head
        if indexes["global"]["bu"]:
            bu_fmd = from_mixed_dataset[indexes["global"]["bu"]]
            temp_bu = []
            x_bu = y_list[0][indexes["global"]["bu"]]
            y_bu = self.bu_final_layers[0](x_bu)
            temp_bu.append(y_bu)
            for i in range(self.num_deconvs):
                if self.deconv_config.CAT_OUTPUT[i]:
                    x_bu = torch.cat((x_bu, y_bu), 1)

                x_bu = self.bu_deconv_layers[i](x_bu)
                y_bu = self.bu_final_layers[i + 1](x_bu)
                temp_bu.append(y_bu)

            if indexes["relative"]["bu"]:
                _bu = [tbu[indexes["relative"]["bu"]] for tbu in temp_bu]
                relative_bu_fmd = bu_fmd[indexes["relative"]["bu"]]
                if (relative_bu_fmd == 0).any():
                    bu = [tbu[relative_bu_fmd == 0] for tbu in _bu]

                if relative_bu_fmd.any():
                    bu_mixed = [self.bu_mixed[i](tbu[relative_bu_fmd == 1]) for i, tbu in enumerate(_bu)]

            if indexes["relative"]["fbu"]:
                fusion_bu = [tbu[indexes["relative"]["fbu"]] for tbu in temp_bu]

        # PC Head
        if indexes["global"]["pc"]:
            pc_fmd = from_mixed_dataset[indexes["global"]["pc"]]
            x_pc = [y[indexes["global"]["pc"]] for y in y_list]
            y_pc = self.pc_incre_modules[0](x_pc[0])

            for i in range(len(self.pc_downsamp_modules)):
                y_pc = self.pc_incre_modules[i + 1](x_pc[i + 1]) + \
                    self.pc_downsamp_modules[i](y_pc)
            y_pc = self.pc_final_layer(y_pc)

            if torch._C._get_tracing_state():
                y_pc = y_pc.flatten(start_dim=2).mean(dim=2)
            else:
                y_pc = F.avg_pool2d(y_pc, kernel_size=y_pc.size()[2:]).view(y_pc.size(0), -1)

            y_pc = self.pc_classifier(y_pc)
            if indexes["relative"]["pc"]:
                _pc = y_pc[indexes["relative"]["pc"]]
                relative_pc_fmd = pc_fmd[indexes["relative"]["pc"]]
                if (relative_pc_fmd == 0).any():
                    pc = _pc[relative_pc_fmd == 0]
                if relative_pc_fmd.any():
                    pc_mixed = _pc[relative_pc_fmd == 1]

            if indexes["relative"]["fpc"]:
                fusion_pc = y_pc[indexes["relative"]["fpc"]]
                relative_fusion_fmd = pc_fmd[indexes["relative"]["fpc"]]

        # Context Head
        if indexes["global"]["context"]:
            context_fmd = from_mixed_dataset[indexes["global"]["context"]]
            x_context = [y[indexes["global"]["context"]] for y in y_list]
            y_context = self.incre_modules[0](x_context[0])

            for i in range(len(self.downsamp_modules)):
                y_context = self.incre_modules[i + 1](x_context[i + 1]) + \
                    self.downsamp_modules[i](y_context)
            y_context = self.final_layer(y_context)

            if torch._C._get_tracing_state():
                y_context = y_context.flatten(start_dim=2).mean(dim=2)
            else:
                y_context = F.avg_pool2d(y_context, kernel_size=y_context.size()[2:]).view(y_context.size(0), -1)

            y_context = self.classifier(y_context)
            if indexes["relative"]["context"]:
                _context = y_context[indexes["relative"]["context"]]
                relative_context_fmd = context_fmd[indexes["relative"]["context"]]
                if (relative_context_fmd == 0).any():
                    context = _context[relative_context_fmd == 0]
                if relative_context_fmd.any():
                    context_mixed = _context[relative_context_fmd == 1]

            if indexes["relative"]["fcontext"]:
                fusion_context = y_context[indexes["relative"]["fcontext"]]

        # Fusion Head
        if fusion_bu is not None:
            assert fusion_pc is not None
            assert fusion_context is not None
            _final_fusion = self.fusion(fusion_bu, fusion_pc, fusion_context, inputs["centers"])
            if (relative_fusion_fmd == 0).any():
                final_fusion = _final_fusion[relative_fusion_fmd == 0]

            if relative_fusion_fmd.any():
                final_fusion_mixed = self.fusion_mixed(_final_fusion[relative_fusion_fmd == 1])

        return {
            "hm": hm,
            "hw": hw,
            "bu": bu,
            "bu_mixed": bu_mixed,
            "pc": pc,
            "pc_mixed": pc_mixed,
            "context": context,
            "context_mixed": context_mixed,
            "fusion": final_fusion,
            "fusion_mixed": final_fusion_mixed
            }

    def init_weights(self, pretrained='', verbose=True):
        logger.info('=> init weights from normal distribution')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)

        parameters_names = set()
        for name, _ in self.named_parameters():
            parameters_names.add(name)

        buffers_names = set()
        for name, _ in self.named_buffers():
            buffers_names.add(name)

        if os.path.isfile(pretrained):
            pretrained_state_dict = torch.load(pretrained)
            logger.info('=> loading pretrained model {}'.format(pretrained))

            need_init_state_dict = {}

            for name, m in pretrained_state_dict.items():
                if name.split('.')[0] in self.pretrained_layers \
                   or self.pretrained_layers[0] == '*':
                    if name in parameters_names or name in buffers_names:
                        if verbose:
                            logger.info(
                                '=> init {} from {}'.format(name, pretrained)
                            )
                        need_init_state_dict[name] = m
            self.load_state_dict(need_init_state_dict, strict=False)


def get_pose_net(cfg, model_cfg, is_train, **kwargs):
    model = BENet(cfg, model_cfg, **kwargs)

    if is_train and model_cfg.INIT_WEIGHTS:
        model.init_weights(model_cfg.PRETRAINED, verbose=cfg.VERBOSE)

    return model
