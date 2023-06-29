from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.utils.data

from .MixedCenterEmotion import MixedCenterEmotion as mixed_center_emotion
from .MixedDataset import MixedDataset as mixed
from .transforms import build_transforms
from .target_generators import BBGenerator
from .target_generators import EmotionsGenerator
from .transforms import ExtractSubject, MaskAllSubjects, RandomMaskSubject, Fusion, RandomChoice


def build_dataset(cfg, dataset_type, pretransforms):

    pre_trans_dict = {
        "ExtractSubject": ExtractSubject(),
        "MaskAllSubjects": MaskAllSubjects(),
        "RandomMaskSubject": RandomMaskSubject(),
        "Fusion": Fusion(),
        "Det": "det",
    }
    if dataset_type == 'train':
        is_train = True
        dataset_name = cfg.DATASET.TRAIN
    elif dataset_type == 'val':
        is_train = True
        dataset_name = cfg.DATASET.VAL
    else:
        is_train = False
        dataset_name = cfg.DATASET.TEST

    transforms = build_transforms(cfg, is_train)

    _trans = [pre_trans_dict[name] for name in pretransforms[0]]
    pre_transforms = RandomChoice(_trans, pretransforms[1])

    _BBGenerator = BBGenerator

    bb_generator = [
        _BBGenerator(
            output_size, cfg.DATASET.MAX_NUM_PEOPLE, cfg.DATASET.SIGMA
        ) for output_size in cfg.DATASET.OUTPUT_SIZE
    ]

    bu_cat_generator = [
        EmotionsGenerator(
            cfg.DATASET.MAX_NUM_PEOPLE,
            cfg.DATASET.NUM_CAT_EMOTIONS,
            output_size
        ) for output_size in cfg.DATASET.OUTPUT_SIZE
    ]

    bu_cont_generator = [
        EmotionsGenerator(
            cfg.DATASET.MAX_NUM_PEOPLE,
            cfg.DATASET.NUM_CONT_EMOTIONS,
            output_size
        ) for output_size in cfg.DATASET.OUTPUT_SIZE
    ]

    if cfg.DATASET.NUM_CAT_MIXED:
        bu_cat_generator += [
            EmotionsGenerator(cfg.DATASET.MAX_NUM_PEOPLE,
                              cfg.DATASET.NUM_CAT_MIXED,
                              output_size)
            for output_size in cfg.DATASET.OUTPUT_SIZE
        ]

    dataset = eval(cfg.DATASET.DATASET)(
        cfg,
        dataset_name,
        is_train,
        bb_generator,
        bu_cat_generator,
        bu_cont_generator,
        transforms,
        pre_transforms
    )

    return dataset


def collate_func(batch):
    custom_batch = {
        "inputs": {"images": [], "from_mixed_dataset": []},
        "hm": [[], []],
        "hw": [[], []],
        "bu": {"cat_emo": {"main": [[], []], "mixed": [[], []]},
               "cont_emo": {"main": [[], []], "mixed": [[], []]}},
        "pc": {"cat_emo": {"main": [], "mixed": []},
               "cont_emo": {"main": [], "mixed": []}},
        "context": {"cat_emo": {"main": [], "mixed": []},
                    "cont_emo": {"main": [], "mixed": []}},
        "fusion": {"cat_emo": {"main": [], "mixed": []},
                   "cont_emo": {"main": [], "mixed": []}}
        }
    ind_glob = 0
    ind_relative_bu = 0
    ind_relative_pc = 0
    ind_relative_context = 0
    centers = []
    indexes = {"relative": {"bu": [], "fbu": [],
                            "pc": [], "fpc": [],
                            "context": [], "fcontext": []},
               "global": {"det": [],
                          "bu": [],
                          "pc": [],
                          "context": []}
               }

    for images, bb_list, cat_list, cont_list, from_mixed_dataset in batch:
        img_type = images["head"]
        img = images["image"]

        if from_mixed_dataset:
            key = "mixed"
        else:
            key = "main"

        if img_type == "fusion":
            custom_batch["inputs"]["from_mixed_dataset"] += [from_mixed_dataset] * 3
            centers.append(images["centers"])
            custom_batch["inputs"]["images"].append(img[0])
            indexes["global"]["bu"].append(ind_glob)
            ind_glob += 1
            indexes["relative"]["fbu"].append(ind_relative_bu)
            ind_relative_bu += 1
            custom_batch["inputs"]["images"].append(img[1])
            indexes["global"]["pc"].append(ind_glob)
            ind_glob += 1
            indexes["relative"]["fpc"].append(ind_relative_pc)
            ind_relative_pc += 1
            custom_batch["inputs"]["images"].append(img[2])
            indexes["global"]["context"].append(ind_glob)
            ind_glob += 1
            indexes["relative"]["fcontext"].append(ind_relative_context)
            ind_relative_context += 1
        else:
            custom_batch["inputs"]["from_mixed_dataset"].append(from_mixed_dataset)
            custom_batch["inputs"]["images"].append(img[0])
            indexes["global"][img_type].append(ind_glob)
            ind_glob += 1
            if img_type == "bu":
                indexes["relative"]["bu"].append(ind_relative_bu)
                ind_relative_bu += 1
            if img_type == "pc":
                indexes["relative"]["pc"].append(ind_relative_pc)
                ind_relative_pc += 1
            if img_type == "context":
                indexes["relative"]["context"].append(ind_relative_context)
                ind_relative_context += 1

        if img_type == "det":
            custom_batch["hm"][0].append(torch.from_numpy(bb_list["hm"][0]))
            custom_batch["hm"][1].append(torch.from_numpy(bb_list["hm"][1]))
            custom_batch["hw"][0].append(torch.from_numpy(bb_list["hw"][0]))
            custom_batch["hw"][1].append(torch.from_numpy(bb_list["hw"][1]))

        elif img_type == "bu":
            custom_batch["bu"]["cat_emo"][key][0].append(torch.from_numpy(cat_list[0]))
            custom_batch["bu"]["cat_emo"][key][1].append(torch.from_numpy(cat_list[1]))
            custom_batch["bu"]["cont_emo"][key][0].append(torch.from_numpy(cont_list[0]))
            custom_batch["bu"]["cont_emo"][key][1].append(torch.from_numpy(cont_list[1]))

        else:
            custom_batch[img_type]["cat_emo"][key].append(torch.from_numpy(cat_list))
            custom_batch[img_type]["cont_emo"][key].append(torch.from_numpy(cont_list))

    custom_batch["inputs"]["images"] = torch.stack(custom_batch["inputs"]["images"])
    custom_batch["inputs"]["from_mixed_dataset"] = torch.tensor(custom_batch["inputs"]["from_mixed_dataset"])
    custom_batch["hm"] = [torch.stack(res) for res in custom_batch["hm"]] if custom_batch["hm"][0] else None
    custom_batch["hw"] = [torch.stack(res) for res in custom_batch["hw"]] if custom_batch["hw"][0] else None
    custom_batch["bu"]["cat_emo"]["main"] = [torch.stack(res) for res in custom_batch["bu"]["cat_emo"]["main"]]\
        if custom_batch["bu"]["cat_emo"]["main"][0] else None
    custom_batch["bu"]["cat_emo"]["mixed"] = [torch.stack(res) for res in custom_batch["bu"]["cat_emo"]["mixed"]]\
        if custom_batch["bu"]["cat_emo"]["mixed"][0] else None
    custom_batch["bu"]["cont_emo"]["main"] = [torch.stack(res) for res in custom_batch["bu"]["cont_emo"]["main"]] \
        if custom_batch["bu"]["cont_emo"]["main"][0] else None
    custom_batch["bu"]["cont_emo"]["mixed"] = [torch.stack(res) for res in custom_batch["bu"]["cont_emo"]["mixed"]] \
        if custom_batch["bu"]["cont_emo"]["mixed"][0] else None
    custom_batch["pc"]["cat_emo"]["main"] = torch.stack(custom_batch["pc"]["cat_emo"]["main"]) \
        if custom_batch["pc"]["cat_emo"]["main"] else None
    custom_batch["pc"]["cat_emo"]["mixed"] = torch.stack(custom_batch["pc"]["cat_emo"]["mixed"]) \
        if custom_batch["pc"]["cat_emo"]["mixed"] else None
    custom_batch["pc"]["cont_emo"]["main"] = torch.stack(custom_batch["pc"]["cont_emo"]["main"]) \
        if custom_batch["pc"]["cont_emo"]["main"] else None
    custom_batch["pc"]["cont_emo"]["mixed"] = torch.stack(custom_batch["pc"]["cont_emo"]["mixed"]) \
        if custom_batch["pc"]["cont_emo"]["mixed"] else None
    custom_batch["context"]["cat_emo"]["main"] = torch.stack(custom_batch["context"]["cat_emo"]["main"]) \
        if custom_batch["context"]["cat_emo"]["main"] else None
    custom_batch["context"]["cat_emo"]["mixed"] = torch.stack(custom_batch["context"]["cat_emo"]["mixed"]) \
        if custom_batch["context"]["cat_emo"]["mixed"] else None
    custom_batch["context"]["cont_emo"]["main"] = torch.stack(custom_batch["context"]["cont_emo"]["main"]) \
        if custom_batch["context"]["cont_emo"]["main"] else None
    custom_batch["context"]["cont_emo"]["mixed"] = torch.stack(custom_batch["context"]["cont_emo"]["mixed"]) \
        if custom_batch["context"]["cont_emo"]["mixed"] else None
    custom_batch["fusion"]["cat_emo"]["main"] = torch.stack(custom_batch["fusion"]["cat_emo"]["main"]) \
        if custom_batch["fusion"]["cat_emo"]["main"] else None
    custom_batch["fusion"]["cat_emo"]["mixed"] = torch.stack(custom_batch["fusion"]["cat_emo"]["mixed"]) \
        if custom_batch["fusion"]["cat_emo"]["mixed"] else None
    custom_batch["fusion"]["cont_emo"]["main"] = torch.stack(custom_batch["fusion"]["cont_emo"]["main"]) \
        if custom_batch["fusion"]["cont_emo"]["main"] else None
    custom_batch["fusion"]["cont_emo"]["mixed"] = torch.stack(custom_batch["fusion"]["cont_emo"]["mixed"]) \
        if custom_batch["fusion"]["cont_emo"]["mixed"] else None

    custom_batch["inputs"]["centers"] = centers
    custom_batch["inputs"]["indexes"] = indexes

    return custom_batch


def make_dataloader(cfg, dataset_type, distributed=False, pre_transforms=[]):

    if dataset_type == 'train' or dataset_type == 'val':
        is_train = True
    else:
        is_train = False

    if is_train:
        images_per_gpu = cfg.TRAIN.IMAGES_PER_GPU
        shuffle = True
    else:
        images_per_gpu = cfg.TRAIN.IMAGES_PER_GPU
        shuffle = False
    images_per_batch = images_per_gpu * len(cfg.GPUS)

    dataset = build_dataset(cfg, dataset_type, pre_transforms)

    if is_train and distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset
        )
        shuffle = False
    else:
        train_sampler = None

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=images_per_batch,
        shuffle=shuffle,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY,
        sampler=train_sampler,
        collate_fn=collate_func
    )

    return data_loader


def make_test_dataloader(cfg):

    dataset = eval(cfg.DATASET.DATASET_TEST)(
        cfg.DATASET.ROOT,
        cfg.DATASET.TEST,
        transform=Fusion()
    )

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    return data_loader, dataset
