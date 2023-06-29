import numpy as np
import argparse
from matplotlib import pyplot as plt
import os
import torch
import torchvision
from pathlib import Path
from PIL import Image

import _init_paths

from models.BENet import get_pose_net
from config import cfg_mt as cfg
from config import update_config_mt as update_config
from utils.transforms import resize_align_multi_scale
from utils.transforms import get_multi_scale_size
from core.inference import get_outputs
from utils.utils import EmotionConverter


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize pre_processed heatmaps '
                                                 'and tag maps from the chosen dataset')

    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    parser.add_argument('--model',
                        help='Path to the pre-trained emo_model',
                        default='',
                        type=str)

    parser.add_argument('--img',
                        help='Path to the image',
                        required=True,
                        type=str)

    parser.add_argument('--emo',
                        nargs='+',
                        help='A list containing the emotions we want to see',
                        required=True)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    return args


def show(title, img, ax):
    ax.cla()
    if len(img.shape) == 3:
        ax.imshow(img, vmin=0, vmax=1)
    else:
        ax.imshow(img, cmap='gray', vmin=0, vmax=1)
    ax.set_title(title)
    ax.set_axis_off()


def main():
    args = parse_args()
    emos = args.emo
    update_config(cfg, args)
    converter = EmotionConverter(cfg.DATASET.TEST)
    model = get_pose_net(cfg, cfg.MODEL, is_train=False)
    if len(args.model) > 0:
        try:
            model.load_state_dict(torch.load(args.model), strict=True)
        except:
            temp = torch.load(args.model)
            new_dict = dict()
            for key in temp:
                new_dict[f"{key.replace('module.', '')}"] = temp[key]
            model.load_state_dict(new_dict, strict=True)

    else:
        root_output_dir = Path(cfg.OUTPUT_DIR)
        dataset = cfg.DATASET.DATASET
        dataset = dataset.replace(':', '_')
        model_name = cfg.MODEL.NAME
        cfg_name = os.path.basename(args.cfg).split('.')[0]

        final_output_dir = root_output_dir / dataset / model_name / cfg_name
        state_file = os.path.join(final_output_dir, 'model_best.pth.tar')
        model.load_state_dict(torch.load(state_file), strict=True)

    model = model.cuda()
    temp_img = Image.open(args.img).convert('RGB')
    img = np.array(temp_img)

    imagenet_means = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=imagenet_means,
                std=imagenet_std
            )
        ]
    )

    nb_plot = len(emos) + 2
    nb_col = 4
    nb_rows = (nb_plot - 1) // nb_col + 1

    fig, axes = plt.subplots(nb_rows, nb_col, gridspec_kw={'width_ratios': [1] * nb_col,
                                                           'height_ratios': [1] * nb_rows})

    indexes = {"relative": {"bu": [0], "fbu": [],
                            "pc": [], "fpc": [],
                            "context": [], "fcontext": []},
               "global": {"det": [0],
                          "bu": [0],
                          "pc": [],
                          "context": []}
               }
    cat2ind = converter.cat_to_ind
    input_size = cfg.DATASET.INPUT_SIZE
    base_size, img_center, scale = get_multi_scale_size(img, input_size, 1.0, 1.0, [])

    with torch.no_grad():
        resized, center, scale, trans = resize_align_multi_scale(img, input_size, 1.0, 1.0, [])
        resized = transforms(resized).unsqueeze(0).cuda()

        hm, hw, bu_cat, bu_cont, pc_cat, pc_cont, context_cat, context_cont, fusion_cat, fusion_cont = get_outputs(
            cfg, cfg.MODEL, model, {"images": resized, "indexes": indexes, "centers": [],
                                    "from_mixed_dataset": torch.tensor([0, 0, 0]).cuda()},
            cfg.TEST.FLIP_TEST,
            cfg.TEST.PROJECT2IMAGE,
            base_size,
            mt=False
        )

    det_heatmap = hm[0, 0].cpu().numpy()
    displayed_img = np.array(temp_img.resize((det_heatmap.shape[1], det_heatmap.shape[0])))
    r = c = 0
    show("Raw image", displayed_img, axes[r, c])
    c += 1
    show("Detection heatmap", det_heatmap, axes[r, c])
    c += 1
    for emo in emos:
        pos = cat2ind["EMOTIC"][emo]
        out_map = bu_cat[0, pos].cpu().numpy()
        show(f"{emo}, min={out_map.min():0.3f} max={out_map.max():0.3f}", out_map, axes[r, c])
        c += 1
        if c == nb_col:
            r += 1
            c = 0

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
