import argparse
import os
import torch
import torchvision
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from pathlib import Path

import _init_paths

from models.BENet import get_pose_net
from config import cfg_mt as cfg
from config import update_config_mt as update_config
from utils.transforms import get_final_preds, resize_align_multi_scale, get_multi_scale_size
from core.inference import get_outputs, Decoder
from utils.utils import EmotionConverter




def parse_args():
    parser = argparse.ArgumentParser(description="Inference on an image")

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

    parser.add_argument('--emo_thr',
                        help='threshold to decide whether an emotion is present or not (in [0, 1])',
                        type=float,
                        default=0.5)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    update_config(cfg, args)
    converter = EmotionConverter(cfg.DATASET.TEST)
    decoder = Decoder(cfg)
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
    indexes = {"relative": {"bu": [0], "fbu": [],
                            "pc": [], "fpc": [],
                            "context": [], "fcontext": []},
               "global": {"det": [0],
                          "bu": [0],
                          "pc": [],
                          "context": []}
               }

    # size at scale 1.0
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

        ans, scores = decoder.decode(hm, hw, bu_cat, bu_cont, cfg.TEST.ADJUST)
        final_results = get_final_preds(ans, center, scale, [hm.size(3), hm.size(2)])

    fig, ax = plt.subplots()
    ax.imshow(img, vmin=0, vmax=1)
    for pred in final_results:
        x1, y1, x2, y2 = pred[0, :4]
        w = x2 - x1
        h = y2 - y1
        ax.add_patch(Rectangle((x1, y1), w, h, linewidth=2, edgecolor='r', facecolor='none'))

        emotions = pred[0, 5:5+26]
        mask = np.where(emotions > args.emo_thr)[0]

        emo_txt = f"{[converter.ind_to_cat['EMOTIC'][ind.item()] for ind in mask]}"
        ax.text(x1, y1-5, f"Det_score: {pred[0, 4]:4.2}\nEmos: {emo_txt}", color='r', fontsize='small')
    ax.set_axis_off()
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
