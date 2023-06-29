from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import torch
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms
import torch.multiprocessing
import numpy as np
from tqdm import tqdm

import _init_paths

from models.BENet import get_pose_net
from utils.utils import EmotionConverter
from config import cfg_mt, update_config_mt
from core.inference import get_outputs, Decoder
from utils.transforms import resize_align_multi_scale, get_final_preds, get_multi_scale_size, get_affine_transform, \
    affine_transform
from dataset import make_test_dataloader


torch.multiprocessing.set_sharing_strategy('file_system')


def parse_args():
    parser = argparse.ArgumentParser(description='Test keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    parser.add_argument('--state',
                        help='which state file to use (final or best)',
                        required=True,
                        type=str)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    return args


def get_bu_centers(cfg, anno, resized, trans):
    centers = []
    for i, res in enumerate(cfg.DATASET.OUTPUT_SIZE):
        centers.append([])
        factor = cfg.DATASET.INPUT_SIZE / res
        height, width = resized.shape[:2]
        base_size = (int(width/factor), int(height/factor))
        center = np.array([int(width / 2.0 + 0.5), int(height / 2.0 + 0.5)])
        if width < height:
            scale_width = width / 200
            scale_height = base_size[1] / base_size[0] * width / 200
        else:
            scale_height = height / 200
            scale_width = base_size[0] / base_size[1] * height / 200
        scale = np.array([scale_width, scale_height])

        trans2 = get_affine_transform(center, scale, 0, base_size)

        for j, obj in enumerate(anno):
            x1, y1, x2, y2 = obj['bbox']
            xc, yc = (x1+x2)/2, (y1+y2)/2
            xc, yc = affine_transform([xc, yc], trans)
            xc, yc = affine_transform([xc, yc], trans2)
            centers[i].append((int(xc), int(yc)))

    return [centers]


def main():
    args = parse_args()
    update_config_mt(cfg_mt, args)

    dataset_name = cfg_mt.DATASET.DATASET
    dataset_name = dataset_name.replace(':', '_')
    model_name = cfg_mt.MODEL.NAME
    cfg_name = os.path.basename(args.cfg).split('.')[0]
    final_output_dir = f"{cfg_mt.OUTPUT_DIR}/{dataset_name}/{model_name}/{cfg_name}"

    # cudnn related setting
    cudnn.benchmark = cfg_mt.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg_mt.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg_mt.CUDNN.ENABLED

    model = get_pose_net(cfg_mt, cfg_mt.MODEL, is_train=False)
    if args.state == "final":
        model_state_file = os.path.join(final_output_dir, 'final_state0.pth.tar')
    elif args.state == "best":
        model_state_file = os.path.join(final_output_dir, 'model_best.pth.tar')
    model.load_state_dict(torch.load(model_state_file))
    model.eval().cuda()

    data_loader, test_dataset = make_test_dataloader(cfg_mt)
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

    decoder = Decoder(cfg_mt)
    converter = EmotionConverter(cfg_mt.DATASET.TEST)

    # BU_preds with detection
    all_preds = []
    all_scores = []

    indexes = {"relative": {"bu": [0], "fbu": [0],
                            "pc": [0], "fpc": [0],
                            "context": [0], "fcontext": [0]},
               "global": {"det": [0],
                          "bu": [0],
                          "pc": [1],
                          "context": [2]}
               }

    results_without_det = {
        "bu": {"cat": np.array([]), "cont": np.array([])},
        "pc": {"cat": np.array([]), "cont": np.array([])},
        "context": {"cat": np.array([]), "cont": np.array([])},
        "context_rep": {"cat": np.array([]), "cont": np.array([])},
        "fusion": {"cat": np.array([]), "cont": np.array([])},
    }
    pbar = tqdm(total=len(test_dataset))
    for i, (images, annos, all_anns, path, database) in enumerate(data_loader):
        if i == 0:
            new_img = True
            saved_path = path
        else:
            if path == saved_path:
                new_img = False
            else:
                new_img = True
                saved_path = path

        images = [img[0].cpu().numpy() for img in images]

        # size at scale 1.0
        base_size, center, scale = get_multi_scale_size(images[0], cfg_mt.DATASET.INPUT_SIZE, 1.0, 1.0, [])

        with torch.no_grad():
            resized_images = []
            input_size = cfg_mt.DATASET.INPUT_SIZE

            for j, img in enumerate(images):
                saved_size = base_size if j == 1 else []
                resized, center, scale, trans = resize_align_multi_scale(img, input_size, 1, 1, saved_size)
                if j == 0:
                    centers = get_bu_centers(cfg_mt, all_anns, resized, trans)
                resized_images.append(transforms(resized))

            batch = {"images": torch.stack(resized_images).cuda(), "indexes": indexes,
                     "centers": centers, "from_mixed_dataset": torch.tensor([0, 0, 0]).cuda()}
            hm, hw, bu_cat, bu_cont, pc_cat, pc_cont, context_cat, context_cont, fusion_cat, fusion_cont = get_outputs(
                cfg_mt, cfg_mt.MODEL, model, batch, cfg_mt.TEST.FLIP_TEST, cfg_mt.TEST.PROJECT2IMAGE, base_size)
            if new_img:
                ans, scores = decoder.decode(hm, hw, bu_cat, bu_cont, cfg_mt.TEST.ADJUST)
                final_results = get_final_preds(ans, center, scale, [hm.size(3), hm.size(2)])
                all_preds.append(final_results)
                all_scores.append(scores)

                if context_cat is not None:
                    glob_cat_gt = converter.get_cat_emotions(all_anns, database[0]).max(axis=0)
                    temp_context_cat_concat = np.concatenate((glob_cat_gt, context_cat.cpu().numpy()))
                    results_without_det["context"]["cat"] = np.vstack((results_without_det["context"]["cat"],
                                                                       temp_context_cat_concat)) \
                        if results_without_det["context"]["cat"].size else temp_context_cat_concat

                if context_cont is not None:
                    glob_cont_gt = converter.get_cont_emotions(all_anns).max(axis=0)
                    temp_context_cont_concat = np.concatenate((glob_cont_gt, context_cont.cpu().numpy()))
                    results_without_det["context"]["cont"] = np.vstack((results_without_det["context"]["cont"],
                                                                        temp_context_cont_concat)) \
                        if results_without_det["context"]["cont"].size else temp_context_cont_concat

            if bu_cat is not None:
                cat_gt = converter.get_cat_emotions([all_anns[0]], database[0])[0]
                glob_cat_gt = converter.get_cat_emotions(all_anns, database[0]).max(axis=0)

                # BU results
                x1, y1, x2, y2 = all_anns[0]['bbox']
                x_center = int((x1 + x2) / 2)
                y_center = int((y1 + y2) / 2)
                trans = get_affine_transform(center, scale, 0, base_size, inv=0)
                x_center_resized, y_center_resized = affine_transform([x_center, y_center], trans)

                temp_bu_cat = bu_cat[0, :, int(y_center_resized), int(x_center_resized)].cpu().numpy()
                temp_bu_cat_concat = np.concatenate((cat_gt, temp_bu_cat))
                results_without_det["bu"]["cat"] = np.vstack((results_without_det["bu"]["cat"], temp_bu_cat_concat)) \
                    if results_without_det["bu"]["cat"].size else temp_bu_cat_concat

                # PC results
                temp_pc_cat_concat = np.concatenate((cat_gt, pc_cat.cpu().numpy()))
                results_without_det["pc"]["cat"] = np.vstack((results_without_det["pc"]["cat"], temp_pc_cat_concat)) \
                    if results_without_det["pc"]["cat"].size else temp_pc_cat_concat

                # Context results (repeated)
                temp_context_cat_concat = np.concatenate((glob_cat_gt, context_cat.cpu().numpy()))
                results_without_det["context_rep"]["cat"] = np.vstack((results_without_det["context_rep"]["cat"],
                                                                       temp_context_cat_concat)) \
                    if results_without_det["context_rep"]["cat"].size else temp_context_cat_concat

                # Fusion results
                temp_fusion_cat_concat = np.concatenate((cat_gt, fusion_cat.cpu().numpy()))
                results_without_det["fusion"]["cat"] = np.vstack((results_without_det["fusion"]["cat"],
                                                                  temp_fusion_cat_concat)) \
                    if results_without_det["fusion"]["cat"].size else temp_fusion_cat_concat

            if bu_cont is not None:
                cont_gt = converter.get_cont_emotions([all_anns[0]])[0]
                glob_cont_gt = converter.get_cont_emotions(all_anns).max(axis=0)

                # BU results
                x1, y1, x2, y2 = all_anns[0]['bbox']
                x_center = int((x1 + x2) / 2)
                y_center = int((y1 + y2) / 2)
                trans = get_affine_transform(center, scale, 0, base_size, inv=0)
                x_center_resized, y_center_resized = affine_transform([x_center, y_center], trans)

                temp_bu_cont = bu_cont[0, :, int(y_center_resized), int(x_center_resized)].cpu().numpy()
                temp_bu_cont_concat = np.concatenate((cont_gt, temp_bu_cont))
                results_without_det["bu"]["cont"] = np.vstack((results_without_det["bu"]["cont"], temp_bu_cont_concat)) \
                    if results_without_det["bu"]["cont"].size else temp_bu_cont_concat

                # PC results
                temp_pc_cont_concat = np.concatenate((cont_gt, pc_cont.cpu().numpy()))
                results_without_det["pc"]["cont"] = np.vstack((results_without_det["pc"]["cont"], temp_pc_cont_concat)) \
                    if results_without_det["pc"]["cont"].size else temp_pc_cont_concat

                # Context results (repeated)
                temp_context_cont_concat = np.concatenate((glob_cont_gt, context_cont.cpu().numpy()))
                results_without_det["context_rep"]["cont"] = np.vstack((results_without_det["context_rep"]["cont"],
                                                                       temp_context_cont_concat)) \
                    if results_without_det["context_rep"]["cont"].size else temp_context_cont_concat

                # Fusion results
                temp_fusion_cont_concat = np.concatenate((cont_gt, fusion_cont.cpu().numpy()))
                results_without_det["fusion"]["cont"] = np.vstack((results_without_det["fusion"]["cont"],
                                                                  temp_fusion_cont_concat)) \
                    if results_without_det["fusion"]["cont"].size else temp_fusion_cont_concat

        pbar.update()
    pbar.close()
    info_str, results_emo = test_dataset.evaluate(
        cfg_mt, all_preds, results_without_det, all_scores, final_output_dir, temp=False, state=args.state
    )

    message = ""
    if info_str is not None:
        message += f"<{'DETECTION'}>"
        for key, val in info_str.items():
            if val < 0:
                message += f"\n| {key:<6} | {val:05.2f} |"
            else:
                message += f"\n| {key:<6} | {val*100:05.2f} |"
        message += f"\n\n\n"

    if results_emo is not None:
        ind2thr = results_emo.pop("ind2thr")
        thr = list(ind2thr.values())
        ind2cat = results_emo.pop("ind2cat")
        cat = list(ind2cat["EMOTIC"].values())
        det_recall = results_emo.pop("det_recall")
        for key, val in results_emo.items():
            if val is not None:
                if len(val.shape) == 2:
                    val = val.transpose()
                    message += f"<{key.upper()}>\n| {'Thresholds':<15} |"
                    for t in thr:
                        message += f" {t} |"
                    message += f"\n"
                    for i, line in enumerate(val):
                        message += f"\n| {cat[i]:<15} |"
                        for score in line:
                            message += f" {score:05.2f} |"
                    mean_per_cat = val.mean(0)
                    message += f"\n\n| {'mAP':<15} |"
                    for mpc in mean_per_cat:
                        message += f" {mpc:05.2f} |"
                    message += f"\n| {'Global mAP':<15} | {mean_per_cat.mean():05.2f} |"
                    if key == "bu_cat_all_dets":
                        message += f"\n| {'Det Recall':<15} |"
                        for rec in det_recall:
                            message += f" {rec:05.2f} |"
                        message += f"\n| {'Mean Det Rec':<15} | {sum(det_recall)/len(det_recall):05.2f} |"
                else:
                    message += f"\n\n<{key.upper()}>\n"
                    for i, score in enumerate(val):
                        message += f"| {cat[i]:<15} | {score:05.2f} |\n"
                    message += f"\n| {'mAP':<15} | {val.mean():05.2f} |"
                message += f"\n\n\n"

    with open(f"{final_output_dir}/{args.state}_state_scores.txt", "w") as text_file:
        print(message, file=text_file)


if __name__ == '__main__':
    main()

