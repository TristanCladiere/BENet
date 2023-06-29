import pandas as pd
import numpy as np
import argparse
import json
import time
from pathlib import Path
from PIL.Image import open as imread


def parse_args():
    parser = argparse.ArgumentParser(description='Choose file to proceed and where to save the results')

    parser.add_argument('--heco',
                        help='Path to HECO directory',
                        required=True,
                        type=str)
    parser.add_argument('--out',
                        help='Where to save the results',
                        required=True,
                        type=str)
    parser.add_argument('--format',
                        help="bbox format: xywh or x1y1x2y2",
                        default="x1y1x2y2")

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(fr"{args.heco}/Labels/HECO_Labels.csv", delimiter=",", index_col=False).to_dict(orient='list')
    dataset = {
        "images": [],
        "annotations": [],
        "categories": [{'supercategory': 'person',
                        'id': 1,
                        'name': 'person',
                        'keypoints': ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
                                      'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
                                      'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee',
                                      'right_knee', 'left_ankle', 'right_ankle'],
                        'skeleton': [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13],
                                     [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3],
                                     [2, 4], [3, 5], [4, 6], [5, 7]]
                        }]
    }

    all_imgs = np.array(data["Image"])
    ids = np.arange(len(all_imgs))
    _format = args.format
    for i in ids:
        i = int(i)
        img_name = data["Image"][i]
        img = imread(f"{args.heco}/Images/{img_name}")
        dataset["images"].append({"database": "HECO",
                                  "file_name": img_name,
                                  "folder": "HECO/Images",
                                  "height": img.height,
                                  "width": img.width,
                                  "id": i})
        x1, y1, x2, y2 = int(data["xmin"][i]), int(data["ymin"][i]), int(data["xmax"][i]), int(data["ymax"][i])
        if _format == "xywh":
            w = x2 - x1
            h = y2 - y1
            temp = [int(x1), int(y1), int(w), int(h)]
        elif _format == "x1y1x2y2":
            temp = [int(x1), int(y1), int(x2), int(y2)]
        bbox = [temp]
        cat_ann = [[data["Category"][i]]]
        cont_ann = [{"valence": data["Valence"][i] / 10,
                     "arousal": data["Arousal"][i] / 10,
                     "dominance": data["Dominance"][i] / 10}]

        indexes_other_anns = np.where((img_name == all_imgs) == (i != ids))[0]
        for ind in indexes_other_anns:
            x1, y1, x2, y2 = int(data["xmin"][i]), int(data["ymin"][i]), int(data["xmax"][i]), int(data["ymax"][i])
            if _format == "xywh":
                w = x2 - x1
                h = y2 - y1
                temp = [int(x1), int(y1), int(w), int(h)]
            elif _format == "x1y1x2y2":
                temp = [int(x1), int(y1), int(x2), int(y2)]
            bbox.append(temp)

            cat_ann.append([data["Category"][ind]])
            cont_ann.append({"valence": data["Valence"][ind] / 10,
                             "arousal": data["Arousal"][ind] / 10,
                             "dominance": data["Dominance"][ind] / 10})

        dataset["annotations"].append({"image_id": i,
                                       "id": i,
                                       "category_id": 1,
                                       "bbox": bbox,
                                       "annotations_categories": cat_ann,
                                       "annotations_continuous": cont_ann})

    # Save the new dataset
    with open(f'{args.out}/HECO_{args.format}_tdpc.json', 'w') as fp:
        # json.dump(coco, fp, sort_keys=True, indent=4)  # more human readable on a text editor
        json.dump(dataset, fp)


if __name__ == '__main__':
    start_time = time.time()
    print('Working...')
    main()
    print(f'Total Time : {time.time() - start_time:03.1f} seconds')
