import json
import argparse
import time
import numpy as np
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description='Create emotic person centric annotations')

    parser.add_argument('--emotic',
                        help="absolute path to emotic NEW annotations directory (after emotic_mat_to_json.py)",
                        required=True,
                        type=str)

    parser.add_argument('--out_dir',
                        help="where to save the merged annotations",
                        required=True,
                        type=str)

    parser.add_argument('--x1y1x2y2',
                        help="Specify if we use the x1y1x2y2 format rather than xywh",
                        required=True,
                        type=bool)

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    datasets = ["train", "val", "test"]

    for dataset in datasets:
        tic = time.time()
        if args.x1y1x2y2:
            dataset += '_x1y1x2y2'
        # Load emotic annotations
        emotic = json.load(open(fr'{args.emotic}/EMOTIC_{dataset}.json', 'r'))
        top_down_dataset = {
            "images": [],
            "annotations": [],
            "categories": emotic["categories"]
        }
        img_ids = []
        ann_ids = []
        for ann in emotic["annotations"]:
            img_ids.append(ann["image_id"])
            ann_ids.append(ann["id"])

        img_ids = np.array(img_ids)
        ann_ids = np.array(ann_ids)

        global_id = 0
        for ann in emotic["annotations"]:
            img_id = ann["image_id"]
            _id = ann["id"]
            indexes = np.where((img_ids == img_id) == (ann_ids != _id))[0]
            bbox = [ann["bbox"]]
            cat_ann = [ann["annotations_categories"]]
            cont_ann = [ann["annotations_continuous"]]
            for ind in indexes:
                bbox.append(emotic["annotations"][ind]["bbox"])
                cat_ann.append(emotic["annotations"][ind]["annotations_categories"])
                cont_ann.append(emotic["annotations"][ind]["annotations_continuous"])
            img = emotic["images"][img_id]
            top_down_dataset["images"].append({"database": "EMOTIC",
                                               "file_name": img["file_name"],
                                               "folder": img["folder"],
                                               "name": img["name"],
                                               "height": img["height"],
                                               "width": img["width"],
                                               "id": global_id})
            top_down_dataset["annotations"].append({"image_id": global_id,
                                                    "id": global_id,
                                                    "category_id": 1,
                                                    "bbox": bbox,
                                                    "annotations_categories": cat_ann,
                                                    "annotations_continuous": cont_ann})
            global_id += 1

        # Save the new dataset
        with open(f'{args.out_dir}/EMOTIC_{dataset}_tdpc.json', 'w') as fp:
            # json.dump(coco, fp, sort_keys=True, indent=4)  # more human readable on a text editor
            json.dump(top_down_dataset, fp)

        print(f'{dataset.upper()} annotations done\n'
              f'    Time Elapsed: {time.time() - tic:03.1f} seconds\n')


if __name__ == '__main__':
    start_time = time.time()
    print('Working...')
    main()
    print(f'Total Time : {time.time() - start_time:03.1f} seconds')
