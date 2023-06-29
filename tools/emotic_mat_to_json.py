import json
from scipy.io import loadmat
import argparse
import numpy as np
import math
import time
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description='Convert Emotic annotations to coco json format')

    parser.add_argument('--emotic',
                        help="path to emotic annotations directory",
                        required=True,
                        type=str)

    parser.add_argument('--out_dir',
                        help="where to save the new annotations",
                        required=True,
                        type=str)
    parser.add_argument('--format',
                        help="bbox format: xywh or x1y1x2y2",
                        default="x1y1x2y2")
    args = parser.parse_args()

    return args


def mat_to_dict(mat, _format):
    """
    Convert the loaded annotations (mat) with scipy.io.loadmat to a dict with the same structure as coco annotations (out)
    """

    out_dict = {'images': [], 'annotations': [], 'categories': []}

    # we add the key 'categories' to reproduce the style of coco and to be able to use this dataset with the cocoapi
    # later
    out_dict['categories'].append({'supercategory': 'person',
                                   'id': 1,
                                   'name': 'person',
                                   'keypoints': ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
                                                 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
                                                 'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee',
                                                 'right_knee', 'left_ankle', 'right_ankle'],
                                   'skeleton': [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13],
                                                [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3],
                                                [2, 4], [3, 5], [4, 6], [5, 7]]
                                   })

    ann_id = 0

    for idx_img, img in enumerate(mat):
        filename = img.filename
        folder = img.folder

        height = img.image_size.n_row
        width = img.image_size.n_col

        name = img.original_database.name

        if name == "mscoco":
            image_id = img.original_database.info.image_id
            annotations_id = img.original_database.info.annotations_id

        out_dict['images'].append({'database': 'EMOTIC',
                                   'file_name': filename,
                                   'folder': folder,
                                   'name': name,
                                   'height': height,
                                   'width': width,
                                   'id': idx_img,
                                   'coco_ids': {'image_id': image_id,
                                                'annotations_id': annotations_id} if name == "mscoco" else []
                                   })

        persons = img.person
        if not type(persons).__module__ == np.__name__:
            persons = np.array([persons])

        for person in persons:
            x, y, x2, y2 = person.body_bbox
            if x < 0:
                x = 0
            if y < 0:
                y = 0
            if x2 > width:
                x2 = width
            if y2 > height:
                y2 = height
            if _format == "xywh":
                w = x2 - x
                h = y2 - y
                bbox = [int(x), int(y), int(w), int(h)]
            elif _format == "x1y1x2y2":
                bbox = [int(x), int(y), int(x2), int(y2)]

            # For test and val sets, there are multiple annotators. Therefore, we only use the categories for which
            # they all agreed (saved in 'combined_categories' instead of 'annotations_categories'). Same thing for
            # continuous annotations ('combined_continuous' instead of 'annotations_continuous').

            if 'combined_categories' in person._fieldnames:
                if isinstance(person.combined_categories, str):
                    annotations_categories = [person.combined_categories]
                else:
                    annotations_categories = [cat for cat in person.combined_categories]
            else:
                if isinstance(person.annotations_categories.categories, str):
                    annotations_categories = [person.annotations_categories.categories]
                else:
                    annotations_categories = [cat for cat in person.annotations_categories.categories]

            if 'combined_continuous' in person._fieldnames:
                valence = person.combined_continuous.valence
                arousal = person.combined_continuous.arousal
                dominance = person.combined_continuous.dominance

            else:
                valence = person.annotations_continuous.valence
                arousal = person.annotations_continuous.arousal
                dominance = person.annotations_continuous.dominance

            # If we don't have values for VAD, don't use None type, it will stop the dataloader. Use str instead.
            annotations_continuous = {"valence": valence if not math.isnan(valence) else 'None',
                                      "arousal": arousal if not math.isnan(valence) else 'None',
                                      "dominance": dominance if not math.isnan(valence) else 'None'}

            gender = person.gender
            age = person.age

            out_dict['annotations'].append({'image_id': idx_img,
                                            'id': ann_id,
                                            'category_id': 1,
                                            'bbox': bbox,
                                            'coco_ids': {'image_id': image_id,
                                                         'annotations_id': annotations_id} if name == "mscoco" else [],
                                            'annotations_categories': annotations_categories,
                                            'annotations_continuous': annotations_continuous,
                                            'gender': gender,
                                            'age': age
                                            })
            ann_id += 1

    return out_dict


def main():
    args = parse_args()
    assert args.format in ["x1y1x2y2", "xywh"], f"Wrong format, please choose either 'x1y1x2y2' or 'xywh'"
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    EMOTIC = loadmat(fr'{args.emotic}/Annotations.mat', struct_as_record=False, squeeze_me=True)
    keys = [key for key in EMOTIC.keys() if '_' not in key]

    for key in keys:
        tic = time.time()
        emotic = EMOTIC[key]
        emotic_dict = mat_to_dict(emotic, args.format)

        with open(f'{args.out_dir}/EMOTIC_{key}_{args.format}.json', 'w') as fp:
            # json.dump(emotic_dict, fp, sort_keys=True, indent=4)  # more human readable on a text editor
            json.dump(emotic_dict, fp)

        print(f'EMOTIC {key.upper()} annotations have been converted to coco json format \n'
              f'    Time Elapsed: {time.time() - tic:03.1f} seconds\n')


if __name__ == '__main__':
    start_time = time.time()
    print('Working...')
    main()
    print(f'Total Time : {time.time() - start_time:03.1f} seconds')
