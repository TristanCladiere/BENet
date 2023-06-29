import json
import argparse
import time
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description='Merge EMOTIC and HECO annotations')

    parser.add_argument('--indir',
                        help="new_annotations dir (after creating json files)",
                        required=True,
                        type=str)

    parser.add_argument('--outdir',
                        help="out dir",
                        required=True,
                        type=str)

    parser.add_argument('--format',
                        help="bbox format: xywh or x1y1x2y2",
                        default="x1y1x2y2")

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    # Load emotic annotations
    emotic = json.load(open(fr'{args.indir}/EMOTIC_train_{args.format}_tdpc.json', 'r'))
    emotic_max_id = len(emotic["annotations"])

    # Load HECO annotations
    heco = json.load(open(fr'{args.indir}/HECO_{args.format}_tdpc.json', 'r'))

    for img in heco["images"]:
        img['id'] += emotic_max_id

    for ann in heco["annotations"]:
        ann['image_id'] += emotic_max_id
        ann['id'] += emotic_max_id

    # Merge the datasets
    for data in heco['images']:
        emotic['images'].append(data)
    for data in heco["annotations"]:
        emotic['annotations'].append(data)

    # Save the merged dataset
    with open(f'{args.outdir}/EMOTIC_and_HECO_train_{args.format}_tdpc.json', 'w') as fp:
        json.dump(emotic, fp)


if __name__ == '__main__':
    start_time = time.time()
    print('Working...')
    main()
    print(f'Total Time : {time.time() - start_time:03.1f} seconds')
