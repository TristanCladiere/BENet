# BENet: A lightweight bottom-up framework for context-aware emotion recognition


## Abstract

Emotion recognition from images is a challenging task. The latest and most common approach to solve this problem is to 
fuse information from different contexts, such as person-centric features, scene features, object features, interactions
features and so on. This requires specialized pre-trained models, and multiple pre-processing steps, resulting in long 
and complex frameworks, not always practicable in real time scenario with limited resources. Moreover, these methods do 
not deal with person detection, and treat each subject sequentially, which is even slower for scenes with many people. 
Therefore, we propose a new approach, based on a single end-to-end trainable architecture that can both detect and 
process all subjects simultaneously by creating emotion maps. We also introduce a new multitask training protocol which 
enhances the model predictions. Finally, we present a new baseline for emotion recognition on EMOTIC dataset, which
considers the detection of the agents.

![Overview](/figures/overview.png "Overview of the proposed BENet framework")

## Main Results

### mAP scores for emotion recognition on EMOTIC test set.

| BU    | BU + Det | BU + Det + PC | BU + Det + PC + BG | BU + Det + PC + BG + ED |
|-------|----------|---------------|--------------------|-------------------------|
| 23.10 | 27.22    | 27.49         | 27.73              | **28.75**               | 
  
  

### AP scores for human detection on EMOTIC test set

| BU + Det | BU + Det + PC | BU + Det + PC + BG | BU + Det + PC + BG + ED |
|----------|---------------|--------------------|-------------------------|
| 49.66    | 49.16         | 51.49              | **51.71**               |

### mAP scores for emotion recognition on EMOTIC test set, depending on the human detector predictions for thresholds from 0.50 to 0.9

| Detection threshold     | 0.50      | 0.55      | 0.60      | 0.65      | 0.70      | 0.75      | 0.80      | 0.85      | 0.90      | 0.95      | Average   |
|-------------------------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|
| BU + Det                | 25.73     | 25.38     | 25.04     | 24.68     | 24.37     | 24.00     | 23.62     | 22.83     | 21.71     | 19.37     | 23.67     |
| BU + Det + PC           | 26.19     | 25.85     | 25.51     | 25.32     | 25.13     | 24.81     | 24.20     | 23.24     | 21.91     | 19.37     | 24.15     |
| BU + Det + PC + BG      | 26.01     | 25.86     | 25.57     | 25.24     | 24.97     | 24.58     | 23.99     | 23.29     | 22.00     | 19.48     | 24.10     |
| BU + Det + PC + BG + ED | **26.96** | **26.66** | **26.38** | **26.07** | **25.82** | **25.28** | **24.61** | **23.71** | **22.21** | **19.57** | **24.73** |


*Note: **BU**=Bottom-Up Head / **Det**=Detection Head / **PC**=Person-Centric Head / **BG**=Background Head / **ED**=Extra Data*

## Environment
The code is developed using python 3.9 on Windows 10, but also works on Linux Ubuntu 20.04. NVIDIA GPUs are needed.

## Quick start
### Installation
1. Clone this repo, and we'll call the directory that you cloned as ${BENET_ROOT}.
2. Install pytorch following [official instruction](https://pytorch.org/get-started/locally).
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Install [COCOAPI](https://github.com/cocodataset/cocoapi):
      
   - COCOAPI=/path/to/clone/cocoapi -> pick a path where you'd like to have the software cloned and then set an environment variable (COCOAPI in this case) accordingly.
   ```
   git clone https://github.com/cocodataset/cocoapi.git $COCOAPI
   ```
   - Move the file **cocoeval_center_emotions.py** into `$COCOAPI/PythonAPI/pycocotools`
   - If you are on Linux:
     - Move inside `$COCOAPI/PythonAPI`, make the installation into global site-packages: 
       ```
       make install
       ```
     - Alternatively, if you do not have permissions or prefer not to install the COCO API into global site-packages:
       ```
       python setup.py install --user
       ```
   - If you are on Windows:
     - You need to have [Visual Studio](https://visualstudio.microsoft.com/) installed.
     - In the Visual Studio Installer, modify your installation and ensure that the C++ Desktop Builds Tools are ticked. 
     - Move inside `$COCOAPI/PythonAPI` 
     - Modify the file **setup.py** by deleting the line 12: `extra_compile_args=[‘-Wno-cpp’, ‘-Wno-unused-function’, ‘-std=c99’]`
     - Run:
       ```
       python setup.py install
       ```


Your directory tree should look like this:
```
   ${BENET_ROOT}
   ├── experiments
   ├── figures
   ├── lib
   ├── tools 
   ├── README.md
   └── requirements.txt
```
   
### Data preparation

Come back under ${BENET_ROOT}, and create a new dir to store the data: ``mkdir data`` 

Then, download [EMOTIC](https://s3.sunai.uoc.edu/emotic/download.html) and extract it under ``${BENET_ROOT}/data``:
```
${BENET_ROOT}
| data
   | EMOTIC
      | PAMI
         | Annotations
         | emotic
            | ade20k
            | emodb_small
            | framesdb
            | mscoco
```

If you want to use extra data for training, please download [HECO](https://heco2022.github.io/), and extract it under ``${BENET_ROOT}/data``:
```
${BENET_ROOT}
| data
   | HECO
      | Data
      | Labels 
```

---
**WARNING:** There is a typo line **9736** in ``HECO_Labels.csv``. You must replace **Happinesst** by **Happiness** before going further.

---

After downloading data, run under `${BENET_ROOT}`:
```
python tools/emotic_mat_to_json.py --emotic data/EMOTIC/PAMI/Annotations --out_dir new_annotations --format x1y1x2y2
```
It will convert the `Annotations.mat` file from **EMOTIC** into 3 .json files following **COCO** dataset structure.

Then, use `emotic_top_down_anns.py` to convert the newly created .json files into "person-centric" versions that are necessary to perform multitasks training:
```
python tools/emotic_top_down_anns.py --emotic new_annotations --out_dir new_annotations --x1y1x2y2 True
```

If you want to use **HECO** as extra training data, run `heco_csv_to_json.py` then `merge_tdpc_emotic_heco.py`:
```
python tools/heco_csv_to_json.py --heco data/HECO --out new_annotations --format x1y1x2y2
```
```
python tools/merge_tdpc_emotic_heco.py --indir new_annotations --outdir new_annotations --format x1y1x2y2
```

### Download pretrained models and paper results

All the config files related to the experiments presented in our ACIVS 2023 paper can be found in ``experiments/emotic``.

The **Imagenet** and **COCO Pose** pretrained backbones, as well as the tensorboard log files and all the results of those experiments can be downloaded from [here](https://drive.google.com/file/d/1KFx-Mbrz9-xc9iqa5QzKKxD8r1xZyqkZ/view?usp=drive_link). 
Add the 3 folders (`output`, `models`, and `log`) in ${BENET_ROOT}.


At this point, your directory tree should look like this:
```
   ${BENET_ROOT}
   ├── data
   ├── experiments
   ├── figures
   ├── lib
   ├── log
   ├── models
   ├── output
   ├── tools 
   ├── README.md
   └── requirements.txt
```

### Training

---
**NOTE:** if you are on Windows, please replace ``nccl`` by ``gloo`` in line 20 of ``lib/config/default_mt.py`` before going further. 

---
**WARNING:** If you want to redo one of the experiments presented in our paper (``emotic_acivs_bu_det.yaml`` for example), you should duplicate the corresponding experiment config file and use a **new** name. If you do not, it will erase the provided model weights. 

---
Then run:

```
python tools/train_mt.py --cfg experiments/emotic/emotic_acivs_bu_det.yaml --test 10 TRAIN.IMAGES_PER_GPU 5
```
The argument ``--test <n>`` is used to test the model on **EMOTIC** test set every *n* epochs. You can ignore it if you are not interested in monitoring these temporary scores.
Please also adapt the batch size with ``TRAIN.IMAGES_PER_GPU <batch_size>`` to fit with your GPU memory.

### Testing

To test a model (*emotic_acivs_bu_det.yaml* for example), run:
```
python tools/test_mt.py --cfg experiments/emotic/emotic_acivs_bu_det.yaml --state best
```
The argument ``--state <model_state>`` can be:
   - ``best``: version with the smallest validation loss
   - ``final``: version obtained at the end of the training

## Image inference

To perform single image inference, use ``tools/img_inference.py`` and choose your parameters:

```
    --cfg <config_file> 
    --img <path> 
    --emo_thr <val> (between 0 and 1)
    TEST.DETECTION_THRESHOLD <val> (between 0 and 1)
```
For example, you can try:

```
python tools/img_inference.py --cfg experiments/emotic/emotic_and_heco_acivs_1_no_fusion.yaml --img data/EMOTIC/PAMI/emotic/mscoco/images/COCO_val2014_000000039106.jpg --emo_thr 0.4 TEST.DETECTION_THRESHOLD 0.1
```

---
**NOTE:** It is note trivial to choose good threshold values, especially for the emotions : currently, the given value 
applies to **all** the emotions, whereas a tuned threshold **per** emotion may result in better predictions. For example, 
"Engagement" tends to have high confidence, so the threshold for this one should be high to avoid a constant prediction.

---

## Visualize detection and emotions maps

To visualize the detection heatmap and some emotion maps, you can use ``tools/show_maps.py`` with the main parameters:
```
    --cfg <config_file> 
    --emo <emotion(s)_map_to_display>
    --img <path> 
```

For example, you can try:

```
python tools/show_maps.py --cfg experiments/emotic/emotic_and_heco_acivs_1_no_fusion.yaml --emo Affection Anger Anticipation Confidence Disconnection Engagement Excitement Happiness Peace Pleasure --img data/EMOTIC/PAMI/emotic/mscoco/images/COCO_val2014_000000039106.jpg
```


Illustration of the heatmap produced by the detection head, and two emotion maps
given by the bottom-up head. Normalisation is between 0 (black, emotion is absent)
and 1 (white, emotion is present). The predicted and annotated bounding boxes are
also added to the raw image.

![Example1](/figures/example1.png)

## Acknowledgement

Our codes are largely inspired by [HigherHRNet](https://github.com/HRNet/HigherHRNet-Human-Pose-Estimation).

