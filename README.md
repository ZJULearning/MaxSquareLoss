# Domain Adaptation for Semantic Segmentation with Maximum Squares Loss

By Minghao Chen, Hongyang Xue, Deng Cai.

## Introduction

A **PyTorch** implementation for our paper ["Domain Adaptation for Semantic Segmentation with Maximum Squares Loss"](). The segmentation model is based on Deeplabv2 with ResNet-101 backbone. "MaxSquare+IW+Multi" introduced in the paper achieve competitive result on all three UDA datasets: GTA5, SYNTHIA, NTHU. Moreover, our method achieves the state-of-the-art results in GTA5-to-Cityscapes and Cityscapes-to-NTHU adaptation.

### Citation

If you use this code in your research, please cite:

```

```

## Requirements
The code is implemented with Python(3.6) and Pytorch(1.0.0).

Install the newest Pytorch from https://pytorch.org/.

To install the required python packages, run

```python
pip install -r requirements.txt
```

## Setup

#### GTA5-to-Cityscapes:

- Download [GTA5 datasets](https://download.visinf.tu-darmstadt.de/data/from_games/), which contains 24,966 annotated images with 1914×1052 resolution taken from the GTA5 game. We use the sample code for reading the label maps and a split into training/validation/test set from [here](https://download.visinf.tu-darmstadt.de/data/from_games/code/read_mapping.zip). In the experiments, we resize GTA5 images to 1280x720.
- Download [Cityscapes](https://www.cityscapes-dataset.com/), which contains 5,000 annotated images with 2048 × 1024 resolution taken from real urban street scenes. We resize Cityscapes images to 1024x512 (or 1280x640 which yields sightly better results but costs more time). 
- Download the checkpoint pretrained on GTA5.
- If you want to pretrain the model by yourself, download the model pretrained on ImageNet.

#### SYNTHIA-to-Cityscapes:

- Download [SYNTHIA-RAND-CITYSCAPES](http://synthia-dataset.net/download/808/) consisting of 9,400 1280 × 760 synthetic images. We resize images to 1280x760.
- Download the checkpoint pretrained on SYNTHIA.

#### Cityscapes-to-NTHU

- Download [NTHU dataset](https://yihsinchen.github.io/segmentation_adaptation_dataset/), which consists 2048 × 1024 resolution from four different cities: Rio,
  Rome, Tokyo, and Taipei. We resize images to 1024x512, the same as Cityscapes.
- Download the checkpoint pretrained on Cityscapes.

Put all datasets in "datasets" folder and all checkpoints in "pretrained_model" folder.

## Results

We present several trained results reported in our paper and provide the corresponding checkpoints.

#### GTA5-to-Cityscapes:

| Method  | Source | MinEnt$^\dagger$ | MaxSquare | MaxSquare+IW | MaxSquare+IW+Multi |
| :-----: | :----: | :--------------: | :-------: | :----------: | :----------------: |
| mIoU(%) |  36.9  |       42.2       |   44.3    |     45.2     |        46.4        |

To get above results, download corresponding checkpoints and put them to "pretrained_model/checkpoint" folder. Use the following code to evaluate.

**MaxSquare**

```
python3 tools/evaluate.py --gpu "0" --dataset 'cityscapes' --checkpoint_dir "./log/eval_city" --pretrained_ckpt_file "./pretrained_model/checkpoint/GTA5_to_Cityscapes_MaxSquare.pth" --image_summary True
```

**MaxSquare+IW**

```
python3 tools/evaluate.py --gpu "0" --dataset 'cityscapes' --checkpoint_dir "./log/eval_city" --pretrained_ckpt_file "./pretrained_model/checkpoint/GTA5_to_Cityscapes_MaxSquare_IW.pth" --image_summary True
```

**MaxSquare+IW+Multi**

```
python3 tools/evaluate.py --gpu "0" --dataset 'cityscapes' --checkpoint_dir "./log/eval_city" --pretrained_ckpt_file "./pretrained_model/checkpoint/GTA5_to_Cityscapes_MaxSquare_IW_Multi.pth" --image_summary True
```

Since we use random mirror to augment the data, ensembling the model prediction and the prediction on the filpped image will sightly boost the performance, which can be done with "--flip True"

```
python3 tools/evaluate.py --gpu "0" --dataset 'cityscapes' --checkpoint_dir "./log/eval_city" --pretrained_ckpt_file "./pretrained_model/checkpoint/GTA5_to_Cityscapes_MaxSquare_IW_Multi.pth" --image_summary True --flip True
```

To have a look at predicted examples, run tensorboard as follows:

```
tensorboard --logdir=./log/eval_city  --port=6009
```



#### SYNTHIA-to-Cityscapes:

|  Method  | Source | MinEnt$^\dagger$ | MaxSquare | MaxSquare+IW | MaxSquare+IW+Multi |
| :------: | :----: | :--------------: | :-------: | :----------: | :----------------: |
| mIoU(%)  |  30.1  |       38.0       |   39.3    |     40.4     |        41.4        |
| mIoU*(%) |  34.3  |       44.5       |   45.8    |     46.9     |        48.2        |

**MaxSquare**

```
python3 tools/evaluate.py --gpu "0" --source_dataset 'synthia' --dataset 'cityscapes' --checkpoint_dir "./log/eval_city" --pretrained_ckpt_file "./pretrained_model/checkpoint/synthia_to_Cityscapes_MaxSquare.pth" --image_summary True  --num_classes 16
```

**MaxSquare+IW**

```
python3 tools/evaluate.py --gpu "0" --source_dataset 'synthia' --dataset 'cityscapes' --checkpoint_dir "./log/eval_city" --pretrained_ckpt_file "./pretrained_model/checkpoint/synthia_to_Cityscapes_MaxSquare_IW.pth" --image_summary True  --num_classes 16
```

**MaxSquare+IW+Multi**

```
python3 tools/evaluate.py --gpu "0" --source_dataset 'synthia' --dataset 'cityscapes' --checkpoint_dir "./log/eval_city" --pretrained_ckpt_file "./pretrained_model/checkpoint/synthia_to_Cityscapes_MaxSquare_IW_Multi.pth" --image_summary True --flip True --num_classes 16
```



#### Cityscapes-to-NTHU

##### **Rome**

| Method  | Source | MaxSquare | MaxSquare+IW |
| :-----: | :----: | :-------: | :----------: |
| mIoU(%) |  51.0  |   53.9    |     54.5     |

**Rio**

| Method  | Source | MaxSquare | MaxSquare+IW |
| :-----: | :----: | :-------: | :----------: |
| mIoU(%) |  48.9  |   52.0    |     53.3     |

**Tokyo**

| Method  | Source | MaxSquare | MaxSquare+IW |
| :-----: | :----: | :-------: | :----------: |
| mIoU(%) |  47.8  |   49.7    |     50.5     |

**Taipei**

| Method  | Source | MaxSquare | MaxSquare+IW |
| :-----: | :----: | :-------: | :----------: |
| mIoU(%) |  46.3  |   49.8    |     50.6     |

Evaluation (take "Rome" for example):

**MaxSquare**

```
python3 tools/evaluate.py --gpu "0" --city_name 'Rome' --checkpoint_dir "./log/eval_Rome" --pretrained_ckpt_file "./pretrained_model/checkpoint/Cityscapes_to_Rome_MaxSquare.pth" --image_summary True  --num_classes 13
```

**MaxSquare+IW**

```
python3 tools/evaluate.py --gpu "0" --city_name 'Rome' --checkpoint_dir "./log/eval_Rome" --pretrained_ckpt_file "./pretrained_model/checkpoint/Cityscapes_to_Rome_MaxSquare_IW.pth" --image_summary True  --num_classes 13
```



## Training

#### GTA5-to-Cityscapes:

(Optional) Pretrain the model on the source domain (GTA5). 

You can skip this step by downloading the checkpoint pretrained on GTA5 in "Setup" section.

```
python3 tools/train_source.py --gpu "0" --dataset 'gta5' --checkpoint_dir "./log/gta5_pretrain/" --iter_max 200000 --iter_stop 80000 --freeze_bn False --weight_decay 5e-4 --lr 2.5e-4 --crop_size "1280,720"
```

Then in next step, set `--pretrained_ckpt_file "./log/gta5_pretrain/gta5best.pth"`.

**MaxSquare**

```
python3 tools/solve_gta5.py --gpu "0" --source_dataset 'gta5'  --dataset 'cityscapes'   --checkpoint_dir "./log/gta2city_maxsquare/" --pretrained_ckpt_file "./pretrained_model/GTA5_source.pth"  --crop_size "1280,720" --target_crop_size "1024,512"  --epoch_num 10 --target_mode "maxsquare" --lr 2.5e-4 --lambda_target 0.1
```

**MaxSquare+IW**

```

```

(Optional) Pretrain the multi-level model on the source domain (GTA5) by adding "--multi True". 

Otherwise, download this checkpoint pretrained on GTA5.

**MaxSquare+IW+Multi**

```

```



#### **SYNTHIA-to-Cityscapes:**

(Optional) Pretrain the model on the source domain (SYNTHIA). 

```

```

**MaxSquare** (example)

```

```



#### Cityscapes-to-NTHU

(Optional) Pretrain the model on the source domain (Cityscapes). 

```

```

**MaxSquare** (take "Rome" for example)

```

```



## Acknowledgment