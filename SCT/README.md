# Self-Calibrated Tuning of Vision-Language Models for Out-of-Distribution Detection

## Requirement
### Package
Our experiments are conducted with Python 3.8 and Pytorch 1.8.1.

All required packages are based on [CoOp](https://github.com/KaiyangZhou/CoOp) (for training) and [MCM](https://github.com/deeplearning-wisc/MCM/blob/main/utils/plot_util.py) (for evaluation).
This code is built on top of the awesome toolbox [Dassl.pytorch](https://github.com/KaiyangZhou/Dassl.pytorch) so you need to install the `dassl` environment first. Simply follow the instructions described [here](https://github.com/KaiyangZhou/Dassl.pytorch#installation) to install `dassl` as well as PyTorch. After that, run `pip install -r requirements.txt` under `LoCoOp/` to install a few more packages required by [CLIP](https://github.com/openai/CLIP) and [MCM](https://github.com/deeplearning-wisc/MCM/blob/main/utils/plot_util.py) (this should be done when `dassl` is activated).


### Datasets
Please create `data` folder and download the following ID and OOD datasets to `data`.

#### In-distribution Datasets
We use ImageNet-1K as the ID dataset.
- Create a folder named `imagenet/` under `data` folder.
- Create `images/` under `imagenet/`.
- Download the dataset from the [official website](https://image-net.org/index.php) and extract the training and validation sets to `$DATA/imagenet/images`.

#### Out-of-distribution Datasets
We use the large-scale OOD datasets [iNaturalist](https://arxiv.org/abs/1707.06642), [SUN](https://vision.princeton.edu/projects/2010/SUN/), [Places](https://arxiv.org/abs/1610.02055), and [Texture](https://arxiv.org/abs/1311.3618) curated by [Huang et al. 2021](https://arxiv.org/abs/2105.01879). We follow instructions from this [repository](https://github.com/deeplearning-wisc/large_scale_ood#out-of-distribution-dataset) to download the subsampled datasets.

The overall file structure is as follows:
```
LoCoOp
|-- data
    |-- imagenet
        |-- images/
            |--train/ # contains 1,000 folders like n01440764, n01443537, etc.
            |-- val/ # contains 1,000 folders like n01440764, n01443537, etc.
    |-- iNaturalist
    |-- SUN
    |-- Places
    |-- Texture
    ...
```

## Quick Start
The training script is in `LoCoOp/scripts/sct/train.sh`.

e.g., 1-shot training with ViT-B/16
```train
CUDA_VISIBLE_DEVICES=0 bash scripts/sct/train.sh data imagenet vit_b16_ep25 end 16 1 False 0.25 200
```

e.g., 16-shot training with ViT-B/16
```train
CUDA_VISIBLE_DEVICES=0 bash scripts/sct/train.sh data imagenet vit_b16_ep25 end 16 16 False 0.25 200
```