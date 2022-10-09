# Robustness Transfer Benchmark 

This repository contains the code to reproduce the results of our paper:
[Does Robustness on ImageNet Transfer to Downstream Tasks?](https://arxiv.org/abs/2204.03934) | Yutaro Yamada (Yale University), Mayu Otani (CyberAgent Inc.), CVPR 2022.

![](https://i.imgur.com/7P0w0yo.png)

## Requirements
To install requirements, follow the installation guide in [MMDetection](https://mmdetection.readthedocs.io/en/latest/get_started.html#installation) and [MMSegmentation](https://mmsegmentation.readthedocs.io/en/latest/get_started.html#installation).
For example,

```
conda install pytorch torchvision -c pytorch
pip install -U openmim
mim install mmcv-full
pip install mmdet
pip install mmsegmentation
pip install imagecorruptions
```

Specific versions we tested are in `requirements.txt`.

## Data
Download [COCO2017](https://cocodataset.org/#download) and save the data in `data/coco/`.

```
./
└── data
    └── coco
        ├── annotations
        └── val2017
```

## Benchmarking

- To measure how well a model transfers robustness from ImageNet classification to downstream tasks, we have to prepare the same set of distributional shifts that can be applied to both classification and downstream tasks. We focus on 15 synthetic image corruption types, originally introduced in ImageNet-C [1]. 
- We use the [python package](https://github.com/bethgelab/imagecorruptions) to generate synthetic corruptions. Inspired by the [Corruption Benchmarking tool](https://mmdetection.readthedocs.io/en/latest/robustness_benchmarking.html) for object detection [2] provided in [mmdetection](https://github.com/open-mmlab/mmdetection), we modified the data pipeline of [mmsegmentation](https://github.com/open-mmlab/mmsegmentation) to incorporate image corruptions into the performance evaluation.
- Example script to evaluate robustness of trained networks:
```python
# Object detection (inside mmdetection folder)
python tools/analysis_tools/test_robustness.py \
  config_file_used_to_train_your_model \
  your_trained_model.pth \
  --eval bbox segm \
  --out corrupt_results_summary.pkl \
  --corruptions "corruption_type" \
  --severities 1

# Semantic segmentation (inside mmsegmentation folder)
python tools/analysis_tools/test_robustness.py \
  config_file_used_to_train_your_model \
  your_trained_model.pth \
  --eval mIoU \
  --out corrupt_results_summary.pkl \
  --corruptions "corruption_type" \
  --severities 1
```
- For all of our examples, we used severities=1 but you can set up to severities=5.


For training/fine-tuning, we based off of the following repositories:
- [mmdetection](https://github.com/open-mmlab/mmdetection)
- [mmsegmentation](https://github.com/open-mmlab/mmsegmentation)
- [robust-models-transfer](https://github.com/Microsoft/robust-models-transfer)

## Trained models 

The trained models and config files for each training setting is listed below:

- Fixed-Feature Transfer Learning

| Fixed-Feature Transfer Learning| Regular | ANT  | DeepAug+AugMix | Swin-T |
| ----------- | ----------- |----------- | -----------|-----------|
| Detection (MSCOCO)      |   [model](https://github.com/runopti/RobustDownstreamTransfer/releases/download/v1.0.0/fixedfeature_cocodetect_regular_epoch_24.pth) / [config](https://github.com/runopti/RobustDownstreamTransfer/blob/main/mmdetection/configs/mask_rcnn/mask_rcnn_r50_fpn_2x_lr0.04_coco_fixedfeature.py)     | [model](https://github.com/runopti/RobustDownstreamTransfer/releases/download/v1.0.0/fixedfeature_cocodetect_ant_epoch_24.pth) / [config](https://github.com/runopti/RobustDownstreamTransfer/blob/main/mmdetection/configs/mask_rcnn/mask_rcnn_r50_ant_fpn_2x_lr0.04_coco_fixedfeature.py) | [model](https://github.com/runopti/RobustDownstreamTransfer/releases/download/v1.0.0/fixedfeature_cocodetect_deepaugaugmix_epoch_24.pth) / [config](https://github.com/runopti/RobustDownstreamTransfer/blob/main/mmdetection/configs/mask_rcnn/mask_rcnn_r50_deepaug_augmix_fpn_2x_lr0.04_coco_fixedfeature.py) | [model](https://github.com/runopti/RobustDownstreamTransfer/releases/download/v1.0.0/fixedfeature_cocodetect_swinT_epoch_24.pth) / [config](https://github.com/runopti/RobustDownstreamTransfer/blob/main/configs/swin/mask_rcnn_swin-t-p4-w7_fpn_fp16_ms-crop-2x_coco_fixedfeature.py) |
| Segmentation (ADE20K)   | model / [config](https://github.com/runopti/RobustDownstreamTransfer/blob/main/mmsegmentation/configs/upernet/upernet_r50_512x512_80k_ade20k_regular_fixedfeature.py)      | model / [config](https://github.com/runopti/RobustDownstreamTransfer/blob/main/mmsegmentation/configs/upernet/upernet_r50_512x512_80k_ade20k_ant_fixedfeature.py) | model / [config](https://github.com/runopti/RobustDownstreamTransfer/blob/main/mmsegmentation/configs/upernet/upernet_r50_512x512_80k_ade20k_deepaug_augmix_fixedfeature.py) | model / [config](https://github.com/runopti/RobustDownstreamTransfer/blob/main/mmsegmentation/configs/swin/upernet_swin_tiny_patch4_window7_512x512_80k_ade20k_pretrain_224x224_1K_fixedfeature.py) |


| Full-Network Transfer Learning| Regular | ANT  | DeepAug+AugMix | Swin-T |
| ----------- | ----------- |----------- | -----------|-----------|
| Detection (MSCOCO)      |   model / [config](https://github.com/runopti/RobustDownstreamTransfer/blob/main/mmdetection/configs/mask_rcnn/mask_rcnn_r50_fpn_2x_coco.py)     | model / [config](https://github.com/runopti/RobustDownstreamTransfer/blob/main/mmdetection/configs/mask_rcnn/mask_rcnn_r50_ant_fpn_2x_lr0.04_coco.py) | model / [config](https://github.com/runopti/RobustDownstreamTransfer/blob/main/mmdetection/configs/mask_rcnn/mask_rcnn_r50_deepaug_augmix_fpn_2x_lr0.04_coco.py) | model / [config](https://github.com/runopti/RobustDownstreamTransfer/blob/main/mmdetection/configs/swin/mask_rcnn_swin-t-p4-w7_fpn_fp16_ms-crop-2x_coco.py) |
| Segmentation (ADE20K)   | model / [config](https://github.com/runopti/RobustDownstreamTransfer/blob/main/mmsegmentation/configs/upernet/upernet_r50_512x512_80k_ade20k_regular.py)      | model / [config](https://github.com/runopti/RobustDownstreamTransfer/blob/main/mmsegmentation/configs/upernet/upernet_r50_512x512_80k_ade20k_ant.py) | model / [config](https://github.com/runopti/RobustDownstreamTransfer/blob/main/mmsegmentation/configs/upernet/upernet_r50_512x512_80k_ade20k_deepaug_augmix.py) | model / [config](https://github.com/runopti/RobustDownstreamTransfer/blob/main/mmsegmentation/configs/swin/upernet_swin_tiny_patch4_window7_512x512_80k_ade20k_pretrain_224x224_1K.py) |


- For classification (cifar10), we used the code from [3] with epochs=150, lr=0.01, step-lr=30, batch-size=64, weight-decay=5e-4 for all cases (e.g. Regular, ANT, DeepAug+, and Swin-T.)

- We also investigat a recently proposed data augmentation technique to improve common corruption robustness called 'PRIME' [4] and apply it during transfer learning from ImageNet-pretrained models.


| PRIME Data Aug during Transfer Learning| Regular | PRIME  | DeepAug+AugMix | Swin-T |
| ----------- | ----------- |----------- | -----------|-----------|
| Detection (MSCOCO)      |   model / [config](https://github.com/runopti/RobustDownstreamTransfer/blob/main/mmdetection/configs/mask_rcnn/mask_rcnn_r50_prime_fpn_2x_lr0.04_coco.py)     | model / [config](https://github.com/runopti/RobustDownstreamTransfer/blob/main/mmdetection/configs/mask_rcnn/mask_rcnn_r50_prime_while_finetune_prime_fpn_2x_lr0.04_coco.py) | model / [config](https://github.com/runopti/RobustDownstreamTransfer/blob/main/mmdetection/configs/mask_rcnn/mask_rcnn_r50_prime_while_finetune_deepaug_fpn_2x_lr0.04_coco.py) | model / [config](https://github.com/runopti/RobustDownstreamTransfer/blob/main/mmdetection/configs/swin/mask_rcnn_prime_while_finetune_swin-t-p4-w7_fpn_fp16_ms-crop-2x_coco.py) |
| Segmentation (ADE20K)   | model / [config](https://github.com/runopti/RobustDownstreamTransfer/blob/main/mmsegmentation/configs/upernet/upernet_r50_512x512_80k_ade20k_prime_while_finetune_regular.py)      | model / [config](https://github.com/runopti/RobustDownstreamTransfer/blob/main/mmsegmentation/configs/upernet/upernet_r50_512x512_80k_ade20k_prime_while_finetune_prime.py) | model / [config](https://github.com/runopti/RobustDownstreamTransfer/blob/main/mmsegmentation/configs/upernet/upernet_r50_512x512_80k_ade20k_prime_while_finetune_deepaug_augmix.py) | model / [config](https://github.com/runopti/RobustDownstreamTransfer/blob/main/mmsegmentation/configs/swin/upernet_swin_tiny_patch4_window7_512x512_80k_ade20k_pretrain_224x224_1K_prime_while_finetune.py) |




## Bibtex
```
@inproceedings{robusttransfer,
      title={Does Robustness on ImageNet Transfer to Downstream Tasks?},
      author={Yutaro Yamada, Mayu Otani},
      year={2022},
      booktitle={CVPR},
}
```

## Reference
 - [1] Hendrycks et al. "[Benchmarking Neural Network Robustness to Common Corruptions and Perturbations](https://arxiv.org/abs/1903.12261)" ICLR 2019
 - [2]  Michaelis et al. "[Benchmarking Robustness in Object Detection:
Autonomous Driving when Winter is Coming](https://arxiv.org/abs/1907.07484)" Machine Learning for Autonomous Driving Workshop at NeurIPS 2019
 - [3] Salman et al. "[Do Adversarially Robust ImageNet Models Transfer Better?](https://github.com/Microsoft/robust-models-transfer)" NeurIPS 2020
 - [4] Modas and Rade et al. "[PRIME: A Few Primitives Can Boost
Robustness to Common Corruptions](https://arxiv.org/abs/2112.13547)" arXiv 2021. 


