# Enhancing Surgical Instrument Segmentation: Integrating Vision Transformer Insights with Adapter

Model structure:

![](https://github.com/weimengmeng1999/AdapterSIS/blob/main/figures/model.png)

## Requirements

- `torch==2.0.0`
- `torchvision==0.15.0`
- `torchmetrics==0.10.3`
- `albumentations`

## Datasets
We use five datasets (Robust-MIS 2019, EndoVis 2017, EndoVis 2018, CholecSeg8k, AutoLaparo) in our paper.

For `Robust-MIS 2019`, you can download the dataset from [here](https://www.synapse.org/#!Synapse:syn20575265) and then put the files in `data/robomis`.

For `EndoVis 2017`, You can apply the dataset [here](https://endovissub2017-roboticinstrumentsegmentation.grand-challenge.org/) by registration.

For `EndoVis 2018`, You can apply the dataset [here](https://endovissub2018-roboticscenesegmentation.grand-challenge.org/) by registration.

For `CholecSeg8k`, you can download the dataset from [here](https://www.kaggle.com/datasets/newslab/cholecseg8k).

For `AutoLaparo`, you can request the dataset from [here](https://autolaparo.github.io/).

NOTE: The `Robust-MIS 2019' dataset includes 3 stages of testing, and the stage 3 is unseen images during training process.

## Training
Train with ViT-L on a single GPU
```python
python train.py \
        --data_path ../data/robomis \
        --output_dir .../eval_adapter_plus_vitl \
        --arch vit_base \
        --patch_size 14 \
        --n_last_blocks 4 \
        --imsize 588 \
        --lr 0.01 \
        --config_file dinov2/configs/eval/vitl14_pretrain.yaml \
        --pretrained_weights https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth \
        --num_workers 2 \
        --epochs 500 \
```
Train with ViT-L on a multiple GPUs
```python
export CUDA_VISIBLE_DEVICES=0,1
export PYTHONPATH=/nfs/home/mwei/AdapterSIS-dgx
python -m torch.distributed.launch --nproc_per_node=2 eval_paper.py \
        --data_path /nfs/home/mwei/mmsegmentation/data/robo \
        --output_dir /nfs/home/mwei/AdapterExp/paperonn \
        --arch vit_base \
        --patch_size 14 \
        --n_last_blocks 4 \
        --imsize 588 \
        --lr 0.01 \
        --config_file dinov2/configs/eval/vitl14_pretrain.yaml \
        --pretrained_weights https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth \
        --num_workers 2 \
        --epochs 500 \
        --batch_size_per_gpu 12
```

##  Evaluation
For evaluation, simply add --evaluate for the training file
```python
python train.py \
        --data_path ../data/robomis \
        --output_dir .../eval_adapter_plus_vitl \
        --arch vit_base \
        --patch_size 14 \
        --n_last_blocks 4 \
        --imsize 588 \
        --lr 0.01 \
        --config_file dinov2/configs/eval/vitl14_pretrain.yaml \
        --pretrained_weights https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth \
        --num_workers 2 \
        --epochs 500 \
        --evaluate
```
