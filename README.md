# Enhancing Surgical Instrument Segmentation: Integrating Vision Transformer Insights with Adapter

Model structure:

![](https://github.com/weimengmeng1999/AdapterSIS/blob/main/figures/model.png)


# Training
Train with ViT-L
```python
python train.py \
        --data_path ../robomis \
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

#  Evaluation
For evaluation, simply add --evaluate for the training file
```python
python train.py \
        --data_path ../robomis \
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
