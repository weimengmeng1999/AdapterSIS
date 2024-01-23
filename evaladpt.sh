# python eval_adapter.py \
#         --data_path /nfs/home/mwei/mmsegmentation/data/robo \
#         --output_dir /nfs/home/mwei/SelfSL4MIS_experiment/eval_adapter \
#         --arch vit_small \
#         --patch_size 14 \
#         --n_last_blocks 4 \
#         --imsize 588 \
#         --lr 0.01 \
#         --config_file dinov2/configs/eval/vits14_pretrain.yaml \
#         --pretrained_weights https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth \
#         --num_workers 2 \
#         --epochs 500 \
#         --evaluate
# python eval_adapter.py \
#         --data_path /nfs/home/mwei/mmsegmentation/data/robo \
#         --output_dir /nfs/home/mwei/SelfSL4MIS_experiment/eval_adapter_plus_vit \
#         --arch vit_base \
#         --patch_size 14 \
#         --n_last_blocks 4 \
#         --imsize 588 \
#         --lr 0.01 \
#         --config_file dinov2/configs/eval/vits14_pretrain.yaml \
#         --pretrained_weights https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth \
#         --num_workers 2 \
#         --epochs 500 \
python train.py \
        --data_path /nfs/home/mwei/mmsegmentation/data/robo \
        --output_dir /nfs/home/mwei/AdapterExp/exp1 \
        --arch vit_base \
        --patch_size 14 \
        --n_last_blocks 4 \
        --imsize 588 \
        --lr 0.01 \
        --config_file dinov2/configs/eval/vitl14_pretrain.yaml \
        --pretrained_weights https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth \
        --num_workers 2 \
        --epochs 500 \