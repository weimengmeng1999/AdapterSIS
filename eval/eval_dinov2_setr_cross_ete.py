# Copyright (c) Facebook, Inc. and its affiliates.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import argparse
import json
from pathlib import Path
import glob

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torchvision import datasets
from torchvision import transforms as pth_transforms
from torchvision import models as torchvision_models
from PIL import Image, ImageFile
import numpy as np
import math 

import utils
import vision_transformer as vits
from eval_knn import extract_features
from timm.models.layers import trunc_normal_ 
from einops import rearrange
import matplotlib.pyplot as plt
from masktrans_block import Block, FeedForward
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
# from torchmetrics.functional import dice_score

import utils
import vision_transformer as vits
from setr_decoder import TransModel2d, TransConfig
from dataset import EndoVis2017, EndoVis2018, Robomis

from functools import partial
from dinov2.eval.setup import build_model_for_eval, get_autocast_dtype
from dinov2.utils.config import get_cfg_from_args
from dinov2.eval.utils import ModelWithIntermediateLayers
from dinov2.models import build_model_from_cfg


# from mmcv.cnn import build_norm_layer

class DC(nn.Module):
    def __init__(self,nb_classes):
        super(DC, self).__init__()
        
        self.softmax = nn.Softmax(1)
        self.nb_classes = nb_classes

    @staticmethod 
    def onehot(gt,shape):
        shp_y = gt.shape
        gt = gt.long()
        y_onehot = torch.zeros(shape)
        y_onehot = y_onehot.cuda()
        y_onehot.scatter_(1, gt, 1)
        return y_onehot


    def dice(self, output, target):
        output = self.softmax(output)
        if not all([i == j for i, j in zip(output.shape, target.shape)]):
            target = self.onehot(target, output.shape)

        sum_axis = list(range(2,len(target.shape)))

        s = (10e-20)
        intersect = torch.sum(output * target,sum_axis)
        dice = (2 * intersect) / (torch.sum(output,sum_axis) + torch.sum(target,sum_axis) + s)
        #dice shape is (batch_size, nb_classes)
        return 1.0 - dice.mean()  

    def forward(self, output, target):
        result = self.dice(output, target)
        return result
    
def iou(y_true, y_pred):
    intersection = (y_true * y_pred).sum()
    union = y_true.sum() + y_pred.sum() - intersection
    return (intersection + 1e-6) / (union + 1e-6)
    
def ch_iou(y_true, y_pred):
    result = []

    if y_true.sum() == 0:
        if y_pred.sum() == 0:
            return 1
        else:
            return 0

    for type_id in set(y_true.flatten()):
        if type_id == 0:
            continue
        result += [iou(y_true == type_id, y_pred == type_id)]

    return np.mean(result)

def isi_iou(y_true, y_pred, problem_type='binary'):
    result = []

    if problem_type == 'binary':
        type_number = 2
    elif problem_type == 'parts':
        type_number = 4
    elif problem_type == 'instruments':
        type_number = 8

    if y_true.sum() == 0:
        if y_pred.sum() == 0:
            return 1
        else:
            return 0

    for type_id in range(type_number):
        if type_id == 0:
            continue
        if (y_true == type_id).sum() != 0 or (y_pred == type_id).sum() != 0:
            result += [iou(y_true == type_id, y_pred == type_id)]

    return np.mean(result)


def eval_linear(args):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    # ============ building network ... ============
    cfg = get_cfg_from_args(args)
    feature_model, _ = build_model_from_cfg(cfg, only_teacher=True)
    feature_model = feature_model.cuda()
    feature_model = nn.parallel.DistributedDataParallel(feature_model, device_ids=[args.gpu], find_unused_parameters=True)
    # model = build_model_for_eval(cfg, args.pretrained_weights)
    # autocast_dtype = get_autocast_dtype(cfg)
    # model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')



    # n_last_blocks_list = [1, 4]
    # n_last_blocks = max(n_last_blocks_list)
    # autocast_ctx = partial(torch.cuda.amp.autocast, enabled=True, dtype=autocast_dtype)
    # feature_model = ModelWithIntermediateLayers(model, n_last_blocks, autocast_ctx)
    # sample_output = feature_model(train_dataset[0][0].unsqueeze(0).cuda())

    # load weights to evaluate
    # utils.load_pretrained_weights(model, args.pretrained_weights, args.checkpoint_key, args.arch, args.patch_size)
    # print(f"Model {args.arch} built.")
    seg_decoder = Decoder2D(in_channels=384, 
                            out_channels=2, 
                            features=[256, 128, 64])
    seg_decoder = seg_decoder.cuda()
    seg_decoder = nn.parallel.DistributedDataParallel(seg_decoder, device_ids=[args.gpu], find_unused_parameters=True)

    # ============ preparing data ... ============
    val_transform = A.Compose([
        A.Resize(588, 588, interpolation=Image.BICUBIC),
        # A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

    dataset_val = Robomis(args.cross_test_path, split="validation", transform = val_transform, imsize=args.imsize)
    val_loader = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
    )


    train_transform = A.Compose([
                A.OneOf([
                    A.RandomSizedCrop(min_max_height=(int(
                        588 * 0.5), 588),
                                      height=588,
                                      width=588,
                                      p=0.5),
                A.PadIfNeeded(min_height=588, min_width=588, 
                              border_mode=cv2.BORDER_CONSTANT)
                ],p=1),
                A.HorizontalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.OneOf([
                    A.ElasticTransform(alpha=120,
                                       sigma=120 * 0.05,
                                       alpha_affine=120 * 0.03,
                                       p=0.5),
                    A.GridDistortion(p=0.5),
                    A.OpticalDistortion(distort_limit=2, shift_limit=0.5, p=1)
                ], p=0),
                        #p=0.8 if self.use_vis_aug_non_rigid else 0),
                A.CLAHE(p=0.8),
                A.RandomBrightnessContrast(p=0.8),
                A.RandomGamma(p=0.8),
            ])

    dataset_train = Robomis(args.data_path, split="training", transform = train_transform,
                            imsize=args.imsize)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset_train)
    train_loader = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    print(f"Data loaded with {len(dataset_train)} train and {len(dataset_val)} val imgs.")

    # set optimizer
    optimizer = torch.optim.SGD(
        seg_decoder.parameters(),
        args.lr * (args.batch_size_per_gpu * utils.get_world_size()) / 16., # linear scaling rule
        momentum=0.9,
        weight_decay=0, # we do not apply weight decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=0)

    # Optionally resume from a checkpoint
    to_restore = {"epoch": 0, "best_acc": 0.}
    utils.restart_from_checkpoint(
        os.path.join(args.output_dir, "checkpoint.pth.tar"),
        run_variables=to_restore,
        state_dict=seg_decoder,
        optimizer=optimizer,
        scheduler=scheduler,
    )
    start_epoch = to_restore["epoch"]
    best_acc = to_restore["best_acc"]

    # seed = 123
    # torch.manual_seed(seed)
    # np.random.seed(seed)
    # eval_metrics = []

    if args.evaluate:
        to_restore = {"epoch": 0, "best_acc": 0.}
        utils.restart_from_checkpoint(
            os.path.join(args.output_dir, "checkpoint.pth.tar"),
            run_variables=to_restore,
            state_dict=seg_decoder,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        start_epoch = to_restore["epoch"]
        best_acc = to_restore["best_acc"]
        # utils.load_pretrained_linear_weights(seg_decoder, args.arch, args.patch_size)
        # checkpoint = torch.load(os.path.join(args.output_dir, "checkpoint.pth.tar"), map_location="cpu")
        # for key in checkpoint.keys():
        #    print(key)
        # seg_decoder.load_state_dict(checkpoint['state_dict'], strict=False)

        # test_stats = validate_network(val_loader, model, seg_decoder, args.n_last_blocks, args.avgpool_patchtokens)
        test_stats = validate_network(val_loader, feature_model, seg_decoder, args.n_last_blocks, args.avgpool_patchtokens)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        return

    for epoch in range(start_epoch, args.epochs):
        train_loader.sampler.set_epoch(epoch)

        train_stats = train(feature_model, seg_decoder, optimizer, train_loader, epoch, args.n_last_blocks, args.avgpool_patchtokens)
        scheduler.step()

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch}
        if epoch % args.val_freq == 0 or epoch == args.epochs - 1:
            test_stats = validate_network(val_loader, feature_model, seg_decoder, args.n_last_blocks, args.avgpool_patchtokens)
            print(f"Accuracy at epoch {epoch} of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
            best_acc = max(best_acc, test_stats["acc1"])
            print(f'Max accuracy so far: {best_acc:.2f}%')
            log_stats = {**{k: v for k, v in log_stats.items()},
                         **{f'test_{k}': v for k, v in test_stats.items()}}
            # eval_metrics.append(test_stats["dice"])
        if utils.is_main_process():
            with (Path(args.output_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
            save_dict = {
                "epoch": epoch + 1,
                "encoder_state_dict": seg_decoder.state_dict(),
                "decoder_state_dict": seg_decoder.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_acc": best_acc,
            }
            torch.save(save_dict, os.path.join(args.output_dir, "checkpoint.pth.tar"))
    # mean_metric = np.mean(eval_metrics)
    # std_metric = np.std(eval_metrics)
    # print("mean dice:{:.4f}".format(mean_metric))
    # print("std dice:{:.4f}".format(std_metric ))
    print("Training of the supervised linear classifier on frozen features completed.\n"
                "Top-1 test accuracy: {acc:.1f}".format(acc=best_acc))


def train(model, seg_decoder, optimizer, loader, epoch, n, avgpool):
    model.train()
    seg_decoder.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    dice_loss = DC(2)
    for (inp, target, idx) in metric_logger.log_every(loader, 20, header):
        # move to gpu
        inp = inp.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        # forward
        H, W = target.size(1), target.size(2)

        x_tokens_list = model(inp, is_training=True)
        
        output_en = x_tokens_list["x_norm_patchtokens"]
        #   output_en = torch.cat([outputs for outputs, _ in intermediate_output], dim=-1)
        output_en = rearrange(output_en, "b (h w) (p1 p2 c) -> b c (h p1) (w p2)", 
                             p1 = 1, p2 = 1, 
                             h = H // 14, w = W // 14, 
                             c = 384)
        output = seg_decoder(output_en)
        # print(output_en.size())
        output = F.interpolate(output, size=(H, W), mode="bilinear")

        # compute cross entropy loss
        loss_ce = nn.CrossEntropyLoss()(output, target)
        # loss_ce= nn.CrossEntropyLoss(#weight=None, 
        #                              reduction='mean', weight = torch.Tensor([0.1, 10]).cuda(non_blocking=True))(output, target)
        # loss_ce= nn.CrossEntropyLoss(ignore_index = 0, reduction='mean')(output, target)
        # loss_ce = F.binary_cross_entropy_with_logits(output, target)
        loss_dce = dice_loss(output, target.unsqueeze(1))
        loss =  loss_ce + loss_dce
        # loss = loss_ce
        # loss = loss_dce
        # loss=nn.BCEWithLogitsLoss()(output, target)

        # compute the gradients
        optimizer.zero_grad()
        loss.backward()

        # step
        optimizer.step()

        # log 
        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def validate_network(val_loader, model, seg_decoder, n, avgpool):
    seg_decoder.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    epoch_num = 0
    dice_loss = DC(2)
    for (inp, target, idx) in metric_logger.log_every(val_loader, 20, header):
        epoch_num += 1
        # move to gpu
        inp = inp.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # forward
        H, W = target.size(1), target.size(2)
        x_tokens_list = model(inp, is_training=True)
        
        output_en = x_tokens_list["x_norm_patchtokens"]

        output = rearrange(output_en, "b (h w) (p1 p2 c) -> b c (h p1) (w p2)", 
                    p1 = 1, p2 = 1, 
                    h = H // 14, w = W // 14, 
                    c = 384)
        output = seg_decoder(output)
        output = F.interpolate(output, size=(H, W), mode="bilinear")
        # loss = nn.CrossEntropyLoss()(output, target)
        loss = nn.CrossEntropyLoss(#weight=None, 
                                   reduction='mean', weight = torch.Tensor([0.1, 10]).cuda(non_blocking=True))(output, target)
        dice = 1 - dice_loss(output, target.unsqueeze(1))

        #save images for visualization
        # fname = os.path.join(args.output_dir, "pred_" + str(epoch_num) + ".png")
        # plt.imsave(fname=fname, arr=preds[0].cpu().numpy(), format='png', cmap='gray')
        # print(f"{fname} saved.")
        
        acc = (torch.max(output, 1)[1] == target).float().mean()

        probs = torch.softmax(output, dim=1)
        _, preds = torch.max(probs, dim=1)
        preds = preds.cpu().detach().numpy()
        target_n = target.cpu().detach().numpy()
        iou_c = ch_iou(target_n, preds)
        # iou_c = torch.from_numpy([iou_c])
        iou_i = isi_iou(target_n, preds)
    

        batch_size = inp.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc.item(), n=batch_size)
        metric_logger.meters['dice'].update(dice.item(), n=batch_size)
        metric_logger.meters['ch_iou'].update(iou_c, n=batch_size)
        metric_logger.meters['isi_iou'].update(iou_i, n=batch_size)

    print('* Acc@1 {top1.global_avg:.3f} loss {losses.global_avg:.3f} Dice {dice.global_avg:.3f} Ch_iou {ch_iou.global_avg:.3f} ISI_iou {isi_iou.global_avg:.3f}'
          .format(top1=metric_logger.acc1, losses=metric_logger.loss, dice=metric_logger.meters['dice'], 
           ch_iou=metric_logger.meters['ch_iou'], isi_iou=metric_logger.meters['isi_iou']))
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}



class Decoder2D(nn.Module):
    def __init__(self, in_channels, out_channels, features=[256, 128, 64]):
        super().__init__()
        self.decoder_1 = nn.Sequential(
                    nn.Conv2d(in_channels, features[0], 3, padding=1),
                    nn.BatchNorm2d(features[0]),
                    nn.ReLU(inplace=True),
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
                )
        self.decoder_2 = nn.Sequential(
                    nn.Conv2d(features[0], features[1], 3, padding=1),
                    nn.BatchNorm2d(features[1]),
                    nn.ReLU(inplace=True),
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
                )
        self.decoder_3 = nn.Sequential(
            nn.Conv2d(features[1], features[2], 3, padding=1),
            nn.BatchNorm2d(features[2]),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        )
        # self.decoder_4 = nn.Sequential(
        #     nn.Conv2d(features[2], features[3], 3, padding=1),
        #     nn.BatchNorm2d(features[3]),
        #     nn.ReLU(inplace=True),
        #     nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        # )

        self.final_out = nn.Conv2d(features[-1], out_channels, 3, padding=1)

    def forward(self, x):
        x = self.decoder_1(x)
        x = self.decoder_2(x)
        x = self.decoder_3(x)
        # x = self.decoder_4(x)
        x = self.final_out(x)
        return x


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Evaluation with semantic segmentation on RobustMIS2019')
    parser.add_argument('--n_last_blocks', default=4, type=int, help="""Concatenate [CLS] tokens
        for the `n` last blocks. We use `n=4` when evaluating ViT-Small and `n=1` with ViT-Base.""")
    parser.add_argument('--avgpool_patchtokens', default=False, type=utils.bool_flag,
        help="""Whether ot not to concatenate the global average pooled features to the [CLS] token.
        We typically set this to False for ViT-Small and to True with ViT-Base.""")
    parser.add_argument('--arch', default='vit_small', type=str, help='Architecture')
    parser.add_argument('--patch_size', default=16, type=int, help='Patch resolution of the model.')
    parser.add_argument('--imsize', default=224, type=int, help='Image size')
    parser.add_argument("--checkpoint_key", default="teacher", type=str, help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs of training.')
    parser.add_argument("--lr", default=0.01, type=float, help="""Learning rate at the beginning of
        training (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.
        We recommend tweaking the LR depending on the checkpoint evaluated.""")
    parser.add_argument('--batch_size_per_gpu', default=16, type=int, help='Per-GPU batch-size')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    parser.add_argument('--data_path', default='/path/to/imagenet/', type=str)
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument('--val_freq', default=1, type=int, help="Epoch frequency for validation.")
    parser.add_argument('--output_dir', default=".", help='Path to save logs and checkpoints')
    parser.add_argument("--opts", default=[], nargs=argparse.REMAINDER, help="Additional configuration options")
    parser.add_argument('--num_labels', default=1000, type=int, help='Number of labels for linear classifier')
    parser.add_argument('--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
    parser.add_argument("--config_file", type=str, help="Model configuration file")
    parser.add_argument("--pretrained_weights", type=str, help="Pretrained model weights")
    parser.add_argument("--cross_test_path", type=str, help="Cross testing dataset path")
    args = parser.parse_args()
    eval_linear(args)
