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
from torchvision.utils import draw_segmentation_masks
from PIL import Image, ImageFile
import numpy as np
import math 
from tqdm import tqdm

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

from functools import partial
from dinov2.eval.setup import build_model_for_eval, get_autocast_dtype
from dinov2.utils.config import get_cfg_from_args
from dinov2.eval.utils import ModelWithIntermediateLayers

from SegLoss.losses_pytorch.dice_loss import TverskyLoss, SoftDiceLoss, DC_and_CE_loss
# from SegLoss.losses_pytorch.dice_loss import *
from tools.dataset import EndoVis2017
from segloss.dice import DC
from segloss.iou_multi import *
from backbones.decoders import DecoderMLA #.....decoder structure

from backbones.adapter_blocks import CAViT, CACNN

# from mmcv.cnn import build_norm_layer
from ops.modules import MSDeformAttn


def train_seg(args):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    # ============ building network ... ============

    cfg = get_cfg_from_args(args)
    model = build_model_for_eval(cfg, args.pretrained_weights)
    autocast_dtype = get_autocast_dtype(cfg)



    n_last_blocks_list = [1, 4]
    n_last_blocks = max(n_last_blocks_list)
    autocast_ctx = partial(torch.cuda.amp.autocast, enabled=True, dtype=autocast_dtype)
    feature_model = ModelWithIntermediateLayers(model, n_last_blocks, autocast_ctx)

    backbone_encoder = SpatialPriorModule()
    backbone_encoder = backbone_encoder.cuda()
    backbone_encoder = nn.parallel.DistributedDataParallel(backbone_encoder,device_ids=[args.gpu])

    cross_vit = CAViT(
            dim=1024,
            n_levels=3,
            num_heads=8,
            init_values=0.0,
            n_points=4,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            deform_ratio=1.0,
            with_cp=False)
    cross_vit = cross_vit.cuda()
    cross_vit = nn.parallel.DistributedDataParallel(cross_vit,device_ids=[args.gpu])

    cross_cnn = CACNN(
                        dim=1024,
                        n_levels=1,
                        num_heads=8,
                        n_points=4,
                        norm_layer=partial(nn.LayerNorm, eps=1e-6),
                        with_cffn=True,
                        cffn_ratio=0.25,
                        deform_ratio=1.0,
                        drop=0.0,
                        drop_path=0.0,
                        with_cp=False
                    )
    cross_cnn = cross_cnn.cuda()
    cross_cnn = nn.parallel.DistributedDataParallel(cross_cnn,device_ids=[args.gpu])

    seg_decoder = DecoderMLA(num_classes=2)
    seg_decoder = seg_decoder.cuda()
    seg_decoder = nn.parallel.DistributedDataParallel(seg_decoder,device_ids=[args.gpu])

    # ============ preparing data ... ============
    val_transform = A.Compose([
        A.Resize(588, 588, interpolation=Image.BICUBIC),
        # A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

    dataset_val = EndoVis2017(args.cross_test_path, split="Test", transform = val_transform, imsize=args.imsize, task = "multi")
    val_loader = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # if args.evaluate:
    #     utils.load_pretrained_linear_weights(seg_decoder, args.arch, args.patch_size)
    #     # test_stats = validate_network(val_loader, model, seg_decoder, side_encoder, fusion_model, args.n_last_blocks, args.avgpool_patchtokens)
    #     test_stats = validate_network(val_loader, feature_model, seg_decoder, args.n_last_blocks, args.avgpool_patchtokens)
    #     print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
    #     return

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

    dataset_train = EndoVis2017(args.data_path, split="Train", transform = train_transform,
                            imsize=args.imsize, task = "multi")
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
        # seg_decoder.load_state_dict(checkpoint['state_dict'], strict=False)

        # test_stats = validate_network(val_loader, model, seg_decoder, args.n_last_blocks, args.avgpool_patchtokens)
        test_stats = validate_network(val_loader, feature_model, seg_decoder, args.n_last_blocks, args.avgpool_patchtokens)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        return

    for epoch in range(start_epoch, args.epochs):
        train_loader.sampler.set_epoch(epoch)

        # train_stats = train(feature_model, seg_decoder, side_encoder, fusion_model, fusion_model_2, fusion_model_3, fusion_model_4, optimizer, train_loader, epoch, args.n_last_blocks, args.avgpool_patchtokens)
        train_stats = train(model, feature_model, backbone_encoder, cross_vit, cross_cnn, seg_decoder, optimizer, train_loader, epoch, args.n_last_blocks, args.avgpool_patchtokens)

        scheduler.step()

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch}
        if epoch % args.val_freq == 0 or epoch == args.epochs - 1:
            # test_stats = validate_network(val_loader, feature_model, seg_decoder, side_encoder, fusion_model, fusion_model_2, fusion_model_3, fusion_model_4, args.n_last_blocks, args.avgpool_patchtokens)
            test_stats = validate_network(val_loader, model, feature_model, backbone_encoder, cross_vit, cross_cnn, seg_decoder, args.n_last_blocks, args.avgpool_patchtokens)           
            print(f"Accuracy at epoch {epoch} of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
            best_acc = max(best_acc, test_stats["acc1"])
            print(f'Max accuracy so far: {best_acc:.2f}%')
            log_stats = {**{k: v for k, v in log_stats.items()},
                         **{f'test_{k}': v for k, v in test_stats.items()}}
        if utils.is_main_process():
            with (Path(args.output_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

            save_dict = {
                "epoch": epoch + 1,
                "state_dict": seg_decoder.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_acc": best_acc,
            }
            torch.save(save_dict, os.path.join(args.output_dir, "checkpoint.pth.tar"))
    print("Training of the supervised linear classifier on frozen features completed.\n"
                "Top-1 test accuracy: {acc:.1f}".format(acc=best_acc))


def train(model, feature_model, backbone_encoder, cross_vit, cross_cnn, seg_decoder, optimizer, loader, epoch, n, avgpool):
    # side_encoder.train()
    seg_decoder.train()
    # fusion_model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    dice_loss = DC(2)
    for (inp, target, idx) in tqdm(metric_logger.log_every(loader, 20, header)):
        # move to gpu
        inp = inp.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        # forward
        H, W = target.size(1), target.size(2)

        deform_inputs1, deform_inputs2 = deform_inputs(inp, 14)
        H_c, W_c = inp.shape[2] // 16, inp.shape[3] // 16
        level_embed = nn.Parameter(torch.zeros(3, 1024)).cuda()

        c1,c2,c3,c4 = backbone_encoder(inp)
        c2 = c2 + level_embed[0]
        c3 = c3 + level_embed[1]
        c4 = c4 + level_embed[2]
        c = torch.cat([c2, c3, c4], dim=1)

    
        with torch.no_grad():
            x = model.patch_embed(inp)
            for idx, blk in enumerate(model.blocks[0 : -3]):
                x = blk(x)
                # cls, x = (x[:,:1,], x[:,1:,],)

        x = cross_vit(
            query=x,
            reference_points=deform_inputs1[0],
            feat=c,
            spatial_shapes=deform_inputs1[1],
            level_start_index=deform_inputs1[2],
        )

        # x = torch.cat((cls, x), dim=1)
        output_last_4 = x
########################################################################################
        with torch.no_grad():
            for idx, blk in enumerate(model.blocks[-3 : -2]):
                x = blk(x)
                # cls, x = (x[:,:1,], x[:,1:,],)
    
        c=cross_cnn(query=c,
            reference_points=deform_inputs2[0],
            feat=x,
            spatial_shapes=deform_inputs2[1],
            level_start_index=deform_inputs2[2],
            H=H_c,
            W=W_c)
        x = cross_vit(
            query=x,
            reference_points=deform_inputs1[0],
            feat=c,
            spatial_shapes=deform_inputs1[1],
            level_start_index=deform_inputs1[2],
        )
        # x = torch.cat((cls, x), dim=1)
        output_last_3 = x
########################################################################################
        with torch.no_grad():
            for idx, blk in enumerate(model.blocks[-2 : -1]):
                x = blk(x)
                # cls, x = (x[:,:1,], x[:,1:,],)
    
        c=cross_cnn(query=c,
            reference_points=deform_inputs2[0],
            feat=x,
            spatial_shapes=deform_inputs2[1],
            level_start_index=deform_inputs2[2],
            H=H_c,
            W=W_c)
        x = cross_vit(
            query=x,
            reference_points=deform_inputs1[0],
            feat=c,
            spatial_shapes=deform_inputs1[1],
            level_start_index=deform_inputs1[2],
        )
        # x = torch.cat((cls, x), dim=1)
        output_last_2 = x
########################################################################################
        with torch.no_grad():
            for idx, blk in enumerate(model.blocks[-2 : -1]):
                x = blk(x)
                # cls, x = (x[:,:1,], x[:,1:,],)
    
        c=cross_cnn(query=c,
            reference_points=deform_inputs2[0],
            feat=x,
            spatial_shapes=deform_inputs2[1],
            level_start_index=deform_inputs2[2],
            H=H_c,
            W=W_c)
        x = cross_vit(
            query=x,
            reference_points=deform_inputs1[0],
            feat=c,
            spatial_shapes=deform_inputs1[1],
            level_start_index=deform_inputs1[2],
        )
        # x = torch.cat((cls, x), dim=1)
        output_last = x

        with torch.no_grad():
          x_tokens_list = feature_model(inp)
          intermediate_output_last = x_tokens_list[-1:]

          output_last_vit = torch.cat([outputs for outputs, _ in intermediate_output_last], dim=-1)
          output_last = output_last_vit + output_last

          output_last = rearrange(output_last, "b (h w) (p1 p2 c) -> b c (h p1) (w p2)", 
                    p1 = 1, p2 = 1, 
                    h = H // 14, w = W // 14, 
                    c = 1024)
          output_last_2 = rearrange(output_last_2, "b (h w) (p1 p2 c) -> b c (h p1) (w p2)", 
                    p1 = 1, p2 = 1, 
                    h = H // 14, w = W // 14, 
                    c = 1024)
          output_last_3 = rearrange(output_last_3, "b (h w) (p1 p2 c) -> b c (h p1) (w p2)", 
                    p1 = 1, p2 = 1, 
                    h = H // 14, w = W // 14, 
                    c = 1024)
          output_last_4 = rearrange(output_last_4, "b (h w) (p1 p2 c) -> b c (h p1) (w p2)", 
                    p1 = 1, p2 = 1, 
                    h = H // 14, w = W // 14, 
                    c = 1024)
        output = seg_decoder(output_last, output_last_2, output_last_3, output_last_4)

        output = nn.Softmax(1)(output)
        # loss_tky = SoftDiceLoss()(output, target.unsqueeze(1))
        loss = iou_loss(output, target)
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
def validate_network(val_loader, model, feature_model, backbone_encoder, cross_vit, cross_cnn, seg_decoder, n, avgpool):
    # side_encoder.eval()
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
        # sideout1, sideout2, sideout3 = side_encoder(inp)
        # print(sideout3.shape)
        deform_inputs1, deform_inputs2 = deform_inputs(inp, 14)
        H_c, W_c = inp.shape[2] // 16, inp.shape[3] // 16
        level_embed = nn.Parameter(torch.zeros(3, 1024)).cuda()

        c1,c2,c3,c4 = backbone_encoder(inp)
        c2 = c2 + level_embed[0]
        c3 = c3 + level_embed[1]
        c4 = c4 + level_embed[2]
        c = torch.cat([c2, c3, c4], dim=1)

    
        with torch.no_grad():
            x = model.patch_embed(inp)
            for idx, blk in enumerate(model.blocks[0 : -3]):
                x = blk(x)
                # cls, x = (x[:,:1,], x[:,1:,],)

        x = cross_vit(
            query=x,
            reference_points=deform_inputs1[0],
            feat=c,
            spatial_shapes=deform_inputs1[1],
            level_start_index=deform_inputs1[2],
        )

        # x = torch.cat((cls, x), dim=1)
        output_last_4 = x
########################################################################################
        with torch.no_grad():
            for idx, blk in enumerate(model.blocks[-3 : -2]):
                x = blk(x)
                # cls, x = (x[:,:1,], x[:,1:,],)
    
        c=cross_cnn(query=c,
            reference_points=deform_inputs2[0],
            feat=x,
            spatial_shapes=deform_inputs2[1],
            level_start_index=deform_inputs2[2],
            H=H_c,
            W=W_c)
        x = cross_vit(
            query=x,
            reference_points=deform_inputs1[0],
            feat=c,
            spatial_shapes=deform_inputs1[1],
            level_start_index=deform_inputs1[2],
        )
        # x = torch.cat((cls, x), dim=1)
        output_last_3 = x
########################################################################################
        with torch.no_grad():
            for idx, blk in enumerate(model.blocks[-2 : -1]):
                x = blk(x)
                # cls, x = (x[:,:1,], x[:,1:,],)
    
        c=cross_cnn(query=c,
            reference_points=deform_inputs2[0],
            feat=x,
            spatial_shapes=deform_inputs2[1],
            level_start_index=deform_inputs2[2],
            H=H_c,
            W=W_c)
        x = cross_vit(
            query=x,
            reference_points=deform_inputs1[0],
            feat=c,
            spatial_shapes=deform_inputs1[1],
            level_start_index=deform_inputs1[2],
        )
        # x = torch.cat((cls, x), dim=1)
        output_last_2 = x
########################################################################################
        with torch.no_grad():
            for idx, blk in enumerate(model.blocks[-2 : -1]):
                x = blk(x)
                # cls, x = (x[:,:1,], x[:,1:,],)
    
        c=cross_cnn(query=c,
            reference_points=deform_inputs2[0],
            feat=x,
            spatial_shapes=deform_inputs2[1],
            level_start_index=deform_inputs2[2],
            H=H_c,
            W=W_c)
        x = cross_vit(
            query=x,
            reference_points=deform_inputs1[0],
            feat=c,
            spatial_shapes=deform_inputs1[1],
            level_start_index=deform_inputs1[2],
        )
        # x = torch.cat((cls, x), dim=1)
        output_last = x

        with torch.no_grad():
          x_tokens_list = feature_model(inp)
        #   intermediate_output = x_tokens_list[-n:-1]
          intermediate_output_last = x_tokens_list[-1:]
        #   intermediate_output_last_2 = x_tokens_list[-2:-1]
        #   intermediate_output_last_3 = x_tokens_list[-3:-2]
        #   intermediate_output_last_4 = x_tokens_list[-4:-3]
        #   print(x_tokens_list[-1:].shape)
          output_last_vit = torch.cat([outputs for outputs, _ in intermediate_output_last], dim=-1)
          output_last = output_last_vit + output_last
          
          output_last = rearrange(output_last, "b (h w) (p1 p2 c) -> b c (h p1) (w p2)", 
                    p1 = 1, p2 = 1, 
                    h = H // 14, w = W // 14, 
                    c = 1024)
          output_last_2 = rearrange(output_last_2, "b (h w) (p1 p2 c) -> b c (h p1) (w p2)", 
                    p1 = 1, p2 = 1, 
                    h = H // 14, w = W // 14, 
                    c = 1024)
          output_last_3 = rearrange(output_last_3, "b (h w) (p1 p2 c) -> b c (h p1) (w p2)", 
                    p1 = 1, p2 = 1, 
                    h = H // 14, w = W // 14, 
                    c = 1024)
          output_last_4 = rearrange(output_last_4, "b (h w) (p1 p2 c) -> b c (h p1) (w p2)", 
                    p1 = 1, p2 = 1, 
                    h = H // 14, w = W // 14, 
                    c = 1024)

        output = seg_decoder(output_last, output_last_2, output_last_3, output_last_4)
        # output = F.interpolate(output, size=(H, W), mode="bilinear")
        # loss = nn.CrossEntropyLoss()(output, target)
        loss = nn.CrossEntropyLoss(#weight=None, 
                                   reduction='mean', weight = torch.Tensor([0.1, 10]).cuda(non_blocking=True))(output, target)
        dice = 1 - dice_loss(output, target.unsqueeze(1))

        #save images for visualization
        ########################################################
        # for i in range(output.shape[0]):
        #     pred_i = torch.argmax(output[i], dim=0)
        #     fname = os.path.join(args.output_dir, "pred_img", "pred_" + str(epoch_num)+ "_" + str(i) + ".png")
        #     fname_gt = os.path.join(args.output_dir, "pred_img", "gt_" + str(epoch_num)+ "_" + str(i) + ".png")
        #     fname_msk = os.path.join(args.output_dir, "pred_img", "gt_msk_" + str(epoch_num)+ "_" + str(i) + ".png")
        #     result_img = draw_segmentation_masks((255*inp[i]).cpu().to(torch.uint8), masks=pred_i.to(torch.bool).cpu(), alpha=.5, colors = "green")
        #     # result_gt = draw_segmentation_masks((255*inp[i]).cpu().to(torch.uint8), masks=target.unsqueeze(1)[i].to(torch.bool).cpu(), alpha=.5, colors = "green")
        #     result_gt = 255*inp[i].cpu().to(torch.uint8)
        #     print(result_gt.size())
        #     result_msk = 255*pred_i.cpu().numpy()
        #     # result_msk = np.stack((result_msk,) * 3, axis=-1)
        #     print(result_msk.size())
        #     print(result_img.size())
        #     result_img = torch.transpose(result_img, 0, 2)
        #     result_gt = torch.transpose(result_gt, 0, 2)
        #     plt.imsave(fname=fname, arr=result_img.numpy(), format='png')
        #     plt.imsave(fname=fname_gt, arr=result_gt.numpy(), format='png')
        #     plt.imsave(fname=fname_msk, arr=result_msk, format='png')
        #     print(f"{fname} saved.")
        
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
    args = parser.parse_args()
    train_seg(args)
