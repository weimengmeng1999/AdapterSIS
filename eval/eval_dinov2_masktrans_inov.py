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

# from mmcv.cnn import build_norm_layer

class Robomis(torch.utils.data.Dataset):
    def __init__(self, dir_main, split, transform = None, imsize=None):
        super(Robomis, self).__init__()
        self.transform = transform
        # self.mask_transform=mask_transform
        self.imsize = imsize
        self.img_files = glob.glob(os.path.join(dir_main,'images',split,'*.png'))
        self.mask_files = []
        for img_path in self.img_files:
             self.mask_files.append(os.path.join(dir_main, 'annotations', split, os.path.basename(img_path)))

    def __getitem__(self, index):
        img_path = self.img_files[index]
        mask_path = self.mask_files[index]
        with open(img_path, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
        with open(mask_path, 'rb') as f:
            mask = Image.open(f)
            mask = mask.point(lambda x: 1 if x > 0 else 0, mode='1')
            # mask = mask.convert('L')  # or mask = mask.convert('1')
        if self.imsize is not None:
            img = img.resize((self.imsize, self.imsize), resample=Image.BILINEAR)
            mask = mask.resize((self.imsize, self.imsize), resample=Image.NEAREST)
        if self.transform is not None:
            # mat, mat_inv = self.getTransformMat(self.imsize, True)
            img_np = np.array(img).astype(np.uint8)
            mask_np = np.array(mask).astype(np.uint8)
            transformed = self.transform(image=img_np, mask=mask_np)

            # Access the transformed image and mask
            # trans_img = transformed["image"]
            trans_img = torch.from_numpy(transformed['image'].transpose(2, 0, 1)) / 255.0
            trans_mask = torch.from_numpy(transformed["mask"])
        else:
            trans_img = img
            trans_mask = mask
        return trans_img, trans_mask.long(), index
    
    def __len__(self):
        return len(self.img_files)

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


def eval_linear(args):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    # ============ building network ... ============
    cfg = get_cfg_from_args(args)
    model = build_model_for_eval(cfg, args.pretrained_weights)
    autocast_dtype = get_autocast_dtype(cfg)
    # model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')



    n_last_blocks_list = [1, 4]
    n_last_blocks = max(n_last_blocks_list)
    autocast_ctx = partial(torch.cuda.amp.autocast, enabled=True, dtype=autocast_dtype)
    feature_model = ModelWithIntermediateLayers(model, n_last_blocks, autocast_ctx)
    # sample_output = feature_model(train_dataset[0][0].unsqueeze(0).cuda())

    # load weights to evaluate
    # utils.load_pretrained_weights(model, args.pretrained_weights, args.checkpoint_key, args.arch, args.patch_size)
    # print(f"Model {args.arch} built.")
    # side_encoder = CNNEncoder(in_channels=3, features=[64, 128, 256])
    # side_encoder = side_encoder.cuda()
    # side_encoder = nn.parallel.DistributedDataParallel(side_encoder, device_ids=[args.gpu])

    dim=1536
    seg_decoder=MaskTransformer(n_cls=2, patch_size=14, d_encoder=dim,
                                n_layers=2, d_ff=4*dim, d_model=dim,
                                n_heads=dim // 64, drop_path_rate= 0.0,
                                dropout= 0.1) #d_model
    seg_decoder = seg_decoder.cuda()
    seg_decoder = nn.parallel.DistributedDataParallel(seg_decoder,device_ids=[args.gpu])

    # ============ preparing data ... ============
    val_transform = A.Compose([
        A.Resize(588, 588, interpolation=Image.BICUBIC),
        # A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

    dataset_val = Robomis(args.data_path, split="validation", transform = val_transform, imsize=args.imsize)
    val_loader = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # if args.evaluate:
    #     utils.load_pretrained_linear_weights(seg_decoder, args.arch, args.patch_size)
    #     test_stats = validate_network(val_loader, model, seg_decoder, side_encoder, fusion_model, args.n_last_blocks, args.avgpool_patchtokens)
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

        # train_stats = train(feature_model, seg_decoder, side_encoder, fusion_model, fusion_model_2, fusion_model_3, fusion_model_4, optimizer, train_loader, epoch, args.n_last_blocks, args.avgpool_patchtokens)
        train_stats = train(feature_model, seg_decoder, optimizer, train_loader, epoch, args.n_last_blocks, args.avgpool_patchtokens)

        scheduler.step()

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch}
        if epoch % args.val_freq == 0 or epoch == args.epochs - 1:
            # test_stats = validate_network(val_loader, feature_model, seg_decoder, side_encoder, fusion_model, fusion_model_2, fusion_model_3, fusion_model_4, args.n_last_blocks, args.avgpool_patchtokens)
            test_stats = validate_network(val_loader, feature_model, seg_decoder, args.n_last_blocks, args.avgpool_patchtokens)           
            print(f"Accuracy at epoch {epoch} of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
            best_acc = max(best_acc, test_stats["acc1"])
            print(f'Max accuracy so far: {best_acc:.2f}%')
            log_stats = {**{k: v for k, v in log_stats.items()},
                         **{f'test_{k}': v for k, v in test_stats.items()}}
        if utils.is_main_process():
            with (Path(args.output_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
            # save_dict = {
            #     "epoch": epoch + 1,
            #     "state_dict": seg_decoder.state_dict(),
            #     "state_dict_sdencoder": side_encoder.state_dict(),
            #     "state_dict_fusion": fusion_model.state_dict(),
            #     "optimizer": optimizer.state_dict(),
            #     "scheduler": scheduler.state_dict(),
            #     "best_acc": best_acc,
            # }
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


def train(model, seg_decoder, optimizer, loader, epoch, n, avgpool):
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
        # sideout1, sideout2, sideout3 = side_encoder(inp)
        # print(sideout3.shape)
        with torch.no_grad():
          x_tokens_list = model(inp)

          intermediate_output = x_tokens_list[-n:]
          output_en = torch.cat([outputs for outputs, _ in intermediate_output], dim=-1)
        # print(output_en.size())
        output = seg_decoder(output_en, (H, W))
        output = F.interpolate(output, size=(H, W), mode="bilinear")
        # output_last = fusion_model(sideout3, output_last, output_last_2, output_last_3, output_last_4)
        # output_last = output_last + output_last_2 + output_last_3 + output_last_4
        # output_en_f = torch.cat((output_en, output_last), dim=1)
        # output_last_2 = fusion_model_2(output_last_2, sideout3)
        # output_last_3 = fusion_model_3(output_last_3, sideout2)
        # output_last_4 = fusion_model_4(output_last_4, sideout1)
        # output = seg_decoder(output_last, output_last_4, output_last_3, output_last_2)
        # output = seg_decoder(output_last, output_last_2, output_last_3, output_last_4)
        # print(output_en.size())
        # output = F.interpolate(output, size=(H, W), mode="bilinear")

        # compute cross entropy loss
        # loss_ce = nn.CrossEntropyLoss()(output, target)
        # loss_ce= nn.CrossEntropyLoss(#weight=None, 
        #                              reduction='mean', weight = torch.Tensor([0.1, 10]).cuda(non_blocking=True))(output, target)
        # loss_ce= nn.CrossEntropyLoss(ignore_index = 0, reduction='mean')(output, target)
        # loss_ce = F.binary_cross_entropy_with_logits(output, target)
        loss_dce = dice_loss(output, target.unsqueeze(1))
        # loss =  loss_ce + loss_dce
        # loss = loss_ce
        loss = loss_dce
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
        # with torch.no_grad():
        #   x_tokens_list = model(inp)
        # #   intermediate_output = x_tokens_list[-n:-1]
        #   intermediate_output_last = x_tokens_list[-1:]
        # #   print(x_tokens_list[-1:].shape)
        #   output_last = torch.cat([outputs for outputs, _ in intermediate_output_last], dim=-1)
        # #   output_en = torch.cat([outputs for outputs, _ in intermediate_output], dim=-1)
        # #   output_en = rearrange(output_en, "b (h w) (p1 p2 c) -> b c (h p1) (w p2)", 
        # #             p1 = 1, p2 = 1, 
        # #             h = H // 14, w = W // 14, 
        # #             c = 1152)
        #   output_last = rearrange(output_last, "b (h w) (p1 p2 c) -> b c (h p1) (w p2)", 
        #             p1 = 1, p2 = 1, 
        #             h = H // 14, w = W // 14, 
        #             c = 384)
        # output_last = fusion_model(sideout3, output_last)
        # # output_en_f = torch.cat((output_en, output_last), dim=1)
        # output = seg_decoder(output_last, sideout1, sideout2, sideout3)
        with torch.no_grad():
          x_tokens_list = model(inp)

          intermediate_output = x_tokens_list[-n:]
          output_en = torch.cat([outputs for outputs, _ in intermediate_output], dim=-1)
        # print(output_en.size())
        output = seg_decoder(output_en, (H, W))
        output = F.interpolate(output, size=(H, W), mode="bilinear")
        # output_last = fusion_model(sideout3, output_last, output_last_2, output_last_3, output_last_4)
        # output_last = output_last + output_last_2 + output_last_3 + output_last_4
        # output_en_f = torch.cat((output_en, output_last), dim=1)
        # output_last_2 = fusion_model_2(output_last_2, sideout3)
        # output_last_3 = fusion_model_3(output_last_3, sideout2)
        # output_last_4 = fusion_model_4(output_last_4, sideout1)
        # output = seg_decoder(output_last, output_last_2, output_last_3, output_last_4)
        # output = F.interpolate(output, size=(H, W), mode="bilinear")
        # loss = nn.CrossEntropyLoss()(output, target)
        loss = nn.CrossEntropyLoss(#weight=None, 
                                   reduction='mean', weight = torch.Tensor([0.1, 10]).cuda(non_blocking=True))(output, target)
        dice = 1 - dice_loss(output, target.unsqueeze(1))

        #save images for visualization
        # fname = os.path.join(args.output_dir, "pred_" + str(epoch_num) + ".png")
        # plt.imsave(fname=fname, arr=preds[0].cpu().numpy(), format='png', cmap='gray')
        # print(f"{fname} saved.")

        #save images for visualization
        ########################################################
        for i in range(output.shape[0]):
            pred_i = torch.argmax(output[i], dim=0)
            fname = os.path.join(args.output_dir, "pred", "pred_" + str(epoch_num)+ "_" + str(i) + ".png")
            result_img = draw_segmentation_masks((255*inp[i]).cpu().to(torch.uint8), masks=pred_i.to(torch.bool).cpu(), alpha=.5, colors = "green")
            print(result_img.size())
            result_img = torch.transpose(result_img, 0, 2)
            plt.imsave(fname=fname, arr=result_img.numpy(), format='png')
            print(f"{fname} saved.")
        
        acc = (torch.max(output, 1)[1] == target).float().mean()

        batch_size = inp.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc.item(), n=batch_size)
        metric_logger.meters['dice'].update(dice.item(), n=batch_size)

    print('* Acc@1 {top1.global_avg:.3f} loss {losses.global_avg:.3f} Dice {dice.global_avg:.3f}'
          .format(top1=metric_logger.acc1, losses=metric_logger.loss, dice=metric_logger.meters['dice']))
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


class CNNEncoder(nn.Module):
    def __init__(self, in_channels, features=[64, 128, 256]):
        super().__init__()
        self.sin_conv_1 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels, features[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(features[0]),
            nn.ReLU(inplace=True),
        )

        self.sin_conv_2 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(features[0], features[1], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(features[1]),
            nn.ReLU(inplace=True),
        )

        self.sin_conv_3 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(features[1], features[2], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(features[2]),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        c1 = self.sin_conv_1(x)
        c2 = self.sin_conv_2(c1)
        c3 = self.sin_conv_3(c2)
        return c1, c2, c3


class FusionModel(nn.Module):
    def __init__(self):
        super().__init__()
        # self.flatten = nn.Flatten()
        self.conv = nn.Conv2d(256, 384, kernel_size=1)
        self.resize = nn.Upsample(size=(42, 42), mode='bilinear', align_corners=False)
        self.activation = nn.ReLU()
    
    def forward(self, x, x1, x2, x3, x4):
        x = self.conv(x)
        x = self.resize(x)
        x = x + x1 + x2 + x3 + x4
        x = self.activation(x)
        return x

def init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)



class MaskTransformer(nn.Module):
    def __init__(
        self,
        n_cls,
        patch_size,
        d_encoder,
        n_layers,
        n_heads,
        d_model,
        d_ff,
        drop_path_rate,
        dropout,
    ):
        super().__init__()
        self.d_encoder = d_encoder
        self.patch_size = patch_size
        self.n_layers = n_layers
        self.n_cls = n_cls
        self.d_model = d_model
        self.d_ff = d_ff
        self.scale = d_model ** -0.5

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, n_layers)]
        self.blocks = nn.ModuleList(
            [Block(d_model, n_heads, d_ff, dropout, dpr[i]) for i in range(n_layers)]
        )

        self.cls_emb = nn.Parameter(torch.randn(1, n_cls, d_model))
        self.proj_dec = nn.Linear(d_encoder, d_model)

        self.proj_patch = nn.Parameter(self.scale * torch.randn(d_model, d_model))
        self.proj_classes = nn.Parameter(self.scale * torch.randn(d_model, d_model))

        self.decoder_norm = nn.LayerNorm(d_model)
        self.mask_norm = nn.LayerNorm(n_cls)

        self.apply(init_weights)
        trunc_normal_(self.cls_emb, std=0.02)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"cls_emb"}

    def forward(self, x, im_size):
        H, W = im_size
        GS = H // self.patch_size

        x = self.proj_dec(x)
        cls_emb = self.cls_emb.expand(x.size(0), -1, -1)
        x = torch.cat((x, cls_emb), 1)
        for blk in self.blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        patches, cls_seg_feat = x[:, : -self.n_cls], x[:, -self.n_cls :]
        patches = patches @ self.proj_patch
        cls_seg_feat = cls_seg_feat @ self.proj_classes

        patches = patches / patches.norm(dim=-1, keepdim=True)
        cls_seg_feat = cls_seg_feat / cls_seg_feat.norm(dim=-1, keepdim=True)

        masks = patches @ cls_seg_feat.transpose(1, 2)
        masks = self.mask_norm(masks)
        masks = rearrange(masks, "b (h w) n -> b n h w", h=int(GS))

        return masks

    def get_attention_map(self, x, layer_id):
        if layer_id >= self.n_layers or layer_id < 0:
            raise ValueError(
                f"Provided layer_id: {layer_id} is not valid. 0 <= {layer_id} < {self.n_layers}."
            )
        x = self.proj_dec(x)
        cls_emb = self.cls_emb.expand(x.size(0), -1, -1)
        x = torch.cat((x, cls_emb), 1)
        for i, blk in enumerate(self.blocks):
            if i < layer_id:
                x = blk(x)
            else:
                return blk(x, return_attention=True)


    

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
    eval_linear(args)
