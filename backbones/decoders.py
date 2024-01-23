import torch
from torch import nn
import torch.nn.functional as F
from functools import partial
from .unet_parts import DoubleConv, Up, Down, UNet, OutConv

class MLAHead(nn.Module):
    def __init__(self, mla_channels=1024, mlahead_channels=128, norm_cfg=None):
        super(MLAHead, self).__init__()
        self.head2 = nn.Sequential(nn.Conv2d(mla_channels, mlahead_channels, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(mlahead_channels), 
                                   nn.ReLU(),
                                   nn.Conv2d(mlahead_channels, mlahead_channels, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(mlahead_channels), 
                                   nn.ReLU())
        self.head3 = nn.Sequential(nn.Conv2d(mla_channels, mlahead_channels, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(mlahead_channels),  
                                   nn.ReLU(),
                                   nn.Conv2d(mlahead_channels, mlahead_channels, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(mlahead_channels), 
                                   nn.ReLU())
        self.head4 = nn.Sequential(nn.Conv2d(mla_channels, mlahead_channels, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(mlahead_channels), 
                                   nn.ReLU(),
                                   nn.Conv2d(mlahead_channels, mlahead_channels, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(mlahead_channels), 
                                   nn.ReLU())
        self.head5 = nn.Sequential(nn.Conv2d(mla_channels, mlahead_channels, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(mlahead_channels), 
                                   nn.ReLU(),
                                   nn.Conv2d(mlahead_channels, mlahead_channels, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(mlahead_channels), 
                                   nn.ReLU())

    def forward(self, mla_p2, mla_p3, mla_p4, mla_p5):
        # head2 = self.head2(mla_p2)
        head2 = F.interpolate(self.head2(
            mla_p2), 4*mla_p2.shape[-1], mode='bilinear', align_corners=True)
        head3 = F.interpolate(self.head3(
            mla_p3), 4*mla_p3.shape[-1], mode='bilinear', align_corners=True)
        head4 = F.interpolate(self.head4(
            mla_p4), 4*mla_p4.shape[-1], mode='bilinear', align_corners=True)
        head5 = F.interpolate(self.head5(
            mla_p5), 4*mla_p5.shape[-1], mode='bilinear', align_corners=True)
        return torch.cat([head2, head3, head4, head5], dim=1)

class DecoderMLA(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self, img_size=588, mla_channels=1024, mlahead_channels=128,
                 norm_layer=nn.BatchNorm2d, num_classes=2, norm_cfg=None):
        super().__init__()
        self.img_size = img_size
        self.norm_cfg = norm_cfg
        self.mla_channels = mla_channels
        self.BatchNorm = norm_layer
        self.mlahead_channels = mlahead_channels
        self.num_classes = 2

        self.mlahead = MLAHead(mla_channels=self.mla_channels,
                               mlahead_channels=self.mlahead_channels, norm_cfg=self.norm_cfg)
        # self.cls = nn.Conv2d(4 * self.mlahead_channels,
        #                      self.num_classes, 3, padding=1)
        self.cls = nn.Sequential(
                    nn.Conv2d(4 * self.mlahead_channels, 256, 3, padding=1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True)#56 256
                )
        self.cls_1 = nn.Sequential(
                    nn.Conv2d(256, 128, 3, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True) #56 256
                )
        self.cls_2 = nn.Sequential(
                    nn.Conv2d(128, 64, 3, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True) #56 256
                )
        self.cls_3 = nn.Conv2d(64, self.num_classes, 3, padding=1)

    def forward(self, input, input1, input2, input3):
        x = self.mlahead(input, input1, input2, input3)
        x = self.cls(x)
        x = self.cls_1(x)
        x = self.cls_2(x)
        x = self.cls_3(x)
        x = F.interpolate(x, size=self.img_size, mode='bilinear')
        return x

class DecoderSETR(nn.Module):
    def __init__(self, in_channels, out_channels, features=[512, 256, 128, 64]):
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
        self.decoder_4 = nn.Sequential(
            nn.Conv2d(features[2], features[3], 3, padding=1),
            nn.BatchNorm2d(features[3]),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        )

        self.final_out = nn.Conv2d(features[-1], out_channels, 3, padding=1)

    def forward(self, x):
        x = self.decoder_1(x)
        x = self.decoder_2(x)
        x = self.decoder_3(x)
        x = self.decoder_4(x)
        x = self.final_out(x)
        return x

class DecoderSETRF(nn.Module):
    def __init__(self, in_channels, out_channels, features=[512, 256, 128, 64]):
        super().__init__()
        self.decoder_1 = nn.Sequential(
                    nn.Conv2d(in_channels, features[0], 3, padding=1),
                    nn.BatchNorm2d(features[0]),
                    nn.ReLU(inplace=True),
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True) #28 512
                )
        self.decoder_2 = nn.Sequential(
                    nn.Conv2d(features[0], features[1], 3, padding=1),
                    nn.BatchNorm2d(features[1]),
                    nn.ReLU(inplace=True),
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True) #56 256
                )
        self.decoder_3 = nn.Sequential(
            nn.Conv2d(features[1]*2, features[2], 3, padding=1),
            nn.BatchNorm2d(features[2]),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True) #112 128
        )
        self.decoder_4 = nn.Sequential(
            nn.Conv2d(features[2]*2, features[3], 3, padding=1),
            nn.BatchNorm2d(features[3]),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True) # 224 64
        )

        self.final_out = nn.Conv2d(features[-1]*2, out_channels, 3, padding=1) #448 2

    def forward(self, x, c1, c2, c3):
        x = self.decoder_1(x)
        x = self.decoder_2(x)
        diffy = c3.size()[2] - x.size()[2]
        diffx = c3.size()[3] - x.size()[3]
        x = F.pad(x, [diffx // 2, diffx - diffx // 2,
                       diffy // 2, diffy - diffy // 2])
        x = torch.cat([x, c3], dim=1)       
        x = self.decoder_3(x)
        diffy = c2.size()[2] - x.size()[2]
        diffx = c2.size()[3] - x.size()[3]
        x = F.pad(x, [diffx // 2, diffx - diffx // 2,
                       diffy // 2, diffy - diffy // 2])
        x = torch.cat([x, c2], dim=1)
        x = self.decoder_4(x)
        diffy = c1.size()[2] - x.size()[2]
        diffx = c1.size()[3] - x.size()[3]
        x = F.pad(x, [diffx // 2, diffx - diffx // 2,
                       diffy // 2, diffy - diffy // 2])
        x = torch.cat([x, c1], dim=1)      
        x = self.final_out(x)
        return x


    
class FusionModel(nn.Module):
    def __init__(self):
        super().__init__()
        # self.flatten = nn.Flatten()
        self.conv = nn.Conv2d(256, 384, kernel_size=1)
        self.resize = nn.Upsample(size=(42, 42), mode='bilinear', align_corners=False)
        self.activation = nn.ReLU()
    
    def forward(self, x, x1):
        x = self.conv(x)
        x = self.resize(x)
        x = x + x1
        x = self.activation(x)
        return x


class FCUUp(nn.Module):
    """ Transformer patch embeddings -> CNN feature maps
    """

    def __init__(self, inplanes, outplanes, up_stride, act_layer=nn.ReLU,
                 norm_layer=partial(nn.BatchNorm2d, eps=1e-6),):
        super(FCUUp, self).__init__()

        self.up_stride = up_stride
        self.conv_project = nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1, padding=0)
        self.bn = norm_layer(outplanes)
        self.act = act_layer()

    def forward(self, x_r, H, W):
        # B, C, _ = x.shape
        # [N, 197, 384] -> [N, 196, 384] -> [N, 384, 196] -> [N, 384, 14, 14]
        # x_r = x[:, 1:].transpose(1, 2).reshape(B, C, H, W)
        x_r = self.act(self.bn(self.conv_project(x_r)))

        return F.interpolate(x_r, size=(H * self.up_stride, W * self.up_stride))

class ConvBlock(nn.Module):

    def __init__(self, inplanes, outplanes, stride=1, res_conv=False, act_layer=nn.ReLU, groups=1,
                 norm_layer=partial(nn.BatchNorm2d, eps=1e-6), drop_block=None, drop_path=None):
        super(ConvBlock, self).__init__()

        expansion = 4
        med_planes = outplanes // expansion

        self.conv1 = nn.Conv2d(inplanes, med_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = norm_layer(med_planes)
        self.act1 = act_layer(inplace=True)

        self.conv2 = nn.Conv2d(med_planes, med_planes, kernel_size=3, stride=stride, groups=groups, padding=1, bias=False)
        self.bn2 = norm_layer(med_planes)
        self.act2 = act_layer(inplace=True)

        self.conv3 = nn.Conv2d(med_planes, outplanes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = norm_layer(outplanes)
        self.act3 = act_layer(inplace=True)

        if res_conv:
            self.residual_conv = nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=stride, padding=0, bias=False)
            self.residual_bn = norm_layer(outplanes)

        self.res_conv = res_conv
        self.drop_block = drop_block
        self.drop_path = drop_path

    def zero_init_last_bn(self):
        nn.init.zeros_(self.bn3.weight)

    def forward(self, x, x_t=None):
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.act1(x)

        x = self.conv2(x) if x_t is None else self.conv2(x + x_t)
        x = self.bn2(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x2 = self.act2(x)

        x = self.conv3(x2)
        x = self.bn3(x)
        if self.drop_block is not None:
            x = self.drop_block(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        if self.res_conv:
            residual = self.residual_conv(residual)
            residual = self.residual_bn(residual)

        x += residual
        x = self.act3(x)

        return x

class DecoderUNet(nn.Module):
    def __init__(self, n_channels = 3, n_classes = 2, outplanes = 1024, embed_dim=384, dw_stride= 3, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.dw_stride = dw_stride

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))

        expansion = 4
        self.expand_block = FCUUp(inplanes=embed_dim, outplanes=outplanes // expansion, up_stride=dw_stride)

        self.fusion_block = ConvBlock(inplanes=outplanes, outplanes=outplanes, groups=1)

    def forward(self, x, xv):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        _, _, H, W = x5.shape
        xv_r = self.expand_block(xv, H // self.dw_stride, W // self.dw_stride)
        # print(x5.shape)
        # print(xv_r.shape)
        x5 = self.fusion_block(x5, xv_r)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits