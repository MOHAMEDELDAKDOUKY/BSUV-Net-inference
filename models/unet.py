"""
16-layer U-Net model
"""
import torch
import torch.nn as nn
from models.unet_tools import UNetDown, UNetUp, ConvSig, FCNN
from models.vggnet import VGGNet

class unet_vgg16(nn.Module):
    """
    Args:
        inp_ch (int): Number of input channels
        kernel_size (int): Size of the convolutional kernels
        skip (bool, default=True): Use skip connections
    """
    def __init__(self, inp_ch, kernel_size=3, skip=True):
        super().__init__()
        self.skip = skip
        
        self.empty_bg_enc  = VGGNet()
        self.recent_bg_enc = VGGNet()
        self.curr_fr_enc   = VGGNet()
        
        self.dec4 = UNetUp(512, skip*512, 512, 2, batch_norm=True, kernel_size=kernel_size)
        self.dec3 = UNetUp(512, skip*256, 256, 2, batch_norm=True, kernel_size=kernel_size)
        self.dec2 = UNetUp(256, skip*128, 128, 2, batch_norm=True, kernel_size=kernel_size)
        self.dec1 = UNetUp(128, skip*64, 64, 2, batch_norm=True, kernel_size=kernel_size)
        self.out = ConvSig(64)

        self.conv1 = nn.Conv2d(64  * 3, 64,  1, padding="same")
        self.conv2 = nn.Conv2d(128 * 3, 128, 1, padding="same")
        self.conv3 = nn.Conv2d(256 * 3, 256, 1, padding="same")
        self.conv4 = nn.Conv2d(512 * 3, 512, 1, padding="same")
        self.conv5 = nn.Conv2d(512 * 3, 512, 1, padding="same")

    def forward(self, inp):
        """
        Args:
            inp (tensor) :              Tensor of input Minibatch

        Returns:
            (tensor): Change detection output
            (tensor): Domain output. Will not be returned when self.adversarial="no"
        """

        empty_bg  = inp[:,1:4,:,:]
        recent_bg = inp[:,5:8,:,:]
        curr_fr   = inp[:,9:,:,:]


        empty_bg_out, empty_bg_feats = self.empty_bg_enc(empty_bg)
        recent_bg_out, recent_bg_feats = self.recent_bg_enc(recent_bg)
        curr_fr_out, curr_fr_feats = self.curr_fr_enc(curr_fr)

        out = torch.cat([empty_bg, recent_bg, curr_fr], dim=1)
        d1 = torch.cat([empty_bg_feats[0], recent_bg_feats[0], curr_fr_feats[0]], dim = 1)
        d1 = self.conv1(d1)
        d2 = torch.cat([empty_bg_feats[1], recent_bg_feats[1], curr_fr_feats[1]], dim = 1)
        d2 = self.conv2(d2)
        d3 = torch.cat([empty_bg_feats[2], recent_bg_feats[2], curr_fr_feats[2]], dim = 1)
        d3 = self.conv3(d3)
        d4 = torch.cat([empty_bg_feats[3], recent_bg_feats[3], curr_fr_feats[3]], dim = 1)
        d4 = self.conv4(d4)
        d5 = torch.cat([empty_bg_feats[4], recent_bg_feats[4], curr_fr_feats[4]], dim = 1)
        d5 = self.conv5(d5)

        if self.skip:
            u4 = self.dec4(d5, d4)
            u3 = self.dec3(u4, d3)
            u2 = self.dec2(u3, d2)
            u1 = self.dec1(u2, d1)
        else:
            u4 = self.dec4(d5)
            u3 = self.dec3(u4)
            u2 = self.dec2(u3)
            u1 = self.dec1(u2)
        cd_out = self.out(u1)

        return cd_out



 