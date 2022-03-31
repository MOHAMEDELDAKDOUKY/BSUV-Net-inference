"""
16-layer U-Net model
"""
import torch.nn as nn
from models.unet_tools import UNetDown, UNetUp, ConvSig, FCNN, Embed
from models.vggnet import VGGNet


class unet_vgg16(nn.Module):
    """
    Args:
        inp_ch (int): Number of input channels
        kernel_size (int): Size of the convolutional kernels
        skip (bool, default=True): Use skip connections
    """
    def __init__(self, inp_ch, kernel_size=3, skip=False):
        super().__init__()
        self.skip = skip
        
        self.empty_bg_enc = VGGNet()
        self.curr_fr_enc = VGGNet()
        
        # self.dec4 = UNetUp(512, skip*512, 512, 2, batch_norm=True, kernel_size=kernel_size)
        self.dec3 = UNetUp(512, skip * 256, 256, 2, batch_norm=True, kernel_size=kernel_size)
        self.dec2 = UNetUp(256, skip * 128, 128, 2, batch_norm=True, kernel_size=kernel_size)
        self.dec1 = UNetUp(128, skip * 64 , 64 , 2, batch_norm=True, kernel_size=kernel_size)
        self.out  = ConvSig(64)

    def forward(self, inp):
        """
        Args:
            inp (tensor) :              Tensor of input Minibatch

        Returns:
            (tensor): Change detection output
            (tensor): Domain output. Will not be returned when self.adversarial="no"
        """
        empty_bg = inp[:,0:3,:,:]
        curr_fr  = inp[:,3:6,:,:]

        _, empty_bg_feats = self.empty_bg_enc(empty_bg)
        _, curr_fr_feats = self.curr_fr_enc(curr_fr)

        d1 =  curr_fr_feats[0]
        d2 =  curr_fr_feats[1]
        d3 =  empty_bg_feats[2] - curr_fr_feats[2]
        d4 =  empty_bg_feats[3] - curr_fr_feats[3]
        #d5 = empty_bg_feats[4] - curr_fr_feats[4]

        if self.skip:
            #u4 = self.dec4(d5, d4)
            u3 = self.dec3(d4, d3)
            u2 = self.dec2(u3, d2)
            u1 = self.dec1(u2, d1 ) 
        else:
            #u4 = self.dec4(d5)
            u3 = self.dec3(d4)
            u2 = self.dec2(u3)
            u1 = self.dec1(u2)
        cd_out = self.out(u1)

        return cd_out




 