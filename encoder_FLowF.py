
import torch.nn as nn
import torch.nn.functional as F
from segformer_modified import SegFormerModified


# ---------------------------------------------------------------
# The following is the implementation of Encoder for F with Conv layers (Fig.A10)
# ---------------------------------------------------------------

class FLowFEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.segformer_mod = SegFormerModified(img_size=64,
                                        patch_size=3,
                                        stride=2,
                                        in_chans=256,
                                        embed_dims=1024,
                                        num_heads= 4,
                                        mlp_ratios= 2,
                                        sr_ratios=1,
                                        qkv_bias=True,
                                        drop_rate=0.0,
                                        norm_layer=nn.LayerNorm,
                                        depths= 5,
                                        )
        self.conv1 = nn.Conv2d(256,128,kernel_size = (3,3), stride = (1,1), padding = (1,1))
        self.conv2 = nn.Conv2d(128,128,kernel_size = (3,3), stride = (1,1), padding = (1,1))
        self.conv3 = nn.Conv2d(128,96,kernel_size = (3,3), stride = (1,1), padding = (1,1))
        
    def forward(self, x):
        x = self.segformer_mod(x)
        x = F.pixle_shuffle(x,2)
        x = F.upsampling_bilinear2d(x, scale_factor=2)
        x = self.conv1(x)
        x = F.relu(x)
        x = F.upsampling_bilinear2d(x, scale_factor=2)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        return x


class FLowFEncoderLT(nn.Module):
    def __init__(self):
        super().__init__()
        self.segformer_mod = SegFormerModified(img_size=64,
                                        patch_size=3,
                                        stride=2,
                                        in_chans=256,
                                        embed_dims=1024,
                                        num_heads= 4,
                                        mlp_ratios= 2,
                                        sr_ratios=1,
                                        qkv_bias=True,
                                        drop_rate=0.0,
                                        norm_layer=nn.LayerNorm,
                                        depths= 2,
                                        )
        self.conv1 = nn.Conv2d(256,128,kernel_size = (3,3), stride = (1,1), padding = (1,1))
        self.conv2 = nn.Conv2d(128,128,kernel_size = (3,3), stride = (1,1), padding = (1,1))
        
    def forward(self, x):
        x = self.segformer_mod(x)
        x = F.pixle_shuffle(x,2)
        x = F.upsampling_bilinear2d(x, scale_factor=2)
        x = self.conv1(x)
        x = F.leaky_relu(x,negative_slope=0.01)
        x = self.conv2(x)
        return x