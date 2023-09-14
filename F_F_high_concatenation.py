import torch.nn as nn
import torch.nn.functional as F
from segformer_modified import SegFormerModified

# The input x to this encoder should be the concatenated F and F_high image
# which yeilds a 192 (96 + 96) channel image

class FFHighEncoder(nn.Module):
    # This is the encoder used for concatenating the F and F_high image
    # The structure implemented here is the same as the one used in the live portrait paper Fig.A12
    # The output of this encoder is the triplane representation
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(192,256,kernel_zie = (3,3), stride = (1,1), padding = (1,1))
        self.conv2 = nn.Conv2d(256,128,kernel_size = (3,3), stride = (1,1), padding = (1,1))
        
        # The paper doesn't explicitly mention the patch size, qkv_bias, drop_rate, and batch normalizaiton
        # Here we keep the config the same as in the FLowFEncoder:
        # patch_size = 3, without dropout, with kqv_bias, with layer normalization
        self.segformer_mod = SegFormerModified(img_size=256,
                                        patch_size=3, 
                                        stride=2,
                                        in_chans=128,
                                        embed_dims=1024,
                                        num_heads= 2,
                                        mlp_ratios= 2,
                                        sr_ratios=2,
                                        qkv_bias= True,
                                        drop_rate=0.0,
                                        norm_layer=nn.LayerNorm,
                                        depths= 2,
                                        )	
        
        # concat with output of endcoder for F(output of ViT) with Conv layers
        # 352 = 256(F) + 96(F_high)
        self.conv3 = nn.Conv2d(352,256,kernel_size = (3,3), stride = (1,1), padding = (1,1))
        self.conv4 = nn.Conv2d(256,128,kernel_size = (3,3), stride = (1,1), padding = (1,1))
        self.conv5 = nn.Conv2d(128,128,kernel_size = (3,3), stride = (1,1), padding = (1,1))
        self.conv6 = nn.Conv2d(128,96,kernel_size = (3,3), stride = (1,1), padding = (1,1))
        

        
        
    def forward(self, x, y):
        # The x here should be the 
        x = self.conv1(x)
        x = F.leaky_relu(x,negative_slope=0.01)
        x = self.conv2(x)
        x = F.leaky_relu(x,negative_slope=0.01)
        x = self.segformer_mod(x)
        x = F.pixle_shuffle(x,2)
        x = self.pixel_shuffle(x)
        # concat with output of endcoder for F(output of ViT) with Conv layers
        # 352 = 256(F) + 96(F_high)
        x = self.conv3(x)
        x = F.leaky_relu(x,negative_slope=0.01)
        x = self.conv4(x)
        x = F.leaky_relu(x,negative_slope=0.01)
        x = self.conv5(x)
        x = F.leaky_relu(x,negative_slope=0.01)
        x = self.conv6(x)
        return x


class FFHighEncoderLT(nn.Module):
    # This is the light version of encoder used for concatenating the F and F_high image
    # The structure implemented here is the same as the one used in the live portrait paper Fig.A12
    # The output of this encoder is the triplane representation
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(256,256,kernel_zie = (3,3), stride = (1,1), padding = (1,1))
        
        self.overlap_patch_embed = OverLapPatchEmbed(img_size = 128, strid = 2, in_chans = 128, embed_dim = 1024)
        self.transformer_block = MixVisionTransformer(dim = 256, num_heads = 2, mlp_ratio = 2, sr_ratio = 2) 	
        self.conv2 = nn.Conv2d(256,256,kernel_size = (3,3), stride = (1,1), padding = (1,1))
        self.conv3 = nn.Conv2d(256,128,kernel_size = (3,3), stride = (1,1), padding = (1,1))
        self.conv4 = nn.Conv2d(128,128,kernel_size = (3,3), stride = (1,1), padding = (1,1))
        self.conv5 = nn.Conv2d(128,96,kernel_size = (3,3), stride = (1,1), padding = (1,1))
        self.conv6 = nn.Conv2d(96,96,kernel_size = (3,3), stride = (1,1), padding = (1,1))
        
        

        
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.overlap_patch_embed(x)
        x = self.transformer_block(x)
        x = F.pixle_shuffle(x,2)
        x = self.conv2(x)
        x = F.leaky_relu(x,negative_slope=0.01)
        x = F.upsampling_bilinear2d(x, scale_factor=2)
        x = self.conv3(x)
        x = F.leaky_relu(x,negative_slope=0.01)
        x = self.conv4(x)
        x = F.leaky_relu(x,negative_slope=0.01)
        x = self.conv5(x)
        x = F.leaky_relu(x,negative_slope=0.01)
        x = self.conv6(x)
        return x
