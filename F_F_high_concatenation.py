import torch.nn as nn
import torch.nn.functional as F

class FFHighEncoder(nn.Module):
    # This is the encoder used for concatenating the F and F_high image
    # The structure implemented here is the same as the one used in the live portrait paper Fig.A12
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(192,256,kernel_zie = (3,3), stride = (1,1), padding = (1,1))
        self.conv2 = nn.Conv2d(256,128,kernel_size = (3,3), stride = (1,1), padding = (1,1))
        
        # self.overlap_patch_embed = OverLapPatchEmbed(img_size = 256, strid = 2, in_chans = 128, embed_dim = 1024)
        # self.transformer_block = TransformerBlock(dim = 1024, num_heads = 2, mlp_ratio = 2, sr_ratio = 2) 	
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor = 2)
        # concat with output of endcoder for F(output of ViT) with Conv layers
        # 352 = 256(F) + 96(F_high)
        self.conv3 = nn.Conv2d(352,256,kernel_size = (3,3), stride = (1,1), padding = (1,1))
        self.conv4 = nn.Conv2d(256,128,kernel_size = (3,3), stride = (1,1), padding = (1,1))
        self.conv5 = nn.Conv2d(128,128,kernel_size = (3,3), stride = (1,1), padding = (1,1))
        self.conv6 = nn.Conv2d(128,96,kernel_size = (3,3), stride = (1,1), padding = (1,1))
        

        
        
    def forward(self, x):
        x = self.conv1(x)
        x = F.leaky_relu(x,negative_slope=0.01)
        x = self.conv2(x)
        x = F.leaky_relu(x,negative_slope=0.01)
        # x = self.overlap_patch_embed(x)
        # x = self.transformer_block(x)
        x = self.pixel_shuffle(x)
        x = self.conv3(x)
        x = F.leaky_relu(x,negative_slope=0.01)
        x = self.conv4(x)
        x = F.leaky_relu(x,negative_slope=0.01)
        x = self.conv5(x)
        x = F.leaky_relu(x,negative_slope=0.01)
        x = self.conv6(x)
        return x
