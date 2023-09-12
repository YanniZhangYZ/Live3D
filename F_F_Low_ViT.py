import torch.nn as nn
import torch.nn.functional as F


class FFLowEncoder(nn.Module):
    # This is the encoder used to get feature map F from F_low
    # The structure implemented here is the same as the one used in the live portrait paper Fig.A10
    def __init__(self):
        super().__init__()
        # patch_size shuold be 3
        # self.overlap_patch_embed = OverLapPatchEmbed(img_size = 64, strid = 2, in_chans = 256, embed_dim = 1024)
        # without dropout, with qkv_bias = True, with layer normalization
        # self.transformer_block1 = TransformerBlock(dim = 1024, num_heads = 4, mlp_ratio = 2, sr_ratio = 1) 
        # self.transformer_block2 = TransformerBlock(dim = 1024, num_heads = 4, mlp_ratio = 2, sr_ratio = 1) 	
        # self.transformer_block3 = TransformerBlock(dim = 1024, num_heads = 4, mlp_ratio = 2, sr_ratio = 1)
        # self.transformer_block4 = TransformerBlock(dim = 1024, num_heads = 4, mlp_ratio = 2, sr_ratio = 1)
        # self.transformer_block5 = TransformerBlock(dim = 1024, num_heads = 4, mlp_ratio = 2, sr_ratio = 1) 	
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor = 2)

        self.conv1 = nn.Conv2d(256,128,kernel_size = (3,3), stride = (1,1), padding = (1,1))
        self.conv2 = nn.Conv2d(128,128,kernel_size = (3,3), stride = (1,1), padding = (1,1))
        self.conv3 = nn.Conv2d(128,96,kernel_size = (3,3), stride = (1,1), padding = (1,1))
        

        
        
    def forward(self, x):
        # x = self.overlap_patch_embed(x)
        # x = self.transformer_block1(x)
        # x = self.transformer_block2(x)
        # x = self.transformer_block3(x)
        # x = self.transformer_block4(x)
        # x = self.transformer_block5(x)
        x = self.pixel_shuffle(x)
        x = F.upsampling_bilinear2d(x, scale_factor=2)
        x = self.conv1(x)
        x = F.relu(x)
        x = F.upsampling_bilinear2d(x, scale_factor=2)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        return x


class FFLowEncoderLT(nn.Module):
    # This is the light weight version of encoder used to get feature map F from F_low
    # The structure implemented here is the same as the one used in the live portrait paper Fig.A10
    def __init__(self):
        super().__init__()
        # patch_size shuold be 3
        # self.overlap_patch_embed = OverLapPatchEmbed(img_size = 64, strid = 2, in_chans = 256, embed_dim = 1024)
        # without dropout, with qkv_bias = True, with layer normalization
        # self.transformer_block1 = TransformerBlock(dim = 1024, num_heads = 4, mlp_ratio = 2, sr_ratio = 1) 
        # self.transformer_block2 = TransformerBlock(dim = 1024, num_heads = 4, mlp_ratio = 2, sr_ratio = 1) 	
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor = 2)
        self.conv1 = nn.Conv2d(256,128,kernel_size = (3,3), stride = (1,1), padding = (1,1))
        self.conv2 = nn.Conv2d(128,128,kernel_size = (3,3), stride = (1,1), padding = (1,1))
        

        
        
    def forward(self, x):
        # x = self.overlap_patch_embed(x)
        # x = self.transformer_block1(x)
        # x = self.transformer_block2(x)
        x = self.pixel_shuffle(x)
        x = F.upsampling_bilinear2d(x, scale_factor=2)
        x = self.conv1(x)
        x = F.leaky_relu(x,negative_slope=0.01)
        x = self.conv2(x)
        return x
