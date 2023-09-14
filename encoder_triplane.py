from encoder_FLowF import FLowFEncoder
from encoder_high import IFHighEncoder
from F_F_high_concatenation import FFHighEncoder
from deeplabv3_modified import DeepLabV3Modified

import torch.nn as nn

class TriplaneEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder_FlowF = FLowFEncoder()
        self.encoder_high = IFHighEncoder()
        self.encoder_concat = FFHighEncoder()
        self.deeplabv3 = DeepLabV3Modified()
        
    def forward(self, x):
        # should have concatenating 2D pixel channel to the original image
        x_5 = x
        
        f_high = self.encoder_high(x_5)
        f_low = self.deeplabv3(x_5)
        f = self.encoder_FlowF(f_low)
        triplane = self.encoder_concat(f, f_high)
        return triplane
