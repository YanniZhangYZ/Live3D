import torch.nn as nn
import torch.nn.functional as F


class EncoderHigh(nn.Module):
    # This is the module that encodes the input original image to a feature map with high frequency
    # The structure implemented here is the same as the one used in the live portrait paper Fig.A11
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(5,64,kernel_zie = (7,7), stride = (2,2), padding = (3,3))
        self.conv2 = nn.Conv2d(64,96,kernel_size = (3,3), stride = (1,1), padding = (1,1))
        self.conv3 = nn.Conv2d(96,96,kernel_size = (3,3), stride = (1,1), padding = (1,1))
        self.conv4 = nn.Conv2d(96,96,kernel_size = (3,3), stride = (1,1), padding = (1,1))
        self.conv5 = nn.Conv2d(96,96,kernel_size = (3,3), stride = (1,1), padding = (1,1))
       
        
        
    def forward(self, x):
        x = self.conv1(x)
        x = F.leaky_relu(x,negative_slope=0.01)
        x = self.conv2(x)
        x = F.leaky_relu(x,negative_slope=0.01)
        x = self.conv3(x)
        x = F.leaky_relu(x,negative_slope=0.01)
        x = self.conv4(x)
        x = F.leaky_relu(x,negative_slope=0.01)
        x = self.conv5(x)
        x = F.leaky_relu(x,negative_slope=0.01)
        return x
