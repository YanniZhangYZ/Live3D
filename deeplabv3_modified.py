# ---------------------------------------------------------------
# This is the modified DeepLabV3 model originally from segmentation_models.pytorch github repo
# ---------------------------------------------------------------



from torch import nn
from typing import Optional

from segmentation_models_pytorch.base import (
    SegmentationModel,
    SegmentationHead,
    ClassificationHead,
)

from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.decoders import ASPP


class DeepLabV3Modified(SegmentationModel):
    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        # encoder_weights: Optional[str] = "imagenet",
        decoder_channels: int = 256,
        in_channels: int = 5,
        classes: int = 1,
        activation: Optional[str] = None,
        upsampling: int = 8,
        aux_params: Optional[dict] = None,
    ):
        super().__init__()

        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            # weights=encoder_weights,
            output_stride=8,
        )

        self.decoder = DeepLabV3Decoder(
            in_channels=self.encoder.out_channels[-1],
            out_channels=decoder_channels,
        )

        # According to the paper, the final output they use is from the layer before bilinear upsampling and segmentation head
        
        # self.segmentation_head = SegmentationHead(
        #     in_channels=self.decoder.out_channels,
        #     out_channels=classes,
        #     activation=activation,
        #     kernel_size=1,
        #     upsampling=upsampling,
        # )

        if aux_params is not None:
            self.classification_head = ClassificationHead(in_channels=self.encoder.out_channels[-1], **aux_params)
        else:
            self.classification_head = None


class DeepLabV3Decoder(nn.Sequential):
    def __init__(self, in_channels, out_channels=256, atrous_rates=(12, 24, 36)):
        super().__init__(
            ASPP(in_channels, out_channels, atrous_rates),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            # nn.BatchNorm2d(out_channels), # the paper mentioned that they remove all the instance of batch norm
            nn.ReLU(),
        )
        self.out_channels = out_channels

    def forward(self, *features):
        return super().forward(features[-1])