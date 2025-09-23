import torch.nn as nn
from models.basic.densenet_encoder import DenseNetEncoder
from  models.basic.block import ConvTransposeBnRelu, BottleNeck

class DUET(nn.Module):
    def __init__(self, image_size=128, in_channels=1, out_channels=1,
                pretrained=True, out_channels_pre=512, mid_num=2048,
                latent_size=16, pretrained_idx=1, config=None):
        super(DUET, self).__init__()
        self.dense = DenseNetEncoder(image_size=image_size, in_channels=in_channels,
                                     out_channels=out_channels_pre, pretrained=pretrained
                                     ,pretrained_idx=pretrained_idx)

        self.fm = image_size//16 if out_channels_pre==256 else image_size//32
        self.bottle_neck = BottleNeck(out_channels_pre, self.fm, mid_num=mid_num, latent_size=latent_size)
        self.in_channels_decoder = out_channels_pre
        self.out_channels_decoder = out_channels_pre // 2
        blocks = []
        while self.in_channels_decoder != 32:
            blocks.append(ConvTransposeBnRelu(input_channels=self.in_channels_decoder,
                                              output_channels=self.out_channels_decoder,
                                              kernel_size=4, stride=2))
            self.in_channels_decoder = self.out_channels_decoder
            self.out_channels_decoder = self.out_channels_decoder // 2
        blocks.append(ConvTransposeBnRelu(input_channels=self.in_channels_decoder,
                                          output_channels=2*out_channels, kernel_size=3, stride=2, last_layer=True))
        self.decoder = nn.ModuleList(blocks)

    def forward(self, x):
        x = self.dense(x)
        out, z = self.bottle_neck(x)
        for block in self.decoder:
            out = block(out)
        x_hat, log_var = out.chunk(2, dim=1)
        return x_hat, log_var, z