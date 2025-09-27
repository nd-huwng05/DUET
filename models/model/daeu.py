import torch.nn as nn
from  models.basic.block import BottleNeck, ResBlock, BasicBlock

class DAEU(nn.Module):
    def __init__(self, input_size=64, in_planes=1, base_width=16, expansion=1, mid_num=2048, latent_size=16,
                 en_num_layers=None, de_num_layers=None,residual=False, layer=4):
        super(DAEU, self).__init__()
        block = ResBlock if residual else BasicBlock
        self.fm = input_size // 16  # down-sample for 4 times. 2^4=16
        self.layer = layer

        if en_num_layers is None:
            en_num_layers = 1
        if de_num_layers is None:
            de_num_layers = 1

        self.en_block1 = block(in_planes, 1 * base_width * expansion, en_num_layers, downsample=True)

        self.en_block2 = block(1 * base_width * expansion, 2 * base_width * expansion, en_num_layers,
                               downsample=True)
        self.en_block3 = block(2 * base_width * expansion, 4 * base_width * expansion, en_num_layers,
                               downsample=True)
        self.en_block4 = block(4 * base_width * expansion, 4 * base_width * expansion, en_num_layers,
                               downsample=True)

        self.bottle_neck = BottleNeck(4 * base_width * expansion, feature_size=self.fm, mid_num=mid_num,
                                      latent_size=latent_size)

        self.de_block1 = block(4 * base_width * expansion, 4 * base_width * expansion, de_num_layers,
                               upsample=True)
        self.de_block2 = block(4 * base_width * expansion, 2 * base_width * expansion, de_num_layers,
                               upsample=True)
        self.de_block3 = block(2 * base_width * expansion, 1 * base_width * expansion, de_num_layers,
                               upsample=True)
        self.de_block4 = block(1 * base_width * expansion,  2 * in_planes, 1, upsample=True,
                            last_layer=True)

        for param in self.en_block1.parameters():
            param.requires_grad = False
        for param in self.en_block2.parameters():
            param.requires_grad = False
        for param in self.en_block3.parameters():
            param.requires_grad = False

    def forward(self, x):
        en1 = self.en_block1(x)
        en2 = self.en_block2(en1)
        en3 = self.en_block3(en2)
        en4 = self.en_block4(en3)

        de1, z = self.bottle_neck(en4)

        de2 = self.de_block1(de1)
        de3 = self.de_block2(de2)
        de4 = self.de_block3(de3)
        x_hat, log_var = self.de_block4(de4).chunk(2, 1)
        return  x_hat, log_var, z

import torch
if __name__ == '__main__':
    # Giả sử input_size = 64, in_planes = 1
    input_size = 64
    batch_size = 2
    channels = 1
    height = width = input_size

    # Tạo model
    model = DAEU(input_size=input_size, in_planes=channels, latent_size=16, residual=False)

    # Tạo input dummy
    x = torch.randn(batch_size, channels, height, width)

    # Forward pass
    try:
        x_hat, log_var, z = model(x)
        print("Forward pass thành công!")
        print("x_hat shape:", x_hat.shape)
        print("log_var shape:", log_var.shape)
        print("z shape:", z.shape)
    except Exception as e:
        print("Lỗi khi forward pass:", e)