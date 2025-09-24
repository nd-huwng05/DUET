from torch import nn
import torch
import torchxrayvision as xrv
import torchvision.models as models

class DenseNetEncoder(nn.Module):
    def __init__(self, image_size=128, in_channels=1, out_channels=256,
                 pretrained=True, pretrained_idx=0):
        super(DenseNetEncoder, self).__init__()
        self.image_size = image_size
        densenet121 = models.densenet121()

        if pretrained:
            if pretrained_idx == 0:
                state_dict = torch.load("data/DenseNet121.pt", map_location="cpu")
                if "state_dict" in state_dict:
                    state_dict = state_dict["state_dict"]
                densenet121.load_state_dict(state_dict, strict=False)
            elif pretrained_idx == 1:
                densenet121 = xrv.models.DenseNet(weights="densenet121-res224-all")
            else:
                densenet121 = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)

        if in_channels == 1:
            old_conv = densenet121.features.conv0
            new_conv = nn.Conv2d(in_channels, old_conv.out_channels,
                                 kernel_size=old_conv.kernel_size,
                                 stride=old_conv.stride,
                                 padding=old_conv.padding,
                                 bias=False)
            if pretrained:
                new_conv.weight.data = old_conv.weight.data.mean(dim=1, keepdim=True)
            densenet121.features.conv0 = new_conv

        self.stem = nn.Sequential(
            densenet121.features.conv0,
            densenet121.features.norm0,
            densenet121.features.relu0,
            densenet121.features.pool0,
        )

        self.block1 = nn.Sequential(densenet121.features.denseblock1,
                                    densenet121.features.transition1)
        self.block2 = nn.Sequential(densenet121.features.denseblock2,
                                    densenet121.features.transition2)
        self.block3 = nn.Sequential(densenet121.features.denseblock3,
                                    densenet121.features.transition3)
        self.block4 = densenet121.features.denseblock4

        for param in self.stem.parameters():
            param.requires_grad = False
        for param in self.block1.parameters():
            param.requires_grad = False
        for param in self.block2.parameters():
            param.requires_grad = False
        for param in self.block3.parameters():
            param.requires_grad = False
        for param in self.block4.parameters():
            param.requires_grad = False

        if out_channels == 256:
            self.model = nn.Sequential(self.stem, self.block1, self.block2)
            for param in self.block2.parameters():
                param.requires_grad = True
        elif out_channels == 512:
            self.model = nn.Sequential(self.stem, self.block1, self.block2, self.block3)
            for param in self.block3.parameters():
                param.requires_grad = True
        elif out_channels == 1024:
            self.model = nn.Sequential(self.stem, self.block1, self.block2, self.block3, self.block4)
            for param in self.block4.parameters():
                param.requires_grad = True
        else:
            raise ValueError("out_channels isn't 256, 512, 1024")


    def forward(self, x):
        return self.model(x)