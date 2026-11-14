"""U-Net with a ResNet-34 encoder and a decoder that runs to full resolution."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class Up(nn.Module):
    """Nearest-neighbour 2x upsampling, optional skip concatenation, conv-BN-ReLU."""

    def __init__(self, cin, cskip, cout, nconv=2):
        super().__init__()
        layers = [nn.Conv2d(cin + cskip, cout, 3, padding=1, bias=False),
                  nn.BatchNorm2d(cout), nn.ReLU(inplace=True)]
        for _ in range(nconv - 1):
            layers += [nn.Conv2d(cout, cout, 3, padding=1, bias=False),
                       nn.BatchNorm2d(cout), nn.ReLU(inplace=True)]
        self.block = nn.Sequential(*layers)

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if skip is not None:
            x = torch.cat([x, skip], 1)
        return self.block(x)


class FilamentUNet(nn.Module):
    """The decoder upsamples all the way to stride 1 instead of stopping at
    stride 4 and interpolating: filament barbs are a few pixels wide, and the
    last two stages are where their boundaries are decided. Those two stages
    dominate the decoder cost, so they use a single convolution and few channels.
    """

    def __init__(self, pretrained_encoder=False):
        super().__init__()
        weights = torchvision.models.ResNet34_Weights.DEFAULT if pretrained_encoder else None
        enc = torchvision.models.resnet34(weights=weights)
        self.stem = nn.Sequential(enc.conv1, enc.bn1, enc.relu)   # stride 2, 64 ch
        self.pool = enc.maxpool
        self.l1, self.l2, self.l3, self.l4 = enc.layer1, enc.layer2, enc.layer3, enc.layer4
        self.d4 = Up(512, 256, 256)
        self.d3 = Up(256, 128, 128)
        self.d2 = Up(128, 64, 64)
        self.d1 = Up(64, 64, 32, nconv=1)
        self.d0 = Up(32, 0, 24, nconv=1)
        self.head = nn.Conv2d(24, 1, 1)

    def forward(self, x):
        x0 = self.stem(x)              # stride 2
        x1 = self.l1(self.pool(x0))    # stride 4
        x2 = self.l2(x1)               # stride 8
        x3 = self.l3(x2)               # stride 16
        x4 = self.l4(x3)               # stride 32
        y = self.d4(x4, x3)
        y = self.d3(y, x2)
        y = self.d2(y, x1)
        y = self.d1(y, x0)
        y = self.d0(y)
        return self.head(y)            # logits at stride 1


def load_model(weights_path, device):
    """Model in eval mode with trained weights. Does not download anything."""
    model = FilamentUNet(pretrained_encoder=False)
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    return model.to(device).to(memory_format=torch.channels_last).eval()
