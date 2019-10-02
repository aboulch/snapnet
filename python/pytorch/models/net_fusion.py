"""Unet segmentation network."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class FusionNet(nn.Module):
    """Unet segmentation network."""

    def __init__(self, in_channels, out_channels):
        """Init Unet fields."""
        super(FusionNet, self).__init__()

        self.in_channels = in_channels

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, out_channels, kernel_size=3, padding=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x1, x2, f1, f2):
        """Forward method."""

        x = torch.cat([f1,f2], dim=1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return x + x1 + x2

    
    def load_from_filename(self, model_path):
        """Load weights from filename."""
        th = torch.load(model_path)  # load the weigths
        self.load_state_dict(th)


def fusion_net(in_channels, out_channels, pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = FusionNet(in_channels, out_channels)
    return model