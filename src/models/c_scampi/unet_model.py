__author__ = "Thomas Siedler, Volker Herold"
__year__ = "2023"
__version__ = "0.0.1"

from .unet_parts import *
from utils.cartesian.transforms import image2kspace_torch, cartesian_backward
from utils.data_utils import toComplex, toReal
from utils.cartesian.sampling import data_consistency


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x, skip_connections=1.0):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, skip_connections * x4)
        x = self.up2(x, skip_connections * x3)
        x = self.up3(x, skip_connections * x2)
        x = self.up4(x, skip_connections * x1)
        logits = self.outc(x)
        return logits


class DipUnet(UNet):
    """Implementation of UNet for Deep Image Prior in MRI."""

    def __init__(self, n_channels, n_classes, mask, bilinear=True, produce=True, apply_data_consistency=False, k0=None,
                 coilmap=None, skip_connections=1.0):

        super(DipUnet, self).__init__(n_channels, n_classes, bilinear=bilinear)
        self.mask = mask
        self.produce = produce
        self.data_consistency = apply_data_consistency
        self.k0 = k0
        self.coilmap = coilmap
        self.skip_connections = skip_connections
        if self.data_consistency:
            assert self.k0 is not None, "When data_consistency is selected, DipUnet needs a k0 to be passed!"

    def forward(self, x, **kwargs):
        logits = super(DipUnet, self).forward(x, self.skip_connections)

        if self.data_consistency and self.produce:
            logits = toComplex(logits, dim=1)
            if self.coilmap is not None:
                logits = self.coilmap * logits.broadcast_to(self.coilmap.shape)
            logits = image2kspace_torch(logits, [0, 0, 1, 1])
            logits = toReal(logits, dim=1)

            logits = data_consistency(logits, self.k0, self.mask.int())
            logits = toComplex(logits, dim=1)
            logits = cartesian_backward(logits, self.coilmap)

        return logits
