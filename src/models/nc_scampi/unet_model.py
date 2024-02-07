__author__ = "Volker Herold, Thomas Siedler"
__year__ = "2023"
__version__ = "0.0.1"


from .unet_parts import *


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

    def forward(self, x,skip_connections=1.0):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, skip_connections*x4)
        x = self.up2(x, skip_connections*x3)
        x = self.up3(x, skip_connections*x2)
        x = self.up4(x, skip_connections*x1)
        logits = self.outc(x)
        return logits


class DipUnet(UNet):
    """Implementation of UNet for Deep Image Prior in MRI."""

    def __init__(self, n_channels, n_classes, bilinear=True, produce=True,
                 imagespace=True, data_consistency=False, skip_connections=1.0):
        super(DipUnet, self).__init__(n_channels, n_classes, bilinear=bilinear)

        self.produce = produce
        self.imagespace = imagespace
        self.data_consistency = data_consistency
        self.skip_connections = skip_connections


    def forward(self, x):
        logits = super(DipUnet, self).forward(x,self.skip_connections)



        return logits
