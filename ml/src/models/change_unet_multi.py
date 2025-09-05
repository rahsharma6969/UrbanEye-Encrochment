import torch, torch.nn as nn, torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.net(x)

class ChangeUNetMulti(nn.Module):
    def __init__(self, in_ch=8, n_classes=3):  # 4 bands * 2 times = 8; classes: {0..4}, we will ignore 255 in loss
        super().__init__()
        ch = [64,128,256]
        self.enc1 = DoubleConv(in_ch, ch[0])
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(ch[0], ch[1])
        self.pool2 = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(ch[1], ch[2])

        self.up2 = nn.ConvTranspose2d(ch[2], ch[1], 2, stride=2)
        self.dec2 = DoubleConv(ch[2], ch[1])
        self.up1 = nn.ConvTranspose2d(ch[1], ch[0], 2, stride=2)
        self.dec1 = DoubleConv(ch[1], ch[0])

        self.outc = nn.Conv2d(ch[0], n_classes, 1)

    def forward(self, t0, t1):
        # concat times (simple Siamese by concatenation)
        x = torch.cat([t0, t1], dim=1)  # Bx8xHxW
        e1 = self.enc1(x)               # Bx64
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)              # Bx128
        p2 = self.pool2(e2)
        b = self.bottleneck(p2)         # Bx256
        u2 = self.up2(b)
        d2 = self.dec2(torch.cat([u2, e2], dim=1))
        u1 = self.up1(d2)
        d1 = self.dec1(torch.cat([u1, e1], dim=1))
        logits = self.outc(d1)          # BxC
        return logits
