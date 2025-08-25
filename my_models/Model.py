import torch
import torch.nn as nn
from my_models.ResNet import ResNet
from my_models.GMAModel import GMAModel
from my_models.EMAttention import EMA


class PyramidPool(nn.Module):
    def __init__(self, in_channels, out_channels, pool_size):
        super(PyramidPool, self).__init__()
        self.features = nn.Sequential(
            nn.AdaptiveAvgPool2d(pool_size),
            nn.Conv2d(in_channels, out_channels, kernel_size=3,padding=pool_size,dilation=pool_size, bias=False),
            nn.BatchNorm2d(out_channels,momentum=0.99),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        size = x.shape
        output = nn.functional.interpolate(self.features(x), size[2:])
        return output


class APPM(nn.Module):
    def __init__(self):
        super(APPM, self).__init__()

        self.layer5a = PyramidPool(512, 1024, 1)
        self.layer5b = PyramidPool(512, 1024, 4)
        self.layer5c = PyramidPool(512, 1024, 6)
        self.layer5d = PyramidPool(512, 1024, 8)

        self.final = nn.Sequential(
            nn.Conv2d(4608, 1024, 3, padding=1, bias=False),
            nn.BatchNorm2d(1024, momentum=.95),
            nn.ReLU(inplace=True)
        )



    def forward(self, x):
        x = self.final(torch.cat([
            x,
            self.layer5a(x),
            self.layer5b(x),
            self.layer5c(x),
            self.layer5d(x),
        ], 1))
        return x


class GMA_block(nn.Module):

    def __init__(self):
        super(GMA_block, self).__init__()
        self.transformer = GMAModel(dim=512, mlp_dim=2*512, heads=8, depth=12)

    def forward(self,x):
        b, c, h, w = x.shape
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(b, -1, 512)
        vit, _ = self.transformer(x)
        vit = vit.view(b, h, w, 512)
        out = vit.permute(0, 3, 1, 2).contiguous()
        return out


class Model(nn.Module):
    def __init__(self, num_classes=2, reduction_factor=8):
        super(Model, self).__init__()
        self.bottle_planes = (512 // reduction_factor) * 5
        self.num_classes = num_classes
        print("-" *20, "initializing model", "-" *20)
        # encoder
        self.cnn = ResNet()
        # bottleneck
        self.mslayer = APPM()
        self.down = nn.Conv2d(1024,512,3,padding=1,bias=False)
        self.ViT = GMA_block()
        self.conv = nn.Sequential(
            nn.Conv2d(512,512,kernel_size=3,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        #decoder
        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up_con1 = nn.Sequential(
            nn.Conv2d(256,256,kernel_size=3,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256,256,kernel_size=3,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
            )
        self.block1 = EMA(256)

        self.up2 = nn.ConvTranspose2d(256, 128 ,kernel_size=2, stride=2)
        self.up_con2 = nn.Sequential(
            nn.Conv2d(128,128,kernel_size=3,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128,128,kernel_size=3,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
            )
        self.block2 = EMA(128)
        
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up_con3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), 
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
            )
        self.block3 = EMA(64)

        self.up4 = nn.ConvTranspose2d(64, 32, kernel_size=(2,2), stride=(2,2))
        self.out = nn.Conv2d(32, num_classes, kernel_size=1)


    def forward(self, x):
        # resnet encoder
        out, x1, x2, x3 = self.cnn(x)

        # bottleneck 
        out1 = self.down(self.mslayer(out))
        out2 = self.ViT(out)
        out3 = self.conv(out) 
        out = out1 + out2 + out3

        # decoder
        out = self.up1(out) + x3
        out = self.up_con1(out)
        out = self.block1(out)

        out = self.up2(out) + x2
        out = self.up_con2(out)
        out = self.block2(out)

        out = self.up3(out) + x1
        out = self.up_con3(out)
        out = self.block3(out)

        out = self.up4(out)
        
        return self.out(out)

if __name__ == '__main__':
    x = torch.rand([4, 3, 256, 256])
    a = Model(num_classes=2)
    print(a(x).shape)
    pass