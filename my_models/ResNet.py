import torch
import torch.nn as nn
from torchvision.models import resnet34, ResNet34_Weights
# from torchvision.models import resnet101,ResNet101_Weights
from torchvision.models import resnet18,ResNet18_Weights


class ResNet(nn.Module):

    def __init__(self):
        """Generic resnet video generator.

        Args:
            block (nn.Module): resnet building block
            conv_makers (list(functions)): generator function for each layer
            layers (List[int]): number of blocks per layer
            stem (nn.Module, optional): Resnet stem, if None, defaults to conv-bn-relu. Defaults to None.
            num_classes (int, optional): Dimension of the final FC layer. Defaults to 400.
            zero_init_residual (bool, optional): Zero init bottleneck residual BN. Defaults to False.
        """
        super(ResNet, self).__init__()
        # self.backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.backbone = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
        # self.backbone = resnet101(weights=ResNet101_Weights.IMAGENET1K_V2)
        self.con1 = self.backbone.conv1
        self.bt1 = self.backbone.bn1
        self.re = self.backbone.relu


    def forward(self, x):
        x = self.re(self.bt1(self.con1(x)))
        x = self.backbone.layer1(x)
        x1 = x
        x = self.backbone.layer2(x)
        x2 = x
        x = self.backbone.layer3(x)
        x3 = x
        x = self.backbone.layer4(x)

        return x, x1, x2, x3


if __name__ == '__main__':
    x = torch.rand([2, 3, 256, 256], device='cuda')
    network = ResNet().to('cuda')
    print(network(x)[0].shape)
    print(network(x)[1].shape)
    print(network(x)[2].shape)
    print(network(x)[3].shape)
