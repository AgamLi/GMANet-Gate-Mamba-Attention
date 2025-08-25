import torch
from torch import nn

"Efficient Multi-Scale Attention Module with Cross-Spatial Learning"

class EMA(nn.Module):
    def __init__(self, channels, c2=None, factor=32):
        super(EMA, self).__init__()
        self.groups = factor
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        #(B,C,H,W)
        b, c, h, w = x.size()

        
        group_x = x.reshape(b * self.groups, -1, h, w)  # : (B,C,H,W)-->(B*G,C/G,H,W)
        x_h = self.pool_h(group_x) # : (B*G,C/G,H,W)-->(B*G,C/G,H,1)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2) # : (B*G,C/G,H,W)-->(B*G,C/G,1,W)-->(B*G,C/G,W,1)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))# : (B*G,C/G,H+W,1)
        x_h, x_w = torch.split(hw, [h, w], dim=2) # : x_h:(B*G,C/G,H,1); x_w:(B*G,C/G,W,1)

       
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid()) #  (B*G,C/G,H,W) * (B*G,C/G,H,1) * (B*G,C/G,1,W)=(B*G,C/G,H,W)
        x2 = self.conv3x3(group_x) # : (B*G,C/G,H,W)-->(B*G,C/G,H,W)

        
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1)) #  (B*G,C/G,H,W)-->agp-->(B*G,C/G,1,1)-->reshape-->(B*G,C/G,1)-->permute-->(B*G,1,C/G)
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)  #  (B*G,C/G,H,W)-->reshape-->(B*G,C/G,H*W)
        y1 = torch.matmul(x11, x12) # (B*G,1,C/G) @ (B*G,C/G,H*W) = (B*G,1,H*W)

        
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1)) # (B*G,C/G,H,W)-->agp-->(B*G,C/G,1,1)-->reshape-->(B*G,C/G,1)-->permute-->(B*G,1,C/G)
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw  #  (B*G,C/G,H,W)-->reshape-->(B*G,C/G,H*W)
        y2 = torch.matmul(x21, x22)  # (B*G,1,C/G) @ (B*G,C/G,H*W) = (B*G,1,H*W)

        
        weights = (y1+y2).reshape(b * self.groups, 1, h, w)  # (B*G,1,H*W)-->reshape-->(B*G,1,H,W)
        weights_ =  weights.sigmoid() #  (B*G,1,H,W)
        out = (group_x * weights_).reshape(b, c, h, w) # (B*G,C/G,H,W)*(B*G,1,H,W)==(B*G,C/G,H,W)-->reshape(B,C,H,W)
        return out


if __name__ == '__main__':
    # (B,C,H,W)
    input=torch.randn(4,32,64,64)
    Model = EMA(channels=32)
    output=Model(input)
    print(output.shape)
