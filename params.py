import torch
import torch.nn as nn
from thop import profile
from torchinfo import summary
import numpy as np

from my_models.Model import Model



def calculate_metrics(model):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()  
    
    input_tensor = torch.randn(1, 3, 256, 256).to(device)

    flops, params = profile(model, inputs=(input_tensor,))
    
    params_m = params / 1e6  
    flops_g = flops / 1e9    
    
    return params_m, flops_g


if __name__ == "__main__":
    num_classes = 3
    model = Model(num_classes=num_classes).cuda()
    
    params_m, flops_g = calculate_metrics(model)
    
    print(f"模型参数量: {params_m:.2f} M")
    print(f"浮点运算量: {flops_g:.2f} G")