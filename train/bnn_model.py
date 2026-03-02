import torch
import torch.nn as nn
import torch.nn.functional as F

class Binarize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight):
        return torch.where(weight >= 0, torch.tensor(1.0, device=weight.device), torch.tensor(-1.0, device=weight.device))

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

class BNNClassifier(nn.Module):
    def __init__(self):
        super(BNNClassifier, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(1, 8704))
        nn.init.uniform_(self.weight, -1.0, 1.0)
        
        # 【最强抗过拟合魔法】：随机致盲 50% 的像素
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        # 让输入经过 Dropout（注意：PyTorch 会在评估和导出模型时自动关闭它）
        x = self.dropout(x)
        bin_weight = Binarize.apply(self.weight)
        return F.linear(x, bin_weight, bias=None)

def unpack_to_bits(x_float):
    x_byte = (x_float * 255).to(torch.uint8)
    bits = []
    for i in range(8):
        bit = (x_byte >> i) & 1
        bit_float = bit.float() * 2.0 - 1.0
        bits.append(bit_float)
    x_bits = torch.stack(bits, dim=-1).view(x_float.shape[0], -1)
    return x_bits