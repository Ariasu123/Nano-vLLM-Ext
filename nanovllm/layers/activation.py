# 【这个文件做什么】
# 实现 Qwen3 MLP 使用的 SwiGLU 激活：
# output = SiLU(gate) * up。
# gate 决定哪些信息通过，up 提供被保留的内容。
import torch
from torch import nn
import torch.nn.functional as F


class SiluAndMul(nn.Module):

    # torch.compile 会尝试把切分、SiLU 和乘法融合成更少的 GPU 内核。
    # 将gate和value合成一个矩阵，减少一件矩阵计算以及kernel调用
    @torch.compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # chunk(2,-1) 沿最后一维平均切成两份。
        x, y = x.chunk(2, -1)

        # SiLU(z)=z*sigmoid(z)，结果再与另一半逐元素相乘。
        return F.silu(x) * y
