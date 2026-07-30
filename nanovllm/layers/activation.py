# 实现 Qwen3 MLP 使用的 SwiGLU 激活：
# output = SiLU(gate) * up
# gate 决定哪些信息通过，up 提供被保留的内容
import torch
from torch import nn
import torch.nn.functional as F


class SiluAndMul(nn.Module):

    # torch.compile 会对计算图进行优化，尝试融合切分、SiLU 和乘法等连续算子， 减少 GPU kernel 启动次数，提高推理性能
    # 将gate和up合成一个矩阵，减少一件矩阵计算以及kernel调用
    @torch.compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        x, y = x.chunk(2, -1)   # chunk(2,-1) 沿最后一维平均切成两份

        # SiLU(x)=x*sigmoid(x)，结果再与另一半(y)逐元素相乘。
        return F.silu(x) * y
