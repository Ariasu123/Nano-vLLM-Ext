# RMSNorm 用来控制隐藏向量的数值尺度，让深层网络更稳定。
# 它计算每个 token 向量的均方根，再将向量除以该尺度，最后乘可学习权重。

# 与 LayerNorm 相比，RMSNorm 不减均值，只按均方根缩放。
# 本实现还有“残差相加 + RMSNorm”的融合版本，可减少一次显存读写。
import torch
from torch import nn


class RMSNorm(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        # eps 是防止除以零的小常数。
        self.eps = eps

        # 每个隐藏维度都有一个可学习缩放系数，初始值为 1。
        self.weight = nn.Parameter(torch.ones(hidden_size))

    # torch.compile 尝试将下列逐元素操作融合。
    @torch.compile
    def rms_forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        orig_dtype = x.dtype

        # 用 float32 计算统计量，减少 float16/bfloat16 的数值误差。
        x = x.float()

        # pow(2) 平方，mean(...,-1) 对每个 token 的隐藏维求平均。
        # keepdim=True 保留长度为 1 的维度，使结果可广播回 x。
        var = x.pow(2).mean(dim=-1, keepdim=True)

        # rsqrt(z) = 1/sqrt(z)；mul_ 是原地乘法。
        x.mul_(torch.rsqrt(var + self.eps))
        x = x.to(orig_dtype).mul_(self.weight)
        return x

    # 融合“x + residual”与 RMSNorm，同时返回相加后的残差流。
    @torch.compile
    def add_rms_forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        orig_dtype = x.dtype
        x = x.float().add_(residual.float())

        # residual 保存未归一化的相加结果，供下一子层继续累积。
        residual = x.to(orig_dtype)
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + self.eps))
        x = x.to(orig_dtype).mul_(self.weight)
        return x, residual

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            return self.rms_forward(x)
        else:
            return self.add_rms_forward(x, residual)
