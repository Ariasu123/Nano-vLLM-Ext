# RoPE（Rotary Position Embedding，旋转位置编码）让 Attention 知道 token 的先后位置。
# 它把 Q/K 向量的每两个相关分量看成二维坐标，再按位置对应角度旋转。

from functools import lru_cache
import torch
from torch import nn


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    # 先转 float32 提高三角运算精度，再把最后一维平均分成两半。
    x1, x2 = torch.chunk(x.float(), 2, dim=-1)

    # 这是二维旋转公式，等价于复数乘法：
    # (x1 + i*x2) * (cos + i*sin)。
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    # cat 沿最后一维把两半拼回去，再恢复输入 dtype。
    return torch.cat((y1, y2), dim=-1).to(x.dtype)


class RotaryEmbedding(nn.Module):

    # 初始化时预计算从位置 0 到最大长度的全部 cos/sin，避免每次 forward 重算。
    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        assert rotary_dim == head_size

        # 不同维度使用不同频率：低维变化快，高维变化慢。
        inv_freq = 1.0 / (base**(torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))

        # t 是所有可能的位置 [0,1,2,...,max_position-1]。
        t = torch.arange(max_position_embeddings, dtype=torch.float)

        # einsum("i,j->ij") 计算外积，得到 [位置数, rotary_dim/2] 的角度表。
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        # 插入长度为 1 的 head 维，使形状 [position,1,rotary_dim] 可广播到所有注意力头。
        cache = torch.cat((cos, sin), dim=-1).unsqueeze_(1)

        # buffer 会跟随模型移动设备，但不是可训练参数。
        # persistent=False 表示不写入 checkpoint，因为它可以根据配置重新计算。
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    # torch.compile 会尝试把多个 PyTorch 操作编译、融合成更高效的 GPU 执行。
    @torch.compile
    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # 用 positions 直接索引预计算表，取出本轮 token 对应的 cos/sin。
        cos_sin = self.cos_sin_cache[positions]
        cos, sin = cos_sin.chunk(2, dim=-1)
        query = apply_rotary_emb(query, cos, sin)
        key = apply_rotary_emb(key, cos, sin)
        return query, key


# lru_cache(1) 记住最近一次返回值。
# 所有配置相同的 Decoder 层可复用同一个 RoPE 表，避免重复分配。
@lru_cache(1)
def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
):
    rotary_emb = RotaryEmbedding(head_size, rotary_dim, max_position, base)
    return rotary_emb
