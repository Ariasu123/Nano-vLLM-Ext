# 实现模型的张量并行 Linear（线性层）。
# 普通线性层计算 y = x @ W^T + bias；当 W 太大时，可把它切到多张 GPU。
#
# 【用两张 GPU 举例】
# 假设完整权重 W 形状为 [8,4]：
# - Column Parallel（列并行）按输出维切成两个 [4,4]，每张卡算 4 个不同输出特征；
# - Row Parallel（行并行）按输入维切成两个 [8,2]，每张卡算一部分和，最后相加。
#
# 【两个通信操作】
# all_reduce：每张 GPU 都提供一个 Tensor，将它们相加后，每张 GPU 都得到结果。
# gather：收集每张 GPU 的不同分片，只让指定 GPU 得到完整结果。

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist


def divide(numerator, denominator):
    # assert 确保能整除；否则无法平均切到每张 GPU。
    assert numerator % denominator == 0
    return numerator // denominator


class LinearBase(nn.Module):

    # 所有并行 Linear 的公共基类，只负责创建参数和记录并行信息。
    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        tp_dim: int | None = None,
    ):
        super().__init__()

        # tp_dim 指加载完整 checkpoint 时沿哪一维切片。
        # 0 是输出维，1 是输入维，None 表示不切。
        self.tp_dim = tp_dim
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()

        # PyTorch Linear 的 weight 形状是 [output_size, input_size]。
        self.weight = nn.Parameter(torch.empty(output_size, input_size))

        # Python 对象可以动态添加属性。
        # loader.py 会通过这个属性找到当前层专用的权重加载方法。
        self.weight.weight_loader = self.weight_loader
        if bias:
            self.bias = nn.Parameter(torch.empty(output_size))
            self.bias.weight_loader = self.weight_loader
        else:
            # 正式注册一个值为 None 的参数，使 state_dict 等 PyTorch 机制知道本层没有 bias。
            self.register_parameter("bias", None)

    # 基类留下占位，子类必须重写forward。
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class ReplicatedLinear(LinearBase):
    # Replicated 表示“不分片”：每张 GPU 都保存完整相同权重。

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        super().__init__(input_size, output_size, bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        # copy_ 原地复制数据，不创建新的 Parameter 对象。
        param.data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class ColumnParallelLinear(LinearBase):
    # 沿输出维切分。输入 x 在各 GPU 相同，但每张卡计算不同的输出特征。

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        tp_size = dist.get_world_size()

        # 每张卡只创建 output_size / tp_size 行权重。
        super().__init__(input_size, divide(output_size, tp_size), bias, 0)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data

        # size(0) 是当前 GPU 参数在输出维的长度。
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size

        # narrow(dim,start,length) 从完整权重取出当前 rank 的连续分片。
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # F.linear 完成 x @ weight.T + bias。
        # 返回的最后一维只是完整输出的一段，此处不需要 GPU 通信。
        return F.linear(x, self.weight, self.bias)


class MergedColumnParallelLinear(ColumnParallelLinear):
    # 将多个独立列并行层拼成一个大层，例如 MLP 的 gate_proj 与 up_proj。

    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        bias: bool = False,
    ):
        self.output_sizes = output_sizes
        super().__init__(input_size, sum(output_sizes), bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: int):
        param_data = param.data

        # loaded_shard_id 表示当前加载的是第几个原始权重。
        # shard_offset 是它在本地融合参数中的起始位置。
        shard_offset = sum(self.output_sizes[:loaded_shard_id]) // self.tp_size
        shard_size = self.output_sizes[loaded_shard_id] // self.tp_size
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        # chunk 把完整权重平均切成 tp_size 份，再取当前 rank 对应的一份。
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)


class QKVParallelLinear(ColumnParallelLinear):
    # QKV 专用融合层。输出内存按 [本地Q, 本地K, 本地V] 排列。

    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int | None = None,
        bias: bool = False,
    ):
        tp_size = dist.get_world_size()

        # 没有单独指定 KV 头时，退化为普通多头注意力：KV 头数等于 Q 头数。
        total_num_kv_heads = total_num_kv_heads or total_num_heads
        self.head_size = head_size
        self.num_heads = divide(total_num_heads, tp_size)
        self.num_kv_heads = divide(total_num_kv_heads, tp_size)
        # K 和 V 各占一份，所以总头数是 Q + 2*KV。
        output_size = (total_num_heads + 2 * total_num_kv_heads) * self.head_size
        super().__init__(hidden_size, output_size, bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: str):
        param_data = param.data
        assert loaded_shard_id in ["q", "k", "v"]

        # 根据 q/k/v 计算它在融合参数中的本地长度和起点。
        if loaded_shard_id == "q":
            shard_size = self.num_heads * self.head_size
            shard_offset = 0
        elif loaded_shard_id == "k":
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size
        else:
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size + self.num_kv_heads * self.head_size
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)


class RowParallelLinear(LinearBase):
    # 沿输入维切分。每张卡得到完整输出形状，但数值只是部分和。

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        tp_size = dist.get_world_size()

        # 每张卡只接收 input_size / tp_size 个输入特征。
        super().__init__(divide(input_size, tp_size), output_size, bias, 1)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data

        # 一维参数通常是 bias，不沿输入维切分，每张卡加载相同副本。
        if param_data.ndim == 1:
            param_data.copy_(loaded_weight)
            return
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 每张卡先计算局部部分和。bias 只在 rank 0 加一次，
        # 否则 all_reduce 相加后 bias 会被重复 tp_size 次。
        y = F.linear(x, self.weight, self.bias if self.tp_rank == 0 else None)
        if self.tp_size > 1:
            # all_reduce 将所有 GPU 的 y 相加，并把完整结果发回每张 GPU。
            dist.all_reduce(y)
        return y
