# 【这个文件做什么】
# 这里实现词表并行的 Embedding 和 LM Head。
# 两者其实使用形状相同的权重 [vocab_size, hidden_size]：
# Embedding 用 token id 查一行；LM Head 用隐藏向量与每一行做点积，得到词表 logits。
#
# 【词表并行示例】
# 假设词表有 1000 个 token、两张 GPU：
# rank 0 保存 token 0~499，rank 1 保存 token 500~999。
# 这样每张 GPU 只保存一半词表权重。
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from nanovllm.utils.context import get_context


class VocabParallelEmbedding(nn.Module):

    # num_embeddings 是词表大小，embedding_dim 是每个 token 向量维度。
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
    ):
        super().__init__()
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()

        # 为了平均切分，词表大小必须整除 GPU 数。
        assert num_embeddings % self.tp_size == 0
        self.num_embeddings = num_embeddings
        self.num_embeddings_per_partition = self.num_embeddings // self.tp_size
        # 当前 rank 负责半开区间 [start,end)，包含 start、不包含 end。
        self.vocab_start_idx = self.num_embeddings_per_partition * self.tp_rank
        self.vocab_end_idx = self.vocab_start_idx + self.num_embeddings_per_partition
        # 每张卡只创建本地词表大小的权重。
        self.weight = nn.Parameter(torch.empty(self.num_embeddings_per_partition, embedding_dim))
        self.weight.weight_loader = self.weight_loader

    # 从完整 checkpoint 的第 0 维取出当前 rank 的词表分片。
    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(0)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(0, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    # x 是 token id Tensor，形状通常为 [num_tokens]。
    def forward(self, x: torch.Tensor):
        if self.tp_size > 1:
            # mask 标记哪些 token 属于当前 rank 的词表范围。
            mask = (x >= self.vocab_start_idx) & (x < self.vocab_end_idx)

            # 全局 token id 要减去 start，变成本地权重下标。
            # 不属于本 rank 的位置通过 mask 变成 0，先做一次安全的占位查询。
            x = mask * (x - self.vocab_start_idx)

        # embedding 相当于按 x 中每个整数从 weight 查一行。
        y = F.embedding(x, self.weight)
        if self.tp_size > 1:
            # unsqueeze(1) 把 [num_tokens] 变成 [num_tokens,1]，
            # 这样 mask 可以广播到 [num_tokens, hidden_size]。
            y = mask.unsqueeze(1) * y

            # 每个 token 只在一张 GPU 上非零；all_reduce 相加后每张卡都得到完整 embedding。
            dist.all_reduce(y)
        return y


class ParallelLMHead(VocabParallelEmbedding):

    # LM Head 复用词表并行权重布局，但计算方向从 hidden_state 到 logits。
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        bias: bool = False,
    ):
        # 当前实现没有并行 LM Head bias。
        assert not bias
        super().__init__(num_embeddings, embedding_dim)

    # x: [候选位置数, hidden_size]
    # 返回 rank 0 上的完整 logits: [请求数, vocab_size]。
    def forward(self, x: torch.Tensor):
        context = get_context()
        if context.is_prefill:
            # Prefill 计算了 prompt 中很多位置，但生成下一个 token 只需要每条请求最后一个位置。
            # 例：cu_seqlens_q=[0,3,5]，两个末位下标是 3-1=2、5-1=4。
            last_indices = context.cu_seqlens_q[1:] - 1

            # contiguous 确保筛选后的 Tensor 在内存中连续，便于后续矩阵乘。
            x = x[last_indices].contiguous()

        # 每张 GPU 只计算自己负责词表区间的 logits。
        logits = F.linear(x, self.weight)
        if self.tp_size > 1:
            # 只有 rank 0 需要准备列表接收所有 GPU 的 logits 分片。
            all_logits = [torch.empty_like(logits) for _ in range(self.tp_size)] if self.tp_rank == 0 else None

            # gather 将每张 GPU 的分片收集到目标 rank 0。
            dist.gather(logits, all_logits, 0)

            # cat(...,-1) 沿最后一维拼接，恢复完整词表顺序。
            logits = torch.cat(all_logits, -1) if self.tp_rank == 0 else None
        return logits
