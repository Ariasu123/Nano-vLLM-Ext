# 【这个文件做什么】
# 这里完成自注意力中最核心的计算，并管理每一层的 K/V Cache：
# 1. 使用自定义 Triton 内核把本轮新 K/V 写到指定物理槽位；
# 2. Prefill 时调用支持变长序列的 FlashAttention；
# 3. Decode 时从分页 KV Cache 读取完整历史。
#
# 【Q、K、V 的直观理解】
# Q（Query）表示“当前 token 想寻找什么信息”；
# K（Key）表示“每个历史 token 能提供什么线索”；
# V（Value）表示“真正要汇总的内容”。
# Q 与 K 的相似度经过 softmax 后成为权重，再对 V 做加权求和。
#
# 【Triton 与 FlashAttention】
# Triton 是编写 GPU 内核的 Python 风格语言；这里用它完成简单而高效的缓存写入。
# FlashAttention 是优化后的注意力实现，数学结果与普通 Attention 一致，但减少显存读写。
import torch
from torch import nn
import triton
import triton.language as tl

from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from nanovllm.utils.context import get_context


# @triton.jit 是装饰器：第一次调用时会把下面函数编译成 GPU 内核。
@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    # 启动网格为 (N,)，所以每个 program 负责一个 token。
    idx = tl.program_id(0)

    # 读取这个 token 应写入的缓存槽位。
    slot = tl.load(slot_mapping_ptr + idx)

    # -1 表示 CUDA Graph 中多出来的无效填充行。
    if slot == -1: return

    # tl.arange(0,D) 生成 0 到 D-1，用来并行读写当前 token 的全部 K/V 元素。
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    # 缓存可看成 [num_slots, D]，所以第 slot 行从 slot*D 开始。
    cache_offsets = slot * D + tl.arange(0, D)

    # 将结果写入 GPU 全局内存。
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)

# 普通 Python 包装函数：检查 Tensor 内存布局，再启动 Triton 内核。
def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    # key/value 形状为 [N, 本地KV头数, head_dim]，N 是本轮新 token 数。
    N, num_heads, head_dim = key.shape

    # 将“头数 × 每头维度”展平为每个 token 的总元素数 D。
    D = num_heads * head_dim

    # stride 表示某一维相邻元素在内存中相隔多少个元素。
    # 下列 assert 保证 Triton 内核假设的连续布局成立。
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N

    # [(N,)] 是 Triton 启动语法：创建 N 个并行 program。
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)


class Attention(nn.Module):

    # num_heads 和 num_kv_heads 都是“当前 GPU 本地”的头数。
    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        # 先用空 Tensor 占位；ModelRunner 分配大缓存后会替换为当前层的视图。
        self.k_cache = self.v_cache = torch.tensor([])

    # q: [新token数, 本地Q头数, head_dim]
    # k/v: [新token数, 本地KV头数, head_dim]
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        # Context 提供当前阶段、序列边界、块表和缓存写入位置。
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache

        # numel() 返回 Tensor 元素总数；预热阶段缓存为空，因此跳过写入。
        if k_cache.numel() and v_cache.numel():
            # 当前新 K/V 必须保存，之后的 Decode 才能直接复用。
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)

        if context.is_prefill:
            if context.block_tables is not None:
                # 有 Prefix Cache 时，旧 K/V 和新 K/V 都在分页缓存中。
                # block_table 告诉 FlashAttention 按什么顺序读取物理块。
                k, v = k_cache, v_cache

            # varlen 是 variable length（变长）的缩写。
            # cu_seqlens 把拼接 token 流重新划分为互不干扰的请求。
            o = flash_attn_varlen_func(q, k, v,
                                       max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                       max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                       softmax_scale=self.scale, causal=True, block_table=context.block_tables)
        else:
            # Decode 中每条请求只有一个 Query。
            # unsqueeze(1) 在下标 1 插入长度为 1 的维度：
            # [batch, head, dim] -> [batch, query_len=1, head, dim]。
            o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                        cache_seqlens=context.context_lens, block_table=context.block_tables, 
                                        softmax_scale=self.scale, causal=True)
        # o 是每个 Query 从历史 Value 中汇总出的上下文向量。
        return o
