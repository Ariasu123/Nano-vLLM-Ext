# 【这个文件做什么】
# 这里用 PyTorch 重新搭建了 Qwen3 的推理结构。
# 权重并不在代码里，而是由 loader.py 从模型目录加载。
#
# 【一枚 token 的完整路线】
# token id
# -> Embedding（查表得到向量）
# -> 多层 DecoderLayer（Attention + MLP）
# -> 最终 RMSNorm
# -> LM Head（把向量投影为整个词表的 logits）
# -> Sampler 选择下一个 token。
#
# 【常见形状】
# 本项目为了配合变长批处理，把不同请求的 token 拼成一维：
# input_ids: [num_tokens]
# hidden_states: [num_tokens, hidden_size]
# Q/K/V: [num_tokens, num_heads, head_dim]
#
# 【建议的阅读位置】
# 先关注每个 forward 中 Tensor 如何改变形状，不必一开始就研究张量并行权重加载。
import torch
from torch import nn
import torch.distributed as dist
from transformers import Qwen3Config

from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.attention import Attention
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.linear import QKVParallelLinear, MergedColumnParallelLinear, RowParallelLinear
from nanovllm.layers.rotary_embedding import get_rope
from nanovllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead


class Qwen3Attention(nn.Module):

    # 这个模块完成一层自注意力：
    # hidden_states -> Q/K/V 投影 -> Q/K 位置编码 -> Attention -> 输出投影。
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int = 4096 * 32,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-06,
        qkv_bias: bool = False,
        rope_theta: float = 10000,
        rope_scaling: dict | None = None,
    ) -> None:
        super().__init__()

        # GQA（Grouped Query Attention，分组查询注意力）允许多个 Q 头共享一组 K/V 头。
        # 例：32 个 Q 头、8 个 KV 头时，每 4 个 Q 头共享 1 组 K/V。
        # KV Cache 只保存 8 个头，而不是 32 个头，因此显著节省显存。

        # tp_size 是张量并行 GPU 数量；注意力头平均分到各 GPU。
        tp_size = dist.get_world_size()
        self.total_num_heads = num_heads

        # assert 保证头数能整除 GPU 数，否则每张 GPU 无法获得相同大小的分片。
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        assert self.total_num_kv_heads % tp_size == 0
        self.num_kv_heads = self.total_num_kv_heads // tp_size

        # 每个注意力头的维度通常是 hidden_size / 总 Q 头数。
        # `a or b` 在 a 为 None/0 时使用 b。
        self.head_dim = head_dim or hidden_size // self.total_num_heads

        # 当前 GPU 上，Q/K/V 展平后的最后一维大小。
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim

        # 点积注意力要除以 sqrt(head_dim)，等价于乘 head_dim**-0.5，
        # 防止维度增大时 Q·K 数值过大、softmax 过于尖锐。
        self.scaling = self.head_dim ** -0.5
        self.qkv_bias = qkv_bias

        # 一次线性变换同时计算 Q、K、V，输出随后再切开。
        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=qkv_bias,
        )
        # Attention 多头结果拼接后，通过输出投影回到 hidden_size。
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
        )
        if isinstance(rope_scaling, dict):
            # dict.get("key", default) 在 key 不存在时返回默认值。
            rope_theta = rope_scaling.get("rope_theta", rope_theta)

        # RoPE（旋转位置编码）把 token 位置信息写入 Q/K。
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=rope_theta,
        )
        # Attention 类负责写入 KV Cache，并调用 FlashAttention 内核。
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
        )
        if not self.qkv_bias:
            # 某些 Qwen3 配置在每个注意力头内部对 Q/K 再做 RMSNorm。
            self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

    # positions: [num_tokens]，每个 token 的绝对位置。
    # hidden_states: [num_tokens, hidden_size]。
    # 返回: [num_tokens, hidden_size]。
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        # QKV 投影后最后一维按 [Q部分, K部分, V部分] 排列。
        qkv = self.qkv_proj(hidden_states)

        # split 根据给定长度切分最后一维。
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        # view 只改变 Tensor 的“观察形状”，不复制数据。
        # -1 让 PyTorch 自动推断 token 数。
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        if not self.qkv_bias:
            q = self.q_norm(q)
            k = self.k_norm(k)
        # 只旋转 Q/K，因为注意力权重由 Q 与 K 的关系决定；V 不参与位置相似度计算。
        q, k = self.rotary_emb(positions, q, k)

        # 内部先把新 K/V 写进当前层缓存，再选择 Prefill 或 Decode 内核。
        o = self.attn(q, k, v)

        # flatten(1,-1) 把 head 和 head_dim 合并回一个特征维度。
        output = self.o_proj(o.flatten(1, -1))
        return output


class Qwen3MLP(nn.Module):

    # MLP 是每个 Decoder 层中 Attention 之后的前馈网络。
    # Qwen3 使用 SwiGLU：SiLU(gate) * up，然后再投影回 hidden_size。
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
    ) -> None:
        super().__init__()
        # gate 和 up 原本是两个独立线性层，这里合并为一次更大的矩阵乘。
        # [intermediate_size] * 2 等价于 [intermediate_size, intermediate_size]。
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
        )
        assert hidden_act == "silu"
        self.act_fn = SiluAndMul()

    def forward(self, x):
        # 输出最后一维包含拼接的 gate 和 up 两部分。
        gate_up = self.gate_up_proj(x)

        # SiluAndMul 将最后一维一分为二并计算 SiLU(gate) * up。
        x = self.act_fn(gate_up)

        # down_proj 将扩大后的中间维度降回 hidden_size。
        x = self.down_proj(x)
        return x


class Qwen3DecoderLayer(nn.Module):

    # 一层 Decoder 包含两个子模块：
    # 1. Self-Attention：让 token 读取前文信息；
    # 2. MLP：对每个 token 的表示做非线性变换。
    # 两个子模块前都有 RMSNorm，并通过 residual（残差）保留原信息。
    def __init__(
        self,
        config: Qwen3Config,
    ) -> None:
        super().__init__()
        self.self_attn = Qwen3Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, 'attention_bias', True),
            head_dim=getattr(config, 'head_dim', None),
            rope_theta=getattr(config, "rope_theta", 1000000),
            rope_scaling=getattr(config, "rope_scaling", None),
        )
        self.mlp = Qwen3MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    # residual 是尚未与当前 hidden_states 相加的残差流。
    # 返回新的 hidden_states 和更新后的 residual，交给下一层继续使用。
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # 第一层刚进入时 residual=None，因此先把原 embedding 保存为残差。
        if residual is None:
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            # 其他层使用融合版本：先做 hidden_states + residual，再进行 RMSNorm。
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        hidden_states = self.self_attn(positions, hidden_states)

        # 将 Attention 输出加到残差流，再归一化后送进 MLP。
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class Qwen3Model(nn.Module):

    # Qwen3 主干：Embedding + 多个 DecoderLayer + 最终 RMSNorm。
    def __init__(
        self,
        config: Qwen3Config,
    ) -> None:
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        # ModuleList 与普通 list 类似，但能让 PyTorch 正确发现并管理其中各层参数。
        self.layers = nn.ModuleList([Qwen3DecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    # input_ids: [num_tokens]；positions: [num_tokens]。
    # 返回最后一层隐藏状态 [num_tokens, hidden_size]，尚未变成词表 logits。
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        # Embedding 可以理解为查表：用 token id 找到对应的 hidden_size 维向量。
        hidden_states = self.embed_tokens(input_ids)
        residual = None

        # 所有 Decoder 层依次处理同一批 token；每层都有自己独立的 K/V Cache。
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)

        # 最后一次把 MLP 输出加入残差并归一化。
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class Qwen3ForCausalLM(nn.Module):
    # Hugging Face 权重中 Q/K/V、gate/up 是独立参数，而本实现为效率将它们合并。
    # 此映射告诉 loader 应把独立权重写入融合参数的哪一部分。
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(
        self,
        config: Qwen3Config
    ) -> None:
        super().__init__()
        self.model = Qwen3Model(config)

        # LM Head 把 hidden_size 维向量投影成 vocab_size 个 logits。
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        if config.tie_word_embeddings:
            # 某些模型让输入 Embedding 与输出 LM Head 共用同一份权重，以减少参数量。
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    # forward 只运行模型主干，返回隐藏状态。
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(input_ids, positions)

    # 单独计算 logits，便于 ModelRunner 在 CUDA Graph 外只处理真实 batch 的输出。
    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return self.lm_head(hidden_states)
