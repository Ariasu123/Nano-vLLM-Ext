# Context 保存“当前这一批请求”的公共元数据，例如本轮是 Prefill 还是 Decode、
# token 应写到哪个 KV Cache 槽位，以及每条请求的块表。

# 【为什么使用全局 Context】
# Qwen3 有很多层，每层 Attention 都需要相同的批次信息。
# 如果把这些参数从 Qwen3Model 一层层传到每个 Attention，函数签名会很长。
# 本项目选择在每次模型前向前设置一次全局 Context，各层需要时直接 get_context()。

# 【注意】
# 这里的全局变量只在当前进程内共享；张量并行的每个进程都有自己的 _CONTEXT。
# 每次 run 结束必须 reset_context，否则下一批可能错误使用上一批的块表。
from dataclasses import dataclass
import torch


@dataclass(slots=True)
class Context:
    # @dataclass 会根据这些字段自动生成 __init__ 等常用方法。
    # slots=True 禁止对象随意增加新属性，并能略微减少内存占用。

    is_prefill: bool = False

    # cu_seqlens_q/k 是多条变长序列的累计边界，长度为 batch_size+1，带一个结束位置。
    # 它们只在 Prefill 使用。
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None

    # 当前批次中最长的 Query 和 Key 序列长度，FlashAttention 用它选择内核配置。
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0

    # slot_mapping[i] 表示第 i 个新 token 的 K/V 应写到哪个扁平缓存槽。
    slot_mapping: torch.Tensor | None = None

    # Decode 时每条请求的真实上下文长度。
    context_lens: torch.Tensor | None = None

    # block_tables[请求下标, 逻辑块下标] = 物理块 id。
    block_tables: torch.Tensor | None = None

# 模块级变量只创建一次，导入本模块的代码访问的是同一个对象。
_CONTEXT = Context()

def get_context():
    # 返回当前全局 Context；这里不复制对象。
    return _CONTEXT

def set_context(is_prefill, cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=0, max_seqlen_k=0, slot_mapping=None, context_lens=None, block_tables=None):
    # global 告诉 Python：这里要修改模块级 _CONTEXT，而不是创建同名局部变量。
    global _CONTEXT

    # 每次整体创建新对象，确保 Prefill/Decode 不需要的字段恢复为默认值，
    # 不会残留上一批的数据。
    _CONTEXT = Context(is_prefill, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, context_lens, block_tables)

def reset_context():
    global _CONTEXT

    # 无参数 Context() 会使用数据类中声明的全部默认值。
    _CONTEXT = Context()
