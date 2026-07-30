import os
from dataclasses import dataclass
from transformers import AutoConfig

# @dataclass 自动生成 __init__，所以不用手写大量 self.xxx = xxx
@dataclass(slots=True)
class Config:
    
    model: str                            # 本地模型目录。
    max_num_batched_tokens: int = 16384   # 一个 Prefill 批次最多处理多少 token
    max_num_seqs: int = 512               # GPU 批次中最多同时处理多少条 Sequence
    max_model_len: int = 4096             # 用户允许的最大上下文长度
    gpu_memory_utilization: float = 0.9   # 最多使用 GPU 总显存的比例
    tensor_parallel_size: int = 1         # 张量并行使用的 GPU 数量
    enforce_eager: bool = False           # True 强制使用普通 Eager 执行；False 时允许 CUDA Graph
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256         # 每个物理 KV Cache 块容纳多少 token。
    num_kvcache_blocks: int = -1          # 可分配多少 KV Cache 块，要等 ModelRunner 预热并检查显存后计算

    # 数据类完成自动 __init__ 后，会自动调用 __post_init__ 做校验和补充初始化。
    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8

        # 读取模型目录中的 config.json，得到层数、头数、隐藏维度、dtype 等
        self.hf_config = AutoConfig.from_pretrained(self.model)

        # 最终长度取“用户限制”和“模型本身上限”中的较小值。
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
