# 【这个文件做什么】
# Config 集中保存整个推理引擎的配置。
# 用户传入的是模型路径和少量可选参数；这里还会从 Hugging Face 配置中读取模型结构信息。
import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass(slots=True)
class Config:
    # @dataclass 自动生成 __init__，所以不用手写大量 self.xxx = xxx。
    # 字段冒号后的内容是类型标注，等号右侧是默认值。

    # 本地模型目录。
    model: str

    # 一个 Prefill 批次最多处理多少 token。
    max_num_batched_tokens: int = 16384

    # GPU 批次中最多同时处理多少条 Sequence。
    max_num_seqs: int = 512

    # 用户允许的最大上下文长度。
    max_model_len: int = 4096

    # 最多使用 GPU 总显存的比例；留一部分给驱动和其他程序。
    gpu_memory_utilization: float = 0.9

    # 张量并行使用的 GPU 数量。
    tensor_parallel_size: int = 1

    # True 强制使用普通 Eager 执行；False 时允许 CUDA Graph。
    enforce_eager: bool = False

    # `AutoConfig | None` 表示该字段可以是 Hugging Face 配置，也可以暂时为空。
    hf_config: AutoConfig | None = None

    # EOS token id 会在分词器加载后填写，-1 是尚未填写的占位值。
    eos: int = -1

    # 每个物理 KV Cache 块容纳多少 token。
    kvcache_block_size: int = 256

    # 可分配多少 KV Cache 块，要等 ModelRunner 预热并检查显存后计算。
    num_kvcache_blocks: int = -1

    # 数据类完成自动 __init__ 后，会自动调用 __post_init__ 做校验和补充初始化。
    def __post_init__(self):
        # assert 条件为 False 时立即报错，避免初始化 GPU 后才发现配置无效。
        assert os.path.isdir(self.model)

        # 当前 FlashAttention 分页缓存要求块大小是 256 的倍数。
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8

        # 读取模型目录中的 config.json，得到层数、头数、隐藏维度、dtype 等。
        self.hf_config = AutoConfig.from_pretrained(self.model)

        # 最终长度取“用户限制”和“模型本身上限”中的较小值。
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
