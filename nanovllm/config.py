import os
from dataclasses import dataclass

# transformers 是较重的依赖（且间接依赖 torch）。这里不在模块顶层导入，改到 __post_init__ 内
# 按需导入：纯逻辑单测（scheduler/block_manager）只需构造假 config，无需安装 transformers/torch。

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
    hf_config: "AutoConfig | None" = None   # 字符串注解：避免类定义期就需要 transformers。
    eos: int = -1
    kvcache_block_size: int = 256         # 每个物理 KV Cache 块容纳多少 token。
    num_kvcache_blocks: int = -1          # 可分配多少 KV Cache 块，要等 ModelRunner 预热并检查显存后计算

    # ---------- 功能一：Chunked Prefill（vLLM 式 prefill/decode 混批）----------
    # 默认关=原版两阶段调度（prefill 步与 decode 步互斥），零回归；
    # 打开后同一步既给 running 序列各算 1 个 decode token，又用剩余预算给 waiting 序列做分块 prefill，
    # 使 decode 不再被长 prompt 的连续 prefill 饿住。
    enable_chunked_prefill: bool = False

    # ---------- 功能二：Prefix Cache 增强 ----------
    enable_lru: bool = True               # True 使用显式 LRU 驱逐；False 回退原版 FIFO，用于对照基线。

    # ---------- 功能三：Cache-Aware Scheduling（LPM 前缀感知调度）----------
    # 默认关=等待队列严格 FIFO 出队（原版行为），零回归。
    # 打开后 prefill 优先调度“与已缓存前缀匹配最长（命中块最多）”的等待请求，把同前缀请求在时间上
    # 聚拢，减少热点前缀被无关请求挤出缓存后的重复 prefill；配合 aging 阈值防止低命中请求饿死。
    enable_cache_aware_schedule: bool = False

    # ---------- 功能四：Draft-Target Speculative Decoding（投机解码）----------
    # 默认关=完全走原 decode 路径，不构建 draft 模型/KV，零回归。
    # 打开后用小 draft 模型连续 propose K 个候选 token，大 target 模型一次并行 verification，
    # 通过 Exact Rejection Sampling 决定接受长度，使一个 target step 推进 1~K+1 个 token。
    enable_speculative_decode: bool = False
    speculative_model: "str | None" = None    # draft 模型本地目录（需与 target 共享同一 tokenizer/vocab）
    num_speculative_tokens: int = 1            # 每步 draft propose 的候选 token 数 K（默认 1：融合后满批 64 eager 下 wall/TPOT/吞吐全指标最优；higher-K 需 draft-decode CUDA Graph 摊薄 draft forward 才有收益）
    draft_hf_config: "AutoConfig | None" = None  # draft 模型 config，若开启则在 __post_init__ 懒加载
    num_draft_kvcache_blocks: int = -1         # draft KV Cache 块数，ModelRunner 预热后计算

    # 数据类完成自动 __init__ 后，会自动调用 __post_init__ 做校验和补充初始化。
    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8

        # 读取模型目录中的 config.json，得到层数、头数、隐藏维度、dtype 等。
        # 按需导入 transformers：仅真正构造 Config（跑引擎）时才需要，纯逻辑单测无需安装。
        from transformers import AutoConfig
        self.hf_config = AutoConfig.from_pretrained(self.model)

        # 最终长度取“用户限制”和“模型本身上限”中的较小值。
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)

        # 功能四：开启投机时懒加载 draft config，并做 hf_config 级别的 vocab_size 断言
        # （token 空间必须一致，否则 rejection sampling 的 p/q 无法对齐）。
        # tokenizer 层面的 token-id 映射与特殊 token 一致性由 LLMEngine 加载 tokenizer 后校验。
        if self.enable_speculative_decode:
            assert self.speculative_model is not None and os.path.isdir(self.speculative_model)
            assert self.num_speculative_tokens >= 1
            self.draft_hf_config = AutoConfig.from_pretrained(self.speculative_model)
            assert self.draft_hf_config.vocab_size == self.hf_config.vocab_size, \
                f"draft/target vocab_size 不一致: {self.draft_hf_config.vocab_size} vs {self.hf_config.vocab_size}"
