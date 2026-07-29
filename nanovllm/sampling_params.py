# 【这个文件做什么】
# SamplingParams 保存一条请求的“如何采样”和“何时停止”。
# LLM.generate 可以让所有 prompt 共用一个实例，也可以给每条 prompt 单独配置。
from dataclasses import dataclass


@dataclass(slots=True)
class SamplingParams:
    # 温度越小越确定，越大越随机；本实现不支持 0。
    temperature: float = 1.0

    # 最多生成多少新 token，不包含 prompt token。
    max_tokens: int = 64

    # False：生成 EOS 时停止；True：即使生成 EOS 也继续到 max_tokens。
    ignore_eos: bool = False

    def __post_init__(self):
        # Sampler 会用 logits / temperature，所以 temperature 不能为 0。
        # “greedy sampling”指始终选择最高分 token，本项目没有实现该分支。
        assert self.temperature > 1e-10, "greedy sampling is not permitted"
