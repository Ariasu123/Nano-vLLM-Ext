# LLM.generate 可以让所有 prompt 共用一个实例，也可以给每条 prompt 单独配置。
from dataclasses import dataclass


@dataclass(slots=True)
class SamplingParams:
    temperature: float = 1.0
    max_tokens: int = 64

    # False：生成 EOS 时停止；True：即使生成 EOS 也继续到 max_tokens。
    ignore_eos: bool = False

    def __post_init__(self):
        # Sampler 会用 logits / temperature，所以 temperature 不能为 0。
        assert self.temperature > 1e-10, "greedy sampling is not permitted"
