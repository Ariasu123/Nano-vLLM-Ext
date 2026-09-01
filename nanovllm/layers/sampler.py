# 模型输出 logits（词表中每个 token 的原始分数），Sampler 将它们变成概率，
# 再根据概率随机选择每条请求的下一个 token。
#
# 本实现只支持 temperature，没有实现 top-k、top-p 或重复惩罚。
import torch
from torch import nn


class Sampler(nn.Module):

    @torch.compile
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        # logits 形状为 [请求数, 词表大小]；temperature 原本是 [请求数]。
        # unsqueeze(1) 变成 [请求数,1]，借助广播让每行除以自己的温度。
        logits = logits.float().div_(temperatures.unsqueeze(dim=1))

        # 温度越小，最高分 token 的优势越明显；温度越大，分布越平坦。
        # softmax 将每行分数转换为总和为 1 的概率。
        probs = torch.softmax(logits, dim=-1)

        # 为每个候选 token 生成指数随机数，用 概率/随机数 得到带噪分数，再取最大值。
        # clamp_min 防止随机数过小造成除零或数值溢出。
        # argmax 返回每行最大值的下标，也就是采样出的 token id。
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
        return sample_tokens

    # ---------- 功能四：Speculative Decoding 复用的概率/采样工具 ----------
    # 故意不加 @torch.compile：投机路径走 eager，且这两个方法要在 rejection_sample 前后被普通
    # Python 逻辑调用（compute_probs 得到 target p / draft q，sample_from_probs 采 draft 候选 d）。
    # lossless 关键：draft 的 q 与其采样、target 的 p 必须来自同一 compute_probs(同一 temperature)。

    # 把 logits 变成概率分布（非原地，保留入参）。temperatures: [S] → 广播到 [S,1]。
    def compute_probs(self, logits: torch.Tensor, temperatures: torch.Tensor) -> torch.Tensor:
        logits = logits.float() / temperatures.unsqueeze(dim=1)
        return torch.softmax(logits, dim=-1)

    # 从给定概率分布采样一个 token（Gumbel/exponential-argmax，与 forward 同源）。
    # 非原地除法，绝不修改传入的 probs（draft q 之后还要喂给 rejection_sample）。
    def sample_from_probs(self, probs: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
        noise = torch.empty_like(probs).exponential_(1, generator=generator).clamp_min_(1e-10)
        return probs.div(noise).argmax(dim=-1)
