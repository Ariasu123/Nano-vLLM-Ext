# 【这个文件做什么】
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

        # 下面是“指数竞赛”采样技巧，与按照 probs 做分类采样等价：
        # 为每个候选 token 生成指数随机数，用 概率/随机数 得到带噪分数，再取最大值。
        # clamp_min 防止随机数过小造成除零或数值溢出。
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)

        # argmax 返回每行最大值的下标，也就是采样出的 token id。
        return sample_tokens
