# 引擎运行指标：调度组成与 Prefix Cache 命中情况。
# 这些数据本身不参与推理，只用于 benchmark 汇总，量化各优化项（公平调度、Prefix Cache）的收益。

from dataclasses import dataclass


@dataclass
class SchedulerStats:
    # 分别统计 prefill / decode 步数，可看出反饥饿策略是否让 decode 得到足够调度机会。
    num_prefill_steps: int = 0
    num_decode_steps: int = 0
    # 抢占次数：decode 缺块时挤掉其他请求的次数，越多说明显存压力越大。
    num_preemptions: int = 0
    # 累计处理的 prefill / decode token 数，供分阶段吞吐计算。
    total_prefill_tokens: int = 0
    total_decode_tokens: int = 0

    def record_prefill_step(self, num_tokens: int):
        self.num_prefill_steps += 1
        self.total_prefill_tokens += num_tokens

    def record_decode_step(self, num_seqs: int):
        self.num_decode_steps += 1
        self.total_decode_tokens += num_seqs

    def record_preemption(self):
        self.num_preemptions += 1


@dataclass
class PrefixCacheStats:
    num_queries: int = 0    # 曾向缓存查询过的“完整块”总数（分母）。
    num_hits: int = 0       # 其中命中的完整块总数（分子）。
    num_evictions: int = 0  # 真正复写掉一个仍带有效哈希的块的次数（LRU 驱逐）。
    block_size: int = 256

    def record_query(self, num_query_blocks: int, num_hit_blocks: int):
        self.num_queries += num_query_blocks
        self.num_hits += num_hit_blocks

    def record_eviction(self):
        self.num_evictions += 1

    @property
    def hit_rate(self) -> float:
        # 命中率 = 命中块 / 查询块；无查询时定义为 0。
        return self.num_hits / self.num_queries if self.num_queries else 0.0

    @property
    def saved_tokens(self) -> int:
        # 每个命中的完整块省下 block_size 个 token 的 prefill 计算。
        return self.num_hits * self.block_size


# ---------- 延迟指标纯函数（不依赖 torch，可单独单测）----------

def percentile(values: list[float], p: float) -> float:
    # 线性插值分位数：p 取 50 得中位数，取 99 得 P99。空列表返回 0。
    if not values:
        return 0.0
    s = sorted(values)
    if len(s) == 1:
        return float(s[0])
    # k 是 p 分位对应的（可能非整）下标，再在相邻两点间线性插值。
    k = (len(s) - 1) * (p / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return float(s[lo] + (s[hi] - s[lo]) * (k - lo))


def compute_ttft(arrival_time: float, first_token_time: float) -> float:
    # TTFT（Time To First Token）：请求到达到吐出第一个 token 的耗时。
    return first_token_time - arrival_time


def compute_tpot(first_token_time: float, finish_time: float, num_completion_tokens: int) -> float:
    # TPOT（Time Per Output Token）：首 token 之后，平均每个新 token 的耗时。
    # 只有 1 个（或 0 个）completion token 时没有 inter-token 间隔，定义为 0。
    if num_completion_tokens <= 1:
        return 0.0
    return (finish_time - first_token_time) / (num_completion_tokens - 1)
