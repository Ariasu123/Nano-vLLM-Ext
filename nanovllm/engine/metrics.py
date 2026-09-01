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

    # ---------- 功能四：Speculative Decoding 统计 ----------
    # num_speculative_steps：走 spec 路径的 step 数（一步可提交 1~K+1 个 token）。
    # total_proposed_tokens：draft 累计 propose 的候选 token 数（= K * spec 序列数累加）。
    # total_accepted_tokens：其中被 rejection sampling 接受的 draft token 数（不含 recovered/bonus）。
    # total_emitted_tokens：spec 路径实际提交的 token 数（accepted + recovered/bonus），用于吞吐口径。
    # num_bonus：全接受触发 bonus token 的次数。
    num_speculative_steps: int = 0
    total_proposed_tokens: int = 0
    total_accepted_tokens: int = 0
    total_emitted_tokens: int = 0
    num_bonus: int = 0
    # record_acceptance 调用次数 = 累计「序列 × spec step」的 verification 次数（一次 target forward）。
    # avg_accept_len 的正确分母：一个 spec step 会并行验证多条序列，不能用 num_speculative_steps。
    num_acceptance_records: int = 0

    # ---------- 功能四诊断：为什么 spec batch 装不满 / draft KV lag 从哪来 ----------
    # 每次 _select_speculative 里被拒序列按「首个失败闸门」归因，回答「首个 spec batch 为何只有 51/64」：
    #   spec_reject_draft  —— draft KV 池装不下（几乎必然的限流点）
    #   spec_reject_target —— target KV 池装不下
    #   spec_reject_budget —— token 预算不足（默认 16384 下几乎不触发）
    # spec_fallback_decode_steps：spec 开启但本步无序列能组 spec、退化为普通 decode 的步数
    #   （>0 = 存在「target 前进、draft 冻结」的隐藏 fallback，会累积 draft KV lag —— 用来确认有没有它）。
    spec_reject_draft: int = 0
    spec_reject_target: int = 0
    spec_reject_budget: int = 0
    spec_fallback_decode_steps: int = 0

    def record_prefill_step(self, num_tokens: int):
        self.num_prefill_steps += 1
        self.total_prefill_tokens += num_tokens

    def record_decode_step(self, num_seqs: int):
        self.num_decode_steps += 1
        self.total_decode_tokens += num_seqs

    def record_preemption(self):
        self.num_preemptions += 1

    # 记录一个 spec step 的规模：num_seqs 条序列、每条 propose 了 num_spec 个候选。
    def record_speculative_step(self, num_seqs: int, num_spec: int):
        self.num_speculative_steps += 1
        self.total_proposed_tokens += num_seqs * num_spec

    # 记录单条序列一次 verification 的结果：接受 accepted 个 draft token、实际提交 emitted 个 token、
    # 是否触发 bonus。spec 提交的 token 也计入 decode 吞吐（total_decode_tokens），保证开/关口径一致。
    def record_acceptance(self, accepted: int, emitted: int, bonus: bool):
        self.num_acceptance_records += 1
        self.total_accepted_tokens += accepted
        self.total_emitted_tokens += emitted
        self.total_decode_tokens += emitted
        if bonus:
            self.num_bonus += 1

    @property
    def acceptance_rate(self) -> float:
        # 接受率 = 被接受的 draft token / 全部 propose 的 draft token；无 propose 时定义为 0。
        return self.total_accepted_tokens / self.total_proposed_tokens if self.total_proposed_tokens else 0.0

    @property
    def avg_accept_len(self) -> float:
        # 平均每次 verification 真正被接受的 draft token 数（不含每轮必然的 recovered/bonus），范围 0..K。
        # 分母是 verification 次数（record_acceptance 调用数），不能用 num_speculative_steps
        # （一个 spec step 并行验证多条序列）。这才是字面意义的 "accept length"。
        return self.total_accepted_tokens / self.num_acceptance_records if self.num_acceptance_records else 0.0

    @property
    def tokens_per_target_step(self) -> float:
        # 平均每次 target verification 最终推进的 token 数（= accepted + 1 个 recovered/bonus，触及
        # max_tokens/EOS 截断时略少）——投机解码的加速上限，范围 1..K+1。与 avg_accept_len 区分：
        # 后者只数被接受的 draft token，这里含每轮那个必然产生的 recovered/bonus token。
        return self.total_emitted_tokens / self.num_acceptance_records if self.num_acceptance_records else 0.0


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
