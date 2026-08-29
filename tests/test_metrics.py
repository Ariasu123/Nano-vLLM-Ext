# 指标纯函数与统计 dataclass 的单测（不依赖 torch/xxhash/transformers）。
from nanovllm.engine.metrics import (
    percentile, compute_ttft, compute_tpot, PrefixCacheStats, SchedulerStats,
)


def test_percentile_basic():
    assert percentile([], 50) == 0.0
    assert percentile([5], 99) == 5.0
    assert percentile([1, 2, 3, 4], 0) == 1
    assert percentile([1, 2, 3, 4], 100) == 4
    assert percentile([1, 2, 3, 4], 50) == 2.5   # 线性插值中位数


def test_ttft_tpot():
    assert compute_ttft(1.0, 1.5) == 0.5
    assert compute_tpot(1.0, 3.0, 5) == 0.5       # (3-1)/(5-1)
    assert compute_tpot(1.0, 3.0, 1) == 0.0       # 只有首 token，无间隔
    assert compute_tpot(1.0, 3.0, 0) == 0.0


def test_prefix_cache_stats():
    s = PrefixCacheStats(block_size=256)
    s.record_query(4, 2)
    s.record_eviction()
    assert s.num_queries == 4 and s.num_hits == 2
    assert s.hit_rate == 0.5
    assert s.saved_tokens == 2 * 256
    assert s.num_evictions == 1
    assert PrefixCacheStats().hit_rate == 0.0     # 无查询定义为 0


def test_scheduler_stats():
    s = SchedulerStats()
    s.record_prefill_step(100)
    s.record_prefill_step(80)
    s.record_decode_step(8)
    s.record_preemption()
    assert s.num_prefill_steps == 2
    assert s.total_prefill_tokens == 180
    assert s.num_decode_steps == 1
    assert s.total_decode_tokens == 8
    assert s.num_preemptions == 1
