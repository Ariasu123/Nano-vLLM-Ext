# 功能四 P4：Scheduler 投机路径单测（SimpleNamespace 假 config + 假 token，不依赖 GPU）。
# 覆盖：spec 组批与 K+1 预算、预算/资源不足降级、关闭时零回归、postprocess_speculative
# 的多 token 提交 / num_cached & num_draft_cached 语义 / 复用停止判定截断。
from types import SimpleNamespace

from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence
from nanovllm.sampling_params import SamplingParams


def make_scheduler(block_size=4, num_blocks=100, draft_blocks=100, max_num_seqs=16,
                   max_num_batched_tokens=1000, enable_speculative_decode=True,
                   num_speculative_tokens=4, eos=-1):
    Sequence.block_size = block_size
    cfg = SimpleNamespace(
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        eos=eos,
        kvcache_block_size=block_size,
        num_kvcache_blocks=num_blocks,
        enable_lru=True,
        enable_chunked_prefill=False,
        enable_cache_aware_schedule=False,
        enable_speculative_decode=enable_speculative_decode,
        num_speculative_tokens=num_speculative_tokens,
        num_draft_kvcache_blocks=draft_blocks,
    )
    return Scheduler(cfg)


def make_seq(token_ids, max_tokens=100, ignore_eos=True):
    return Sequence(token_ids, SamplingParams(temperature=0.6, ignore_eos=ignore_eos, max_tokens=max_tokens))


def prefill_one(sched, token_ids, ignore_eos=True, max_tokens=100):
    # 完整 prefill 并生成 1 个 token，进入 running（decode/spec 就绪）。
    seq = make_seq(token_ids, max_tokens=max_tokens, ignore_eos=ignore_eos)
    sched.add(seq)
    seqs, is_prefill = sched.schedule()
    assert is_prefill
    sched.postprocess(seqs, [999] * len(seqs), True)
    assert seq in sched.running
    return seq


def test_spec_step_selected_and_budget():
    sched = make_scheduler(num_speculative_tokens=4)
    a = prefill_one(sched, [1, 2, 3, 4, 5])
    b = prefill_one(sched, [6, 7, 8, 9, 10])
    seqs, is_prefill = sched.schedule()
    assert sched.pending_kind == "speculative"
    assert is_prefill is False
    assert set(seqs) == {a, b}
    for s in seqs:
        assert s.num_scheduled_tokens == 5          # K+1
        assert s.draft_block_table                  # draft tentative 块已备
    assert sched.stats.num_speculative_steps == 1
    assert sched.stats.total_proposed_tokens == 2 * 4


def test_budget_limits_spec_count():
    sched = make_scheduler(num_speculative_tokens=4, max_num_batched_tokens=5)
    prefill_one(sched, [1, 2, 3, 4, 5])
    prefill_one(sched, [6, 7, 8, 9, 10])
    seqs, _ = sched.schedule()
    assert sched.pending_kind == "speculative"
    assert len(seqs) == 1                            # 预算 5 只容得下 1 条（K+1=5）


def test_insufficient_draft_blocks_falls_back_to_decode():
    sched = make_scheduler(num_blocks=100, draft_blocks=1, num_speculative_tokens=4)
    a = prefill_one(sched, [1, 2, 3, 4, 5])
    seqs, is_prefill = sched.schedule()
    assert sched.pending_kind == "normal"            # draft 备块不足 → 降级
    assert is_prefill is False
    assert a in seqs and a.num_scheduled_tokens == 1  # 普通 decode


def test_disabled_no_spec_zero_regression():
    sched = make_scheduler(enable_speculative_decode=False)
    assert sched.draft_block_manager is None
    a = prefill_one(sched, [1, 2, 3, 4, 5])
    seqs, is_prefill = sched.schedule()
    assert sched.pending_kind == "normal"
    assert is_prefill is False
    assert a in seqs and a.num_scheduled_tokens == 1


def test_prefill_pending_not_selected_for_spec():
    # running 有 decode 就绪序列、waiting 有新请求时：两阶段本步优先 prefill，不走 spec。
    sched = make_scheduler(num_speculative_tokens=4)
    prefill_one(sched, [1, 2, 3, 4, 5])
    waiting = make_seq([6, 7, 8, 9])
    sched.add(waiting)
    seqs, is_prefill = sched.schedule()
    assert is_prefill is True and sched.pending_kind == "normal"
    assert waiting in seqs


def test_postprocess_speculative_partial_accept_counts():
    sched = make_scheduler(num_speculative_tokens=4)
    a = prefill_one(sched, [1, 2, 3, 4, 5])          # len=6
    sched.schedule()                                  # spec step
    n_before = a.num_tokens                           # 6
    sched.postprocess_speculative([a], [([101, 102, 103], 2)])  # 接受2 draft + 1 recovered
    assert a.num_tokens == n_before + 3               # 9
    assert a.num_cached_tokens == a.num_tokens - 1    # 8（末 token 不缓存）
    # draft 真实写入上界 = n_before-1+K = 5+4 = 9；min(8,9)=8（无落后）
    assert a.num_draft_cached_tokens == 8
    assert len(a.block_table) == a.num_blocks
    assert sched.stats.total_accepted_tokens == 2
    assert sched.stats.total_emitted_tokens == 3
    assert sched.stats.num_bonus == 0


def test_postprocess_speculative_all_accept_bonus_draft_lag():
    sched = make_scheduler(num_speculative_tokens=4)
    a = prefill_one(sched, [1, 2, 3, 4, 5])          # len=6
    sched.schedule()
    n_before = a.num_tokens                           # 6
    sched.postprocess_speculative([a], [([201, 202, 203, 204, 205], 4)])  # 全接受+bonus
    assert a.num_tokens == n_before + 5               # 11
    num_written = a.num_tokens - 1                    # 10
    assert a.num_cached_tokens == num_written
    # cap = 6-1+4 = 9 < 10 → draft 落后 1，下一步 propose 第 0 步融合补 dK
    assert a.num_draft_cached_tokens == 9
    assert sched.stats.num_bonus == 1
    assert sched.stats.total_accepted_tokens == 4


def test_postprocess_speculative_stops_on_eos():
    sched = make_scheduler(num_speculative_tokens=4, eos=7)
    a = prefill_one(sched, [1, 2, 3, 4, 5], ignore_eos=False)   # len=6
    sched.schedule()
    sched.postprocess_speculative([a], [([10, 7, 20], 2)])       # 提交到 EOS(7) 即停
    assert a.is_finished
    assert a.num_tokens == 6 + 2                       # 只提交 10,7；丢弃 20
    assert a not in sched.running
    assert not a.block_table                            # finish 释放 target KV
    assert not a.draft_block_table                      # 及 draft KV


def test_postprocess_speculative_stops_on_max_tokens():
    sched = make_scheduler(num_speculative_tokens=4)
    a = prefill_one(sched, [1, 2, 3], max_tokens=3)     # completion=1, len=4
    sched.schedule()
    sched.postprocess_speculative([a], [([11, 12, 13, 14, 15], 4)])
    assert a.is_finished
    assert a.num_completion_tokens == 3                 # 999,11,12 → 命中 max_tokens 停
    assert a.num_tokens == 3 + 3
