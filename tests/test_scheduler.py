# 功能一 Chunked Prefill（vLLM 式 prefill/decode 混批）单测：
# enable_chunked_prefill=False 时行为=原版两阶段（回归保护）；True 时同一步混排 decode + 分块 prefill。
# 用 SimpleNamespace 假 config 绕开 Config.__post_init__（AutoConfig/isdir），不 import torch。
from types import SimpleNamespace

from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.sampling_params import SamplingParams


def make_scheduler(block_size=4, num_blocks=100, max_num_seqs=16,
                   max_num_batched_tokens=1000, enable_chunked_prefill=False,
                   enable_lru=True, enable_cache_aware_schedule=False, eos=-1):
    Sequence.block_size = block_size
    cfg = SimpleNamespace(
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        eos=eos,
        kvcache_block_size=block_size,
        num_kvcache_blocks=num_blocks,
        enable_lru=enable_lru,
        enable_chunked_prefill=enable_chunked_prefill,
        enable_cache_aware_schedule=enable_cache_aware_schedule,
    )
    return Scheduler(cfg)


def make_seq(token_ids, max_tokens=100):
    return Sequence(token_ids, SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=max_tokens))


def _prefill_one(sched, token_ids):
    # 让一条请求完整 prefill 并生成 1 个 token，进入 running（decode 就绪）。
    seq = make_seq(token_ids)
    sched.add(seq)
    seqs, is_prefill = sched.schedule()
    assert is_prefill and seq in seqs
    sched.postprocess(seqs, [999] * len(seqs), True)   # 999 非 eos，请求不结束
    assert seq in sched.running
    return seq


def test_default_behavior_unchanged():
    # enable_chunked_prefill=False（默认）：长 prompt 一步 prefill 到底，等价原版两阶段。
    sched = make_scheduler()
    seq = make_seq(list(range(9)))
    sched.add(seq)
    seqs, is_prefill = sched.schedule()
    assert is_prefill
    assert seq.num_scheduled_tokens == 9               # 无 chunk 上限，一次到位
    assert seq in sched.running


def test_two_phase_never_mixes():
    # 两阶段：running 有 decode、waiting 有新请求时，本步只做 prefill（原版永不混批）。
    sched = make_scheduler()
    running = _prefill_one(sched, [1, 2, 3, 4, 5])
    waiting = make_seq([6, 7, 8, 9])
    sched.add(waiting)
    seqs, is_prefill = sched.schedule()
    assert is_prefill is True                          # 优先 prefill
    assert waiting in seqs and running not in seqs      # decode 序列本步未被调度


def test_chunked_mixed_step():
    # 混批：running 有 decode + waiting 有长 prompt → 一步同时返回二者，is_prefill=True。
    sched = make_scheduler(max_num_batched_tokens=8, enable_chunked_prefill=True)
    running = _prefill_one(sched, [1, 2, 3, 4, 5])
    waiting = make_seq(list(range(20)))                # 长 prompt，放不下整段
    sched.add(waiting)
    seqs, is_prefill = sched.schedule()
    assert is_prefill is True                          # 混批步走 varlen 路径
    assert running in seqs and running.is_prefill is False and running.num_scheduled_tokens == 1
    assert waiting in seqs and waiting.is_prefill is True
    assert waiting.num_scheduled_tokens == 7           # 预算 8 - 1 个 decode = 7 给 prefill
    assert waiting in sched.waiting                     # 未完成，留 waiting 队首续跑


def test_chunked_postprocess_per_seq():
    # 混批 postprocess 按序列判定：decode 序列 append token，仍在 prefill 的长 prompt 不 append。
    sched = make_scheduler(max_num_batched_tokens=8, enable_chunked_prefill=True)
    running = _prefill_one(sched, [1, 2, 3, 4, 5])
    n_before = running.num_completion_tokens
    waiting = make_seq(list(range(20)))
    sched.add(waiting)
    seqs, is_prefill = sched.schedule()
    sched.postprocess(seqs, [999] * len(seqs), is_prefill)
    assert running.num_completion_tokens == n_before + 1   # decode 序列吐了一个 token
    assert waiting.num_completion_tokens == 0              # 仍在 prefill，不 append
    assert waiting in sched.waiting


def test_chunked_pure_decode_when_waiting_empty():
    # 混批模式下 waiting 为空 → 本步退化为纯 decode（is_prefill=False，保留 CUDA Graph 路径）。
    sched = make_scheduler(enable_chunked_prefill=True)
    running = _prefill_one(sched, [1, 2, 3, 4, 5])
    seqs, is_prefill = sched.schedule()
    assert is_prefill is False
    assert running in seqs and running.num_scheduled_tokens == 1


# ---------- 功能三：Cache-Aware Scheduling（LPM 前缀感知调度）----------
# block_size=4：一条 [1..8] 的请求完整 prefill 后会登记 block0=[1,2,3,4]、block1=[5,6,7,8] 的前缀哈希，
# 后续请求带相同开头即命中。can_allocate 探针只数“连续命中的完整前缀块”，末块不参与。

def test_cache_aware_selects_high_hit_over_fifo():
    # 开启 LPM 后：队首是 0 命中请求、队尾是 2 命中请求时，本步应优先调度高命中者而非 FIFO 队首。
    sched = make_scheduler(enable_cache_aware_schedule=True, max_num_seqs=1)
    _prefill_one(sched, list(range(1, 13)))                # 登记前缀 [1,2,3,4]/[5,6,7,8]
    low_hit = make_seq(list(range(50, 63)))                # block0=[50,51,52,53] 不命中 → 0
    high_hit = make_seq(list(range(1, 9)) + [100, 101, 102, 103, 104])  # 命中 [1..4]/[5..8] → 2
    sched.add(low_hit)                                     # 先加 → FIFO 队首
    sched.add(high_hit)
    seqs, is_prefill = sched.schedule()
    assert is_prefill is True
    assert high_hit in seqs and low_hit not in seqs        # LPM 选高命中，跳过 FIFO 队首
    assert low_hit in sched.waiting                        # 低命中仍留 waiting


def test_cache_aware_continues_midprefill_first():
    # 续跑红线：一个处于分块 prefill 中途的请求（block_table 非空）即使排在 FIFO 队首之后，
    # 也必须被优先续上，而不是被 LPM 换成命中更高的 fresh 请求。
    sched = make_scheduler(max_num_batched_tokens=8, enable_cache_aware_schedule=True)
    _prefill_one(sched, list(range(1, 9)))                 # 登记前缀 [1,2,3,4]/[5,6,7,8]
    a_low = make_seq(list(range(50, 70)))                  # 20 token，0 命中；先加 → 永在队首
    b_high = make_seq(list(range(1, 9)) + list(range(300, 312)))  # 20 token，命中 2
    sched.add(a_low)
    sched.add(b_high)
    # 第一步：LPM 选中命中更高的 b_high（此时不在队首）做分块 prefill，预算 8 只切一段 → b_high 变续跑请求，
    # 仍留在 waiting 队列中部（未被移除），队首仍是从未被调度的 a_low。
    seqs, _ = sched.schedule()
    assert b_high in seqs and a_low not in seqs
    sched.postprocess(seqs, [999] * len(seqs), True)
    assert b_high.block_table and b_high in sched.waiting  # b_high 成为中途续跑请求
    assert not a_low.block_table and sched.waiting[0] is a_low   # a_low 仍是 FIFO 队首、未分配
    # 第二步：续跑红线应返回 b_high（队列中部的续跑请求），而非 FIFO 队首 a_low。
    seqs2, _ = sched.schedule()
    assert b_high in seqs2 and b_high.num_scheduled_tokens > 0
    assert a_low not in seqs2 and a_low in sched.waiting


def test_cache_aware_aging_falls_back_to_fifo():
    # aging 阀门：每 K 步定序强制走 FIFO 到达顺序，防低命中请求饿死。K=2 时第 2 步定序的队首应是低命中请求。
    sched = make_scheduler(enable_cache_aware_schedule=True)
    _prefill_one(sched, list(range(1, 9)))                 # 登记前缀 [1,2,3,4]
    sched._aging_interval = 2
    sched._prefill_select_calls = 0                        # 重置计数，隔离本用例
    low_hit = make_seq(list(range(50, 58)))                # 0 命中；队首
    high_hit = make_seq(list(range(1, 9)))                 # 命中 [1,2,3,4] → 1
    sched.add(low_hit)
    sched.add(high_hit)
    assert sched._build_prefill_order()[0] is high_hit     # 第1次：非触发轮 → LPM 高命中排最前
    assert sched._build_prefill_order()[0] is low_hit      # 第2次：触发 aging → 回退 FIFO 到达顺序（队首低命中）


def test_cache_aware_disabled_equals_fifo():
    # 回归锁定：开关关闭时，即使队尾有更高命中请求，也严格服务 FIFO 队首（与原版逐字一致）。
    sched = make_scheduler(enable_cache_aware_schedule=False, max_num_seqs=1)
    _prefill_one(sched, list(range(1, 9)))
    low_hit = make_seq(list(range(50, 58)))                # 0 命中；队首
    high_hit = make_seq(list(range(1, 9)))                 # 1 命中；队尾
    sched.add(low_hit)
    sched.add(high_hit)
    seqs, _ = sched.schedule()
    assert low_hit in seqs and high_hit not in seqs        # 严格 FIFO：先服务队首
    assert high_hit in sched.waiting


def test_preempt_counts():
    sched = make_scheduler()
    seq = _prefill_one(sched, [1, 2, 3, 4, 5])
    sched.running.remove(seq)
    sched.preempt(seq)
    assert sched.stats.num_preemptions == 1
    assert seq.status == SequenceStatus.WAITING
    assert seq in sched.waiting
    assert not seq.block_table                         # 抢占已释放 KV Cache
