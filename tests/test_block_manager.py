# BlockManager 的 Prefix Cache 增强单测：LRU 驱逐、优先复写无哈希块、命中重激活、
# 统计计数、以及 enable_lru=False 回退 FIFO。需要 xxhash（block_manager 运行依赖）。
from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.sequence import Sequence
from nanovllm.sampling_params import SamplingParams


def make_seq(token_ids, block_size=4):
    Sequence.block_size = block_size
    return Sequence(token_ids, SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=100))


def prefill(bm: BlockManager, seq: Sequence):
    # 模拟一次完整 prefill 的块管理流程（不含 GPU 计算）。
    n = bm.can_allocate(seq)
    assert n != -1
    bm.allocate(seq, n)
    seq.num_scheduled_tokens = seq.num_tokens - seq.num_cached_tokens
    bm.hash_blocks(seq)
    seq.num_cached_tokens += seq.num_scheduled_tokens
    seq.num_scheduled_tokens = 0
    return n


def _setup_two_cached_one_nohash(enable_lru):
    # 构造 free 池：block0/1 带哈希（缓存块），block2 无哈希。
    bm = BlockManager(3, 4, enable_lru=enable_lru)
    for bid in (0, 1, 2):
        assert bm._allocate_block() == bid          # 初始 free 顺序 0,1,2
    for bid, h in [(0, 100), (1, 101)]:
        bm.blocks[bid].hash = h
        bm.hash_to_block_id[h] = bid
    for bid in (0, 1, 2):
        bm.blocks[bid].ref_count = 0                # _deallocate_block 要求 ref_count==0
        bm._deallocate_block(bid)
    return bm


def test_lru_prefers_nohash_then_evicts_oldest():
    bm = _setup_two_cached_one_nohash(enable_lru=True)
    # LRU 下无哈希块被移到队首：先复写 block2，不算驱逐。
    assert bm._allocate_block() == 2
    assert bm.prefix_stats.num_evictions == 0
    # 再分配复写最久未用的缓存块 block0，算一次驱逐，其哈希索引被删除。
    assert bm._allocate_block() == 0
    assert bm.prefix_stats.num_evictions == 1
    assert 100 not in bm.hash_to_block_id
    assert 101 in bm.hash_to_block_id


def test_fifo_fallback_ignores_nohash_priority():
    bm = _setup_two_cached_one_nohash(enable_lru=False)
    # FIFO 下按回插顺序 0,1,2 取出，不因无哈希而优先。
    assert bm._allocate_block() == 0
    assert bm._allocate_block() == 1
    assert bm._allocate_block() == 2


def test_prefix_hit_reactivation_and_stats():
    bm = BlockManager(100, 4, enable_lru=True)
    a = make_seq([1, 2, 3, 4, 5, 6, 7, 8, 9])       # 9 token → 2 个可复用完整块
    prefill(bm, a)
    assert (bm.prefix_stats.num_queries, bm.prefix_stats.num_hits) == (2, 0)
    bm.deallocate(a)

    b = make_seq([1, 2, 3, 4, 5, 6, 7, 8, 42])       # 前 8 token 与 a 相同
    n = bm.can_allocate(b)
    assert n == 2                                    # 命中 2 个完整前缀块
    bm.allocate(b, n)
    assert bm.prefix_stats.num_hits == 2
    assert bm.prefix_stats.hit_rate == 0.5           # 2 命中 / 4 查询
    assert bm.prefix_stats.saved_tokens == 2 * 4
    assert bm.prefix_stats.num_evictions == 0        # 命中重激活不驱逐
