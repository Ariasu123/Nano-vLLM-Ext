# 功能四 P3：BlockManager 投机解码块管理单测（block_size=4，纯逻辑，不依赖 GPU）。
# 覆盖：tentative 块申请、commit 后按接受长度回收多余块、全接受+bonus 需额外块、
# 资源不足降级、hash_blocks_committed 防污染、draft_block_table 对称回收。
from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.sequence import Sequence
from nanovllm.sampling_params import SamplingParams


def make_seq(token_ids, block_size=4):
    Sequence.block_size = block_size
    return Sequence(token_ids, SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=100))


def running_decode_seq(bm, token_ids):
    # 走一次 prefill 分配，再置成 running-decode 状态（末 token 不缓存：num_cached=len-1）。
    seq = make_seq(token_ids)
    n = bm.can_allocate(seq)
    assert n != -1
    bm.allocate(seq, n)
    seq.is_prefill = False
    seq.num_cached_tokens = len(seq) - 1
    seq.num_scheduled_tokens = 0
    return seq


def _ceil_blocks(n, bs=4):
    return (n + bs - 1) // bs


def test_spec_append_and_trim_releases_tentative_blocks():
    bm = BlockManager(10, 4)
    seq = running_decode_seq(bm, list(range(6)))       # len=6 → 2 块
    assert len(seq.block_table) == 2
    free_before = len(bm.free_block_ids)

    assert bm.can_append_spec(seq, 4)
    bm.may_append_spec(seq, 4)
    # 最坏最终长度 6+4+1=11 → ceil(11/4)=3 块，追加 1 个 tentative 块
    assert len(seq.block_table) == 3
    assert len(bm.free_block_ids) == free_before - 1

    # 全拒绝：只提交 1 个 recovered token → len=7，保留 ceil(7/4)=2 块
    seq.append_token(999)
    bm.trim_blocks(seq, _ceil_blocks(len(seq)))
    assert len(seq.block_table) == 2
    assert len(bm.free_block_ids) == free_before        # tentative 块已归还


def test_spec_all_accept_bonus_keeps_extra_block():
    bm = BlockManager(10, 4)
    seq = running_decode_seq(bm, list(range(8)))       # len=8 → 恰好 2 满块
    bm.may_append_spec(seq, 4)
    # 最坏 8+4+1=13 → ceil(13/4)=4 块
    assert len(seq.block_table) == 4

    # 全接受 + bonus：提交 K+1=5 个 token → len=13，bonus（index12）需要第 4 块
    for t in range(5):
        seq.append_token(1000 + t)
    bm.trim_blocks(seq, _ceil_blocks(len(seq)))
    assert len(seq.block_table) == 4                    # 不回收，bonus 块保留供下一步写 KV


def test_can_append_spec_insufficient_blocks():
    bm = BlockManager(2, 4)
    seq = running_decode_seq(bm, list(range(6)))       # 用掉 2 块，free=0
    assert bm.can_append_spec(seq, 4) is False          # 需追加 1 块但无空闲 → 降级普通 decode


def test_hash_blocks_committed_excludes_last_and_tentative():
    bm = BlockManager(20, 4)
    seq = running_decode_seq(bm, list(range(9)))       # len=9 → 3 块
    bm.may_append_spec(seq, 4)                          # ceil(14/4)=4 块
    assert len(seq.block_table) == 4

    # 全接受 + bonus 提交 5 个 → len=14，末 token index13 不缓存
    for t in range(5):
        seq.append_token(2000 + t)
    bm.hash_blocks_committed(seq, len(seq) - 1)         # num_written=13 → 满块 13//4=3

    # 只登记完全落在 [0,13) 的满块 0/1/2；含末 token 的 block3 不登记（防污染）
    assert bm.blocks[seq.block_table[0]].hash != -1
    assert bm.blocks[seq.block_table[1]].hash != -1
    assert bm.blocks[seq.block_table[2]].hash != -1
    assert bm.blocks[seq.block_table[3]].hash == -1
    assert len(bm.hash_to_block_id) == 3


def test_draft_block_table_symmetric_recycle():
    # draft 用独立 BlockManager + block_table 参数，回收逻辑与 target 对称。
    draft_bm = BlockManager(10, 4, enable_lru=False)
    seq = make_seq(list(range(6)))
    seq.is_prefill = False
    seq.draft_block_table = [draft_bm._allocate_block(), draft_bm._allocate_block()]
    free_before = len(draft_bm.free_block_ids)

    assert draft_bm.can_append_spec(seq, 4, seq.draft_block_table)
    draft_bm.may_append_spec(seq, 4, seq.draft_block_table)
    assert len(seq.draft_block_table) == 3
    assert len(draft_bm.free_block_ids) == free_before - 1

    seq.append_token(999)                               # 全拒绝 → len=7
    draft_bm.trim_blocks(seq, _ceil_blocks(len(seq)), seq.draft_block_table)
    assert len(seq.draft_block_table) == 2
    assert len(draft_bm.free_block_ids) == free_before
