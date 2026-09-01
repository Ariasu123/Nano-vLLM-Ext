# 功能四 P2：Sequence pickle round-trip。
# draft_block_table / num_draft_cached_tokens 成对追加到 __getstate__/__setstate__ 末尾，
# 必须能跨 pickle 往返；瞬态 draft_tokens/draft_probs 不进 pickle。
import pickle

from nanovllm.engine.sequence import Sequence
from nanovllm.sampling_params import SamplingParams


def _roundtrip(seq):
    return pickle.loads(pickle.dumps(seq))


def test_decode_roundtrip_carries_draft_fields():
    seq = Sequence([1, 2, 3, 4], SamplingParams(ignore_eos=True, max_tokens=100))
    seq.is_prefill = False
    seq.num_cached_tokens = 3
    seq.block_table = [7, 3]
    seq.draft_block_table = [11, 5]
    seq.num_draft_cached_tokens = 3
    seq.draft_tokens = object()   # 瞬态，不应影响 pickle
    seq.draft_probs = object()

    out = _roundtrip(seq)
    assert out.block_table == [7, 3]
    assert out.draft_block_table == [11, 5]
    assert out.num_draft_cached_tokens == 3
    assert out.last_token == 4
    assert out.token_ids == []               # decode 子进程不需要历史 token


def test_prefill_roundtrip_defaults():
    # 关闭投机（默认）时字段为 []/0，pickle 往返仍正确。
    seq = Sequence([9, 8, 7], SamplingParams(ignore_eos=True, max_tokens=50))
    out = _roundtrip(seq)
    assert out.draft_block_table == []
    assert out.num_draft_cached_tokens == 0
    assert out.token_ids == [9, 8, 7]        # prefill 发送完整 token_ids
    assert out.last_token == 7
