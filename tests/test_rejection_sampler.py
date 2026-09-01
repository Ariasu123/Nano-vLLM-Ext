# rejection_sampler 的纯 torch CPU 单测：lossless 正确性 + 边界行为。
import torch

from nanovllm.layers.rejection_sampler import rejection_sample


def _tvd(a, b):
    return 0.5 * (a - b).abs().sum().item()


def test_lossless_first_token_matches_target():
    # 黄金判据：draft_tokens 真从 q 采样、uniform 独立 U[0,1) 时，
    # 首个 emitted token 的经验分布应 ≈ target 分布 p。
    torch.manual_seed(0)
    V = 6
    p0 = torch.tensor([0.35, 0.05, 0.20, 0.10, 0.25, 0.05])
    q0 = torch.tensor([0.10, 0.30, 0.20, 0.15, 0.10, 0.15])
    S = 400_000
    gen = torch.Generator().manual_seed(123)

    # draft_tokens ~ q0
    d = torch.multinomial(q0, S, replacement=True, generator=gen).unsqueeze(1)  # [S,1]
    target_probs = p0.repeat(S, 2, 1).clone()   # [S, K+1=2, V]
    draft_probs = q0.repeat(S, 1, 1).clone()     # [S, K=1, V]
    uniform = torch.rand(S, 1, generator=gen)
    rec_u = torch.rand(S, generator=gen)

    accept_len, out = rejection_sample(target_probs, draft_probs, d,
                                       uniform=uniform, recovered_uniform=rec_u)
    emitted0 = out[:, 0]
    emp = torch.bincount(emitted0, minlength=V).float() / S
    assert _tvd(emp, p0) < 0.01, (emp, p0)


def test_all_accept_when_q_equals_p():
    V, S, K = 5, 8, 3
    p = torch.rand(V); p /= p.sum()
    target_probs = p.repeat(S, K + 1, 1).clone()
    draft_probs = p.repeat(S, K, 1).clone()
    d = torch.randint(0, V, (S, K))
    uniform = torch.rand(S, K)  # 任意 uniform 都应全接受（p/q==1）
    accept_len, out = rejection_sample(target_probs, draft_probs, d, uniform=uniform)
    assert torch.all(accept_len == K)
    # 前 K 个是被接受的 draft token，第 K 个是 bonus
    assert torch.equal(out[:, :K], d)


def test_all_reject_disjoint_onehot():
    V, S, K = 4, 16, 2
    a, b = 1, 3
    q = torch.zeros(V); q[a] = 1.0
    p = torch.zeros(V); p[b] = 1.0
    target_probs = p.repeat(S, K + 1, 1).clone()
    draft_probs = q.repeat(S, K, 1).clone()
    d = torch.full((S, K), a)          # draft 只会采到 a
    uniform = torch.rand(S, K)
    accept_len, out = rejection_sample(target_probs, draft_probs, d, uniform=uniform)
    assert torch.all(accept_len == 0)
    assert torch.all(out[:, 0] == b)   # recovered = argmax(relu(p-q)) = b


def test_greedy_branch():
    # temperature==0：接受当且仅当 draft token == target argmax；recovered/bonus 取 argmax。
    V, K = 5, 3
    p = torch.zeros(1, K + 1, V)
    # 各位置 target argmax 分别为 2, 4, 1, 0
    for j, am in enumerate([2, 4, 1, 0]):
        p[0, j, am] = 1.0
    q = torch.full((1, K, V), 1.0 / V)
    greedy = torch.ones(1, dtype=torch.bool)

    # draft = [2, 4, 9?] -> 用合法 token：匹配前两位，第三位不匹配
    d = torch.tensor([[2, 4, 3]])
    accept_len, out = rejection_sample(p, q, d, greedy=greedy)
    assert accept_len.item() == 2            # 前两位匹配 argmax，第三位 3 != 1 拒绝
    assert out[0, 0].item() == 2 and out[0, 1].item() == 4
    assert out[0, 2].item() == 1             # recovered = 位置2 的 argmax = 1

    # 全匹配 -> 全接受 + bonus(argmax=0)
    d2 = torch.tensor([[2, 4, 1]])
    accept_len2, out2 = rejection_sample(p, q, d2, greedy=greedy)
    assert accept_len2.item() == 3
    assert out2[0, 3].item() == 0


def test_shapes_and_partial_accept():
    # 形状与部分接受：K=1、S=1 等边界不报错，out_tokens 形状为 [S,K+1]。
    for S, K, V in [(1, 1, 3), (1, 4, 7), (5, 2, 4)]:
        p = torch.rand(S, K + 1, V); p /= p.sum(-1, keepdim=True)
        q = torch.rand(S, K, V); q /= q.sum(-1, keepdim=True)
        d = torch.randint(0, V, (S, K))
        accept_len, out = rejection_sample(p, q, d)
        assert out.shape == (S, K + 1)
        assert torch.all((0 <= accept_len) & (accept_len <= K))
        assert out.dtype == torch.int64
