# Exact Speculative Sampling / Rejection Sampling。
#
# 投机解码的正确性核心：draft 模型从分布 q 采样出候选 token d，target 模型给出分布 p，
# 用 rejection sampling 决定是否接受，使最终输出分布严格等于 target 分布 p（lossless）。
#
# 单个 token 的判据（Leviathan et al. 2023 / Chen et al. 2023）：
#   - 以概率 alpha = min(1, p(d)/q(d)) 接受 d；
#   - 拒绝时从 residual 分布 normalize(relu(p - q)) 采样 recovered token，并停止；
#   - 若 K 个候选全部接受，则额外从 target 分布再采一个 bonus token。
# 逐位置从左到右串行判定：一旦某位置拒绝，后面的候选立即作废。
#
# 纯 torch 实现（CPU 可测）：不依赖 flash-attn / triton / CUDA。
import torch


def _sample_from_probs(probs: torch.Tensor, u: torch.Tensor | None, generator) -> torch.Tensor:
    # 逆变换采样：按行做 CDF，再用均匀数定位。probs [..., V]，返回 [...]（int64）。
    # 用 uniform 注入（而非 Gumbel）便于测试确定性，也便于与接受判据共享随机源语义。
    if u is None:
        u = torch.rand(probs.shape[:-1], generator=generator, device=probs.device, dtype=probs.dtype)
    cdf = probs.cumsum(dim=-1)
    # searchsorted 找到第一个使 cdf >= u 的下标；clamp 防止浮点误差越界。
    idx = torch.searchsorted(cdf, u.unsqueeze(-1).clamp(max=1.0), right=False).squeeze(-1)
    return idx.clamp_(max=probs.size(-1) - 1)


def rejection_sample(
    target_probs: torch.Tensor,   # [S, K+1, V] 已按各序列 temperature 归一化的 target 概率
    draft_probs: torch.Tensor,    # [S, K, V]   draft 实际 proposal 分布 q
    draft_tokens: torch.Tensor,   # [S, K]      draft 采出的候选 token d（从 q 采样）
    *,
    greedy: torch.Tensor | None = None,   # [S] bool，True 表示该序列 temperature==0 走贪心特例
    uniform: torch.Tensor | None = None,          # [S, K] in [0,1)，接受判据用；None 时内部 rand
    recovered_uniform: torch.Tensor | None = None,  # [S] in [0,1)，采 recovered/bonus 用
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact speculative sampling。

    对齐关系：设序列已提交长度 N，target_probs[:, j] 是预测索引 N+j 的目标分布 p，
    draft_probs[:, j] = q_{j+1}，draft_tokens[:, j] = d_{j+1}（j = 0..K-1）；
    target_probs[:, K] 是全部接受时的 bonus 分布。

    返回:
      accept_len: [S] int64，被接受的 draft token 数 L（0..K）。
      out_tokens: [S, K+1] int64，每序列提交 token；位置 0..accept_len 有效
                  （前 accept_len 个是被接受的 draft token，第 accept_len 个是 recovered 或 bonus）。
      调用方按 out_tokens[i, :accept_len[i]+1] 截取。
    """
    S, K, V = draft_probs.shape
    device = draft_probs.device
    assert target_probs.shape == (S, K + 1, V)
    assert draft_tokens.shape == (S, K)

    if uniform is None:
        uniform = torch.rand(S, K, generator=generator, device=device, dtype=target_probs.dtype)
    if greedy is None:
        greedy = torch.zeros(S, dtype=torch.bool, device=device)

    # 取出候选 token 处的 p(d) 与 q(d)：gather 到 [S, K]
    tok = draft_tokens.unsqueeze(-1)                                  # [S,K,1]
    p_at = target_probs[:, :K, :].gather(-1, tok).squeeze(-1)        # [S,K]
    q_at = draft_probs.gather(-1, tok).squeeze(-1)                    # [S,K]

    # log 空间判据，数值更稳：accept iff log(u) <= log p(d) - log q(d)
    # （等价 u <= min(1, p/q)；u>0，q>0 因为 d 是从 q 采样出来的）。
    log_ratio = p_at.clamp_min(1e-30).log() - q_at.clamp_min(1e-30).log()
    accept_sample = uniform.clamp_min(1e-30).log() <= log_ratio       # [S,K] bool

    # 贪心特例（temperature==0）：接受当且仅当 d 恰为 target 的 argmax。
    target_argmax = target_probs[:, :K, :].argmax(dim=-1)             # [S,K]
    accept_greedy = draft_tokens == target_argmax                    # [S,K]
    accept = torch.where(greedy.unsqueeze(-1), accept_greedy, accept_sample)  # [S,K]

    # 累积接受：只有前缀全接受才算接受。cumprod 沿 K 维。
    accept_prefix = torch.cumprod(accept.to(torch.int64), dim=-1)    # [S,K]，首个拒绝及之后为 0
    accept_len = accept_prefix.sum(dim=-1)                           # [S] in 0..K

    # recovered / bonus 分布：
    #   - 若 accept_len < K（存在拒绝）：在位置 L 从 residual = normalize(relu(p_L - q_L)) 采样；
    #   - 若 accept_len == K（全接受）：从 bonus = target_probs[:, K] 采样。
    # 贪心序列：recovered/bonus 都取 argmax。
    idxL = accept_len.clamp(max=K - 1)                               # [S]，用于取第 L 个 p/q（L<K 时）
    ar = torch.arange(S, device=device)
    p_L = target_probs[ar, idxL]                                     # [S,V] 第 L 位置 target 分布
    q_L = draft_probs[ar, idxL]                                      # [S,V] 第 L 位置 draft 分布
    residual = torch.relu(p_L - q_L)
    residual_sum = residual.sum(dim=-1, keepdim=True)
    # 数值兜底：residual 全 0（浮点误差）时退回直接从 p_L 采样。
    residual = torch.where(residual_sum > 0, residual / residual_sum.clamp_min(1e-30), p_L)

    bonus_dist = target_probs[:, K, :]                               # [S,V]
    full_accept = accept_len == K                                    # [S]
    recover_dist = torch.where(full_accept.unsqueeze(-1), bonus_dist, residual)  # [S,V]

    extra_sampled = _sample_from_probs(recover_dist, recovered_uniform, generator)  # [S]
    extra_greedy = recover_dist.argmax(dim=-1)                       # [S]
    extra = torch.where(greedy, extra_greedy, extra_sampled)         # [S]

    # 组装 out_tokens：前 accept_len 个位置放被接受的 draft token，第 accept_len 个位置放 extra。
    out_tokens = torch.zeros(S, K + 1, dtype=torch.int64, device=device)
    out_tokens[:, :K] = draft_tokens
    out_tokens[ar, accept_len] = extra
    return accept_len, out_tokens
