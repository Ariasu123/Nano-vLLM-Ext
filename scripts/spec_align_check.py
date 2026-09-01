# 功能四 P6：verification 张量/KV 对齐 debug guard（需 GPU，TP=1）。
#
# 为什么单独存在：端到端 lossless 判据（run_gpu.sh 的黄金测试）能发现"结果不对"，但无法定位
# 是 draft 采样、rejection sampling、还是 verification 的张量构造出错。本脚本把中间层单独拎出来——
# 只比 logits、不比采样结果（排除随机数/temperature/rejection sampler 干扰），验证 target 一次并行
# verification 在每个位置 (e,d1..dK) 产出的 logits，与对同一 token 串逐 token 单步 decode 的 logits
# 数值一致。定位规则：worst_position==0 通常是 context/position/slot 错；第 0 位对、后面开始错，
# 往往是 causal 对齐 / cu_seqlens / speculative KV 写入位置有偏移。
#
# 用法（有卡模式）：
#   python scripts/spec_align_check.py    # 默认读 env.sh 的数据盘路径；自定义时传 SPEC_TARGET=/path SPEC_DRAFT=/path 覆盖
import os
from random import seed, randint

from nanovllm import LLM, SamplingParams


def _model_root():
    # 与 scripts/env.sh 一致：AutoDL 优先数据盘 /root/autodl-tmp，否则 ~/huggingface；MODEL_ROOT 可覆盖。
    root = os.environ.get("MODEL_ROOT")
    if not root:
        root = "/root/autodl-tmp/models" if os.path.isdir("/root/autodl-tmp") else os.path.expanduser("~/huggingface")
    return root


TARGET = os.path.expanduser(os.environ.get("SPEC_TARGET", os.path.join(_model_root(), "Qwen3-8B")))
DRAFT = os.path.expanduser(os.environ.get("SPEC_DRAFT", os.path.join(_model_root(), "Qwen3-0.6B")))
K = int(os.environ.get("SPEC_K", "4"))
# 对齐判据 = 逐位置最大概率差 <= PROB_ATOL：对 BF16 内核噪声鲁棒（softmax 后 ~1e-2），对结构
# 错位敏感（错位位置某 token 概率差≈1）。取 0.3 可干净区分二者，可用环境变量收紧/放宽。
PROB_ATOL = float(os.environ.get("SPEC_PROB_ATOL", "0.3"))
# 以下仅作诊断打印：BF16 下 varlen vs decode 两内核有正常 ULP 级 logit 差异，raw-logit allclose 不作判据。
ATOL = float(os.environ.get("SPEC_ATOL", "1e-2"))
RTOL = float(os.environ.get("SPEC_RTOL", "1e-2"))


def main():
    seed(0)
    # enforce_eager=True：投机路径本就走 eager，且对齐诊断不需要 CUDA Graph。
    llm = LLM(TARGET, enforce_eager=True, max_model_len=4096,
              enable_speculative_decode=True, speculative_model=DRAFT,
              num_speculative_tokens=K)
    sp = SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=64)
    for _ in range(2):
        llm.add_request([randint(0, 10000) for _ in range(randint(40, 60))], sp)

    # 推进到「下一步是投机 step」：prefill/普通 decode 步正常执行，直到 scheduler 选出 spec 组批。
    while True:
        seqs, is_prefill = llm.scheduler.schedule()
        if llm.scheduler.pending_kind == "speculative":
            break
        token_ids = llm.model_runner.call("run", seqs, is_prefill)
        llm.scheduler.postprocess(seqs, token_ids, is_prefill)

    mr = llm.model_runner
    draft_tokens, _ = mr._draft_propose(seqs)          # 第 0 步已融合补写 draft KV（首次进入=写全部 0..N-2）；返回真实 propose 的 d1..dK 作对齐 token 串
    diag = mr.verify_alignment(seqs, draft_tokens, atol=ATOL, rtol=RTOL)

    # 先 dump verification 的 4 个元数据 + block_table/context_len，便于人工核对偏移。
    meta = diag["meta"]
    print(f"===== verification 元数据（seqs={len(seqs)}, K={K}）=====")
    print(f"positions        = {meta['positions']}")
    print(f"slot_mapping     = {meta['slot_mapping']}")
    print(f"cu_seqlens_q     = {meta['cu_seqlens_q']}")
    print(f"cu_seqlens_k     = {meta['cu_seqlens_k']}")
    print(f"context_len/seq  = {meta['context_len_per_seq']}")
    print(f"block_tables     = {meta['block_tables']}")

    print(f"\n===== logits 对齐诊断（Path A verification vs Path B 逐 token decode）=====")
    print(f"max_prob_diff={diag['max_prob_diff']:.4e}  per_position_max_prob_diff={diag['per_position_max_prob_diff']}   (对齐判据，index 0=e..K=dK)")
    print(f"argmax_match={diag['argmax_match']}  num_argmax_mismatch={diag['num_argmax_mismatch']}   (仅参考：near-tie 处 BF16 噪声会翻面)")
    print(f"per_position_max_diff = {diag['per_position_max_diff']}   (logit 绝对差，仅诊断)")
    print(f"max_abs_diff={diag['max_abs_diff']:.4e}  mean_abs_diff={diag['mean_abs_diff']:.4e}")
    print(f"worst_position={diag['worst_position']}  worst_vocab_index={diag['worst_vocab_index']}")
    print(f"allclose(atol={ATOL},rtol={RTOL})={diag['allclose']}   (仅诊断：BF16 下 varlen/decode 两内核的 ULP 级差异，不作判据)")

    # 判据 = 逐位置最大概率差 <= PROB_ATOL：结构错位会让某位置整条 logits 全错 → 某 token 概率差≈1；
    # 而 BF16 两内核只在 near-tie 处翻 argmax，概率差仅 ~1e-2 级。故不用 raw-logit allclose(过严)、
    # 也不用 strict argmax(near-tie 伪报)。max_abs_diff 的量级(结构错 O(1)+ vs ULP ~0.1)是旁证。
    if diag["max_prob_diff"] > PROB_ATOL:
        hint = ("worst_position==0 → 多半是 context/position/slot 构造错；"
                "第 0 位对、后面开始错 → 多半是 causal 对齐 / cu_seqlens / KV 写入偏移。")
        raise AssertionError(
            f"verification 与逐 token decode 的概率分布不一致（max_prob_diff={diag['max_prob_diff']:.4e} > {PROB_ATOL}；{hint}）")
    if not diag["argmax_match"]:
        print(f"\n注：{diag['num_argmax_mismatch']} 处 argmax 翻面但概率差在阈内——BF16 下 varlen(prefill) 与 "
              "decode 两内核在 near-tie 处的正常翻面，采样分布一致，不影响 lossless。")
    print("\nOK: verification tensor/KV 对齐（概率分布一致），可放心进端到端 lossless 判据。")
    # 不显式 llm.exit()：LLMEngine 注册了 atexit 退出回调，进程退出时自动清理一次；
    # 若这里再显式调一次，atexit 第二次进入时 model_runner 已被 del → AttributeError。


if __name__ == "__main__":
    main()
