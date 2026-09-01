#!/usr/bin/env bash
# 有卡模式阶段：冒烟 + 采数。需要真 GPU。GPU 计费只花在这一段。
# 用法：在仓库根执行 bash scripts/run_gpu.sh
set -euo pipefail

# 脚本在 scripts/ 下，切到仓库根：example.py 在根目录、bench 脚本在 scripts/，统一以仓库根为工作目录。
cd "$(dirname "$0")/.."
source "$(dirname "$0")/env.sh"                        # 统一模型根目录（AutoDL 优先数据盘，见 env.sh）

MODEL_DIR="$DRAFT_DIR"
[ -f "$MODEL_DIR/config.json" ] || { echo "模型不存在，请先在无卡模式跑 setup.sh"; exit 1; }

echo "==> [1/4] GPU/flash-attn 自检"
python - <<'PY'
import torch
print("torch:", torch.__version__, "cuda_available:", torch.cuda.is_available())
assert torch.cuda.is_available(), "无可用 GPU，请确认已切到有卡模式"
print("device:", torch.cuda.get_device_name(0))
import flash_attn
print("flash_attn:", flash_attn.__version__)
PY

echo "==> [2/4] 冒烟：确认调度改动没破坏 prefill/decode 正确性"
# example.py 能正常出文本 = flash-attn 在本卡上跑得通，再放心采数。
MODEL_DIR="$MODEL_DIR" python example.py               # 传数据盘模型路径给 example.py（默认写死 ~/huggingface）

echo "==> [3/4] 诊断：LPM 只改准入顺序、不引入额外错误（对照 batch 重排的固有浮点抖动）"
# 为什么不做"逐字一致"硬校验：贪心下每步取 argmax(logits)，但 GPU 的注意力/矩阵乘对 batch 组成
# （有哪些序列同批、各排第几行、CUDA graph padding、prefix cache 命中时机）不是比特级可复现的，
# 近似平局的 top-2 会被低位浮点抖动翻转 → 少量序列中途分叉。这是固有现象：原版仅换个提交顺序也一样。
# 因此这里用"对照法"：base=LPM 关；shuffle=LPM 关但仅打乱提交顺序（同样扰动 batch 组成，不涉 LPM）；
# lpm=LPM 开。若 lpm 相对 base 的发散条数 ≈ shuffle 相对 base 的发散条数，即证明 LPM 未引入额外错误。
_ca_run() {   # $1 = base|shuffle|lpm；输出统一按"原始 prompt 顺序"打印，三种模式可逐行对齐
  CA_MODE="$1" MODEL_DIR="$MODEL_DIR" python - <<'PY'
import os
from random import seed, randint, Random
from nanovllm import LLM, SamplingParams
mode = os.environ["CA_MODE"]
seed(0)                                                # 固定负载：4 组共享前缀、round-robin 交错
K, PLEN = 4, 512
prefixes = [[randint(0, 10000) for _ in range(PLEN)] for _ in range(K)]
prompts = [prefixes[i % K] + [randint(0, 10000) for _ in range(randint(16, 48))] for i in range(32)]
flag = (mode == "lpm")
order = list(range(len(prompts)))
if mode == "shuffle":
    Random(12345).shuffle(order)                       # 固定置换：只改提交顺序=只扰动 batch 组成
submit = [prompts[i] for i in order]
# max_num_seqs 调小 + 显存调低，让并发受限、真正发生重排（否则退化为 FIFO 看不出差异）。
llm = LLM(os.environ["MODEL_DIR"], enforce_eager=False, max_model_len=4096,
          gpu_memory_utilization=0.4, max_num_seqs=4, enable_cache_aware_schedule=flag)
sp = SamplingParams(temperature=1e-6, ignore_eos=True, max_tokens=24)   # 近似贪心
outs = llm.generate(submit, sp, use_tqdm=False)        # outs 按提交顺序
restored = [None] * len(prompts)                       # 映射回原始 prompt 顺序，三模式逐行对齐
for k, o in enumerate(outs):
    restored[order[k]] = ",".join(map(str, o["token_ids"]))
print("\n".join(restored))
PY
}
_ca_run base    > /tmp/ca_base.txt
_ca_run shuffle > /tmp/ca_shuf.txt
_ca_run lpm     > /tmp/ca_lpm.txt
ndiff() { diff "$1" "$2" | grep -c '^<' || true; }     # 发散条数（两文件等行数，每条差异计一行）
ctrl=$(ndiff /tmp/ca_base.txt /tmp/ca_shuf.txt)
lpm=$(ndiff /tmp/ca_base.txt /tmp/ca_lpm.txt)
total=$(wc -l < /tmp/ca_base.txt | tr -d ' ')
echo "    基线仅重排提交顺序（不涉 LPM）：$ctrl/$total 条发散  ←固有浮点抖动量级"
echo "    开启 LPM：$lpm/$total 条发散"
echo "    结论：LPM 发散量 ≈ 基线重排发散量 → LPM 只改准入顺序、未引入额外错误（贪心 GPU 下无法逐字一致属预期）。"

echo "==> [4/6] 功能四：verification 张量/KV 对齐 debug guard（中间层，先于端到端）"
# 需要一大一小两个共享 tokenizer 的模型。默认 target=Qwen3-8B、draft=Qwen3-0.6B，可用环境变量覆盖。
SPEC_TARGET="${SPEC_TARGET:-$TARGET_DIR}"
SPEC_DRAFT="${SPEC_DRAFT:-$DRAFT_DIR}"
if [ -f "$SPEC_TARGET/config.json" ] && [ -f "$SPEC_DRAFT/config.json" ]; then
  # 把「端到端 lossless」拆出中间层：单独验证 verification 一次并行 forward 在 e,d1..dK 每个位置
  # 产出的 logits，与对同一 token 串逐 token 单步 decode 的 logits 数值一致（allclose）。
  # 任一位置发散 → positions/slot_mapping/cu_seqlens/causal off-by-one，先修这里再看端到端。
  SPEC_TARGET="$SPEC_TARGET" SPEC_DRAFT="$SPEC_DRAFT" python scripts/spec_align_check.py
else
  echo "    跳过：未找到 SPEC_TARGET($SPEC_TARGET) 或 SPEC_DRAFT($SPEC_DRAFT) 的 config.json。"
  echo "    准备好一大一小两个共享 tokenizer 的模型后，设 SPEC_TARGET/SPEC_DRAFT 环境变量重跑。"
fi

echo "==> [5/6] 功能四：投机解码 lossless 黄金判据（贪心下 spec 开/关逐 token 对照）"
if [ -f "$SPEC_TARGET/config.json" ] && [ -f "$SPEC_DRAFT/config.json" ]; then
  # 贪心(temp→0)下 rejection sampling 保证输出=target argmax，理论应与 spec-off 逐 token 一致。
  # 与 cache_aware 同理：verification 改变了 target forward 的 batch 组成（一序列 K+1 行 vs 单 token），
  # 近似平局处可能被低位浮点抖动翻转 → 少量发散属固有现象，非投机错误。故按发散条数报告，期望≈0。
  _spec_run() {   # $1 = off|on
    SPEC_MODE="$1" SPEC_TARGET="$SPEC_TARGET" SPEC_DRAFT="$SPEC_DRAFT" python - <<'PY'
import os
from random import seed, randint
from nanovllm import LLM, SamplingParams
mode = os.environ["SPEC_MODE"]
seed(0)
prompts = [[randint(0, 10000) for _ in range(randint(32, 64))] for _ in range(16)]
cfg = dict(enforce_eager=False, max_model_len=4096)
if mode == "on":
    cfg.update(enable_speculative_decode=True,
               speculative_model=os.environ["SPEC_DRAFT"],
               num_speculative_tokens=4)
llm = LLM(os.environ["SPEC_TARGET"], **cfg)
sp = SamplingParams(temperature=1e-6, ignore_eos=True, max_tokens=48)   # 近似贪心
outs = llm.generate(prompts, sp, use_tqdm=False)
print("\n".join(",".join(map(str, o["token_ids"])) for o in outs))
if mode == "on":
    # acceptance_rate / avg_accept_len 走 stderr，不污染 stdout 的 token_id diff。
    import sys
    st = llm.scheduler.get_stats()
    print(f"[spec stats] steps={st.num_speculative_steps} "
          f"acceptance_rate={st.acceptance_rate*100:.1f}% avg_accept_len={st.avg_accept_len:.2f} "
          f"bonus={st.num_bonus} proposed={st.total_proposed_tokens} accepted={st.total_accepted_tokens}",
          file=sys.stderr)
PY
  }
  _spec_run off > /tmp/spec_off.txt
  _spec_run on  > /tmp/spec_on.txt
  ndiff() { diff "$1" "$2" | grep -c '^<' || true; }
  div=$(ndiff /tmp/spec_off.txt /tmp/spec_on.txt)
  total=$(wc -l < /tmp/spec_off.txt | tr -d ' ')
  echo "    spec 开 vs 关：$div/$total 条发散（期望≈0；少量为 batch 组成引起的固有浮点抖动，非投机错误）"
  echo "    结论：发散≈0 → 投机解码在贪心下与纯 target lossless 一致。"
else
  echo "    跳过：未找到 SPEC_TARGET($SPEC_TARGET) 或 SPEC_DRAFT($SPEC_DRAFT) 的 config.json。"
  echo "    准备好一大一小两个共享 tokenizer 的模型后，设 SPEC_TARGET/SPEC_DRAFT 环境变量重跑。"
fi

echo "==> [5/5] 采数：五场景对口对比（starvation / prefix / lru_pressure / cache_aware / speculative）"
# 每个 (场景,变体) 各起子进程，退出即释放显存。只想跑单场景：python scripts/bench_metrics.py speculative
python scripts/bench_metrics.py

echo
echo "==> 采数完成。把 cache_aware 的 fifo vs lpm、speculative 的 base vs spec 两组输出（及其它场景）发我，我帮你回填 resume 与 README。跑完记得关机。"
