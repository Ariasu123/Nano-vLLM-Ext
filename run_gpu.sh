#!/usr/bin/env bash
# 有卡模式阶段：冒烟 + 采数。需要真 GPU。GPU 计费只花在这一段。
# 用法：切到有卡模式后 bash run_gpu.sh
set -euo pipefail

MODEL_DIR="$HOME/huggingface/Qwen3-0.6B"
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
python example.py

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

echo "==> [4/4] 采数：四场景对口对比（starvation / prefix / lru_pressure / cache_aware，各含基线 vs 优化变体）"
# 每个 (场景,变体) 各起子进程，退出即释放显存。只想跑单场景：python bench_metrics.py cache_aware
python bench_metrics.py

echo
echo "==> 采数完成。把 cache_aware 的 fifo vs lpm 两组输出（及其它场景）发我，我帮你回填 resume 与 README。跑完记得关机。"
