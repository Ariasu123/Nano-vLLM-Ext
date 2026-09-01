<p align="center">
<img width="300" src="assets/logo.png">
</p>

# Nano-vLLM-Ext

[English](README.md)

Nano-vLLM-Ext 是一个独立维护的 Nano-vLLM 扩展版本:


### 扩展特性

- Draft–Target 投机解码
- Chunked Prefill（prefill/decode 混批）
- 缓存感知调度（LPM），底层 LRU 前缀缓存支撑

## 架构

### Qwen 推理架构

![Qwen 推理架构](assets/Qwen_arch.png)

输入 Token ID 先经过 Token Embedding 与 RoPE，再穿过重复的 Decoder 层，最后由归一化层和 LM Head 生成下一个 Token 的 logits。每个 Decoder 层内，GQA 使用紧凑的 KV Cache 完成注意力计算，SwiGLU 前馈网络则变换隐藏状态。该图展示了这些组件如何串成一条完整的推理路径。

### 引擎与 KV Cache

![引擎与 KV Cache 架构](assets/Engine.png)

输入请求从 Scheduler 的 waiting 队列进入 running 队列，并被组装为执行批次。BlockManager 通过每条序列的 block table、物理块池和 Prefix Cache 管理 KV Cache 块的分配、复用与释放。Rank 0 负责协调调度和命令，Worker 则携带相应 block table 执行自己的模型分片。

### 张量并行执行

![张量并行执行流程](assets/TP-expanded.png)

Checkpoint 权重被拆分到各个张量并行 Rank，其中列并行层和行并行层共同分担模型计算。Prefill 和 Decode 阶段中，每个 Rank 执行本地分片，并在需要合并部分激活值时执行 `all_reduce`。Rank 0 汇聚词表 logits、采样下一个 Token 并返回结果；Worker Rank 只完成分配给自己的分片计算。


## 快速开始（复现评测）

笔者本地是 macOS（无 NVIDIA GPU），评测在租用的 GPU 服务器上进行，下文以 [AutoDL](https://www.autodl.com/) 为例：选购实例时选**带 PyTorch 的官方镜像**（CUDA 12），开机后把项目 `scp` 到服务器；模型体积大，务必下到数据盘 `/root/autodl-tmp`（系统盘通常仅 ~30GB，脚本已默认落数据盘，见下）。换其他云厂商或本地有卡机器同理，满足下方环境前提即可。

评测脚本都在 `scripts/`，按「无卡准备 → 有卡评测」两阶段组织，把 GPU 计费只压在最后一步。把整个项目拷到服务器后，在**仓库根**依次执行：

```bash
bash scripts/setup.sh              # 无卡：装依赖 + pip install -e . + flash-attn + 下 Qwen3-0.6B + 跑 CPU 单测
bash scripts/download_models.sh    # 下投机解码用的一大一小两个模型（Qwen3-8B target + Qwen3-0.6B draft）
bash scripts/run_gpu.sh            # 有卡：GPU 自检 → 冒烟 → LPM 对齐诊断 → 投机 lossless 判据 → 五场景采数
```

**环境前提**：Linux x86_64 + 已装 PyTorch 的 CUDA 12 环境（如 AutoDL 官方 PyTorch 镜像）。脚本本身不装 torch，flash-attn 直接下官方预编译 wheel（`cu12` + `linux_x86_64`）。

**模型存放**：默认落数据盘——检测到 AutoDL 数据盘 `/root/autodl-tmp` 时用 `/root/autodl-tmp/models`，否则回退 `~/huggingface`；可用 `MODEL_ROOT` / `TARGET_DIR` / `DRAFT_DIR` 覆盖，统一定义在 `scripts/env.sh`。下载默认走 `hf-mirror` 国内镜像（`HF_ENDPOINT` 可覆盖）。

**显存**：Qwen3-8B(bf16) 权重约 16GB，另需 KV Cache 余量。显存紧张可换更小的 target：`TARGET_REPO=Qwen/Qwen3-4B bash scripts/download_models.sh`，再 `SPEC_TARGET=$MODEL_ROOT/Qwen3-4B bash scripts/run_gpu.sh`。

只跑功能一~三（不含投机解码）时，`setup.sh` 下好 Qwen3-0.6B 即可，跳过 `download_models.sh`；`run_gpu.sh` 里投机相关步骤会因缺模型自动跳过。

## Benchmark

`scripts/bench_metrics.py` 在针对每个优化项专门构造的场景下，对比原版引擎与该优化项。每个 `(场景, 变体)` 在独立子进程中运行，确保 GPU 显存在两次运行间完全释放；各场景使用固定随机种子，因此同一场景的所有变体看到完全相同的请求批次。需要 CUDA GPU（实测环境：RTX 5090、Qwen3-0.6B、`max_model_len=4096`）。

```bash
python scripts/bench_metrics.py                 # 全部场景、全部变体
python scripts/bench_metrics.py prefix          # 仅某一个场景
python scripts/bench_metrics.py prefix shared   # 单个变体（子进程内部使用）
```

五个场景，每个隔离一个优化项（原版 vs 优化项）。每张结果表都是基线 vs 优化对比。

> 功能一~三优化的是调度层与缓存层，收益来源（混批消除 decode 饥饿、前缀块复用与驱逐、LPM 出队顺序）与模型规模解耦，故在 Qwen3-0.6B 上测量以隔离"模型计算"这一变量、加快迭代并节省 GPU，换更大模型上述比率结论同样成立。功能四投机解码必须一大一小两个模型才有意义（target 越大、decode 单步成本越高，"一步 target forward 提交多 token"的吞吐收益才显著），故用 Qwen3-8B 作 target、Qwen3-0.6B 作 draft。

### `starvation` —— 混批消除 decode 饥饿

先提交 96 条短请求（输出=2），再提交 96 条长请求（输入 1200–1600）制造连续 prefill。`no_mix`（`enable_chunked_prefill=False`）是原版两阶段调度；`mix`（`enable_chunked_prefill=True`）把 prefill 与 decode 混排在同一步。两阶段调度下短请求要等所有长 prefill 跑完才能 decode，其唯一的 token 间隔横跨整段 prefill 突发；把每条 running 序列的 1 个 decode token 混入每一步后，该间隔收敛为一步。

| 短请求 TPOT | `no_mix`（两阶段） | `mix` | 变化 |
|---|---|---|---|
| P50 | 552.6ms | 117.2ms | −79% |
| P99 | 822.9ms | 117.6ms | −86% |

### `prefix` —— 共享前缀复用

128 条请求，每条带 1024-token 系统前缀。`no_reuse` 每条前缀唯一；`shared` 共用一条可被缓存复用的前缀。

| 指标 | `no_reuse` | `shared` | 变化 |
|---|---|---|---|
| 命中率 | 0% | 88.3% | — |
| TTFT P50 | 579ms | 241ms | −58% |
| 端到端耗时 | 1.36s | 0.54s | −60% |
| prefill 步数 | 10 | 3 | — |

### `lru_pressure` —— 缓存受压下 LRU vs FIFO

128 条不同的 512-token 前缀，384 条请求按幂律偏斜复用，`gpu_memory_utilization=0.4` 强制驱逐。`fifo`（`enable_lru=False`）vs `lru`。

| 指标 | `fifo` | `lru` | 变化 |
|---|---|---|---|
| 命中率 | 60.4% | 62.5% | +2.1pp |
| 驱逐次数 | 82 | 62 | −24% |

### `cache_aware` —— 缓存驱逐压力下 LPM vs FIFO 调度

64 个不同的 1024-token 前缀，512 条请求按 round-robin 交错到达（相邻请求命中不同前缀），`gpu_memory_utilization=0.15` 使前缀工作集超出缓存容量。`fifo`（`enable_cache_aware_schedule=False`）按到达顺序服务等待队列；`lpm`（`enable_cache_aware_schedule=True`）优先服务"已缓存前缀最长"的请求，把同前缀请求聚拢、赶在被驱逐前复用其块。两变体都开 `enable_lru=True`，仅调度顺序不同。

| 指标 | `fifo` | `lpm` | 变化 |
|---|---|---|---|
| 命中率 | 0% | 72.9% | — |
| 驱逐次数 | 1968 | 470 | −76% |
| TTFT P50 | 2894ms | 1773ms | −39% |
| 端到端耗时 | 5.53s | 3.82s | −31% |

Tradeoff：LPM 让 prefill 更快排空，更多序列并发 decode，TPOT P50 由 5.1ms 升至 8.4ms；该收益仅在前缀工作集超出缓存容量时出现，故为默认关闭的开关。

### `speculative` —— draft/target 投机解码（Qwen3-8B 作 target + Qwen3-0.6B 作 draft）

在同一个 64 请求闭合批次上做 apples-to-apples 对比。三个变体把投机算法本身与 CUDA Graph 这一混淆变量分离开：`base`（仅 target，CUDA Graph）、`base_eager`（仅 target，eager）、`spec`（投机，eager）。`base_eager` vs `spec` 才是公平对比——两者都 eager。

默认 **K=1**。**sync–propose 融合**把「补写落后 draft KV 的独立 forward」折叠进 propose 第 0 步：每序列 `[nc, N)` 的一次 varlen prefill 同时补写 draft KV（含末 token e）并从末位 logit 采出 d1，使每步 draft forward 数从 K+1 降到 K。融合只改变 draft 的 proposal 分布（prefill vs decode 两内核的数值差异），由拒绝采样无条件校正，losslessness 不受影响；收益在低 K 最大（K=1 每个全接受步都省下一次 forward），实测 K=1 端到端 8.01s → 6.58s（−18%）。

| 指标 | `base`（graph） | `base_eager` | `spec` K=1 |
|---|---|---|---|
| 端到端耗时 | 2.40s | 3.91s | 6.58s |
| TPOT P50 | 15.3ms | 27ms | 39.1ms |

K-sweep（融合后，满批 64）：K 越大单请求 `avg_accept_len` 越高（K=1→4：0.77→2.14），但 eager 下每个 draft forward 固定 ~20ms、与 target 的 ~30ms 同量级，堆 draft 成本比省 target verify 更快，故 wall/TPOT/吞吐三项均随 K 单调变差——**K=1 全指标最优**：

| K | 端到端 | TPOT P50 | decode 吞吐 | acceptance | avg_accept_len |
|---|---|---|---|---|---|
| 1 | 6.58s | 39.1ms | 1332 tok/s | 76.5% | 0.77 |
| 2 | 7.30s | 40.5ms | 1198 tok/s | 67.0% | 1.34 |
| 3 | 8.29s | 41.7ms | 1047 tok/s | 61.0% | 1.83 |
| 4 | 9.28s | 46.5ms |  933 tok/s | 53.5% | 2.14 |

正确性独立于速度已验证：draft/target 的 logit 对齐误差 `max_prob_diff=1.7e-2`（P6，判据 0.3）；拒绝采样无损——贪心输出与单独跑 target 模型逐 token 一致（P7）；全 K 下 `reject: draft=0 target=0 fallback=0`，无结构性拒绝或降级。

**但融合后 K=1 仍比 eager 基线慢 1.7×（6.58s vs 3.91s），profiling 给出了原因。** `SPEC_PROFILE=1` 用 CUDA Event 记录一步内各分段（步末只做一次 `torch.cuda.synchronize()`，关闭时零开销）。K=1 稳态（batch 64，单步 ~55ms）：

| 分段 | 耗时 | 占比 |
|---|---|---|
| draft forward ×1（融合：补写 KV + propose d1） | 22.6ms | 41% |
| target 8B verify forward ×1 | 29.8ms | 54% |
| logits + softmax + 拒绝采样 + 后处理 | ~2.6ms | 5% |

关键观察：0.6B draft 的单次 eager forward（~22ms）与 8B target verify（~30ms）在 batch 从 64 降到 1 时*几乎不变*——耗时是每次 forward 固定的 kernel 启动 + Python 调度开销，而非 GPU 矩阵乘（后者在这个 batch 规模下根本看不见）。所以"draft 小、所以便宜"是错的：开销才是主导，一次 draft forward 就吃掉 target verify 的约 2/3。这既解释了「K 越大越慢」，也是默认取 K=1（forward 数最少）的原因。早期"全词表 LM head + FP32 概率张量很贵"的假设被证伪——那条路径只占 ~2.6ms。

在这个*闭合*微批上（没有新请求补入），序列每步前进 1–2 token 而逐渐失步，batch 占用率衰减成长尾——真实服务有持续到达的请求会把 batch 填满。

结论：算法与接受率正确且健康，融合消除了 per-step 的独立 sync forward。剩余瓶颈是 draft 的 eager forward（纯 kernel 启动开销）；已识别的下一步是 **draft-decode CUDA Graph**——定形状的单 token decode 正是 graph capture 能消除的开销，它会把 ~22ms 的 draft forward 压到数 ms，从而反转 K 的权衡、让 higher-K 的多接受收益真正兑现。

每个变体报告的指标：TTFT/TPOT P50/P99、prefill/decode 吞吐、峰值显存、调度统计（prefill/decode 步数、抢占次数）、prefix cache 统计（命中率、节省 token 数、驱逐次数）。

## 上游致谢

Nano-vLLM-Ext 基于 [GeeeekExplorer/nano-vllm](https://github.com/GeeeekExplorer/nano-vllm)，并由本仓库独立维护。感谢上游项目及其贡献者的工作。
