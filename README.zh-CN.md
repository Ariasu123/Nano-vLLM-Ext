# Nano-vLLM-Ext

[English](README.md)

在 [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) 基线上补齐 serving 层能力——
Chunked Prefill、LRU 前缀缓存、Cache-Aware（LPM）调度、Draft-Target 投机解码。
每项优化都配一个**专门构造的对口微基准**，与原版做单变量对比。

## 亮点

- **Chunked Prefill 消除 decode 饥饿：短请求 TPOT P99 822.9ms → 117.6ms（−86%）**。
  长 Prompt 连续 prefill 期间，在途请求不再被饿住。
- **Cache-Aware（LPM）调度：前缀命中率 0 → 72.9%、驱逐 −76%、端到端 −31%**。
  只改等待队列的出队顺序，不改缓存容量。
- **Draft-Target 投机解码 lossless 已验证**（贪心输出与单独跑 target 逐 token 一致；
  draft/target logit 对齐 `max_prob_diff=1.7e-2`），并用分段 profiling **证伪了初始假设**，
  把瓶颈归因到 kernel 启动开销而非模型算力。
- **四项优化全部默认关闭、关时逐字节等价于原版、含 CPU 单测**——每个数字都能一键对照复现。

## 实测环境

| | |
|---|---|
| GPU | RTX 5090（投机解码场景为 AutoDL 租用实例） |
| 模型 | Qwen3-0.6B（功能一~三）；Qwen3-8B target + Qwen3-0.6B draft（投机解码） |
| 配置 | `max_model_len=4096`，固定随机种子 |
| 采数 | `scripts/bench_metrics.py`，每个 `(场景, 变体)` 跑在独立子进程，保证 GPU 显存在两次运行间完全释放 |

> **为什么功能一~三只用 0.6B**：它们优化的是调度层与缓存层，收益来源（混批消除 decode 饥饿、
> 前缀块复用与驱逐、LPM 出队顺序）与模型规模解耦。用小模型可以隔离"模型计算"这一变量、
> 加快迭代并节省 GPU，换更大模型上述比率结论同样成立。
> **为什么功能四必须一大一小**：投机解码的收益前提是 target 单步 decode 成本远高于 draft，
> 因此用 Qwen3-8B 作 target、Qwen3-0.6B 作 draft。

## 优化项一览

| 功能 | 机制 | 对口场景 | 关键结果 | Takeaway |
|---|---|---|---|---|
| Chunked Prefill | prefill/decode 混批，一步内先给在途 decode 排 1 token，剩余预算做分块 prefill | `starvation` | TPOT P50 −79%、P99 −86% | decode 饥饿是**调度**问题，不是算力问题 |
| Prefix Cache | block-hash 前缀复用（上游能力，做基线量化） | `prefix` | 命中率 88.3%、TTFT −58%、端到端 −60% | 共享系统前缀是最便宜的一笔优化 |
| LRU 驱逐 | 容量受限 LRU 取代 FIFO 驱逐 | `lru_pressure` | 驱逐 −24%、命中率 +2.1pp | 只有缓存受压时策略才有区分度 |
| Cache-Aware 调度 | 等待队列按"已缓存前缀最长优先"出队 + aging 防饿死 | `cache_aware` | 命中率 0→72.9%、驱逐 −76%、端到端 −31% | **只改顺序、不改容量**，也能造出命中率 |
| 投机解码 | Draft-Target + 精确拒绝采样 + KV 事务化提交 | `speculative` | lossless ✅；吞吐当前为负 | 小 batch eager 下 **kernel 启动开销主导**，"draft 小所以便宜"不成立 |

## 架构

![引擎与 KV Cache 架构](assets/Engine.png)

输入请求从 Scheduler 的 waiting 队列进入 running 队列，并被组装为执行批次。
BlockManager 通过每条序列的 block table、物理块池和 Prefix Cache 管理 KV Cache 块的分配、复用与释放。
Rank 0 负责协调调度和命令，Worker 则携带相应 block table 执行自己的模型分片。
上述四项优化全部落在 Scheduler / BlockManager 这一层，模型层零改动。

## 结果

每张表都是**原版 vs 优化项**，同一场景下所有变体看到完全相同的请求批次。

### 1. `starvation` —— 混批消除 decode 饥饿

先提交 96 条短请求（输出=2），再提交 96 条长请求（输入 1200–1600）制造连续 prefill。
`no_mix`（`enable_chunked_prefill=False`）是原版两阶段调度；`mix`（`enable_chunked_prefill=True`）
把 prefill 与 decode 混排在同一步。

两阶段调度下短请求要等所有长 prefill 跑完才能 decode，其唯一的 token 间隔横跨整段 prefill 突发；
把每条 running 序列的 1 个 decode token 混入每一步后，该间隔收敛为一步。

| 短请求 TPOT | `no_mix`（两阶段） | `mix` | 变化 |
|---|---|---|---|
| P50 | 552.6ms | 117.2ms | **−79%** |
| P99 | 822.9ms | 117.6ms | **−86%** |

### 2. `prefix` —— 共享前缀复用

128 条请求，每条带 1024-token 系统前缀。`no_reuse` 每条前缀唯一；`shared` 共用一条可被缓存复用的前缀。

| 指标 | `no_reuse` | `shared` | 变化 |
|---|---|---|---|
| 命中率 | 0% | 88.3% | — |
| TTFT P50 | 579ms | 241ms | **−58%** |
| 端到端耗时 | 1.36s | 0.54s | **−60%** |
| prefill 步数 | 10 | 3 | — |

### 3. `lru_pressure` —— 缓存受压下 LRU vs FIFO

128 条不同的 512-token 前缀，384 条请求按幂律偏斜复用，`gpu_memory_utilization=0.4` 强制驱逐。
`fifo`（`enable_lru=False`）vs `lru`。

| 指标 | `fifo` | `lru` | 变化 |
|---|---|---|---|
| 命中率 | 60.4% | 62.5% | +2.1pp |
| 驱逐次数 | 82 | 62 | **−24%** |

### 4. `cache_aware` —— 缓存驱逐压力下 LPM vs FIFO 调度

64 个不同的 1024-token 前缀，512 条请求按 round-robin 交错到达（相邻请求命中不同前缀），
`gpu_memory_utilization=0.15` 使前缀工作集超出缓存容量。
`fifo`（`enable_cache_aware_schedule=False`）按到达顺序服务等待队列；
`lpm`（`enable_cache_aware_schedule=True`）优先服务"已缓存前缀最长"的请求，
把同前缀请求聚拢、赶在被驱逐前复用其块。**两变体都开 `enable_lru=True`，仅调度顺序不同。**

| 指标 | `fifo` | `lpm` | 变化 |
|---|---|---|---|
| 命中率 | 0% | 72.9% | — |
| 驱逐次数 | 1968 | 470 | **−76%** |
| TTFT P50 | 2894ms | 1773ms | **−39%** |
| 端到端耗时 | 5.53s | 3.82s | **−31%** |

**Tradeoff（诚实披露）**：LPM 让 prefill 更快排空，更多序列并发 decode，TPOT P50 由 5.1ms 升至 8.4ms；
该收益**仅在前缀工作集超出缓存容量时出现**，故设计为默认关闭的开关。

### 5. `speculative` —— Draft-Target 投机解码

在同一个 64 请求闭合批次上做 apples-to-apples 对比。Qwen3-8B 作 target、Qwen3-0.6B 作 draft，默认 **K=1**。

#### 5.1 先隔离混淆变量

三个变体把**投机算法本身**与 **CUDA Graph** 分开：
`base`（仅 target，开 CUDA Graph）、`base_eager`（仅 target，eager）、`spec`（投机，eager）。
投机路径当前跑在 eager 下，所以**只有 `base_eager` vs `spec` 是公平对比**——
拿 `base` 去比会把 CUDA Graph 的收益错算成投机算法的损失。

| 指标 | `base`（graph） | `base_eager` | `spec` K=1 |
|---|---|---|---|
| 端到端耗时 | 2.40s | 3.91s | 6.58s |
| TPOT P50 | 15.3ms | 27ms | 39.1ms |

#### 5.2 正确性先立住（独立于速度）

- draft/target 的 logit 对齐误差 `max_prob_diff=1.7e-2`（判据 0.3），argmax 一致
- 拒绝采样**无损**：贪心输出与单独跑 target 模型**逐 token 一致**
- 全 K 下 `reject: draft=0 target=0 fallback=0`，无结构性拒绝或降级

#### 5.3 sync–propose 融合：每步少一次 draft forward

原设计每步要跑 K+1 次 draft forward——一次独立的 forward 补写落后的 draft KV，再加 K 次 propose。
**融合**把补写折叠进 propose 第 0 步：每序列 `[nc, N)` 的**一次 varlen prefill** 同时完成
(1) 补写 draft KV（含末 token e）(2) 从末位 logit 采出 d1，使每步 draft forward 数从 K+1 降到 K。

融合只改变 draft 的 **proposal 分布**（prefill vs decode 两内核的数值差异），由拒绝采样无条件校正，
losslessness 不受影响。收益在低 K 最大（K=1 时每个全接受步都省下一次 forward）：
**端到端 8.01s → 6.58s（−18%）**。

#### 5.4 K-sweep：与直觉相反

| K | 端到端 | TPOT P50 | decode 吞吐 | acceptance | avg_accept_len |
|---|---|---|---|---|---|
| **1** | **6.58s** | **39.1ms** | **1332 tok/s** | 76.5% | 0.77 |
| 2 | 7.30s | 40.5ms | 1198 tok/s | 67.0% | 1.34 |
| 3 | 8.29s | 41.7ms | 1047 tok/s | 61.0% | 1.83 |
| 4 | 9.28s | 46.5ms |  933 tok/s | 53.5% | 2.14 |

K 越大 `avg_accept_len` 确实越高（0.77 → 2.14，**说明算法本身健康**），
但 wall / TPOT / 吞吐三项**均随 K 单调变差**，K=1 全指标最优。这个反直觉结果需要解释——见下。

#### 5.5 当前它跑不赢基线，profiling 说明了原因

K=1 仍比 `base_eager` 慢 **1.7×**（6.58s vs 3.91s）。
`SPEC_PROFILE=1` 用 CUDA Event 记录一步内各分段（步末只做一次 `torch.cuda.synchronize()`，关闭时零开销）。
K=1 稳态（batch 64，单步 ~55ms）：

| 分段 | 耗时 | 占比 |
|---|---|---|
| draft forward ×1（融合：补写 KV + propose d1） | 22.6ms | 41% |
| target 8B verify forward ×1 | 29.8ms | 54% |
| logits + softmax + 拒绝采样 + 后处理 | ~2.6ms | 5% |

**关键观察**：0.6B draft 的单次 eager forward（~22ms）与 8B target verify（~30ms），
在 batch 从 64 降到 1 时**几乎不变**。说明耗时是每次 forward 固定的
**kernel 启动 + Python 调度开销**，而非 GPU 矩阵乘——后者在这个 batch 规模下根本看不见。

所以「draft 模型小、所以 propose 便宜」是**错的**：开销才是主导，一次 draft forward 就吃掉
target verify 的约 2/3。投机解码要赚钱的前提 `draft_cost ≪ target_cost` 在 eager 下不成立，
于是无论 K 取多少都赢不了——这同时解释了 5.4 的「K 越大越慢」和「K=1 最优」。

**被证伪的假设**：最初猜「全词表 LM head + FP32 概率张量（`[S,K+1,V]`）很贵」。
profiling 显示这条路径合计只占 ~2.6ms / 5%，假设不成立。先 profile、再优化。

**口径说明**：这是个*闭合*微批（没有新请求补入），序列每步前进 1–2 token 而逐渐失步，
batch 占用率衰减成长尾；真实服务有持续到达的请求会把 batch 填满。

#### 5.6 下一步

两侧 forward 都上 **CUDA Graph** 摊薄固定开销——定形状的前向正是 graph capture 能消除的开销。
target verify 侧的难点在于它走 varlen 路径（`max_seqlen_k` 是随序列增长的 Python host int，无法图捕获），
需改走 `flash_attn_with_kvcache` 的多-query 路径（每序列固定 K+1 个 query）才可捕获；
draft 侧 propose 步 1..K−1 是标准单 token decode，可直接套现有 decode 图的模式。

---

每个变体报告的指标：TTFT/TPOT P50/P99、prefill/decode 吞吐、峰值显存、
调度统计（prefill/decode 步数、抢占次数）、prefix cache 统计（命中率、节省 token 数、驱逐次数）。

## 深入阅读

每项功能的设计取舍、边界条件与实现细节：

- [chunked-prefill.md](docs/chunked-prefill.md) —— prefill/decode 混批的预算分配与 varlen attention 复用
- [prefix-cache-lru.md](docs/prefix-cache-lru.md) —— block-hash 前缀缓存与容量受限 LRU 驱逐
- [cache-aware-scheduling.md](docs/cache-aware-scheduling.md) —— LPM 打分、aging 防饿死与队列深度解耦
- [metrics-and-benchmark.md](docs/metrics-and-benchmark.md) —— 指标采集底座与 benchmark 框架

## 复现

需要 Linux x86_64 + CUDA 12 + PyTorch 的环境。脚本在 `scripts/`，按「无卡准备 → 有卡评测」
两阶段组织，把 GPU 计费只压在最后一步：

```bash
bash scripts/setup.sh              # 无卡：装依赖 + flash-attn + 下 Qwen3-0.6B + 跑 CPU 单测
bash scripts/download_models.sh    # 下投机解码用的两个模型（Qwen3-8B target + Qwen3-0.6B draft）
bash scripts/run_gpu.sh            # 有卡：自检 → 冒烟 → LPM 对齐诊断 → 投机 lossless 判据 → 五场景采数

python scripts/bench_metrics.py prefix   # 也可单独跑某个场景
```

模型路径、镜像源等统一在 `scripts/env.sh` 中用环境变量覆盖。

## 上游致谢

Nano-vLLM-Ext 基于 [GeeeekExplorer/nano-vllm](https://github.com/GeeeekExplorer/nano-vllm)，
并由本仓库独立维护。感谢上游项目及其贡献者的工作。
