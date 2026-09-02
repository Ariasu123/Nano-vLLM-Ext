# Nano-vLLM-Ext

[简体中文](README.zh-CN.md)

Serving-layer extensions on top of the [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) baseline —
Chunked Prefill, LRU prefix cache, Cache-Aware (LPM) scheduling, and Draft-Target speculative decoding.
Every enhancement ships with a **purpose-built microbenchmark** that isolates it against the original engine.

## Highlights

- **Chunked Prefill eliminates decode starvation: victim TPOT P99 822.9ms → 117.6ms (−86%)**.
  In-flight requests are no longer starved during a burst of long-prompt prefills.
- **Cache-Aware (LPM) scheduling: prefix hit rate 0 → 72.9%, evictions −76%, wall time −31%**.
  Achieved by changing only the dequeue order of the waiting queue — cache capacity is untouched.
- **Draft-Target speculative decoding verified lossless** (greedy output is token-identical to running
  the target alone; draft/target logit alignment holds to `max_prob_diff=1.7e-2`), and segment-level
  profiling **disproved the initial hypothesis**, attributing the bottleneck to kernel-launch overhead
  rather than model compute.
- **All four features are default-off, byte-identical to the original when off, and unit-tested** —
  every number here reproduces as a one-flag A/B.

## Measurement Setup

| | |
|---|---|
| GPU | RTX 5090 (speculative-decoding runs on a rented AutoDL instance) |
| Models | Qwen3-0.6B (features 1–3); Qwen3-8B target + Qwen3-0.6B draft (speculative) |
| Config | `max_model_len=4096`, fixed random seed |
| Harness | `scripts/bench_metrics.py` — each `(scenario, variant)` runs in its own subprocess so GPU memory is fully released between runs |

> **Why features 1–3 use only 0.6B**: they optimize the scheduling and caching layers, and their gains
> (mixed batching removing decode starvation, prefix-block reuse and eviction, LPM dequeue order) are
> decoupled from model size. A small model isolates the "model compute" variable, speeds up iteration,
> and saves GPU; the same ratios hold on larger models.
> **Why feature 4 needs a large + small pair**: speculative decoding only pays off when the target's
> per-step decode cost far exceeds the draft's, hence Qwen3-8B as target and Qwen3-0.6B as draft.

## Enhancements at a Glance

| Feature | Mechanism | Scenario | Key result | Takeaway |
|---|---|---|---|---|
| Chunked Prefill | Mixed prefill/decode batching: schedule 1 token for each running decode first, then chunk-prefill with the remaining budget | `starvation` | TPOT P50 −79%, P99 −86% | Decode starvation is a **scheduling** problem, not a compute problem |
| Prefix Cache | Block-hash prefix reuse (upstream capability, quantified as a baseline) | `prefix` | Hit rate 88.3%, TTFT −58%, wall −60% | A shared system prefix is the cheapest optimization available |
| LRU eviction | Capacity-bounded LRU replacing FIFO eviction | `lru_pressure` | Evictions −24%, hit rate +2.1pp | Eviction policy only differentiates itself under cache pressure |
| Cache-Aware scheduling | Dequeue by "longest cached prefix first" + aging to prevent starvation | `cache_aware` | Hit rate 0→72.9%, evictions −76%, wall −31% | **Order alone, without more capacity**, can manufacture hit rate |
| Speculative decoding | Draft-Target + exact rejection sampling + transactional KV commit | `speculative` | Lossless ✅; throughput currently negative | At small batch in eager, **kernel-launch overhead dominates** — "the draft is small, therefore cheap" is false |

## Architecture

![Engine and KV cache architecture](assets/Engine.png)

Incoming requests move from the scheduler's waiting queue to the running queue and are assembled into
execution batches. BlockManager maintains each sequence's block table, physical block pool, and Prefix
Cache so KV-cache blocks can be allocated, reused, or released. Rank 0 coordinates scheduling and
commands, while workers execute their model shards with the required block tables.
All four enhancements live in the Scheduler / BlockManager layer — the model layer is unchanged.

## Results

Every table is **original vs. enhancement**; within a scenario, all variants see the identical request batch.

### 1. `starvation` — mixed batching eliminates decode starvation

96 short victims (out=2) are submitted first, then 96 long blockers (input 1200–1600) force consecutive
prefill. `no_mix` (`enable_chunked_prefill=False`) is the original two-phase scheduler; `mix`
(`enable_chunked_prefill=True`) mixes prefill and decode in one step.

Under two-phase scheduling a victim cannot decode until every long prefill finishes, so its single
inter-token gap spans the whole prefill burst; mixing one decode token per running sequence into each
step collapses that gap to a single step.

| Victim TPOT | `no_mix` (two-phase) | `mix` | Change |
|---|---|---|---|
| P50 | 552.6ms | 117.2ms | **−79%** |
| P99 | 822.9ms | 117.6ms | **−86%** |

### 2. `prefix` — shared-prefix reuse

128 requests each carry a 1024-token system prefix. `no_reuse` gives every request a unique prefix;
`shared` gives them one common prefix the cache can reuse.

| Metric | `no_reuse` | `shared` | Change |
|---|---|---|---|
| Hit rate | 0% | 88.3% | — |
| TTFT P50 | 579ms | 241ms | **−58%** |
| Wall time | 1.36s | 0.54s | **−60%** |
| Prefill steps | 10 | 3 | — |

### 3. `lru_pressure` — LRU vs FIFO under cache pressure

128 distinct 512-token prefixes, 384 requests with power-law skewed reuse,
`gpu_memory_utilization=0.4` to force eviction. `fifo` (`enable_lru=False`) vs `lru`.

| Metric | `fifo` | `lru` | Change |
|---|---|---|---|
| Hit rate | 60.4% | 62.5% | +2.1pp |
| Evictions | 82 | 62 | **−24%** |

### 4. `cache_aware` — LPM scheduling under prefix-eviction pressure

64 distinct 1024-token prefixes, 512 requests arriving round-robin (adjacent requests hit different
prefixes), `gpu_memory_utilization=0.15` so the prefix working set exceeds cache capacity.
`fifo` (`enable_cache_aware_schedule=False`) serves the waiting queue in arrival order;
`lpm` (`enable_cache_aware_schedule=True`) prioritizes the request whose cached prefix is longest,
clustering same-prefix requests so their blocks are reused before eviction.
**Both variants keep `enable_lru=True`; only the scheduling order differs.**

| Metric | `fifo` | `lpm` | Change |
|---|---|---|---|
| Hit rate | 0% | 72.9% | — |
| Evictions | 1968 | 470 | **−76%** |
| TTFT P50 | 2894ms | 1773ms | **−39%** |
| Wall time | 5.53s | 3.82s | **−31%** |

**Tradeoff (disclosed)**: LPM drains prefill faster so more sequences decode concurrently, raising
TPOT P50 from 5.1ms to 8.4ms; the gain **only appears when the prefix working set exceeds cache
capacity**, so it ships as a default-off flag.

### 5. `speculative` — Draft-Target speculative decoding

Apples-to-apples on the same closed batch of 64 requests. Qwen3-8B as target, Qwen3-0.6B as draft,
default **K=1**.

#### 5.1 Isolate the confound first

Three variants separate the **speculative algorithm** from **CUDA Graph**:
`base` (target only, CUDA Graph on), `base_eager` (target only, eager), `spec` (speculative, eager).
The speculative path currently runs eager, so **only `base_eager` vs `spec` is a fair comparison** —
comparing against `base` would miscount CUDA Graph's gain as the speculative algorithm's loss.

| Metric | `base` (graph) | `base_eager` | `spec` K=1 |
|---|---|---|---|
| Wall time | 2.40s | 3.91s | 6.58s |
| TPOT P50 | 15.3ms | 27ms | 39.1ms |

#### 5.2 Correctness established independently of speed

- Draft/target logit alignment: `max_prob_diff=1.7e-2` (threshold 0.3), argmax matches
- The rejection sampler is **lossless**: greedy output is **token-identical** to running the target alone
- Across all K: `reject: draft=0 target=0 fallback=0` — no structural rejections or fallbacks

#### 5.3 Sync–propose fusion: one fewer draft forward per step

The original design ran K+1 draft forwards per step — one standalone forward to backfill the lagging
draft KV, plus K propose forwards. The **fusion** folds the backfill into propose step 0: a **single
varlen prefill** over `[nc, N)` per sequence both (1) backfills the draft KV (including the last token e)
and (2) samples d1 from the last position's logit, cutting draft forwards per step from K+1 to K.

The fusion only changes the draft's **proposal distribution** (prefill vs decode kernel numerics), which
the rejection sampler corrects unconditionally, so losslessness is unaffected. The gain is largest at
low K (at K=1 every all-accept step saves a forward): **8.01s → 6.58s end-to-end (−18%)**.

#### 5.4 K-sweep: the counterintuitive part

| K | Wall | TPOT P50 | decode tput | acceptance | avg_accept_len |
|---|---|---|---|---|---|
| **1** | **6.58s** | **39.1ms** | **1332 tok/s** | 76.5% | 0.77 |
| 2 | 7.30s | 40.5ms | 1198 tok/s | 67.0% | 1.34 |
| 3 | 8.29s | 41.7ms | 1047 tok/s | 61.0% | 1.83 |
| 4 | 9.28s | 46.5ms |  933 tok/s | 53.5% | 2.14 |

Larger K does raise `avg_accept_len` (0.77 → 2.14, **so the algorithm itself is healthy**), yet wall,
TPOT and throughput all degrade **monotonically** with K, making K=1 optimal on every metric.
That inversion demands an explanation — below.

#### 5.5 It currently loses to the baseline, and profiling says why

K=1 is still **1.7× slower** than `base_eager` (6.58s vs 3.91s).
`SPEC_PROFILE=1` records CUDA Events across segments of a step (one `torch.cuda.synchronize()` at step
end, zero overhead when off). K=1 steady state (batch 64, ~55ms/step):

| Segment | Time | Share |
|---|---|---|
| draft forward ×1 (fused: backfill KV + propose d1) | 22.6ms | 41% |
| target 8B verify forward ×1 | 29.8ms | 54% |
| logits + softmax + rejection + postprocess | ~2.6ms | 5% |

**Key observation**: the 0.6B draft's single eager forward (~22ms) and the 8B target verify (~30ms) are
both **flat as batch drops from 64 to 1**. The cost is fixed per-forward **kernel-launch + Python-dispatch
overhead**, not GPU matmul — the latter is invisible at this batch scale.

So "the draft model is small, therefore proposing is cheap" is **false**: overhead dominates, and one
draft forward eats roughly 2/3 of a target verify. Speculative decoding's premise
`draft_cost ≪ target_cost` does not hold in eager mode, so no choice of K can win — which explains both
"larger K is slower" in 5.4 and why K=1 is optimal.

**The disproven hypothesis**: the initial guess was that the full-vocab LM head plus FP32 probability
tensors (`[S,K+1,V]`) were expensive. Profiling put that entire path at ~2.6ms / 5%. Profile first,
then optimize.

**Scope note**: this is a *closed* microbatch (no arriving requests to refill). Sequences advance 1–2
tokens per step and desync, so batch occupancy drains to a long tail; real serving with continuous
arrivals would keep batches full.

#### 5.6 Next step

Put **CUDA Graph** on both forwards to amortize the fixed overhead — fixed-shape forwards are exactly
what graph capture eliminates. The difficulty on the target-verify side is that it takes the varlen path
(`max_seqlen_k` is a Python host int that grows with the sequence, so it cannot be captured); it must be
rerouted through `flash_attn_with_kvcache`'s multi-query path (a fixed K+1 queries per sequence) to
become capturable. On the draft side, propose steps 1..K−1 are standard single-token decodes and can
reuse the existing decode-graph pattern directly.

---

Metrics reported per variant: TTFT/TPOT P50/P99, prefill/decode throughput, peak memory, scheduler stats
(prefill/decode steps, preemptions), and prefix-cache stats (hit rate, saved tokens, evictions).

## Deep Dives

Design tradeoffs, edge cases, and implementation details per feature:

- [chunked-prefill.md](docs/chunked-prefill.md) — budget allocation for mixed prefill/decode and varlen-attention reuse
- [prefix-cache-lru.md](docs/prefix-cache-lru.md) — block-hash prefix cache and capacity-bounded LRU eviction
- [cache-aware-scheduling.md](docs/cache-aware-scheduling.md) — LPM scoring, aging against starvation, decoupling from queue depth
- [metrics-and-benchmark.md](docs/metrics-and-benchmark.md) — the metrics-collection substrate and benchmark harness

## Reproducing

Requires Linux x86_64 with CUDA 12 and PyTorch. The scripts in `scripts/` are organized in two stages —
"CPU-only prep → GPU eval" — so GPU billing is confined to the last step:

```bash
bash scripts/setup.sh              # CPU-only: deps + flash-attn + download Qwen3-0.6B + CPU unit tests
bash scripts/download_models.sh    # download the two speculative-decoding models (Qwen3-8B target + Qwen3-0.6B draft)
bash scripts/run_gpu.sh            # GPU: self-check -> smoke -> LPM alignment -> speculative lossless check -> 5-scenario benchmark

python scripts/bench_metrics.py prefix   # or run a single scenario
```

Model paths, mirrors, and related settings are overridable via environment variables defined in `scripts/env.sh`.

## Upstream Acknowledgement

Nano-vLLM-Ext is based on [GeeeekExplorer/nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) and is
independently maintained in this repository. We thank the upstream project and its contributors for their work.
