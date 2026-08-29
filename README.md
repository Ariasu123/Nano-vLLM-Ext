<p align="center">
<img width="300" src="assets/logo.png">
</p>

# Nano-vLLM-Ext

[简体中文](README.zh-CN.md)

Nano-vLLM-Ext is an independently maintained extension of Nano-vLLM.

### Extension Features

- Draft–Target speculative decoding
- Chunked Prefill (mixed prefill/decode batching)
- Cache-aware (LPM) scheduling, backed by LRU prefix cache

## Architecture

### Qwen Inference Architecture

![Qwen inference architecture](assets/Qwen_arch.png)

Input token IDs first pass through token embeddings and RoPE, then flow through repeated Decoder layers before the final normalization and LM head produce next-token logits. Within each layer, GQA reuses a compact KV cache for attention, while the SwiGLU feed-forward network transforms the hidden states. The diagram shows how these components form one end-to-end inference path.

### Engine and KV Cache

![Engine and KV cache architecture](assets/Engine.png)

Incoming requests move from the scheduler's waiting queue to the running queue and are assembled into execution batches. BlockManager maintains each sequence's block table, physical block pool, and Prefix Cache so KV-cache blocks can be allocated, reused, or released. Rank 0 coordinates scheduling and commands, while workers execute their model shards with the required block tables.

### Tensor Parallel Execution

![Tensor parallel execution flow](assets/TP-expanded.png)

Checkpoint weights are split across tensor-parallel ranks, with column-parallel and row-parallel layers distributing the model computation. During Prefill and Decode, every rank runs its local shard and uses `all_reduce` where partial activations must be combined. Rank 0 gathers vocabulary logits, samples the next token, and returns the result; worker ranks complete only their assigned shard work.

## Benchmark

`bench_metrics.py` compares the original engine against each enhancement under a
scenario built to exercise that specific feature. Each `(scenario, variant)` runs
in its own subprocess so GPU memory is fully released between runs; workloads use a
fixed seed, so all variants of a scenario see the same request batch. Requires a
CUDA GPU (measured on an RTX 5090, Qwen3-0.6B, `max_model_len=4096`).

```bash
python bench_metrics.py                 # all scenarios, all variants
python bench_metrics.py prefix          # one scenario only
python bench_metrics.py prefix shared   # a single variant (used internally per subprocess)
```

Four scenarios, each isolating one enhancement (original engine vs. enhancement). Each results table is baseline vs. optimized.

### `starvation` — mixed batching eliminates decode starvation

96 short victims (out=2) are submitted first, then 96 long blockers (input 1200–1600) force consecutive prefill. `no_mix` (`enable_chunked_prefill=False`) is the original two-phase scheduler; `mix` (`enable_chunked_prefill=True`) mixes prefill and decode in one step. Under two-phase scheduling a victim cannot decode until every long prefill finishes, so its single inter-token gap spans the whole prefill burst; mixing one decode token per running seq into each step collapses that gap to a single step.

| Victim TPOT | `no_mix` (two-phase) | `mix` | Change |
|---|---|---|---|
| P50 | 552.6ms | 117.2ms | −79% |
| P99 | 822.9ms | 117.6ms | −86% |

### `prefix` — shared-prefix reuse

128 requests each carry a 1024-token system prefix. `no_reuse` gives every request a unique prefix; `shared` gives them one common prefix the cache can reuse.

| Metric | `no_reuse` | `shared` | Change |
|---|---|---|---|
| Hit rate | 0% | 88.3% | — |
| TTFT P50 | 579ms | 241ms | −58% |
| Wall time | 1.36s | 0.54s | −60% |
| Prefill steps | 10 | 3 | — |

### `lru_pressure` — LRU vs FIFO under cache pressure

128 distinct 512-token prefixes, 384 requests with power-law skewed reuse, `gpu_memory_utilization=0.4` to force eviction. `fifo` (`enable_lru=False`) vs `lru`.

| Metric | `fifo` | `lru` | Change |
|---|---|---|---|
| Hit rate | 60.4% | 62.5% | +2.1pp |
| Evictions | 82 | 62 | −24% |

### `cache_aware` — LPM scheduling under prefix-eviction pressure

64 distinct 1024-token prefixes, 512 requests arriving round-robin (adjacent requests hit different prefixes), `gpu_memory_utilization=0.15` so the prefix working set exceeds cache capacity. `fifo` (`enable_cache_aware_schedule=False`) serves the waiting queue in arrival order; `lpm` (`enable_cache_aware_schedule=True`) prioritizes the request whose cached prefix is longest, clustering same-prefix requests so their blocks are reused before eviction. Both variants keep `enable_lru=True`; only the scheduling order differs.

| Metric | `fifo` | `lpm` | Change |
|---|---|---|---|
| Hit rate | 0% | 72.9% | — |
| Evictions | 1968 | 470 | −76% |
| TTFT P50 | 2894ms | 1773ms | −39% |
| Wall time | 5.53s | 3.82s | −31% |

Tradeoff: LPM drains prefill faster so more sequences decode concurrently, raising TPOT P50 from 5.1ms to 8.4ms; the gain only appears when the prefix working set exceeds cache capacity, so it ships as a default-off flag.

Metrics reported per variant: TTFT/TPOT P50/P99, prefill/decode throughput, peak
memory, scheduler stats (prefill/decode steps, preemptions), and prefix-cache
stats (hit rate, saved tokens, evictions).

## Upstream Acknowledgement

Nano-vLLM-Ext is based on [GeeeekExplorer/nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) and is independently maintained in this repository. We thank the upstream project and its contributors for their work.
