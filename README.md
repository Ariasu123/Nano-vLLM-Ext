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

## Quick Start (Reproduce the Benchmarks)

The author develops on macOS (no NVIDIA GPU), so the benchmarks run on a rented GPU server — the examples below use [AutoDL](https://www.autodl.com/): pick the **official PyTorch image** (CUDA 12) when creating the instance, then `scp` the project over after boot. The models are large, so download them to the data disk `/root/autodl-tmp` (the system disk is usually only ~30GB; the scripts default to the data disk, see below). Any other cloud provider or a local GPU box works the same, as long as the prerequisites below are met.

The evaluation scripts live in `scripts/`, organized as two stages — "CPU-only prep → GPU eval" — so GPU billing is confined to the last step. After copying the whole project to the server, run these from the **repo root**:

```bash
bash scripts/setup.sh              # CPU-only: deps + pip install -e . + flash-attn + download Qwen3-0.6B + CPU unit tests
bash scripts/download_models.sh    # download the two models for speculative decoding (Qwen3-8B target + Qwen3-0.6B draft)
bash scripts/run_gpu.sh            # GPU: self-check -> smoke -> LPM alignment -> speculative lossless check -> 5-scenario benchmark
```

**Prerequisites**: Linux x86_64 with PyTorch on CUDA 12 already installed (e.g. AutoDL's official PyTorch image). The scripts do not install torch; flash-attn is pulled as an official prebuilt wheel (`cu12` + `linux_x86_64`).

**Model location**: defaults to the data disk — `/root/autodl-tmp/models` when the AutoDL data disk `/root/autodl-tmp` is detected, otherwise `~/huggingface`. Override with `MODEL_ROOT` / `TARGET_DIR` / `DRAFT_DIR` (all defined in one place, `scripts/env.sh`). Downloads default to the `hf-mirror` mirror (`HF_ENDPOINT` to override).

**GPU memory**: Qwen3-8B (bf16) weights are ~16GB plus KV-cache headroom. If memory is tight, use a smaller target: `TARGET_REPO=Qwen/Qwen3-4B bash scripts/download_models.sh`, then `SPEC_TARGET=$MODEL_ROOT/Qwen3-4B bash scripts/run_gpu.sh`.

To run features 1–3 only (no speculative decoding), the Qwen3-0.6B from `setup.sh` is enough — skip `download_models.sh`; the speculative steps in `run_gpu.sh` are auto-skipped when the models are missing.

## Benchmark

`scripts/bench_metrics.py` compares the original engine against each enhancement under a
scenario built to exercise that specific feature. Each `(scenario, variant)` runs
in its own subprocess so GPU memory is fully released between runs; workloads use a
fixed seed, so all variants of a scenario see the same request batch. Requires a
CUDA GPU (measured on an RTX 5090, Qwen3-0.6B, `max_model_len=4096`).

```bash
python scripts/bench_metrics.py                 # all scenarios, all variants
python scripts/bench_metrics.py prefix          # one scenario only
python scripts/bench_metrics.py prefix shared   # a single variant (used internally per subprocess)
```

Five scenarios, each isolating one enhancement (original engine vs. enhancement). Each results table is baseline vs. optimized.

> Features 1–3 optimize the scheduling and caching layers; their gains (mixed batching to eliminate decode starvation, prefix-block reuse and eviction, LPM dequeue order) are decoupled from model size, so they are measured on Qwen3-0.6B to isolate the "model compute" variable, iterate faster, and save GPU — the same ratios hold on larger models. Feature 4 (speculative decoding) only makes sense with a large + small model pair (the larger the target, the higher the per-step decode cost, and the greater the throughput gain from committing multiple tokens per target forward), so it uses Qwen3-8B as target and Qwen3-0.6B as draft.

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

### `speculative` — draft/target speculative decoding (Qwen3-8B target + Qwen3-0.6B draft)

Apples-to-apples on the same closed batch of 64 requests. Three variants isolate the speculative algorithm from the CUDA Graph confound: `base` (target only, CUDA Graph), `base_eager` (target only, eager), `spec` (speculative, eager). `base_eager` vs `spec` is the fair comparison — both run eager.

Default **K=1**. A **sync–propose fusion** folds the standalone "backfill the lagging draft KV" forward into propose step 0: one varlen prefill over `[nc, N)` per sequence backfills the draft KV (including the last token e) *and* samples d1 from the last position's logit, cutting draft forwards per step from K+1 to K. The fusion only changes the draft's proposal distribution (prefill vs decode kernel numerics), which the rejection sampler corrects unconditionally, so losslessness is unaffected; the gain is largest at low K (at K=1 every all-accept step saves a forward), measured at 8.01s → 6.58s end-to-end (−18%).

| Metric | `base` (graph) | `base_eager` | `spec` K=1 |
|---|---|---|---|
| Wall time | 2.40s | 3.91s | 6.58s |
| TPOT P50 | 15.3ms | 27ms | 39.1ms |

K-sweep (post-fusion, full batch 64): larger K raises per-request `avg_accept_len` (K=1→4: 0.77→2.14), but each eager draft forward is a fixed ~20ms — the same order as the target's ~30ms — so piling on draft cost outruns the saved target verifies, and wall/TPOT/throughput all degrade monotonically with K. **K=1 is optimal on every metric:**

| K | Wall | TPOT P50 | decode tput | acceptance | avg_accept_len |
|---|---|---|---|---|---|
| 1 | 6.58s | 39.1ms | 1332 tok/s | 76.5% | 0.77 |
| 2 | 7.30s | 40.5ms | 1198 tok/s | 67.0% | 1.34 |
| 3 | 8.29s | 41.7ms | 1047 tok/s | 61.0% | 1.83 |
| 4 | 9.28s | 46.5ms |  933 tok/s | 53.5% | 2.14 |

Correctness is verified independently of speed: draft/target logit alignment holds to `max_prob_diff=1.7e-2` (P6, threshold 0.3); the rejection sampler is lossless — greedy output is bit-identical to the target model alone (P7); and across all K, `reject: draft=0 target=0 fallback=0` (no structural rejections or fallbacks).

**But even post-fusion, K=1 is still 1.7× slower than the eager baseline (6.58s vs 3.91s), and profiling says why.** `SPEC_PROFILE=1` records CUDA Events across segments of the step (one `torch.cuda.synchronize()` at step end, zero overhead when off). K=1 steady state (batch 64, ~55ms/step):

| Segment | Time | Share |
|---|---|---|
| draft forward ×1 (fused: backfill KV + propose d1) | 22.6ms | 41% |
| target 8B verify forward ×1 | 29.8ms | 54% |
| logits + softmax + rejection + postprocess | ~2.6ms | 5% |

The key observation: the 0.6B draft's single eager forward (~22ms) and the 8B target verify (~30ms) are both *flat as batch drops 64→1* — the cost is fixed per-forward kernel-launch + Python-dispatch overhead, not GPU matmul, which is invisible at this batch scale. So "the draft is small, therefore cheap" is false: overhead dominates, and one draft forward costs ~2/3 of the target verify. This is why larger K is slower, and why the default is K=1 (fewest forwards). An early hypothesis that the full-vocab LM head + FP32 probability tensor was expensive is disproven — that path is ~2.6ms.

On this *closed* microbatch (no arriving requests to refill), sequences advance 1–2 tokens per step and desync, so batch occupancy drains to a long tail — real serving with continuous arrivals would keep batches full.

Conclusion: the algorithm and acceptance are correct and healthy, and the fusion removed the per-step standalone sync forward. The remaining bottleneck is the draft's eager forward (pure kernel-launch overhead). The identified next step is a **draft-decode CUDA Graph** — fixed-shape single-token decode is exactly what graph capture eliminates; it would collapse the ~22ms draft forward to a few ms, flipping the K trade-off so that higher-K's extra acceptance finally pays off.

Metrics reported per variant: TTFT/TPOT P50/P99, prefill/decode throughput, peak
memory, scheduler stats (prefill/decode steps, preemptions), and prefix-cache
stats (hit rate, saved tokens, evictions).

## Upstream Acknowledgement

Nano-vLLM-Ext is based on [GeeeekExplorer/nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) and is independently maintained in this repository. We thank the upstream project and its contributors for their work.
