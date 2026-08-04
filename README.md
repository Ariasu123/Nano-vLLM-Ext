<p align="center">
<img width="300" src="assets/logo.png">
</p>

# Nano-vLLM-Ext

[简体中文](README.zh-CN.md)

Nano-vLLM-Ext is an independently maintained extension of Nano-vLLM.

### Extension Features

- Draft–Target speculative decoding
- Chunked Prefill with fair scheduling
- Enhanced Prefix Cache

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

## Upstream Acknowledgement

Nano-vLLM-Ext is based on [GeeeekExplorer/nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) and is independently maintained in this repository. We thank the upstream project and its contributors for their work.
