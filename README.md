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

See how the Qwen inference path connects embeddings, attention, KV cache, and the language-model head.

### Engine and KV Cache

![Engine and KV cache architecture](assets/Engine.png)

Learn how request scheduling, KV cache management, and multi-process execution work together.

### Tensor Parallel Execution

![Tensor parallel execution flow](assets/TP-expanded.png)

Follow parameter sharding, collective communication, and sampling across tensor-parallel ranks.

## Upstream Acknowledgement

Nano-vLLM-Ext is based on [GeeeekExplorer/nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) and is independently maintained in this repository. We thank the upstream project and its contributors for their work.
