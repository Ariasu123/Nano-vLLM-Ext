<p align="center">
<img width="300" src="assets/logo.png">
</p>

# Nano-vLLM-Ext

[简体中文](README.zh-CN.md)

An independently maintained extension of Nano-vLLM, a lightweight vLLM implementation built from scratch.

## Key Features

* 🚀 **Fast offline inference** - Comparable inference speeds to vLLM
* 📖 **Readable codebase** - Clean implementation in ~1,200 lines of Python code
* ⚡ **Optimization Suite** - Prefix caching, Tensor Parallelism, Torch compilation, CUDA graph, and more

## Installation

```bash
pip install git+https://github.com/Ariasu123/Nano-vLLM-Ext.git
```

## Model Download

To download the model weights manually, use the following command:

```bash
huggingface-cli download --resume-download Qwen/Qwen3-0.6B \
  --local-dir ~/huggingface/Qwen3-0.6B/ \
  --local-dir-use-symlinks False
```

## Quick Start

See `example.py` for usage. The API mirrors vLLM's interface with minor differences in the `LLM.generate` method:

```python
from nanovllm import LLM, SamplingParams

llm = LLM("/YOUR/MODEL/PATH", enforce_eager=True, tensor_parallel_size=1)
sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
prompts = ["Hello, Nano-vLLM."]
outputs = llm.generate(prompts, sampling_params)
outputs[0]["text"]
```

## Benchmark

See `bench.py` for the benchmark implementation. The following are existing benchmark results and do not include the planned roadmap features below.

**Test Configuration:**

- Hardware: RTX 4070 Laptop (8GB)
- Model: Qwen3-0.6B
- Total Requests: 256 sequences
- Input Length: Randomly sampled between 100–1024 tokens
- Output Length: Randomly sampled between 100–1024 tokens

**Performance Results:**

| Inference Engine | Output Tokens | Time (s) | Throughput (tokens/s) |
| --- | ---: | ---: | ---: |
| vLLM | 133,966 | 98.37 | 1361.84 |
| Nano-vLLM | 133,966 | 93.41 | 1434.13 |

## Roadmap

The following features are planned and are not yet available:

- Draft–Target speculative decoding
- Chunked Prefill with fair scheduling
- Enhanced Prefix Cache

## Upstream Acknowledgement

Nano-vLLM-Ext is independently maintained and is based on [GeeeekExplorer/nano-vllm](https://github.com/GeeeekExplorer/nano-vllm). We thank the upstream project and its contributors for their work.
