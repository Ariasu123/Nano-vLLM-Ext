<p align="center">
<img width="300" src="assets/logo.png">
</p>

# Nano-vLLM-Ext

[English](README.md)

Nano-vLLM-Ext 是一个独立维护的 Nano-vLLM 扩展版本；Nano-vLLM 是从零实现的轻量级 vLLM 实现。

## 核心特性

* 🚀 **快速离线推理** - 推理速度可与 vLLM 相当
* 📖 **易读的代码库** - 以约 1,200 行 Python 代码实现的清晰架构
* ⚡ **优化能力集** - 前缀缓存、张量并行、Torch 编译、CUDA Graph 等

## 安装

```bash
pip install git+https://github.com/Ariasu123/Nano-vLLM-Ext.git
```

## 下载模型

如需手动下载模型权重，请使用以下命令：

```bash
huggingface-cli download --resume-download Qwen/Qwen3-0.6B \
  --local-dir ~/huggingface/Qwen3-0.6B/ \
  --local-dir-use-symlinks False
```

## 快速开始

使用方式见 `example.py`。API 与 vLLM 的接口保持一致，`LLM.generate` 方法存在少量差异：

```python
from nanovllm import LLM, SamplingParams

llm = LLM("/YOUR/MODEL/PATH", enforce_eager=True, tensor_parallel_size=1)
sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
prompts = ["Hello, Nano-vLLM."]
outputs = llm.generate(prompts, sampling_params)
outputs[0]["text"]
```

## 基准测试

基准测试实现见 `bench.py`。下列为既有的基准测试结果，不包含下文路线图中的计划功能。

**测试配置：**

- 硬件：RTX 4070 Laptop（8GB）
- 模型：Qwen3-0.6B
- 请求总数：256 个序列
- 输入长度：在 100–1024 个 token 间随机采样
- 输出长度：在 100–1024 个 token 间随机采样

**性能结果：**

| 推理引擎 | 输出 Token 数 | 时间（秒） | 吞吐量（tokens/s） |
| --- | ---: | ---: | ---: |
| vLLM | 133,966 | 98.37 | 1361.84 |
| Nano-vLLM | 133,966 | 93.41 | 1434.13 |

## 路线图

以下功能仍在计划中，尚未可用：

- Draft–Target 投机解码
- Chunked Prefill 与公平调度
- Prefix Cache 增强

## 上游致谢

Nano-vLLM-Ext 基于 [GeeeekExplorer/nano-vllm](https://github.com/GeeeekExplorer/nano-vllm)，并由本仓库独立维护。感谢上游项目及其贡献者的工作。
