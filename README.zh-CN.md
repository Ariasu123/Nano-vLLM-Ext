<p align="center">
<img width="300" src="assets/logo.png">
</p>

# Nano-vLLM-Ext

[English](README.md)

Nano-vLLM-Ext 是一个独立维护的 Nano-vLLM 扩展版本:


### 扩展特性

- Draft–Target 投机解码
- Chunked Prefill 与公平调度
- Prefix Cache 增强

## 架构

### Qwen 推理架构

![Qwen 推理架构](assets/Qwen_arch.png)

了解 Qwen 推理路径如何串联嵌入、注意力、KV Cache 与语言模型头。

### 引擎与 KV Cache

![引擎与 KV Cache 架构](assets/Engine.png)

理解请求调度、KV Cache 管理与多进程执行如何协同工作。

### 张量并行执行

![张量并行执行流程](assets/TP-expanded.png)

跟随参数分片、集合通信与采样在各张量并行 Rank 间的执行流程。


## 上游致谢

Nano-vLLM-Ext 基于 [GeeeekExplorer/nano-vllm](https://github.com/GeeeekExplorer/nano-vllm)，并由本仓库独立维护。感谢上游项目及其贡献者的工作。
