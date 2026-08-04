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

输入 Token ID 先经过 Token Embedding 与 RoPE，再穿过重复的 Decoder 层，最后由归一化层和 LM Head 生成下一个 Token 的 logits。每个 Decoder 层内，GQA 使用紧凑的 KV Cache 完成注意力计算，SwiGLU 前馈网络则变换隐藏状态。该图展示了这些组件如何串成一条完整的推理路径。

### 引擎与 KV Cache

![引擎与 KV Cache 架构](assets/Engine.png)

输入请求从 Scheduler 的 waiting 队列进入 running 队列，并被组装为执行批次。BlockManager 通过每条序列的 block table、物理块池和 Prefix Cache 管理 KV Cache 块的分配、复用与释放。Rank 0 负责协调调度和命令，Worker 则携带相应 block table 执行自己的模型分片。

### 张量并行执行

![张量并行执行流程](assets/TP-expanded.png)

Checkpoint 权重被拆分到各个张量并行 Rank，其中列并行层和行并行层共同分担模型计算。Prefill 和 Decode 阶段中，每个 Rank 执行本地分片，并在需要合并部分激活值时执行 `all_reduce`。Rank 0 汇聚词表 logits、采样下一个 Token 并返回结果；Worker Rank 只完成分配给自己的分片计算。


## 上游致谢

Nano-vLLM-Ext 基于 [GeeeekExplorer/nano-vllm](https://github.com/GeeeekExplorer/nano-vllm)，并由本仓库独立维护。感谢上游项目及其贡献者的工作。
