# LLMEngine 是用户 API 与底层 GPU 推理代码之间的“总指挥”。
# 它本身不实现 Transformer 计算，而是负责：
# 1. 把 prompt 转成 Sequence 请求对象；
# 2. 让 Scheduler（调度器）决定本轮运行哪些请求；
# 3. 让 ModelRunner 在 GPU 上执行模型；
# 4. 重复以上过程，直到全部请求完成；
# 5. 把生成的 token id 解码回字符串。

# 【两个最重要的推理阶段】
# Prefill（预填充）：一次处理 prompt 中的多个 token，建立 KV Cache。
# Decode（逐 token 解码）：之后每轮每个请求只输入最新的一个 token，并预测下一个 token。
import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner


class LLMEngine:

    # __init__ 是创建 LLM(...) 时自动调用的初始化函数。
    # model 是本地模型目录；**kwargs 会把其余“名称=值”参数收集成字典。
    # 例如 LLM(path, enforce_eager=True) 中，kwargs 就是 {"enforce_eager": True}。
    def __init__(self, model, **kwargs):
        # fields(Config) 返回 Config 数据类声明的所有字段。
        # 这里先得到合法字段名集合，再过滤 kwargs，防止把无关参数传给 Config。
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)

        # Sequence.block_size 是类属性，即所有 Sequence 对象共享同一个值。
        # 将它设成配置值，可保证 Sequence 的逻辑分块大小与 GPU KV Cache 物理块大小完全一致。
        Sequence.block_size = config.kvcache_block_size

        # ps 保存子进程对象，events 保存主进程通知子进程工作的事件对象。
        self.ps = []
        self.events = []

        # multiprocessing 简写为 mp。spawn 会启动全新的 Python 进程，
        # 不会直接复制当前进程已初始化的 CUDA 状态，因此更适合 CUDA 多进程。
        ctx = mp.get_context("spawn")

        # tensor_parallel_size 表示用几张 GPU 共同运行同一份模型。
        # rank 是每个参与进程的编号：rank 0 使用当前主进程，rank 1、2... 使用子进程。
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        # 主进程也创建一个 ModelRunner，它负责 rank 0 对应的 GPU 分片，并汇总最终 logits。
        self.model_runner = ModelRunner(config, 0, self.events)

        # use_fast=True 优先使用 Rust 实现的快速分词器。
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)

        # EOS 是 End Of Sequence（序列结束）的缩写。
        # 当模型生成这个 token 时，Scheduler 可以把请求标记为完成。
        config.eos = self.tokenizer.eos_token_id

        # 功能四：投机要求 draft/target 共用同一 token 空间。config 已断言 vocab_size 相等，
        # 这里进一步校验 tokenizer 的 token-id 映射与特殊 token 完全一致，否则 p/q 无法对齐、
        # 提交的 token 在两模型含义不同 → 直接报错终止（不做静默兼容）。
        if config.enable_speculative_decode:
            draft_tokenizer = AutoTokenizer.from_pretrained(config.speculative_model, use_fast=True)
            assert self.tokenizer.get_vocab() == draft_tokenizer.get_vocab(), \
                "draft/target tokenizer 词表映射不一致，无法用于投机解码"
            for attr in ("eos_token_id", "bos_token_id", "pad_token_id"):
                assert getattr(self.tokenizer, attr) == getattr(draft_tokenizer, attr), \
                    f"draft/target tokenizer {attr} 不一致，无法用于投机解码"

        self.scheduler = Scheduler(config)

        # atexit.register 表示 Python 进程正常退出时自动调用 self.exit，
        # 以免遗留 GPU 通信组、共享内存或子进程。
        atexit.register(self.exit)

    # 释放 ModelRunner 和张量并行子进程。
    def exit(self):
        # call("exit") 不仅调用 rank 0 的 exit，还会通知其他 rank 同时退出。
        self.model_runner.call("exit")
        del self.model_runner

        # join 会让主进程等待每个子进程真正结束，防止主程序先退出。
        for p in self.ps:
            p.join()

    # 把一个新请求转换成 Sequence 并放入 Scheduler 的等待队列。
    
    # prompt 可以是：
    # - str：普通字符串，例如 "Hello"；
    # - list[int]：已经分好词的 token id，例如 [9707]。
    # `str | list[int]` 是 Python 的联合类型标注，表示两种类型都可以。
    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):
            # isinstance 用来判断运行时类型。字符串需要先转成模型可读的整数列表。
            prompt = self.tokenizer.encode(prompt)

        # Sequence 保存这个请求的 token、生成参数、状态和 KV Cache 块表。
        seq = Sequence(prompt, sampling_params)
        # 记录请求到达时刻（TTFT 起点）；benchmark 需要，正常 generate 不受影响。
        seq.arrival_time = perf_counter()
        self.scheduler.add(seq)
        # 返回 seq，便于 benchmark 保留引用、在结束后读取时间戳。
        return seq

    # 执行“一轮”推理，而不是一次完成所有生成。
    # 返回：
    # - outputs：本轮刚刚完成的请求；
    # - num_tokens：用于进度条计算吞吐量的 token 数。
    def step(self):
        # Scheduler 会返回本轮选中的序列，以及本轮走 varlen(prefill) 还是固定形状(decode) 路径。
        seqs, is_prefill = self.scheduler.schedule()

        # 功能四：投机 step 走独立分派。run_speculative 一次并行 verification 提交 1~K+1 token。
        # 计入 decode 吞吐的 token 数取本步各序列名义提交量（accept_len+1）；截断到 EOS/max_tokens
        # 的极少数末尾差异不影响对照结论，权威口径由 metrics.total_decode_tokens 提供。
        if self.scheduler.pending_kind == "speculative":
            result = self.model_runner.call("run_speculative", seqs)
            self.scheduler.postprocess_speculative(seqs, result)
            decode_tokens = sum(accept_len + 1 for _, accept_len in result) if result else 0
            outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
            return outputs, 0, decode_tokens

        # 按每条序列的 is_prefill 分别统计 prefill / decode token 数：混批步二者并存，
        # 不能再用单一符号区分。decode 序列 num_scheduled_tokens==1，prefill 序列为本步 chunk 大小。
        # 必须在 postprocess 之前统计（postprocess 会把 num_scheduled_tokens 清零）。
        prefill_tokens = sum(seq.num_scheduled_tokens for seq in seqs if seq.is_prefill)
        decode_tokens = sum(seq.num_scheduled_tokens for seq in seqs if not seq.is_prefill)

        # ModelRunner.run 会准备 GPU tensor、执行 Qwen3 并采样下一个 token。
        # call 还负责在多 GPU 时让所有 rank 执行相同操作。
        token_ids = self.model_runner.call("run", seqs, is_prefill)

        # GPU 计算完成后，把新 token、缓存进度和请求完成状态写回 Sequence。
        self.scheduler.postprocess(seqs, token_ids, is_prefill)

        # 列表推导式只收集已经结束的请求。
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        return outputs, prefill_tokens, decode_tokens

    # waiting 和 running 队列都为空时，说明没有未完成请求。
    def is_finished(self):
        return self.scheduler.is_finished()

    # 批量生成接口。
    # prompts 是多个字符串或多个 token id 列表；
    # sampling_params 可以是共用配置，也可以是逐请求配置列表。
    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        # tqdm 是终端进度条。total 是总请求数，而不是总 token 数。
        pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True, disable=not use_tqdm)

        # 如果只传入一个 SamplingParams，就让所有 prompt 共用它。
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)

        # zip 按下标把 prompt 与对应参数配对。
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)

        # 字典暂存已完成结果。请求完成顺序可能与提交顺序不同，因此不能直接 append 到列表。
        outputs = {}
        prefill_throughput = decode_throughput = 0.

        # 每次循环只前进一步：长请求会经历 prefill，再经历很多次 decode。
        while not self.is_finished():
            t = perf_counter()
            output, prefill_tokens, decode_tokens = self.step()
            dt = perf_counter() - t
            # 吞吐量 = 本轮处理 token 数 / 本轮耗时，单位为 token/s。混批步二者都非零，各自更新。
            if prefill_tokens:
                prefill_throughput = prefill_tokens / dt
            if decode_tokens:
                decode_throughput = decode_tokens / dt
            pbar.set_postfix({
                "Prefill": f"{int(prefill_throughput)}tok/s",
                "Decode": f"{int(decode_throughput)}tok/s",
            })
            for seq_id, token_ids in output:
                # 本轮可能完成 0 个、1 个或多个请求。
                outputs[seq_id] = token_ids
                pbar.update(1)
        pbar.close()
        # sorted 把 seq_id 从小到大排序，从而恢复用户提交 prompts 的顺序。
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]

        # tokenizer.decode 把 completion token id 还原成可读文字。
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        return outputs

    # benchmark 专用：跑一批请求并汇总性能指标，不返回解码文本。
    # 指标：TTFT/TPOT 的 P50/P99、prefill/decode 分阶段吞吐、峰值显存，以及调度与 Prefix Cache 统计。
    def benchmark(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = False,
    ) -> dict:
        from nanovllm.engine.metrics import percentile, compute_ttft, compute_tpot

        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)

        # 清零峰值显存统计，保证测到的是本次 benchmark 期间的峰值。
        self.model_runner.call("reset_peak_memory")

        # 保留每个 seq 的引用，结束后从中读取时间戳（seq 完成后仍存活）。
        seqs = [self.add_request(p, sp) for p, sp in zip(prompts, sampling_params)]

        pbar = tqdm(total=len(prompts), desc="Benchmarking", dynamic_ncols=True, disable=not use_tqdm)
        prefill_tokens = decode_tokens = 0
        prefill_time = decode_time = 0.
        t_start = perf_counter()
        while not self.is_finished():
            t = perf_counter()
            output, prefill_tokens_step, decode_tokens_step = self.step()
            dt = perf_counter() - t
            # 混批步 prefill/decode token 并存：dt 同时计入两侧（都略偏保守），非混批步只命中一侧、与原版一致。
            if prefill_tokens_step:
                prefill_tokens += prefill_tokens_step
                prefill_time += dt
            if decode_tokens_step:
                decode_tokens += decode_tokens_step
                decode_time += dt
            pbar.update(len(output))
        pbar.close()
        wall_time = perf_counter() - t_start

        # 逐请求延迟指标。
        ttfts, tpots = [], []
        for seq in seqs:
            if seq.first_token_time is not None:
                ttfts.append(compute_ttft(seq.arrival_time, seq.first_token_time))
                if seq.finish_time is not None:
                    tpot = compute_tpot(seq.first_token_time, seq.finish_time, seq.num_completion_tokens)
                    if tpot > 0:
                        tpots.append(tpot)
        return {
            "num_requests": len(prompts),
            "wall_time_s": wall_time,
            "ttft_p50_s": percentile(ttfts, 50),
            "ttft_p99_s": percentile(ttfts, 99),
            "tpot_p50_s": percentile(tpots, 50),
            "tpot_p99_s": percentile(tpots, 99),
            "prefill_throughput_tok_s": prefill_tokens / prefill_time if prefill_time else 0.,
            "decode_throughput_tok_s": decode_tokens / decode_time if decode_time else 0.,
            "total_prefill_tokens": prefill_tokens,
            "total_decode_tokens": decode_tokens,
            "peak_memory_bytes": self.model_runner.call("get_peak_memory"),
            "scheduler_stats": self.scheduler.get_stats(),
            "prefix_cache_stats": self.scheduler.block_manager.get_stats(),
        }
