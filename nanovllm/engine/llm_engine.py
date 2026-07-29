# 【这个文件做什么】
# LLMEngine 是用户 API 与底层 GPU 推理代码之间的“总指挥”。
# 它本身不实现 Transformer 计算，而是负责：
# 1. 把 prompt 转成 Sequence 请求对象；
# 2. 让 Scheduler（调度器）决定本轮运行哪些请求；
# 3. 让 ModelRunner 在 GPU 上执行模型；
# 4. 重复以上过程，直到全部请求完成；
# 5. 把生成的 token id 解码回字符串。
#
# 【建议的阅读位置】
# 在 example.py 之后阅读本文件。读完 generate() 和 step() 后，再读 sequence.py 和 scheduler.py。
#
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
    #
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
        self.scheduler.add(seq)

    # 执行“一轮”推理，而不是一次完成所有生成。
    # 返回：
    # - outputs：本轮刚刚完成的请求；
    # - num_tokens：用于进度条计算吞吐量的 token 数。
    def step(self):
        # Scheduler 会返回本轮选中的序列，以及本轮属于 prefill 还是 decode。
        seqs, is_prefill = self.scheduler.schedule()

        # Prefill 时可能一次处理很多 prompt token，所以把每条序列的本轮 token 数相加。
        # Decode 时每条序列只有 1 个 token，使用负数只是本项目内部区分两种吞吐量的简写。
        num_tokens = sum(seq.num_scheduled_tokens for seq in seqs) if is_prefill else -len(seqs)

        # ModelRunner.run 会准备 GPU tensor、执行 Qwen3 并采样下一个 token。
        # call 还负责在多 GPU 时让所有 rank 执行相同操作。
        token_ids = self.model_runner.call("run", seqs, is_prefill)

        # GPU 计算完成后，把新 token、缓存进度和请求完成状态写回 Sequence。
        self.scheduler.postprocess(seqs, token_ids, is_prefill)

        # 列表推导式只收集已经结束的请求。
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        return outputs, num_tokens

    # waiting 和 running 队列都为空时，说明没有未完成请求。
    def is_finished(self):
        return self.scheduler.is_finished()

    # 面向用户的批量生成接口。
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
        # 这里不会修改 SamplingParams，因此共享同一个对象是安全的。
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
            output, num_tokens = self.step()
            # 吞吐量 = 本轮处理 token 数 / 本轮耗时，单位为 token/s。
            if num_tokens > 0:
                prefill_throughput = num_tokens / (perf_counter() - t)
            else:
                decode_throughput = -num_tokens / (perf_counter() - t)
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
