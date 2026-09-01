# ModelRunner 是真正驱动 GPU 的执行器。Scheduler 只说“本轮运行哪些 Sequence”，
# ModelRunner 则把 Python 列表整理成 PyTorch Tensor，执行 Qwen3，并返回新 token。

# 【主要工作】
# 1. 每张 GPU 创建模型分片并加载相应权重；
# 2. 多 GPU 时让所有进程执行相同命令；
# 3. 预热模型并用剩余显存分配 KV Cache；
# 4. 分别准备 Prefill 和 Decode 的输入；
# 5. 选择普通 Eager 执行或 CUDA Graph 回放。

# 【Tensor 与 rank】
# Tensor（张量）可理解为支持 GPU 运算的多维数组。shape=[2,3] 就是 2 行 3 列。
# 多 GPU 时，每个参与者有一个从 0 开始的编号，叫 rank；参与者总数叫 world_size。

import os
import pickle
import torch
import torch.distributed as dist
from contextlib import contextmanager
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.layers.sampler import Sampler
from nanovllm.layers.rejection_sampler import rejection_sample
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.loader import load_model


# ---------- 功能四 profiling：spec step 分段 CUDA Event 计时（SPEC_PROFILE=1 时启用）----------
# 诊断用，只在 rank0 创建。用 CUDA Event 记录各段 start/end，全程不 synchronize（避免探针本身
# 改变执行序），到 spec step 末尾统一 synchronize 一次再读各段 elapsed_time。关闭时根本不创建
# 事件，正常 benchmark 零开销、走原路径。
class _SpecProfiler:
    def __init__(self):
        self.sections = {}      # name -> [(start_event, end_event), ...]（可重复段如 draft 每步累积多对）
        self.counters = {}      # name -> int（synced_seqs / synced_tokens 等标量）
        self._open = {}         # name -> start_event（begin/stop 跨语句配对）

    def begin(self, name):
        e = torch.cuda.Event(enable_timing=True)
        e.record()
        self._open[name] = e

    def stop(self, name):
        e = torch.cuda.Event(enable_timing=True)
        e.record()
        self.sections.setdefault(name, []).append((self._open.pop(name), e))

    @contextmanager
    def section(self, name):
        self.begin(name)
        try:
            yield
        finally:
            self.stop(name)

    def set(self, name, value):
        self.counters[name] = value

    def summarize(self):
        # 唯一一次同步：此时 GPU 时间线已 join，安全读取所有 elapsed_time（毫秒）。
        torch.cuda.synchronize()
        sums, per_call = {}, {}
        for name, pairs in self.sections.items():
            per = [s.elapsed_time(e) for s, e in pairs]
            sums[name] = sum(per)
            per_call[name] = per
        return sums, per_call, self.counters


class ModelRunner:

    # config 是全局配置；rank 是当前进程/GPU 编号；
    # event 用于让 rank 0 通知其他 rank“共享内存中有新命令”。
    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        # hf_config 来自模型目录，包含层数、隐藏维度、头数和 dtype 等结构信息。
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event

        # 建立多 GPU 通信组。NCCL 是 NVIDIA 提供的 GPU 通信库；
        # tcp 地址用于各进程互相发现并建立连接。
        dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)

        # 当前 rank 使用编号相同的 GPU，例如 rank 1 使用 cuda:1。
        torch.cuda.set_device(rank)

        # 保存默认 dtype，初始化结束后再恢复，避免影响用户的其他 PyTorch 代码。
        default_dtype = torch.get_default_dtype()

        # 模型参数通常使用 float16 或 bfloat16。
        # 设置默认 device 后，未显式指定 device 的 Tensor 会直接创建在 GPU 上。
        torch.set_default_dtype(hf_config.dtype)
        torch.set_default_device("cuda")

        # 这里只创建 Qwen3 网络结构，随后 load_model 才填入真实权重。
        self.model = Qwen3ForCausalLM(hf_config)
        load_model(self.model, config.model)
        self.sampler = Sampler()

        # ---------- 功能四：Speculative Decoding ----------
        # 同一 ModelRunner 内嵌 draft 子模型，复用同一 TP 进程组与命令通道
        # （不能起第二个 ModelRunner：会重复 init_process_group + 抢占同名 SharedMemory）。
        self.enable_speculative_decode = config.enable_speculative_decode
        self.num_spec_tokens = config.num_speculative_tokens
        self.draft_model = None
        self.draft_kv_cache = None
        # SPEC_PROFILE=1：开启 spec step 分段计时（仅诊断，仅 rank0 打印）。_spec_prof 由 run_speculative
        # 按步创建；_draft_propose 直调时（如 spec_align_check）恒为 None → 零开销。
        self.spec_profile = os.environ.get("SPEC_PROFILE") == "1"
        self._spec_prof = None
        if self.enable_speculative_decode:
            self.draft_model = Qwen3ForCausalLM(config.draft_hf_config)
            load_model(self.draft_model, config.speculative_model)

        # 顺序不能随意调整：先预热得到模型前向的峰值显存，
        # 再用余下显存分配 KV Cache，否则正式运行可能显存溢出（OOM）。
        self.warmup_model()
        self.allocate_kv_cache()
        if not self.enforce_eager:
            self.capture_cudagraph()

        # 初始化结束，恢复 CPU 与原默认 dtype。
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        if self.world_size > 1:
            if rank == 0:
                # rank 0 创建 1 MiB 共享内存。
                # 只传方法名和 Sequence 元数据，不传模型权重或 GPU Tensor。
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)

                # barrier 是集合点：所有 rank 到达后才能继续。
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name="nanovllm")

                # 非零 rank 初始化完后一直停留在命令循环中。
                self.loop()

    # 释放共享内存、CUDA Graph 和 GPU 通信资源。
    def exit(self):
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                # close 关闭当前句柄，unlink 才真正删除操作系统中的共享内存对象。
                self.shm.unlink()
        if not self.enforce_eager:
            del self.graphs, self.graph_pool
        # 等待 GPU 异步任务完成后再销毁通信组。
        torch.cuda.synchronize()
        dist.destroy_process_group()

    # 非零 rank 的常驻命令循环。
    def loop(self):
        while True:
            method_name, args = self.read_shm()

            # *args 会展开列表。例如 args=[seqs, True] 会变成 call("run", seqs, True)。
            self.call(method_name, *args)
            if method_name == "exit":
                break

    # 等待 rank 0 写入命令，并从共享内存还原 Python 对象。
    def read_shm(self):
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()
        # 前 4 字节保存后续 pickle 数据的长度 n。
        n = int.from_bytes(self.shm.buf[0:4], "little")

        # pickle.loads 把字节还原成 Python 列表。
        # 赋值左侧的 *args 收集第一个元素之后的所有元素。
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()
        return method_name, args

    # rank 0 将一次方法调用写入共享内存，并唤醒所有子 rank。
    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and self.rank == 0

        # pickle.dumps 将 Python 对象转换成字节。
        data = pickle.dumps([method_name, *args])
        n = len(data)
        # 先写 4 字节长度，再写数据正文。
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        for event in self.event:
            event.set()

    # 统一的方法调用入口。
    def call(self, method_name, *args):
        # rank 0 先通知其他 rank，再在本进程执行同名方法。
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        # getattr(self, "run") 等价于 self.run，但允许方法名在运行时决定。
        method = getattr(self, method_name, None)
        return method(*args)

    # benchmark 用：返回本进程 GPU 已分配显存的历史峰值（字节）。
    # 经 call("get_peak_memory") 调用时取 rank 0 的返回值即可代表整体规模。
    def get_peak_memory(self) -> int:
        return torch.cuda.memory_stats()["allocated_bytes.all.peak"]

    # benchmark 用：清零峰值显存统计，确保测量的是本次 benchmark 期间的峰值。
    def reset_peak_memory(self):
        torch.cuda.reset_peak_memory_stats()

    # 用接近配置上限的虚拟输入执行一次 Prefill。
    # 这不是为了得到有意义的输出，而是为了触发内核编译、显存分配并记录峰值。
    def warmup_model(self):
        # empty_cache 释放 PyTorch 缓存分配器中当前没有使用的显存。
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        # seq_len 不能超过批 token 上限或模型最大上下文长度。
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        seq_len = min(max_num_batched_tokens, max_model_len)

        # `//` 是整数除法；请求数还受到 max_num_seqs 限制。
        num_seqs = min(max_num_batched_tokens // seq_len, self.config.max_num_seqs)

        # [0] * seq_len 创建虚拟 token；`_` 表示循环变量不会被使用。
        seqs = [Sequence([0] * seq_len) for _ in range(num_seqs)]
        for seq in seqs:
            seq.num_scheduled_tokens = seq_len
        self.run(seqs, True)
        # draft 也要预热：触发其内核编译并把它的前向峰值计入下方 KV 预算。
        # 此时 draft KV 未分配（k_cache.numel()==0 → store 跳过），与 target 预热同理。
        if self.draft_model is not None:
            # 必须与 target 预热同处 inference_mode：否则直调 draft forward 时梯度仍开启，
            # layernorm 上的 @torch.compile 会走 AOTAutograd 编前向+反向联合图，RMSNorm 里的
            # inplace 会撞 version counter 报 "modified by an inplace operation"。
            with torch.inference_mode():
                input_ids, positions = self.prepare_prefill(seqs)
                self.draft_model.compute_logits(self.draft_model(input_ids, positions))
            reset_context()
        torch.cuda.empty_cache()

    # 根据预热后的可用显存，一次性创建所有层的 KV Cache。
    def allocate_kv_cache(self):
        config = self.config
        hf_config = config.hf_config
        # free 和 total 是当前空闲显存与总显存，单位为字节。
        free, total = torch.cuda.mem_get_info()
        used = total - free

        # peak 是预热期间 PyTorch 同时占用过的最大值；current 是现在仍在使用的值。
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        # KV 头按 GPU 数量平分，这里只计算当前 rank 持有的头数。
        num_kv_heads = hf_config.num_key_value_heads // self.world_size

        # 某些配置直接给 head_dim；没有时用隐藏维度除以注意力头数。
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)

        # 一个物理块的字节数：
        # 2(K/V) * 层数 * 每块 token 数 * 本地 KV 头数 * 每头维度 * 单元素字节数。
        block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * hf_config.dtype.itemsize

        # KV Cache 总字节预算 = 总显存预算 - 当前已用显存 - 正式前向仍要预留的临时峰值。
        budget = int(total * config.gpu_memory_utilization - used - peak + current)

        # 功能四：开启投机时，先切一块给 draft KV，其余给 target。
        # 注意 Qwen3 draft/target 共用 8 个 KV head × head_dim 128，只差层数（0.6B≈28 vs 8B≈36），
        # 故 draft 单块字节 ≈ target 的 ~0.78×（并非小一个量级）。固定 15% 只够 ~51 条并发（诊断实测
        # reject 全部来自 draft 池），提到 22% 让 draft 覆盖满批 64；target 侧仍有余量（reject_target=0）。
        if self.draft_model is not None:
            dhf = config.draft_hf_config
            d_num_kv_heads = dhf.num_key_value_heads // self.world_size
            d_head_dim = getattr(dhf, "head_dim", dhf.hidden_size // dhf.num_attention_heads)
            draft_block_bytes = 2 * dhf.num_hidden_layers * self.block_size * d_num_kv_heads * d_head_dim * dhf.dtype.itemsize
            draft_budget = int(budget * 0.22)
            config.num_draft_kvcache_blocks = draft_budget // draft_block_bytes
            assert config.num_draft_kvcache_blocks > 0
            budget -= draft_budget

        config.num_kvcache_blocks = budget // block_bytes
        assert config.num_kvcache_blocks > 0

        # 最终 Tensor 形状：
        # [2(K/V), 层数, 物理块数, 块内 token 数, 本地 KV 头数, head_dim]。
        self.kv_cache = torch.empty(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, head_dim)
        layer_id = 0

        # modules() 递归遍历所有子模块；只有 Attention 有 k_cache/v_cache。
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                # Tensor 切片是指向大缓存的视图，不会复制整层数据。
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

        # draft KV Cache：独立张量，挂到 draft_model 各 Attention 层。
        if self.draft_model is not None:
            self.draft_kv_cache = torch.empty(2, dhf.num_hidden_layers, config.num_draft_kvcache_blocks, self.block_size, d_num_kv_heads, d_head_dim)
            layer_id = 0
            for module in self.draft_model.modules():
                if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                    module.k_cache = self.draft_kv_cache[0, layer_id]
                    module.v_cache = self.draft_kv_cache[1, layer_id]
                    layer_id += 1

    # 将不同长度的 Python block_table 补齐为二维 GPU Tensor。
    # use_draft=True 时取 seq.draft_block_table（投机路径 draft 侧专用）。
    def prepare_block_tables(self, seqs: list[Sequence], use_draft: bool = False):
        tables = [seq.draft_block_table if use_draft else seq.block_table for seq in seqs]
        # 例：[[7,3], [5]] 会补成 [[7,3], [5,-1]]。
        max_len = max(len(t) for t in tables)
        block_tables = [t + [-1] * (max_len - len(t)) for t in tables]

        # pin_memory 使用锁页 CPU 内存，使 CPU -> GPU 复制更快；
        # non_blocking=True 允许复制与其他 GPU 工作重叠。
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    # 为 Prefill 准备输入。
    # 多条请求不补 padding，而是首尾拼接；cu_seqlens 记录各请求边界。
    def prepare_prefill(self, seqs: list[Sequence]):
        # 例：两条片段长度为 3、2，拼接后 input_ids=[a,b,c,d,e]，
        # cu_seqlens_q=[0,3,5] 表示第一条取 [0:3]、第二条取 [3:5]。
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None
        for seq in seqs:
            # 已缓存前缀不再作为 Query 输入，但仍是 Attention 可见的 Key/Value。
            start = seq.num_cached_tokens
            seqlen_q = seq.num_scheduled_tokens
            end = start + seqlen_q

            # 本轮 Query 可关注位置 0 到 end-1，所以 Key/Value 总长度是 end。
            seqlen_k = end

            # extend 把切片中的多个 token 逐个加入总列表。
            input_ids.extend(seq[start:end])

            # positions 是 token 在原请求中的绝对位置，供 RoPE 使用。
            positions.extend(range(start, end))

            # cu 表示 cumulative（累计）；每项都在前一项上加当前长度。
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            # 预热 Sequence 没有真实块表，因此不写 KV Cache。
            if not seq.block_table:
                continue

            # 求出本轮 token 覆盖的逻辑块范围。
            start_block = start // self.block_size
            end_block = (end + self.block_size - 1) // self.block_size
            for i in range(start_block, end_block):
                # 扁平槽位 = 物理块 id * block_size + 块内偏移。
                # 例：block_size=4、物理块 7 的槽位是 28、29、30、31。
                slot_start = seq.block_table[i] * self.block_size
                if i == start_block:
                    # 本轮可能从一个已有部分缓存的块中间开始。
                    slot_start += start % self.block_size
                if i != end_block - 1:
                    slot_end = seq.block_table[i] * self.block_size + self.block_size
                else:
                    # 最后一个块可能未写满，只计算到 end。
                    slot_end = seq.block_table[i] * self.block_size + end - i * self.block_size
                slot_mapping.extend(range(slot_start, slot_end))

        # 总 K 长度大于总 Q 长度，说明存在缓存前缀，需要块表定位旧 K/V。
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:
            block_tables = self.prepare_block_tables(seqs)

        # 将普通 Python 列表变为 GPU Tensor。
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        # 所有 Attention 层通过全局 Context 读取同一批元数据。
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables)
        return input_ids, positions

    # 为 Decode 准备输入。
    # 每条请求只输入 last_token，更早 token 的 Key/Value 已经在缓存中。
    def prepare_decode(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        for seq in seqs:
            # 模型在位置 t 输入 token[t]，该位置的 logits 用来预测 token[t+1]。
            input_ids.append(seq.last_token)
            positions.append(len(seq) - 1)

            # context_lens 包含当前 last_token，表示 Attention 可见的完整上下文长度。
            context_lens.append(len(seq))

            # 例：物理块 7、block_size=4、last_token 是块内第 2 个，
            # 则写入槽位为 7*4+(2-1)=29。
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens  - 1)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)

        # Decode 不需要 cu_seqlens_q/k，因此只设置实际需要的字段。
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        return input_ids, positions

    # 将每条请求的 temperature 转成 GPU Tensor。
    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = [seq.temperature for seq in seqs]
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    # @torch.inference_mode() 是装饰器：进入函数时关闭梯度记录。
    # 推理不需要反向传播，关闭梯度可减少显存和运行开销。
    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        # Eager 模式表示 PyTorch 每次按 Python 代码逐个提交 GPU 操作。
        # Prefill 形状变化很大，不适合这里的固定形状 CUDA Graph。
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            return self.model.compute_logits(self.model(input_ids, positions))
        else:
            # CUDA Graph 会提前记录一串 GPU 操作，回放时减少 CPU 提交开销。
            # 它要求 Tensor 地址和形状固定，所以要把本轮数据复制到捕获时的缓冲区。
            bs = input_ids.size(0)
            context = get_context()

            # next 找到第一个不小于真实 batch size 的图。
            # 例：bs=10 时使用 bs=16 的图，其余 6 行当作无效填充。
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars

            # 把真实数据复制到固定地址。
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions

            # 名字末尾的下划线表示原地修改；fill_(-1) 将全部槽先标成无效。
            graph_vars["slot_mapping"].fill_(-1)
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"].zero_()
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables

            # replay 直接重放捕获好的 GPU 操作。
            graph.replay()
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    # 执行一次“准备输入 -> 模型前向 -> 采样”。
    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        # `A if 条件 else B` 是 Python 条件表达式。
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)

        # 只有 rank 0 会收集完整词表 logits，因此只有它需要采样温度。
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        logits = self.run_model(input_ids, positions, is_prefill)

        # logits 是模型对词表每个 token 给出的未归一化分数，
        # 形状为 [本轮请求数, 词表大小]。
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None

        # 清空 Context，防止下一批误用旧块表。
        reset_context()
        return token_ids

    # ---------- 功能四：Speculative Decoding 的 GPU 执行 ----------

    # 把 rank0 的一维 int64 token 列表广播给所有 rank（长度未知，先播长度再播内容）。
    # 仅在 world_size>1 时走 NCCL；单卡直接建张量返回。历史 token 只有 rank0（真实
    # Sequence）持有，ranks>0 的 pickle 副本不带 token_ids，故 draft 同步区间需广播。
    def _broadcast_ids(self, ids_list):
        if self.world_size == 1:
            return torch.tensor(ids_list, dtype=torch.int64, device="cuda")
        if self.rank == 0:
            n = torch.tensor([len(ids_list)], dtype=torch.int64, device="cuda")
        else:
            n = torch.zeros(1, dtype=torch.int64, device="cuda")
        dist.broadcast(n, 0)
        length = int(n.item())
        if self.rank == 0:
            t = torch.tensor(ids_list, dtype=torch.int64, device="cuda")
        else:
            t = torch.zeros(length, dtype=torch.int64, device="cuda")
        dist.broadcast(t, 0)
        return t

    # 给定全局 token 索引与目标 block_table，算扁平 KV 槽位。
    def _slot(self, block_table, idx):
        return block_table[idx // self.block_size] * self.block_size + idx % self.block_size

    # 连续 K 次 draft 前向，逐步 propose d1..dK。第 0 步是「融合步」：用一次 varlen prefill
    # 覆盖每序列 [num_draft_cached_tokens, N)，一次同时完成 (1) 补写落后的 draft KV（含 e=token[N-1]）、
    # (2) 取末位（N-1）logit 采样 d1 —— 省去原先独立的 draft-sync 前向，每步 draft 前向从 K+1 降到 K。
    # 落后区间统一在此写入（首次进入 / fallback / preempt 重算 / all-accept 后补 dK）；稳态 partial-accept
    # 下 nc==N-1，第 0 步退化为单 token varlen，等价原 decode，无损无罚。写入的 draft KV 位置集合
    # （nc..N+K-2）与融合前完全一致，故 scheduler 计数与 lossless 语义不变。块表由 may_append_spec
    # 预留（覆盖同步区+propose 区），这里只写不分配。步>=1 为普通 draft decode（输入上一步 d_i）。
    # 每步 rank0 采样后 dist.broadcast，保证各 rank draft KV 与后续 verification 输入一致（陷阱4）。
    # 返回 draft_tokens[S,K]（全 rank 一致）；draft_probs[S,K,V] 仅 rank0，其余 None。
    @torch.inference_mode()
    def _draft_propose(self, seqs: list[Sequence]):
        S, K, V = len(seqs), self.num_spec_tokens, self.config.hf_config.vocab_size
        N = [seq.num_tokens for seq in seqs]
        draft_tokens = torch.empty(S, K, dtype=torch.int64, device="cuda")
        if self.rank == 0:
            temps = self.prepare_sample(seqs)                      # [S]
            greedy = temps == 0
            safe_temps = torch.where(greedy, torch.ones_like(temps), temps)
            draft_probs = torch.empty(S, K, V, device="cuda")
        prof = self._spec_prof                                  # 非 None ⟹ rank0（profiler 仅在 rank0 创建）
        cur = None                                              # 步>=1 输入上一步 d_i；第 0 步走 varlen 分支
        for i in range(K):
            if i == 0:
                # 融合步：每序列 [nc, N) 的 varlen prefill —— 补写落后 draft KV（含 e），取末位 logit 采样 d1。
                positions_l, cu_q, cu_k, slot = [], [0], [0], []
                max_q = max_k = 0
                ids_list = [] if self.rank == 0 else None
                backfill = 0
                for seq, n in zip(seqs, N):
                    nc = seq.num_draft_cached_tokens
                    if self.rank == 0:
                        ids_list.extend(seq.token_ids[nc:n])       # [nc, n)，含 e=token[n-1]
                    positions_l.extend(range(nc, n))
                    cu_q.append(cu_q[-1] + (n - nc))
                    cu_k.append(cu_k[-1] + n)                      # 可见 keys 0..n-1
                    max_q = max(max_q, n - nc)
                    max_k = max(max_k, n)
                    for idx in range(nc, n):
                        slot.append(self._slot(seq.draft_block_table, idx))
                    backfill += n - 1 - nc                         # e 之前的补写量（诊断用）
                if prof is not None:
                    prof.set("synced_seqs", sum(1 for seq, n in zip(seqs, N) if seq.num_draft_cached_tokens < n - 1))
                    prof.set("synced_tokens", backfill)
                input_ids = self._broadcast_ids(ids_list)
                positions = torch.tensor(positions_l, dtype=torch.int64, device="cuda")
                set_context(True,
                            torch.tensor(cu_q, dtype=torch.int32, device="cuda"),
                            torch.tensor(cu_k, dtype=torch.int32, device="cuda"),
                            max_q, max_k,
                            torch.tensor(slot, dtype=torch.int32, device="cuda"),
                            None, self.prepare_block_tables(seqs, use_draft=True))
                for seq, n in zip(seqs, N):
                    seq.num_draft_cached_tokens = n - 1            # 与原 draft-sync 后状态一致（postprocess 再定稿）
            else:
                positions = torch.tensor([n - 1 + i for n in N], dtype=torch.int64, device="cuda")
                slot = torch.tensor([self._slot(seq.draft_block_table, n - 1 + i) for seq, n in zip(seqs, N)], dtype=torch.int32, device="cuda")
                context_lens = torch.tensor([n + i for n in N], dtype=torch.int32, device="cuda")
                set_context(False, slot_mapping=slot, context_lens=context_lens, block_tables=self.prepare_block_tables(seqs, use_draft=True))
                input_ids = cur
            if prof is not None:
                # 拆分：draft_forward = transformer 主干；draft_logits_sample = LM head + softmax + 采样
                # （大词表 LM head/softmax 是否为大头，看这两段的比例即知）。逐 step 各记一对事件；
                # 第 0 步的 forward 同时含 backfill，其耗时计入 draft_forward[0]（sync 已融合进来）。
                with prof.section("draft_forward"):
                    hidden = self.draft_model(input_ids, positions)
                with prof.section("draft_logits_sample"):
                    logits = self.draft_model.compute_logits(hidden)            # [S,V]
                    probs = self.sampler.compute_probs(logits, safe_temps)      # [S,V]
                    d = torch.where(greedy, probs.argmax(dim=-1), self.sampler.sample_from_probs(probs))
                    draft_probs[:, i] = probs
                reset_context()
            else:
                logits = self.draft_model.compute_logits(self.draft_model(input_ids, positions))   # [S,V]
                reset_context()
                if self.rank == 0:
                    probs = self.sampler.compute_probs(logits, safe_temps)      # [S,V]
                    d = torch.where(greedy, probs.argmax(dim=-1), self.sampler.sample_from_probs(probs))
                    draft_probs[:, i] = probs
                else:
                    d = torch.empty(S, dtype=torch.int64, device="cuda")
            if self.world_size > 1:
                dist.broadcast(d, 0)
            draft_tokens[:, i] = d
            cur = d
        return draft_tokens, (draft_probs if self.rank == 0 else None)

    # 构造 verification 的一次 target varlen forward 输入：
    # 每序列 input=[e,d1..dK]（K+1 token），positions=N-1..N+K-1，写真实槽 N-1..N+K-1，
    # cu_seqlens_k=N+K（可见全部前缀），return_all_logits=True 让 LM head 返回每个位置 logits。
    # causal=True 的 varlen 在 q_len<k_len 时按 bottom-right 对齐，正是「e 看 0..N-1、dj 看 0..N-1+j」。
    def prepare_verify(self, seqs: list[Sequence], draft_tokens: torch.Tensor):
        K = self.num_spec_tokens
        dt = draft_tokens.tolist()
        input_ids, positions, cu_q, cu_k, slot = [], [], [0], [0], []
        max_q = max_k = 0
        for seq, drow in zip(seqs, dt):
            N = seq.num_tokens
            toks = [seq.last_token] + drow                          # e,d1..dK
            input_ids.extend(toks)
            positions.extend(range(N - 1, N - 1 + len(toks)))       # N-1..N+K-1
            cu_q.append(cu_q[-1] + len(toks))
            seqlen_k = N + K                                         # keys 0..N+K-1
            cu_k.append(cu_k[-1] + seqlen_k)
            max_q = max(max_q, len(toks))
            max_k = max(max_k, seqlen_k)
            for j in range(len(toks)):
                slot.append(self._slot(seq.block_table, N - 1 + j))
        input_ids = torch.tensor(input_ids, dtype=torch.int64, device="cuda")
        positions = torch.tensor(positions, dtype=torch.int64, device="cuda")
        cu_q = torch.tensor(cu_q, dtype=torch.int32, device="cuda")
        cu_k = torch.tensor(cu_k, dtype=torch.int32, device="cuda")
        slot = torch.tensor(slot, dtype=torch.int32, device="cuda")
        block_tables = self.prepare_block_tables(seqs)              # target 块表
        set_context(True, cu_q, cu_k, max_q, max_k, slot, None, block_tables, return_all_logits=True)
        return input_ids, positions

    # 一个完整 spec step（全 rank 执行；仅 rank0 返回提交结果）。
    # 返回 [(committed_token_ids, accept_len), ...]，交给 scheduler.postprocess_speculative。
    @torch.inference_mode()
    def run_speculative(self, seqs: list[Sequence]):
        K = self.num_spec_tokens
        # SPEC_PROFILE=1 且 rank0 时按步创建 profiler；否则 prof=None → 下方全走原路径、零开销。
        prof = _SpecProfiler() if (self.spec_profile and self.rank == 0) else None
        self._spec_prof = prof
        try:
            if prof is None:
                draft_tokens, draft_probs = self._draft_propose(seqs)      # 第 0 步已融合补写 draft KV
                input_ids, positions = self.prepare_verify(seqs, draft_tokens)
                logits = self.model.compute_logits(self.model(input_ids, positions))   # rank0: [S*(K+1),V]
                reset_context()
                if self.rank != 0:
                    return None
                S, V = len(seqs), logits.size(-1)
                temps = self.prepare_sample(seqs)                          # [S]
                greedy = temps == 0
                safe_temps = torch.where(greedy, torch.ones_like(temps), temps)
                # 每序列 K+1 行共享该序列温度；顺序与 prepare_verify 拼接一致。
                row_temps = safe_temps.repeat_interleave(K + 1)
                target_probs = self.sampler.compute_probs(logits, row_temps).view(S, K + 1, V)
                accept_len, out_tokens = rejection_sample(target_probs, draft_probs, draft_tokens, greedy=greedy)
                accept_len = accept_len.tolist()
                out = out_tokens.tolist()
                return [(out[s][: accept_len[s] + 1], accept_len[s]) for s in range(S)]

            # ---- profiling 路径（仅 rank0，逐段 CUDA Event，末尾统一 synchronize）----
            prof.begin("step_total")
            draft_tokens, draft_probs = self._draft_propose(seqs)      # 第 0 步融合补写 draft KV；内部记 draft_forward / draft_logits_sample / synced_*
            input_ids, positions = self.prepare_verify(seqs, draft_tokens)
            with prof.section("target_verify_forward"):
                hidden = self.model(input_ids, positions)
            with prof.section("target_logits_probs"):                  # 第 1 段：verification 的 LM head
                logits = self.model.compute_logits(hidden)             # rank0: [S*(K+1),V]
            reset_context()
            S, V = len(seqs), logits.size(-1)
            temps = self.prepare_sample(seqs)                          # [S]
            greedy = temps == 0
            safe_temps = torch.where(greedy, torch.ones_like(temps), temps)
            row_temps = safe_temps.repeat_interleave(K + 1)
            with prof.section("target_logits_probs"):                  # 第 2 段：大词表 softmax（与上段累加）
                target_probs = self.sampler.compute_probs(logits, row_temps).view(S, K + 1, V)
            with prof.section("rejection_sampler"):
                accept_len, out_tokens = rejection_sample(target_probs, draft_probs, draft_tokens, greedy=greedy)
            with prof.section("postprocess"):                          # tolist 的 D2H 同步也计入本段
                accept_len = accept_len.tolist()
                out = out_tokens.tolist()
                result = [(out[s][: accept_len[s] + 1], accept_len[s]) for s in range(S)]
            prof.stop("step_total")
            self._report_spec_profile(prof, S, K, accept_len)
            return result
        finally:
            self._spec_prof = None

    # 打印一行本 spec step 的分段耗时（毫秒）+ batch_size/proposed/accepted/emitted + cost/token。
    # emitted 用 accepted+batch（每序列提交 accept_len+1；stop 截断在 scheduler，未计入此处的成本口径）。
    def _report_spec_profile(self, prof, S, K, accept_len):
        sums, per_call, cnt = prof.summarize()
        accepted = sum(accept_len)
        emitted = accepted + S
        total = sums.get("step_total", 0.0)
        cost = total / emitted if emitted else 0.0
        draft_steps = [round(x, 2) for x in per_call.get("draft_forward", [])]
        g = lambda k: sums.get(k, 0.0)
        print(f"[SPEC_PROFILE] bs={S} prop={S * K} acc={accepted} emit={emitted} | "
              f"total={total:.1f} "
              f"backfill(seq={cnt.get('synced_seqs', 0)},tok={cnt.get('synced_tokens', 0)},fused_into_draft_fwd[0]) "
              f"draft_fwd={g('draft_forward'):.1f} draft_smp={g('draft_logits_sample'):.1f} "
              f"tgt_fwd={g('target_verify_forward'):.1f} tgt_prob={g('target_logits_probs'):.1f} "
              f"rej={g('rejection_sampler'):.1f} post={g('postprocess'):.1f} | "
              f"cost/tok={cost:.3f}ms draft_fwd_steps={draft_steps}", flush=True)

    # ---------- 功能四 P6：verification 张量/KV 对齐 debug guard（需 GPU，仅 TP=1 诊断）----------
    # 把「端到端 lossless」拆出一个中间层：单独验证 verification 的一次并行 forward 在每个位置
    # (e,d1..dK) 产出的 target logits，与对同一 token 串逐 token 单步 decode 得到的 logits 数值一致。
    # 只比 logits、不比采样结果 → 排除随机数/temperature/rejection sampler 干扰，纯查张量构造。
    # 返回诊断 dict：逐位置 diff + worst_position/worst_vocab（第 0 位就错通常是 context/position/slot；
    # 第 0 位对、后面错往往是 causal 对齐 / cu_seqlens / KV 写入偏移），并附 verification 的元数据。
    # atol/rtol 默认 1e-2（BF16/FP16 下 varlen vs decode 两内核的正常差异）。
    @torch.inference_mode()
    def verify_alignment(self, seqs: list[Sequence], draft_tokens: torch.Tensor, atol: float = 1e-2, rtol: float = 1e-2):
        K, S = self.num_spec_tokens, len(seqs)
        # Path A：一次并行 verification（prefill 式 varlen + return_all_logits）。
        input_ids, positions = self.prepare_verify(seqs, draft_tokens)
        ctx = get_context()                                   # reset 前抓 verification 元数据
        meta = {
            "positions": positions.tolist(),
            "slot_mapping": ctx.slot_mapping.tolist(),
            "cu_seqlens_q": ctx.cu_seqlens_q.tolist(),
            "cu_seqlens_k": ctx.cu_seqlens_k.tolist(),
            "block_tables": ctx.block_tables.tolist(),
            "context_len_per_seq": [seq.num_tokens + K for seq in seqs],   # = cu_seqlens_k 相邻差
        }
        logits_a = self.model.compute_logits(self.model(input_ids, positions))    # [S*(K+1),V]
        reset_context()
        V = logits_a.size(-1)
        logits_a = logits_a.view(S, K + 1, V).float()
        # Path B：对同一 token 串 [e,d1..dK] 逐 token 单步 decode，收集每位置 logits。
        # 每步写自身槽 N-1+i、读 keys 0..N-1+i，完全自洽（不依赖 Path A 的写入）。
        dt = draft_tokens.tolist()
        logits_b = torch.empty(S, K + 1, V, device=logits_a.device, dtype=torch.float32)
        for i in range(K + 1):
            tok = torch.tensor([([seq.last_token] + dt[s])[i] for s, seq in enumerate(seqs)], dtype=torch.int64, device="cuda")
            pos = torch.tensor([seq.num_tokens - 1 + i for seq in seqs], dtype=torch.int64, device="cuda")
            slot = torch.tensor([self._slot(seq.block_table, seq.num_tokens - 1 + i) for seq in seqs], dtype=torch.int32, device="cuda")
            context_lens = torch.tensor([seq.num_tokens + i for seq in seqs], dtype=torch.int32, device="cuda")
            block_tables = self.prepare_block_tables(seqs)
            set_context(False, slot_mapping=slot, context_lens=context_lens, block_tables=block_tables)
            logits_b[:, i] = self.model.compute_logits(self.model(tok, pos)).float()
            reset_context()
        # 结构对齐判据 = 概率分布是否接近（对 BF16 噪声鲁棒、对结构错位敏感）：
        # 位置/slot/causal 若错位，该位置读到错误 keys → 整条 logits 全错 → 某 token 概率差≈1；
        # 而 BF16 下 varlen(prefill) 与 decode 两内核累加顺序不同，logits 仅有 1-ULP 级（量级几十时
        # 约 0.1~0.25）差异，只会在 near-tie 处让 argmax 翻面，概率差仅 ~1e-2 级。故既不用 raw-logit
        # allclose（1e-2 对 BF16 过严，正确代码也过不了），也不用 strict argmax（near-tie 处伪报）。
        probs_a = logits_a.softmax(dim=-1)
        probs_b = logits_b.softmax(dim=-1)
        prob_diff = (probs_a - probs_b).abs()                 # [S,K+1,V]
        per_position_max_prob = prob_diff.amax(dim=2).amax(dim=0)   # [K+1]
        max_prob_diff = float(prob_diff.max().item())
        # argmax 一致性仅作参考：near-tie 处 BF16 噪声会翻面，不代表结构错。
        argmax_a = logits_a.argmax(dim=-1)                    # [S,K+1]
        argmax_b = logits_b.argmax(dim=-1)
        argmax_match = bool((argmax_a == argmax_b).all().item())
        num_argmax_mismatch = int((argmax_a != argmax_b).sum().item())
        # logit 绝对差：仅诊断，量级即可区分结构错（O(1)+）与 ULP 噪声（~0.1）。
        diff = (logits_a - logits_b).abs()                    # [S,K+1,V]
        per_position_max = diff.amax(dim=2).amax(dim=0)       # [K+1]，对 S、V 取 max
        worst_position = int(per_position_max.argmax().item())
        worst_vocab_index = int(diff[:, worst_position, :].amax(dim=0).argmax().item())
        allclose = torch.allclose(logits_a, logits_b, atol=atol, rtol=rtol)
        return {
            "max_prob_diff": max_prob_diff,
            "per_position_max_prob_diff": [round(x, 6) for x in per_position_max_prob.tolist()],
            "argmax_match": argmax_match,
            "num_argmax_mismatch": num_argmax_mismatch,
            "allclose": allclose,
            "max_abs_diff": float(diff.max().item()),
            "mean_abs_diff": float(diff.mean().item()),
            "worst_position": worst_position,
            "worst_vocab_index": worst_vocab_index,
            "per_position_max_diff": [round(x, 6) for x in per_position_max.tolist()],
            "meta": meta,
        }

    # 捕获多个常用 Decode batch size 的 CUDA Graph。
    @torch.inference_mode()
    def capture_cudagraph(self):
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        # 小批次捕获 1/2/4/8，大批次每隔 16 捕获一次。
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None

        for bs in reversed(self.graph_bs):
            # 从大到小捕获，多个图共享内存池，减少重复显存。
            graph = torch.cuda.CUDAGraph()
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])

            # 捕获前先运行一次，让编译和临时分配完成。
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        # 保存捕获时的固定 Tensor；run_model 只修改内容，再调用 replay。
        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )
