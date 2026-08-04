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

import pickle
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.layers.sampler import Sampler
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.loader import load_model


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

        # KV Cache 预算 = 总显存预算 - 当前已用显存 - 正式前向仍要预留的临时峰值。
        config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
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

    # 将不同长度的 Python block_table 补齐为二维 GPU Tensor。
    def prepare_block_tables(self, seqs: list[Sequence]):
        # 例：[[7,3], [5]] 会补成 [[7,3], [5,-1]]。
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]

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
