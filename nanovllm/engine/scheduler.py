# Scheduler（调度器）决定“这一轮让哪些请求进入 GPU，以及每条请求处理几个 token”。
# 它不执行神经网络，只管理请求队列和 KV Cache 资源。
#
# 【为什么需要调度】
# GPU 同时处理多条请求通常比逐条处理更快，这叫 batching（批处理）。
# 每条请求的 prompt 长度和生成进度不同，所以每个生成步骤都要重新组合批次，
# 这叫 continuous batching（连续批处理）。
#
# 【两个队列】
# waiting：等待 Prefill，或被抢占后等待重新计算的请求。
# running：prompt 已处理完，正在逐 token Decode 的请求。

from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:

    def __init__(self, config: Config):
        # 一个批次最多同时包含多少条请求。
        self.max_num_seqs = config.max_num_seqs

        # 一个 Prefill 批次最多处理多少 token，防止批次过大导致 GPU 显存不足。
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_size = config.kvcache_block_size
        # BlockManager 管理所有物理 KV Cache 块。
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)

        # deque 是双端队列，可高效地从左端或右端添加、删除元素。
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    # 两个队列都为空时，整个 generate() 才真正完成。
    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        # append 加到右侧，后面用 popleft 从左侧取出，形成先入先出（FIFO）。
        self.waiting.append(seq)

    # 选择下一批请求。
    # 返回 (scheduled_seqs, is_prefill)：
    # scheduled_seqs 是本轮进入 GPU 的 Sequence 列表；
    # is_prefill 告诉 ModelRunner 应准备 prompt 片段还是单个 last_token。
    def schedule(self) -> tuple[list[Sequence], bool]:
        scheduled_seqs = []
        num_batched_tokens = 0

        # ---------- 阶段一：优先尝试 Prefill ----------
        # Prefill 和 Decode 的输入形状不同，本实现不会把二者混在同一个批次。
        while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
            # 只查看队首，确认资源足够后才真正弹出。
            seq = self.waiting[0]

            # 例：上限 100、本轮已安排 60 个 token，则 remaining=40。
            remaining = self.max_num_batched_tokens - num_batched_tokens
            if remaining == 0:
                break
            if not seq.block_table:
                # 空 block_table 表示该请求还没有持有 KV Cache 块。
                # can_allocate 会同时检查缓存命中数量和剩余物理块是否足够。
                num_cached_blocks = self.block_manager.can_allocate(seq)
                if num_cached_blocks == -1:
                    # -1 表示当前连队首请求都无法安全分配。
                    break

                # 例：10 个 token、块大小 4、命中 2 块，只需计算 10 - 2*4 = 2 个 token。
                num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
            else:
                # 块表已存在，说明之前只处理了 prompt 的一部分（Chunked Prefill）。
                num_tokens = seq.num_tokens - seq.num_cached_tokens

            # 剩余预算放不下完整请求时：
            # 当前批次为空则允许只处理一部分；已有其他请求则停止继续添加。
            if remaining < num_tokens and scheduled_seqs:
                break
            if not seq.block_table:
                # 真正修改空闲块、引用计数，并为 seq 创建 对应的block_table。
                self.block_manager.allocate(seq, num_cached_blocks)

            # min 保证本轮不会超过 token 预算。chunked prefill思想
            seq.num_scheduled_tokens = min(num_tokens, remaining)
            num_batched_tokens += seq.num_scheduled_tokens
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                # 本轮结束后，完整 prompt 都有 KV Cache，下一轮可以进入 Decode。
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)
            # 即使只处理 prompt 的一部分，本轮也必须把它送进 GPU。
            scheduled_seqs.append(seq)

        if scheduled_seqs:
            # 成功安排任何 Prefill 后立即返回，不再继续构造 Decode 批次。
            return scheduled_seqs, True

        # ---------- 阶段二：没有 Prefill 时执行 Decode ----------
        # 每条 running 序列本轮只处理自己的 last_token。
        while self.running and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.running.popleft()

            # 如果 last_token 需要一个新块，而当前没有空闲块，就先抢占其他请求。
            while not self.block_manager.can_append(seq):
                if self.running:
                    # pop 从右侧取队尾请求。
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                # Python 的 while...else 表示循环没有执行 break 时进入 else。
                seq.num_scheduled_tokens = 1
                seq.is_prefill = False
                # 如果跨入新逻辑块，此处真正分配物理块。
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
        # 正常情况下至少应该调度一条请求，否则说明缓存或状态有错误。
        assert scheduled_seqs

        # extendleft 会逐个插到左侧，所以先 reversed 才能保持原顺序。
        self.running.extendleft(reversed(scheduled_seqs))  # 保持未完成序列的相对调度顺序。
        return scheduled_seqs, False

    # 抢占一条请求：释放它持有的物理块，并移回 waiting 队首。
    # token_ids 不会丢失，之后可以重新 Prefill。
    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        # 释放kv cache
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    # GPU 完成本轮计算后，更新 Sequence 的缓存进度、生成 token 和完成状态。
    def postprocess(self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool):
        # zip 把第 i 条序列与第 i 个采样结果配对。
        for seq, token_id in zip(seqs, token_ids):
            # 如果本轮填满了完整块，为它建立前缀缓存索引。
            self.block_manager.hash_blocks(seq)

            # 本轮安排的 token 现在已经计算并写入 KV Cache。
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0
            # Chunked Prefill 还没覆盖完整 prompt 时，下一轮继续处理剩余 prompt。
            if is_prefill and seq.num_cached_tokens < seq.num_tokens:
                continue

            # 完整 Prefill 后，这是第一个生成 token；Decode 时则是下一个生成 token。
            seq.append_token(token_id)

            # 满足任一停止条件即结束：生成 EOS，或达到最大生成长度。
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED

                # 请求完成后立刻释放 KV Cache，供其他请求使用。
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
