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
from time import perf_counter
import os

# Cache-Aware 调度参数：
# _LPM_WINDOW —— 只对 waiting 前 W 个 fresh 请求打分排序，把每步开销 bound 在与队列深度无关的常量。
#                每步只打分一次（见 _build_prefill_order），故 W 可取较大值而不吞收益。
# _AGING_INTERVAL —— 每 K 步 prefill 定序强制走一次 FIFO 到达顺序，bound 低命中请求最坏等待、防饿死。
_LPM_WINDOW = 128
_AGING_INTERVAL = 8

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.metrics import SchedulerStats


class Scheduler:

    def __init__(self, config: Config):
        # 一个批次最多同时包含多少条请求。
        self.max_num_seqs = config.max_num_seqs

        # 一个 Prefill 批次最多处理多少 token，防止批次过大导致 GPU 显存不足。
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_size = config.kvcache_block_size
        # BlockManager 管理所有物理 KV Cache 块。enable_lru 控制驱逐策略（LRU / 回退 FIFO）。
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size, config.enable_lru)

        # deque 是双端队列，可高效地从左端或右端添加、删除元素。
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

        # ---------- 功能一：Chunked Prefill（vLLM 式 prefill/decode 混批）----------
        # 关：原版两阶段调度（prefill 步与 decode 步互斥），作对照基线。
        # 开：同一步先给 running 序列各排 1 个 decode token，再用剩余预算给 waiting 做分块 prefill，
        #     使 decode 不再被长 prompt 的连续 prefill 饿住。
        self.enable_chunked_prefill = config.enable_chunked_prefill

        # ---------- 功能三：Cache-Aware Scheduling（LPM 前缀感知调度）----------
        # 关：waiting 严格 FIFO 出队（原版行为），作对照基线。
        # 开：prefill 优先选“与已缓存前缀匹配最长”的等待请求，把同前缀请求在时间上聚拢。
        self.enable_cache_aware_schedule = config.enable_cache_aware_schedule
        self._lpm_window = _LPM_WINDOW
        self._aging_interval = _AGING_INTERVAL   # 暴露为属性，单测可调低以强制触发 aging
        self._prefill_select_calls = 0

        # ---------- 功能四：Speculative Decoding ----------
        # 关：pending_kind 恒 "normal"，不建 draft BlockManager，逐字节走原路径（零回归）。
        # 开：无 prefill 可调度时，尝试组一个 spec step（每序列 propose K 个候选、计入 K+1 预算），
        #     draft KV 用独立 BlockManager 管理（不做 prefix 复用/hash，enable_lru=False）。
        self.enable_speculative_decode = getattr(config, "enable_speculative_decode", False)
        self.num_speculative_tokens = getattr(config, "num_speculative_tokens", 4)
        # SPEC_DIAG=1：每个 spec step 打印一行 [SPEC_SCHED]（组批规模 + 各闸门拒绝数），
        # 用来查清「首个 spec batch 为何只有 51/64」与逐步 draft lag 走势；默认关、零开销。
        self.spec_diag = os.environ.get("SPEC_DIAG") == "1"
        # pending_kind 告诉 LLMEngine 本步该走哪条执行路径："normal" | "speculative"。
        self.pending_kind = "normal"
        self.draft_block_manager = None
        if self.enable_speculative_decode:
            self.draft_block_manager = BlockManager(
                getattr(config, "num_draft_kvcache_blocks", 0),
                config.kvcache_block_size,
                enable_lru=False,
            )

        # 调度统计，仅用于 benchmark 汇总。
        self.stats = SchedulerStats()

    # 两个队列都为空时，整个 generate() 才真正完成。
    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        # append 加到右侧，后面用 popleft 从左侧取出，形成先入先出（FIFO）。
        self.waiting.append(seq)

    # Cache-Aware：一次性给出本轮 prefill 的候选尝试顺序（续跑红线 → aging → LPM）。
    # 【关键】每步只打分一次（O(W×hash)），而非每轮循环重打分（O(seqs×W×hash)）——否则调度开销吞掉收益。
    # 同一 step 内不重打分是安全的：新前缀哈希只在 postprocess 的 hash_blocks 才登记，step 内命中数不变。
    def _build_prefill_order(self) -> list[Sequence]:
        cont = None
        fresh = []
        for seq in self.waiting:
            if seq.block_table:
                cont = seq                       # 续跑请求（分块 prefill 中途），同一时刻至多一条
            else:
                fresh.append(seq)
        # 1) 续跑红线：已开始 prefill 的请求必须先续上，绝不被 LPM 重排晾着（否则分块续跑语义被破坏）。
        order = [cont] if cont is not None else []
        if not fresh:
            return order
        # 2) aging 阀门：每 K 步定序强制走 FIFO 到达顺序，bound 低命中请求最坏等待、防饿死。
        self._prefill_select_calls += 1
        if self._prefill_select_calls % self._aging_interval == 0:
            order.extend(fresh)
            return order
        # 3) LPM：窗口内按命中前缀块数降序（Python sort 稳定 + reverse → 平手保留先到者=FIFO）；窗口外维持 FIFO 追加。
        w = self._lpm_window
        order.extend(sorted(fresh[:w],
                            key=lambda s: self.block_manager.count_cached_prefix_blocks(s),
                            reverse=True))
        order.extend(fresh[w:])
        return order

    # 从 waiting 移除已被调度的请求：关闭时 popleft（O(1)，同原版）；开启时按 identity 从中间移除（O(n)）。
    def _remove_from_waiting(self, seq: Sequence):
        if self.enable_cache_aware_schedule:
            self.waiting.remove(seq)
        else:
            self.waiting.popleft()

    # 选择下一批请求。
    # 返回 (scheduled_seqs, is_prefill)：
    # scheduled_seqs 是本轮进入 GPU 的 Sequence 列表；
    # is_prefill 告诉 ModelRunner 应准备 prompt 片段还是单个 last_token。
    def schedule(self) -> tuple[list[Sequence], bool]:
        # 每步先复位为普通路径；仅 _select_speculative 成功组批时才改成 "speculative"。
        self.pending_kind = "normal"
        if self.enable_chunked_prefill:
            return self._schedule_chunked()
        return self._schedule_two_phase()

    # 原版两阶段调度：Prefill 步与 Decode 步永不混批。enable_chunked_prefill=False 时使用，作对照基线。
    def _schedule_two_phase(self) -> tuple[list[Sequence], bool]:
        scheduled_seqs = []
        num_batched_tokens = 0

        # ---------- 阶段一：优先尝试 Prefill ----------
        # Prefill 和 Decode 的输入形状不同，本路径不会把二者混在同一个批次。
        # Cache-Aware 开启时本步一次性定序（LPM，见 _build_prefill_order）；关闭时严格 FIFO 队首（零回归）。
        order = self._build_prefill_order() if self.enable_cache_aware_schedule else None
        oi = 0
        while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
            if order is None:
                seq = self.waiting[0]
            else:
                if oi >= len(order):
                    break
                seq = order[oi]
                oi += 1
            remaining = self.max_num_batched_tokens - num_batched_tokens
            if remaining == 0:
                break
            if not seq.block_table:
                # 空 block_table 表示该请求还没有持有 KV Cache 块。
                num_cached_blocks = self.block_manager.can_allocate(seq)
                if num_cached_blocks == -1:
                    break
                num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
            else:
                # 块表已存在，说明之前只处理了 prompt 的一部分（长 prompt 分块续跑）。
                num_tokens = seq.num_tokens - seq.num_cached_tokens
            # 剩余预算放不下完整请求：批次为空则允许只处理一部分，否则停止继续添加。
            if remaining < num_tokens and scheduled_seqs:
                break
            if not seq.block_table:
                self.block_manager.allocate(seq, num_cached_blocks)
            chunk = min(num_tokens, remaining)
            seq.num_scheduled_tokens = chunk
            num_batched_tokens += chunk
            if seq.num_cached_tokens + chunk == seq.num_tokens:
                seq.status = SequenceStatus.RUNNING
                self._remove_from_waiting(seq)
                self.running.append(seq)
                scheduled_seqs.append(seq)
            else:
                # 仅处理了 prompt 的一部分：留在 waiting（下一轮被续跑红线优先续上），从 num_cached_tokens 续。
                scheduled_seqs.append(seq)
                break

        if scheduled_seqs:
            self.stats.record_prefill_step(num_batched_tokens)
            return scheduled_seqs, True

        # ---------- 阶段一半：无 prefill 时优先尝试组一个 speculative step ----------
        # 成功则本步走投机路径（pending_kind="speculative"），返回 is_prefill=False（执行路径由
        # LLMEngine 按 pending_kind 分派，不看该布尔）；组不起来（无 running / 资源不足）则落到普通 decode。
        if self.enable_speculative_decode:
            spec_seqs = self._select_speculative()
            if spec_seqs:
                self.pending_kind = "speculative"
                return spec_seqs, False
            # spec 开启却组不起一个 spec step（全部序列装不下）：本步退化为普通 decode。
            # target 会前进而 draft KV 冻结 → 下一 spec step 需补 lag。计数以确认这条隐藏 fallback 是否发生。
            if self.running:
                self.stats.spec_fallback_decode_steps += 1

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

        # 记录一次 decode 步统计（benchmark 用）。
        self.stats.record_decode_step(len(scheduled_seqs))
        return scheduled_seqs, False

    # vLLM 式混批：同一步先给所有 running 序列各排 1 个 decode token，再用剩余预算给 waiting 做
    # 分块 prefill。decode 不再被长 prompt 的连续 prefill 饿住（victim 的 token 间隔 ~1s → ~一步）。
    def _schedule_chunked(self) -> tuple[list[Sequence], bool]:
        scheduled_seqs = []

        # ---------- 阶段一：所有 running 序列各排 1 个 decode token ----------
        decode_seqs = []
        while self.running and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                seq.num_scheduled_tokens = 1
                seq.is_prefill = False
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
                decode_seqs.append(seq)
        num_decode_tokens = len(decode_seqs)
        # decode 序列放回 running（保持相对顺序），下一轮继续 decode。
        self.running.extendleft(reversed(decode_seqs))

        # ---------- 阶段二：剩余 token 预算内给 waiting 队首做 chunked prefill ----------
        num_prefill_tokens = 0
        prefill_budget = self.max_num_batched_tokens - num_decode_tokens
        # Cache-Aware 开启时本步一次性定序（LPM）；关闭时严格 FIFO 队首（零回归）。
        order = self._build_prefill_order() if self.enable_cache_aware_schedule else None
        oi = 0
        while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
            if order is None:
                seq = self.waiting[0]
            else:
                if oi >= len(order):
                    break
                seq = order[oi]
                oi += 1
            remaining = prefill_budget - num_prefill_tokens
            if remaining <= 0:
                break
            if not seq.block_table:
                num_cached_blocks = self.block_manager.can_allocate(seq)
                if num_cached_blocks == -1:
                    break
                num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
            else:
                num_tokens = seq.num_tokens - seq.num_cached_tokens
            if not seq.block_table:
                self.block_manager.allocate(seq, num_cached_blocks)
            # 混批下总是允许部分 prefill：切一段填满剩余预算即可，不要求空批才切。
            chunk = min(num_tokens, remaining)
            seq.num_scheduled_tokens = chunk
            num_prefill_tokens += chunk
            if seq.num_cached_tokens + chunk == seq.num_tokens:
                seq.status = SequenceStatus.RUNNING
                self._remove_from_waiting(seq)
                self.running.append(seq)
                scheduled_seqs.append(seq)
            else:
                # 只处理一部分：留在 waiting（下一轮被续跑红线优先续上），下轮从 num_cached_tokens 续；本步 prefill 到此为止。
                scheduled_seqs.append(seq)
                break

        assert scheduled_seqs
        if num_prefill_tokens > 0:
            # 混批步（含 prefill chunk）走 varlen+eager 路径：LM head 用 cu_seqlens_q 每序列取一个 logit。
            self.stats.record_prefill_step(num_prefill_tokens)
            self.stats.total_decode_tokens += num_decode_tokens
            return scheduled_seqs, True
        # waiting 为空：本步退化为纯 decode，走固定形状 CUDA Graph（稳态 decode 不丢图加速）。
        self.stats.record_decode_step(num_decode_tokens)
        return scheduled_seqs, False

    # 组一个 speculative step：从 running 中挑能容纳「末 token e + K 个候选」的序列。
    # 每条序列按 K+1 计入 token 预算；target 与 draft 双侧都要能备足 tentative 块才纳入
    # （原子性：先 can_append_spec 双查，都过再 may_append_spec，避免只申请到一半）。
    # 挑不满不影响正确性：未入选序列留在 running，下一步继续尝试或降级普通 decode。
    def _select_speculative(self) -> list[Sequence]:
        K = self.num_speculative_tokens
        need = K + 1
        selected = []
        num_batched_tokens = 0
        rej_budget = rej_target = rej_draft = 0     # 本次组批各闸门拒绝数（诊断用）
        # 遍历当前 running 快照：逐条 popleft 判定后再 append 回队尾，一轮后顺序不变。
        for _ in range(len(self.running)):
            if len(selected) >= self.max_num_seqs:
                # 已达批上限，剩余序列原样留在 running。
                self.running.append(self.running.popleft())
                continue
            seq = self.running.popleft()
            # 按短路顺序（预算 → target KV → draft KV）归因到首个失败闸门：
            # 定位「首个 spec batch 为何只有 51/64」的确切限流点（几乎必然是 draft KV 池）。
            if num_batched_tokens + need > self.max_num_batched_tokens:
                rej_budget += 1
            elif not self.block_manager.can_append_spec(seq, K):
                rej_target += 1
            elif not self.draft_block_manager.can_append_spec(seq, K, seq.draft_block_table):
                rej_draft += 1
            else:
                self.block_manager.may_append_spec(seq, K)
                self.draft_block_manager.may_append_spec(seq, K, seq.draft_block_table)
                seq.is_prefill = False
                seq.num_scheduled_tokens = need
                num_batched_tokens += need
                selected.append(seq)
            self.running.append(seq)   # 无论是否入选，都留在 running
        if selected:
            self.stats.record_speculative_step(len(selected), K)
        self.stats.spec_reject_budget += rej_budget
        self.stats.spec_reject_target += rej_target
        self.stats.spec_reject_draft += rej_draft
        if self.spec_diag:
            print(f"[SPEC_SCHED] running={len(self.running)} selected={len(selected)} "
                  f"reject: draft={rej_draft} target={rej_target} budget={rej_budget}", flush=True)
        return selected

    # 抢占一条请求：释放它持有的物理块，并移回 waiting 队首。
    # token_ids 不会丢失，之后可以重新 Prefill。
    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        # 释放kv cache
        self.block_manager.deallocate(seq)
        # spec 序列的 draft KV 也一并释放并清零计数：重算后由 _draft_propose 第 0 步融合补写从头重建，
        # 避免 draft KV 与被重新 prefill 的 target 序列错位。
        if self.draft_block_manager is not None:
            self.draft_block_manager.deallocate_draft(seq)
        self.waiting.appendleft(seq)
        # 记录一次抢占，抢占越多说明显存压力越大（benchmark 用）。
        self.stats.record_preemption()

    # GPU 完成本轮计算后，更新 Sequence 的缓存进度、生成 token 和完成状态。
    def postprocess(self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool):
        # zip 把第 i 条序列与第 i 个采样结果配对。
        for seq, token_id in zip(seqs, token_ids):
            # 如果本轮填满了完整块，为它建立前缀缓存索引。
            self.block_manager.hash_blocks(seq)

            # 本轮安排的 token 现在已经计算并写入 KV Cache。
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0
            # 按序列判定（混批步一步内 prefill/decode 序列并存）：prompt 还没覆盖完的序列丢弃本步采样，
            # 下一轮继续 prefill 剩余部分；decode 序列与刚 prefill 完的序列才 append token。
            if seq.num_cached_tokens < seq.num_tokens:
                continue

            # 完整 Prefill 后，这是第一个生成 token；Decode 时则是下一个生成 token。
            seq.append_token(token_id)

            # 刚生成第一个 completion token：记录 TTFT 终点时刻。
            if seq.num_completion_tokens == 1:
                seq.first_token_time = perf_counter()

            # 满足任一停止条件即结束（EOS 或达到 max_tokens）。停止判定与善后抽成
            # _is_stop/_finish_seq，供 postprocess_speculative 逐 token 复用同一套逻辑。
            if self._is_stop(seq, token_id):
                self._finish_seq(seq)

    # 停止判据：生成 EOS（且未 ignore_eos），或已达最大生成长度。
    def _is_stop(self, seq: Sequence, token_id: int) -> bool:
        return (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens

    # 请求结束的统一善后：打完成时刻、释放 target（及 draft）KV、移出 running。
    def _finish_seq(self, seq: Sequence):
        seq.status = SequenceStatus.FINISHED
        seq.finish_time = perf_counter()
        self.block_manager.deallocate(seq)
        if self.draft_block_manager is not None:
            self.draft_block_manager.deallocate_draft(seq)
        self.running.remove(seq)

    # 投机步善后：一步内一条序列提交 1~K+1 个 token。results[i] = (committed_token_ids, accept_len)。
    # committed_token_ids 已由 rejection_sample 截到 accept_len+1 个（accepted draft + recovered/bonus）。
    def postprocess_speculative(self, seqs: list[Sequence], results):
        K = self.num_speculative_tokens
        for seq, (committed, accept_len) in zip(seqs, results):
            # 本步开始前的长度：draft propose 第 0 步融合已把 draft KV 补到 n_before-1，
            # propose 又真实写了 K 个索引，故 draft 真实写入上界 = n_before-1+K（见下 num_draft_cached）。
            n_before = seq.num_tokens
            emitted = 0
            stopped = False
            # 逐 token 提交并复用同一套停止判定：命中即在该 token 处截断，丢弃其后 committed（off-by-one 安全）。
            for token_id in committed:
                seq.append_token(token_id)
                emitted += 1
                if seq.num_completion_tokens == 1:
                    seq.first_token_time = perf_counter()
                if self._is_stop(seq, token_id):
                    stopped = True
                    break
            seq.num_scheduled_tokens = 0

            # 接受统计：accepted 只数被接受的 draft token（=accept_len）；emitted 计入 decode 吞吐口径。
            self.stats.record_acceptance(accepted=accept_len, emitted=emitted, bonus=(accept_len == K))

            # 「末 token 不缓存」：提交后 target 缓存前进到 len-1。
            num_written = seq.num_tokens - 1
            seq.num_cached_tokens = num_written
            # draft 真实已写入 KV 上界 = min(len-1, n_before-1+K)：全接受+bonus 时 dK/bonus 未写，
            # 故 num_draft_cached 落后 1，下一步 propose 第 0 步融合补齐（反映真实写入，非纯计数）。
            seq.num_draft_cached_tokens = min(num_written, n_before - 1 + K)

            # 只登记完全提交的满块（防污染），再把双侧多余 tentative 块裁回 free。
            self.block_manager.hash_blocks_committed(seq, num_written)
            keep = seq.num_blocks
            self.block_manager.trim_blocks(seq, keep)
            self.draft_block_manager.trim_blocks(seq, keep, seq.draft_block_table)

            if stopped:
                self._finish_seq(seq)

    # 暴露调度统计给 LLMEngine（benchmark 汇总用）。
    def get_stats(self) -> SchedulerStats:
        return self.stats
