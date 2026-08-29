# BlockManager 管理 GPU 上的 KV Cache（键值缓存）空间。

# 【逻辑块与物理块】
# 假设 block_size=4，请求 token 为 [A,B,C,D,E,F]，逻辑上分为：
# 逻辑块 0=[A,B,C,D]，逻辑块 1=[E,F]。
# 如果 GPU 物理块 7 和 3 分配给它，Sequence.block_table 就是 [7,3]。

# 【前缀缓存 Prefix Cache】
# 两个请求如果开头有相同的完整 token 块，它们对应的 K/V 也完全相同，
# 因此可以让两个请求共享同一个物理块，避免重复 Prefill。
from collections import OrderedDict
import xxhash
import numpy as np

from nanovllm.engine.sequence import Sequence
from nanovllm.engine.metrics import PrefixCacheStats


class Block:

    # Block 是一个物理 KV Cache 块在 CPU 侧的“管理记录”。
    # 真正的大块 K/V 张量在 ModelRunner.kv_cache 中，这里只保存编号和元数据。
    def __init__(self, block_id):
        # block_id 是这个物理块在 GPU 大张量中的下标。
        self.block_id = block_id

        # ref_count 是引用计数：有多少条 Sequence 正在使用这个块。
        # 0 表示没有请求占用；2 表示两个相同前缀的请求正在共享。
        self.ref_count = 0

        # hash 是截至当前块的“累计前缀哈希”。-1 表示尚未建立有效缓存索引。
        self.hash = -1

        # 保存这个完整块对应的 token，用于命中后再次核对，避免极小概率的哈希碰撞。
        self.token_ids = []

    # 当一个块刚被完整写满时，记录它的累计哈希与 token 内容。
    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids

    # 物理块将被新内容覆盖前，重置管理信息。
    def reset(self):
        # 新分配后立即有一个 Sequence 使用，因此 ref_count 从 1 开始。
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class BlockManager:

    def __init__(self, num_blocks: int, block_size: int, enable_lru: bool = True):
        self.block_size = block_size

        # enable_lru=True 使用显式 LRU 驱逐；False 回退原版 FIFO，用于对照基线。
        self.enable_lru = enable_lru

        # 创建 num_blocks 个 CPU 管理对象；它们与 GPU 物理块一一对应。
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]

        # 字典实现“累计前缀哈希 -> 物理块 id”的快速查询。
        self.hash_to_block_id: dict[int, int] = dict()

        # 空闲块用 OrderedDict 维护“释放先后顺序”：队首是最久之前释放的块（LRU 端），
        # 队尾是最近释放的块。相比 deque，它支持 O(1) 按 id 删除（命中重激活时用到）。
        # 初始化时全部块都空闲，插入顺序为 0..num_blocks-1。
        self.free_block_ids: OrderedDict[int, None] = OrderedDict.fromkeys(range(num_blocks))

        # set（集合）记录正在被至少一个请求使用的块 id，查询复杂度接近 O(1)。
        self.used_block_ids: set[int] = set()

        # Prefix Cache 命中/驱逐统计，仅用于 benchmark 汇总，不参与推理逻辑。
        self.prefix_stats = PrefixCacheStats(block_size=block_size)

    # @classmethod 表示方法接收类 cls，而不是某个实例 self。
    # 本函数不依赖某个 BlockManager 的字段，放在类中只是为了归类。
    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        # 哈希可以把很长的 token 列表压缩成一个整数，便于放入字典查询。
        h = xxhash.xxh64()

        # 当前块的 K/V 不只由当前块 token 决定，也由它前面的上下文决定。
        # 所以先写入前一块的哈希，再写当前 token，形成“链式累计哈希”。
        # 例如第二块内容相同，但第一块不同，最终哈希也会不同。
        if prefix != -1:
            # to_bytes 把整数转成 8 个字节；little 表示低位字节放在前面。
            h.update(prefix.to_bytes(8, "little"))

        # xxhash 接受字节，不直接接受 Python 列表，因此先转成 NumPy 数组再取原始字节。
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    # 从空闲队列取出一个物理块，并准备让新请求使用。
    def _allocate_block(self) -> int:
        # 统一从队首取块：
        # - FIFO 模式（enable_lru=False）：队首是最久之前释放的块，等价于原 deque.popleft。
        # - LRU 模式：_deallocate_block 已把“无缓存价值”的块移到队首，因此队首要么是无哈希块
        #   （复写零损失），要么是最久未用的缓存块（真正的 LRU 驱逐）。
        block_id = next(iter(self.free_block_ids))
        del self.free_block_ids[block_id]
        block = self.blocks[block_id]

        # 空闲块理论上不应仍有使用者；assert 用来尽早暴露内部状态错误。
        assert block.ref_count == 0

        # 一个已释放块可能保留旧 K/V 供 Prefix Cache 命中。现在它将被新内容覆盖：
        # 若它仍带有效哈希且索引指向自己，说明这是一次真正的缓存驱逐，删除索引并计数。
        if block.hash != -1 and self.hash_to_block_id.get(block.hash) == block_id:
            del self.hash_to_block_id[block.hash]
            self.prefix_stats.record_eviction()

        block.reset()
        self.used_block_ids.add(block_id)
        return block_id

    # 引用计数归零后，把块放回空闲队列。
    def _deallocate_block(self, block_id: int):
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        # 回插到队尾，表示“最近释放”。
        self.free_block_ids[block_id] = None

        # LRU 模式下，无缓存价值的块（hash==-1）移到队首，让它们最先被复用，
        # 从而尽量保留仍可能命中的缓存块，把驱逐推迟到真正显存不足时。
        if self.enable_lru and self.blocks[block_id].hash == -1:
            self.free_block_ids.move_to_end(block_id, last=False)

        # 注意：这里故意不清除 block.hash 和 token_ids。
        # 只要该物理块尚未被新内容覆盖，旧 K/V 仍然有效，可被 Prefix Cache 重新激活。

    # 沿前缀逐块匹配 Prefix Cache，返回 (命中的完整前缀块数, 最坏还需新占用的物理块数)。
    # can_allocate 与 count_cached_prefix_blocks 共用这一个哈希循环，避免逻辑重复。
    def _match_cached_prefix(self, seq: Sequence) -> tuple[int, int]:
        # -1 作为“还没有前一个块哈希”的哨兵值。
        h = -1
        num_cached_blocks = 0

        # 最坏情况：所有逻辑块都要新占用一个空闲物理块。
        num_new_blocks = seq.num_blocks

        # range(seq.num_blocks - 1) 故意排除最后一块。
        # 最后一块通常没填满，未来还会追加 token，不能作为稳定的共享前缀。
        for i in range(seq.num_blocks - 1):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block_id = self.hash_to_block_id.get(h, -1)

            # 哈希相同不代表 100% 相同，因此还要比较实际 token 列表。
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                # 前缀必须连续；第 i 块未命中后，后面的块不能跳过它继续复用。
                break

            num_cached_blocks += 1
            if block_id in self.used_block_ids:
                # 正在使用的命中块通过增加引用计数共享，不占用 free 队列中的新块。
                num_new_blocks -= 1

        return num_cached_blocks, num_new_blocks

    # 在真正修改状态前，检查一个新请求是否能完整获得所需块。
    # 返回值：
    # - -1：可用物理块不足；
    # - 0：能分配，但没有命中前缀缓存；
    # - 正整数 n：能分配，并且前 n 个完整逻辑块可直接复用。
    def can_allocate(self, seq: Sequence) -> int:
        num_cached_blocks, num_new_blocks = self._match_cached_prefix(seq)

        # 闲置的缓存命中块仍在 free_block_ids 中，重新启用它也会占掉一个空闲名额，
        # 因此只有“当前已被使用的共享块”能从 num_new_blocks 中减掉。
        if len(self.free_block_ids) < num_new_blocks:
            return -1
        return num_cached_blocks

    # 只返回命中的完整前缀块数，不做分配可行性判断、无任何副作用（不 record_query、不改状态）。
    # 供 Cache-Aware 调度给多个等待请求打分排序用。
    def count_cached_prefix_blocks(self, seq: Sequence) -> int:
        return self._match_cached_prefix(seq)[0]

    # 真正为 Sequence 建立 block_table。
    # 调用前必须先通过 can_allocate，避免分配到一半才发现资源不足。
    def allocate(self, seq: Sequence, num_cached_blocks: int):
        # 同一个 Sequence 不能重复分配块表。
        assert not seq.block_table
        h = -1

        # ---------- 先挂载命中的 Prefix Cache 块 ----------
        for i in range(num_cached_blocks):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block_id = self.hash_to_block_id[h]
            block = self.blocks[block_id]
            if block_id in self.used_block_ids:
                # 该块正在被其他 Sequence 使用，引用计数加一即可共享。
                block.ref_count += 1
            else:
                # 该块虽然在空闲队列，但旧 K/V 仍有效；从空闲队列移除并重新激活。
                # OrderedDict 支持按 id O(1) 删除，优于原 deque.remove 的 O(n) 扫描。
                block.ref_count = 1
                del self.free_block_ids[block_id]
                self.used_block_ids.add(block_id)
            seq.block_table.append(block_id)

        # ---------- 为未命中部分分配新物理块 ----------
        for i in range(num_cached_blocks, seq.num_blocks):
            seq.block_table.append(self._allocate_block())

        # 每个命中块都是完整块，所以缓存 token 数 = 命中块数 * 块大小。
        seq.num_cached_tokens = num_cached_blocks * self.block_size

        # 记录一次缓存查询：分母为可被复用的完整前缀块数（末块不参与），分子为命中块数。
        # 放在 allocate（每个 Sequence 恰好首次分配时调用一次），避免 can_allocate 被反复
        # 试探调用时重复计数。
        self.prefix_stats.record_query(max(seq.num_blocks - 1, 0), num_cached_blocks)

    # Sequence 完成或被抢占时，释放它对所有物理块的引用。
    def deallocate(self, seq: Sequence):
        # reversed 从最后一个逻辑块向前释放。
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1

            # 共享块只有最后一个使用者离开时才真正回到空闲队列。
            if block.ref_count == 0:
                self._deallocate_block(block_id)

        # Sequence 已不再拥有任何有效 KV Cache。
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    # 暴露 Prefix Cache 统计给上层（Scheduler → LLMEngine → benchmark）。
    def get_stats(self) -> PrefixCacheStats:
        return self.prefix_stats

    # 检查下一次 Decode 是否需要新块，以及当前是否有空闲块可用。
    def can_append(self, seq: Sequence) -> bool:
        # Scheduler 调用这里时，上轮采样 token 已 append 到 Sequence，但尚未写入 KV Cache。
        # 若 len(seq) % block_size == 1，说明这个 last_token 是新逻辑块的第一个 token。
        # Python 中 bool 可当作 0/1，所以右侧结果要么是 0（无需新块），要么是 1（需要一块）。
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    # 如果 last_token 跨入新逻辑块，就真正分配物理块。
    def may_append(self, seq: Sequence):
        if len(seq) % self.block_size == 1:
            seq.block_table.append(self._allocate_block())

    # 将本轮新填满的完整块登记到 Prefix Cache。
    def hash_blocks(self, seq: Sequence):
        # start 是本轮开始前已有多少个完整缓存块；
        # end 是本轮结束后共有多少个完整缓存块。
        start = seq.num_cached_tokens // self.block_size
        end = (seq.num_cached_tokens + seq.num_scheduled_tokens) // self.block_size

        # 没有新填满的块时直接返回。单行 if 是普通 if 的紧凑写法。
        if start == end: return

        # 若不是第一块，就从前一块的累计哈希继续计算。
        h = self.blocks[seq.block_table[start - 1]].hash if start > 0 else -1
        for i in range(start, end):
            block = self.blocks[seq.block_table[i]]
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block.update(h, token_ids)
            self.hash_to_block_id[h] = block.block_id

            # 字典中同一个哈希只需保留一个物理块：
            # 相同前缀产生的 K/V 相同，后续请求共享任意一个副本即可。
