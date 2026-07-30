# Sequence 表示“一条正在生成的请求”。可以把它理解成一张请求进度表：
# 它记录输入和已生成的 token、当前状态、采样参数，以及 token 对应的 KV Cache 块。
#
# 【为什么需要单独的 Sequence】
# 调度器每轮只选择一部分请求送入 GPU。所有需要跨轮保存的状态都放在 Sequence 中，
# 这样 Scheduler 只负责决定“谁运行”，ModelRunner 只负责“如何在 GPU 上运行”。

from copy import copy
from enum import Enum, auto
from itertools import count

from nanovllm.sampling_params import SamplingParams


class SequenceStatus(Enum):
    # Enum（枚举）表示变量只能从固定选项中取值。
    # auto() 会自动生成递增的值，我们只关心 WAITING 等名称。
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


class Sequence:
    block_size = 256

    # count() 是无限计数器，每次 next(counter) 依次得到 0、1、2……
    # 它用于给请求分配唯一且递增的 seq_id。
    counter = count()

    def __init__(self, token_ids: list[int], sampling_params = SamplingParams()):
        # seq_id 不仅区分请求，LLMEngine 最后还会用它恢复提交顺序。
        self.seq_id = next(Sequence.counter)
        self.status = SequenceStatus.WAITING

        # copy 创建浅拷贝。token id 都是整数，因此浅拷贝已经足够。
        # 后面 append 新 token 时，不会意外修改用户传入的原列表。
        self.token_ids = copy(token_ids)
        self.last_token = token_ids[-1]          # last_token 是 Decode 阶段真正送进模型的 token。
        self.num_tokens = len(self.token_ids)    # 当前完整长度，包含 prompt 和已经生成的 token
        self.num_prompt_tokens = len(token_ids)  # prompt 长度创建后不再变化，因此可以据此计算生成部分的长度。
        self.num_cached_tokens = 0               # 已经拥有有效 KV Cache、不必重复计算的 token 数。
        self.num_scheduled_tokens = 0            # 调度器为“当前这一轮”安排的 token 数；执行完后会重新清零。
        self.is_prefill = True

        # block_table 是“逻辑块编号 -> GPU 物理块编号”的映射。
        # 例：逻辑块 0、1 被分配到物理块 7、3，则 block_table == [7, 3]。
        self.block_table = []

        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos

    # __len__ 让 len(seq) 等价于 seq.num_tokens。
    def __len__(self):
        return self.num_tokens

    # __getitem__ 让 seq[0] 或 seq[2:5] 像列表一样访问 token_ids。
    def __getitem__(self, key):
        return self.token_ids[key]

    # @property 把无参数方法包装成只读属性。
    # 调用方写 seq.is_finished，而不是 seq.is_finished()。
    @property
    def is_finished(self):
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self):
        # completion 指模型新生成的部分，不包括输入 prompt。
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self):
        # 切片 [:n] 表示从开头取到下标 n 之前。
        return self.token_ids[:self.num_prompt_tokens]

    @property
    def completion_token_ids(self):
        # 切片 [n:] 表示从下标 n 取到列表末尾。
        return self.token_ids[self.num_prompt_tokens:]

    @property
    def num_blocks(self):
        # 这是整数“向上取整除法”。
        # 例：块大小 256 时，256 个 token 需要 1 块，257 个 token 需要 2 块。
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self):
        # 计算最后一个逻辑块里已有多少 token。
        # 例：长度 257、块大小 256，前一块装满，最后一块只有 1 个。
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    # 返回第 i 个“逻辑块”包含的 token，而不是读取 GPU 物理缓存。
    def block(self, i):
        assert 0 <= i < self.num_blocks
        # 例：block_size=256、i=1 时，切片范围是 [256:512]。
        return self.token_ids[i*self.block_size: (i+1)*self.block_size]

    # 把模型刚采样出的一个 token 添加到序列末尾。
    def append_token(self, token_id: int):
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1

    # __getstate__ / __setstate__ 控制对象被 pickle（序列化）时保存哪些数据。
    # 序列化就是把 Python 对象转换成字节，以便通过共享内存传给其他进程（TP架构进行传递到子进程）。
    def __getstate__(self):
        # Prefill 要读取 prompt 片段，因此发送 token_ids；
        # Decode 只使用 last_token，因此只发一个整数，减少通信量。
        last_state = self.last_token if not self.is_prefill else self.token_ids
        return (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.num_scheduled_tokens, self.block_table, last_state)

    def __setstate__(self, state):
        # tuple 解包顺序必须与 __getstate__ 返回顺序完全一致。
        self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.num_scheduled_tokens, self.block_table, last_state = state
        if isinstance(last_state, list):
            self.token_ids = last_state
            self.last_token = self.token_ids[-1]
        else:
            # Decode 子进程不需要历史 token；历史注意力信息已保存在 GPU KV Cache 中。
            self.token_ids = []
            self.last_token = last_state
