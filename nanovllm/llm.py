# 【这个文件做什么】
# 这是用户看到的 LLM 类入口。它直接继承 LLMEngine，因此拥有 generate 等全部方法。
# 单独保留这个空子类，可以让外部始终使用简洁稳定的 `from nanovllm import LLM`。
from nanovllm.engine.llm_engine import LLMEngine


class LLM(LLMEngine):
    # pass 表示类体暂时没有新增内容，但继承来的功能仍然全部可用。
    pass
