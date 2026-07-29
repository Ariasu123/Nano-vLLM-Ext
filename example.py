# 【这个文件做什么】
# 这是整个项目最适合开始阅读的文件。它演示了用户如何把两句话交给 Nano-vLLM，
# 再从返回结果中取出模型生成的文字。
#
# 【建议的阅读顺序】
# 先读懂本文件的 main()，然后进入 nanovllm/engine/llm_engine.py，观察 LLM.generate()
# 如何把 prompt（提示词）转换成 token（模型使用的整数编号）并启动生成循环。
#
# 【需要先知道的概念】
# 大模型不能直接读取字符串。Tokenizer（分词器）会把文字转换成 token id 列表，
# 例如“你好”可能被转换为 [108386]。模型输出的也仍然是 token id，最后再由分词器还原成文字。
import os
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer


def main():
    # expanduser 会把路径开头的 "~" 展开成当前用户的主目录。
    # 这里的 path 必须指向已经下载好的 Hugging Face 模型文件夹。
    path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")

    # AutoTokenizer 根据模型目录里的配置选择正确的分词器。
    # 它只负责“文字 <-> token id”的转换，不负责运行神经网络。
    tokenizer = AutoTokenizer.from_pretrained(path)

    # 创建推理引擎时会完成较重的初始化工作：
    # 1. 读取模型配置；2. 在 GPU 上创建 Qwen3；3. 加载模型权重；
    # 4. 预热模型；5. 使用剩余显存分配 KV Cache。
    # enforce_eager=True 表示先不使用 CUDA Graph 优化，便于调试和理解执行过程。
    llm = LLM(path, enforce_eager=True, tensor_parallel_size=1)

    # temperature 控制随机程度：值越小，模型越倾向于选择概率最高的 token。
    # max_tokens=256 表示每个请求最多生成 256 个新 token，不包含输入 prompt。
    sampling_params = SamplingParams(temperature=0.6, max_tokens=256)

    # prompts 是普通 Python 字符串列表，因此下面会一次提交两个独立请求。
    prompts = [
        "introduce yourself",
        "list all prime numbers within 100",
    ]

    # 指令模型并不是直接用原始问题训练的，而是使用“角色化”的对话格式。
    # apply_chat_template 会把一条用户消息包装成模型要求的特殊格式。
    # tokenize=False 表示此处仍然返回字符串，真正的分词稍后由 LLMEngine.add_request 完成。
    # add_generation_prompt=True 会在末尾添加“现在轮到 assistant 回答”的特殊标记。
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]

    # generate 会一直循环，直到两个请求都遇到 EOS（结束 token）或达到 max_tokens。
    # 返回列表和 prompts 顺序一致，每项都是 {"text": ..., "token_ids": ...}。
    outputs = llm.generate(prompts, sampling_params)

    # zip 会把两个等长列表按相同下标配对：
    # 第一个 prompt 对应第一个 output，第二个 prompt 对应第二个 output。
    for prompt, output in zip(prompts, outputs):
        print("\n")
        print(f"Prompt: {prompt!r}")
        print(f"Completion: {output['text']!r}")


# 这是 Python 脚本常见的入口保护：
# 直接运行 `python example.py` 时会调用 main()；
# 如果其他文件只是 `import example`，main() 不会自动执行。
if __name__ == "__main__":
    main()
