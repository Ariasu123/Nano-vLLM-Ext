import os
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer


def main():
    # 模型路径优先读 MODEL_DIR 环境变量（run_gpu.sh 会按 env.sh 的数据盘位置设好），
    # 否则回退 ~/huggingface；expanduser 把开头的 "~" 展开成当前用户的主目录。
    path = os.path.expanduser(os.environ.get("MODEL_DIR", "~/huggingface/Qwen3-0.6B/"))
    tokenizer = AutoTokenizer.from_pretrained(path)

    # 创建推理引擎时会完成较重的初始化工作：
    # 1. 读取模型配置；2. 在 GPU 上创建 Qwen3；3. 加载模型权重；
    # 4. 预热模型；5. 使用剩余显存分配 KV Cache。
    llm = LLM(path, enforce_eager=True, tensor_parallel_size=1)

    sampling_params = SamplingParams(temperature=0.6, max_tokens=256)

    prompts = [
        "introduce yourself",
        "list all prime numbers within 100",
    ]

    # apply_chat_template 会把一条用户消息包装成模型要求的特殊格式。
    # tokenize=False 表示此处仍然返回字符串，真正的分词稍后由 LLMEngine.add_request 完成
    # add_generation_prompt=True 会在末尾添加“现在轮到 assistant 回答”的特殊标记
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]

    outputs = llm.generate(prompts, sampling_params)
    for prompt, output in zip(prompts, outputs):
        print("\n")
        print(f"Prompt: {prompt!r}")
        print(f"Completion: {output['text']!r}")

if __name__ == "__main__":
    main()
