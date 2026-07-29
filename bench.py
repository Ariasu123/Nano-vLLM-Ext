# 【这个文件做什么】
# 这是吞吐量 benchmark（基准测试），用于测量 Nano-vLLM 每秒能生成多少 token。
# 它直接构造随机 token id，避免分词器耗时影响 GPU 推理结果。
import os
import time
from random import randint, seed
from nanovllm import LLM, SamplingParams
# 若要与官方 vLLM 比较，可切换为：from vllm import LLM, SamplingParams


def main():
    # 固定随机种子后，每次运行产生相同随机长度，结果更容易比较。
    seed(0)

    # 共生成 256 条请求，每条输入/输出长度都在指定范围随机选择。
    num_seqs = 256
    max_input_len = 1024
    max_ouput_len = 1024

    path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")

    # enforce_eager=False 允许 Decode 使用 CUDA Graph，测量的是优化后性能。
    llm = LLM(path, enforce_eager=False, max_model_len=4096)

    # 嵌套列表推导式为每条请求创建不同长度的 token id 列表。
    prompt_token_ids = [[randint(0, 10000) for _ in range(randint(100, max_input_len))] for _ in range(num_seqs)]

    # 每条请求也有独立的随机生成长度。
    # ignore_eos=True 保证一定生成到 max_tokens，便于准确计算总输出 token。
    sampling_params = [SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=randint(100, max_ouput_len)) for _ in range(num_seqs)]

    # 如果改用官方 vLLM，需要按它的接口包装 token id：
    # prompt_token_ids = [dict(prompt_token_ids=p) for p in prompt_token_ids]

    # 先用一个小请求预热，避免首次编译和图捕获时间计入正式测量。
    llm.generate(["Benchmark: "], SamplingParams())

    # 记录开始时间，关闭进度条后运行全部请求。
    t = time.time()
    llm.generate(prompt_token_ids, sampling_params, use_tqdm=False)
    t = (time.time() - t)
    # 生成器表达式逐个取 max_tokens，再由 sum 求总输出 token 数。
    total_tokens = sum(sp.max_tokens for sp in sampling_params)

    # 吞吐量 = 总生成 token 数 / 总秒数。
    throughput = total_tokens / t
    print(f"Total: {total_tokens}tok, Time: {t:.2f}s, Throughput: {throughput:.2f}tok/s")


if __name__ == "__main__":
    main()
