# 指标化 benchmark：按“对口场景”对比原版与各优化项，产出可回填简历的真实数字。
# 每个 (场景, 变体) 单独起子进程跑：进程退出即彻底释放显存，避免上一个变体的权重/KV Cache
# 残留导致下一个变体分配 KV 块数 <=0；工作负载用固定种子构造，同场景各变体是同一批请求。
#
# 场景：
#   starvation   —— 功能一：短请求先进入 decode，长请求连续 prefill。no_mix（原版两阶段）vs
#                   mix（enable_chunked_prefill：同一步混排 decode + 分块 prefill）。混批消除 decode
#                   饥饿：victim 的 token 间隔从"等完所有长 prompt 的 prefill 步"降到"一步"。
#   prefix       —— 功能二：长共享系统前缀 vs 各自独立前缀，看 Prefix Cache 复用带来的命中率与 TTFT 收益。
#   lru_pressure —— 功能二：多前缀 + 偏斜复用 + 调低显存制造缓存紧张，对比 FIFO vs LRU 的命中率/驱逐。
#
# 需要 CUDA GPU。用法：python bench_metrics.py              # 跑全部场景全部变体
#                     python bench_metrics.py <场景>        # 只跑某场景
#                     python bench_metrics.py <场景> <变体>  # 子进程内跑单个（内部用）
import os
import sys
import subprocess
from random import randint, seed, random

from nanovllm import LLM, SamplingParams

MODEL = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
MAX_MODEL_LEN = 4096


def _sp(max_tokens):
    return SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=max_tokens)


def build_starvation(**_):
    # 短请求先提交：第一步就 prefill 完进入 running；随后长请求连续 prefill，原版两阶段调度期间完全
    # 不 decode → 短请求被饿住。victim 输出=2：其 TPOT 恰好等于“首个→第二个 token”这唯一一个
    # 间隔，即被 prefill 垄断的饥饿时长本身，从而让 TPOT P99 真实反映混批消除饥饿的收益（隔离变量的微基准）。
    seed(0)
    prompts, sps = [], []
    for _ in range(96):                                    # victims（短、先到，输出=2）
        prompts.append([randint(0, 10000) for _ in range(randint(48, 96))])
        sps.append(_sp(2))
    for _ in range(96):                                    # blockers（中长、后到，制造连续慢 prefill）
        prompts.append([randint(0, 10000) for _ in range(randint(1200, 1600))])
        sps.append(_sp(randint(4, 8)))
    return prompts, sps


def build_prefix(share=True, **_):
    # 128 条请求各带一段 1024-token“系统前缀”。share=True 全员共享同一前缀（Prefix Cache
    # 跨请求命中，省重复 prefill）；share=False 各自独立前缀（无复用），作为对照基线。
    seed(0)
    PREFIX_LEN = 1024
    common = [randint(0, 10000) for _ in range(PREFIX_LEN)]
    prompts, sps = [], []
    for _ in range(128):
        prefix = common if share else [randint(0, 10000) for _ in range(PREFIX_LEN)]
        prompts.append(prefix + [randint(0, 10000) for _ in range(randint(32, 96))])
        sps.append(_sp(32))
    return prompts, sps


def build_lru_pressure(**_):
    # 128 个不同前缀（各 512 token = 2 块），请求按幂律偏斜复用（random()**2 偏向小 index →
    # 热点前缀）。配合调低 gpu_memory_utilization 制造缓存紧张，考察 FIFO/LRU 谁能保住热点块。
    seed(0)
    NUM_PREFIXES, PREFIX_LEN = 128, 512
    prefixes = [[randint(0, 10000) for _ in range(PREFIX_LEN)] for _ in range(NUM_PREFIXES)]
    prompts, sps = [], []
    for _ in range(384):
        idx = min(int(NUM_PREFIXES * (random() ** 2)), NUM_PREFIXES - 1)
        prompts.append(prefixes[idx] + [randint(0, 10000) for _ in range(randint(32, 96))])
        sps.append(_sp(16))
    return prompts, sps


def build_cache_aware(**_):
    # 功能三：64 个不同系统前缀（各 1024 token），512 条请求按 round-robin 交错到达（相邻请求命中不同前缀）。
    # 机制：FIFO 命中率低的充要条件 = 同一前缀两次命中之间穿过的不同前缀数 ≥ 缓存能容纳的前缀组数。
    #   round-robin 下该间距 = 前缀总数 N=64，故只要缓存装不下 64 组前缀（配 gpu_memory_utilization=0.15
    #   压小缓存），FIFO 每次回到某前缀时它已被其余 63 个挤出 → 反复重算（近乎 0 复用、evictions 高）；
    #   LPM 优先服务"当前已缓存前缀"的请求，把同前缀请求在时间上聚拢（窗口 W=128 覆盖约 2 个 round-robin
    #   轮次）→ 在被驱逐前把同前缀请求成批排完、复用前缀块。
    # 两变体请求/种子/显存/enable_lru 全同，收益只来自出队顺序，不靠调容量硬造。
    # 判据：若这轮 fifo 的 evictions 仍为 0（说明缓存仍装得下），把 gpu_memory_utilization 再往下压。
    seed(0)
    NUM_PREFIXES, PREFIX_LEN = 64, 1024
    prefixes = [[randint(0, 10000) for _ in range(PREFIX_LEN)] for _ in range(NUM_PREFIXES)]
    prompts, sps = [], []
    for i in range(512):
        idx = i % NUM_PREFIXES                             # round-robin 交错，最大化 FIFO 抖动
        prompts.append(prefixes[idx] + [randint(0, 10000) for _ in range(randint(32, 96))])
        sps.append(_sp(16))
    return prompts, sps


SCENARIOS = {
    "starvation": {
        "build": build_starvation,
        "variants": {
            "no_mix": {"cfg": dict(enable_lru=True, enable_chunked_prefill=False)},
            "mix": {"cfg": dict(enable_lru=True, enable_chunked_prefill=True)},
        },
    },
    "prefix": {
        "build": build_prefix,
        "variants": {
            "no_reuse": {"cfg": dict(enable_lru=True), "wl": dict(share=False)},
            "shared": {"cfg": dict(enable_lru=True), "wl": dict(share=True)},
        },
    },
    "lru_pressure": {
        "build": build_lru_pressure,
        "variants": {
            "fifo": {"cfg": dict(enable_lru=False, gpu_memory_utilization=0.4)},
            "lru": {"cfg": dict(enable_lru=True, gpu_memory_utilization=0.4)},
        },
    },
    "cache_aware": {
        "build": build_cache_aware,
        "variants": {
            "fifo": {"cfg": dict(enable_lru=True, gpu_memory_utilization=0.15,
                                 enable_cache_aware_schedule=False)},
            "lpm": {"cfg": dict(enable_lru=True, gpu_memory_utilization=0.15,
                                enable_cache_aware_schedule=True)},
        },
    },
}


def fmt_bytes(n):
    return f"{n / (1024 ** 3):.2f} GiB"


def run_variant(scenario, variant):
    sc = SCENARIOS[scenario]
    v = sc["variants"][variant]
    prompts, sps = sc["build"](**v.get("wl", {}))
    llm = LLM(MODEL, enforce_eager=False, max_model_len=MAX_MODEL_LEN, **v.get("cfg", {}))
    llm.generate(["warmup"], _sp(1), use_tqdm=False)       # 预热：排除首次内核编译/CUDA Graph 捕获
    r = llm.benchmark(prompts, sps, use_tqdm=False)
    # 不显式 llm.exit()：子进程退出时 LLMEngine 注册的 atexit 会自动清理一次。
    sched, pref = r["scheduler_stats"], r["prefix_cache_stats"]
    print(f"\n===== {scenario} / {variant} ({v.get('cfg', {})}) =====")
    print(f"requests={r['num_requests']}  wall={r['wall_time_s']:.2f}s")
    print(f"TTFT  P50={r['ttft_p50_s']*1000:.1f}ms  P99={r['ttft_p99_s']*1000:.1f}ms")
    print(f"TPOT  P50={r['tpot_p50_s']*1000:.2f}ms  P99={r['tpot_p99_s']*1000:.2f}ms")
    print(f"throughput  prefill={r['prefill_throughput_tok_s']:.0f}tok/s  decode={r['decode_throughput_tok_s']:.0f}tok/s")
    print(f"peak_memory={fmt_bytes(r['peak_memory_bytes'])}")
    print(f"scheduler  prefill_steps={sched.num_prefill_steps}  decode_steps={sched.num_decode_steps}  "
          f"preemptions={sched.num_preemptions}")
    print(f"prefix_cache  hit_rate={pref.hit_rate*100:.1f}%  hits={pref.num_hits}/{pref.num_queries}  "
          f"saved_tokens={pref.saved_tokens}  evictions={pref.num_evictions}")


def main():
    if len(sys.argv) >= 3:                                 # 子进程内：跑单个 (场景, 变体)
        run_variant(sys.argv[1], sys.argv[2])
        return
    only = sys.argv[1] if len(sys.argv) == 2 else None
    for scenario, sc in SCENARIOS.items():
        if only and scenario != only:
            continue
        for variant in sc["variants"]:
            rc = subprocess.run([sys.executable, __file__, scenario, variant]).returncode
            if rc != 0:
                print(f"\n!!! {scenario}/{variant} 退出码 {rc}（可能显存不足，见上方 traceback），继续下一个。")


if __name__ == "__main__":
    main()
