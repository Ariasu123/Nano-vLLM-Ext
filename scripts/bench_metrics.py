# 每个 (场景, 变体) 单独起子进程跑：进程退出即彻底释放显存，避免上一个变体的权重/KV Cache
# 残留导致下一个变体分配 KV 块数 <=0；工作负载用固定种子构造，同场景各变体是同一批请求。
#
# 场景：
#   starvation   —— 功能一：短请求先进入 decode，长请求连续 prefill。no_mix（原版两阶段）vs
#                   mix（enable_chunked_prefill：同一步混排 decode + 分块 prefill）。混批消除 decode
#                   饥饿：victim 的 token 间隔从"等完所有长 prompt 的 prefill 步"降到"一步"。
#   prefix       —— 功能二：长共享系统前缀 vs 各自独立前缀，看 Prefix Cache 复用带来的命中率与 TTFT 收益。
#   lru_pressure —— 功能二：多前缀 + 偏斜复用 + 调低显存制造缓存紧张，对比 FIFO vs LRU 的命中率/驱逐。
#   cache_aware  —— 功能三：多前缀 round-robin 交错到达 + 压小缓存，对比 FIFO vs LPM 出队顺序的命中率/驱逐。
#   speculative  —— 功能四：同一 target 模型下 base 关投机 vs spec 开投机，对比 decode 吞吐/TPOT 与接受率。
#
# 需要 CUDA GPU。用法：python scripts/bench_metrics.py              # 跑全部场景全部变体
#                     python scripts/bench_metrics.py <场景>        # 只跑某场景
#                     python scripts/bench_metrics.py <场景> <变体>  # 子进程内跑单个（内部用）
import os
import sys
import subprocess
from random import randint, seed, random

from nanovllm import LLM, SamplingParams


def _model_root():
    # 模型根目录：AutoDL 系统盘（含 $HOME）小，Qwen3-8B ≈16GB 应落数据盘 /root/autodl-tmp；
    # 非 AutoDL 环境回退 ~/huggingface。可用 MODEL_ROOT 覆盖，与 scripts/env.sh 保持一致。
    root = os.environ.get("MODEL_ROOT")
    if not root:
        root = "/root/autodl-tmp/models" if os.path.isdir("/root/autodl-tmp") else os.path.expanduser("~/huggingface")
    return root


MODEL = os.path.join(_model_root(), "Qwen3-0.6B")
MAX_MODEL_LEN = 4096

# 功能四：投机场景要一大一小两个模型（共享同一 tokenizer/vocab）。可用环境变量覆盖路径。
# target 用较大模型放大 decode 单步成本，draft 用小模型 propose，收益才明显。
SPEC_TARGET = os.path.expanduser(os.environ.get("SPEC_TARGET", os.path.join(_model_root(), "Qwen3-8B")))
SPEC_DRAFT = os.path.expanduser(os.environ.get("SPEC_DRAFT", os.path.join(_model_root(), "Qwen3-0.6B")))


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


def build_speculative(**_):
    # 功能四：中等长度 prompt + 较长输出（每条 128 token），让 decode 阶段占主导——投机解码的收益
    # 全在 decode（一步 target forward 提交多 token）。两变体同批请求/同种子，唯一差别是是否开投机。
    seed(0)
    prompts, sps = [], []
    for _ in range(64):
        prompts.append([randint(0, 10000) for _ in range(randint(64, 128))])
        sps.append(_sp(128))
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
    "speculative": {
        # 功能四：同一 target 模型（SPEC_TARGET），base 关投机 vs spec 开投机（draft=SPEC_DRAFT，K=4）。
        # 对照 decode 吞吐 / TPOT，并看 spec 的 acceptance_rate / avg_accept_len。
        "build": build_speculative,
        # A(base)=CUDA Graph baseline，B(base_eager)=eager baseline，C(spec)=eager 投机。
        # A vs C 回答「当前完整 serving 配置下开投机是否值得」（混入 CUDA Graph 开/关变量）；
        # B vs C 才隔离掉该变量、单独回答「投机算法本身有没有收益」——spec path 本就走 eager，只有
        # 和 eager baseline 比才公平。三者同批请求/同种子/同 target 模型。
        "variants": {
            "base": {"model": SPEC_TARGET, "cfg": dict(enable_lru=True)},
            "base_eager": {"model": SPEC_TARGET, "enforce_eager": True, "cfg": dict(enable_lru=True)},
            "spec": {"model": SPEC_TARGET, "cfg": dict(enable_lru=True,
                                                       enable_speculative_decode=True,
                                                       speculative_model=SPEC_DRAFT,
                                                       num_speculative_tokens=int(os.environ.get("SPEC_K", "4")))},
        },
    },
}


def fmt_bytes(n):
    return f"{n / (1024 ** 3):.2f} GiB"


def run_variant(scenario, variant):
    sc = SCENARIOS[scenario]
    v = sc["variants"][variant]
    prompts, sps = sc["build"](**v.get("wl", {}))
    model = v.get("model", MODEL)                          # speculative 场景用较大的 target 模型
    # enforce_eager 默认 False（走 CUDA Graph decode）。speculative 场景额外提供 base_eager 变体
    # 关掉 Graph，做「eager-vs-eager」苹果对苹果对照：spec path 本就走 eager，只有和 eager baseline
    # 比才隔离掉「CUDA Graph 开/关」这个巨大变量，单独衡量投机算法本身的收益。
    llm = LLM(model, enforce_eager=v.get("enforce_eager", False), max_model_len=MAX_MODEL_LEN, **v.get("cfg", {}))
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
    if sched.num_speculative_steps:
        # 原始量（可自行核对）：verification_steps=record_acceptance 调用数（每序列每 spec step 一次
        # target forward）；proposed/accepted/emitted 为累计 token 数。派生：
        #   acceptance_rate       = accepted / proposed        （被接受的 draft token 占比）
        #   avg_accept_len        = accepted / verification     （每次真正接受的 draft token 数，0..K）
        #   tokens_per_target_step= emitted  / verification     （每次 target forward 推进 token 数，含 bonus，1..K+1）
        print(f"speculative  spec_steps={sched.num_speculative_steps}  verification_steps={sched.num_acceptance_records}  "
              f"acceptance_rate={sched.acceptance_rate*100:.1f}%  avg_accept_len={sched.avg_accept_len:.2f}  "
              f"tokens_per_target_step={sched.tokens_per_target_step:.2f}  bonus={sched.num_bonus}")
        print(f"speculative  raw: proposed={sched.total_proposed_tokens}  "
              f"accepted={sched.total_accepted_tokens}  emitted={sched.total_emitted_tokens}")
        # 诊断：spec batch 装不满的限流点（各闸门累计拒绝数）+ 隐藏 fallback 步数。
        # reject.draft 远大于其它 ⟹ draft KV 池是限流点；fallback_decode_steps>0 ⟹ 存在 target 前进/draft 冻结的降级。
        print(f"speculative  reject: draft={sched.spec_reject_draft}  target={sched.spec_reject_target}  "
              f"budget={sched.spec_reject_budget}  fallback_decode_steps={sched.spec_fallback_decode_steps}")


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
