#!/usr/bin/env bash
# 模型存放根目录的统一定义，供 setup.sh / run_gpu.sh / download_models.sh 共享 source。
#
# 为什么优先落数据盘：AutoDL 实例的系统盘（/root，含 $HOME）通常仅 ~30GB 且随实例释放，
# 而 Qwen3-8B(bf16) 权重就 ≈16GB，直接下到 $HOME 会撑爆系统盘；数据盘 /root/autodl-tmp
# 才是大容量持久盘。故检测到 AutoDL 数据盘时落 /root/autodl-tmp/models，否则回退 ~/huggingface。
# 三者均可用 MODEL_ROOT / TARGET_DIR / DRAFT_DIR 环境变量覆盖，以适配非 AutoDL 环境。
if [ -z "${MODEL_ROOT:-}" ]; then
  if [ -d /root/autodl-tmp ]; then
    MODEL_ROOT=/root/autodl-tmp/models
  else
    MODEL_ROOT="$HOME/huggingface"
  fi
fi
export MODEL_ROOT
export TARGET_DIR="${TARGET_DIR:-$MODEL_ROOT/Qwen3-8B}"     # 投机解码的 target（大模型）
export DRAFT_DIR="${DRAFT_DIR:-$MODEL_ROOT/Qwen3-0.6B}"     # 投机解码的 draft（小模型），也是单模型场景默认模型
