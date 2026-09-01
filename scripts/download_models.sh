#!/usr/bin/env bash
# 在服务器上下载投机解码需要的一大一小两个 Qwen3 模型到数据盘（AutoDL 优先 /root/autodl-tmp）。
# target 与 draft 必须同属 Qwen3 系列（共享同一 tokenizer/vocab），否则 rejection sampling 无法对齐。
#
# 用法：
#   bash download_models.sh                       # 默认下 Qwen3-8B(target) + Qwen3-0.6B(draft)
#   TARGET_REPO=Qwen/Qwen3-4B bash download_models.sh   # 换更省显存的 4B 作 target
#   DEST=/data/hf bash download_models.sh         # 换下载目录
#
# 显存参考（bf16 权重）：Qwen3-8B ≈16GB、Qwen3-4B ≈8GB、Qwen3-0.6B ≈1.2GB，另需 KV Cache 余量。
set -euo pipefail
source "$(dirname "$0")/env.sh"                 # 统一模型根目录（AutoDL 优先数据盘，见 env.sh）

DEST="${DEST:-$MODEL_ROOT}"
TARGET_REPO="${TARGET_REPO:-Qwen/Qwen3-8B}"     # 注意：Qwen3 无 7B，7B 级对应 8B
DRAFT_REPO="${DRAFT_REPO:-Qwen/Qwen3-0.6B}"

# 默认走 hf-mirror 国内镜像（与 setup.sh 一致）；关掉 Xet 后端避免不认镜像时 401。均可用环境变量覆盖。
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET=1

mkdir -p "$DEST"

# huggingface_hub 的 CLI：新版命令是 hf，旧版是 huggingface-cli，二者都支持 download。
if command -v hf >/dev/null 2>&1; then
  HF="hf"
elif command -v huggingface-cli >/dev/null 2>&1; then
  HF="huggingface-cli"
else
  echo "未找到 hf / huggingface-cli，先安装：pip install -U 'huggingface_hub[cli]'"
  exit 1
fi

# hf_transfer 可显著加速大文件下载（可选，装了就开）。
if python -c "import hf_transfer" >/dev/null 2>&1; then
  export HF_HUB_ENABLE_HF_TRANSFER=1
fi

_pull() {   # $1=repo，落到 $DEST/<repo 末段>，跳过 .pth 等非必要格式
  local repo="$1" dir="$DEST/$(basename "$1")"
  echo "==> 下载 $repo -> $dir"
  # 每个 pattern 必须各带一个 --exclude：新版 hf CLI 下 `--exclude a b c` 只把 a 当排除模式，
  # b、c 会被当成"指定要下载的文件名"（positional），触发 "Ignoring --exclude since filenames
  # have been explicitly set" → 只找这些字面文件名 → 下 0 个文件。Qwen3 本无这些文件，排除只是保险。
  "$HF" download "$repo" --local-dir "$dir" \
    --exclude "original/*" --exclude "*.pth" --exclude "consolidated*.safetensors"
  [ -f "$dir/config.json" ] || { echo "!! $dir 缺 config.json，下载可能失败"; exit 1; }
}

_pull "$TARGET_REPO"
_pull "$DRAFT_REPO"

echo
echo "==> 完成。设置环境变量后即可跑 run_gpu.sh 的投机步骤："
echo "    export SPEC_TARGET=$DEST/$(basename "$TARGET_REPO")"
echo "    export SPEC_DRAFT=$DEST/$(basename "$DRAFT_REPO")"
