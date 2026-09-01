#!/usr/bin/env bash
# 无卡模式阶段：装依赖 + 下模型 + 跑纯逻辑单测。全程不需要 GPU，省 GPU 计费。
# 用法：在仓库根执行 bash scripts/setup.sh
set -euo pipefail

# 脚本在 scripts/ 下，切到仓库根：pip install -e . 与 pytest tests/ 都需以仓库根为工作目录。
cd "$(dirname "$0")/.."
source "$(dirname "$0")/env.sh"                        # 统一模型根目录（AutoDL 优先数据盘，见 env.sh）

MODEL_DIR="$DRAFT_DIR"

echo "==> [1/4] 安装依赖（transformers/xxhash/pytest 及模型下载工具）"
pip install -U transformers xxhash pytest "huggingface_hub[cli]"
# editable 安装本包：脚本已收进 scripts/，装成 editable 后从任意目录都能 import nanovllm。
pip install -e .

echo "==> [2/4] 安装 flash-attn（走镜像下官方预编译 wheel，不走源码编译）"

if python -c "import flash_attn" 2>/dev/null; then
  echo "    flash_attn 已安装，跳过。"
else
  # wheel 名里的 torch 主次版本、python tag、C++ ABI 全部按当前环境自动探测，换机不用改。
  FA_ABI=$(python -c "import torch; print('TRUE' if torch.compiled_with_cxx11_abi() else 'FALSE')")
  FA_TORCH=$(python -c "import torch; print('.'.join(torch.__version__.split('+')[0].split('.')[:2]))")
  FA_PY=$(python -c "import sys; print(f'cp{sys.version_info.major}{sys.version_info.minor}')")
  FA_WHL="flash_attn-2.8.3+cu12torch${FA_TORCH}cxx11abi${FA_ABI}-${FA_PY}-${FA_PY}-linux_x86_64.whl"
  echo "    torch=${FA_TORCH} py=${FA_PY} abi=${FA_ABI} → 下载 ${FA_WHL}"
  wget -O "$FA_WHL" "https://ghfast.top/https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/${FA_WHL}"
  pip install "./$FA_WHL" --no-build-isolation
fi
python -c "import flash_attn; print('    flash_attn OK', flash_attn.__version__)"

echo "==> [3/4] 下载模型到 $MODEL_DIR（走 hf-mirror 国内镜像）"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
# 新版 hf 默认用 Xet 后端（cas-server.xethub.hf.co），不认 hf-mirror 镜像会 401，
# 关掉它回退普通 HTTPS 从镜像下。
export HF_HUB_DISABLE_XET=1
if [ -f "$MODEL_DIR/config.json" ]; then
  echo "    已存在，跳过下载。"
elif command -v hf >/dev/null 2>&1; then
  hf download Qwen/Qwen3-0.6B --local-dir "$MODEL_DIR"      # 新版 huggingface_hub 命令
else
  huggingface-cli download Qwen/Qwen3-0.6B --local-dir "$MODEL_DIR"
fi

echo "==> [4/4] 跑纯逻辑单测（三项功能的调度/缓存/指标逻辑，不吃 GPU）"
python -m pytest tests/ -q

echo
echo "==> 无卡阶段完成。三份单测通过后，切到【有卡模式】执行：bash scripts/run_gpu.sh"
