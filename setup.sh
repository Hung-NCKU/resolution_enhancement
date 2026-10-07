#!/usr/bin/env bash
# 在 WSL (Ubuntu) 內安裝執行環境。用法：bash setup.sh
set -euo pipefail
VENV="$HOME/.venvs/resolution"

if ! command -v ffmpeg >/dev/null; then
  echo "安裝 ffmpeg（需要 sudo 密碼）"
  sudo apt-get update -qq && sudo apt-get install -y -qq ffmpeg
fi

if [ ! -x "$HOME/.local/bin/uv" ]; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
UV="$HOME/.local/bin/uv"

[ -d "$VENV" ] || "$UV" venv "$VENV" --python 3.12
source "$VENV/bin/activate"
# RTX 50 系列 (Blackwell, sm_120) 需要 CUDA 12.8 版 PyTorch
"$UV" pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
"$UV" pip install -r "$(dirname "$0")/requirements.txt"

# SeedVR2 影片修復（ByteDance Seed，一步擴散）— 放在 Linux 檔案系統以加快模型載入
SEED="$HOME/seedvr2"
SEED_COMMIT=4490bd1f482e026674543386bb2a4d176da245b9   # v2.5.24，本專案測試過的版本
if [ ! -d "$SEED" ]; then
  git clone https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler.git "$SEED"
  git -C "$SEED" checkout -q "$SEED_COMMIT"
fi
"$UV" pip install safetensors psutil einops "omegaconf>=2.3.0" "diffusers>=0.33.1" "peft>=0.17.0" \
  "rotary_embedding_torch>=0.5.3" gguf matplotlib
mkdir -p "$SEED/models/SEEDVR2"
( cd "$(dirname "$0")" && python - <<'P'
import engine, video_seedvr2 as s
for f in ["ema_vae_fp16.safetensors", s.DIT_MODELS["3b-fp8"][0]]:
    info = engine.ModelInfo(f, f, f"https://huggingface.co/numz/SeedVR2_comfyUI/resolve/main/{f}", "", "aux")
    engine.download(info, dest_dir=s.SEEDVR2_MODELS)
P
)

python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
echo "安裝完成。執行 ./run.sh 或在 Windows 點兩下 start.bat"
