#!/usr/bin/env bash
# 選用：編譯 SageAttention 2 / 3 與 FlashAttention 2（只針對 RTX 50 系列 sm_120）
# 需要 sudo（安裝 CUDA 12.8 編譯工具與 Python 標頭檔）。FlashAttention 約需 40 分鐘。
set -euo pipefail

if [ ! -x /usr/local/cuda-12.8/bin/nvcc ]; then
  cd /tmp
  wget -q https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
  sudo dpkg -i cuda-keyring_1.1-1_all.deb
  sudo apt-get update -qq
fi
sudo apt-get install -y -qq cuda-toolkit-12-8 build-essential ninja-build python3.12-dev

export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export TORCH_CUDA_ARCH_LIST="12.0"
export MAX_JOBS=4          # 16GB RAM 下同時編譯太多會記憶體不足
export NVCC_THREADS=2
export LIBRARY_PATH=/usr/lib/wsl/lib:${LIBRARY_PATH:-}   # WSL 的 libcuda.so 在這裡（SageAttention 3 需要）
source ~/.venvs/resolution/bin/activate
UV=~/.local/bin/uv
$UV pip install setuptools wheel packaging ninja

[ -d ~/SageAttention ] || git clone --depth 1 https://github.com/thu-ml/SageAttention.git ~/SageAttention
(cd ~/SageAttention && EXT_PARALLEL=2 $UV pip install --no-build-isolation .)
(cd ~/SageAttention/sageattention3_blackwell && $UV pip install --no-build-isolation .)
FLASH_ATTN_CUDA_ARCHS=120 FLASH_ATTENTION_FORCE_BUILD=TRUE $UV pip install --no-build-isolation flash-attn

python -c "import sageattention, sageattn3, flash_attn; print('OK', flash_attn.__version__)"
