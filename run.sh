#!/usr/bin/env bash
# 啟動網頁介面（WSL 內）。有參數時改為執行命令列：./run.sh -i 圖片 -m photo-best
cd "$(dirname "$0")"
source "$HOME/.venvs/resolution/bin/activate"
export PYTHONWARNINGS=ignore
if [ $# -gt 0 ]; then
  exec python restore.py "$@"
fi
exec python app.py
