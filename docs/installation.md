# 安裝教學（從零開始）

本文假設你有一台 **Windows 11 + NVIDIA 顯示卡** 的電腦，從完全沒有 WSL 開始。全部完成約需 30~60 分鐘（大部分是下載時間），若要編譯加速套件再加約 45 分鐘。

- [1. 安裝 NVIDIA 驅動（Windows 端）](#1-安裝-nvidia-驅動windows-端)
- [2. 安裝 WSL2 與 Ubuntu 24.04](#2-安裝-wsl2-與-ubuntu-2404)
- [3. 確認 WSL 看得到 GPU](#3-確認-wsl-看得到-gpu)
- [4. 下載本專案](#4-下載本專案)
- [5. 執行 setup.sh](#5-執行-setupsh)
- [6.（選用）編譯加速套件](#6選用編譯加速套件)
- [7. 啟動](#7-啟動)
- [8. 疑難排解](#8-疑難排解)

---

## 1. 安裝 NVIDIA 驅動（Windows 端）

到 [NVIDIA 驅動下載頁](https://www.nvidia.com/drivers/) 安裝最新的 Game Ready 或 Studio 驅動。

> **重點：只在 Windows 安裝驅動，不要在 WSL 的 Ubuntu 裡安裝任何 NVIDIA 驅動。**
> WSL2 會自動把 Windows 的驅動提供給 Linux 使用（`/usr/lib/wsl/lib/libcuda.so`）。在 Ubuntu 裡另外裝驅動會把它覆蓋掉，導致 GPU 無法使用。

RTX 50 系列（Blackwell）需要 570 版以上的驅動；本專案測試時使用 617.14。

## 2. 安裝 WSL2 與 Ubuntu 24.04

以**系統管理員身分**開啟 PowerShell：

```powershell
wsl --install -d Ubuntu-24.04
```

完成後重新開機，第一次開啟 Ubuntu 會要求建立 Linux 使用者名稱與密碼（這個密碼之後 `sudo` 會用到）。

確認版本是 WSL 2：

```powershell
wsl -l -v
#   NAME            STATE    VERSION
# * Ubuntu-24.04    Running  2
```

> 本專案的 `start.bat` 寫死使用名為 `Ubuntu-24.04` 的發行版。若你的名稱不同，請修改 `start.bat` 最後一行的 `-d` 參數。

## 3. 確認 WSL 看得到 GPU

在 Ubuntu 終端機：

```bash
nvidia-smi
```

應該看到你的顯示卡名稱與 VRAM。若出現 `command not found` 或錯誤，請回到步驟 1 更新 Windows 驅動並重新開機。

## 4. 下載本專案

```bash
sudo apt-get update && sudo apt-get install -y git
cd /mnt/d                      # Windows 的 D 槽；也可以放 /mnt/c/Users/你的名字/...
git clone https://github.com/Hung-NCKU/resolution_enhancement.git
cd resolution_enhancement
```

**為什麼放在 Windows 磁碟（/mnt/d）？** 方便用 Windows 檔案總管存取輸出檔，也能點兩下 `start.bat` 啟動。
效能上的代價是 WSL 讀取 Windows 磁碟較慢，所以 `setup.sh` 會把**最大的 SeedVR2 模型（約 3.7 GB）放到 Linux 檔案系統** `~/seedvr2/models`。實測從 `/mnt/d` 載入 3.2 GB 權重要 30 秒以上，放在 Linux 端只要幾秒（見 [experiments.md](experiments.md#實驗-4模型放置位置)）。

## 5. 執行 setup.sh

```bash
bash setup.sh
```

這個腳本會依序：

| 步驟 | 內容 | 大小 / 時間 |
|---|---|---|
| ffmpeg | 用 apt 安裝（會要求 sudo 密碼） | 小 |
| uv | 安裝 Python 套件管理工具 [uv](https://docs.astral.sh/uv/)（比 pip 快很多） | 小 |
| 虛擬環境 | 建立 `~/.venvs/resolution`（Python 3.12） | |
| PyTorch | 安裝 **CUDA 12.8 版**（RTX 50 系列必須 12.8 以上） | 約 3 GB |
| 相依套件 | spandrel、Gradio、facexlib、OpenCV… | 約 500 MB |
| SeedVR2 | clone 並固定在測試過的版本 v2.5.24 到 `~/seedvr2` | |
| SeedVR2 模型 | 3B FP8 DiT（3.2 GB）＋ VAE（0.5 GB） | 約 3.7 GB |

最後會印出 `CUDA: True NVIDIA GeForce ...` 表示成功。

**圖片模型**（約 1.8 GB）會在第一次使用時自動下載到 `models/`，也可以先一次下載：

```bash
./run.sh --download
```

> **下載很慢？** GitHub 的 release 檔案單一連線常只有數百 KB/s。本專案的下載器會用 8 條連線分段下載（見 `engine.py` 的 `download()`），實測快 5~10 倍。

## 6.（選用）編譯加速套件

SeedVR2 啟動時會顯示：

```
SeedVR2 optimizations check: SageAttention ❌ | Flash Attention ❌ | Triton ✅
```

❌ 只代表沒安裝選用的加速套件，**不影響功能與畫質**，會自動改用 PyTorch 內建的 SDPA。

想要再快約 10%，可以執行：

```bash
bash build_attention.sh
```

這個腳本會：

1. 安裝 CUDA 12.8 編譯工具（`nvcc`，使用 NVIDIA 的 WSL 專用套件庫，**不會動到驅動**）與 `python3.12-dev`（提供編譯需要的 `Python.h`）
2. 編譯 SageAttention 2、SageAttention 3（RTX 50 專用）、FlashAttention 2，**只針對 sm_120 架構**以縮短時間

實測編譯時間：SageAttention 2 約 4 分鐘、SageAttention 3 約 1 分鐘、FlashAttention 約 40 分鐘（12 核心、`MAX_JOBS=4`）。

完成後再啟動 SeedVR2 應顯示三個 ✅。程式會自動使用 FlashAttention（實測最快且結果與 SDPA 幾乎相同）；SageAttention 為 8/4-bit 量化，誤差較大且沒有更快，所以保留但不預設使用。比較數據見 [experiments.md](experiments.md#實驗-6注意力後端)。

> **不是 RTX 50 系列？** 修改 `build_attention.sh` 中的 `TORCH_CUDA_ARCH_LIST`（RTX 30 = `8.6`、RTX 40 = `8.9`）與 `FLASH_ATTN_CUDA_ARCHS`，並刪掉 SageAttention 3 那一行（它只支援 Blackwell）。

## 7. 啟動

**Windows**：點兩下專案資料夾中的 `start.bat`，瀏覽器會自動開啟 <http://localhost:7860>。

**WSL**：

```bash
./run.sh            # 網頁介面
./run.sh --help     # 命令列說明
```

停止：在終端機按 `Ctrl+C`；網頁上的「停止」按鈕只會停止目前的影片任務。

## 8. 疑難排解

| 症狀 | 原因與解法 |
|---|---|
| `CUDA: False` | WSL 看不到 GPU。更新 Windows NVIDIA 驅動、重新開機；確認沒有在 Ubuntu 內安裝 NVIDIA 驅動。 |
| 執行 `.sh` 出現 `$'\r': command not found` | 檔案被轉成 Windows 換行（CRLF）。本專案的 `.gitattributes` 已強制 LF；若仍發生：`sed -i 's/\r$//' *.sh` |
| **速度突然慢 10 倍以上、GPU 使用率 100% 但功耗很低** | **VRAM 溢出到系統記憶體**。WSL 不會報錯，而是默默變慢。降低輸出解析度或每批影格數，關閉其他佔用顯示卡的程式（動態桌布、瀏覽器影片、遊戲平台）。詳見 [experiments.md](experiments.md#實驗-1wsl-的-vram-溢出)。 |
| 想讓 VRAM 不足時直接報錯 | NVIDIA 控制台 →「管理 3D 設定」→「CUDA - 系統記憶體備援原則」→「偏好無系統記憶體備援」 |
| `INTERNAL ASSERT FAILED at CUDACachingAllocator.cpp` | 設了 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。這在 WSL 不支援，請移除該環境變數。 |
| 編譯加速套件出現 `Python.h: No such file or directory` | `sudo apt install python3.12-dev` |
| SageAttention 3 連結失敗、提示 `libcuda` | WSL 的 `libcuda.so` 在 `/usr/lib/wsl/lib`，需 `export LIBRARY_PATH=/usr/lib/wsl/lib`（`build_attention.sh` 已處理） |
| 網頁打不開 | 確認終端機有出現 `Running on local URL: http://0.0.0.0:7860`；WSL2 預設會把 localhost 轉發到 Windows。若埠號衝突可用 `PORT=7861 ./run.sh`。 |
| 輸出影片沒有聲音 | 原影片沒有音軌，或音訊格式無法放入 MP4。SeedVR2 輸出後會用 ffmpeg 把原音軌轉成 AAC 合併回去。 |
| 預估時間不準 | 預估使用本機實測速度，第一次會用預設值；每次完成後自動校正（存在 `outputs/.speed.json`，刪除即重設）。 |
