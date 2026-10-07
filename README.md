# AI 影片畫質修復工作站（Resolution Enhancement）

在 **Windows 11 + WSL2 (Ubuntu 24.04)** 上以 NVIDIA GPU 執行的影片 / 圖片畫質修復與超解析度工具，供**教學與研究**使用。

- 影片核心：**SeedVR2**（ByteDance Seed，2025，一步擴散影片修復），保持前後格一致、不閃爍
- 圖片 / 逐格：DAT2、DRCT-L、ATD、RealPLKSR、Real-ESRGAN 等超解析度模型
- 臉部修復：CodeFormer、GFPGAN；前處理：SCUNet 去雜訊、FBCNN 去 JPEG 壓縮痕
- 網頁介面（Gradio）＋ 命令列；開始前預估處理時間、執行中顯示剩餘時間
- 已針對 **8GB VRAM（RTX 5060）** 實測調校，實驗過程完整記錄於 [docs/experiments.md](docs/experiments.md)

> 本專案整合多個開源模型，各模型授權不同（例如 CodeFormer 限非商業使用），請見文末〈[授權](#授權)〉。

---

## 文件導覽

| 文件 | 內容 | 適合對象 |
|---|---|---|
| [docs/installation.md](docs/installation.md) | 從零開始安裝：WSL、驅動、環境、加速套件、疑難排解 | 第一次使用 |
| [docs/principles.md](docs/principles.md) | 超解析度原理：GAN、Transformer、擴散模型、臉部修復、分塊推論、數值精度 | 課程教學 |
| [docs/experiments.md](docs/experiments.md) | 8GB VRAM 調校實驗：VRAM 溢出、批次大小、注意力後端、各模型速度 | 研究 / 調校 |
| [docs/code-walkthrough.md](docs/code-walkthrough.md) | 程式架構導讀：每個檔案在做什麼、關鍵設計 | 想修改程式的人 |

---

## 快速開始

**需求**：Windows 11、NVIDIA 顯示卡（建議 8GB VRAM 以上）、WSL2 Ubuntu 24.04、約 15 GB 硬碟空間。

```bash
# 1. 在 WSL (Ubuntu) 內 clone（建議放在 Windows 磁碟，方便用 start.bat）
cd /mnt/d
git clone https://github.com/Hung-NCKU/resolution_enhancement.git
cd resolution_enhancement

# 2. 安裝（PyTorch、相依套件、ffmpeg、SeedVR2 與其模型；需 sudo 密碼）
bash setup.sh

# 3.（選用）編譯 FlashAttention 等加速套件，約 45 分鐘，SeedVR2 約快 10%
bash build_attention.sh

# 4. 啟動網頁介面 → 瀏覽器開 http://localhost:7860
./run.sh
```

Windows 使用者也可直接**點兩下 `start.bat`**，會自動啟動並開啟瀏覽器。

詳細步驟與常見錯誤見 [docs/installation.md](docs/installation.md)。

---

## 使用方式

### 網頁介面

| 分頁 | 功能 |
|---|---|
| 🎬 影片修復 | 選擇引擎（SeedVR2 / 逐格放大）、輸出解析度、試跑格數；上傳後即顯示**預估時間**與警告 |
| 🖼️ 圖片修復 | 單張（含前後對比滑桿）與批次處理（ZIP 打包下載） |

每個參數下方都有說明文字。建議流程：**上傳 → 看預估時間 → 用「試跑」處理 50~100 格確認效果 → 再跑整部**。

### 命令列

```bash
./run.sh -i 影片.mp4 --res 1080                         # SeedVR2 修復到 1080p（預設引擎）
./run.sh -i 影片.mp4 --res 720 --frames 100             # 只處理前 100 格試跑
./run.sh -i 影片.mp4 --video-engine frame -m photo-fast -s 2   # 逐格快速放大 2 倍
./run.sh -i 照片.jpg -m photo-best --face codeformer     # 圖片 + 臉部修復
./run.sh -i 資料夾/ -o outputs -m photo-fast             # 整個資料夾
./run.sh --list                                         # 列出所有模型
./run.sh --download                                     # 預先下載所有圖片模型（約 1.8 GB）
```

執行前會先印出預估時間，例如 `⏱ 影片.mp4：1,800 格，預估約 3 小時 5 分`。

---

## 模型一覽

### 影片引擎

| 引擎 | 特色 | RTX 5060 8GB 實測 |
|---|---|---|
| **SeedVR2 3B FP8**（預設） | 擴散模型重建細節、前後格一致；適合老影片、模糊、重壓縮 | 720p 約 4 秒/格、1080p 約 6 秒/格 |
| SeedVR2 3B Q8 / 7B Sharp Q4 | 品質更高的變體，首次使用自動下載 | 更慢 |
| 逐格放大 | 每格獨立用圖片模型處理，速度快 | 0.2~3 秒/格（依模型） |

> 例：1 分鐘 30fps 影片 = 1,800 格，SeedVR2 1080p 約 3 小時。輸出超過約 1920×1000 時 8GB VRAM 不足，會再慢約 2.4 倍。

### 圖片 / 逐格模型

| 代號 | 架構 | 用途 |
|---|---|---|
| `photo-best` | DAT2 | 真實照片：模糊、雜訊、JPEG，品質最高 |
| `photo-fast` | RealPLKSR | 同上，快約 8 倍 |
| `web-photo` | ATD | 社群網站轉存、重壓縮 |
| `clean-max` | DRCT-L | 乾淨輸入的極致放大，最慢 |
| `general` / `anime` / `video-fast` | Real-ESRGAN 系列 | 通用 / 動漫 / 輕量 |
| 臉部 `codeformer` / `gfpgan` | CodeFormer / GFPGAN v1.4 | 臉部重建 |
| 前處理 `scunet` / `fbcnn` | SCUNet / FBCNN | 去雜訊 / 去壓縮痕 |

---

## 專案結構

```
├── app.py               # Gradio 網頁介面
├── restore.py           # 命令列介面
├── engine.py            # 圖片 / 逐格引擎：模型管理、分塊推論、臉部修復、影片管線
├── video_seedvr2.py     # SeedVR2 包裝：8GB 調校參數、進度解析、停止、合併音訊
├── eta.py               # 處理時間預估與自動校正
├── setup.sh             # 環境安裝
├── build_attention.sh   # 選用：編譯 FlashAttention / SageAttention
├── run.sh / start.bat   # 啟動（WSL / Windows）
└── docs/                # 詳細教學文件
```

執行後會產生（不納入版本控制）：`models/`（圖片模型）、`outputs/`（輸出）、`~/seedvr2/`（SeedVR2 程式與模型，放在 Linux 檔案系統以加快載入）。

---

## 測試環境

| 項目 | 版本 |
|---|---|
| 作業系統 | Windows 11 + WSL2 Ubuntu 24.04 |
| GPU | NVIDIA GeForce RTX 5060 8GB（Blackwell, sm_120），Windows 驅動 617.14 |
| Python | 3.12.3 |
| PyTorch | 2.11.0+cu128 |
| Gradio / spandrel | 6.29.0 / 0.4.2 |
| SeedVR2 CLI | v2.5.24（commit `4490bd1`） |
| FlashAttention | 2.8.3（自行編譯 sm_120） |

其他 NVIDIA 顯示卡（RTX 20/30/40 系列）理論上可用同一套環境，但未經測試；調校參數（批次大小、分塊）可能需要依 VRAM 調整。

---

## 授權

本專案程式碼僅整合與呼叫下列開源專案，**模型權重與上游程式碼依各自授權**，使用前請自行確認：

| 專案 | 授權 | 備註 |
|---|---|---|
| [SeedVR2](https://github.com/ByteDance-Seed/SeedVR) / [CLI 實作](https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler) | Apache-2.0 | |
| [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) | BSD-3-Clause | |
| [CodeFormer](https://github.com/sczhou/CodeFormer) | S-Lab License 1.0 | **限非商業用途** |
| [GFPGAN](https://github.com/TencentARC/GFPGAN) | Apache-2.0（部分元件另有授權） | 見其 LICENSE |
| [SCUNet](https://github.com/cszn/SCUNet) / [FBCNN](https://github.com/jiaxi-jiang/FBCNN) | Apache-2.0 | |
| [Phhofm 模型](https://github.com/Phhofm/models)（DAT2、ATD、DRCT-L、RealPLKSR） | 依各模型 release 頁面 | |
| [spandrel](https://github.com/chaiNNer-org/spandrel) / [facexlib](https://github.com/xinntao/facexlib) | MIT | |
| [FlashAttention](https://github.com/Dao-AILab/flash-attention) / [SageAttention](https://github.com/thu-ml/SageAttention) | BSD-3-Clause / Apache-2.0 | |

學術引用請見 [docs/principles.md#參考文獻](docs/principles.md#參考文獻)。
