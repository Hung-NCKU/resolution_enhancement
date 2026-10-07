# 原理說明：影像與影片超解析度

本文說明本專案用到的技術原理，適合作為影像處理 / 深度學習課程的補充教材。每一節都標出對應的程式位置，方便對照閱讀。

- [1. 問題定義：超解析度與盲修復](#1-問題定義超解析度與盲修復)
- [2. 評估：失真與感知的取捨](#2-評估失真與感知的取捨)
- [3. CNN 與 GAN：Real-ESRGAN](#3-cnn-與-ganreal-esrgan)
- [4. Transformer：SwinIR、HAT、DAT、DRCT、ATD](#4-transformerswinirhatdatdrctatd)
- [5. 擴散模型與 SeedVR2](#5-擴散模型與-seedvr2)
- [6. 臉部修復：GFPGAN 與 CodeFormer](#6-臉部修復gfpgan-與-codeformer)
- [7. 前處理：去雜訊與去壓縮](#7-前處理去雜訊與去壓縮)
- [8. 工程技術：分塊推論、數值精度、記憶體管理](#8-工程技術分塊推論數值精度記憶體管理)
- [參考文獻](#參考文獻)

---

## 1. 問題定義：超解析度與盲修復

**超解析度（Super-Resolution, SR）** 是從低解析度影像 $y$ 還原高解析度影像 $x$。經典的劣化模型為：

$$y = (x \otimes k)\downarrow_s + n$$

- $k$：模糊核（鏡頭失焦、動態模糊）
- $\downarrow_s$：以倍率 $s$ 縮小
- $n$：雜訊（感光元件雜訊、壓縮失真）

真實世界的影像劣化通常**未知且混合**：多次轉存的 JPEG、手機降噪、影片串流壓縮、縮放。處理劣化未知的影像稱為**盲修復（blind restoration）**。

這是**病態問題（ill-posed）**：同一張低解析度影像可能對應無限多張高解析度影像，缺失的高頻細節無法唯一決定。因此所有方法都需要某種「先驗知識」：

| 方法 | 先驗從哪裡來 |
|---|---|
| 雙三次插值 | 假設影像平滑（沒有新細節） |
| CNN / Transformer 回歸 | 從大量成對資料學到的對應關係 |
| GAN | 判別器要求輸出「看起來像真實影像」 |
| 擴散模型 | 學到整個自然影像的分布，從中取樣合理的細節 |
| 臉部專用模型 | 人臉的結構（五官位置、紋理） |

> 教學重點：模型「重建」出的細節是**合理的推測**，不是真實資訊的還原。用於鑑識、醫學或證據用途時必須特別小心。

## 2. 評估：失真與感知的取捨

| 指標 | 衡量什麼 | 特性 |
|---|---|---|
| PSNR / SSIM | 與真實影像逐像素的差異 | 偏好平滑、模糊的輸出 |
| LPIPS、FID 等 | 深度特徵或分布上的相似度 | 偏好銳利、紋理豐富的輸出 |

以 MSE 訓練的模型 PSNR 高但看起來糊；以 GAN / 擴散訓練的模型看起來清楚，PSNR 反而較低。兩者無法同時最佳化，這稱為**失真與感知的取捨（perception–distortion tradeoff）**。

本專案的模型多為感知導向（GAN 或擴散），目標是「看起來清楚自然」。

## 3. CNN 與 GAN：Real-ESRGAN

**程式**：`engine.py` 中 `general`、`anime`、`video-fast` 三個模型。

Real-ESRGAN [1] 的貢獻主要在**訓練資料**而非架構：

- **高階劣化模型**：把「模糊 → 縮小 → 雜訊 → JPEG」重複兩次、每次參數隨機，模擬多次轉存的網路影像。
- **純合成資料**：只需要高解析度影像，低解析度版本即時合成，不需要成對收集。
- **U-Net 判別器 + 頻譜正規化**：提供逐像素的真實度回饋並穩定訓練。

生成器是 ESRGAN 的 RRDB（Residual-in-Residual Dense Block）CNN；`video-fast` 是輕量的 SRVGG 架構，速度快數十倍。

> 延伸觀察：本專案的 Phhofm 模型名稱中的 `real`、`otf`、`multiblur` 等字尾，代表訓練時模擬的劣化類型。**同一架構、不同劣化設定，結果差異可能比換架構更大**，選模型時應先看它是為哪種劣化訓練的。

## 4. Transformer：SwinIR、HAT、DAT、DRCT、ATD

**程式**：`photo-best`（DAT2）、`web-photo`（ATD）、`clean-max`（DRCT-L）。

CNN 的感受野有限；Transformer 的自注意力可以讓每個像素參考遠處的相似紋理（例如磚牆重複的花紋）。但全圖自注意力的計算量是像素數的平方，因此 SR Transformer 都在**限制注意力範圍**上下功夫：

| 模型 | 核心想法 |
|---|---|
| SwinIR [2] | 在固定大小的視窗內做注意力，並交錯平移視窗讓資訊跨視窗流動 |
| HAT [3] | 結合通道注意力與視窗注意力，加上重疊的跨視窗注意力，「啟動更多像素」 |
| DAT [4] | 交替使用**空間**注意力與**通道**注意力，兼顧局部與全域資訊 |
| DRCT [5] | 加入密集殘差連接，避免深層網路中資訊流失（資訊瓶頸） |
| ATD [6] | 學習一組「token 字典」作為全域先驗，突破局部視窗限制 |

另一條路線是**高效率 CNN**：RealPLKSR（`photo-fast`）基於 PLKSR [7]，只對部分通道使用大卷積核，用很少的計算取得大感受野，速度比 DAT2 快約 8 倍。

**課堂討論**：本專案實測 Transformer 模型（DAT2、ATD、DRCT-L）比 CNN（RealPLKSR）慢 8~35 倍（見 [experiments.md](experiments.md#實驗-7圖片模型速度)）。品質提升是否值得這樣的成本？取決於應用。

## 5. 擴散模型與 SeedVR2

**程式**：`video_seedvr2.py`（呼叫 SeedVR2 CLI）。

### 5.1 擴散模型基本概念

- **前向過程**：逐步對影像加高斯雜訊，直到變成純雜訊。
- **反向過程**：訓練神經網路一步步去除雜訊；生成時從雜訊開始反覆去噪，得到影像。
- **條件生成**：把低解析度影像當作條件，模型就會生成「與輸入一致」的高解析度影像。

擴散模型學到的是完整的影像分布，能合成 GAN 難以產生的自然紋理。缺點是**慢**，傳統上需要數十到數百步。

### 5.2 潛空間與 DiT

直接在像素上做擴散太耗運算，所以先用 **VAE** 把影像壓縮到較小的**潛空間（latent space）**，在潛空間做擴散，最後再解碼回像素。去噪網路使用 **DiT（Diffusion Transformer）**。

一次執行 SeedVR2 分成四個階段（對應程式日誌的 Phase 1~4）：

```
輸入影格 ──VAE 編碼──▶ 潛空間 ──DiT 去噪（1 步）──▶ 潛空間 ──VAE 解碼──▶ 輸出影格 ──色彩校正
          Phase 1                Phase 2                    Phase 3            Phase 4
```

### 5.3 SeedVR 與 SeedVR2

- **SeedVR** [8]：以擴散 Transformer 處理任意長度與解析度的影片修復，在時間與空間上使用**視窗注意力**，讓模型同時參考前後影格。
- **SeedVR2** [9]：透過**擴散對抗後訓練（Diffusion Adversarial Post-Training）**，把多步擴散壓縮成**一步**，速度大幅提升。

### 5.4 為什麼影片要一批一批處理？

逐格處理的模型每一格各自「猜」細節，前後格猜得不一樣，看起來就會**閃爍**。SeedVR2 把連續多格放在同一批，透過時空注意力讓它們互相參考，所以前後一致。

- **批次越大越一致**，但 VRAM 用量也越大（見 [experiments.md](experiments.md#實驗-2每批影格數)）。
- 批次大小必須是 **4n+1**（5、9、13、21…），因為影片 VAE 在時間軸上以 4 倍壓縮，第一格單獨編碼。
- 批次與批次之間用重疊影格（`temporal_overlap`）混合，減少接縫。

### 5.5 色彩校正

擴散模型可能讓整體色調偏移。SeedVR2 在輸出後把**輸入影像的低頻色彩**轉移回來：LAB 是在感知色彩空間比對統計量，Wavelet 是只保留輸出的高頻細節、低頻換成輸入的。

## 6. 臉部修復：GFPGAN 與 CodeFormer

**程式**：`engine.py` 的 `restore_faces()`。

一般 SR 模型在人臉上常不夠好，因為人眼對五官的錯誤非常敏感。臉部模型利用「人臉結構固定」的強先驗：

| 模型 | 先驗 |
|---|---|
| GFPGAN [10] | 預訓練 StyleGAN2 人臉生成器，把它當作「人臉長什麼樣子」的知識庫 |
| CodeFormer [11] | 在高品質人臉上學一個**離散碼本（codebook）**，Transformer 預測每個位置該用哪個碼；**保真度權重 w** 控制要多相信輸入 |

**完整流程**（facexlib）：

1. **偵測**：RetinaFace 找出臉的位置與 5 個特徵點
2. **對齊**：仿射變換把臉轉正、縮放到 512×512 標準模板
3. **修復**：GFPGAN 或 CodeFormer
4. **貼回**：反仿射變換，以 ParseNet 產生的臉部分割遮罩柔和地貼回已放大的影像

> 實作細節（容易踩雷）：這兩個模型的輸入輸出範圍是 **[-1, 1]**，不是一般影像模型的 [0, 1]。`engine.py` 呼叫時先做 `x * 2 - 1`，輸出再 `(y + 1) / 2`。

**CodeFormer 保真度**：w 越低，越依賴碼本（修得越清楚，但可能不像本人）；w 越高，越保留輸入特徵。這是「失真與感知取捨」的直接體現。

## 7. 前處理：去雜訊與去壓縮

SR 模型會把雜訊和壓縮方塊**當成細節一起放大**。劣化嚴重時可以先清理：

- **SCUNet** [12]：結合 Swin Transformer 與 CNN 的 U-Net，配合真實雜訊的資料合成，做盲去雜訊。
- **FBCNN** [13]：預測 JPEG 品質因子並據以去除方塊與振鈴，品質因子也可手動調整。

## 8. 工程技術：分塊推論、數值精度、記憶體管理

### 8.1 分塊推論與羽化混合

**程式**：`engine.py` 的 `run_tiled()`。

大圖一次放進 GPU 會記憶體不足，所以切成小塊（tile）處理再拼回。直接拼接會在邊界看到接縫，因為每塊邊緣的像素缺少鄰近的上下文。解法：

1. 相鄰分塊**重疊**一段（預設 32 像素）
2. 每塊的輸出乘上**線性漸變權重**（重疊區由 0 漸變到 1），影像外緣保持 1
3. 累加後除以權重總和

分塊大小是速度與記憶體的取捨：分塊越大越快（重疊浪費少、GPU 利用率高），但記憶體越多。本專案依剩餘 VRAM 與架構自動決定（`auto_tile()`），記憶體不足時自動縮小重試。

### 8.2 數值精度

| 格式 | 位元 | 特性 |
|---|---|---|
| FP32 | 32 | 最精確、最慢 |
| TF32 | 19（矩陣運算內部） | Ampere 以後的 GPU 自動使用，幾乎無損 |
| FP16 | 16 | 快、省一半記憶體；數值範圍小（最大 65504），容易溢位 |
| BF16 | 16 | 範圍與 FP32 相同、精度較低，比 FP16 不易溢位 |
| FP8（E4M3） | 8 | SeedVR2 的權重儲存格式，再省一半 |
| INT8 / FP4 | 8 / 4 | SageAttention 2 / 3 用於注意力計算，誤差更大 |

**實作觀察**：模型宣稱支援半精度，不代表真的能跑。本專案實測 ATD 在 FP16/BF16 下發生型別錯誤、CodeFormer 在 BF16 下失敗。因此 `engine.py` 載入模型時會**實際用一張測試輸入試跑**（`_try_dtype()`），失敗就退回 FP32。

### 8.3 記憶體管理（8GB VRAM 的生存之道）

| 技術 | 做法 | 代價 |
|---|---|---|
| 分塊（tiling） | 圖片 / VAE 切塊處理 | 稍慢、需處理接縫 |
| 卸載（offload） | 不用的模型暫存到 CPU 記憶體 | 搬移時間 |
| BlockSwap | DiT 的一部分 Transformer 區塊放在 CPU，用到時才搬上 GPU | 每次搬移約數百毫秒 |
| 降低精度 | FP8 權重、半精度運算 | 些微品質損失 |
| 限制用量 | `torch.cuda.set_per_process_memory_fraction(0.92)` | 讓 VRAM 不足時**報錯**而不是默默溢出 |

最後一項是 WSL 特有的問題：Windows 驅動允許 CUDA 記憶體溢出到系統記憶體，程式不會出錯，但速度會掉 10~20 倍。詳細實驗見 [experiments.md](experiments.md#實驗-1wsl-的-vram-溢出)。

---

## 參考文獻

1. X. Wang, L. Xie, C. Dong, Y. Shan. *Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data.* ICCVW 2021. [arXiv:2107.10833](https://arxiv.org/abs/2107.10833)
2. J. Liang et al. *SwinIR: Image Restoration Using Swin Transformer.* ICCVW 2021. [arXiv:2108.10257](https://arxiv.org/abs/2108.10257)
3. X. Chen et al. *Activating More Pixels in Image Super-Resolution Transformer.* CVPR 2023. [arXiv:2205.04437](https://arxiv.org/abs/2205.04437)
4. Z. Chen et al. *Dual Aggregation Transformer for Image Super-Resolution.* ICCV 2023. [arXiv:2308.03364](https://arxiv.org/abs/2308.03364)
5. C.-C. Hsu, C.-M. Lee, Y.-S. Chou. *DRCT: Saving Image Super-resolution away from Information Bottleneck.* CVPRW 2024. [arXiv:2404.00722](https://arxiv.org/abs/2404.00722)
6. L. Zhang et al. *Transcending the Limit of Local Window: Advanced Super-Resolution Transformer with Adaptive Token Dictionary.* CVPR 2024. [arXiv:2401.08209](https://arxiv.org/abs/2401.08209)
7. D. Lee, S. Yun, Y. Ro. *Partial Large Kernel CNNs for Efficient Super-Resolution.* 2024. [arXiv:2404.11848](https://arxiv.org/abs/2404.11848)
8. J. Wang et al. *SeedVR: Seeding Infinity in Diffusion Transformer Towards Generic Video Restoration.* CVPR 2025. [arXiv:2501.01320](https://arxiv.org/abs/2501.01320)
9. J. Wang et al. *SeedVR2: One-Step Video Restoration via Diffusion Adversarial Post-Training.* 2025. [arXiv:2506.05301](https://arxiv.org/abs/2506.05301)
10. X. Wang, Y. Li, H. Zhang, Y. Shan. *Towards Real-World Blind Face Restoration with Generative Facial Prior.* CVPR 2021. [arXiv:2101.04061](https://arxiv.org/abs/2101.04061)
11. S. Zhou, K. C. K. Chan, C. Li, C. C. Loy. *Towards Robust Blind Face Restoration with Codebook Lookup Transformer.* NeurIPS 2022. [arXiv:2206.11253](https://arxiv.org/abs/2206.11253)
12. K. Zhang et al. *Practical Blind Image Denoising via Swin-Conv-UNet and Data Synthesis.* 2022. [arXiv:2203.13278](https://arxiv.org/abs/2203.13278)
13. J. Jiang, K. Zhang, R. Timofte. *Towards Flexible Blind JPEG Artifacts Removal.* ICCV 2021. [arXiv:2109.14573](https://arxiv.org/abs/2109.14573)
14. T. Dao. *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning.* 2023. [arXiv:2307.08691](https://arxiv.org/abs/2307.08691)
15. J. Zhang et al. *SageAttention2: Efficient Attention with Thorough Outlier Smoothing and Per-thread INT4 Quantization.* 2024. [arXiv:2411.10958](https://arxiv.org/abs/2411.10958)
