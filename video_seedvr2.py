"""SeedVR2 影片修復（ByteDance Seed 一步擴散影片超解析度）的包裝。

以子程序呼叫 numz/ComfyUI-SeedVR2_VideoUpscaler 的 inference_cli.py，
並針對 8GB VRAM (RTX 5060) 調整參數：
- VAE 分塊 512：避免 VRAM 溢出到系統記憶體（WSL 會默默溢出，速度慢 20 倍以上）
- BlockSwap 16：DiT 一半區塊放 CPU，實測比全部放 GPU 或全部交換都快
- 串流分段 (chunk_size)：長影片不會吃光 RAM
完成後把原始音軌合併回去。
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

SEEDVR2_DIR = Path(os.environ.get("SEEDVR2_DIR", Path.home() / "seedvr2"))
# 模型放在 Linux 檔案系統：從 /mnt/d 讀 3GB 權重要 30 秒以上
SEEDVR2_MODELS = Path(os.environ.get("SEEDVR2_MODELS", SEEDVR2_DIR / "models" / "SEEDVR2"))

DIT_MODELS = {
    "3b-fp8": ("seedvr2_ema_3b_fp8_e4m3fn.safetensors", "3B FP8（建議，8GB 最佳平衡）", 16),
    "3b-q8": ("seedvr2_ema_3b-Q8_0.gguf", "3B GGUF Q8（品質接近 FP16）", 16),
    "7b-sharp-q4": ("seedvr2_ema_7b_sharp-Q4_K_M.gguf", "7B Sharp Q4（細節最多，非常慢）", 36),
}

_STAGE = re.compile(r"(Encoding|Upscaling|Decoding) batch (\d+)/(\d+)")
_CHUNK = re.compile(r"[Cc]hunk (\d+)/(\d+)")
# 各階段佔每段時間的比例（實測：編碼約 12%、擴散約 63%、解碼約 25%）
_STAGE_W = {"Encoding": (0.0, 0.12), "Upscaling": (0.12, 0.75), "Decoding": (0.75, 1.0)}
_STAGE_ZH = {"Encoding": "編碼", "Upscaling": "擴散修復", "Decoding": "解碼"}


@dataclass
class SeedOptions:
    resolution: int = 1080          # 輸出短邊像素
    model: str = "3b-fp8"
    batch_size: int = 0             # 每批影格數（需 4n+1），0＝依解析度自動；越大越穩定但越吃 VRAM
    color_correction: str = "lab"
    load_cap: int = 0               # >0 時只處理前 N 格（試跑）
    chunk_size: int = 105           # 串流分段，限制 RAM 用量
    seed: int = 42


_current: Optional[subprocess.Popen] = None


def stop() -> None:
    """終止正在執行的 SeedVR2。"""
    if _current is not None and _current.poll() is None:
        _current.kill()


def attention_mode() -> str:
    """有裝 FlashAttention 就用（RTX 5060 實測 DiT 快約 9%，結果與 SDPA 幾乎相同）。
    SageAttention 2/3 也已安裝，但為 8/4-bit 量化，誤差較大且沒有更快，不預設使用。"""
    import importlib.util
    return "flash_attn_2" if importlib.util.find_spec("flash_attn") else "sdpa"


def available() -> bool:
    return (SEEDVR2_DIR / "inference_cli.py").exists()


def auto_batch(resolution: int) -> int:
    """RTX 5060 8GB 實測：1080p 用 13 格（21 格會溢出 VRAM，慢 3~4 倍）。"""
    if resolution <= 720:
        return 21
    if resolution <= 1080:
        return 13
    return 5


def _fix_batch(n: int) -> int:
    n = max(1, int(n))
    return n - ((n - 1) % 4)  # 往下取到 4n+1


def upscale(src: str, dst: str, opt: SeedOptions,
            progress: Optional[Callable[[float, str], None]] = None,
            cancel: Optional[threading.Event] = None,
            log: Optional[Callable[[str], None]] = None) -> str:
    if not available():
        raise RuntimeError(f"找不到 SeedVR2（{SEEDVR2_DIR}），請執行 setup.sh")
    fname, _, swap = DIT_MODELS[opt.model]
    SEEDVR2_MODELS.mkdir(parents=True, exist_ok=True)
    dst = str(Path(dst).resolve())
    Path(dst).parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="seedvr2_") as tmp:
        raw = str(Path(tmp) / "video.mp4")
        cmd = [
            sys.executable, "-u", "inference_cli.py",
            str(Path(src).resolve()), "--output", raw, "--model_dir", str(SEEDVR2_MODELS),
            "--dit_model", fname, "--resolution", str(opt.resolution),
            "--batch_size", str(_fix_batch(opt.batch_size or auto_batch(opt.resolution))), "--temporal_overlap", "3",
            "--color_correction", opt.color_correction, "--seed", str(opt.seed),
            "--dit_offload_device", "cpu", "--vae_offload_device", "cpu",
            "--blocks_to_swap", str(swap), "--cache_dit", "--cache_vae",
            "--vae_encode_tiled", "--vae_encode_tile_size", "512", "--vae_encode_tile_overlap", "64",
            # 解碼分塊 384：速度與 512 相同，但 VRAM 峰值較低，其他程式（瀏覽器、桌布）佔用 VRAM 時不易溢出
            "--vae_decode_tiled", "--vae_decode_tile_size", "384", "--vae_decode_tile_overlap", "48",
            "--video_backend", "ffmpeg", "--attention_mode", attention_mode(),
        ]
        if opt.chunk_size:
            cmd += ["--chunk_size", str(opt.chunk_size)]
        if opt.load_cap:
            cmd += ["--load_cap", str(opt.load_cap)]

        env = dict(os.environ, PYTHONWARNINGS="ignore", PYTHONUNBUFFERED="1")
        proc = subprocess.Popen(cmd, cwd=SEEDVR2_DIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1, env=env)
        global _current
        _current = proc
        tail: list[str] = []
        chunk, chunks = 1, 1
        try:
            for line in proc.stdout:
                line = line.rstrip()
                if not line:
                    continue
                tail = (tail + [line])[-40:]
                if log:
                    log(line)
                if cancel is not None and cancel.is_set():
                    proc.kill()
                    raise InterruptedError("已取消")
                if m := _CHUNK.search(line):
                    chunk, chunks = int(m.group(1)), int(m.group(2))
                if progress and (m := _STAGE.search(line)):
                    stage, i, n = m.group(1), int(m.group(2)), int(m.group(3))
                    lo, hi = _STAGE_W[stage]
                    frac = lo + (hi - lo) * (i - 1) / n
                    total = ((chunk - 1) + frac) / chunks
                    label = f"{_STAGE_ZH[stage]} {i}/{n}" + (f"（分段 {chunk}/{chunks}）" if chunks > 1 else "")
                    progress(min(total, 0.99), label)
            proc.wait()
        finally:
            if proc.poll() is None:
                proc.kill()
        if proc.returncode in (-9, -15):
            raise InterruptedError("已停止")
        if proc.returncode != 0 or not Path(raw).exists():
            raise RuntimeError("SeedVR2 執行失敗：\n" + "\n".join(tail[-15:]))

        # 合併原始音訊（SeedVR2 輸出沒有聲音）
        if progress:
            progress(0.99, "合併音訊")
        r = subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", raw, "-i", str(Path(src).resolve()),
                            "-map", "0:v:0", "-map", "1:a?", "-c:v", "copy", "-c:a", "aac", "-b:a", "256k",
                            "-shortest", "-movflags", "+faststart", dst], capture_output=True, text=True)
        if r.returncode != 0:
            shutil.copy(raw, dst)
    if progress:
        progress(1.0, "完成")
    return dst
