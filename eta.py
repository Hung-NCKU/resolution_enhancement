"""處理時間預估。

速度單位：
- SeedVR2：秒 / (輸出百萬像素 × 格)
- 逐格放大：秒 / (輸入百萬像素 × 格)
預設值來自 RTX 5060 8GB 實測；每次完成後以實際耗時更新（存在 outputs/.speed.json），越用越準。
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SPEED_FILE = ROOT / "outputs" / ".speed.json"

# SeedVR2 輸出超過約 1.9 MP（例如 2160×1080）時 8GB VRAM 會溢出，實測慢約 2.4 倍，分開記錄
SEED_BIG_MP = 1.9
DEFAULT_RATES = {
    "seedvr2:3b-fp8": 2.6, "seedvr2:3b-fp8:big": 6.2,
    "seedvr2:3b-q8": 3.0, "seedvr2:3b-q8:big": 7.0,
    "seedvr2:7b-sharp-q4": 8.0, "seedvr2:7b-sharp-q4:big": 18.0,
    "frame:photo-best": 26.0, "frame:photo-fast": 3.4, "frame:web-photo": 45.0, "frame:clean-max": 111.0,
    "frame:general": 16.5, "frame:anime": 14.6, "frame:video-fast": 0.6,
}
SEED_OVERHEAD = 40.0   # 載入模型等固定時間（秒）
FACE_PER_FRAME = 0.3   # 開臉部修復時每格額外秒數（粗估）


def _load() -> dict:
    try:
        return json.loads(SPEED_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def rate(key: str) -> float:
    learned = _load().get(key)
    return float(learned) if learned else DEFAULT_RATES.get(key, 5.0)


def record(key: str, work: float, seconds: float) -> None:
    """以實際耗時更新速度（指數平均，避免單次異常影響太大）。"""
    if work <= 0 or seconds <= 0:
        return
    new = seconds / work
    data = _load()
    old = data.get(key)
    data[key] = new if old is None else 0.5 * old + 0.5 * new
    try:
        SPEED_FILE.parent.mkdir(parents=True, exist_ok=True)
        SPEED_FILE.write_text(json.dumps(data, indent=1), encoding="utf-8")
    except OSError:
        pass


def seed_output_size(w: int, h: int, resolution: int) -> tuple[int, int]:
    s = resolution / min(w, h)
    return round(w * s) // 2 * 2, round(h * s) // 2 * 2


def seed_key(model: str, out_mp: float) -> str:
    return f"seedvr2:{model}" + (":big" if out_mp > SEED_BIG_MP else "")


def estimate_seed(w: int, h: int, frames: int, resolution: int, model: str) -> tuple[float, str, float]:
    """回傳 (秒數, 速度 key, 工作量)。"""
    ow, oh = seed_output_size(w, h, resolution)
    mp = ow * oh / 1e6
    key = seed_key(model, mp)
    work = mp * frames
    return SEED_OVERHEAD + rate(key) * work, key, work


def estimate_frame(w: int, h: int, frames: int, model: str, face: bool) -> tuple[float, str, float]:
    key = f"frame:{model}"
    work = w * h / 1e6 * frames
    return rate(key) * work + (FACE_PER_FRAME * frames if face else 0), key, work


def fmt(seconds: float) -> str:
    if not math.isfinite(seconds) or seconds < 0:
        return "?"
    m = seconds / 60
    if m < 1:
        return "不到 1 分鐘"
    if m < 60:
        return f"{m:.0f} 分鐘"
    h = m / 60
    if h < 24:
        return f"{int(h)} 小時 {int(m % 60)} 分"
    return f"{int(h // 24)} 天 {int(h % 24)} 小時"


class Live:
    """依實際進度推算剩餘時間。前 3% 進度與前 20 秒資料不足，不顯示。"""

    def __init__(self):
        self.t0 = time.time()

    def remaining(self, p: float) -> str:
        el = time.time() - self.t0
        if p < 0.03 or el < 20:
            return "計算剩餘時間中…"
        left = fmt(el / p - el)
        return f"已過 {fmt(el)}，剩餘{left if left.startswith('不到') else '約 ' + left}"
