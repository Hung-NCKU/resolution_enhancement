"""畫質修復核心引擎：模型管理、分塊推論、臉部修復、影片處理。"""
from __future__ import annotations

import gc
import math
import os
import shutil
import subprocess
import threading
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import spandrel
import spandrel_extra_arches

spandrel_extra_arches.install()

ROOT = Path(__file__).resolve().parent
MODEL_DIR = Path(os.environ.get("RESOLUTION_MODEL_DIR", ROOT / "models"))
MODEL_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == "cuda":
    torch.backends.cudnn.benchmark = False  # 圖片尺寸多變時反而變慢；影片時才開啟
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # WSL 下 VRAM 不足會默默溢出到系統記憶體（慢 10 倍以上），限制用量讓它改拋 OOM，
    # 由 run_tiled 自動縮小分塊重試
    torch.cuda.set_per_process_memory_fraction(0.92)

PH = "https://github.com/Phhofm/models/releases/download"
RE = "https://github.com/xinntao/Real-ESRGAN/releases/download"


@dataclass(frozen=True)
class ModelInfo:
    key: str
    label: str
    url: str
    desc: str
    kind: str = "sr"  # sr | face | pre

    subdir: str = ""

    @property
    def filename(self) -> str:
        name = self.url.rsplit("/", 1)[-1]
        return f"{self.subdir}/{name}" if self.subdir else name


SR_MODELS = [
    ModelInfo("photo-best", "真實照片・最高品質 (DAT2)",
              f"{PH}/4xBHI_dat2_real/4xBHI_dat2_real.safetensors",
              "處理模糊、雜訊、JPEG 壓縮等真實劣化，細節最自然。速度較慢。"),
    ModelInfo("photo-fast", "真實照片・快速 (RealPLKSR)",
              f"{PH}/4xbhi_realplksr/4xBHI_realplksr_dysample_real.safetensors",
              "與最高品質相同訓練資料，速度快數倍，適合大量圖片或影片。"),
    ModelInfo("web-photo", "網路壓縮照片 (ATD)",
              f"{PH}/4xNomosWebPhoto_atd/4xNomosWebPhoto_atd.safetensors",
              "針對社群網站、多次轉存的照片（重壓縮、縮圖）。"),
    ModelInfo("clean-max", "乾淨高清圖・極致細節 (DRCT-L)",
              f"{PH}/4xNomos2_hq_drct-l/4xNomos2_hq_drct-l.safetensors",
              "輸入本身乾淨時放大效果最好（SOTA 級 Transformer），非常吃 VRAM 與時間。"),
    ModelInfo("general", "通用經典 (Real-ESRGAN x4plus)",
              f"{RE}/v0.1.0/RealESRGAN_x4plus.pth",
              "最穩定的經典模型，各種圖都能用。"),
    ModelInfo("anime", "動漫/插畫 (Real-ESRGAN Anime)",
              f"{RE}/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
              "動畫、漫畫、插圖，線條乾淨。"),
    ModelInfo("video-fast", "影片・極速 (realesr-general-x4v3)",
              f"{RE}/v0.2.5.0/realesr-general-x4v3.pth",
              "輕量模型，適合長影片。"),
]

FACE_MODELS = [
    ModelInfo("codeformer", "CodeFormer", "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth",
              "修復力最強，可調整保真度。", kind="face"),
    ModelInfo("gfpgan", "GFPGAN v1.4", "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth",
              "自然、保留原貌。", kind="face"),
]

PRE_MODELS = [
    ModelInfo("scunet", "去雜訊 (SCUNet)", "https://github.com/cszn/KAIR/releases/download/v1.0/scunet_color_real_psnr.pth",
              "放大前先移除強烈雜訊。", kind="pre"),
    ModelInfo("fbcnn", "去 JPEG 壓縮痕 (FBCNN)", "https://github.com/jiaxi-jiang/FBCNN/releases/download/v1.0/fbcnn_color.pth",
              "放大前先移除方塊與振鈴。", kind="pre"),
]

ALL_MODELS = {m.key: m for m in SR_MODELS + FACE_MODELS + PRE_MODELS}

# 臉部偵測 / 分割權重（facexlib 會從 models/facexlib 讀取）
FX = "https://github.com/xinntao/facexlib/releases/download"
AUX_MODELS = [
    ModelInfo("retinaface", "臉部偵測 RetinaFace", f"{FX}/v0.1.0/detection_Resnet50_Final.pth", "", "aux", "facexlib"),
    ModelInfo("parsenet", "臉部分割 ParseNet", f"{FX}/v0.2.2/parsing_parsenet.pth", "", "aux", "facexlib"),
]


def log(msg: str) -> None:
    print(f"[resolution] {msg}", flush=True)


def download(info: ModelInfo, progress: Optional[Callable[[float, str], None]] = None,
             dest_dir: Optional[Path] = None) -> Path:
    path = (dest_dir or MODEL_DIR) / info.filename
    if path.exists() and path.stat().st_size > 0:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.part")
    log(f"下載模型 {info.label} ...")
    # 先解析重新導向後的實際網址與大小，GitHub 單連線常只有數百 KB/s，改用多連線分段下載
    req = urllib.request.Request(info.url, method="HEAD")
    with urllib.request.urlopen(req, timeout=30) as r:
        final_url = r.geturl()
        total = int(r.headers.get("Content-Length", 0))
        ranged = r.headers.get("Accept-Ranges", "") == "bytes"

    done = 0
    done_lock = threading.Lock()

    def report(n: int) -> None:
        nonlocal done
        with done_lock:
            done += n
            if progress and total:
                progress(done / total, f"下載 {info.label} {done >> 20}/{total >> 20} MB")

    def fetch(start: int, end: int, f) -> None:
        pos, attempts = start, 0
        while pos <= end:
            try:
                rq = urllib.request.Request(final_url, headers={"Range": f"bytes={pos}-{end}"})
                with urllib.request.urlopen(rq, timeout=60) as r:
                    while chunk := r.read(1 << 18):
                        with done_lock:
                            f.seek(pos)
                            f.write(chunk)
                        pos += len(chunk)
                        report(len(chunk))
            except OSError:
                attempts += 1
                if attempts > 8:
                    raise

    if ranged and total > 8 << 20:
        n = 8
        size = math.ceil(total / n)
        with open(tmp, "wb") as f:
            f.truncate(total)
            threads = [threading.Thread(target=fetch, args=(i * size, min(total, (i + 1) * size) - 1, f))
                       for i in range(n)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
        if tmp.stat().st_size != total or done < total:
            raise IOError(f"下載不完整：{info.filename}")
    else:
        with urllib.request.urlopen(final_url, timeout=60) as r, open(tmp, "wb") as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
                report(len(chunk))
    os.replace(tmp, path)
    return path


def download_all(progress=None) -> None:
    for info in list(ALL_MODELS.values()) + AUX_MODELS:
        download(info, progress)


# ---------------------------------------------------------------- 模型快取

_cache: dict[str, spandrel.ModelDescriptor] = {}
_lock = threading.Lock()


def load(key: str, progress=None) -> spandrel.ModelDescriptor:
    with _lock:
        if key in _cache:
            return _cache[key]
        info = ALL_MODELS[key]
        path = download(info, progress)
        desc = spandrel.ModelLoader(device="cpu").load_from_file(str(path))
        desc.model.eval()
        for p in desc.model.parameters():
            p.requires_grad_(False)
        desc.to(DEVICE)
        # 半精度：fp16 > bf16 > fp32。部分架構（如 ATD）宣稱支援卻會出錯，實際試跑確認。
        # 臉部模型很小且內部有 fp32 運算，維持 fp32
        if DEVICE.type == "cuda" and info.kind != "face":
            for dt, ok in ((torch.float16, desc.supports_half), (torch.bfloat16, desc.supports_bfloat16)):
                if ok and _try_dtype(desc, dt):
                    break
            else:
                desc.model.float()
        # 保持少量模型常駐，避免 8GB VRAM 爆掉
        sr_loaded = [k for k in _cache if ALL_MODELS[k].kind == info.kind]
        for k in sr_loaded:
            del _cache[k]
        _cache[key] = desc
        free_memory()
        log(f"已載入 {info.label} ({desc.architecture.name}, x{desc.scale}, {model_dtype(desc)})")
        return desc


@torch.inference_mode()
def _try_dtype(desc, dtype) -> bool:
    try:
        desc.model.to(dtype)
        out = desc(torch.rand(1, desc.input_channels, 128, 128, device=DEVICE, dtype=dtype))  # 太小的輸入可能走不到會出錯的路徑
        return bool(torch.isfinite(out).all())
    except Exception:  # noqa: BLE001
        return False


def model_dtype(desc) -> torch.dtype:
    return next(desc.model.parameters()).dtype


def free_memory() -> None:
    gc.collect()
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()


def unload_all() -> None:
    with _lock:
        _cache.clear()
    global _face_helper
    _face_helper = None
    free_memory()


# ---------------------------------------------------------------- 影像 <-> tensor

def to_tensor(img_rgb: np.ndarray, dtype) -> torch.Tensor:
    t = torch.from_numpy(np.ascontiguousarray(img_rgb)).to(DEVICE)
    t = t.permute(2, 0, 1).unsqueeze(0)
    return (t.float() / 255.0).to(dtype)


def to_image(t: torch.Tensor) -> np.ndarray:
    t = t.squeeze(0).float().clamp_(0, 1).mul_(255).round_().byte()
    return t.permute(1, 2, 0).cpu().numpy()


# ---------------------------------------------------------------- 分塊推論

def auto_tile(desc) -> int:
    """依 VRAM 與架構估計安全的分塊大小（輸入像素邊長）。"""
    if DEVICE.type != "cuda":
        return 256
    free, _ = torch.cuda.mem_get_info()
    free += torch.cuda.memory_reserved() - torch.cuda.memory_allocated()  # PyTorch 快取中可重用的部分
    gb = free / 1024 ** 3
    arch = desc.architecture.name.lower()
    if "drct" in arch and model_dtype(desc) == torch.float32:
        return 256  # DRCT-L fp32 非常吃記憶體
    heavy = any(a in arch for a in ("drct", "hat", "atd", "dat", "swin", "rgt", "grl"))
    # 實測 RTX 5060：DAT2 半精度 512 分塊約 1.3 GB、ATD fp32 512 約 4.1 GB；分塊越大越快
    fp32 = model_dtype(desc) == torch.float32
    if heavy:
        if gb < 3:
            return 192
        if gb < 5.5:
            return 256 if fp32 else 384
        return 512 if (fp32 or gb < 12) else 768
    return 512 if gb < 3 else 1024


@torch.inference_mode()
def run_tiled(desc, img_rgb: np.ndarray, tile: int = 0, overlap: int = 32,
              progress: Optional[Callable[[float, str], None]] = None) -> np.ndarray:
    """對 RGB uint8 影像進行分塊推論，使用羽化混合避免接縫。OOM 時自動縮小分塊。"""
    tile = tile or auto_tile(desc)
    while True:
        try:
            return _run_tiled(desc, img_rgb, tile, overlap, progress)
        except torch.OutOfMemoryError:
            free_memory()
            if tile <= 64:
                raise
            tile = tile // 2
            log(f"VRAM 不足，分塊縮小為 {tile}")


def _run_tiled(desc, img_rgb, tile, overlap, progress):
    dtype = model_dtype(desc)
    s = desc.scale
    h, w = img_rgb.shape[:2]
    x = to_tensor(img_rgb, dtype)

    if h <= tile and w <= tile:
        if progress:
            progress(0.0, "處理中")
        return to_image(desc(x))

    overlap = min(overlap, tile // 4)
    step = tile - overlap
    ys = list(range(0, max(h - tile, 0) + 1, step))
    xs = list(range(0, max(w - tile, 0) + 1, step))
    if ys[-1] + tile < h:
        ys.append(h - tile)
    if xs[-1] + tile < w:
        xs.append(w - tile)
    ys = [max(0, v) for v in ys]
    xs = [max(0, v) for v in xs]

    out = torch.zeros((1, 3, h * s, w * s), device=DEVICE, dtype=torch.float32)
    wsum = torch.zeros((1, 1, h * s, w * s), device=DEVICE, dtype=torch.float32)

    def ramp(n_in: int, lo_edge: bool, hi_edge: bool) -> torch.Tensor:
        n = n_in * s
        r = torch.ones(n, device=DEVICE)
        f = overlap * s
        if f > 0:
            fade = torch.linspace(0, 1, f + 2, device=DEVICE)[1:-1]
            if not lo_edge:
                r[:f] = fade
            if not hi_edge:
                r[-f:] = fade.flip(0)
        return r

    total = len(ys) * len(xs)
    i = 0
    for y0 in ys:
        for x0 in xs:
            y1, x1 = min(y0 + tile, h), min(x0 + tile, w)
            patch = desc(x[:, :, y0:y1, x0:x1]).float()
            wy = ramp(y1 - y0, y0 == 0, y1 == h)
            wx = ramp(x1 - x0, x0 == 0, x1 == w)
            wm = (wy[:, None] * wx[None, :])[None, None]
            out[:, :, y0 * s:y1 * s, x0 * s:x1 * s] += patch * wm
            wsum[:, :, y0 * s:y1 * s, x0 * s:x1 * s] += wm
            i += 1
            if progress:
                progress(i / total, f"分塊 {i}/{total}")
    return to_image(out / wsum.clamp_min(1e-6))


# ---------------------------------------------------------------- 臉部修復

_face_helper = None


def _get_face_helper(upscale: float):
    global _face_helper
    from facexlib.utils.face_restoration_helper import FaceRestoreHelper
    if _face_helper is None:
        for info in AUX_MODELS:
            download(info)
        _face_helper = FaceRestoreHelper(
            upscale, face_size=512, crop_ratio=(1, 1), det_model="retinaface_resnet50",
            save_ext="png", use_parse=True, device=DEVICE,
            model_rootpath=str(MODEL_DIR / "facexlib"))
    _face_helper.upscale_factor = upscale
    _face_helper.clean_all()
    return _face_helper


@torch.inference_mode()
def restore_faces(orig_bgr: np.ndarray, upscaled_bgr: np.ndarray, face_key: str,
                  fidelity: float = 0.7, progress=None) -> np.ndarray:
    """偵測原圖中的臉，用臉部模型修復後貼回已放大的影像。"""
    desc = load(face_key, progress)
    net = desc.model
    dtype = model_dtype(desc)
    scale = upscaled_bgr.shape[1] / orig_bgr.shape[1]
    helper = _get_face_helper(scale)
    helper.read_image(orig_bgr)
    n = helper.get_face_landmarks_5(only_center_face=False, resize=640, eye_dist_threshold=5)
    if n == 0:
        log("未偵測到臉部")
        return upscaled_bgr
    helper.align_warp_face()
    for i, face in enumerate(helper.cropped_faces):
        rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        t = to_tensor(rgb, dtype) * 2 - 1
        try:
            if face_key == "codeformer":
                out = net(t, weight=fidelity, adain=True)[0]
            else:
                out = net(t, return_rgb=False, randomize_noise=False)[0]
            res = to_image((out.float() + 1) / 2)
            res = cv2.cvtColor(res, cv2.COLOR_RGB2BGR)
        except Exception as e:  # noqa: BLE001
            log(f"臉部 {i} 修復失敗：{e}")
            res = face
        helper.add_restored_face(res)
        if progress:
            progress((i + 1) / n, f"臉部 {i + 1}/{n}")
    helper.get_inverse_affine(None)
    result = helper.paste_faces_to_input_image(upsample_img=upscaled_bgr)
    return np.clip(result, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------- 完整流程

@dataclass
class Options:
    model: str = "photo-best"
    outscale: float = 0.0          # 0 = 使用模型原生倍率
    face: str = "none"             # none | codeformer | gfpgan
    fidelity: float = 0.7          # CodeFormer 保真度 (0=修復力強, 1=貼近原圖)
    pre: str = "none"              # none | scunet | fbcnn
    tile: int = 0                  # 0 = 自動
    face_blend: float = 1.0        # 臉部修復結果混合比例


def restore_image(img: np.ndarray, opt: Options, progress=None) -> np.ndarray:
    """輸入 BGR/BGRA/灰階 uint8 (OpenCV 格式)，回傳相同通道格式的修復結果。"""
    if img.dtype == np.uint16:
        img = (img / 257).round().astype(np.uint8)
    alpha = None
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        alpha = img[:, :, 3]
        img = img[:, :, :3]
    orig_bgr = img
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def stage(name, lo, hi):
        if not progress:
            return None
        return lambda p, m="": progress(lo + (hi - lo) * p, f"{name} {m}".strip())

    if opt.pre != "none":
        pre = load(opt.pre, stage("下載", 0, 0.05))
        rgb = run_tiled(pre, rgb, tile=512, progress=stage("前處理", 0.05, 0.2))
        orig_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    desc = load(opt.model, stage("下載", 0.2, 0.25))
    out = run_tiled(desc, rgb, tile=opt.tile, progress=stage("放大", 0.25, 0.85))
    native = desc.scale
    out_bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)

    h, w = orig_bgr.shape[:2]
    target = opt.outscale or native
    if abs(target - native) > 1e-3:
        tw, th = round(w * target), round(h * target)
        interp = cv2.INTER_AREA if target < native else cv2.INTER_LANCZOS4
        out_bgr = cv2.resize(out_bgr, (tw, th), interpolation=interp)

    if opt.face != "none":
        restored = restore_faces(orig_bgr, out_bgr, opt.face, opt.fidelity, stage("臉部", 0.85, 0.98))
        if opt.face_blend < 1.0:
            restored = cv2.addWeighted(restored, opt.face_blend, out_bgr, 1 - opt.face_blend, 0)
        out_bgr = restored

    if alpha is not None:
        a = cv2.resize(alpha, (out_bgr.shape[1], out_bgr.shape[0]), interpolation=cv2.INTER_LANCZOS4)
        out_bgr = np.dstack([out_bgr, a])
    if progress:
        progress(1.0, "完成")
    return out_bgr


def read_image(path: str | Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"無法讀取影像：{path}")
    return img


def write_image(path: str | Path, img: np.ndarray, quality: int = 95) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower()
    params = []
    if ext in (".jpg", ".jpeg"):
        if img.ndim == 3 and img.shape[2] == 4:
            img = img[:, :, :3]
        params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    elif ext == ".webp":
        params = [cv2.IMWRITE_WEBP_QUALITY, quality]
    ok, buf = cv2.imencode(ext, img, params)
    if not ok:
        raise ValueError(f"無法寫入：{path}")
    buf.tofile(str(path))


# ---------------------------------------------------------------- 影片

def ffmpeg_bin() -> str:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def probe_video(path: str) -> tuple[int, int, float, int]:
    import json
    probe = shutil.which("ffprobe")
    if probe:
        # 不用 -count_packets：長影片要掃描整個檔案，兩小時的片要等很久
        r = subprocess.run([probe, "-v", "error", "-select_streams", "v:0",
                            "-show_entries", "stream=width,height,r_frame_rate,nb_frames,duration:format=duration",
                            "-of", "json", path], capture_output=True, text=True, check=True)
        j = json.loads(r.stdout)
        s = j["streams"][0]
        num, den = s["r_frame_rate"].split("/")
        fps = float(num) / float(den)
        n = int(s.get("nb_frames") or 0)
        if not n:
            dur = float(s.get("duration") or j.get("format", {}).get("duration") or 0)
            n = round(dur * fps)
        return int(s["width"]), int(s["height"]), fps, n
    cap = cv2.VideoCapture(path)
    info = (int(cap.get(3)), int(cap.get(4)), cap.get(5) or 30.0, int(cap.get(7)))
    cap.release()
    return info


def restore_video(src: str, dst: str, opt: Options, crf: int = 16, progress=None,
                  cancel: Optional[threading.Event] = None) -> str:
    """逐格修復影片並保留原始音訊。"""
    ff = ffmpeg_bin()
    w, h, fps, n = probe_video(src)
    desc = load(opt.model)
    scale = opt.outscale or desc.scale
    ow, oh = round(w * scale) // 2 * 2, round(h * scale) // 2 * 2

    torch.backends.cudnn.benchmark = DEVICE.type == "cuda"  # 影格尺寸固定，自動調校最快的卷積演算法
    dec = subprocess.Popen([ff, "-v", "error", "-i", src, "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
                           stdout=subprocess.PIPE)
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    enc = subprocess.Popen([ff, "-v", "error", "-y",
                            "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{ow}x{oh}", "-r", f"{fps}", "-i", "-",
                            "-i", src, "-map", "0:v:0", "-map", "1:a?", "-c:a", "copy",
                            "-c:v", "libx264", "-preset", "slow", "-crf", str(crf), "-pix_fmt", "yuv420p",
                            "-movflags", "+faststart", dst], stdin=subprocess.PIPE)
    frame_bytes = w * h * 3
    i = 0
    try:
        while True:
            if cancel is not None and cancel.is_set():
                raise InterruptedError("已取消")
            buf = dec.stdout.read(frame_bytes)
            if len(buf) < frame_bytes:
                break
            frame = np.frombuffer(buf, np.uint8).reshape(h, w, 3)
            out = restore_image(frame, opt)
            if out.shape[1] != ow or out.shape[0] != oh:
                out = cv2.resize(out, (ow, oh), interpolation=cv2.INTER_AREA)
            enc.stdin.write(out.tobytes())
            i += 1
            if progress:
                progress(i / n if n else 0, f"影格 {i}/{n or '?'}")
    finally:
        torch.backends.cudnn.benchmark = False
        dec.stdout.close()
        dec.kill()
        enc.stdin.close()
        enc.wait()
    if enc.returncode != 0:
        raise RuntimeError("ffmpeg 編碼失敗")
    return dst


def system_info() -> str:
    if DEVICE.type == "cuda":
        p = torch.cuda.get_device_properties(0)
        free, total = torch.cuda.mem_get_info()
        return f"GPU：{p.name}｜VRAM {free / 1024 ** 3:.1f}/{total / 1024 ** 3:.1f} GB 可用｜PyTorch {torch.__version__}"
    return f"CPU 模式（未偵測到 CUDA）｜PyTorch {torch.__version__}"
