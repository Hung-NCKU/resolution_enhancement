"""網頁介面：python app.py 後開啟 http://localhost:7860"""
import os
import threading
import time
import zipfile
from pathlib import Path

import cv2
import gradio as gr
import numpy as np

import engine
import eta
import video_seedvr2 as seed

OUT = engine.ROOT / "outputs"
OUT.mkdir(exist_ok=True)

SR_CHOICES = [(m.label, m.key) for m in engine.SR_MODELS]
SR_DESC = {m.key: m.desc for m in engine.SR_MODELS}
FACE_CHOICES = [("不使用", "none"), ("CodeFormer（修復力最強）", "codeformer"), ("GFPGAN v1.4（自然）", "gfpgan")]
PRE_CHOICES = [("不使用", "none"), ("去雜訊 SCUNet", "scunet"), ("去 JPEG 壓縮痕 FBCNN", "fbcnn")]
SCALE_CHOICES = [("模型原生 (4x)", 0), ("2x", 2), ("3x", 3), ("4x", 4), ("6x", 6), ("8x", 8)]

ENGINE_SEED = "seedvr2"
ENGINE_FRAME = "frame"
ENGINE_CHOICES = [("SeedVR2 擴散修復（最佳畫質、畫面不閃爍）", ENGINE_SEED),
                  ("逐格 AI 放大（快速）", ENGINE_FRAME)]
SEED_MODEL_CHOICES = [(v[1], k) for k, v in seed.DIT_MODELS.items()]
RES_CHOICES = [("720p", 720), ("1080p (Full HD)", 1080), ("1440p (2K)", 1440), ("2160p (4K，很慢)", 2160)]
COLOR_CHOICES = [("LAB（建議）", "lab"), ("Wavelet", "wavelet"), ("AdaIN", "adain"), ("不校正", "none")]


def make_opt(model, scale, face, fidelity, pre, tile):
    return engine.Options(model=model, outscale=float(scale), face=face, fidelity=float(fidelity),
                          pre=pre, tile=int(tile))


def stamp(name: str, tag: str, ext: str) -> Path:
    stem = Path(name).stem if name else "output"
    return OUT / f"{stem}_{tag}_{time.strftime('%H%M%S')}.{ext}"


def gr_progress(progress, live: "eta.Live | None" = None):
    def update(v, m=""):
        v = min(max(v, 0.0), 1.0)
        progress(v, desc=f"{m}｜{live.remaining(v)}" if live else m)
    return update


# ---------------------------------------------------------------- 影片

def _frames(n: int, cap) -> int:
    cap = int(cap or 0)
    return min(cap, n) if cap > 0 else n


def _about(sec: float) -> str:
    s = eta.fmt(sec)
    return s if s.startswith("不到") else f"約 {s}"


def _frame_estimate(w, h, frames, fmodel, face):
    return eta.estimate_frame(w, h, frames, fmodel, face != "none")[0]


def video_estimate(path, eng, res, smodel, cap, fmodel, fscale, face):
    """上傳影片或改設定時，顯示輸出尺寸與預估處理時間。"""
    if not path:
        return ""
    try:
        w, h, fps, n = engine.probe_video(path)
    except Exception as e:  # noqa: BLE001
        return f"無法讀取影片資訊：{e}"
    frames = _frames(n, cap)
    lines = [f"**輸入**：{w}×{h}，{fps:.2f} fps，{n:,} 格（{eta.fmt(n / fps) if fps else '?'}）"]
    warn = []
    if eng == ENGINE_SEED:
        ow, oh = eta.seed_output_size(w, h, int(res))
        factor = int(res) / min(w, h)
        sec = eta.estimate_seed(w, h, frames, int(res), smodel)[0]
        if factor > 4.5:
            warn.append(f"放大 {factor:.1f} 倍超過 SeedVR2 擅長的約 4 倍，可能出現不自然的細節，建議降低輸出解析度。")
        if ow * oh / 1e6 > eta.SEED_BIG_MP:
            warn.append("輸出畫面超過約 190 萬像素（約 1920×1000），8GB 顯卡記憶體會不夠用，速度約慢 2.4 倍。")
    else:
        s = float(fscale) or 4
        ow, oh, factor = round(w * s), round(h * s), s
        sec = _frame_estimate(w, h, frames, fmodel, face)
    lines.append(f"**輸出**：{ow}×{oh}（放大 {factor:.1f} 倍）")
    part = f"（只處理前 {frames:,} 格）" if frames < n else ""
    lines.append(f"### ⏱ 預估處理時間：{_about(sec)} {part}")
    if sec > 3 * 3600:
        alt = _frame_estimate(w, h, frames, "photo-fast", "none")
        tip = f"⚠️ 超過 3 小時。建議先用「試跑」看效果"
        if eng == ENGINE_SEED:
            tip += f"，或整部改用「逐格 AI 放大」（photo-fast {_about(alt)}），只把重要片段剪出來用 SeedVR2"
        warn.append(tip + "。")
    lines += [f"- {x}" for x in warn]
    lines.append("<sub>預估依本機實測速度，每次完成後會自動校正；其他程式佔用顯示卡時會變慢。</sub>")
    return "\n\n".join(lines)


def run_video(path, eng, res, smodel, batch, color, cap,
              fmodel, fscale, face, fidelity, crf, progress=gr.Progress()):
    if not path:
        raise gr.Error("請先上傳影片")
    VIDEO_CANCEL.clear()
    t0 = time.time()
    try:
        return _run_video(path, eng, res, smodel, batch, color, cap, fmodel, fscale, face, fidelity, crf,
                          progress, t0)
    except InterruptedError:
        return None, "⏹ 已停止"


def _run_video(path, eng, res, smodel, batch, color, cap, fmodel, fscale, face, fidelity, crf, progress, t0):
    live = eta.Live()
    iw, ih, _, n_in = engine.probe_video(path)
    frames = _frames(n_in, cap)
    if eng == ENGINE_SEED:
        dst = stamp(path, f"seedvr2_{res}p", "mp4")
        opt = seed.SeedOptions(resolution=int(res), model=smodel, batch_size=0 if int(batch) <= 1 else int(batch),
                               color_correction=color, load_cap=int(cap))
        seed.upscale(path, str(dst), opt, progress=gr_progress(progress, live), log=print)
        _, key, work = eta.estimate_seed(iw, ih, frames, int(res), smodel)
        eta.record(key, work, max(time.time() - t0 - eta.SEED_OVERHEAD, 1))
    else:
        opt = make_opt(fmodel, fscale, face, fidelity, "none", 0)
        dst = stamp(path, fmodel, "mp4")
        engine.restore_video(path, str(dst), opt, crf=int(crf), progress=gr_progress(progress, live),
                             cancel=VIDEO_CANCEL)
        if face == "none":  # 有臉部修復時速度受人臉數量影響，不納入校正
            _, key, work = eta.estimate_frame(iw, ih, frames, fmodel, False)
            eta.record(key, work, time.time() - t0)
    w, h, fps, n = engine.probe_video(str(dst))
    el = time.time() - t0
    return str(dst), f"✅ 完成：{w}×{h}，{n} 格，耗時 {eta.fmt(el)}（每格 {el / max(n, 1):.1f} 秒）\n已儲存：{dst}"


VIDEO_CANCEL = threading.Event()


def stop_video():
    VIDEO_CANCEL.set()
    seed.stop()
    return "⏹ 已要求停止"


def toggle_engine(eng):
    return gr.update(visible=eng == ENGINE_SEED), gr.update(visible=eng == ENGINE_FRAME)


# ---------------------------------------------------------------- 圖片

def run_single(path, model, scale, face, fidelity, pre, tile, fmt, progress=gr.Progress()):
    if not path:
        raise gr.Error("請先上傳圖片")
    t0 = time.time()
    img = engine.read_image(path)
    opt = make_opt(model, scale, face, fidelity, pre, tile)
    out = engine.restore_image(img, opt, progress=gr_progress(progress))
    dst = stamp(path, opt.model + (f"_{opt.face}" if opt.face != "none" else ""), fmt)
    engine.write_image(dst, out)

    # 對比圖：原圖以雙三次放大到相同尺寸
    rgb_out = cv2.cvtColor(out[:, :, :3], cv2.COLOR_BGR2RGB)
    src = (img / 257).round().astype(np.uint8) if img.dtype == np.uint16 else img
    src = src if src.ndim == 3 else cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)
    src = cv2.cvtColor(src[:, :, :3], cv2.COLOR_BGR2RGB)
    src = cv2.resize(src, (rgb_out.shape[1], rgb_out.shape[0]), interpolation=cv2.INTER_CUBIC)
    h, w = img.shape[:2]
    info = (f"✅ 完成：{w}×{h} → {out.shape[1]}×{out.shape[0]}，耗時 {time.time() - t0:.1f} 秒\n"
            f"已儲存：{dst}")
    return (src, rgb_out), str(dst), info


def run_batch(files, model, scale, face, fidelity, pre, tile, fmt, progress=gr.Progress()):
    if not files:
        raise gr.Error("請先選擇圖片")
    opt = make_opt(model, scale, face, fidelity, pre, tile)
    folder = OUT / f"batch_{time.strftime('%Y%m%d_%H%M%S')}"
    folder.mkdir(parents=True)
    done, logs = [], []
    for i, f in enumerate(files):
        name = Path(f).name
        progress(i / len(files), desc=f"{i + 1}/{len(files)} {name}")
        try:
            out = engine.restore_image(engine.read_image(f), opt)
            dst = folder / f"{Path(name).stem}.{fmt}"
            engine.write_image(dst, out)
            done.append(str(dst))
            logs.append(f"✓ {name}")
        except Exception as e:  # noqa: BLE001
            logs.append(f"✗ {name}：{e}")
    zpath = folder.with_suffix(".zip")
    with zipfile.ZipFile(zpath, "w") as z:
        for d in done:
            z.write(d, Path(d).name)
    logs.append(f"\n共 {len(done)}/{len(files)} 張，輸出資料夾：{folder}")
    return done, str(zpath), "\n".join(logs)


def free_vram():
    engine.unload_all()
    return engine.system_info()


CSS = """
.gradio-container {max-width: 1400px !important; margin: auto;}
#title {text-align:center; margin-bottom: 0}
#subtitle {text-align:center; opacity:.75; margin-top:0}
"""

with gr.Blocks(title="AI 影片畫質修復") as demo:
    gr.Markdown("# AI 影片畫質修復工作站", elem_id="title")
    gr.Markdown("SeedVR2 一步擴散影片修復 ＋ DAT2 / RealPLKSR / Real-ESRGAN 逐格放大 ＋ CodeFormer 臉部修復",
                elem_id="subtitle")
    sysinfo = gr.Markdown(engine.system_info())

    with gr.Tabs():
        # ============================================================ 影片
        with gr.Tab("🎬 影片修復"):
            with gr.Row():
                with gr.Column(scale=1, min_width=320):
                    eng = gr.Radio(ENGINE_CHOICES, value=ENGINE_SEED, label="修復引擎",
                                   info="SeedVR2 畫質最好但很慢（每格數秒）；逐格放大快很多（每格零點幾秒）。")
                    with gr.Group(visible=True) as seed_box:
                        res = gr.Dropdown(
                            RES_CHOICES, value=1080, label="輸出解析度（短邊）",
                            info="輸出影片較短那一邊的像素。放大倍率最好不超過 4 倍；"
                                 "8GB 顯卡輸出超過約 1920×1000 時速度會慢 2 倍以上。")
                        smodel = gr.Dropdown(
                            SEED_MODEL_CHOICES, value="3b-fp8", label="SeedVR2 模型",
                            info="3B FP8 速度與品質最平衡。7B 細節更多但慢 3 倍以上；"
                                 "非預設模型第一次使用時會自動下載（3~5 GB）。")
                        batch = gr.Slider(
                            1, 45, value=1, step=4, label="每批影格數",
                            info="一次一起處理幾格。越多前後格越一致、越不閃爍，但越吃顯卡記憶體。"
                                 "1＝自動（720p 用 21、1080p 用 13），一般不用改。")
                        color = gr.Dropdown(
                            COLOR_CHOICES, value="lab", label="色彩校正",
                            info="修復後把顏色拉回接近原片，避免偏色。LAB 最準確；結果顏色怪怪的再換 Wavelet。")
                        cap = gr.Number(
                            value=0, precision=0, label="試跑：只處理前 N 格",
                            info="先處理開頭一小段看效果和速度，再決定要不要跑整部。0＝整部影片；"
                                 "建議先試 50~100 格。")
                    with gr.Group(visible=False) as frame_box:
                        fmodel = gr.Dropdown(
                            SR_CHOICES, value="photo-fast", label="放大模型",
                            info="photo-fast 適合大多數影片；video-fast 最快；動畫選動漫模型；"
                                 "photo-best 品質最好但慢約 8 倍。")
                        fscale = gr.Dropdown(
                            SCALE_CHOICES, value=2, label="輸出倍率",
                            info="輸出是原影片的幾倍大。模型固定先放大 4 倍，再縮到這個倍率。")
                        vface = gr.Radio(
                            FACE_CHOICES, value="none", label="臉部修復",
                            info="偵測人臉並另外修復，讓五官更清楚。沒有人臉的影片不用開，會多花時間。")
                        vfid = gr.Slider(
                            0, 1, value=0.7, step=0.05, label="CodeFormer 保真度",
                            info="只對 CodeFormer 有效。低＝修得更清楚但可能不像本人；"
                                 "高＝更像原貌但修復較少。建議 0.5~0.8。")
                        crf = gr.Slider(
                            10, 28, value=16, step=1, label="影片品質 CRF",
                            info="輸出影片的壓縮程度。數字越小畫質越好、檔案越大。"
                                 "16 幾乎看不出壓縮；23 約等於一般網路影片。")
                    gr.Markdown(
                        "**說明**\n"
                        "- **SeedVR2**：目前開源最強的影片修復模型，會「重建」細節並保持前後格一致，"
                        "適合老影片、模糊、壓縮嚴重的影片。速度慢，建議先用「試跑」看效果。\n"
                        "- **逐格放大**：每格獨立處理，速度快，適合本身品質還可以的影片。\n"
                        "- 音訊會原樣保留。")
                with gr.Column(scale=3):
                    with gr.Row():
                        vin = gr.Video(label="輸入影片", height=400)
                        vout = gr.Video(label="修復結果", height=400)
                    vmeta = gr.Markdown()
                    with gr.Row():
                        vid_btn = gr.Button("開始修復影片", variant="primary", size="lg", scale=4)
                        stop_btn = gr.Button("停止", size="lg", scale=1)
                    vinfo = gr.Textbox(label="狀態", lines=2)

        # ============================================================ 圖片
        with gr.Tab("🖼️ 圖片修復"):
            with gr.Row():
                with gr.Column(scale=1, min_width=320):
                    model = gr.Dropdown(SR_CHOICES, value="photo-best", label="放大模型",
                                        info="依圖片類型選擇，選了之後下方會顯示這個模型的特色。")
                    model_desc = gr.Markdown(SR_DESC["photo-best"])
                    scale = gr.Dropdown(SCALE_CHOICES, value=0, label="輸出倍率",
                                        info="輸出是原圖的幾倍大。模型原生為 4 倍；選 6x、8x 是放大後再拉大，細節不會更多。")
                    face = gr.Radio(FACE_CHOICES, value="none", label="臉部修復",
                                    info="偵測人臉並另外修復。人像、老照片很有用；風景或沒有人臉的圖不用開。")
                    fidelity = gr.Slider(0, 1, value=0.7, step=0.05, label="CodeFormer 保真度",
                                         info="只對 CodeFormer 有效。低＝修得更清楚但可能不像本人；"
                                              "高＝更像原貌但修復較少。建議 0.5~0.8。")
                    pre = gr.Radio(PRE_CHOICES, value="none", label="放大前處理",
                                   info="放大前先清理：滿是雜點（暗處噪點）選 SCUNet；"
                                        "有方塊狀壓縮痕（低畫質 JPG）選 FBCNN。")
                    with gr.Accordion("進階", open=False):
                        tile = gr.Slider(0, 1024, value=0, step=32, label="分塊大小",
                                         info="大圖會切成小塊處理，以免顯卡記憶體不足。0＝依顯卡自動決定，"
                                              "一般不用改；出現記憶體錯誤時可調小。")
                        fmt = gr.Radio(["png", "jpg", "webp"], value="png", label="輸出格式",
                                       info="PNG 無損但檔案大，保留透明背景；JPG / WebP 檔案小。")
                        free_btn = gr.Button("釋放 VRAM", size="sm")
                        gr.Markdown("<sub>釋放 VRAM：清掉已載入的模型，讓出顯卡記憶體給其他程式或影片修復。</sub>")
                with gr.Column(scale=3):
                    with gr.Tabs():
                        with gr.Tab("單張"):
                            with gr.Row():
                                inp = gr.Image(type="filepath", label="輸入圖片", height=420)
                                cmp = gr.ImageSlider(label="前後對比（拖曳滑桿）", format="png", height=420,
                                                     buttons=["fullscreen"])
                            run_btn = gr.Button("開始修復", variant="primary", size="lg")
                            with gr.Row():
                                out_file = gr.File(label="下載原始尺寸結果")
                                info = gr.Textbox(label="狀態", lines=3)
                        with gr.Tab("批次"):
                            files = gr.File(file_count="multiple", file_types=["image"], label="選擇多張圖片")
                            batch_btn = gr.Button("批次修復", variant="primary", size="lg")
                            gallery = gr.Gallery(label="結果", columns=4, height=420)
                            with gr.Row():
                                zip_file = gr.File(label="打包下載 (ZIP)")
                                batch_log = gr.Textbox(label="紀錄", lines=6)

    # 影片事件
    eng.change(toggle_engine, eng, [seed_box, frame_box])
    est_inputs = [vin, eng, res, smodel, cap, fmodel, fscale, vface]
    for comp in est_inputs:
        comp.change(video_estimate, est_inputs, vmeta, show_progress="hidden", concurrency_limit=None)
    vid_evt = vid_btn.click(run_video, [vin, eng, res, smodel, batch, color, cap,
                                        fmodel, fscale, vface, vfid, crf], [vout, vinfo])
    vid_evt.then(engine.system_info, None, sysinfo)
    stop_btn.click(stop_video, None, vinfo, concurrency_limit=None)

    # 圖片事件
    common = [model, scale, face, fidelity, pre, tile]
    model.change(lambda k: SR_DESC[k], model, model_desc)
    run_btn.click(run_single, [inp] + common + [fmt], [cmp, out_file, info]).then(engine.system_info, None, sysinfo)
    batch_btn.click(run_batch, [files] + common + [fmt], [gallery, zip_file, batch_log]).then(
        engine.system_info, None, sysinfo)
    free_btn.click(free_vram, None, sysinfo)


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch(
        server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)),
        theme=gr.themes.Soft(primary_hue="indigo"), css=CSS,
        allowed_paths=[str(OUT)], max_file_size="4gb", inbrowser=False, show_error=True)
