"""命令列介面：python restore.py -i 輸入(檔案或資料夾) -o 輸出 [選項]"""
import argparse
import sys
import time
from pathlib import Path

from tqdm import tqdm

import engine
import eta
import video_seedvr2 as seed

IMG_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
VID_EXT = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v"}


def main() -> int:
    p = argparse.ArgumentParser(description="AI 畫質修復 / 超解析度")
    p.add_argument("-i", "--input", help="輸入圖片、影片或資料夾")
    p.add_argument("--download", action="store_true", help="預先下載所有模型")
    p.add_argument("-o", "--output", default="outputs", help="輸出資料夾 (預設 outputs)")
    p.add_argument("-m", "--model", default="photo-best", choices=[m.key for m in engine.SR_MODELS])
    p.add_argument("-s", "--scale", type=float, default=0, help="輸出倍率，0=模型原生(4x)")
    p.add_argument("--face", default="none", choices=["none", "codeformer", "gfpgan"])
    p.add_argument("--fidelity", type=float, default=0.7, help="CodeFormer 保真度 0~1")
    p.add_argument("--pre", default="none", choices=["none", "scunet", "fbcnn"], help="放大前處理")
    p.add_argument("--tile", type=int, default=0, help="分塊大小，0=自動")
    p.add_argument("--ext", default="png", help="圖片輸出格式 png/jpg/webp")
    p.add_argument("--crf", type=int, default=16, help="影片品質 (越小越好)")
    p.add_argument("--video-engine", default="seedvr2", choices=["seedvr2", "frame"],
                   help="影片引擎：seedvr2=擴散修復(最佳) / frame=逐格放大(快)")
    p.add_argument("--res", type=int, default=1080, help="SeedVR2 輸出短邊解析度")
    p.add_argument("--seed-model", default="3b-fp8", choices=list(seed.DIT_MODELS))
    p.add_argument("--batch", type=int, default=0, help="SeedVR2 每批影格數，0=自動")
    p.add_argument("--frames", type=int, default=0, help="只處理前 N 格（試跑）")
    p.add_argument("--list", action="store_true", help="列出模型")
    args = p.parse_args()

    if args.list:
        for m in engine.SR_MODELS + engine.FACE_MODELS + engine.PRE_MODELS:
            print(f"{m.key:12s} {m.label}\n{'':12s} {m.desc}")
        return 0
    if args.download:
        bar = tqdm(total=100, unit="%")

        def prog(v, m):
            bar.n = round(v * 100)
            bar.set_description(m.split(" ")[1] if " " in m else m)
            bar.refresh()
        engine.download_all(prog)
        bar.close()
        print("所有模型已下載至", engine.MODEL_DIR)
        return 0
    if not args.input:
        p.error("需要 -i 輸入")

    opt = engine.Options(model=args.model, outscale=args.scale, face=args.face,
                         fidelity=args.fidelity, pre=args.pre, tile=args.tile)
    src = Path(args.input)
    files = sorted(f for f in src.rglob("*") if f.suffix.lower() in IMG_EXT | VID_EXT) if src.is_dir() else [src]
    if not files:
        print("找不到可處理的檔案", file=sys.stderr)
        return 1

    out_dir = Path(args.output)
    print(engine.system_info())
    tag = f"{args.model}{'_' + args.face if args.face != 'none' else ''}"
    for f in tqdm(files, unit="檔", disable=len(files) == 1):
        rel = f.relative_to(src).parent if src.is_dir() else Path()
        t0 = time.time()
        try:
            if f.suffix.lower() in VID_EXT:
                bar = tqdm(total=100, desc=f.name, unit="%")

                def prog(v, m, bar=bar):
                    bar.n = round(v * 100)
                    bar.set_postfix_str(m)
                    bar.refresh()
                w, h, _, n = engine.probe_video(str(f))
                frames = min(args.frames, n) if args.frames > 0 else n
                if args.video_engine == "seedvr2":
                    sec = eta.estimate_seed(w, h, frames, args.res, args.seed_model)[0]
                else:
                    sec = eta.estimate_frame(w, h, frames, args.model, args.face != "none")[0]
                tqdm.write(f"⏱ {f.name}：{frames:,} 格，預估約 {eta.fmt(sec)}")
                if args.video_engine == "seedvr2":
                    dst = out_dir / rel / f"{f.stem}_seedvr2_{args.res}p.mp4"
                    seed.upscale(str(f), str(dst), seed.SeedOptions(
                        resolution=args.res, model=args.seed_model, batch_size=args.batch,
                        load_cap=args.frames), progress=prog)
                else:
                    dst = out_dir / rel / f"{f.stem}_{tag}.mp4"
                    engine.restore_video(str(f), str(dst), opt, crf=args.crf, progress=prog)
                bar.close()
            else:
                dst = out_dir / rel / f"{f.stem}_{tag}.{args.ext}"
                img = engine.read_image(f)
                out = engine.restore_image(img, opt)
                engine.write_image(dst, out)
            tqdm.write(f"✓ {f.name} → {dst}  ({time.time() - t0:.1f}s)")
        except Exception as e:  # noqa: BLE001
            tqdm.write(f"✗ {f.name}: {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
