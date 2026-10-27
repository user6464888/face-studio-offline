"""Generate local foreground and conservative head feature masks with BiRefNet."""
from pathlib import Path
import argparse
import json
import time
import cv2
import numpy as np
# Importing the existing CUDA-enabled PyTorch loads compatible CUDA/cuDNN DLLs.
import torch
import onnxruntime as ort
from PIL import Image, ImageDraw


def main():
    p = argparse.ArgumentParser()
    p.add_argument("input", type=Path)
    p.add_argument("--model", type=Path, default=Path(__file__).parent / "models/birefnet-lite.onnx")
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()
    root = args.input.resolve()
    for folder in ("foreground_masks", "masks", "cutouts"):
        (root / folder).mkdir(exist_ok=True)
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 6
    providers = ([('CUDAExecutionProvider', {'cudnn_conv_algo_search': 'DEFAULT'})]
                 if 'CUDAExecutionProvider' in ort.get_available_providers() and torch.cuda.is_available() else [])
    session = ort.InferenceSession(str(args.model), opts, providers=providers + ["CPUExecutionProvider"])
    print('Segmentation providers:', session.get_providers(), flush=True)
    name = session.get_inputs()[0].name
    files = sorted((root / "images").glob("*.jpg"))
    if args.limit:
        files = files[:args.limit]
    rows = []
    for index, path in enumerate(files):
        dest = root / "masks" / (path.name + ".png")
        rgb = np.asarray(Image.open(path).convert("RGB"))
        h, w = rgb.shape[:2]
        started = time.time()
        if not dest.exists():
            small = cv2.resize(rgb, (1024, 1024)).astype(np.float32) / 255
            x = ((small - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]).astype(np.float32)
            logit = session.run(None, {name: x.transpose(2, 0, 1)[None]})[0][0, 0]
            prob = 1 / (1 + np.exp(-np.clip(logit, -30, 30)))
            prob = cv2.resize(prob, (w, h), interpolation=cv2.INTER_LINEAR)
            binary = (prob > 0.5).astype(np.uint8)
            count, labels, stats, centers = cv2.connectedComponentsWithStats(binary)
            if count < 2:
                raise RuntimeError(f"No person foreground: {path.name}")
            component = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            foreground = (labels == component).astype(np.uint8) * 255
            cv2.imwrite(str(root / "foreground_masks" / (path.name + ".png")), foreground)
            yy, xx = np.where(foreground > 0)
            top = int(yy.min())
            upper = yy < top + int(h * .25)
            cx = int(np.median(xx[upper]))
            # Feature matching ignores shoulders and the stationary wall. This ROI
            # is deliberately only a feature mask, never treated as a true silhouette.
            head = foreground.copy()
            head[top + int(h * .74):] = 0
            head[:, :max(0, cx - int(w * .20))] = 0
            head[:, min(w, cx + int(w * .20)):] = 0
            head = cv2.erode(head, np.ones((5, 5), np.uint8))
            cv2.imwrite(str(dest), head)
            rgba = np.dstack([rgb, head])
            Image.fromarray(rgba).save(root / "cutouts" / (path.stem + ".png"))
        mask = cv2.imread(str(dest), 0)
        rows.append({"file": path.name, "foreground_fraction": float((mask > 0).mean())})
        print(f"{index + 1}/{len(files)} {path.name} {time.time() - started:.1f}s", flush=True)
    previews = []
    for path in files[::max(1, len(files) // 15)][:15]:
        im = Image.open(root / "cutouts" / (path.stem + ".png"))
        bg = Image.new("RGBA", im.size, "#dce3ec")
        bg.alpha_composite(im)
        bg.thumbnail((320, 180))
        previews.append((bg.convert("RGB"), path.stem))
    sheet = Image.new("RGB", (320 * 3, 204 * ((len(previews) + 2) // 3)), "#1c2738")
    draw = ImageDraw.Draw(sheet)
    for i, (im, label) in enumerate(previews):
        ox, oy = i % 3 * 320, i // 3 * 204
        sheet.paste(im, (ox, oy))
        draw.text((ox + 8, oy + 182), label, fill="white")
    sheet.save(root / "mask_overview.jpg", quality=92)
    (root / "segmentation.json").write_text(json.dumps({"model": "BiRefNet-general-lite", "feature_mask_is_not_silhouette": True, "images": rows}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
