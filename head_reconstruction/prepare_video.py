"""Select sharp frames in temporal windows; keep original pixels and timing."""

import argparse
import csv
import html
import json
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw


def sharpness(frame):
    h, w = frame.shape[:2]
    roi = frame[int(h * .12):int(h * .90), int(w * .30):int(w * .70)]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    signature = cv2.resize(gray, (80, 100), interpolation=cv2.INTER_AREA).astype(np.float32)
    signature -= signature.mean()
    return score, signature


def prepare(source, output, video_number=1, window_seconds=.75, duplicate_threshold=1.5):
    if output.exists():
        raise ValueError(f"输出文件夹已存在，选择新的文件夹以免覆盖：{output}")
    session = json.loads((source / "session.json").read_text(encoding="utf-8"))
    video = session["videos"][video_number - 1]
    with (source / video["timestamps"]).open(encoding="utf-8", newline="") as stream:
        timestamps = list(csv.DictReader(stream))
    if not timestamps:
        raise ValueError("视频没有逐帧时间记录")
    if any(int(item["frame_index"]) != n for n, item in enumerate(timestamps)):
        raise ValueError("逐帧时间记录的索引不连续")
    elapsed = [float(item["elapsed_seconds"]) for item in timestamps]
    if not all(math.isfinite(t) for t in elapsed) or any(b < a for a, b in zip(elapsed, elapsed[1:])):
        raise ValueError("逐帧时间记录无效")
    cap = cv2.VideoCapture(str(source / video["file"]))
    if not cap.isOpened():
        raise ValueError("无法打开视频")
    candidates = []
    best = None
    current_bucket = None
    decoded = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if decoded >= len(timestamps):
                raise ValueError("视频帧数量超过逐帧时间记录")
            bucket = int(elapsed[decoded] / window_seconds)
            if current_bucket != bucket:
                if best is not None:
                    candidates.append(best)
                current_bucket, best = bucket, None
            score, signature = sharpness(frame)
            if best is None or score > best["sharpness_reference"]:
                best = {"frame_index": decoded, "elapsed_seconds": elapsed[decoded],
                        "captured_at": timestamps[decoded]["captured_at_utc"],
                        "sharpness_reference": score, "frame": frame, "signature": signature}
            decoded += 1
        if best is not None:
            candidates.append(best)
    finally:
        cap.release()
    if decoded != len(timestamps) or decoded != video["frames"]:
        raise ValueError("视频解码帧数与采集记录不一致，暂停整理，保留原视频检查")

    # Compare adjacent selected frames only. This is a heuristic, not pose estimation.
    kept, skipped = [], []
    for candidate in candidates:
        distance = None
        if kept:
            distance = float(np.abs(candidate["signature"] - kept[-1]["signature"]).mean())
        candidate["difference_from_previous_selected"] = distance
        if distance is not None and distance < duplicate_threshold:
            skipped.append({"frame_index": candidate["frame_index"],
                            "elapsed_seconds": candidate["elapsed_seconds"],
                            "reason": "adjacent near duplicate", "difference": distance})
        else:
            kept.append(candidate)
    if len(kept) < 2:
        raise ValueError("有效候选太少，检查画面或降低近似重复阈值")

    images = output / "images"
    images.mkdir(parents=True)
    manifest = {
        "source_session": str(source.resolve()), "source_video": video["file"],
        "source_frames_decoded": decoded, "source_duration_seconds": video["elapsed_seconds"],
        "window_seconds": window_seconds, "duplicate_threshold": duplicate_threshold,
        "candidate_windows": len(candidates), "selected_count": len(kept),
        "selection_note": "Sharpest central-region frame per time window; adjacent image difference is only a heuristic.",
        "modality": "RGB", "has_sensor_depth": False, "saved_images_mirrored": False,
        "camera_intrinsics": None, "camera_extrinsics": None,
        "poses_estimated": False, "masks_generated": False,
        "images": [], "skipped_candidates": skipped,
    }
    tiles = []
    for number, item in enumerate(kept, 1):
        frame = item["frame"]
        name = f"frame_{item['frame_index']:06d}.jpg"
        ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not ok:
            raise ValueError(f"无法编码帧 {item['frame_index']}")
        encoded.tofile(str(images / name))
        h, w = frame.shape[:2]
        record = {k: v for k, v in item.items() if k not in {"frame", "signature"}}
        record.update({"file": f"images/{name}", "width": w, "height": h})
        manifest["images"].append(record)
        picture = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        picture.thumbnail((240, 135))
        tiles.append((number, item, picture))

    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    columns = 5
    rows = math.ceil(len(tiles) / columns)
    sheet = Image.new("RGB", (columns * 240, rows * 157), "#17202a")
    draw = ImageDraw.Draw(sheet)
    for i, (number, item, picture) in enumerate(tiles):
        x, y = (i % columns) * 240, (i // columns) * 157
        sheet.paste(picture, (x, y))
        draw.text((x + 6, y + 138), f"{number:02d} | {item['elapsed_seconds']:.1f}s | f{item['frame_index']}", fill="white")
    sheet.save(output / "overview.jpg", quality=92)

    cards = []
    for record in manifest["images"]:
        path = html.escape(record["file"], quote=True)
        cards.append(f'<a class="card" href="{path}"><img src="{path}" loading="lazy"><span>{record["elapsed_seconds"]:.1f} 秒 · 原始帧 {record["frame_index"]}</span></a>')
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>头部重建 · 输入检查</title>
<style>body{font:16px system-ui;background:#101820;color:#e8eef5;max-width:1400px;margin:32px auto;padding:0 20px}p{line-height:1.8;color:#b9cad8}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:16px}.card{background:#1c2a35;border-radius:12px;overflow:hidden;color:#e8eef5;text-decoration:none}.card img{width:100%;display:block}.card span{display:block;padding:12px}</style>
<h1>已筛选的重建候选画面</h1><p>点击照片检查原始尺寸。重点看眼镜反光、运动模糊、头部裁切，以及相邻视角是否有足够重叠。当前只是筛帧结果，尚未分割人物、估计相机位置或重建模型。</p>
<div class="grid">'''+"\n".join(cards)+"</div></html>"
    (output / "review.html").write_text(page, encoding="utf-8")
    note = f"""# 当前重建输入

来自本机第二次采集的戴眼镜、无耳机视频。保留原始视频，不混合第一段配饰不同的录像。

- 完整解码 {decoded} 帧，录像约 {video['elapsed_seconds']:.1f} 秒。
- 每 {window_seconds} 秒选取中心区域清晰度最高的一帧，再用相邻图像差异筛除近似重复候选。
- 保留 {len(kept)} 张候选原尺寸照片，筛除 {len(skipped)} 张近似重复候选。此数字不等于独立视角数；照片数量多也不能替代覆盖检查。
- images/ 是原尺寸彩色图；overview.jpg 是缩略图；review.html 可在浏览器逐张检查；manifest.json 记录来源帧、采集时间与筛选信息。

## 下一步

1. 把人物从背景分离，保留头发、耳朵、眼镜和头部外轮廓，检查各视角分割结果。
2. 在人物区域估计多视角相机位置。人物转动而墙面静止，不能让背景主导匹配；固定镜头时也不能直接把视频运动当作真实相机运动。
3. 检查位姿和稀疏点云是否一致，再做表面重建和贴图。
4. 模型能够稳定旋转后，再增加可替换发型和发色。

普通摄像头没有真实深度，当前也没有相机标定。筛帧不保证重建成功。眼镜的透明、反光区域和头顶遮挡需要在实际重建后评估；不预先删除镜框或承诺自动还原被遮住的皮肤。
"""
    (output / "README.md").write_text(note, encoding="utf-8")
    return {"decoded_frames": decoded, "candidate_windows": len(candidates),
            "selected_images": len(kept), "skipped_near_duplicates": len(skipped), "output": str(output.resolve())}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="从本地采集视频筛选重建候选帧")
    parser.add_argument("source", type=Path, help="含 session.json 的采集文件夹")
    parser.add_argument("output", type=Path, help="新的输出文件夹，不覆盖现有目录")
    parser.add_argument("--video-number", type=int, default=1)
    parser.add_argument("--window-seconds", type=float, default=.75)
    parser.add_argument("--duplicate-threshold", type=float, default=1.5)
    args = parser.parse_args()
    if not math.isfinite(args.window_seconds) or args.window_seconds <= 0:
        parser.error("window-seconds 必须是有限正数")
    if not math.isfinite(args.duplicate_threshold) or args.duplicate_threshold < 0:
        parser.error("duplicate-threshold 必须是有限非负数")
    if args.video_number < 1:
        parser.error("video-number 必须从 1 开始")
    print(json.dumps(prepare(args.source, args.output, args.video_number, args.window_seconds, args.duplicate_threshold), ensure_ascii=False))
