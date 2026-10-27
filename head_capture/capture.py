"""Local RGB webcam capture. Saved images are never mirrored."""

import csv
import json
import os
import queue
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageTk

ROOT = Path(__file__).resolve().parent
CAPTURES = ROOT / "captures"


def utc_now():
    return datetime.now(timezone.utc).isoformat()


class Session:
    def __init__(self, camera):
        CAPTURES.mkdir(exist_ok=True)
        self.path = CAPTURES / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        (self.path / "images").mkdir(parents=True)
        self.lock = threading.RLock()
        self.data = {
            "created_at": utc_now(), "camera": camera, "modality": "RGB",
            "has_sensor_depth": False, "pose_measured": False,
            "angle_labels_are_guidance_only": True, "saved_images_mirrored": False,
            "camera_intrinsics": None, "camera_extrinsics": None,
            "images": [], "videos": [],
        }
        self.flush()

    def flush(self):
        with self.lock:
            temporary = self.path / "session.json.tmp"
            temporary.write_text(json.dumps(self.data, ensure_ascii=False, indent=2), encoding="utf-8")
            temporary.replace(self.path / "session.json")

    def photo(self, frame, stage, angle, timestamp):
        number = len(self.data["images"]) + 1
        name = f"images/{stage}_{angle:03d}_{number:04d}.jpg"
        # imencode/tofile supports Chinese Windows paths, unlike some imwrite builds.
        ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not ok:
            raise RuntimeError("照片编码失败")
        encoded.tofile(str(self.path / name))
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame[h // 5:4 * h // 5, w // 3:2 * w // 3], cv2.COLOR_BGR2GRAY)
        score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        with self.lock:
            self.data["images"].append({
                "file": name, "captured_at": timestamp, "saved_at": utc_now(),
                "stage": stage, "yaw_hint_degrees": angle,
                "width": w, "height": h, "sharpness_reference": score,
            })
            self.flush()
        return score


class Camera:
    """The worker reads every frame; the preview can skip old frames."""
    def __init__(self, index):
        self.index = index
        self.messages = queue.Queue()
        self.lock = threading.Lock()
        self.latest = None
        self.stop_event = threading.Event()
        self.thread = None
        self.writer = None
        self.csv_file = None
        self.video_info = None
        self.fps = 30.0

    def start(self):
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def run(self):
        cap = None
        try:
            backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
            cap = cv2.VideoCapture(self.index, backend)
            if not cap.isOpened():
                raise RuntimeError("无法打开摄像头。请关闭会议软件，检查镜头遮挡和 Windows 摄像头权限，或尝试其他编号。")
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            cap.set(cv2.CAP_PROP_FPS, 30)
            reported_fps = cap.get(cv2.CAP_PROP_FPS)
            self.fps = reported_fps if 1 <= reported_fps <= 120 else 30.0
            self.messages.put(("ready", None))
            failures = 0
            while not self.stop_event.is_set():
                ok, frame = cap.read()
                if not ok:
                    failures += 1
                    if failures >= 10:
                        raise RuntimeError("摄像头连续读取失败，请重新打开摄像头。")
                    self.stop_event.wait(0.05)
                    continue
                failures = 0
                stamp, monotonic = utc_now(), time.monotonic()
                with self.lock:
                    self.latest = (frame, stamp, monotonic)
                    if self.writer is not None:
                        expected = (self.video_info["height"], self.video_info["width"])
                        if frame.shape[:2] != expected:
                            raise RuntimeError("录像期间摄像头分辨率改变，录像已停止。")
                        self.writer.write(frame)
                        self.csv_writer.writerow([self.video_info["frames"], stamp, monotonic - self.video_started])
                        self.video_info["frames"] += 1
                        self.video_info["last_frame_offset_seconds"] = monotonic - self.video_started
        except Exception as error:
            self.messages.put(("error", str(error)))
        finally:
            try:
                self.finish_recording()
            except Exception as error:
                self.messages.put(("error", f"录像保存失败：{error}"))
            finally:
                if cap is not None:
                    cap.release()
                self.messages.put(("closed", None))

    def snapshot(self):
        with self.lock:
            if self.latest is None:
                return None
            frame, stamp, monotonic = self.latest
            return frame.copy(), stamp, monotonic

    def begin_recording(self, session):
        with self.lock:
            if self.latest is None:
                raise RuntimeError("还没有摄像头画面")
            if self.writer is not None:
                return
            h, w = self.latest[0].shape[:2]
            number = len(session.data["videos"]) + 1
            name = f"video_{number:02d}.avi"
            writer = cv2.VideoWriter(str(session.path / name), cv2.VideoWriter_fourcc(*"MJPG"), self.fps, (w, h))
            if not writer.isOpened():
                writer.release()
                raise RuntimeError("无法创建录像文件，请检查文件夹权限和可用空间。")
            info = {"file": name, "timestamps": f"video_{number:02d}_timestamps.csv",
                    "started_at": utc_now(), "width": w, "height": h,
                    "nominal_fps": self.fps, "frames": 0, "last_frame_offset_seconds": None,
                    "timing_note": "AVI uses nominal_fps; use the CSV for real capture timing."}
            try:
                csv_file = (session.path / info["timestamps"]).open("w", newline="", encoding="utf-8")
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow(["frame_index", "captured_at_utc", "elapsed_seconds"])
                with session.lock:
                    session.data["videos"].append(info)
                    session.flush()
            except Exception:
                writer.release()
                if "csv_file" in locals():
                    csv_file.close()
                raise
            self.csv_file, self.csv_writer = csv_file, csv_writer
            self.video_info, self.video_session = info, session
            self.video_started = time.monotonic()
            self.writer = writer

    def finish_recording(self):
        with self.lock:
            if self.writer is None:
                return
            self.writer.release()
            self.csv_file.close()
            self.writer = self.csv_file = None
            with self.video_session.lock:
                self.video_info["ended_at"] = utc_now()
                self.video_info["elapsed_seconds"] = time.monotonic() - self.video_started
                self.video_session.flush()


class App:
    def __init__(self, window):
        self.window, self.camera, self.session = window, None, None
        self.closing = False
        self.position = 0
        window.title("人头采集 · 电脑摄像头")
        window.geometry("1060x850")
        window.minsize(860, 760)
        style = ttk.Style()
        style.configure("TButton", font=("Microsoft YaHei UI", 11), padding=8)
        style.configure("TLabel", font=("Microsoft YaHei UI", 11))
        outer = ttk.Frame(window, padding=16)
        outer.pack(fill="both", expand=True)
        ttk.Label(outer, text="先采集清晰、连续的彩色画面", font=("Microsoft YaHei UI", 18, "bold")).pack(anchor="w")
        toolbar = ttk.Frame(outer)
        toolbar.pack(fill="x", pady=10)
        ttk.Label(toolbar, text="摄像头编号").pack(side="left")
        self.index = tk.StringVar(value="0")
        ttk.Combobox(toolbar, textvariable=self.index, values=[str(i) for i in range(6)], width=3, state="readonly").pack(side="left", padx=8)
        self.open_button = ttk.Button(toolbar, text="打开摄像头", command=self.toggle_camera)
        self.open_button.pack(side="left")
        ttk.Button(toolbar, text="查看保存文件夹", command=self.show_folder).pack(side="right")
        self.preview = tk.Label(outer, bg="#17202a", fg="white", text="点击“打开摄像头”开始预览\n照片和录像只保存在本机", font=("Microsoft YaHei UI", 15))
        self.preview.pack(fill="both", expand=True)
        settings = ttk.Frame(outer)
        settings.pack(fill="x", pady=10)
        self.stage = tk.StringVar(value="水平一圈（24 张）")
        ttk.Combobox(settings, textvariable=self.stage, state="readonly", width=23,
                     values=["水平一圈（24 张）", "高处斜拍补充（8 张）"]).pack(side="left")
        self.stage.trace_add("write", self.reset_position)
        self.mirror = tk.BooleanVar(value=True)
        ttk.Checkbutton(settings, text="预览像照镜子（保存原图）", variable=self.mirror).pack(side="right")
        self.guide = ttk.Label(outer, text="")
        self.guide.pack(anchor="w", pady=3)
        controls = ttk.Frame(outer)
        controls.pack(fill="x", pady=8)
        ttk.Button(controls, text="上一角度", command=lambda: self.move(-1)).pack(side="left")
        ttk.Button(controls, text="保存照片并继续", command=self.save_photo).pack(side="left", padx=8)
        ttk.Button(controls, text="下一角度", command=lambda: self.move(1)).pack(side="left")
        self.record_button = ttk.Button(controls, text="开始录像", command=self.toggle_recording)
        self.record_button.pack(side="right")
        ttk.Label(outer, text="保持中性表情、闭嘴、头发固定。缓慢转动身体，每个角度停稳后拍照。\n一圈应包含正面、左右侧面和后脑；不要仅左右转头。高处补拍时把电脑稳妥抬高并斜向下拍。",
                  wraplength=1000).pack(anchor="w", pady=6)
        self.status = tk.StringVar(value="尚未保存。相机角度需要后续估计，当前标签只用于整理照片。")
        ttk.Label(outer, textvariable=self.status, wraplength=1000).pack(anchor="w")
        self.update_guide()
        window.bind("<space>", self.space_photo)
        window.protocol("WM_DELETE_WINDOW", self.close)
        window.after(40, self.tick)

    def space_photo(self, event):
        if isinstance(self.window.focus_get(), (ttk.Combobox, ttk.Entry)):
            return
        self.save_photo()
        return "break"

    def reset_position(self, *_):
        self.position = 0
        self.update_guide()

    def geometry(self):
        return ("horizontal", 24) if self.stage.get().startswith("水平") else ("elevated", 8)

    def update_guide(self):
        _, count = self.geometry()
        angle = self.position * (360 // count)
        direction = {0: "正面", 90: "侧面", 180: "后脑", 270: "另一侧面"}.get(angle, "相邻过渡视角")
        self.guide.configure(text=f"当前建议：第 {self.position + 1}/{count} 张 · {angle}° · {direction}（固定同一转动方向）")

    def move(self, amount):
        self.position = (self.position + amount) % self.geometry()[1]
        self.update_guide()

    def toggle_camera(self):
        if self.camera:
            self.camera.stop_event.set()
            self.open_button.configure(state="disabled")
            self.status.set("正在关闭摄像头并保存录像…")
        else:
            if self.session and self.session.data["camera"]["index"] != int(self.index.get()):
                self.session = None
                self.reset_position()
            self.camera = Camera(int(self.index.get()))
            self.camera.start()
            self.open_button.configure(state="disabled")
            self.status.set("正在打开摄像头…")

    def require_frame(self):
        snapshot = self.camera.snapshot() if self.camera else None
        if snapshot is None or time.monotonic() - snapshot[2] > 2:
            raise RuntimeError("请先打开摄像头并等待实时画面。")
        return snapshot

    def get_session(self):
        if self.session is None:
            self.session = Session({"index": self.camera.index, "backend": "DirectShow" if os.name == "nt" else "default"})
        return self.session

    def save_photo(self):
        try:
            frame, stamp, _ = self.require_frame()
            stage, count = self.geometry()
            session = self.get_session()
            score = session.photo(frame, stage, self.position * (360 // count), stamp)
            suffix = "；请放大检查是否清晰，必要时重拍" if score < 60 else ""
            self.status.set(f"已保存 {len(session.data['images'])} 张照片{suffix}。文件夹：{session.path.name}")
            if self.position < count - 1:
                self.move(1)
            else:
                self.status.set(self.status.get() + "。已到最后一个角度，可补拍或切换高处斜拍。")
        except Exception as error:
            messagebox.showerror("未能保存", str(error))

    def toggle_recording(self):
        try:
            self.require_frame()
            if self.camera.writer is not None:
                self.camera.finish_recording()
                self.record_button.configure(text="开始录像")
                self.status.set("录像已保存。可以再拍一圈或补拍照片。")
            else:
                self.camera.begin_recording(self.get_session())
                self.record_button.configure(text="停止录像")
                self.status.set("正在录像。缓慢转动一整圈，保持头部在画面内。")
        except Exception as error:
            messagebox.showerror("录像失败", str(error))

    def show_folder(self):
        CAPTURES.mkdir(exist_ok=True)
        os.startfile(str(self.session.path if self.session else CAPTURES))

    def tick(self):
        if self.camera:
            while not self.camera.messages.empty():
                kind, value = self.camera.messages.get_nowait()
                if kind == "ready":
                    self.open_button.configure(text="关闭摄像头", state="normal")
                    self.status.set("摄像头已打开。先调整光线和距离，让头顶、耳朵与下巴都在画面内。")
                elif kind == "error":
                    self.status.set(value)
                    if not self.closing:
                        messagebox.showerror("摄像头提示", value)
                elif kind == "closed":
                    self.camera = None
                    self.open_button.configure(text="打开摄像头", state="normal")
                    self.record_button.configure(text="开始录像")
                    self.preview.configure(image="", text="摄像头已关闭")
                    self.preview.image = None
                    break
            snapshot = self.camera.snapshot() if self.camera else None
            if snapshot:
                frame = snapshot[0]
                if self.mirror.get():
                    frame = cv2.flip(frame, 1)
                picture = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                picture.thumbnail((max(1, self.preview.winfo_width()), max(1, self.preview.winfo_height())))
                w, h = picture.size
                ImageDraw.Draw(picture).ellipse((w * .33, h * .09, w * .67, h * .9), outline="#7df5ce", width=2)
                display = ImageTk.PhotoImage(picture)
                self.preview.configure(image=display, text="")
                self.preview.image = display
        if self.closing and self.camera is None:
            self.window.destroy()
            return
        self.window.after(40, self.tick)

    def close(self):
        self.closing = True
        if self.camera:
            self.camera.stop_event.set()
            self.status.set("正在关闭摄像头并保存文件…")
        else:
            self.window.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    # A hidden launcher console must not hide the interactive Tk window.
    if os.name == "nt":
        def show_window():
            root.withdraw()
            root.deiconify()
            root.lift()
        root.after(150, show_window)
    root.mainloop()
