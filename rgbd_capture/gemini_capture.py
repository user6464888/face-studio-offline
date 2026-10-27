"""Local RGB-D capture UI. Camera remains closed until the user connects it."""
from pathlib import Path
import argparse
import json
import time
import threading
import queue
import shutil
import tkinter as tk
from tkinter import ttk,messagebox
import os
import winsound
import numpy as np
import cv2
from PIL import Image,ImageTk
from orbbec_native import GeminiCamera,NativeSDK
from capture_session import Recording,verify_bag

ROOT=Path(__file__).resolve().parent

class App:
    def __init__(self,root):
        self.root=root;root.title('人头 RGB-D 采集 · Gemini');root.geometry('1140x800')
        self.jobs=queue.Queue();self.events=queue.Queue();self.latest=None;self.last_frame_time=0
        self.connected=False;self.recording=False;self.counting=False;self.closing=False;self.current_path=None
        self.worker=threading.Thread(target=self.engine,daemon=True);self.worker.start()
        self.status=tk.StringVar(value='先连接 Gemini，再点击“连接摄像头”。启动程序不会自动打开镜头。')
        self.detail=tk.StringVar(value='需要同时看到彩色画面和有效深度。请关闭其他正在使用该摄像头的软件。')
        self.counter=tk.StringVar(value='未录制');self.label=tk.StringVar(value='水平一圈 A（无眼镜）');self.glasses=tk.StringVar(value='无眼镜')
        self.duration=tk.IntVar(value=45)
        pad=ttk.Frame(root,padding=16);pad.pack(fill='both',expand=True)
        ttk.Label(pad,text='人头 RGB-D 采集',font=('Microsoft YaHei',18,'bold')).pack(anchor='w')
        ttk.Label(pad,text='原始彩色＋深度 · 相机参数 · 本地保存',font=('Microsoft YaHei',10)).pack(anchor='w',pady=(4,12))
        bar=ttk.Frame(pad);bar.pack(fill='x')
        self.connect=ttk.Button(bar,text='连接摄像头',command=self.open);self.connect.pack(side='left')
        ttk.Label(bar,text='  本段内容：').pack(side='left')
        clip_selector=ttk.Combobox(bar,textvariable=self.label,width=28,state='readonly',values=['5 秒试录','水平一圈 A（无眼镜）','水平一圈 B（无眼镜备用）','高位斜拍一圈（补头顶）','戴眼镜参考']);clip_selector.pack(side='left')
        clip_selector.bind('<<ComboboxSelected>>',lambda _:self.choose_clip())
        ttk.Combobox(bar,textvariable=self.glasses,width=8,state='readonly',values=['无眼镜','戴眼镜']).pack(side='left',padx=8)
        ttk.Label(bar,text='自动停止：').pack(side='left');ttk.Spinbox(bar,textvariable=self.duration,from_=5,to=90,width=5).pack(side='left');ttk.Label(bar,text='秒').pack(side='left')
        ttk.Label(pad,textvariable=self.status,wraplength=1080,font=('Microsoft YaHei',11)).pack(anchor='w',pady=10)
        views=ttk.Frame(pad);views.pack(fill='both',expand=True)
        self.color_view=ttk.Label(views,text='彩色画面',anchor='center');self.color_view.pack(side='left',fill='both',expand=True)
        self.depth_view=ttk.Label(views,text='深度预览（彩色仅用于显示，原始数据另存）',anchor='center');self.depth_view.pack(side='left',fill='both',expand=True)
        ttk.Label(pad,textvariable=self.detail,wraplength=1080).pack(anchor='w',pady=8)
        ttk.Label(pad,textvariable=self.counter,font=('Microsoft YaHei',17,'bold')).pack(anchor='w',pady=8)
        actions=ttk.Frame(pad);actions.pack(fill='x')
        self.start=ttk.Button(actions,text='10 秒倒计时后开始',command=self.countdown,state='disabled');self.start.pack(side='left')
        self.stop=ttk.Button(actions,text='停止 / 取消倒计时',command=self.stop_record,state='disabled');self.stop.pack(side='left',padx=8)
        self.verify=ttk.Button(actions,text='校验刚录的文件',command=self.check,state='disabled');self.verify.pack(side='left')
        ttk.Button(actions,text='查看保存文件夹',command=lambda:os.startfile(str(ROOT/'captures'))).pack(side='left',padx=8)
        ttk.Label(pad,text='先试录 5 秒并校验。正式拍摄：提示音后正面停 2–3 秒，缓慢转动身体一圈；到时间会自动停止。\n摘眼镜采集可在倒计时期间完成。勿戴耳机，勿说话，保持中性表情；头顶、耳朵和下巴都留在画面内。',wraplength=1080).pack(anchor='w',pady=(12,0))
        root.protocol('WM_DELETE_WINDOW',self.close);root.after(70,self.refresh)
    def open(self):
        if self.connected:return
        self.connect.config(state='disabled');self.status.set('正在连接，请稍候…');self.jobs.put(('open',))
    def choose_clip(self):
        label=self.label.get()
        self.duration.set(5 if label=='5 秒试录' else 20 if label=='戴眼镜参考' else 45)
        if label!='5 秒试录':self.glasses.set('戴眼镜' if label=='戴眼镜参考' else '无眼镜')
    def countdown(self):
        if not self.connected or time.perf_counter()-self.last_frame_time>2:
            messagebox.showerror('还不能录制','请先确认彩色与深度都在更新。');return
        if self.latest is None or not np.any(self.latest[1]>0) or not np.isfinite(self.latest[2]) or self.latest[2]<=0:
            messagebox.showerror('还没有有效深度','深度画面没有有效数值。请调整距离、检查设置，或用官方 Viewer 排查。');return
        if shutil.disk_usage(ROOT).free<10*1024**3:
            messagebox.showerror('剩余空间不足','请至少预留 10 GB；建议 30 GB，原始深度录像较大。');return
        self.counting=True;self.start.config(state='disabled');self.verify.config(state='disabled');self.stop.config(state='normal')
        self.tick_countdown(10)
    def tick_countdown(self,seconds):
        if not self.counting or self.closing:return
        self.counter.set(f'{seconds} 秒后开始，可现在摘眼镜并坐好。')
        if seconds<=3 and seconds>0:winsound.MessageBeep()
        if seconds<=0:
            self.counting=False;self.counter.set('正在开始录制…')
            try:duration=int(self.duration.get())
            except Exception:duration=45
            duration=max(5,min(90,duration));self.planned_duration=duration
            self.jobs.put(('record',self.label.get(),self.glasses.get(),duration));return
        self.root.after(1000,lambda:self.tick_countdown(seconds-1))
    def stop_record(self):
        if self.counting:
            self.counting=False;self.start.config(state='normal');self.stop.config(state='disabled');self.counter.set('倒计时已取消');return
        self.stop.config(state='disabled');self.status.set('正在停止并完成文件写入，请勿拔摄像头…');self.jobs.put(('stop',))
    def check(self):
        if self.current_path:
            self.verify.config(state='disabled');self.start.config(state='disabled');self.status.set('正在回读检查：是否同时保存彩色和原始深度…')
            self.jobs.put(('verify',self.current_path/'capture.bag'))
    def engine(self):
        camera=None;record=None
        try:
            while True:
                try:job=self.jobs.get_nowait()
                except queue.Empty:job=None
                if job:
                    action=job[0]
                    try:
                        if action=='close':break
                        if action=='open':camera=GeminiCamera();metadata=camera.open();self.events.put(('connected',metadata))
                        elif action=='record':
                            if camera is None or time.perf_counter()-self.last_frame_time>2:raise RuntimeError('画面已停滞，不能开始录制。')
                            record=Recording(camera,*job[1:]);self.events.put(('recording',record.path,record.started))
                        elif action=='stop' and record:
                            path=record.finish();record=None;self.events.put(('stopped',path))
                        elif action=='verify':self.events.put(('verified',verify_bag(job[1])))
                    except Exception as exc:
                        self.events.put(('error',str(exc)))
                        if action=='open':
                            if camera:camera.close()
                            camera=None
                if camera is not None and camera.started:
                    try:
                        frame=camera.read()
                        if frame is not None:
                            self.latest=frame;self.last_frame_time=time.perf_counter()
                            if record:
                                record.observe(frame)
                                if time.perf_counter()-record.started>=record.metadata['planned_duration_seconds']:
                                    path=record.finish();record=None;self.events.put(('stopped',path))
                    except Exception as exc:
                        if record:record.finish(str(exc));record=None
                        camera.close();camera=None;self.events.put(('camera_error',str(exc)))
                else:time.sleep(.04)
        finally:
            if record:record.finish()
            if camera:camera.close()
            self.events.put(('closed',))
    def refresh(self):
        while not self.events.empty():
            event=self.events.get();kind=event[0]
            if kind=='connected':
                self.connected=True;meta=event[1];self.start.config(state='normal')
                self.status.set(f"已连接 {meta['device_name']} · {meta['usb_type']} · 相机标定参数已读取。")
            elif kind=='recording':
                self.recording=True;self.current_path=event[1];self.record_start=event[2];self.status.set('正在录制彩色＋深度：'+self.current_path.name);winsound.MessageBeep()
            elif kind=='stopped':
                self.recording=False;self.current_path=event[1];self.counter.set('本段已保存，请校验后再录下一段。');winsound.MessageBeep()
                self.start.config(state='normal');self.stop.config(state='disabled');self.verify.config(state='normal');self.status.set('已保存：'+str(self.current_path))
            elif kind=='verified':
                result=event[1];self.verify.config(state='normal');self.start.config(state='normal')
                self.status.set('短片回读检查通过：彩色与原始深度都能读取。' if result['passed'] else '回读检查未通过，请用官方 Viewer 回放检查，勿删除原始文件。')
            elif kind in ['error','camera_error']:
                self.status.set('出现问题：'+event[1]);self.counting=False
                if kind=='camera_error':self.connected=False;self.recording=False
                self.connect.config(state='disabled' if self.connected else 'normal');self.stop.config(state='disabled');self.start.config(state='normal' if self.connected and not self.recording else 'disabled')
            elif kind=='closed':self.root.destroy();return
        if self.closing:self.root.after(70,self.refresh);return
        if self.latest is not None:
            picture,depth,scale,stamps=self.latest
            mm=depth.astype(np.float32)*scale;valid=depth>0
            view=np.clip((mm-200)/1100*255,0,255).astype(np.uint8)
            view=cv2.applyColorMap(view,cv2.COLORMAP_TURBO);view[~valid]=0
            for label,array in [(self.color_view,picture),(self.depth_view,view)]:
                image=Image.fromarray(cv2.cvtColor(array,cv2.COLOR_BGR2RGB));image.thumbnail((520,420))
                label.photo=ImageTk.PhotoImage(image);label.config(image=label.photo,text='')
            h,w=depth.shape;center=depth[int(h*.2):int(h*.7),int(w*.35):int(w*.65)]
            values=center[center>0]*scale;median=float(np.median(values)) if len(values) else 0
            percent=(center>0).mean()*100
            stale=time.perf_counter()-self.last_frame_time
            self.detail.set(f'彩色 {picture.shape[1]}×{picture.shape[0]} · 深度 {w}×{h} · 中央区域有效深度 {percent:.0f}% · 中央距离约 {median/1000:.2f} m · 配对时间差 {stamps["pair_delta_ms"]:.1f} ms'+(' · 画面停止更新，请检查 USB！' if stale>2 else ''))
        if self.recording:self.counter.set(f'正在录制 {int(time.perf_counter()-self.record_start)} / {self.planned_duration} 秒 · 保持表情，缓慢转动身体')
        self.root.after(100,self.refresh)
    def close(self):
        self.closing=True;self.counting=False;self.start.config(state='disabled');self.stop.config(state='disabled');self.status.set('正在完成文件并关闭相机…');self.jobs.put(('close',))

def main():
    p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');args=p.parse_args()
    if args.check:
        print(json.dumps({'runtime_loaded':True,'devices':NativeSDK().probe(),'hardware_recording_tested':False},ensure_ascii=False));return
    root=tk.Tk();App(root);root.mainloop()

if __name__=='__main__':main()
