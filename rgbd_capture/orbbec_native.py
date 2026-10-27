"""Minimal local ctypes bridge for the lab's official Orbbec SDK 1.8.3."""
from pathlib import Path
import ctypes as C
import os
import json
import math
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parent
P=C.c_void_p;E=C.POINTER(P);I=C.c_int;U=C.c_uint32

class Intrinsic(C.Structure):
    _fields_=[('fx',C.c_float),('fy',C.c_float),('cx',C.c_float),('cy',C.c_float),('width',C.c_int16),('height',C.c_int16)]
class Distortion(C.Structure):
    _fields_=[(k,C.c_float) for k in ['k1','k2','k3','k4','k5','k6','p1','p2']]
class Transform(C.Structure):
    _fields_=[('rot',C.c_float*9),('trans',C.c_float*3)]
class CameraParam(C.Structure):
    _fields_=[('depthIntrinsic',Intrinsic),('rgbIntrinsic',Intrinsic),('depthDistortion',Distortion),('rgbDistortion',Distortion),('transform',Transform),('isMirrored',C.c_bool)]

def struct_dict(value):
    result={}
    for name,_ in value._fields_:
        item=getattr(value,name)
        result[name]=struct_dict(item) if isinstance(item,C.Structure) else list(item) if isinstance(item,C.Array) else item
    return result

def decode_color(raw,width,height,fmt):
    buffer=np.frombuffer(raw,np.uint8)
    if fmt==5:
        image=cv2.imdecode(buffer,cv2.IMREAD_COLOR)
        if image is None:raise RuntimeError('无法解码彩色帧。请在官方 Viewer 中选择 MJPG/RGB 格式。')
        return image
    if fmt in (22,23):
        image=buffer.reshape(height,width,3)
        return cv2.cvtColor(image,cv2.COLOR_RGB2BGR) if fmt==22 else image.copy()
    if fmt in (0,1,2):
        code=cv2.COLOR_YUV2BGR_UYVY if fmt==2 else cv2.COLOR_YUV2BGR_YUY2
        return cv2.cvtColor(buffer.reshape(height,width,2),code)
    if fmt in (3,4):
        return cv2.cvtColor(buffer.reshape(height*3//2,width),cv2.COLOR_YUV2BGR_NV12 if fmt==3 else cv2.COLOR_YUV2BGR_NV21)
    raise RuntimeError(f'不支持预览的彩色格式 {fmt}，请改用官方 Viewer。')

def frame_timing(stamps):
    """Preserve the clock domain; zero device times are not valid sync evidence."""
    color=stamps.get('color_device_us',0);depth=stamps.get('depth_device_us',0)
    if color>0 and depth>0:return color,'device_clock',abs(color-depth)/1000
    color=stamps.get('color_system_ms',0);depth=stamps.get('depth_system_ms',0)
    if color>0 and depth>0:return color*1000,'host_receive_clock',abs(color-depth)
    raise ValueError('录像的设备时间和主机接收时间均不可用。')


class NativeSDK:
    def __init__(self):
        if os.name!='nt':raise RuntimeError('This Orbbec SDK bridge supports Windows only.')
        folder=Path(os.environ.get('ORBBEC_SDK_DIR',ROOT/'tools/gemini_sdk'))
        if not (folder/'OrbbecSDK.dll').is_file():raise RuntimeError('Install Orbbec SDK 1.8.3 and set ORBBEC_SDK_DIR; see docs/RECONSTRUCTION.md.')
        self.dll_directory=os.add_dll_directory(str(folder))
        self.lib=C.CDLL(str(folder/'OrbbecSDK.dll'))
        declarations={
            'ob_error_message':(C.c_char_p,[P]),'ob_delete_error':(None,[P]),
            'ob_set_logger_severity':(None,[I,E]),'ob_set_logger_to_file':(None,[I,C.c_char_p,E]),
            'ob_create_context':(P,[E]),'ob_delete_context':(None,[P,E]),'ob_query_device_list':(P,[P,E]),
            'ob_device_list_device_count':(U,[P,E]),'ob_device_list_get_device_name':(C.c_char_p,[P,U,E]),
            'ob_delete_device_list':(None,[P,E]),
            'ob_create_pipeline':(P,[E]),'ob_create_pipeline_with_playback_file':(P,[C.c_char_p,E]),
            'ob_delete_pipeline':(None,[P,E]),'ob_pipeline_get_device':(P,[P,E]),
            'ob_pipeline_get_stream_profile_list':(P,[P,I,E]),'ob_pipeline_enable_frame_sync':(None,[P,E]),
            'ob_pipeline_start_with_config':(None,[P,P,E]),'ob_pipeline_start':(None,[P,E]),
            'ob_pipeline_wait_for_frameset':(P,[P,U,E]),'ob_pipeline_stop':(None,[P,E]),
            'ob_pipeline_start_record':(None,[P,C.c_char_p,E]),'ob_pipeline_stop_record':(None,[P,E]),
            'ob_pipeline_get_camera_param_with_profile':(CameraParam,[P,U,U,U,U,E]),
            'ob_pipeline_get_camera_param':(CameraParam,[P,E]),
            'ob_create_config':(P,[E]),'ob_delete_config':(None,[P,E]),
            'ob_config_enable_stream':(None,[P,P,E]),'ob_config_disable_all_stream':(None,[P,E]),
            'ob_config_set_align_mode':(None,[P,I,E]),
            'ob_config_set_frame_aggregate_output_mode':(None,[P,I,E]),
            'ob_stream_profile_list_count':(U,[P,E]),'ob_stream_profile_list_get_profile':(P,[P,I,E]),
            'ob_delete_stream_profile_list':(None,[P,E]),'ob_delete_stream_profile':(None,[P,E]),
            'ob_video_stream_profile_width':(U,[P,E]),'ob_video_stream_profile_height':(U,[P,E]),
            'ob_video_stream_profile_fps':(U,[P,E]),'ob_stream_profile_format':(I,[P,E]),
            'ob_device_get_device_info':(P,[P,E]),'ob_delete_device':(None,[P,E]),'ob_delete_device_info':(None,[P,E]),
            'ob_device_info_name':(C.c_char_p,[P,E]),'ob_device_info_serial_number':(C.c_char_p,[P,E]),
            'ob_device_info_firmware_version':(C.c_char_p,[P,E]),'ob_device_info_usb_type':(C.c_char_p,[P,E]),
            'ob_frameset_color_frame':(P,[P,E]),'ob_frameset_depth_frame':(P,[P,E]),
            'ob_delete_frame':(None,[P,E]),'ob_frame_data':(P,[P,E]),'ob_frame_data_size':(U,[P,E]),
            'ob_video_frame_width':(U,[P,E]),'ob_video_frame_height':(U,[P,E]),
            'ob_frame_format':(I,[P,E]),'ob_frame_time_stamp_us':(C.c_uint64,[P,E]),
            'ob_frame_system_time_stamp':(C.c_uint64,[P,E]),'ob_frame_index':(C.c_uint64,[P,E]),
            'ob_depth_frame_get_value_scale':(C.c_float,[P,E]),
        }
        for name,(result,args) in declarations.items():
            fn=getattr(self.lib,name);fn.restype=result;fn.argtypes=args
        (ROOT/'logs').mkdir(exist_ok=True)
        self.call('ob_set_logger_severity',3)
        self.call('ob_set_logger_to_file',3,str(ROOT/'logs').encode('utf-8'))
    def call(self,name,*args):
        error=P();result=getattr(self.lib,name)(*args,C.byref(error))
        if error.value:
            message=self.lib.ob_error_message(error)
            self.lib.ob_delete_error(error)
            raise RuntimeError(f'{name}: {(message or b"SDK error").decode("utf-8","replace")}')
        return result
    def probe(self):
        context=self.call('ob_create_context');devices=None
        try:
            devices=self.call('ob_query_device_list',context)
            count=self.call('ob_device_list_device_count',devices)
            return [self.call('ob_device_list_get_device_name',devices,i).decode('utf-8','replace') for i in range(count)]
        finally:
            if devices:self.call('ob_delete_device_list',devices)
            self.call('ob_delete_context',context)

class GeminiCamera:
    def __init__(self):
        self.sdk=NativeSDK();self.pipe=None;self.config=None;self.started=False;self.recording=False;self.metadata={}
    def get_calibration(self,color_profile,depth_profile):
        # The running pipeline returns the parameters for the actual streams.
        # SDK 1.8.3's with_profile variant returned NaNs on our Gemini 2.
        calibration=struct_dict(self.sdk.call('ob_pipeline_get_camera_param',self.pipe))
        for key,profile in [('rgbIntrinsic',color_profile),('depthIntrinsic',depth_profile)]:
            values=calibration[key]
            if (values['width']!=profile['width'] or values['height']!=profile['height']
                    or not all(math.isfinite(values[k]) for k in ('fx','fy','cx','cy'))
                    or values['fx']<=0 or values['fy']<=0):
                raise RuntimeError('相机标定参数与采集分辨率不匹配，请改用官方 Viewer 记录并导出相机参数。')
        coefficients=list(calibration['depthDistortion'].values())+list(calibration['rgbDistortion'].values())
        transform=calibration['transform'];coefficients+=transform['rot']+transform['trans']
        if not all(math.isfinite(v) for v in coefficients):
            raise RuntimeError('相机标定含无效数值，不能用于重建。')
        rotation=np.asarray(transform['rot']).reshape(3,3)
        if not np.allclose(rotation.T@rotation,np.eye(3),atol=.01) or not .99<float(np.linalg.det(rotation))<1.01:
            raise RuntimeError('彩色与深度的相机旋转标定无效，不能用于重建。')
        return calibration
    def pick_profile(self,sensor,preferred,formats):
        s=self.sdk;profiles=s.call('ob_pipeline_get_stream_profile_list',self.pipe,sensor);candidates=[]
        try:
            for i in range(s.call('ob_stream_profile_list_count',profiles)):
                profile=s.call('ob_stream_profile_list_get_profile',profiles,i)
                values={k:s.call(f'ob_video_stream_profile_{k}',profile) for k in ['width','height','fps']}
                values['format']=s.call('ob_stream_profile_format',profile)
                if values['format'] in formats and values['fps']==30:
                    score=abs(values['width']-preferred[0])+abs(values['height']-preferred[1])+formats.index(values['format'])*30
                    candidates.append((score,profile,values))
                else:s.call('ob_delete_stream_profile',profile)
        finally:s.call('ob_delete_stream_profile_list',profiles)
        if not candidates:raise RuntimeError('没有可用的 30fps 彩色/原始深度组合，请使用官方 Viewer 检查设备模式。')
        candidates.sort(key=lambda x:x[0]);best=candidates[0]
        for _,profile,_ in candidates[1:]:s.call('ob_delete_stream_profile',profile)
        return best[1],best[2]
    def open(self):
        s=self.sdk
        try:
            devices=s.probe()
            if len(devices)!=1:raise RuntimeError('请只连接一台 Gemini 相机。当前识别到：'+str(devices))
            self.pipe=s.call('ob_create_pipeline')
            depth,depth_info=self.pick_profile(3,(1280,800),[8])
            color=None
            try:
                color,color_info=self.pick_profile(2,(1920,1080),[5,22,23,0,1,2,3,4])
                self.config=s.call('ob_create_config');s.call('ob_config_disable_all_stream',self.config)
                s.call('ob_config_set_align_mode',self.config,0)
                s.call('ob_config_enable_stream',self.config,depth);s.call('ob_config_enable_stream',self.config,color)
                s.call('ob_config_set_frame_aggregate_output_mode',self.config,0)
                s.call('ob_pipeline_enable_frame_sync',self.pipe)
                s.call('ob_pipeline_start_with_config',self.pipe,self.config);self.started=True
            finally:
                s.call('ob_delete_stream_profile',depth)
                if color:s.call('ob_delete_stream_profile',color)
            device=s.call('ob_pipeline_get_device',self.pipe);info=None
            try:
                info=s.call('ob_device_get_device_info',device)
                self.metadata={k:s.call('ob_device_info_'+fn,info).decode('utf-8','replace') for k,fn in
                    [('device_name','name'),('serial_number','serial_number'),('firmware_version','firmware_version'),('usb_type','usb_type')]}
            finally:
                if info:s.call('ob_delete_device_info',info)
                s.call('ob_delete_device',device)
            calibration=self.get_calibration(color_info,depth_info)
            self.metadata.update({'sdk_version':'1.8.3','color_profile':color_info,'depth_profile':depth_info,
                'calibration':calibration,'alignment':'raw depth and RGB, separate coordinate systems',
                'frame_sync_enabled':True,'streams_requested':['color','depth'],'depth_value_scale_mm':None,
                'sensor_depth':True,'preview_mirrored':False,'raw_data_file':'capture.bag'})
            self.metadata['calibration_source']='ob_pipeline_get_camera_param (running streams, alignment disabled)'
            return self.metadata
        except Exception:
            self.close();raise
    def read(self,timeout=200):
        s=self.sdk;frameset=s.call('ob_pipeline_wait_for_frameset',self.pipe,timeout)
        if not frameset:return None
        color=depth=None
        try:
            color=s.call('ob_frameset_color_frame',frameset);depth=s.call('ob_frameset_depth_frame',frameset)
            if not color or not depth:return None
            w=s.call('ob_video_frame_width',color);h=s.call('ob_video_frame_height',color);fmt=s.call('ob_frame_format',color)
            raw=C.string_at(s.call('ob_frame_data',color),s.call('ob_frame_data_size',color));picture=decode_color(raw,w,h,fmt)
            dw=s.call('ob_video_frame_width',depth);dh=s.call('ob_video_frame_height',depth)
            if s.call('ob_frame_format',depth)!=8:raise RuntimeError('深度帧不是 Y16 原始深度格式。')
            raw=C.string_at(s.call('ob_frame_data',depth),s.call('ob_frame_data_size',depth))
            array=np.frombuffer(raw,'<u2').reshape(dh,dw).copy()
            scale=float(s.call('ob_depth_frame_get_value_scale',depth));self.metadata['depth_value_scale_mm']=scale
            stamps={name:s.call(function,frame) for name,function,frame in
                [('color_device_us','ob_frame_time_stamp_us',color),('depth_device_us','ob_frame_time_stamp_us',depth),
                 ('color_system_ms','ob_frame_system_time_stamp',color),('depth_system_ms','ob_frame_system_time_stamp',depth),
                 ('color_frame_index','ob_frame_index',color),('depth_frame_index','ob_frame_index',depth)]}
            _,stamps['pair_delta_source'],stamps['pair_delta_ms']=frame_timing(stamps)
            return picture,array,scale,stamps
        finally:
            if color:s.call('ob_delete_frame',color)
            if depth:s.call('ob_delete_frame',depth)
            s.call('ob_delete_frame',frameset)
    def start_record(self,path):
        if self.recording:raise RuntimeError('Already recording')
        self.sdk.call('ob_pipeline_start_record',self.pipe,str(Path(path).resolve()).encode('utf-8'));self.recording=True
    def stop_record(self):
        if self.recording:self.sdk.call('ob_pipeline_stop_record',self.pipe);self.recording=False
    def close(self):
        try:self.stop_record()
        finally:
            if self.started:
                try:self.sdk.call('ob_pipeline_stop',self.pipe)
                finally:self.started=False
            if self.pipe:self.sdk.call('ob_delete_pipeline',self.pipe);self.pipe=None
            if self.config:self.sdk.call('ob_delete_config',self.config);self.config=None

if __name__=='__main__':
    sdk=NativeSDK();print(json.dumps({'dll_loaded':True,'camera_param_struct_bytes':C.sizeof(CameraParam),'connected_devices':sdk.probe()},ensure_ascii=False))
