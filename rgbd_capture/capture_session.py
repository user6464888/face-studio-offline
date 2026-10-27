"""Keep raw RGB-D bags, calibration, native depth units and observation timestamps."""
from pathlib import Path
from datetime import datetime
import csv
import json
import time
import cv2

ROOT=Path(__file__).resolve().parent

class Recording:
    def __init__(self,camera,label,glasses,duration=45):
        self.camera=camera;self.started=time.perf_counter();self.rows=0
        self.path=ROOT/'captures'/('gemini_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
        self.path.mkdir(parents=True)
        self.metadata=dict(camera.metadata)
        self.metadata.update({'session_id':self.path.name,'label':label,'glasses':glasses,'planned_duration_seconds':duration,
            'created_local':datetime.now().astimezone().isoformat(),'closed_cleanly':False,
            'timestamps_csv_scope':'preview frames observed by this application; bag is recorded by SDK',
            'sensor_depth':True,'external_upload':False})
        self.file=(self.path/'observed_pairs.csv').open('w',encoding='utf-8',newline='')
        self.writer=csv.DictWriter(self.file,fieldnames=['elapsed_seconds','color_device_us','depth_device_us','color_system_ms','depth_system_ms','pair_delta_ms','pair_delta_source','color_frame_index','depth_frame_index'])
        self.writer.writeheader();self.delta=[]
        self.save_metadata()
        try:self.camera.start_record(self.path/'capture.bag')
        except Exception:
            self.metadata['start_failed']=True;self.file.close();self.save_metadata();raise
    def save_metadata(self):
        (self.path/'session.json').write_text(json.dumps(self.metadata,ensure_ascii=False,indent=2),encoding='utf-8')
    def observe(self,frame):
        picture,depth,scale,stamps=frame
        self.writer.writerow({'elapsed_seconds':round(time.perf_counter()-self.started,6),**stamps});self.rows+=1
        self.delta.append(stamps['pair_delta_ms'])
        if self.rows==1:
            cv2.imwrite(str(self.path/'first_color.jpg'),picture,[cv2.IMWRITE_JPEG_QUALITY,95])
            cv2.imwrite(str(self.path/'first_depth_raw.png'),depth)
            self.metadata.update({'depth_value_scale_mm':scale,'sample_depth_png_dtype':'uint16',
                'depth_conversion':'depth_mm = stored_uint16 * depth_value_scale_mm',
                'pair_delta_clock':stamps.get('pair_delta_source','device_clock'),
                'timestamp_note':'Device clock and host reception clock are stored separately; a zero device time is unavailable, not exact hardware synchronization.'})
            self.save_metadata()
    def finish(self,error=None):
        try:self.camera.stop_record()
        except Exception as exc:error=str(exc)
        finally:self.file.close()
        bag=self.path/'capture.bag'
        self.metadata.update({'actual_duration_seconds':round(time.perf_counter()-self.started,3),
            'observed_preview_pairs':self.rows,'observed_pairs_per_second':round(self.rows/max(time.perf_counter()-self.started,.01),2),
            'bag_bytes':bag.stat().st_size if bag.exists() else 0,'closed_cleanly':error is None,
            'mean_observed_pair_delta_ms':sum(self.delta)/len(self.delta) if self.delta else None,
            'max_observed_pair_delta_ms':max(self.delta) if self.delta else None,
            'recording_error':error})
        self.save_metadata();return self.path

def verify_bag(path,minimum_pairs=5):
    from orbbec_native import GeminiCamera
    path=Path(path);camera=GeminiCamera();s=camera.sdk;depth=color=None
    try:
        camera.pipe=s.call('ob_create_pipeline_with_playback_file',str(path.resolve()).encode('utf-8'))
        depth,di=camera.pick_profile(3,(1280,800),[8]);color,ci=camera.pick_profile(2,(1920,1080),[5,22,23,0,1,2,3,4])
        camera.config=s.call('ob_create_config');s.call('ob_config_disable_all_stream',camera.config)
        s.call('ob_config_enable_stream',camera.config,depth);s.call('ob_config_enable_stream',camera.config,color)
        s.call('ob_config_set_frame_aggregate_output_mode',camera.config,0)
        s.call('ob_pipeline_start_with_config',camera.pipe,camera.config);camera.started=True
        count=0;scale=None;valid=0;last=None;deadline=time.perf_counter()+8
        while count<minimum_pairs and time.perf_counter()<deadline:
            frame=camera.read(200)
            if frame is None:continue
            count+=1;scale=frame[2];valid+=int((frame[1]>0).sum());last=frame
        result={'bag':str(path),'check_kind':'short RGB-D readback sample','paired_frames_read':count,
                'depth_value_scale_mm':scale,'sample_valid_depth_pixels':valid,
                'color_profile':ci,'depth_profile':di,'passed':count>=minimum_pairs and valid>0 and bool(scale and scale>0),
                'entire_file_verified':False,'timestamp_local':datetime.now().astimezone().isoformat()}
        try:
            result['calibration_readback']=camera.get_calibration(ci,di)
            result['calibration_valid']=True
        except Exception as exc:
            result['calibration_readback_error']=str(exc)
            result['calibration_valid']=False
            result['passed']=False
        (path.parent/'readback_check.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
        return result
    finally:
        if depth:s.call('ob_delete_stream_profile',depth)
        if color:s.call('ob_delete_stream_profile',color)
        camera.close()

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('bag');args=parser.parse_args()
    print(json.dumps(verify_bag(args.bag),ensure_ascii=False,indent=2))
