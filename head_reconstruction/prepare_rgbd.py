"""Prepare paired native RGB-D inputs from the user's selected local recordings."""
import argparse
from datetime import datetime, timezone, timedelta
import json
import math
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
CAPTURE = ROOT.parent / 'rgbd_capture'
sys.path.insert(0, str(CAPTURE))
from orbbec_native import GeminiCamera,frame_timing
from rgbd_context import CONFIG,input_folder


def prepare(role, window=.4):
    if CONFIG:
        bag=Path(CONFIG['recordings'][role]);source=bag.parent;output=input_folder(role)
        sidecar=source/'session.json'
        session=json.loads(sidecar.read_text(encoding='utf-8')) if sidecar.exists() and bag.name=='capture.bag' else {}
    else:
        selection = json.loads((CAPTURE / 'reconstruction_selection.json').read_text(encoding='utf-8'))
        clip = next(c for c in selection['clips'] if c['role'] == role)
        source = (CAPTURE / clip['session_metadata']).parent;bag=source/'capture.bag'
        session = json.loads((source / 'session.json').read_text(encoding='utf-8'))
        output = ROOT / 'inputs' / ('rgbd_' + clip['session_id'])
    if session and not session.get('closed_cleanly'):
        raise RuntimeError('原始录像未正常保存。')
    if output.exists():
        previous_path=output/'manifest.json'
        previous=json.loads(previous_path.read_text(encoding='utf-8')) if previous_path.exists() else {}
        empty_folders=all((output/name).is_dir() and not any((output/name).iterdir()) for name in ('images','depth'))
        if previous.get('preparation_status')!='failed' or previous.get('source_session')!=str(source) or not empty_folders:
            raise RuntimeError(f'输出已存在，保留现有结果：{output}')
    (output / 'images').mkdir(parents=True,exist_ok=True)
    (output / 'depth').mkdir(exist_ok=True)
    manifest = {
        'schema_version': 1, 'preparation_status': 'in_progress',
        'created_local': datetime.now(timezone(timedelta(hours=8))).isoformat(),
        'source_session': str(source), 'source_bag': str(bag),
        'role': role, 'source_label': session.get('label',bag.stem), 'modality': 'RGB-D',
        'has_sensor_depth': True, 'window_seconds': window,
        'selection_note': 'Sharpest upper-central RGB candidate per temporal window; this is a heuristic, not pose estimation or head segmentation.',
        'color_profile': session.get('color_profile'), 'depth_profile': session.get('depth_profile'),
        'camera_calibration': session.get('calibration'),
        'calibration_source': 'original live sidecar checked against BAG' if session else 'BAG running playback calibration',
        'depth_value_scale_mm': session.get('depth_value_scale_mm'),
        'source_closed_cleanly_from_sidecar': session.get('closed_cleanly'),
        'depth_conversion': 'depth_mm = stored_uint16 * depth_value_scale_mm',
        'depth_storage': 'lossless uint16 PNG in native depth camera coordinates',
        'color_storage': 'original resolution, decoded BGR saved as JPEG quality 95; original encoded frames remain in source BAG',
        'alignment': 'raw separate depth and RGB coordinate systems; not aligned yet',
        'saved_images_mirrored': False,
        'poses_estimated': False, 'masks_generated': False,
        'maximum_accepted_pair_delta_ms': 20,
        'images': [],
    }
    def save():
        (output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    save()
    camera = GeminiCamera();s = camera.sdk;depth = color = None
    decoded = 0;rejected = 0;bucket = None;best = None;first = last = None
    previews = [];last_report = 0

    def flush(candidate):
        if candidate is None:return
        picture, array, scale, stamps, elapsed, score = candidate
        index = len(manifest['images'])
        name = f'pair_{index:05d}'
        rgb_path = output / 'images' / (name + '.jpg')
        depth_path = output / 'depth' / (name + '.png')
        if not cv2.imwrite(str(rgb_path), picture, [cv2.IMWRITE_JPEG_QUALITY, 95]):
            raise RuntimeError('彩色图片写入失败。')
        if not cv2.imwrite(str(depth_path), array):raise RuntimeError('深度图片写入失败。')
        restored = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if restored is None or restored.dtype != np.uint16 or not np.array_equal(restored, array):
            raise RuntimeError('保存的深度数值与原始帧不同。')
        manifest['images'].append({
            'file': f'images/{name}.jpg', 'depth_file': f'depth/{name}.png',
            'elapsed_seconds': round(elapsed, 6), 'sharpness_reference': score,
            'width': picture.shape[1], 'height': picture.shape[0],
            'depth_width': array.shape[1], 'depth_height': array.shape[0],
            'depth_value_scale_mm': scale, 'valid_depth_pixels': int((array > 0).sum()),
            **stamps,
        })
        mm = array.astype(np.float32) * scale
        display = cv2.applyColorMap(np.clip((mm-200)/1100*255, 0, 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        display[array == 0] = 0
        previews.append((elapsed, cv2.resize(picture, (320, 180)), cv2.resize(display, (320, 200))))

    try:
        camera.pipe = s.call('ob_create_pipeline_with_playback_file', str(bag).encode('utf-8'))
        depth, di = camera.pick_profile(3, (1280,800), [8])
        color, ci = camera.pick_profile(2, (1920,1080), [5,22,23,0,1,2,3,4])
        if session and (ci != session['color_profile'] or di != session['depth_profile']):
            raise RuntimeError('录像分辨率与保存的相机参数不一致。')
        camera.config = s.call('ob_create_config')
        s.call('ob_config_disable_all_stream', camera.config)
        s.call('ob_config_set_align_mode', camera.config, 0)
        s.call('ob_config_enable_stream', camera.config, depth)
        s.call('ob_config_enable_stream', camera.config, color)
        s.call('ob_config_set_frame_aggregate_output_mode', camera.config, 0)
        # SDK 1.8.3 playback has no hardware device and cannot enable its live
        # frame-sync API. Keep the saved timestamps and reject large pair gaps.
        s.call('ob_pipeline_start_with_config', camera.pipe, camera.config);camera.started = True
        live_check = camera.get_calibration(ci, di)
        for key in (('depthIntrinsic','rgbIntrinsic','depthDistortion','rgbDistortion','transform') if session else ()):
            if not np.allclose(list(live_check[key].values()) if key != 'transform' else live_check[key]['rot']+live_check[key]['trans'],
                               list(session['calibration'][key].values()) if key != 'transform' else session['calibration'][key]['rot']+session['calibration'][key]['trans'],
                               rtol=1e-5, atol=1e-4):
                raise RuntimeError(f'录像回读标定与采集时参数不同：{key}')
        manifest['playback_calibration'] = live_check
        manifest.update(color_profile=ci,depth_profile=di,camera_calibration=session.get('calibration',live_check))
        manifest['playback_mirror_flag_differs_from_live'] = live_check['isMirrored'] != session.get('calibration',live_check)['isMirrored']
        manifest['orientation_note'] = 'No pixel flipping applied. SDK playback mirror flag and original live flag are both retained; verify pixel orientation during RGB-depth alignment.'
        started = time.perf_counter();last_received = started
        deadline = started + max(60,session.get('actual_duration_seconds',1800)+20)
        while time.perf_counter() < deadline:
            frame = camera.read(200)
            if frame is None:
                if first is not None and time.perf_counter()-last_received > 3:break
                if first is None and time.perf_counter()-started>15:raise RuntimeError('录像没有可读取的 RGB-D 帧。')
                continue
            last_received = time.perf_counter()
            picture, array, scale, stamps = frame
            ts,clock,_ = frame_timing(stamps)
            used_clocks=manifest.setdefault('elapsed_timestamp_sources',[])
            if clock not in used_clocks:used_clocks.append(clock)
            if len(used_clocks)>1:raise RuntimeError('录像的时间来源在处理中变化，停止整理。')
            if clock=='host_receive_clock':
                manifest['timing_note']='Device timestamps unavailable. Use recorded host reception times for selection; pair deltas describe host reception, not measured hardware capture synchronization.'
            if first is None:first = ts
            if last is not None and ts <= last:
                raise RuntimeError('录像彩色时间戳未保持递增，停止整理。')
            last = ts;decoded += 1;elapsed = (ts-first)/1e6
            next_bucket = int(elapsed/window)
            if next_bucket != bucket:
                flush(best);bucket = next_bucket;best = None
            if stamps['pair_delta_ms'] > 20 or not np.any(array > 0):
                rejected += 1;continue
            expected_scale=manifest['depth_value_scale_mm']
            if not math.isfinite(scale) or scale<=0 or (expected_scale is not None and not math.isclose(scale,expected_scale,rel_tol=1e-5)):
                raise RuntimeError('录像中的深度单位发生变化，停止整理。')
            manifest['depth_value_scale_mm']=scale
            h,w = picture.shape[:2]
            roi = picture[int(h*.12):int(h*.7),int(w*.28):int(w*.72)]
            gray = cv2.resize(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), (240,180), interpolation=cv2.INTER_AREA)
            score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            if best is None or score > best[-1]:best = (picture,array,scale,stamps,elapsed,score)
            if elapsed-last_report >= 10:
                print(json.dumps({'role':role,'elapsed_seconds':round(elapsed,1),'selected_pairs':len(manifest['images'])}),flush=True)
                last_report = elapsed
        flush(best)
        span = (last-first)/1e6 if first is not None else 0
        if time.perf_counter()>=deadline:raise RuntimeError('录像读取超过时限，未确认结束，停止处理。')
        if span < session.get('actual_duration_seconds',0)*.9 or len(manifest['images']) < 30:
            raise RuntimeError('可读取的录像跨度或有效帧数量不足。')
        elapsed_values = [v['elapsed_seconds'] for v in manifest['images']]
        manifest.update({'preparation_status':'complete','source_paired_frames_decoded':decoded,
                         'source_read_span_seconds':span,'rejected_pairs':rejected,
                         'selected_count':len(manifest['images']),
                         'maximum_selected_time_gap_seconds':max(b-a for a,b in zip(elapsed_values,elapsed_values[1:])),
                         'rgb_depth_alignment_verified':False})
        save()
        samples = [previews[i] for i in np.linspace(0,len(previews)-1,12).astype(int)]
        sheet = Image.new('RGB',(1280,3*414),'#f3f6fa');draw = ImageDraw.Draw(sheet)
        for i,(elapsed,rgb,dep) in enumerate(samples):
            x = i%4*320;y = i//4*414
            draw.text((x+10,y+4),f'{elapsed:.2f}s | RGB + raw depth preview',fill='#233243')
            sheet.paste(Image.fromarray(cv2.cvtColor(rgb,cv2.COLOR_BGR2RGB)),(x,y+24))
            sheet.paste(Image.fromarray(cv2.cvtColor(dep,cv2.COLOR_BGR2RGB)),(x,y+208))
        sheet.save(output/'overview.jpg',quality=93)
        (output/'README.md').write_text(
            f'# RGB-D 重建输入\n\n来源：{manifest["source_label"]}，角色：{role}。\n\n'
            f'读取约 {span:.2f} 秒、{decoded} 个配对帧，整理 {len(manifest["images"])} 组彩色与原始深度。'
            f'每 {window} 秒选择中心区域清晰度最高的一组，此指标仅为筛帧启发式。\n\n'
            'images/ 为原尺寸彩色图；depth/ 为数值验证一致的 uint16 深度；manifest.json 保存逐组对应关系、时间戳和相机参数；overview.jpg 为抽样预览。\n\n'
            '两路仍分别位于彩色与深度坐标系。此阶段尚未分割头部、估计头部位姿或融合完整人头。'
            '下一步须验证像素方向与标定投影、分割头部，再对齐多视角并融合深度。原始 BAG 保留在采集目录。\n',encoding='utf-8')
        print(json.dumps({'role':role,'decoded_pairs':decoded,'selected_pairs':len(manifest['images']),
                          'source_span_seconds':span,'output':str(output)},ensure_ascii=False),flush=True)
        return output
    except Exception as exc:
        manifest['preparation_status'] = 'failed';manifest['error'] = str(exc);save();raise
    finally:
        if depth:s.call('ob_delete_stream_profile', depth)
        if color:s.call('ob_delete_stream_profile', color)
        camera.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser();parser.add_argument('--role',choices=['primary','supplement_top'],default='primary')
    parser.add_argument('--window',type=float,default=.4);args = parser.parse_args()
    if not 0.1 <= args.window <= 2:parser.error('--window must be between 0.1 and 2 seconds')
    prepare(args.role,args.window)
