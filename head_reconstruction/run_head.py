"""Offline, unattended video-to-head entry point for the installed local project."""
import argparse
from datetime import datetime,timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent
RGB_SUFFIXES={'.mp4','.avi','.mov','.mkv','.webm'}
VERSION=1
STAGE_LABELS={'prepare_primary':'读取主录像','prepare_supplement_top':'读取高位录像',
              'segment_primary':'分割主录像头部','register_primary':'对齐主录像',
              'segment_top':'筛选高位头部画面','anchor_top':'对齐高位画面',
              'fuse_initial':'生成测量表面','recover_top':'恢复更多高位视角',
              'fuse_recovered':'融合可靠视角','complete_crown':'补全顶部缺口',
              'validate':'检查模型','prepare_styles':'准备发型和发色','rgb_pipeline':'普通视频重建'}

def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8');temporary.replace(path)

def resolve_recording(path):
    path=Path(path).expanduser().resolve()
    if path.is_dir():path=path/'capture.bag'
    if not path.is_file():raise ValueError(f'找不到输入文件：{path}')
    if path.suffix.lower() not in RGB_SUFFIXES|{'.bag'}:raise ValueError('输入需要为 Gemini .bag 或 MP4/AVI/MOV/MKV/WebM 视频。')
    return path

def signature(path):
    stat=path.stat()
    # File identity + edge contents catch accidental source replacement without
    # repeatedly hashing multi-GB recordings on every restart.
    h=hashlib.sha256()
    with path.open('rb') as f:
        h.update(f.read(1024*1024));f.seek(max(0,stat.st_size-1024*1024));h.update(f.read())
    value={'path':str(path),'size':stat.st_size,'mtime_ns':stat.st_mtime_ns,'edge_sha256':h.hexdigest()}
    sidecar=path.parent/'session.json'
    if path.name=='capture.bag' and sidecar.exists():value['sidecar_sha256']=hashlib.sha256(sidecar.read_bytes()).hexdigest()
    return value

def default_inputs(folder):
    folder=folder.resolve()
    primary=next((p for p in folder.iterdir() if p.is_file() and p.name.lower()=='main.bag'),None)
    if primary:
        top=next((p for p in folder.iterdir() if p.is_file() and p.name.lower()=='top.bag'),None)
        return primary,top
    bags=[p for p in folder.iterdir() if p.is_file() and p.suffix.lower()=='.bag']
    videos=[p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in RGB_SUFFIXES]
    if len(bags)==1 and not videos:return bags[0],None
    if len(videos)==1:return videos[0],None
    raise ValueError('请在 rebuild_input 中放 main.bag（可加 top.bag），或仅放一段普通视频；也可用 --primary 和 --top 指定文件。')

class Job:
    def __init__(self,path,identity):
        if path.exists() and any(path.iterdir()) and not (path/'status.json').exists():raise ValueError('输出目录已有其他文件，请选择新的空目录。')
        self.path=path;self.path.mkdir(parents=True,exist_ok=True);self.status_path=path/'status.json'
        self.status=read(self.status_path) if self.status_path.exists() else {'identity':identity,'state':'pending','stages':{},'warnings':[]}
        if self.status['identity']!=identity:raise ValueError('输出目录属于不同输入或参数，请选择新的 --output 目录。')
        self.env=os.environ.copy();self.env.update(PYTHONUTF8='1',PYTHONUNBUFFERED='1',OMP_NUM_THREADS='8')
        self.stage='initializing';self.lock=path/'.running.lock';self.fd=None
    def acquire(self):
        # OS lock is released on process exit/crash, so resumes need no lock cleanup.
        self.fd=self.lock.open('a+b')
        if self.lock.stat().st_size==0:self.fd.write(b'0');self.fd.flush()
        self.fd.seek(0)
        if sys.platform=='win32':
            import msvcrt
            try:msvcrt.locking(self.fd.fileno(),msvcrt.LK_NBLCK,1)
            except OSError:raise ValueError('此任务正在运行，请勿同时启动同一个输出目录。')
        else:
            import fcntl
            fcntl.flock(self.fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
    def persist(self):save(self.status_path,self.status)
    def warning(self,text):
        print('提示：'+text,flush=True)
        if text not in self.status['warnings']:self.status['warnings'].append(text)
        self.persist()
    def step(self,name,script,args=(),expected=(),optional=False):
        self.stage=name
        previous=self.status['stages'].get(name,{})
        if previous.get('state') in ('complete','reused') and all(Path(p).exists() for p in expected):
            print(f'{STAGE_LABELS.get(name,name)}：复用已完成步骤',flush=True);return True
        self.status.update(state='running',current_stage=name)
        self.status['stages'][name]={'state':'running','started_utc':datetime.now(timezone.utc).isoformat()};self.persist()
        log=self.path/'logs'/f'{name}.log';log.parent.mkdir(exist_ok=True)
        command=[sys.executable,'-u',str(ROOT/script),*map(str,args)]
        label=STAGE_LABELS.get(name,name);print(label+'：开始',flush=True)
        with log.open('w',encoding='utf-8') as stream:
            process=subprocess.Popen(command,cwd=ROOT,env=self.env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace')
            try:
                for line in process.stdout:
                    stream.write(line);stream.flush()
                    if self.env.get('HEAD_VERBOSE')=='1':print(line,end='',flush=True)
                    else:
                        try:progress=json.loads(line)
                        except (ValueError,TypeError):continue
                        if isinstance(progress,dict):
                            if 'completed' in progress:print(f'{label}：{progress["completed"]} / {progress.get("total","—")}',flush=True)
                            elif 'elapsed_seconds' in progress:print(f'{label}：已读 {progress["elapsed_seconds"]:.1f} 秒',flush=True)
                code=process.wait()
            except BaseException:
                process.terminate();process.wait();raise
        success=code==0 and all(Path(p).exists() for p in expected)
        self.status['stages'][name].update(state='complete' if success else 'failed',exit_code=code,log=str(log));self.persist()
        if not success:
            if optional:self.warning(f'{label} 未通过检查，将保留可用主模型。详情：{log}');return False
            raise RuntimeError(f'{label} 未完成，请查看 {log}；重新运行相同命令可继续。')
        return True

def copy_preview(job,mode):
    folder=job.path/'preview';folder.mkdir(exist_ok=True)
    text='<!doctype html><meta charset="utf-8"><title>Local reconstruction</title><p>Reconstruction outputs are saved here. Open the salon page launched by main.py to view the model and 19 hairstyles.</p>'
    for name in ['index.html','rgbd.html']:(folder/name).write_text(text,encoding='utf-8')
    return folder

def adopt_geometry(job,cache,config):
    if not cache:return
    cache=Path(cache).resolve()
    for role,folder in config['inputs'].items():
        source=cache/role;manifest=read(Path(folder)/'manifest.json')
        if manifest.get('preparation_status')!='complete':continue
        old_config=cache.parent/'config.json'
        old_folder=Path(read(old_config)['inputs'][role]) if old_config.exists() else ROOT/'inputs'/('rgbd_'+Path(manifest['source_session']).name)
        if not (old_folder/'manifest.json').exists():raise ValueError('不能核对缓存原输入，请省略 --reuse-work 重新计算。')
        original=read(old_folder/'manifest.json')
        if original.get('camera_calibration')!=manifest.get('camera_calibration'):raise ValueError('缓存相机标定与新输入不同。')
        identity={'source_session':manifest['source_session'],'first_color_device_us':manifest['images'][0]['color_device_us']}
        if not (source/'source.json').exists() or read(source/'source.json')!=identity:raise ValueError('指定缓存与输入录像不一致，不能复用。')
        seg=read(source/'segmentation.json');dest=Path(config['work'])/role;dest.mkdir(parents=True,exist_ok=True)
        copied=[]
        for frame in seg['frames']:
            i=frame['index']
            if i>=len(manifest['images']) or abs(frame['time_seconds']-manifest['images'][i]['elapsed_seconds'])>1e-6:continue
            for key in ['file','depth_file']:
                old_image=old_folder/original['images'][i][key];new_image=Path(folder)/manifest['images'][i][key]
                if hashlib.sha256(old_image.read_bytes()).digest()!=hashlib.sha256(new_image.read_bytes()).digest():
                    raise ValueError(f'缓存原图或深度不同（{role}/{i}），请省略 --reuse-work。')
            p=source/f'{i:05d}.npz'
            if p.exists():shutil.copy2(p,dest/p.name);copied.append(i)
        save(dest/'source.json',identity)
        # Registration can only be reused for a fully matching primary sequence.
        if role=='primary' and copied==list(range(len(manifest['images']))) and (source/'registration.json').exists():
            reg=read(source/'registration.json')
            if reg.get('quality_passed') and len(reg['poses_camera_to_first_head'])==len(copied):
                shutil.copy2(source/'registration.json',dest/'registration.json')
                job.status['stages']['register_primary']={'state':'reused','source':str(source),'matched_frames':len(copied)}
        job.status.setdefault('cache_imports',{})[role]={'source':str(source),'matched_geometry_frames':len(copied)}
    job.persist()

def rgbd(job,a,primary,top):
    preview=copy_preview(job,'rgbd');work=job.path/'work';work.mkdir(exist_ok=True)
    config_path=job.path/'config.json'
    config=read(config_path) if config_path.exists() else {'work':str(work),'assets':str(preview/'assets/rgbd_b'),
        'recordings':{'primary':str(primary)},'inputs':{'primary':str(job.path/'inputs/primary')}}
    if top:config['recordings']['supplement_top']=str(top);config['inputs'].setdefault('supplement_top',str(job.path/'inputs/supplement_top'))
    job.env['HEAD_RGBD_CONFIG']=str(config_path)
    for role in config['recordings']:
        folder=Path(config['inputs'][role]);manifest=folder/'manifest.json'
        if manifest.exists() and read(manifest).get('preparation_status')=='complete':
            job.status['stages']['prepare_'+role]={'state':'reused','manifest':str(manifest)};job.persist();continue
        if folder.exists() and any(folder.iterdir()):
            folder=job.path/'inputs'/f'{role}_retry_{time.time_ns()}';config['inputs'][role]=str(folder)
        save(config_path,config)
        prepared=job.step('prepare_'+role,'prepare_rgbd.py',['--role',role,'--window',a.window],[folder/'manifest.json'],optional=role=='supplement_top')
        if not prepared and role=='supplement_top':top=None
    save(config_path,config)
    if a.reuse_work and not job.status.get('cache_imports'):adopt_geometry(job,a.reuse_work,config)
    job.step('segment_primary','reconstruct_rgbd.py',['segment','--role','primary'],[work/'primary/segmentation.json'])
    job.step('register_primary','reconstruct_rgbd.py',['register','--role','primary'],[work/'primary/registration.json'])
    use_top=False
    if top:
        segmented=job.step('segment_top','reconstruct_rgbd.py',['segment','--role','supplement_top'],[work/'supplement_top/segmentation.json'],optional=True)
        if segmented:use_top=job.step('anchor_top','reconstruct_rgbd.py',['register_top'],[work/'supplement_top/registration.json'],optional=True)
    assets=Path(config['assets']);assets.mkdir(parents=True,exist_ok=True)
    fuse_args=['fuse']+([] if use_top else ['--primary-only'])
    baseline=work/'before_top_recovery';baseline.mkdir(exist_ok=True)
    # If final exports already exist, preserve the original baseline on resume.
    if not (baseline/'head.ply').exists():
        job.step('fuse_initial','reconstruct_rgbd.py',fuse_args,[assets/'head.ply',assets/'metadata.json'])
        for name in ['head.ply','head.glb','metadata.json','model_overview.jpg']:shutil.copy2(assets/name,baseline/name)
    if use_top:
        if job.step('recover_top','improve_rgbd_top.py',['--apply'],[work/'supplement_top/recovery_candidate.json'],optional=True):
            job.step('fuse_recovered','reconstruct_rgbd.py',fuse_args,[assets/'head.ply',assets/'metadata.json'])
    if a.no_completion:
        if not (work/'sensor_recovered_head.ply').exists():shutil.copy2(assets/'head.ply',work/'sensor_recovered_head.ply')
    else:
        repair_args=['--refresh-source'] if not (work/'sensor_recovered_head.ply').exists() else []
        job.step('complete_crown','repair_rgbd_crown.py',repair_args,[work/'crown_patch.json',assets/'head.glb'])
    job.step('validate','validate_rgbd_top.py',expected=[work/'top_improvement_validation.json'])
    quality=read(work/'top_improvement_validation.json');meta=read(assets/'metadata.json')
    mandatory=all(quality[k] for k in ['ply_glb_counts_match','finite_geometry','measured_source_vertices_retained_exactly'])
    if not mandatory:raise RuntimeError('模型导出检查失败，查看 work/top_improvement_validation.json。')
    # Check the head hash and styling version on every run; geometry caches stay reusable.
    job.status['stages'].pop('prepare_styles',None)
    job.step('prepare_styles','prepare_hairstyles.py',
             ['--assets',assets,'--style',a.hair_style,'--color',a.hair_color]+(['--bald-only'] if a.salon else []),
             [assets/'styling.json',assets/'head_base.ply',assets/'head_scalp.ply',assets/'head_bald.ply',assets/'head_bald_before_shape.ply',assets/'bald_shape_fit.json',assets/'bald_scalp_mask.bin',assets/'bald_ear_mask.bin',assets/'bald_sideburn_mask.bin',assets/'bald_ear_rim_mask.bin',assets/'head_bald_uv.ply',assets/'head_bald_texture.png',assets/'head_bald_texture_faces.bin',assets/'skin_texture_sources.json',assets/'hair_weights.bin',assets/'scalp_hair_weights.bin',
              *[assets/f'hair_{kind}.ply' for kind in (() if a.salon else ('straight','bob','lob','fringe','part','wavy','waves','curly','ponytail'))]])
    if quality['remaining_crown_boundary_edges']:job.warning('顶部仍有不能自动封合的边界；原始曲面已保留，请查看预览。')
    if not use_top:job.warning('本次未采用高位深度；顶部缺少实测覆盖，补面可信度较低。')
    if a.no_completion or quality['remaining_crown_boundary_edges']:
        p=preview/'rgbd.html';text=p.read_text(encoding='utf-8')
        text=text.replace('头顶已补全。','本次头顶仍有缺口。')
        if a.no_completion:text=text.replace('剩余缺口按周围曲面估计，补面不代表实测发丝。点击“标出补面区域”可查看；','本次仅保留测量表面，未进行估计补面；')
        p.write_text(text,encoding='utf-8')
    report=(f'# 自动重建结果\n\n主输入：{primary}\n\n高位输入：{top or "未提供"}\n\n'
            f'采用水平 {meta["primary_views"]} 组，高位 {meta["top_views"]} 组；最终 {meta["faces"]} 个三角面。\n\n'
            '自动完成筛帧、标定核对、头部分割、配准、融合、着色、可选顶部估计补面和导出验证。\n\n'
            '估计补面可以在预览中标记；不是实测发丝。颈部截面保留开口。自动检查并不保证任意录像都能达到良好质量。\n\n'
            '预览支持九种离线发丝造型（3份基础素材与剪裁、梳理变体）、发色、蓬松度及并排对比，并可将当前造型保存为 GLB、PNG 或旋转 GIF。发丝素材按扫描尺寸与曲面适配，旧头发面从替换显示中移除；脸部、耳部和颈部保留，头皮、发束、长度及既有顶部补面为估计。颜色以采集头发为基础；新增发丝为素材，不是实测。素材来源 Cem Yuksel：https://www.cemyuksel.com/research/hairmodels/ ，限个人及科研使用。文件保存在 preview/exports。\n\n'
            f'提示：{json.dumps(job.status["warnings"],ensure_ascii=False)}\n\n逐步记录在 status.json 和 logs，几何检查在 work/top_improvement_validation.json。\n')
    if a.salon:
        start=report.index('预览支持九种')
        end=report.index('提示：',start)
        report=report[:start]+'本次只准备原始模型和光头底模。19 款混合发型、七种脸型标准头、可更正的几何初判及推荐，由 face_project/main.py 的 salon 界面提供。新增头皮和发型均包含估计，不是实测发丝。\n\n'+report[end:]
    (job.path/'README.md').write_text(report,encoding='utf-8');(assets/'report.md').write_text(report,encoding='utf-8')
    return preview,'rgbd.html',assets/'head.glb'

def import_rgb(job,video):
    import cv2
    import csv
    session_id='auto_rgb_'+job.path.name
    source=ROOT.parent/'head_capture/captures'/session_id
    if (source/'session.json').exists():return session_id
    source.mkdir(parents=True,exist_ok=True);stored=source/('video_01'+video.suffix.lower())
    if not stored.exists():shutil.copy2(video,stored)
    cap=cv2.VideoCapture(str(stored));fps=cap.get(cv2.CAP_PROP_FPS)
    if not cap.isOpened() or not 1<=fps<=240:raise ValueError('普通视频无法读取有效帧率。')
    count=0;previous=-1.;fallback=0
    try:
        with (source/'timestamps.csv').open('w',encoding='utf-8',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=['frame_index','elapsed_seconds','captured_at_utc']);writer.writeheader()
            while True:
                ok,frame=cap.read()
                if not ok:break
                elapsed=cap.get(cv2.CAP_PROP_POS_MSEC)/1000
                if count and elapsed<=previous:elapsed=max(count/fps,previous+1/fps);fallback+=1
                writer.writerow({'frame_index':count,'elapsed_seconds':elapsed,'captured_at_utc':''});previous=elapsed;count+=1
    finally:cap.release()
    if count<30:raise ValueError('普通视频有效帧太少。')
    save(source/'session.json',{'source_video':str(video),'timestamp_note':'decoded media timestamps; FPS fallback when non-increasing; no wall clock capture dates inferred',
         'fps_fallback_frames':fallback,'videos':[{'file':stored.name,'timestamps':'timestamps.csv','frames':count,'elapsed_seconds':previous}]})
    return session_id

def rgb(job,a,primary,top=None):
    preview=copy_preview(job,'rgb');assets=preview/'assets';assets.mkdir(exist_ok=True)
    job.env['HEAD_RGB_ASSETS']=str(assets)
    if top:
        from salon.rgb_pair import import_pair
        session_id=import_pair(job,primary,top)
    else:session_id=import_rgb(job,primary)
    job.step('rgb_pipeline','run_pipeline.py',['--session',session_id],[assets/'head.glb',assets/'metadata.json'])
    import numpy as np
    import trimesh
    mesh=trimesh.load(assets/'head.ply',process=False)
    if not len(mesh.faces) or not np.isfinite(mesh.vertices).all():raise RuntimeError('RGB 模型检查失败。')
    if a.salon:
        span=float(np.ptp(mesh.vertices[:,1]))
        if not .20<span<.80:
            mesh.export(assets/'head_reconstruction_units.ply')
            scale=.32/max(span,1e-8)
            mesh.vertices*=scale;mesh.export(assets/'head.ply')
            metadata=read(assets/'metadata.json')
            if 'rgb_projection' in metadata:metadata['rgb_projection']['display_scale']*=scale
            save(assets/'metadata.json',metadata)
        job.step('prepare_styles','prepare_hairstyles.py',['--assets',assets,'--bald-only'],
                 [assets/'head_bald.ply',assets/'head_bald_uv.ply',assets/'styling.json'])
    save(job.path/'validation.json',{'finite_geometry':True,'faces':len(mesh.faces),'sensor_depth':False,'scale_is_metric':False})
    job.warning('普通视频没有实测深度和尺寸标定；效果受视角、头部运动和光照影响。')
    upgraded=assets/'face_upgrade/head_textured.glb'
    model=upgraded if upgraded.exists() else assets/'head.glb'
    exported=trimesh.load(model,force='scene',process=False).to_geometry()
    if not len(exported.faces) or not np.isfinite(exported.vertices).all():raise RuntimeError('最终 RGB 模型导出检查失败。')
    return preview,'index.html',model

def open_preview(job,preview,page):
    raise RuntimeError('Use main.py to view the public salon; historical standalone viewers are not distributed.')


def main(argv=None):
    parser=argparse.ArgumentParser(description='提供视频，自动生成本地 3D 人头。Gemini 使用 .bag；普通 MP4/AVI 走 RGB 路线。')
    parser.add_argument('--primary',help='主录像文件，或含 capture.bag 的采集文件夹')
    parser.add_argument('--top',help='高位补拍；与主视频同类的 BAG 或普通视频')
    parser.add_argument('--input-folder',type=Path,default=ROOT.parent/'rebuild_input')
    parser.add_argument('--output',type=Path,help='独立结果目录；相同命令可续跑')
    parser.add_argument('--window',type=float,default=.4,help='RGB-D 选帧间隔秒数（0.1–2）')
    parser.add_argument('--no-completion',action='store_true',help='仅保留测量表面，不估计补面')
    parser.add_argument('--open',action='store_true',help='完成后打开本地交互预览')
    parser.add_argument('--check',action='store_true',help='仅检查输入与依赖')
    parser.add_argument('--verbose',action='store_true',help='显示完整处理日志')
    parser.add_argument('--salon',action='store_true',default=True,help='Build the original and bald head for the 19-style salon; always enabled in this release')
    parser.add_argument('--reuse-work',type=Path,help='可选：经输入时间戳核对后复用相同录像的几何缓存')
    parser.add_argument('--hair-style',choices=['original','bald','straight','bob','lob','fringe','part','wavy','waves','curly','ponytail'],default='original',help='Gemini 预览的初始发型')
    parser.add_argument('--hair-color',choices=['original','black','brown','blond','silver','purple'],default='original',help='Gemini 预览的初始发色')
    a=parser.parse_args(argv)
    if not .1<=a.window<=2:parser.error('--window 需要在 0.1–2 秒之间')
    primary,top=(a.primary,a.top) if a.primary else default_inputs(a.input_folder)
    if a.top:top=a.top
    primary=resolve_recording(primary);top=resolve_recording(top) if top else None
    mode='rgbd' if primary.suffix.lower()=='.bag' else 'rgb'
    if top and ((mode=='rgbd' and top.suffix.lower()!='.bag') or (mode=='rgb' and top.suffix.lower() not in RGB_SUFFIXES)):raise ValueError('主录像和顶部补拍须使用同类输入：两段 Gemini BAG，或两段同一相机的普通视频。')
    if mode=='rgb' and a.no_completion:raise ValueError('--no-completion 仅适用于有实测深度的 Gemini BAG。')
    if mode=='rgb' and a.reuse_work:raise ValueError('--reuse-work 仅适用于 Gemini 的几何缓存。')
    dependencies=['cv2','numpy','scipy','trimesh']+(['open3d','mapbox_earcut','shapely'] if mode=='rgbd' else ['torch','onnxruntime']+(['open3d','mapbox_earcut','shapely'] if a.salon else []))
    missing=[name for name in dependencies if importlib.util.find_spec(name) is None]
    if missing:raise RuntimeError('缺少依赖：'+','.join(missing)+'。请安装 requirements-reconstruction.txt。')
    if mode=='rgbd' and not (Path(os.environ.get('ORBBEC_SDK_DIR',ROOT.parent/'rgbd_capture/tools/gemini_sdk'))/'OrbbecSDK.dll').exists():raise RuntimeError('缺少 Orbbec SDK，请设置 ORBBEC_SDK_DIR。')
    identity={'version':VERSION,'mode':mode,'primary':signature(primary),'top':signature(top) if top else None,'window':a.window,'completion':not a.no_completion}
    if a.salon:identity['style_pipeline']='salon_27'
    digest=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:12]
    output=(a.output or ROOT/'runs'/f'{mode}_{digest}').resolve()
    if a.check:print(json.dumps({'input_check':'passed','mode':mode,'primary':str(primary),'top':str(top) if top else None,'output':str(output)},ensure_ascii=False,indent=2));return
    job=Job(output,identity);job.acquire();job.persist()
    if a.verbose:job.env['HEAD_VERBOSE']='1'
    try:
        preview,page,model=(rgbd if mode=='rgbd' else rgb)(job,a,primary,top)
        job.status.update(state='complete',model=str(model),finished_utc=datetime.now(timezone.utc).isoformat());job.persist()
        print('\n完成。模型：'+str(model)+'\n结果目录：'+str(output),flush=True)
        if a.open:
            if a.salon:
                from salon.prepare import prepare
                from salon.server import serve
                prepare(preview/'assets/rgbd_b' if mode=='rgbd' else preview/'assets',recordings={'primary':str(primary),'supplement_top':str(top) if top else None},prebuild=False)
                serve(open_browser=True)
            else:open_preview(job,preview,page)
        return {'output':str(output),'preview':str(preview),'assets':str(preview/'assets/rgbd_b' if mode=='rgbd' else preview/'assets'),'model':str(model),'mode':mode}
    except BaseException as exc:
        job.status.update(state='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',current_stage=job.stage,error=str(exc));job.persist();raise
    finally:
        if job.fd:job.fd.close()

if __name__=='__main__':
    if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8',errors='replace')
    if hasattr(sys.stderr,'reconfigure'):sys.stderr.reconfigure(encoding='utf-8',errors='replace')
    local_python=ROOT/'.venv/Scripts/python.exe'
    if local_python.exists() and Path(sys.executable).resolve()!=local_python.resolve():
        raise SystemExit(subprocess.call([str(local_python),str(Path(__file__).resolve()),*sys.argv[1:]],env={**os.environ,'PYTHONUTF8':'1'}))
    try:main()
    except (Exception,KeyboardInterrupt) as exc:
        print('\n未完成：'+str(exc),file=sys.stderr,flush=True);raise SystemExit(1)
