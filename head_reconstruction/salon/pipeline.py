"""Six-step video -> original -> bald -> shape -> recommendations -> try-on."""
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime,timezone

from salon.geometry import ROOT,RECON
from salon.prepare import prepare,save,STEPS


def same_recording(old,current):
    return all(old.get(k)==current.get(k) for k in ('size','mtime_ns','edge_sha256'))


def cached_assets(primary,top):
    from run_head import signature
    primary_id=signature(primary);top_id=signature(top) if top else None
    runs=sorted((RECON/'runs').glob('*'),key=lambda p:p.name,reverse=True)
    for run in runs:
        status=run/'status.json';config=run/'config.json'
        if not status.exists() or not config.exists():continue
        data=json.loads(status.read_text(encoding='utf-8'));identity=data.get('identity',{})
        if data.get('state')!='complete' or not same_recording(identity.get('primary',{}),primary_id):continue
        if top_id and not same_recording(identity.get('top') or {},top_id):continue
        assets=Path(json.loads(config.read_text(encoding='utf-8'))['assets'])
        if all((assets/name).exists() for name in ['head.ply','head_bald.ply','head_bald_uv.ply','head_bald_texture.png','head_bald_texture_faces.bin','bald_ear_mask.bin','styling.json']):return assets
    return None


def run(primary,top,force=False,prebuild=False):
    try:return _run(primary,top,force,prebuild)
    except Exception as exc:
        if (ROOT/'workflow.json').exists():
            old=json.loads((ROOT/'workflow.json').read_text(encoding='utf-8'))
            if old.get('state')=='running':
                old.update(state='failed',error=str(exc));save(ROOT/'workflow.json',old)
        raise


def _run(primary,top,force=False,prebuild=False):
    from run_head import resolve_recording,RGB_SUFFIXES
    primary=resolve_recording(primary)
    if not top:raise ValueError('新流程需要主视频和顶部补拍，请先选择两段素材。')
    top=resolve_recording(top)
    if (primary.suffix.lower()=='.bag') != (top.suffix.lower()=='.bag'):raise ValueError('请使用两段 Gemini BAG，或同一普通摄像头的两段视频。')
    state={'state':'running','recordings':{'primary':str(primary),'supplement_top':str(top)},'current_step':2,
           'notice':'正在构建新素材；完成前保留上次模型作为查看参考。',
           'steps':[{'number':i+1,'name':s,'state':'complete' if i==0 else 'running' if i==1 else 'pending'} for i,s in enumerate(STEPS)]}
    save(ROOT/'workflow.json',state)
    assets=None if force else cached_assets(primary,top)
    used_cache=assets is not None
    if assets:
        from salon.catalog import STYLES
        print(f'复用相同录像的原始模型与光头底模；更新脸型判断和 {len(STYLES)} 款目录。',flush=True)
    else:
        stamp=datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
        output=RECON/'runs'/('salon_'+stamp)
        state['run_status']=str(output/'status.json');save(ROOT/'workflow.json',state)
        command=[sys.executable,str(RECON/'run_head.py'),'--primary',str(primary),'--top',str(top),'--salon','--output',str(output)]
        logs=ROOT/'logs';logs.mkdir(exist_ok=True)
        log=logs/f'reconstruction_{stamp}.log'
        with log.open('w',encoding='utf-8') as f:
            try:
                subprocess.run(command,cwd=RECON,env={**os.environ,'PYTHONUTF8':'1'},stdout=f,stderr=subprocess.STDOUT,check=True,
                               creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
            except subprocess.CalledProcessError as exc:raise RuntimeError('构建未完成，请查看本次日志：'+str(log)) from exc
        assets=output/'preview/assets/rgbd_b' if primary.suffix.lower()=='.bag' else output/'preview/assets'
    state['current_step']=4
    for s in state['steps']:s['state']='complete' if s['number']<4 else 'running' if s['number']==4 else 'pending'
    save(ROOT/'workflow.json',state)
    result=prepare(assets,recordings={'primary':str(primary),'supplement_top':str(top)},prebuild=prebuild)
    result['using_existing_recordings']=used_cache
    result['notice']='当前展示本次选定素材的结果。更换素材后，在第 1 步重新构建。'
    save(ROOT/'workflow.json',result)
    return result
