"""Prepare the seven reference heads, personal model, catalog and workflow."""
import argparse
import json
from pathlib import Path
from datetime import datetime,timezone

from salon.catalog import catalog,STYLES
from salon.geometry import ROOT,RECON,style_assets
from salon.heads import build_heads

DEFAULT_ASSETS=None
STEPS=['录制正脸视频＋顶部补拍','构建原始脸模','设计为光头','头型／脸型判断','按脸型推荐发型','更换展示']


def save(path,data):
    path=Path(path);temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8');temporary.replace(path)


def prepare(assets=DEFAULT_ASSETS,recordings=None,prebuild=True):
    if assets is None:raise ValueError('Pass the assets directory from your own completed reconstruction.')
    assets=Path(assets).resolve()
    (ROOT/'assets').mkdir(exist_ok=True);(ROOT/'cache').mkdir(exist_ok=True)
    save(ROOT/'catalog.json',catalog())
    heads=build_heads(assets)
    if recordings is None:
        config=assets.parents[2]/'config.json'
        recordings=json.loads(config.read_text(encoding='utf-8')).get('recordings',{}) if config.exists() else {}
    person=next(h for h in heads if h['id']=='person')
    metadata=json.loads((assets/'metadata.json').read_text(encoding='utf-8'))
    reconstruction={'primary_views':metadata.get('primary_views'),'top_views':metadata.get('top_views')}
    warnings=[]
    if metadata.get('top_views')==0 and recordings.get('supplement_top'):
        warnings.append('本次顶部补拍未能可靠融合，原始模型仍有头顶缺口。')
    validation=assets.parents[2]/'work/top_improvement_validation.json'
    if validation.exists():
        check=json.loads(validation.read_text(encoding='utf-8'))
        reconstruction['remaining_crown_boundary_edges']=check.get('remaining_crown_boundary_edges')
        if check.get('remaining_crown_boundary_edges',0)>0 and not warnings:warnings.append('原始模型顶部仍有缺口，请切到头顶查看。')
    workflow={'version':1,'state':'complete','using_existing_recordings':True,'recordings':recordings,
              'source_assets':str(assets),'updated_utc':datetime.now(timezone.utc).isoformat(),
              'steps':[{'number':i+1,'name':name,'state':'needs_confirmation' if i==3 else 'complete'} for i,name in enumerate(STEPS)],
              'analysis':person['analysis'],'active_face':person['analysis'].get('suggested_shape') or 'oval',
              'reconstruction':reconstruction,'quality_warnings':warnings,
              'notice':'当前显示已采集素材的结果；第 1 步可录制或指定新视频，再运行六步流程。',
              'style_count':len(STYLES),'standard_head_count':7}
    save(ROOT/'workflow.json',workflow)
    if prebuild:
        head=next(h for h in heads if h['id']=='oval')
        for style in STYLES:
            print('Build catalog style:',style['number'],style['family'],flush=True)
            style_assets(head,style)
    return workflow


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--assets',type=Path,default=DEFAULT_ASSETS);p.add_argument('--heads-only',action='store_true');a=p.parse_args()
    prepare(a.assets,prebuild=not a.heads_only)
