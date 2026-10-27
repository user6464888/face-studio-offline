"""Combine selected original RGB frames from two videos, without video transcoding."""
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

from salon.geometry import RECON


def merge_prepared(main_folder,top_folder,target):
    main=json.loads((main_folder/'manifest.json').read_text(encoding='utf-8'))
    top=json.loads((top_folder/'manifest.json').read_text(encoding='utf-8'))
    sizes={(row['width'],row['height']) for data in (main,top) for row in data['images']}
    if len(sizes)!=1:raise ValueError('两段普通视频分辨率不同，请使用同一摄像头和相同录制设置。')
    target.mkdir(exist_ok=True,parents=True);(target/'images').mkdir(exist_ok=True)
    result=dict(main);result['images']=[]
    for role,folder,data in [('primary',main_folder,main),('supplement_top',top_folder,top)]:
        for row in data['images']:
            item=dict(row);name=role+'_'+Path(row['file']).name
            shutil.copy2(folder/row['file'],target/'images'/name)
            item.update(file='images/'+name,source_role=role,source_video=data['source_video'])
            if role=='supplement_top':item['elapsed_seconds']+=main['source_duration_seconds']
            result['images'].append(item)
    result.update(selected_count=len(result['images']),source_frames_decoded=main['source_frames_decoded']+top['source_frames_decoded'],
                  source_duration_seconds=main['source_duration_seconds']+top['source_duration_seconds'],
                  source_videos=[main['source_video'],top['source_video']],
                  selection_note='Separately select original RGB frames from frontal-start and top videos; preserve role, no re-encoding. Same camera/resolution required.')
    (target/'manifest.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    return result


def import_pair(job,primary,top):
    from run_head import import_rgb
    from prepare_video import prepare
    session=import_rgb(job,primary)
    second=import_rgb(SimpleNamespace(path=job.path.with_name(job.path.name+'_top')),top)
    target=RECON/'inputs'/session
    if (target/'manifest.json').exists():return session
    main_frames=job.path/'rgb_primary_frames';top_frames=job.path/'rgb_top_frames'
    captures=RECON.parent/'head_capture/captures'
    if not main_frames.exists():prepare(captures/session,main_frames)
    if not top_frames.exists():prepare(captures/second,top_frames)
    merge_prepared(main_frames,top_frames,target)
    return session
