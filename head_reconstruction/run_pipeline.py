"""Rebuild the latest captured session using the already downloaded local tools."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

root=Path(__file__).resolve().parent


def run(script,*args):
    subprocess.run([sys.executable,str(root/script),*map(str,args)],cwd=root.parent,check=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--session');a=p.parse_args()
    captures=root.parent/'head_capture/captures'
    if a.session:
        source=captures/a.session
    else:
        candidates=[]
        for path in captures.iterdir():
            if (path/'session.json').exists():
                info=json.loads((path/'session.json').read_text(encoding='utf-8'))
                if info.get('videos'):candidates.append(path)
        if not candidates:raise RuntimeError('No captured video session was found.')
        source=max(candidates,key=lambda path:path.name)
    target=root/'inputs'/source.name
    if not (target/'manifest.json').exists():
        from prepare_video import prepare
        prepare(source,target)
    run('segment_head.py',target)
    run('reconstruct_colmap.py',target,'--learned','--dense')
    run('refine_volume.py','--session',source.name)
    run('finalize_mesh.py','--session',source.name)
    if (root/'tools/3ddfa_v2/weights/mb1_120x120.pth').exists() and (root/'models/yunet_2023mar.onnx').exists():
        run('fit_face_model.py','--session',source.name)
        run('build_face_upgrade.py','--session',source.name)
    print('Saved result: '+str(root/'viewer'),flush=True)


if __name__=='__main__':main()
