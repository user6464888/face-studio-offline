"""Package only the explicit public source tree; never walk arbitrary output data."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import zipfile

ROOT=Path(__file__).resolve().parents[1]
TOP={'main.py','README.md','LICENSE','THIRD_PARTY_NOTICES.md','.gitignore','requirements-demo.txt',
     'requirements-demo-win-py311.lock.txt','requirements-reconstruction.txt'}
VENDOR={'three.module.js','three.core.js','OrbitControls.js','PLYLoader.js','THREE-LICENSE.txt'}
FORBIDDEN={'.bag','.mp4','.avi','.mov','.mkv','.webm','.ply','.obj','.glb','.gltf','.hair','.data','.pth','.pkl','.onnx','.dll','.exe','.npz','.bin'}


def selected_files(root=ROOT):
    files=[root/name for name in TOP if (root/name).is_file()]
    for folder in ['head_reconstruction','head_capture','rgbd_capture']:
        files.extend((root/folder).glob('*.py'))
    files.extend(p for p in (root/'head_reconstruction/salon').iterdir() if p.is_file() and p.suffix in {'.py','.js','.css','.html'})
    files.extend(root/'head_reconstruction/viewer/vendor'/name for name in VENDOR)
    for folder in ['scripts','tests','docs','.github']:
        files.extend(p for p in (root/folder).rglob('*') if p.is_file() and p.suffix in {'.py','.md','.yml','.yaml','.png','.svg'})
    if (root/'.vscode/launch.json').is_file():files.append(root/'.vscode/launch.json')
    return sorted(set(files))


def audit(files,root=ROOT):
    errors=[]
    for path in files:
        relative=path.relative_to(root).as_posix()
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):errors.append(relative+': symlink/outside root')
        if path.suffix.lower() in FORBIDDEN or path.stat().st_size>10*1024**2:errors.append(relative+': forbidden resource or oversized source')
        if '__pycache__' in path.parts:errors.append(relative+': Python cache')
        if path.suffix in {'.py','.md','.json','.html','.css','.yml','.yaml','.txt'}:
            content=path.read_text(encoding='utf-8')
            # Constructed patterns avoid matching this checker's own source text.
            patterns=[r'[A-Za-z]:[\\/]Users[\\/][^\\/\s]+',r'[A-Za-z]:[\\/]AAA'+r'111[\\/]3d',
                      r'gh' + r'p_[A-Za-z0-9]{30,}',r'github' + r'_pat_[A-Za-z0-9_]{30,}',
                      r'-----BEGIN '+r'(?:RSA |OPENSSH |EC )?PRIVATE KEY-----']
            if any(re.search(pattern,content) for pattern in patterns):errors.append(relative+': local identity/path or credential pattern')
    if errors:raise ValueError('Publication audit failed:\n'+'\n'.join(errors))


def build(root=ROOT,output=None):
    output=Path(output) if output else root/'dist';output.mkdir(parents=True,exist_ok=True)
    files=selected_files(root);audit(files,root)
    rows=[{'file':p.relative_to(root).as_posix(),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in files]
    archive=output/'face-studio-v0.1.0-source.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for row in rows:
            info=zipfile.ZipInfo('face-studio/'+row['file'],date_time=(2026,10,27,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED
            z.writestr(info,(root/row['file']).read_bytes())
    result={'release':'v0.1.0','source_file_count':len(rows),'source_bytes':sum(row['bytes'] for row in rows),
            'archive_bytes':archive.stat().st_size,'archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),
            'contains_recordings_models_weights_or_runtime_state':False,'files':rows}
    (output/'source-manifest.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='files'},indent=2))
    return archive


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path);args=parser.parse_args();build(output=args.output)
