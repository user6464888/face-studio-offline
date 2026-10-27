"""Fail early on optional reconstruction dependencies without reading private frames."""
import importlib.util
import os
from pathlib import Path
import sys
from run_head import resolve_recording,RGB_SUFFIXES

ROOT=Path(__file__).resolve().parent


def check(primary,top):
    primary=resolve_recording(primary);top=resolve_recording(top)
    if primary==top:raise ValueError('Primary and top recordings must be different files.')
    rgbd=primary.suffix.lower()=='.bag'
    if rgbd!=(top.suffix.lower()=='.bag'):raise ValueError('Provide two BAGs or two RGB videos from the same camera.')
    modules=['cv2','PIL','skimage','mapbox_earcut','shapely','rtree']+([] if rgbd else ['torch','onnxruntime'])
    missing=[name for name in modules if importlib.util.find_spec(name) is None]
    if missing:raise RuntimeError('Missing reconstruction packages: '+', '.join(missing)+'. Install requirements-reconstruction.txt.')
    required=[ROOT/'models/yunet_2023mar.onnx']
    if rgbd:
        if sys.platform!='win32':raise RuntimeError('The included Gemini SDK bridge currently supports Windows only.')
        required.append(Path(os.environ.get('ORBBEC_SDK_DIR',ROOT.parent/'rgbd_capture/tools/gemini_sdk'))/'OrbbecSDK.dll')
    else:
        required += [ROOT/'models/birefnet-lite.onnx',ROOT/'models/aliked-n16rot.onnx',ROOT/'models/aliked-lightglue.onnx',
                     Path(os.environ.get('COLMAP_EXE',ROOT/'tools/colmap/bin/colmap.exe')),
                     Path(os.environ.get('COLMAP_DENSE_EXE',ROOT/'tools/colmap-3.11.1/bin/colmap.exe'))]
    absent=[str(path) for path in required if not path.is_file()]
    if absent:raise RuntimeError('Optional external resources are not installed:\n'+'\n'.join(absent)+'\nSee docs/RECONSTRUCTION.md; assets are not downloaded automatically.')
    print('Selected recordings and optional resource paths checked; actual reconstruction quality is not tested.')
