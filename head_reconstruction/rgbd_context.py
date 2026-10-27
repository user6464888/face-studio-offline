"""Optional per-run paths; legacy scripts keep their existing default paths."""
import json
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parent
CONFIG_PATH=os.environ.get('HEAD_RGBD_CONFIG')
CONFIG=json.loads(Path(CONFIG_PATH).read_text(encoding='utf-8')) if CONFIG_PATH else {}
OUT=Path(CONFIG.get('work',ROOT/'outputs/rgbd_b'))
ASSETS=Path(CONFIG.get('assets',ROOT/'viewer/assets/rgbd_b'))

def input_folder(role):
    if CONFIG:return Path(CONFIG['inputs'][role])
    selection=json.loads((ROOT.parent/'rgbd_capture/reconstruction_selection.json').read_text(encoding='utf-8'))
    clip=next(c for c in selection['clips'] if c['role']==role)
    return ROOT/'inputs'/('rgbd_'+clip['session_id'])

def inputs(role):
    folder=input_folder(role)
    return folder,json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
