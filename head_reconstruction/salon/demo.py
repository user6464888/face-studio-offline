"""Build demonstration assets solely from analytic head geometry."""
import json
from salon.catalog import catalog
from salon.geometry import ROOT
from salon.heads import standard_heads

DEMO_REVISION=1


def prepare():
    marker=ROOT/'demo_manifest.json'
    if marker.exists():
        old=json.loads(marker.read_text(encoding='utf-8'))
        if old.get('revision')==DEMO_REVISION:
            heads=old['heads']
            if all((ROOT/h[k]).is_file() for h in heads for k in ['bald_file','radial_file','masks_file']):
                save_state(heads)
                return heads
    heads=standard_heads()
    marker.write_text(json.dumps({'revision':DEMO_REVISION,'heads':heads,'input_source':'analytic_functions_only'},ensure_ascii=False,indent=2),encoding='utf-8')
    save_state(heads)
    return heads


def save_state(heads):
    (ROOT/'heads.json').write_text(json.dumps({'version':1,'heads':heads,'standard_shapes_are_simulated':True},ensure_ascii=False,indent=2),encoding='utf-8')
    (ROOT/'catalog.json').write_text(json.dumps(catalog(),ensure_ascii=False,indent=2),encoding='utf-8')
    state={'version':1,'state':'demo','notice':'示意头演示：程序生成的头部模型。脸型由你选择，推荐用于比较款式。',
           'steps':[],'quality_warnings':[],'style_count':19,'standard_head_count':7}
    (ROOT/'workflow.json').write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding='utf-8')
