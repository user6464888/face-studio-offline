"""Distinct neutral head shapes, copied personal model, and tentative geometry ranking."""
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import ConvexHull,cKDTree

from salon.catalog import FACE_SHAPES,STYLES
from salon.geometry import ROOT,RECON,smooth,radial_field

SHAPES={
 'oval':(0,0,0,0,1.00),
 'round':(.02,.14,.055,.10,.88),
 # The user selected the former round head as the new square head.
 'square':(.02,.14,.16,.22,.88),
 'long':(-.03,-.07,-.07,-.04,1.10),
 'heart':(.14,.01,-.10,-.12,1.01),
 'vshape':(-.01,-.05,-.12,-.18,1.00),
 'diamond':(-.14,.09,-.12,-.22,1.03),
}
SHAPE_REVISION=2


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ear_mask(mesh,c,r):
    q=(mesh.vertices-c)/r;ear=np.zeros(len(q),bool)
    for sign in (-1,1):
        side=sign*q[:,0]>0
        outer=side&(abs(q[:,0])>1.0)&(q[:,1]>-.72)&(q[:,1]<-.13)&(q[:,2]>-.32)&(q[:,2]<.25)
        if outer.sum()<8:continue
        hull=ConvexHull(q[outer][:,[2,1]])
        inside=(q[:,[2,1]]@hull.equations[:,:2].T+hull.equations[:,2]<=.008).all(axis=1)
        ear|=side&inside&(abs(q[:,0])>.90)
    support=ear.copy();support[np.unique(mesh.faces[ear[mesh.faces].any(axis=1)])]=True
    return support


def morph(mesh,c,r,kind,ear):
    v=np.array(mesh.vertices);q=(v-c)/r
    forehead,cheek,jaw,chin,length=SHAPES[kind]
    frontal=smooth((q[:,2]-.06)/.48)*(1-smooth((-q[:,1]-1.46)/.26))
    weights=[np.exp(-((q[:,1]-.10)/.33)**2),np.exp(-((q[:,1]+.45)/.35)**2),
             np.exp(-((q[:,1]+.96)/.27)**2),np.exp(-((q[:,1]+1.28)/.22)**2)]
    width=1+frontal*(forehead*weights[0]+cheek*weights[1]+jaw*weights[2]+chin*weights[3])
    width[ear]=1
    v[:,0]=c[0]+(v[:,0]-c[0])*width
    vertical=smooth((.28-q[:,1])/.38)*(1-smooth((-q[:,1]-1.48)/.25))*frontal
    v[:,1]+=(q[:,1]-.18)*r[1]*(length-1)*vertical
    if kind in ('heart','vshape'):
        # Broaden the lower outline above, then soften the central chin tip.
        lower=weights[3]*frontal
        lift=.002 if kind=='heart' else .003
        v[:,1]+=lift*lower*np.clip(1-abs(q[:,0])/.45,0,1)
    result=trimesh.Trimesh(v,mesh.faces.copy(),process=False)
    result.visual.vertex_colors=np.tile([180,168,152,255],(len(v),1))
    return result


def measure(mesh,c,r,ear):
    v=np.asarray(mesh.vertices);q=(v-c)/r
    front=(q[:,2]>.18)&~ear&(abs(q[:,0])<1.05)&(q[:,1]>-1.58)&(q[:,1]<.35)
    widths=[]
    for low,high in [(.03,.30),(-.60,-.25),(-1.10,-.78),(-1.43,-1.15)]:
        region=front&(q[:,1]>low)&(q[:,1]<high)
        if region.sum()<12:raise ValueError('正面轮廓可用点不足，需手动选择脸型')
        x=v[region,0];widths.append(float(np.quantile(x,.98)-np.quantile(x,.02)))
    chin=np.quantile(v[front,1],.015)
    height=float(c[1]+.35*r[1]-chin)
    return {'forehead_width':widths[0],'cheek_width':widths[1],'jaw_width':widths[2],'chin_width':widths[3],
            'face_height':height,'ratios':[height/widths[1],widths[0]/widths[1],widths[2]/widths[1],widths[3]/widths[1]]}


def standard_heads():
    out=ROOT/'assets';out.mkdir(exist_ok=True)
    from salon.procedural_head import neutral_head
    template,c,r=neutral_head()
    ear=ear_mask(template,c,r)
    heads=[]
    for kind,face in FACE_SHAPES.items():
        mesh=morph(template,c,r,kind,ear)
        target=out/f'head_{kind}.ply';mesh.export(target)
        radius,fallback=radial_field(mesh,c,r)
        radial=out/f'radial_{kind}.npz';np.savez_compressed(radial,radius=radius)
        masks=out/f'masks_{kind}.npz';np.savez_compressed(masks,ear=ear)
        entry={'id':kind,'name':face['name']+'标准头','face_shape':kind,'is_standard':True,
               'centre':c.tolist(),'radii':r.tolist(),'bald_file':target.relative_to(ROOT).as_posix(),
               'radial_file':radial.relative_to(ROOT).as_posix(),'masks_file':masks.relative_to(ROOT).as_posix(),
               'texture_file':None,'fingerprint':digest(target),'radial_fallback_fraction':fallback,
               'shape_revision':SHAPE_REVISION,'shape_profile':dict(zip(('forehead','cheek','jaw','chin','length'),SHAPES[kind])),
               'all_style_ids':[s['id'] for s in STYLES],
               'provenance':'original analytic mannequin with regional deformation; no scan or third-party head dataset'}
        entry['measurements']=measure(mesh,c,r,ear);heads.append(entry)
        print('Standard head:',kind,flush=True)
    return heads


def refresh_standard_heads():
    """Update only standard heads; retain the current scan and its workflow."""
    data=json.loads((ROOT/'heads.json').read_text(encoding='utf-8'))
    data['heads']=standard_heads()+[h for h in data['heads'] if not h['is_standard']]
    data['shape_revision']=SHAPE_REVISION
    from salon.prepare import save
    save(ROOT/'heads.json',data)
    return data


def build_heads(source_assets):
    source_assets=Path(source_assets).resolve();out=ROOT/'assets';out.mkdir(exist_ok=True)
    heads=standard_heads()
    # Work on a local copy, preserving every original reconstruction file.
    protected={name:digest(source_assets/name) for name in ['head.ply','head_bald.ply','head_bald_uv.ply','head_bald_texture.png','styling.json']}
    person=out/'person';person.mkdir(exist_ok=True)
    for name in ['head.ply','head_bald.ply','head_bald_uv.ply','head_bald_texture.png','head_bald_texture_faces.bin','styling.json']:
        shutil.copy2(source_assets/name,person/name)
    styling=json.loads((source_assets/'styling.json').read_text(encoding='utf-8'))
    f=styling['strand_fit_frame'];c=np.array([f['centerX'],f['centerY'],f['centerZ']]);r=np.array([f['radiusX'],f['radiusY'],f['radiusZ']])
    mesh=trimesh.load(person/'head_bald_uv.ply',process=False)
    original=trimesh.load(person/'head_bald.ply',process=False)
    mapping=cKDTree(original.vertices).query(mesh.vertices)[1]
    ear0=np.fromfile(source_assets/'bald_ear_mask.bin',np.uint8).astype(bool)
    support=ear0.copy();support[np.unique(original.faces[ear0[original.faces].any(axis=1)])]=True
    ear=support[mapping]
    radius,fallback=radial_field(mesh,c,r)
    radial=out/'radial_person.npz';np.savez_compressed(radial,radius=radius)
    masks=out/'masks_person.npz';np.savez_compressed(masks,ear=ear)
    entry={'id':'person','name':'本人头模 · 贴合对照','face_shape':None,'is_standard':False,
           'centre':c.tolist(),'radii':r.tolist(),'bald_file':'assets/person/head_bald_uv.ply',
           'original_file':'assets/person/head.ply','texture_file':'assets/person/head_bald_texture.png',
           'texture_faces':'assets/person/head_bald_texture_faces.bin',
           'radial_file':radial.relative_to(ROOT).as_posix(),'masks_file':masks.relative_to(ROOT).as_posix(),
           'fingerprint':digest(person/'head_bald_uv.ply'),'radial_fallback_fraction':fallback,
           'all_style_ids':[s['id'] for s in STYLES],'provenance':'user-provided local reconstruction; not distributed'}
    try:
        entry['measurements']=measure(mesh,c,r,ear)
        a=np.array(entry['measurements']['ratios']);prototypes=np.array([x['measurements']['ratios'] for x in heads])
        spread=np.maximum(np.std(prototypes,axis=0),.035)
        distance=np.linalg.norm((prototypes-a)/spread,axis=1)
        order=np.argsort(distance)
        entry['analysis']={'suggested_shape':heads[order[0]]['id'],
                           'alternatives':[{'shape':heads[i]['id'],'distance':float(distance[i])} for i in order[:3]],
                           'method':'mesh frontal proportions compared with neutral shape examples',
                           'status':'tentative_manual_confirmation_required',
                           'note':'几何初判，受扫描姿态、遮挡与脸型示意模型影响，可手动更正；不把头顶估计曲面当作精确头骨测量。'}
    except ValueError as exc:entry['analysis']={'suggested_shape':None,'status':'manual_selection_required','note':str(exc)}
    heads.append(entry)
    assert all(digest(source_assets/name)==sha for name,sha in protected.items())
    (ROOT/'heads.json').write_text(json.dumps({'version':1,'shape_revision':SHAPE_REVISION,'heads':heads,'source_assets':str(source_assets),
         'protected_sha256':protected,'standard_shapes_are_simulated':True},ensure_ascii=False,indent=2),encoding='utf-8')
    return heads


if __name__=='__main__':
    refresh_standard_heads()
