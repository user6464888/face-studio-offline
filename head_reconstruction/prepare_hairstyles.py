"""Fit simulated three-dimensional hair to the fused RGB-D head.

Keep measured face/ear/neck triangles, remove original hair triangles, fit an
estimated scalp and imported strand grooms, and transfer captured RGB variation.
The new hair geometry and the hidden scalp are simulations, not measurements.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import trimesh
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from rgbd_context import ASSETS
from hair_library import LIBRARY,build_library,ScanAtlas
from groom_variants import fitted_strands
from bald_head import build_bald_head
from skin_texture import build_skin_texture
from bald_regions import bald_regions

VERSION=46
STRAND_FIT_VERSION=5
STYLES={'original':'原始采集','bald':'光头',**LIBRARY}
BALD_DESCRIPTION='移除全部发丝，以本人额头和脸颊的采集纹理补全头皮；头皮形状及纹理分布为估计。'
COLORS={'original':None,'black':'#15171c','brown':'#51352a','blond':'#ba8752','silver':'#a8adb8','purple':'#715780'}

def identify_hair(mesh,c):
    v=np.asarray(mesh.vertices);rgb=np.asarray(mesh.visual.vertex_colors)[:,:3].astype(float)
    brightness=rgb.mean(1);red_blue=rgb[:,0]-rgb[:,2]
    theta=np.arctan2((v[:,0]-c['centerX'])/c['radiusX'],(v[:,2]-c['centerZ'])/c['radiusZ'])
    cos=np.cos(theta)
    line=c['sideLine']+(c['frontLine']-c['sideLine'])*np.maximum(cos,0)**1.4+(c['backLine']-c['sideLine'])*np.maximum(-cos,0)**.6
    # Eyes, eyebrows, beard and ear recesses cannot seed hair detection.
    face=(np.abs(v[:,0]-c['centerX'])<c['radiusX']*.82)&(v[:,2]>c['centerZ']+.025)&(v[:,1]<c['frontLine']-.044)
    ears=(np.abs(v[:,0]-c['centerX'])>c['radiusX']*.76)&(((v[:,2]-c['centerZ']-.012)/.021)**2+((v[:,1]-c['sideLine']+.032)/.031)**2<1)
    protected=face|ears
    scalp=v[:,1]>c['frontLine']+.025
    hair_tone=np.median(red_blue[scalp])
    tone_limit=float(np.clip(hair_tone+17,9,70))
    value_limit=float(np.clip(np.quantile(brightness[scalp],.995)+30,95,240))
    forehead=(np.abs(v[:,0]-c['centerX'])<c['radiusX']*.65)&(v[:,2]>c['centerZ']+.04)&(v[:,1]>c['frontLine']-.042)&(v[:,1]<c['frontLine']-.018)
    skin_value=float(np.median(brightness[forehead])) if forehead.any() else 90.
    # Warm, dim fringe pixels are mixed hair/skin, not exposed forehead. Grow
    # them only from the connected crown, while the brows remain protected.
    warm_fringe=(red_blue<tone_limit+18)&(brightness<min(value_limit,skin_value*.73))
    candidate=(v[:,1]>line-.045)&(((red_blue<tone_limit)&(brightness<value_limit))|warm_fringe)&~protected
    # Connected RGB/geometry regions replace the old hard horizontal hairline.
    edges=mesh.edges_unique;accepted=candidate[edges[:,0]]&candidate[edges[:,1]]
    edge=edges[accepted]
    adjacency=coo_matrix((np.ones(len(edge)),(edge[:,0],edge[:,1])),shape=(len(v),len(v)))
    _,labels=connected_components(adjacency,directed=False)
    seed=candidate&scalp
    region=candidate&np.isin(labels,np.unique(labels[seed]))
    # Connect small scan fragments separated by TSDF seams, without crossing skin.
    if region.sum()<100:raise RuntimeError('可靠头发区域太少，无法自动修改发型。')
    for _ in range(2):
        remaining=np.flatnonzero(candidate&~region)
        if not len(remaining):break
        distance,_=cKDTree(v[region]).query(v[remaining],workers=4)
        region[remaining[distance<.0045]]=True
    # Include the mixed-colour transition immediately next to detected hair.
    edge=edges[region[edges[:,0]]!=region[edges[:,1]]]
    ring=np.unique(edge.ravel())
    transition=ring[(red_blue[ring]<tone_limit+6)&(brightness[ring]<value_limit+12)&~protected[ring]&~region[ring]]
    weights=region.astype(float);weights[transition]=.65
    return weights,protected,theta,{'tone_limit':tone_limit,'brightness_limit':value_limit,'connected_hair_vertices':int(region.sum()),'transition_vertices':len(transition)}

def prepare_bald(assets,result):
    mesh=trimesh.load(assets/'head.ply',process=False)
    scalp=trimesh.load(assets/'head_scalp.ply',process=False)
    weights=np.fromfile(assets/'hair_weights.bin',dtype=np.uint8)
    protected,ear,sideburn=bald_regions(mesh,result['scalp'])
    bald,stats=build_bald_head(mesh,scalp,weights,protected,result['scalp'],sideburn=sideburn,ear=ear)
    baseline=trimesh.Trimesh(bald.metadata['shape_reference_vertices'],bald.faces.copy(),process=False)
    baseline.export(assets/'head_bald_before_shape.ply')
    (assets/'bald_shape_fit.json').write_text(json.dumps(bald.metadata['shape_fit'],ensure_ascii=False,indent=2),encoding='utf-8')
    bald.export(assets/'head_bald.ply')
    shape_weights=bald.metadata['shape_weights']
    (assets/'bald_scalp_mask.bin').write_bytes(shape_weights.tobytes())
    (assets/'bald_ear_mask.bin').write_bytes(ear.astype(np.uint8).tobytes())
    (assets/'bald_sideburn_mask.bin').write_bytes(sideburn.astype(np.uint8).tobytes())
    (assets/'bald_ear_rim_mask.bin').write_bytes(bald.metadata['ear_rim_mask'].tobytes())
    result['bald_stats']=stats
    result['bald_texture_stats']=build_skin_texture(assets,mesh,bald,shape_weights,protected,result['scalp'],ear_guard=ear)
    result['bald_texture']={'geometry':'head_bald_uv.ply','image':'head_bald_texture.png','face_mask':'head_bald_texture_faces.bin','provenance':'skin_texture_sources.json'}


def prepare_bald_grooms(assets,result):
    """Every preset adds strands to exactly the bald geometry shown in the UI."""
    bald=trimesh.load(assets/'head_bald.ply',process=False)
    digest=hashlib.sha256(np.asarray(bald.vertices,dtype='<f4').tobytes()+np.asarray(bald.faces,dtype='<i4').tobytes()).hexdigest()
    expected=[assets/f'{prefix}_{key}.{suffix}' for key in LIBRARY for prefix,suffix in (('hair','ply'),('hair_volume','bin'))]
    if (result.get('preset_base')=='bald' and result.get('strand_base_geometry_sha256')==digest
            and result.get('strand_fit_version')==STRAND_FIT_VERSION and all(p.exists() for p in expected)):
        return
    source=trimesh.load(assets/'head.ply',process=False)
    weights=np.fromfile(assets/'hair_weights.bin',dtype=np.uint8)
    scalp_weights=np.fromfile(assets/'bald_scalp_mask.bin',dtype=np.uint8)
    fit=json.loads((assets/'bald_shape_fit.json').read_text(encoding='utf-8'))
    frame=dict(result['scalp']);r=fit['fit_radii_m']
    frame.update(zip(('centerX','centerY','centerZ'),fit['fit_centre_m']))
    frame.update(radiusX=r['side'],radiusY=r['crown'],radiusZ=(r['front']+r['back'])/2)
    atlas=ScanAtlas(source,frame,weights,result['measured_source_vertices'])
    atlas.attach_scalp(bald,scalp_weights)
    statistics={}
    for key,label in LIBRARY.items():
        print('贴合光头底模：'+label,flush=True)
        hair,stat=fitted_strands(atlas,key)
        hair.export(assets/f'hair_{key}.ply')
        (assets/f'hair_volume_{key}.bin').write_bytes(hair.metadata['volume_weights'].tobytes())
        stat.update(base_model='head_bald.ply',base_geometry_sha256=digest,root_attachment='closest bald triangles',root_embed_depth_mm=.25)
        statistics[key]=stat
    result['preset_stats']['presets']=statistics
    result['preset_stats']['generation']='authored strands retargeted to the shared corrected bald head; no alternate scalp mesh'
    result.setdefault('style_descriptions',{}).update({key:stat['variant']+' · 光头底模上的发丝' for key,stat in statistics.items()})
    result.update(preset_base='bald',strand_base_geometry_sha256=digest,strand_fit_version=STRAND_FIT_VERSION,strand_fit_frame=frame)


def prepare(assets=ASSETS,style='original',color='original',legacy=True):
    assets=Path(assets);source=assets/'head.ply';digest=hashlib.sha256(source.read_bytes()).hexdigest();target=assets/'styling.json'
    if target.exists():
        previous=json.loads(target.read_text(encoding='utf-8'))
        expected=[assets/'head_base.ply',assets/'head_scalp.ply',assets/'hair_weights.bin',assets/'scalp_hair_rgb.bin']
        if legacy:expected += [*[assets/f'scalp_weights_{k}.bin' for k in LIBRARY],*[assets/f'hair_volume_{k}.bin' for k in LIBRARY],*[assets/f'hair_{k}.ply' for k in LIBRARY]]
        # Reuse the depth reconstruction; bald geometry changes invalidate all
        # fitted roots through their geometry hash, including future updates.
        if previous.get('version') in (28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,VERSION) and previous.get('source_head_sha256')==digest and all(p.exists() for p in expected):
            if previous.get('version')!=VERSION or not all((assets/n).exists() for n in ('head_bald.ply','head_bald_before_shape.ply','bald_shape_fit.json','bald_scalp_mask.bin','bald_ear_mask.bin','bald_sideburn_mask.bin','bald_ear_rim_mask.bin','head_bald_uv.ply','head_bald_texture.png','head_bald_texture_faces.bin','head_bald_uv_mapping.npz','skin_texture_sources.json')):prepare_bald(assets,previous)
            previous.update(version=VERSION,styles=STYLES,default_style=style,default_color=color,bald_uses_estimated_scalp=True)
            previous.setdefault('style_descriptions',{})['bald']=BALD_DESCRIPTION
            if legacy:prepare_bald_grooms(assets,previous)
            else:prepare_salon_frame(assets,previous)
            previous.pop('scalp_is_estimated',None)
            previous.update(fitting_frame_is_estimated=True,uses_estimated_scalp_mesh=True)
            for stat in previous.get('preset_stats',{}).values():
                if isinstance(stat,dict) and 'remaining_reversed_triangles' in stat:
                    stat['remaining_reversed_valid_triangles']=stat.pop('remaining_reversed_triangles')
                    stat['degenerate_source_area_threshold_m2']=5e-13
            target.write_text(json.dumps(previous,ensure_ascii=False,indent=2),encoding='utf-8');mark_metadata(assets);return previous
    meta=json.loads((assets/'metadata.json').read_text(encoding='utf-8'))
    mesh=trimesh.load(source,process=False);v=np.asarray(mesh.vertices);rgb=np.asarray(mesh.visual.vertex_colors)[:,:3]
    upper=v[v[:,1]>np.quantile(v[:,1],.55)];bounds=np.quantile(upper,[.005,.995],axis=0)
    unit=float(np.ptp(v[:,1]))/2;top=float(np.quantile(v[:,1],.985));front_line=top-unit*.48-.01
    c={'centerX':float((bounds[0,0]+bounds[1,0])/2),'centerY':front_line-unit*.23,'centerZ':float((bounds[0,2]+bounds[1,2])/2),'radiusX':float((bounds[1,0]-bounds[0,0])*.5*.94),'radiusY':top-(front_line-unit*.23)-.002,'radiusZ':float((bounds[1,2]-bounds[0,2])*.5*.96),'frontLine':front_line,'sideLine':front_line-unit*.45,'backLine':front_line-unit*1.05}
    weights,protected,theta,detection=identify_hair(mesh,c)
    weights=(weights*255).astype(np.uint8);(assets/'hair_weights.bin').write_bytes(weights.tobytes())
    added=meta.get('crown_surface_completion',{}).get('added_vertices',0);measured_count=len(v)-added
    body,scalp,outputs,stats=build_library(mesh,c,weights,measured_count,generate_styles=False)
    body.export(assets/'head_base.ply');scalp.export(assets/'head_scalp.ply')
    (assets/'scalp_hair_weights.bin').write_bytes(scalp.metadata['hair_weights'].tobytes())
    (assets/'scalp_hair_rgb.bin').write_bytes(scalp.metadata['hair_rgb'].astype(np.uint8).tobytes())
    for kind,mask in scalp.metadata['style_hair_weights'].items():
        (assets/f'scalp_weights_{kind}.bin').write_bytes(mask.tobytes())
    for kind,result in outputs.items():
        result.export(assets/f'hair_{kind}.ply')
        (assets/f'hair_volume_{kind}.bin').write_bytes(result.metadata['volume_weights'].tobytes())
    hair_rgb=np.median(rgb[weights>200],axis=0);original_color='#'+''.join(f'{int(channel):02x}' for channel in hair_rgb)
    linear=np.where(rgb/255<=.04045,rgb/255/12.92,((rgb/255+.055)/1.055)**2.4)
    luminance=linear@np.array([.2126,.7152,.0722]);median_luma=float(np.median(luminance[weights>200]))
    result={'version':VERSION,'method':'rgbd_retargeted_strand_library','units':'metres','source_head_sha256':digest,'source_vertices':len(v),'scalp':c,'styles':STYLES,'colors':COLORS,'default_style':style,'default_color':color,'original_preset_color':original_color,'hair_luminance_median_linear':median_luma,'preset_mesh_contains_face':False,'preset_stats':stats,'hair_colour_vertices':int((weights>0).sum()),'protected_face_vertices':int(protected.sum()),'protected_face_geometry_and_colour_unchanged':bool(np.array_equal(body.vertices,v) and np.array_equal(body.visual.vertex_colors,mesh.visual.vertex_colors)),'fitting_frame_is_estimated':True,'uses_estimated_scalp_mesh':True,'preset_hair_is_measured':False,'hair_surface_derived_from_depth':True,'measured_source_vertices':measured_count,'estimated_crown_vertices':added,'measured_hair_vertices':int(((np.arange(len(v))<measured_count)&(weights>0)).sum()),'detection':detection,'asset_attribution':{'author':'Cem Yuksel','url':'https://www.cemyuksel.com/research/hairmodels/','usage':'personal or research'},'style_descriptions':{k:v['variant'] for k,v in stats['presets'].items()},'continuous_scalp_boundary':True,'note':'3份基础发丝素材及剪裁、延长和梳理变体，按RGB-D适配。头皮与脸部共享原扫描边界，发根嵌入、渐细并增加细碎发过渡。新增发型为模拟，不是本人实测发丝。'}
    result['scalp_colour_mode']='boundary_poisson_untinted'
    result['scalp_colour_changes_with_dye']=False
    result['bald_uses_estimated_scalp']=True
    result['style_descriptions']['bald']=BALD_DESCRIPTION
    prepare_bald(assets,result)
    if legacy:prepare_bald_grooms(assets,result)
    else:prepare_salon_frame(assets,result)
    target.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');mark_metadata(assets)
    print(json.dumps({'styling_ready':True,'method':result['method'],'measured_hair_vertices':result['measured_hair_vertices'],'face_preserved':True},ensure_ascii=False),flush=True)
    return result

def mark_metadata(assets):
    p=assets/'metadata.json';meta=json.loads(p.read_text(encoding='utf-8'))
    meta.update(hair_changes_included=True,hairstyle_assets='styling.json',preset_hair_is_measured=False,hair_surface_derived_from_depth=True)
    p.write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')


def prepare_salon_frame(assets,result):
    fit=json.loads((assets/'bald_shape_fit.json').read_text(encoding='utf-8'))
    frame=dict(result['scalp']);r=fit['fit_radii_m']
    frame.update(zip(('centerX','centerY','centerZ'),fit['fit_centre_m']))
    frame.update(radiusX=r['side'],radiusY=r['crown'],radiusZ=(r['front']+r['back'])/2)
    result.update(strand_fit_frame=frame,preset_base='bald',head_pipeline_mode='salon_19',
                  styles={'original':'原始采集','bald':'光头'},
                  note='生成原始脸模和光头底模；19 款新发型由 salon 目录独立构建。')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--assets',type=Path,default=ASSETS);p.add_argument('--style',choices=STYLES,default='original');p.add_argument('--color',choices=COLORS,default='original');p.add_argument('--bald-only',action='store_true');a=p.parse_args();prepare(a.assets,a.style,a.color,legacy=not a.bald_only)
