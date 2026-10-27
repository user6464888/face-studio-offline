"""Clean the measured MVS mesh, apply local photo colors, and prepare viewer assets."""
from pathlib import Path
import argparse
import json
import numpy as np
import cv2
import trimesh
from scipy.spatial import cKDTree
from colmap_data import read_text_model, intrinsics, canonical_frame

ROOT = Path(__file__).resolve().parent
SESSION = None


def photo_colors(vertices, normals, cameras, images, input_dir, basis, origin):
    n = len(vertices)
    sums = np.zeros((n, 3), np.float64)
    weights = np.zeros(n, np.float64)
    canonical=(vertices-origin)@basis.T
    canonical_normals=normals@basis.T
    smooth=cKDTree(canonical).query(canonical,k=24)[1]
    canonical_normals=canonical_normals[smooth].mean(axis=1)
    canonical_normals/=np.maximum(np.linalg.norm(canonical_normals,axis=1,keepdims=True),1e-8)
    normals=canonical_normals@basis
    manifest=json.loads((input_dir/'manifest.json').read_text(encoding='utf-8'))
    sharpness={Path(row['file']).name:row['sharpness_reference'] for row in manifest['images']}
    frontal_candidates=[name for name in images if np.linalg.norm(images[name]['R']-images[min(images)]['R'])<.12]
    preferred=max(frontal_candidates,key=lambda name:sharpness.get(name,0))
    # Prefer one sharp neutral frame over blending different expressions and
    # reflections across the central face. Side/back surfaces still use all views.
    relative=canonical-np.median(canonical,axis=0)
    angle=np.arctan2(relative[:,0],relative[:,2])
    face_prior=np.clip((np.deg2rad(65)-np.abs(angle))/np.deg2rad(25),0,1)
    for name, im in sorted(images.items()):
        cam = cameras[im['camera_id']]
        k = intrinsics(cam)
        xyz = vertices @ im['R'].T + im['t']
        uv = xyz @ k.T
        uv = uv[:, :2] / uv[:, 2:3]
        u, v = np.rint(uv).astype(int).T
        visible = (xyz[:,2] > 0) & (u>=0) & (u<cam['width']) & (v>=0) & (v<cam['height'])
        index = np.flatnonzero(visible)
        picture = cv2.cvtColor(cv2.imread(str(input_dir / 'images' / name)), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(input_dir / 'masks' / (name + '.png')), 0)
        index = index[mask[v[index],u[index]] > 0]
        direction = im['center'] - vertices[index]
        direction /= np.linalg.norm(direction,axis=1,keepdims=True)
        facing = np.einsum('ij,ij->i', normals[index], direction)
        index, facing = index[facing>.15], facing[facing>.15]
        if not len(index):
            continue
        depth = np.full((cam['height'],cam['width']),np.inf,np.float32)
        np.minimum.at(depth,(v[index],u[index]),xyz[index,2].astype(np.float32))
        depth = cv2.erode(depth,np.ones((3,3),np.uint8))
        tolerance = np.median(xyz[index,2]) * .009
        ok = xyz[index,2] <= depth[v[index],u[index]] + tolerance
        index, facing = index[ok], facing[ok]
        weight = np.power(facing,12)
        if name==preferred:
            weight*=1+face_prior[index]*80
        sums[index] += picture[v[index],u[index]] * weight[:,None]
        weights[index] += weight
    found = weights > 1e-7
    colors = np.full((n, 4),255,np.uint8)
    colors[found,:3] = np.clip(sums[found] / weights[found,None],0,255).astype(np.uint8)
    if not found.any():
        raise RuntimeError('No vertices received image colors')
    if not found.all():
        nearest = cKDTree(vertices[found]).query(vertices[~found])[1]
        colors[~found,:3] = colors[found,:3][nearest]
    return colors, float(found.mean())


def main():
    p=argparse.ArgumentParser();p.add_argument('--session',required=True);a=p.parse_args()
    src = ROOT / 'inputs' / a.session
    output = ROOT / 'outputs' / a.session / 'colmap_aliked'
    dense = output / 'dense'
    import os
    assets = Path(os.environ.get('HEAD_RGB_ASSETS',str(ROOT / 'viewer/assets')))
    assets.mkdir(parents=True,exist_ok=True)
    cameras, images, sparse, _, _ = read_text_model(output / 'sparse_text')
    basis, origin = canonical_frame(images,sparse)
    points = trimesh.load(dense / 'fused.ply',process=False)
    refined = (dense / 'mesh_refined.ply').exists()
    mesh = trimesh.load(dense / ('mesh_refined.ply' if refined else 'mesh.ply'),process=False)
    if not isinstance(mesh,trimesh.Trimesh) or not len(mesh.faces):
        raise RuntimeError('No reconstructed surface')
    local_points = (points.vertices-origin) @ basis.T
    colors = np.asarray(points.colors)[:,:3]
    # A horizontal cut just below the chin excludes the shirt and unstable shoulders.
    front = images[min(images)]
    center_cam = front['R'] @ origin + front['t']
    reference_depth = np.median((points.vertices @ front['R'].T + front['t'])[:,2])
    floor = center_cam[1] - (625 - cameras[front['camera_id']]['params'][2]) / cameras[front['camera_id']]['params'][0] * reference_depth
    local_vertices = (mesh.vertices-origin) @ basis.T
    mesh.vertices = local_vertices
    bounds = np.quantile(local_points[local_points[:,1]>floor],[.005,.995],axis=0)
    # Reject far extrapolation outside the measured head volume, then retain the head.
    limits = bounds.copy();limits[0]-=.10;limits[1]+=.10
    if not refined:
        keep = np.all((mesh.triangles_center>limits[0]) & (mesh.triangles_center<limits[1]),axis=1)
        mesh.update_faces(keep);mesh.remove_unreferenced_vertices()
    pieces = mesh.split(only_watertight=False)
    if pieces:
        mesh = max(pieces,key=lambda m:len(m.faces))
    mesh = trimesh.intersections.slice_mesh_plane(mesh,[0,1,0],[0,floor,0],cap=True,engine='earcut')
    mesh.remove_unreferenced_vertices()
    trimesh.smoothing.filter_taubin(mesh,lamb=.5,nu=.53,iterations=3)
    mesh.fix_normals()
    # Texture sampling uses the unscaled world-space coordinates and normals.
    world = mesh.vertices @ basis + origin
    world_normals = mesh.vertex_normals @ basis
    rgba, colored_fraction = photo_colors(world,world_normals,cameras,images,src,basis,origin)
    top = np.quantile(mesh.vertices[:,1],.999)
    height = top-floor
    center = np.array([(bounds[0,0]+bounds[1,0])*.5,(top+floor)*.5,(bounds[0,2]+bounds[1,2])*.5])
    mesh.vertices = (mesh.vertices-center) * (2/height)
    mesh.visual.vertex_colors = rgba
    mesh.export(assets / 'head.ply')
    mesh.export(assets / 'head.glb')
    mesh.export(assets / 'head.obj')
    # For preset hairstyles, replace the measured hair region by an explicitly
    # inferred smooth scalp; the face, ears, and remaining head retain their mesh.
    scalp = {'centerY':.12,'centerZ':-.045,'radiusX':float(np.quantile(np.abs(mesh.vertices[:,0]),.92)*.94),
             'radiusY':.84,'radiusZ':float(np.quantile(np.abs(mesh.vertices[:,2]+.045),.90)*.95),
             'frontLine':.41,'backLine':-.38,'sideLine':.10}
    base = mesh.copy()
    v = base.vertices.copy()
    theta = np.arctan2(v[:,0]/scalp['radiusX'],(v[:,2]-scalp['centerZ'])/scalp['radiusZ'])
    cos=np.cos(theta)
    line = scalp['sideLine'] + (scalp['frontLine']-scalp['sideLine'])*np.maximum(0,cos)**1.4 + (scalp['backLine']-scalp['sideLine'])*np.maximum(0,-cos)**1.3
    hair_region = v[:,1] > line
    coordinates = np.column_stack([v[:,0]/scalp['radiusX'],(v[:,1]-scalp['centerY'])/scalp['radiusY'],(v[:,2]-scalp['centerZ'])/scalp['radiusZ']])
    coordinates /= np.maximum(np.linalg.norm(coordinates,axis=1,keepdims=True),1e-8)
    projected = coordinates * [scalp['radiusX'],scalp['radiusY'],scalp['radiusZ']] + [0,scalp['centerY'],scalp['centerZ']]
    blend = np.clip((v[:,1]-line)/.16,0,1)
    v += (projected-v)*blend[:,None]
    base.vertices=v
    skin_sample = rgba[(mesh.vertices[:,1]>.22)&(mesh.vertices[:,1]<.43)&(mesh.vertices[:,2]>.35)&(np.abs(mesh.vertices[:,0])<.23)&(rgba[:,:3].mean(axis=1)>100),:3]
    skin_color = np.median(skin_sample,axis=0) if len(skin_sample) else [152,119,102]
    base_colors=rgba.copy()
    paint_weight=np.clip((v[:,1]-(line-.09))/.12,0,1)
    base_colors[:,:3]=np.clip(rgba[:,:3]*(1-paint_weight[:,None])+np.asarray(skin_color)*paint_weight[:,None],0,255).astype(np.uint8)
    base.visual.vertex_colors=base_colors
    base.export(assets / 'head_base.ply')
    manifest=json.loads((src / 'manifest.json').read_text(encoding='utf-8'))
    stats=json.loads((output/'reconstruction_stats.json').read_text(encoding='utf-8')) if (output/'reconstruction_stats.json').exists() else {}
    info={
        'source_session':a.session,'source_duration_seconds':manifest['source_duration_seconds'],
        'selected_count':manifest['selected_count'],'registered_count':len(images),
        'dense_views':len((output/'dense_images.txt').read_text().splitlines()),
        'sparse_points':len(sparse),'dense_points':len(points.vertices),
        'vertices':len(mesh.vertices),'faces':len(mesh.faces),'is_watertight':bool(mesh.is_watertight),
        'photo_colored_vertex_fraction':colored_fraction,'sensor_depth':False,'scale_is_metric':False,
        'rgb_projection':{'session':a.session,'canonical_scale':float(2/height),'display_scale':1.,
                          'local_centre':center.tolist(),'world_basis':basis.tolist(),'world_origin':origin.tolist()},
        'scalp':scalp,'method':('BiRefNet + ALIKED/LightGlue COLMAP + calibrated silhouette hull with stable MVS depth fusion + multi-view vertex color' if refined else 'BiRefNet masks + ALIKED/LightGlue COLMAP SfM + PatchMatch MVS + Poisson surface + multi-view vertex color'),
        'summary':'多视角轮廓与稳定深度估计生成的实验模型；未观测的细节按轮廓补全。预设发型与头皮用于造型演示。',
        'preset_hair_is_measured':False,'mean_sparse_reprojection_error_pixels':stats.get('mean_reprojection_error_pixels'),
    }
    (assets/'metadata.json').write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf-8')
    report=f'''# 本次人头重建记录\n\n来源：{a.session} 的 RGB 视频，{info['source_duration_seconds']:.2f} 秒。\n\n- 筛选图像：{info['selected_count']} 张；相机对齐：{info['registered_count']} 张。\n- 参与表面计算的视角：{info['dense_views']}；稀疏点：{info['sparse_points']}；表面点：{info['dense_points']}。\n- 显示模型：{info['vertices']} 个顶点、{info['faces']} 个三角面；闭合：{info['is_watertight']}。\n- 直接取得照片颜色的顶点比例：{colored_fraction:.1%}；其余顶点颜色由相邻表面补全。\n\n## 方法与范围\n\nBiRefNet 背景分割、ALIKED/LightGlue 相机对齐、COLMAP 多视角深度估计。最终显示表面由标定视角的轮廓体与稳定深度估计融合，再进行照片着色。深度不稳定和未观测的区域采用轮廓体补全，表面经过平滑，不能当成精确扫描。详细参数见 outputs/{a.session}/colmap_aliked/dense/volume_refinement.json。\n\n这是 RGB 视频实验原型，没有传感器深度和尺寸标定。细小发丝、透明眼镜、曝光变化、头部轻微移动及缺少俯拍会影响形状与颜色。预设发型是程序生成的造型；切换发型时使用的头皮是近似补全，不能作为真实头皮测量。\n\n## 文件\n\n- head.ply / head.glb：原始重建造型（含顶点颜色）。\n- head.obj：几何交换文件；颜色请以 PLY/GLB 为准。\n- head_base.ply：用于发型演示的近似头皮版本。\n- ../exports：页面生成的 PNG、旋转 GIF 和含当前发型的 GLB。\n\n## 本地工具来源\n\n- COLMAP：https://github.com/colmap/colmap\n- BiRefNet：https://github.com/ZhengPeng7/BiRefNet\n- Three.js：https://github.com/mrdoob/three.js\n- gif.js：https://github.com/jnordberg/gif.js\n\n图像与模型均在本机处理，未上传到外部服务。\n'''
    (assets/'report.md').write_text(report,encoding='utf-8')
    manifest.update({'poses_estimated':True,'masks_generated':True,'reconstruction_result':'../../viewer/assets/metadata.json'})
    (src/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(info,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    main()
