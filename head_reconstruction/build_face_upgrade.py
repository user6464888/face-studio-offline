"""Blend the fitted facial relief into the scan and create photo-textured GLBs."""
from pathlib import Path
import argparse
import json
import shutil
import cv2
import numpy as np
import trimesh
from PIL import Image
from scipy.spatial import cKDTree
from scipy.ndimage import distance_transform_edt, gaussian_filter
from skimage.measure import marching_cubes
from colmap_data import read_text_model, intrinsics, canonical_frame

ROOT=Path(__file__).resolve().parent
SESSION=None

def front_height(mesh,x_values,y_values):
    """Rasterize the nearest front surface instead of interpolating hidden folds."""
    nx,ny=len(x_values),len(y_values)
    pixel=(mesh.vertices[:,:2]-[x_values[0],y_values[0]])/[x_values[1]-x_values[0],y_values[1]-y_values[0]]
    height=np.full((nx,ny),-np.inf,np.float32)
    for face in mesh.faces:
        xy=pixel[face];z=mesh.vertices[face,2]
        lo=np.maximum(np.floor(xy.min(axis=0)).astype(int),0)
        hi=np.minimum(np.ceil(xy.max(axis=0)).astype(int),[nx-1,ny-1])
        if np.any(hi<lo):continue
        a,b,c=xy;den=(b[1]-c[1])*(a[0]-c[0])+(c[0]-b[0])*(a[1]-c[1])
        if abs(den)<1e-8:continue
        xx,yy=np.meshgrid(np.arange(lo[0],hi[0]+1),np.arange(lo[1],hi[1]+1),indexing='ij')
        wa=((b[1]-c[1])*(xx-c[0])+(c[0]-b[0])*(yy-c[1]))/den
        wb=((c[1]-a[1])*(xx-c[0])+(a[0]-c[0])*(yy-c[1]))/den;wc=1-wa-wb
        valid=(wa>=-1e-5)&(wb>=-1e-5)&(wc>=-1e-5)
        depth=wa*z[0]+wb*z[1]+wc*z[2]
        block=height[lo[0]:hi[0]+1,lo[1]:hi[1]+1]
        np.maximum(block,np.where(valid,depth,-np.inf),out=block)
    return height

def blend_volume(head,face,pitch=.01):
    """Re-mesh the closed scan with a smooth front facial depth patch."""
    grid=head.voxelized(pitch).fill()
    padding=max(4,int(np.ceil(max(0,face.bounds[1,2]-head.bounds[1,2])/pitch))+4)
    occupied=np.pad(grid.matrix,padding)
    origin=grid.transform[:3,3]-padding*pitch
    x=origin[0]+np.arange(occupied.shape[0])*pitch
    y=origin[1]+np.arange(occupied.shape[1])*pitch
    z=origin[2]+np.arange(occupied.shape[2])*pitch
    prior=front_height(face,x,y)
    visible=np.isfinite(prior)
    boundary=distance_transform_edt(visible)*pitch
    weight=np.clip(boundary/.085,0,1)
    # The filled scan already supplies its front depth without triangle folds.
    old_front=np.max(np.where(occupied,z[None,None,:],-np.inf),axis=2)
    usable=visible&np.isfinite(old_front)&(old_front>.1)
    target=np.where(usable,old_front*(1-weight)+np.nan_to_num(prior,neginf=0)*weight,old_front)
    # Only the forward facial region changes; the original back remains occupied.
    replace=usable[:,:,None]&(z[None,None,:]>.08)
    occupied=np.where(replace,z[None,None,:]<=target[:,:,None],occupied)
    sdf=distance_transform_edt(~occupied)-distance_transform_edt(occupied)
    sdf=gaussian_filter(sdf.astype(np.float32),.65)
    vertices,faces,_,_=marching_cubes(sdf,0,spacing=(pitch,pitch,pitch),allow_degenerate=False)
    result=trimesh.Trimesh(vertices+origin,faces,process=True);result.fix_normals()
    pieces=result.split(only_watertight=False)
    result=max(pieces,key=lambda m:len(m.faces))
    trimesh.smoothing.filter_taubin(result,lamb=.5,nu=.53,iterations=4)
    result.visual.vertex_colors=np.asarray(head.visual.vertex_colors)[cKDTree(head.vertices).query(result.vertices)[1]]
    return result,int(usable.sum()),float(np.max(np.abs(target[usable]-old_front[usable])))

def photo_atlas(mesh,world,cameras,images,src,names,folder,front_name,front_priority=None,hair_faces=None,photo_blend=None):
    """Each triangle receives photo UVs from a visible calibrated source camera."""
    w,h=1280,720;cols=4;rows=int(np.ceil(len(names)/cols))
    atlas=Image.new('RGB',(w*cols,h*rows),(155,119,103))
    normals=mesh.face_normals
    centers=world[mesh.faces].mean(axis=1)
    weights=[];projections=[]
    for i,name in enumerate(names):
        picture=Image.open(src/'images'/name).convert('RGB');atlas.paste(picture,((i%cols)*w,(i//cols)*h))
        row=images[name];k=intrinsics(cameras[row['camera_id']])
        xyz=world@row['R'].T+row['t'];q=xyz@k.T;uv=q[:,:2]/q[:,2:]
        projections.append(uv)
        direction=row['center']-centers;direction/=np.maximum(np.linalg.norm(direction,axis=1,keepdims=True),1e-9)
        # The caller stores world normals on the mesh for camera scoring.
        facing=np.maximum(0,np.einsum('ij,ij->i',normals,direction))
        pixel=uv[mesh.faces].mean(axis=1);p=np.rint(pixel).astype(int)
        valid=(p[:,0]>=1)&(p[:,0]<w-1)&(p[:,1]>=1)&(p[:,1]<h-1)
        mask=cv2.imread(str(src/'masks'/(name+'.png')),0)
        ids=np.flatnonzero(valid);valid[ids]&=mask[p[ids,1],p[ids,0]]>0
        # A projected vertex depth buffer rejects the opposite surface.
        vp=np.rint(uv).astype(int);ok=(vp[:,0]>=0)&(vp[:,0]<w)&(vp[:,1]>=0)&(vp[:,1]<h)&(xyz[:,2]>0)
        depth=np.full((h,w),np.inf,np.float32)
        np.minimum.at(depth,(vp[ok,1],vp[ok,0]),xyz[ok,2].astype(np.float32))
        depth=cv2.erode(depth,np.ones((5,5),np.uint8))
        face_depth=xyz[mesh.faces,2].mean(axis=1)
        valid[ids]&=face_depth[ids]<depth[p[ids,1],p[ids,0]]+np.median(xyz[:,2])*.012
        score=facing**8*valid
        if name==front_name:
            score*=5
            if front_priority is not None:
                # Keep one reference photo across the central face, including
                # concave eye/nose surfaces, to prevent glasses/reflection seams.
                priority=front_priority.copy()
                inside=(p[:,0]>=1)&(p[:,0]<w-1)&(p[:,1]>=1)&(p[:,1]<h-1)
                ids_inside=np.flatnonzero(inside)
                inside[ids_inside]&=mask[p[ids_inside,1],p[ids_inside,0]]>0
                priority&=inside
                score[priority]=100
        weights.append(score)
    weights=np.stack(weights);choice=weights.argmax(axis=0)
    found=weights.max(axis=0)>1e-8
    # Areas outside all source masks remain explicitly filled from the nearest
    # photographed face, rather than receiving wall/shirt pixels.
    if not found.all():
        nearest=cKDTree(centers[found]).query(centers[~found])[1]
        choice[~found]=choice[found][nearest]
    count=len(mesh.faces)
    flat=mesh.faces.reshape(-1)
    uv=np.empty((count,3,2),float)
    for i,name in enumerate(names):
        select=choice==i;p=projections[i][mesh.faces[select]]
        p[:,:,0]=np.clip(p[:,:,0],0,w-1)+(i%cols)*w
        p[:,:,1]=np.clip(p[:,:,1],0,h-1)+(i//cols)*h
        uv[select,:,0]=p[:,:,0]/(w*cols)
        uv[select,:,1]=1-p[:,:,1]/(h*rows)
    if photo_blend is not None:
        # Keep high-resolution frontal photo details. Blend side/back regions
        # into the existing smooth multi-view colors, avoiding exposure seams
        # and wall pixels on unobserved crown surfaces.
        pixels=np.array(atlas)
        vertex_colors=np.asarray(mesh.visual.vertex_colors)[:,:3]
        face_colors=vertex_colors[mesh.faces].mean(axis=1)
        for i in np.flatnonzero(photo_blend<.999):
            triangle=np.c_[uv[i,:,0]*(w*cols),(1-uv[i,:,1])*(h*rows)]
            lo=np.maximum(np.floor(triangle.min(axis=0)).astype(int),0)
            hi=np.minimum(np.ceil(triangle.max(axis=0)).astype(int)+1,[w*cols,h*rows])
            if np.any(hi<=lo):continue
            block=pixels[lo[1]:hi[1],lo[0]:hi[0]]
            mask=np.zeros(block.shape[:2],np.uint8)
            cv2.fillConvexPoly(mask,np.rint(triangle-lo).astype(np.int32),255)
            select=mask>0;blend=float(photo_blend[i])
            block[select]=np.clip(block[select]*blend+face_colors[i]*(1-blend),0,255).astype(np.uint8)
        atlas=Image.fromarray(pixels)
    atlas.save(folder/'photo_atlas.jpg',quality=96,subsampling=0)
    if hair_faces is not None:
        hair_mask=np.zeros((h*rows,w*cols),np.uint8)
        for i in np.flatnonzero(hair_faces):
            triangle=np.c_[uv[i,:,0]*(w*cols),(1-uv[i,:,1])*(h*rows)]
            cv2.fillConvexPoly(hair_mask,np.rint(triangle).astype(np.int32),255)
        cv2.imwrite(str(folder/'hair_mask.png'),hair_mask)
    material=trimesh.visual.material.PBRMaterial(name='Local photographs',baseColorTexture=atlas,
        metallicFactor=0,roughnessFactor=1,doubleSided=True)
    textured=trimesh.Trimesh(mesh.vertices[flat],np.arange(count*3).reshape(-1,3),process=False)
    textured.vertex_normals=mesh.vertex_normals[flat]
    textured.visual=trimesh.visual.texture.TextureVisuals(uv=uv.reshape(-1,2),material=material)
    textured.metadata.update({'model_inferred':True,'metric_scale':False,'source':'Local RGB video',
        'texture_method':'calibrated photo atlas','unphotographed_fraction':float(1-found.mean())})
    textured.export(folder/'head_textured.glb',include_normals=True)
    return {'photo_assigned_triangle_fraction':float(found.mean()),'source_frames':names,
        'atlas_size':[w*cols,h*rows],'export_vertices':len(textured.vertices)}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--session',required=True);args=parser.parse_args()
    src=ROOT/'inputs'/args.session;out=ROOT/'outputs'/args.session
    import os
    assets=Path(os.environ.get('HEAD_RGB_ASSETS',str(ROOT/'viewer/assets')));folder=assets/'face_upgrade';folder.mkdir(exist_ok=True)
    before=assets/'before_face_upgrade';before.mkdir(exist_ok=True)
    previous=json.loads((before/'metadata.json').read_text(encoding='utf-8')) if (before/'metadata.json').exists() else {}
    if previous and previous.get('source_session')!=args.session:
        archive=assets/'before_face_upgrade_archive'/previous['source_session']
        if not archive.exists():shutil.copytree(before,archive)
    for name in ['head.ply','head_base.ply','head.glb','metadata.json','report.md']:
        if not (before/name).exists() or previous.get('source_session')!=args.session:shutil.copy2(assets/name,before/name)
    cameras,images,sparse,_,_=read_text_model(out/'colmap_aliked/sparse_text')
    basis,origin=canonical_frame(images,sparse)
    raw=trimesh.load(out/'colmap_aliked/dense/mesh_refined.ply',process=False)
    raw.vertices=(raw.vertices-origin)@basis.T
    trimesh.smoothing.filter_taubin(raw,lamb=.5,nu=.53,iterations=3)
    head=trimesh.load(before/'head.ply',process=False)
    # Recover the original viewer normalization; face and scan then share a frame.
    if len(raw.vertices)!=len(head.vertices):raise RuntimeError('Viewer normalization needs a matching reconstructed mesh')
    points=trimesh.load(out/'colmap_aliked/dense/fused.ply',process=False).vertices
    local_points=(points-origin)@basis.T
    front_view=images[min(images)];cam=cameras[front_view['camera_id']]
    center_cam=front_view['R']@origin+front_view['t']
    depth=np.median((points@front_view['R'].T+front_view['t'])[:,2])
    floor=center_cam[1]-(625-cam['params'][2])/cam['params'][0]*depth
    bounds=np.quantile(local_points[local_points[:,1]>floor],[.005,.995],axis=0)
    top=np.quantile(raw.vertices[:,1],.999);scale=2/(top-floor)
    center=np.array([(bounds[0,0]+bounds[1,0])*.5,(top+floor)*.5,(bounds[0,2]+bounds[1,2])*.5])
    affine=np.r_[np.eye(3)*scale,(-center*scale)[None]]
    candidate=np.c_[raw.vertices,np.ones(len(raw.vertices))]@affine
    rms=float(np.sqrt(np.mean(cKDTree(head.vertices).query(candidate)[0]**2)))
    if rms>1e-4:raise RuntimeError(f'Inconsistent mesh normalization: {rms}')
    fit=np.load(out/'face_prior/face_fit.npz')
    face_canonical=(fit['world']-origin)@basis.T
    face_vertices=np.c_[face_canonical,np.ones(len(face_canonical))]@affine
    face=trimesh.Trimesh(face_vertices,fit['faces'],process=False);face.fix_normals()
    if face.vertex_normals[:,2].mean()<0:face.invert()
    face.export(folder/'face_prior.ply')
    head,patch_pixels,maximum_change=blend_volume(head,face)
    head.export(folder/'head_geometry.ply')
    normalized_to_canonical=np.linalg.inv(affine[:3])
    world=(head.vertices-affine[3])@normalized_to_canonical@basis+origin
    face_world=(face.vertices-affine[3])@normalized_to_canonical@basis+origin
    # Camera scores require world-space normals, not viewer-space normals.
    world_mesh=trimesh.Trimesh(world,head.faces,process=False);world_mesh.fix_normals()
    world_mesh.visual.vertex_colors=head.visual.vertex_colors
    world_face=trimesh.Trimesh(face_world,face.faces,process=False)
    if (world_face.vertex_normals@basis.T)[:,2].mean()<0:world_face.invert()
    poses=json.loads((out/'colmap_aliked/pose_review.json').read_text())['cameras']
    names=[]
    preferred_front='frame_000087.jpg' if 'frame_000087.jpg' in images else min(poses,key=lambda r:abs(r['yaw_degrees']))['file']
    for target in range(-150,181,30):
        name=min(poses,key=lambda r:abs(r['yaw_degrees']-target))['file']
        if target==0:name=preferred_front
        if name not in names:names.append(name)
    # Use world geometry for scoring, then move only exported positions back to viewer space.
    center=head.triangles_center
    photo_blend=np.clip((.63-np.abs(center[:,0]))/.16,0,1)*np.clip((center[:,2]-.20)/.15,0,1)
    priority=photo_blend>.001
    scalp=json.loads((before/'metadata.json').read_text(encoding='utf-8'))['scalp']
    theta=np.arctan2(center[:,0]/scalp['radiusX'],(center[:,2]-scalp['centerZ'])/scalp['radiusZ']);cos=np.cos(theta)
    line=scalp['sideLine']+(scalp['frontLine']-scalp['sideLine'])*np.maximum(0,cos)**1.4+(scalp['backLine']-scalp['sideLine'])*np.maximum(0,-cos)**1.3
    hair_faces=center[:,1]>line+.01
    stats=photo_atlas(world_mesh,world,cameras,images,src,names,folder,preferred_front,priority,hair_faces,photo_blend)
    stats['texture_method']='frontal photo UV with a smooth transition into prior multi-view surface colors'
    textured=next(iter(trimesh.load(folder/'head_textured.glb',process=False).geometry.values()))
    transform=np.eye(4);transform[:3,:3]=affine[:3].T@basis;transform[:3,3]=affine[3]-transform[:3,:3]@origin
    textured.apply_transform(transform)
    textured.export(folder/'head_textured.glb',include_normals=True)
    face_folder=folder/'face_only';face_folder.mkdir(exist_ok=True)
    face_names=[n for n in names if abs(next(r['yaw_degrees'] for r in poses if r['file']==n))<70]
    center=face.triangles_center
    priority=(np.abs(center[:,0])<.56)&(center[:,2]>.30)
    face_stats=photo_atlas(world_face,face_world,cameras,images,src,face_names,face_folder,preferred_front,priority)
    face_textured=next(iter(trimesh.load(face_folder/'head_textured.glb',process=False).geometry.values()))
    face_textured.apply_transform(transform)
    face_textured.export(folder/'face_textured.glb',include_normals=True)
    # Keep representative source photographs alongside the comparison viewer.
    refs=[]
    detector=cv2.FaceDetectorYN.create(str(ROOT/'models/yunet_2023mar.onnx'),'',(640,360),.8,.3,5000)
    for target in [-60,0,60,180]:
        name=min(poses,key=lambda r:abs(r['yaw_degrees']-target))['file']
        if target==0:name=preferred_front
        dest=folder/f'source_{target}.jpg';shutil.copy2(src/'images'/name,dest)
        picture=cv2.imread(str(src/'images'/name));_,detections=detector.detect(cv2.resize(picture,(640,360)))
        x,y,w,h=detections[0,:4]*2 if detections is not None else [580,180,340,400]
        cx=x+w/2;cy=y+h/2-70;side=max(w,h)*1.65
        crop=Image.open(dest).crop((int(cx-side*.48),int(cy-side*.55),int(cx+side*.48),int(cy+side*.55)))
        crop.save(folder/f'source_crop_{target}.jpg',quality=96)
        row=images[name]
        refs.append({'angle':target,'name':name,'path':f'assets/face_upgrade/source_{target}.jpg',
            'camera_R':row['R'].tolist(),'camera_t':row['t'].tolist(),'K':intrinsics(cameras[row['camera_id']]).tolist()})
    data={'source_session':args.session,'method':'shared 3DDFA_V2 facial prior + silhouette head + calibrated photo UV atlas',
        'face_is_model_inferred':True,'metric_scale':False,'facial_patch_pixels':patch_pixels,
        'vertices':len(head.vertices),'faces':len(head.faces),'is_watertight':bool(head.is_watertight),
        'maximum_relief_change':maximum_change,
        'normalization_rms':rms,'head':stats,'face':face_stats,'source_views':refs,
        'canonical_basis':basis.tolist(),'canonical_origin':origin.tolist(),'viewer_affine':affine.tolist(),
        'limitations':['Face shape is inferred from a pretrained prior, not measured depth.',
            'Glasses remain in the photograph texture; no separate glasses geometry.',
            'Back of head and hairstyle presets remain approximate.'],
        'prior_repository':'https://github.com/cleardusk/3DDFA_V2',
        'prior_license':'BFM academic use; tools/3ddfa_v2/bfm/readme.md'}
    (folder/'metadata.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
    fit_info=json.loads((out/'face_prior/face_fit.json').read_text(encoding='utf-8'))
    report=f'''# 人脸改进版记录

来源：同一段 {args.session} 本地 RGB 视频。

- 9 个正面与侧面视角拟合共同脸部形状；关键点对齐误差均值 {fit_info['mean_landmark_fit_error_pixels']:.2f} 像素。这是与模型推断关键点的一致程度，不是实测三维精度。
- 新版完整人头：{len(head.vertices)} 个共享顶点，{len(head.faces)} 个三角面；闭合：{head.is_watertight}。
- 正面脸部优先采用单帧照片纹理；向侧面与后脑平滑过渡为旧版多视角照片颜色，降低光照接缝。
- 眼镜仍在照片纹理中，没有单独重建镜框与透明镜片。预设发型仍是程序生成的演示素材，本次未替换发型素材。

文件：head_textured.glb（完整头部）、face_textured.glb（独立脸部）、head_geometry.ply（用于检查的闭合几何）、metadata.json（来源与参数）。网页 /compare.html 可查看前后对比。

人脸先验来自 3DDFA_V2：https://github.com/cleardusk/3DDFA_V2 。BFM 模板仅供学术使用，来源说明见 tools/3ddfa_v2/bfm/readme.md；代码和权重来源、校验值见 tools/3ddfa_v2/source.json。检测器来自 OpenCV YuNet：https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet 。

照片在本机处理，未上传到外部服务；未下载外部人脸采集数据集。脸部包含模型推断，后脑与头皮含轮廓补全，结果不是深度相机扫描，也没有实际尺寸标定。
'''
    (folder/'report.md').write_text(report,encoding='utf-8')
    (assets/'report.md').write_text(report+'\n\n---\n\n'+(before/'report.md').read_text(encoding='utf-8'),encoding='utf-8')
    print(json.dumps({k:v for k,v in data.items() if k not in ['source_views','canonical_basis','canonical_origin','viewer_affine']},indent=2),flush=True)

if __name__=='__main__':main()
