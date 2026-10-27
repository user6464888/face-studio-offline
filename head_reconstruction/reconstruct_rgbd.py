"""Local calibrated head segmentation, registration and measured-depth fusion."""
import os
os.environ.setdefault('OMP_NUM_THREADS','8')
from pathlib import Path
import argparse
import json
import copy
import time
import cv2
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from PIL import Image,ImageDraw
import trimesh

from rgbd_context import ROOT,OUT,ASSETS,inputs
OUT.mkdir(parents=True,exist_ok=True)
REG=o3d.pipelines.registration

def save_json(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
def project(points,cal):
    rot=np.asarray(cal['transform']['rot']).reshape(3,3)
    trans=np.asarray(cal['transform']['trans'])/1000
    xyz=points@rot.T+trans
    intr=cal['rgbIntrinsic'];d=cal['rgbDistortion']
    k=np.array([[intr['fx'],0,intr['cx']],[0,intr['fy'],intr['cy']],[0,0,1.]])
    distortion=np.array([d[n] for n in ['k1','k2','p1','p2','k3','k4','k5','k6']])
    uv=cv2.projectPoints(xyz,np.zeros(3),np.zeros(3),k,distortion)[0].reshape(-1,2)
    return uv,xyz

def prepare_geometry(role,limit=0):
    folder,manifest=inputs(role);cal=manifest['camera_calibration'];di=cal['depthIntrinsic']
    if any(abs(v)>1e-6 for v in cal['depthDistortion'].values()):
        raise RuntimeError('此实现要求已经去畸变的原始深度；当前深度畸变非零。')
    dest=OUT/role;dest.mkdir(exist_ok=True)
    source_record=dest/'source.json'
    identity={'source_session':manifest['source_session'],'first_color_device_us':manifest['images'][0]['color_device_us']}
    if source_record.exists() and json.loads(source_record.read_text(encoding='utf-8'))!=identity:
        raise RuntimeError('输出缓存属于其他采集，停止复用以避免混合人物或录像。请为新采集设置独立输出目录。')
    save_json(source_record,identity)
    previews=[];stats=[];rejected_frames=[]
    rows=list(enumerate(manifest['images']))
    if 'reconstruction_window_seconds' in manifest:
        start,end=manifest['reconstruction_window_seconds']
        rows=[(i,row) for i,row in rows if start<=row['elapsed_seconds']<=end]
    rows=rows[:limit or None]
    first_rgb=cv2.imread(str(folder/rows[0][1]['file']))
    detector=cv2.FaceDetectorYN_create(str(ROOT/'models/yunet_2023mar.onnx'),'',(first_rgb.shape[1],first_rgb.shape[0]),.65)
    faces=detector.detect(first_rgb)[1]
    if faces is None:raise RuntimeError('首帧没有检测到正面人脸，不能自动设定头部深度范围。')
    face=max(faces,key=lambda f:f[2]*f[3])
    first_depth=cv2.imread(str(folder/rows[0][1]['depth_file']),-1).astype(np.float32)*rows[0][1]['depth_value_scale_mm']/1000
    v0,u0=np.where((first_depth>.2)&(first_depth<1.5));z0=first_depth[v0,u0]
    p0=np.column_stack(((u0-di['cx'])/di['fx']*z0,(v0-di['cy'])/di['fy']*z0,z0))
    uv0,_=project(p0,cal)
    inside=(uv0[:,0]>face[0]+face[2]*.15)&(uv0[:,0]<face[0]+face[2]*.85)&(uv0[:,1]>face[1]+face[3]*.1)&(uv0[:,1]<face[1]+face[3]*.85)
    head_distance=float(np.median(z0[inside]))
    if not np.isfinite(head_distance):raise RuntimeError('人脸位置没有可用深度。')
    print(json.dumps({'role':role,'face_depth_m':head_distance}),flush=True)
    sift=cv2.SIFT_create(nfeatures=1400,contrastThreshold=.012,edgeThreshold=12)
    for i,row in rows:
        try:
            path=dest/f'{i:05d}.npz'
            image=cv2.imread(str(folder/row['file']))
            cache_valid=False
            if path.exists():
                with np.load(path) as cached:cache_valid='segmentation_version' in cached and int(cached['segmentation_version'])==3
            if cache_valid:
                with np.load(path) as data:
                    mask=data['rgb_mask'];depth_mask=data['masked_depth']>0;points=data['points']
            else:
                raw=cv2.imread(str(folder/row['depth_file']),cv2.IMREAD_UNCHANGED)
                mm=raw.astype(np.float32)*row['depth_value_scale_mm']
                smooth=cv2.medianBlur(mm,3)
                stable=(mm>200)&(mm<(head_distance+.18)*1000)&(np.abs(mm-smooth)<12)
                binary=cv2.morphologyEx(stable.astype(np.uint8),cv2.MORPH_CLOSE,np.ones((7,7),np.uint8))
                count,labels,table,centres=cv2.connectedComponentsWithStats(binary)
                candidates=[n for n in range(1,count) if table[n,cv2.CC_STAT_AREA]>2000 and abs(centres[n,0]-di['cx'])<di['width']*.3]
                if not candidates:raise RuntimeError(f'没有中央人物深度区域：{role}/{i}')
                component=max(candidates,key=lambda n:table[n,cv2.CC_STAT_AREA])
                person=(labels==component)&stable
                vv,uu=np.where(person);z=smooth[vv,uu]/1000
                pts=np.column_stack(((uu-di['cx'])/di['fx']*z,(vv-di['cy'])/di['fy']*z,z))
                top=float(np.quantile(pts[:,1],.003))
                upper=pts[(pts[:,1]<top+.19)&(pts[:,1]>top+.03)]
                if len(upper)<500:raise RuntimeError('头部深度候选太少。')
                centre=np.median(upper,axis=0)
                half_width=.145-np.clip((pts[:,1]-top-.21)/.055,0,1)*.07
                keep=(pts[:,1]<=top+.265)&(np.abs(pts[:,0]-centre[0])<half_width)&(np.abs(pts[:,2]-centre[2])<.17)
                points=pts[keep].astype(np.float32);uu=uu[keep];vv=vv[keep]
                uv,_=project(points,cal)
                h,w=image.shape[:2]
                valid=(uv[:,0]>=0)&(uv[:,0]<w-1)&(uv[:,1]>=0)&(uv[:,1]<h-1)
                points=points[valid];uv=uv[valid];uu=uu[valid];vv=vv[valid]
                if len(points)<2000:raise RuntimeError(f'头部有效点太少：{role}/{i}')
                pix=np.floor(uv).astype(int);px,py=pix.T;fraction=uv-pix
                fx=fraction[:,0,None];fy=fraction[:,1,None]
                sampled=image[py,px]*(1-fx)*(1-fy)+image[py,px+1]*fx*(1-fy)+image[py+1,px]*(1-fx)*fy+image[py+1,px+1]*fx*fy
                colours=np.clip(sampled[:,::-1],0,255).astype(np.uint8)
                masked=np.zeros(raw.shape,np.float32);masked[vv,uu]=points[:,2]
                native_colour=np.zeros((*raw.shape,3),np.uint8);native_colour[vv,uu]=colours
                mask=np.zeros((h,w),np.uint8)
                ix=np.rint(uv).astype(int);mask[ix[:,1],ix[:,0]]=255
                mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((7,7),np.uint8))
                small=cv2.resize(image,(960,540),interpolation=cv2.INTER_AREA)
                small_mask=cv2.resize(mask,(960,540),interpolation=cv2.INTER_NEAREST)
                keypoints,descriptors=sift.detectAndCompute(cv2.cvtColor(small,cv2.COLOR_BGR2GRAY),small_mask)
                feature_uv=np.array([k.pt for k in keypoints],np.float32).reshape(-1,2)
                if len(feature_uv):
                    distance,indices=cKDTree(uv*np.array([960/w,540/h])).query(feature_uv)
                    usable=distance<1.6
                    feature_xyz=points[indices[usable]];descriptors=descriptors[usable];feature_uv=feature_uv[usable]
                else:
                    feature_xyz=np.zeros((0,3),np.float32);descriptors=np.zeros((0,128),np.float32)
                np.savez_compressed(path,points=points,colours=colours,masked_depth=masked,
                                    native_colour=native_colour,rgb_mask=mask,feature_xyz=feature_xyz,
                                    descriptors=descriptors,feature_uv=feature_uv,segmentation_version=3)
                depth_mask=masked>0
        except (RuntimeError,ValueError) as exc:
            if role!='supplement_top':raise
            rejected_frames.append({'index':i,'reason':str(exc)})
            continue
        stats.append({'index':i,'file':row['file'],'time_seconds':row['elapsed_seconds'],'head_points':len(points)})
        if i%max(1,len(rows)//12)==0 and len(previews)<12:
            preview=image.copy();contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(preview,contours,-1,(30,220,70),3)
            faded=image.copy();faded[mask==0]=(faded[mask==0]*.18+215*.82).astype(np.uint8)
            previews.append((row['elapsed_seconds'],cv2.resize(preview,(480,270)),cv2.resize(faded,(480,270))))
        if i%15==0:print(json.dumps({'stage':'head masks','role':role,'completed':i+1,'total':len(rows)}),flush=True)
    if not stats:raise RuntimeError('没有可靠的头部画面。')
    sheet=Image.new('RGB',(4*480,((len(previews)+3)//4)*572),'#f0f4f8');draw=ImageDraw.Draw(sheet)
    for i,(t,overlay,cutout) in enumerate(previews):
        x=i%4*480;y=i//4*572
        draw.text((x+10,y+4),f'{t:.2f}s | projected depth mask / head crop',fill='#233243')
        sheet.paste(Image.fromarray(cv2.cvtColor(overlay,cv2.COLOR_BGR2RGB)),(x,y+24))
        sheet.paste(Image.fromarray(cv2.cvtColor(cutout,cv2.COLOR_BGR2RGB)),(x,y+298))
    sheet.save(dest/'head_masks.jpg',quality=92)
    save_json(dest/'segmentation.json',{'role':role,'frames':stats,'method':'central near-depth connected component, physical head ROI and calibrated distorted depth-to-RGB projection',
                                      'head_crop_height_m':.265,'head_crop_half_width_m':.145,'masks_are_geometric':True,'initial_face_depth_m':head_distance,
                                      'rgb_depth_projection_visual_review_required':False,'optional_visual_review_recommended':True,'rejected_frames':rejected_frames})
    if not limit:
        manifest['masks_generated']=True;manifest['head_segmentation_result']=str(dest/'segmentation.json')
        save_json(folder/'manifest.json',manifest)

def load_frame(role,index):
    data=np.load(OUT/role/f'{index:05d}.npz')
    pc=o3d.geometry.PointCloud();pc.points=o3d.utility.Vector3dVector(data['points'])
    pc.colors=o3d.utility.Vector3dVector(data['colours']/255)
    pc=pc.voxel_down_sample(.0025)
    pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=.012,max_nn=30))
    pc.orient_normals_towards_camera_location(np.zeros(3))
    return pc,{'xyz':data['feature_xyz'],'desc':data['descriptors']}

def rigid(x,y):
    a=x.mean(axis=0);b=y.mean(axis=0)
    u,_,vt=np.linalg.svd((x-a).T@(y-b));r=vt.T@u.T
    if np.linalg.det(r)<0:vt[-1]*=-1;r=vt.T@u.T
    t=np.eye(4);t[:3,:3]=r;t[:3,3]=b-r@a
    return t

def feature_initial(source,target,seed=0):
    if min(len(source['desc']),len(target['desc']))<8:return None,0
    matches=cv2.BFMatcher().knnMatch(source['desc'],target['desc'],k=2)
    matches=[a for a,b in matches if a.distance<.72*b.distance]
    if len(matches)<6:return None,0
    x=source['xyz'][[m.queryIdx for m in matches]];y=target['xyz'][[m.trainIdx for m in matches]]
    rng=np.random.default_rng(seed);best=np.zeros(len(x),bool)
    for _ in range(350):
        indices=rng.choice(len(x),3,replace=False)
        if np.linalg.norm(np.cross(x[indices[1]]-x[indices[0]],x[indices[2]]-x[indices[0]]))<1e-5:continue
        transform=rigid(x[indices],y[indices])
        good=np.linalg.norm(x@transform[:3,:3].T+transform[:3,3]-y,axis=1)<.008
        if good.sum()>best.sum():best=good
    if best.sum()<6:return None,int(best.sum())
    transform=rigid(x[best],y[best]);return transform,int(best.sum())

def refine(source,target,initial):
    t=initial.copy()
    for voxel,distance,iters in [(.006,.028,35),(.003,.013,25),(.0025,.008,20)]:
        src=source.voxel_down_sample(voxel);dst=target.voxel_down_sample(voxel)
        for pc in (src,dst):
            pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel*4,max_nn=30))
            pc.orient_normals_towards_camera_location(np.zeros(3))
        result=REG.registration_icp(src,dst,distance,t,REG.TransformationEstimationPointToPlane(),REG.ICPConvergenceCriteria(max_iteration=iters))
        t=result.transformation
    try:
        result=REG.registration_colored_icp(source,target,.007,t,REG.TransformationEstimationForColoredICP(lambda_geometric=.985),REG.ICPConvergenceCriteria(max_iteration=15))
    except RuntimeError:
        result=REG.evaluate_registration(source,target,.007,t)
    return result

def register(role,limit=0):
    folder,manifest=inputs(role)
    segmentation=json.loads((OUT/role/'segmentation.json').read_text(encoding='utf-8'))
    available=len(segmentation['frames']);n=min(limit or available,available)
    clouds=[];features=[]
    for i in range(n):pc,feat=load_frame(role,i);clouds.append(pc);features.append(feat)
    poses=[np.eye(4)];edges=[];previous=np.eye(4);graph=REG.PoseGraph();graph.nodes.append(REG.PoseGraphNode(np.eye(4)))
    for j in range(1,n):
        initial,inliers=feature_initial(features[j],features[j-1],j)
        method='SIFT + metric 3D RANSAC'
        if initial is None or Rotation.from_matrix(initial[:3,:3]).magnitude()>np.deg2rad(24):
            initial=previous;method='previous motion prediction'
        result=refine(clouds[j],clouds[j-1],initial)
        if result.fitness<.65 or result.inlier_rmse>.007 or Rotation.from_matrix(result.transformation[:3,:3]).magnitude()>np.deg2rad(28):
            raise RuntimeError(f'{role} 第 {j} 组对齐不可靠：fitness={result.fitness:.3f}, rmse={result.inlier_rmse:.4f}')
        previous=result.transformation
        pose=poses[-1]@previous;poses.append(pose);graph.nodes.append(REG.PoseGraphNode(pose))
        inverse=np.linalg.inv(previous)
        information=REG.get_information_matrix_from_point_clouds(clouds[j-1],clouds[j],.008,inverse)
        graph.edges.append(REG.PoseGraphEdge(j-1,j,inverse,information,uncertain=False))
        edges.append({'source':j,'target':j-1,'fitness':result.fitness,'rmse_m':result.inlier_rmse,'feature_inliers':inliers,'initial_method':method})
        if j%10==0:
            yaw=np.rad2deg(np.unwrap([np.arctan2(p[0,2],p[2,2]) for p in poses]))[-1]
            print(json.dumps({'stage':'registration','role':role,'completed':j,'total':n,'yaw_degrees':round(yaw,1),'rmse_mm':round(result.inlier_rmse*1000,2)}),flush=True)
    links=[(i,i+4) for i in range(0,n-4,4)]
    if n>15:links.extend([(0,n-1),(1,n-4),(3,n-8)])
    loop_count=0;loop_reports=[]
    for i,j in links:
        initial,inliers=feature_initial(features[j],features[i],i*1000+j)
        if initial is None:
            if j-i>10:continue
            initial=np.linalg.inv(poses[i])@poses[j]
        result=refine(clouds[j],clouds[i],initial)
        if result.fitness<.65 or result.inlier_rmse>.0065:continue
        t=np.linalg.inv(result.transformation)
        information=REG.get_information_matrix_from_point_clouds(clouds[i],clouds[j],.008,t)
        trusted_closure=j-i>10 and inliers>=12 and result.fitness>.85 and result.inlier_rmse<.0035
        graph.edges.append(REG.PoseGraphEdge(i,j,t,information,uncertain=not trusted_closure))
        if j-i>10:
            loop_count+=1
            loop_reports.append({'source':j,'target':i,'feature_inliers':inliers,'fitness':result.fitness,
                                 'rmse_mm':result.inlier_rmse*1000,'trusted':trusted_closure})
    REG.global_optimization(graph,REG.GlobalOptimizationLevenbergMarquardt(),REG.GlobalOptimizationConvergenceCriteria(),
                            REG.GlobalOptimizationOption(max_correspondence_distance=.008,edge_prune_threshold=.25,preference_loop_closure=2,reference_node=0))
    poses=[node.pose for node in graph.nodes]
    yaw=np.rad2deg(np.unwrap([np.arctan2(p[0,2],p[2,2]) for p in poses]))
    result={'role':role,'poses_camera_to_first_head':[p.tolist() for p in poses],'yaw_degrees':yaw.tolist(),
            'yaw_span_degrees':float(np.ptp(yaw)),'loop_closures':loop_count,'loop_reports':loop_reports,'edges':edges,
            'median_adjacent_rmse_mm':float(np.median([e['rmse_m'] for e in edges])*1000),
            'quality_passed':bool(limit or role=='supplement_top' or np.ptp(yaw)>330)}
    save_json(OUT/role/'registration.json',result)
    o3d.io.write_pose_graph(str(OUT/role/'pose_graph.json'),graph)
    if not result['quality_passed']:raise RuntimeError(f'{role} 对齐的环绕角度不足：{np.ptp(yaw):.1f} 度')
    if not limit:manifest['poses_estimated']=True;save_json(folder/'manifest.json',manifest)
    return poses,clouds,features

def anchor_top():
    main=json.loads((OUT/'primary/registration.json').read_text(encoding='utf-8'))
    top=json.loads((OUT/'supplement_top/registration.json').read_text(encoding='utf-8'))
    source,sf=load_frame('supplement_top',0)
    candidates=[]
    for i in range(0,len(main['poses_camera_to_first_head']),4):
        target,tf=load_frame('primary',i)
        initial,inliers=feature_initial(sf,tf,10000+i)
        if initial is None or inliers<8:continue
        result=refine(source,target,initial)
        if result.fitness>.65 and result.inlier_rmse<.0065:
            transform=np.asarray(main['poses_camera_to_first_head'][i])@result.transformation
            candidates.append((result.inlier_rmse-inliers*.000005,transform,i,result,inliers))
    if not candidates:raise RuntimeError('高位段暂未找到可靠的同一头部对齐关系，保留 B 段主体，勿强行融合。')
    _,transform,index,result,inliers=min(candidates,key=lambda x:x[0])
    record={'transform_top_to_primary':transform.tolist(),'matched_primary_index':index,'feature_inliers':inliers,
            'fitness':result.fitness,'rmse_mm':result.inlier_rmse*1000}
    save_json(OUT/'top_anchor.json',record);return transform

def register_top_independent(limit=0):
    """Anchor useful crown views directly to B, avoiding a broken top sequence."""
    main=json.loads((OUT/'primary/registration.json').read_text(encoding='utf-8'))
    base_poses=[np.asarray(p) for p in main['poses_camera_to_first_head']]
    bank=[];all_points=[];all_colours=[]
    for i,pose in enumerate(base_poses):
        with np.load(OUT/'primary'/f'{i:05d}.npz') as data:
            all_points.append(data['points']@pose[:3,:3].T+pose[:3,3]);all_colours.append(data['colours']/255)
            if i%4==0:bank.append((i,{'xyz':data['feature_xyz'],'desc':data['descriptors']}))
    template=o3d.geometry.PointCloud();template.points=o3d.utility.Vector3dVector(np.concatenate(all_points))
    template.colors=o3d.utility.Vector3dVector(np.concatenate(all_colours));template=template.voxel_down_sample(.0025)
    template.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=.015,max_nn=40))
    template.orient_normals_towards_camera_location(np.zeros(3))
    segmentation=json.loads((OUT/'supplement_top/segmentation.json').read_text(encoding='utf-8'))
    candidates_indices=[f['index'] for f in segmentation['frames']][:limit or None]
    n=len(candidates_indices)
    poses=[];indices=[];accepted=[];rejected=[]
    for ordinal,j in enumerate(candidates_indices):
        source,features=load_frame('supplement_top',j);candidates=[]
        for i,target_features in bank:
            local,inliers=feature_initial(features,target_features,20000+j*200+i)
            if local is not None and inliers>=10:candidates.append((inliers,base_poses[i]@local,i))
        best=None
        for inliers,initial,index in sorted(candidates,key=lambda x:-x[0])[:3]:
            result=refine(source,template,initial)
            if result.fitness<.5 or result.inlier_rmse>.0045:continue
            score=result.fitness-result.inlier_rmse*25+min(inliers,50)*.001
            if best is None or score>best[0]:best=(score,result,inliers,index)
        if best is not None:
            _,result,inliers,index=best;poses.append(result.transformation.tolist());indices.append(j)
            accepted.append({'index':j,'matched_primary_index':index,'feature_inliers':inliers,'fitness':result.fitness,'rmse_mm':result.inlier_rmse*1000})
        else:rejected.append(j)
        if ordinal%5==0:print(json.dumps({'stage':'independent crown registration','completed':ordinal+1,'total':n,'accepted':len(accepted)}),flush=True)
    if len(poses)<5:raise RuntimeError('可靠的高位补充视角不足，保留主体并检查。')
    record={'role':'supplement_top','poses_camera_to_first_head':poses,'frame_indices':indices,
            'poses_are_in_primary_camera_frame':True,'accepted':accepted,'rejected_indices':rejected,
            'method':'independent SIFT 3D correspondences to B, refined against registered B head points',
            'quality_passed':True,'no_failed_sequential_poses_used':True}
    save_json(OUT/'supplement_top/registration.json',record)
    print(json.dumps({'crown_accepted':len(poses),'crown_rejected':len(rejected)}),flush=True)

def chin_cut_y():
    folder,manifest=inputs('primary');row=manifest['images'][0]
    image=cv2.imread(str(folder/row['file']))
    detector=cv2.FaceDetectorYN_create(str(ROOT/'models/yunet_2023mar.onnx'),'',(image.shape[1],image.shape[0]),.65)
    face=max(detector.detect(image)[1],key=lambda f:f[2]*f[3])
    with np.load(OUT/'primary/00000.npz') as data:points=data['points']
    uv,_=project(points,manifest['camera_calibration'])
    near=(abs(uv[:,0]-(face[0]+face[2]/2))<face[2]*.2)&(abs(uv[:,1]-(face[1]+face[3]))<face[3]*.04)
    if near.sum()<30:raise RuntimeError('无法可靠确定下颌截面。')
    return float(np.median(points[near,1])+.012)

def photo_colour(mesh,jobs):
    vertices=mesh.vertices;normal=mesh.vertex_normals
    total=np.zeros((len(vertices),3),np.float64);weights=np.zeros(len(vertices))
    _,base=inputs('primary');reg=json.loads((OUT/'primary/registration.json').read_text(encoding='utf-8'))
    frontal=[i for i,p in enumerate(reg['poses_camera_to_first_head']) if Rotation.from_matrix(np.asarray(p)[:3,:3]).magnitude()<np.deg2rad(12)]
    preferred=max(frontal,key=lambda i:base['images'][i]['sharpness_reference'])
    for role,anchor in jobs:
        folder,manifest=inputs(role);cal=manifest['camera_calibration'];di=cal['depthIntrinsic']
        reg=json.loads((OUT/role/'registration.json').read_text(encoding='utf-8'))
        indices=reg.get('frame_indices',list(range(len(reg['poses_camera_to_first_head']))))
        for ordinal,(i,p) in enumerate(zip(indices,reg['poses_camera_to_first_head'])):
            if ordinal%2 and not(role=='primary' and i==preferred):continue
            pose=anchor@np.asarray(p);camera=(vertices-pose[:3,3])@pose[:3,:3]
            native_u=np.rint(camera[:,0]/camera[:,2]*di['fx']+di['cx']).astype(int)
            native_v=np.rint(camera[:,1]/camera[:,2]*di['fy']+di['cy']).astype(int)
            direction=pose[:3,3]-vertices;direction/=np.maximum(np.linalg.norm(direction,axis=1,keepdims=True),1e-12)
            facing=np.sum(normal*direction,axis=1)
            good=(camera[:,2]>0)&(native_u>=0)&(native_u<di['width'])&(native_v>=0)&(native_v<di['height'])&(facing>.15)
            if i in reg.get('crown_only_frame_indices',[]):
                good &= vertices[:,1] < reg['crown_cut_y_reference_m']
            ids=np.flatnonzero(good)
            with np.load(OUT/role/f'{i:05d}.npz') as data:depth=data['masked_depth']
            d=depth[native_v[ids],native_u[ids]]
            ids=ids[(d>0)&(abs(d-camera[ids,2])<.007)]
            uv,_=project(camera[ids],cal)
            image=cv2.cvtColor(cv2.imread(str(folder/manifest['images'][i]['file'])),cv2.COLOR_BGR2RGB)
            pix=np.rint(uv).astype(int);h,w=image.shape[:2]
            visible=(pix[:,0]>=0)&(pix[:,0]<w)&(pix[:,1]>=0)&(pix[:,1]<h)
            ids=ids[visible];pix=pix[visible]
            weight=facing[ids]**10
            if role=='primary' and i==preferred:weight*=150
            total[ids]+=image[pix[:,1],pix[:,0]]*weight[:,None];weights[ids]+=weight
    colours=np.asarray(mesh.visual.vertex_colors).copy();found=weights>1e-7
    colours[found,:3]=np.clip(total[found]/weights[found,None],0,255).astype(np.uint8)
    if not found.any():raise RuntimeError('没有顶点取得照片颜色，停止发布模型。')
    if not found.all():
        nearest=cKDTree(vertices[found]).query(vertices[~found])[1]
        colours[~found,:3]=colours[found,:3][nearest]
    mesh.visual.vertex_colors=colours
    return float(found.mean()),preferred

def preview_mesh(mesh,path):
    p=np.asarray(mesh.vertices).copy();col=np.asarray(mesh.visual.vertex_colors)[:,:3];faces=np.asarray(mesh.faces)
    centre=(p.min(axis=0)+p.max(axis=0))/2;p-=centre;p*=2/np.ptp(p[:,1])
    sheet=Image.new('RGB',(1600,960),'#edf2f6');draw=ImageDraw.Draw(sheet)
    for grey in (False,True):
        for j,angle in enumerate([0,90,180,270]):
            a=np.deg2rad(angle);q=p@Rotation.from_euler('y',a).as_matrix().T
            uv=np.column_stack((q[:,0]*185+200,240-q[:,1]*185)).astype(np.int32)
            pic=np.full((480,400,3),[237,242,246],np.uint8)
            order=np.argsort(q[faces,2].mean(axis=1));colours=col[faces].mean(axis=1)
            if grey:
                normal=np.cross(q[faces[:,1]]-q[faces[:,0]],q[faces[:,2]]-q[faces[:,0]])
                normal/=np.maximum(np.linalg.norm(normal,axis=1,keepdims=True),1e-12)
                light=np.array([-.4,.7,1.]);light/=np.linalg.norm(light)
                shade=.45+.55*np.maximum(0,normal@light)
                colours=np.array([163,172,179])[None]*shade[:,None]
            for k in order:cv2.fillConvexPoly(pic,uv[faces[k]],tuple(map(int,colours[k])),lineType=cv2.LINE_AA)
            y=480 if grey else 0;sheet.paste(Image.fromarray(pic),(j*400,y))
            draw.text((j*400+12,y+450),f'{angle} degrees | {"geometry" if grey else "photo color"}',fill='#253c4b')
    sheet.save(path,quality=94)

def fuse(include_top=True):
    jobs=[('primary',np.eye(4))]
    if include_top:
        reg=json.loads((OUT/'supplement_top/registration.json').read_text(encoding='utf-8'))
        jobs.append(('supplement_top',np.eye(4) if reg.get('poses_are_in_primary_camera_frame') else anchor_top()))
    volume=o3d.pipelines.integration.ScalableTSDFVolume(voxel_length=.0015,sdf_trunc=.012,
                                                       color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
    count=0
    for role,anchor in jobs:
        _,manifest=inputs(role);di=manifest['camera_calibration']['depthIntrinsic']
        k=o3d.camera.PinholeCameraIntrinsic(di['width'],di['height'],di['fx'],di['fy'],di['cx'],di['cy'])
        reg=json.loads((OUT/role/'registration.json').read_text(encoding='utf-8'))
        indices=reg.get('frame_indices',list(range(len(reg['poses_camera_to_first_head']))))
        for i,pose in zip(indices,reg['poses_camera_to_first_head']):
            with np.load(OUT/role/f'{i:05d}.npz') as data:
                depth_values=data['masked_depth'].copy()
                if i in reg.get('crown_only_frame_indices',[]):
                    yy,xx=np.where(depth_values>0);z=depth_values[yy,xx]
                    points=np.column_stack(((xx-di['cx'])/di['fx']*z,(yy-di['cy'])/di['fy']*z,z))
                    transform=anchor@np.asarray(pose)
                    world=points@transform[:3,:3].T+transform[:3,3]
                    depth_values[yy[world[:,1]>=reg['crown_cut_y_reference_m']],xx[world[:,1]>=reg['crown_cut_y_reference_m']]]=0
                rgbd=o3d.geometry.RGBDImage.create_from_color_and_depth(o3d.geometry.Image(data['native_colour']),
                      o3d.geometry.Image(depth_values),depth_scale=1.,depth_trunc=1.5,convert_rgb_to_intensity=False)
            volume.integrate(rgbd,k,np.linalg.inv(anchor@np.asarray(pose)));count+=1
            if i%20==0:print(json.dumps({'stage':'depth fusion','role':role,'completed':i+1}),flush=True)
    raw=volume.extract_triangle_mesh();raw.remove_degenerate_triangles();raw.remove_duplicated_triangles();raw.remove_unreferenced_vertices()
    labels,counts,_=raw.cluster_connected_triangles();labels=np.asarray(labels);counts=np.asarray(counts)
    raw.remove_triangles_by_mask(labels!=counts.argmax());raw.remove_unreferenced_vertices()
    raw=raw.filter_smooth_taubin(number_of_iterations=3);raw.compute_vertex_normals()
    vertices=np.asarray(raw.vertices);faces=np.asarray(raw.triangles);rgba=np.column_stack((np.asarray(raw.vertex_colors)*255,np.full(len(vertices),255))).astype(np.uint8)
    world_mesh=trimesh.Trimesh(vertices,faces,vertex_colors=rgba,process=False)
    cut_y=chin_cut_y()
    world_mesh=trimesh.intersections.slice_mesh_plane(world_mesh,[0,-1,0],[0,cut_y,0],cap=False)
    world_mesh.remove_unreferenced_vertices();world_mesh.fix_normals()
    colour_fraction,preferred=photo_colour(world_mesh,jobs)
    triangle_colours=np.asarray(world_mesh.visual.vertex_colors)[world_mesh.faces,:3].mean(axis=1)
    cloth=(world_mesh.triangles_center[:,1]>cut_y-.06)&(triangle_colours[:,0]-triangle_colours[:,1]<3)&(triangle_colours[:,2]-triangle_colours[:,0]>2)&(triangle_colours.mean(axis=1)>45)
    cloth_removed=int(cloth.sum())
    world_mesh.update_faces(~cloth);world_mesh.remove_unreferenced_vertices()
    vertices=np.asarray(world_mesh.vertices);faces=np.asarray(world_mesh.faces);rgba=np.asarray(world_mesh.visual.vertex_colors)
    # Camera X right, Y down, Z away -> model X right, Y up, Z towards front.
    centre=(vertices.min(axis=0)+vertices.max(axis=0))/2
    canonical=(vertices-centre)*[1,-1,-1]
    mesh=trimesh.Trimesh(canonical,faces,vertex_colors=rgba,process=False)
    mesh.fix_normals()
    assets=ASSETS;assets.mkdir(parents=True,exist_ok=True)
    mesh.export(assets/'head.ply');mesh.export(assets/'head.glb');mesh.export(OUT/'head_metric.ply')
    preview_mesh(mesh,assets/'model_overview.jpg')
    main=json.loads((OUT/'primary/registration.json').read_text(encoding='utf-8'))
    metadata={'source_session':inputs('primary')[1]['source_session'],'primary':inputs('primary')[1]['source_label'],
              'sensor_depth':True,'scale_is_metric':True,'units':'metres','method':'calibrated native depth segmentation + SIFT metric RANSAC + colored ICP + pose graph + TSDF',
              'primary_views':len(main['poses_camera_to_first_head']),'top_views':len(json.loads((OUT/'supplement_top/registration.json').read_text(encoding='utf-8'))['poses_camera_to_first_head']) if include_top else 0,
              'fused_views':count,'vertices':len(mesh.vertices),'faces':len(mesh.faces),'is_watertight':bool(mesh.is_watertight),
              'dimensions_mm':(np.ptp(canonical,axis=0)*1000).tolist(),'voxel_mm':1.5,'truncation_mm':12,
              'primary_yaw_span_degrees':main['yaw_span_degrees'],'median_pair_alignment_rmse_mm':main['median_adjacent_rmse_mm'],
              'photo_coloured_vertex_fraction':colour_fraction,'preferred_frontal_frame':preferred,'neck_cut_y_reference_m':cut_y,
              'unobserved_vertex_colour_completion':'nearest photo-coloured surface vertex','cloth_triangles_removed':cloth_removed,
              'reference_camera_centre_m':centre.tolist(),
              'recovered_views_restricted_to_crown':include_top and bool(json.loads((OUT/'supplement_top/registration.json').read_text(encoding='utf-8')).get('crown_only_frame_indices')),
              'summary':'Gemini 2 主录像实测深度融合；头发、耳缘和未观测区域仍可能存在空洞。对齐误差不是物理尺寸准确度。',
              'hair_changes_included':False,'geometry_completed_with_face_prior':False}
    save_json(assets/'metadata.json',metadata);save_json(OUT/'result.json',metadata)
    print(json.dumps(metadata,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['segment','register','register_top','fuse'])
    p.add_argument('--role',choices=['primary','supplement_top'],default='primary');p.add_argument('--limit',type=int,default=0)
    p.add_argument('--primary-only',action='store_true');a=p.parse_args()
    if a.stage=='segment':prepare_geometry(a.role,a.limit)
    elif a.stage=='register':register(a.role,a.limit)
    elif a.stage=='register_top':register_top_independent(a.limit)
    else:fuse(not a.primary_only)
