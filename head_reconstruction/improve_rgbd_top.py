"""Recover crown depth views using temporal, geometric and colour constraints."""
from pathlib import Path
import json
import copy
import argparse
import cv2
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from reconstruct_rgbd import ROOT,OUT,inputs,load_frame,refine,REG,save_json,project


def recover(stop=0):
    current=OUT/'supplement_top/registration.json'
    old=json.loads(current.read_text(encoding='utf-8'))
    main=json.loads((OUT/'primary/registration.json').read_text(encoding='utf-8'))
    folder,manifest=inputs('supplement_top')
    available=json.loads((OUT/'supplement_top/segmentation.json').read_text(encoding='utf-8'))['frames']
    available_indices=[f['index'] for f in available if not stop or f['index']<stop]
    inferred=set(old.get('crown_only_frame_indices',[]))
    direct={i:np.asarray(p) for i,p in zip(old['frame_indices'],old['poses_camera_to_first_head']) if i not in inferred}
    points=[];colours=[]
    for role,indices,poses in [('primary',range(len(main['poses_camera_to_first_head'])),main['poses_camera_to_first_head']),
                             ('supplement_top',list(direct),list(direct.values()))]:
        for i,p in zip(indices,poses):
            pose=np.asarray(p)
            with np.load(OUT/role/f'{i:05d}.npz') as data:
                xyz=data['points']@pose[:3,:3].T+pose[:3,3]
                # Registration ignores the shirt and lower neck.
                valid=xyz[:,1]<.13
                points.append(xyz[valid]);colours.append(data['colours'][valid]/255)
    template=o3d.geometry.PointCloud();template.points=o3d.utility.Vector3dVector(np.concatenate(points))
    template.colors=o3d.utility.Vector3dVector(np.concatenate(colours));template=template.voxel_down_sample(.003)
    template.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=.015,max_nn=35))
    template.orient_normals_towards_camera_location(np.zeros(3))
    xyz=np.asarray(template.points);rgb=np.asarray(template.colors);tree=cKDTree(xyz)
    centre=(np.quantile(xyz,.02,axis=0)+np.quantile(xyz,.98,axis=0))/2
    accepted=[];poses={};failures=[]
    for j in available_indices:
        if j in direct:
            poses[j]=direct[j];accepted.append({'index':j,'method':'existing independent feature anchor'});continue
        if not poses:continue
        previous=max(poses);dt=manifest['images'][j]['elapsed_seconds']-manifest['images'][previous]['elapsed_seconds']
        if dt>2.5:
            failures.append({'index':j,'reason':'temporal gap too large'});continue
        source,_=load_frame('supplement_top',j)
        p=poses[previous]
        before=[i for i in poses if i<previous]
        predicted=p.copy()
        if before:
            k=max(before);delta=p@np.linalg.inv(poses[k])
            prior_dt=manifest['images'][previous]['elapsed_seconds']-manifest['images'][k]['elapsed_seconds']
            factor=min(2,dt/max(prior_dt,.05))
            motion=np.eye(4);motion[:3,:3]=Rotation.from_rotvec(Rotation.from_matrix(delta[:3,:3]).as_rotvec()*factor).as_matrix()
            motion[:3,3]=delta[:3,3]*factor;predicted=motion@p
        candidates=[]
        for angle in [0,-8,8]:
            delta=Rotation.from_euler('y',angle,degrees=True).as_matrix()
            initial=predicted.copy();initial[:3,:3]=delta@predicted[:3,:3]
            initial[:3,3]=centre+delta@(predicted[:3,3]-centre)
            result=refine(source,template,initial)
            transform=result.transformation
            moved=np.asarray(source.points)@transform[:3,:3].T+transform[:3,3]
            distance,nearest=tree.query(moved)
            visible=distance<.007
            error=np.linalg.norm(np.asarray(source.colors)[visible]-rgb[nearest[visible]],axis=1)
            colour_error=float(np.median(error)) if len(error) else 1.
            delta_angle=float(np.rad2deg(Rotation.from_matrix((transform@np.linalg.inv(p))[:3,:3]).magnitude()))
            shift=float(np.linalg.norm(moved.mean(0)-(np.asarray(source.points)@p[:3,:3].T+p[:3,3]).mean(0)))
            plausible=delta_angle<max(18,35*dt) and shift<max(.045,.12*dt)
            if result.fitness>.7 and result.inlier_rmse<.0045 and colour_error<.16 and plausible:
                score=result.fitness-result.inlier_rmse*30-colour_error*.5-delta_angle*.001
                candidates.append((score,transform,{'index':j,'method':'temporal pose prediction + full-head coloured ICP',
                          'fitness':result.fitness,'rmse_mm':result.inlier_rmse*1000,'median_colour_error_rgb01':colour_error,
                          'motion_angle_degrees':delta_angle,'time_gap_seconds':dt}))
        if candidates:
            _,transform,record=max(candidates,key=lambda a:a[0]);poses[j]=transform;accepted.append(record)
        else:failures.append({'index':j,'reason':'geometry/colour/motion consistency check failed'})
        if j%5==0:print(json.dumps({'index':j,'total':len(available_indices),'accepted':len(poses),'recovered':len(poses)-sum(i<j+1 for i in direct)}),flush=True)
    indices=sorted(poses)
    record={'role':'supplement_top','poses_camera_to_first_head':[poses[i].tolist() for i in indices],
            'frame_indices':indices,'poses_are_in_primary_camera_frame':True,'quality_passed':True,
            'accepted':accepted,'rejected_indices':[v['index'] for v in failures],'rejected_diagnostics':failures,
            'method':'independent SIFT anchors + temporal motion prediction + measured full-head geometry and colour checks',
            'old_accepted_count':len(direct),'recovered_count':len(indices)-len(direct),
            'time_span_seconds':[manifest['images'][indices[0]]['elapsed_seconds'],manifest['images'][indices[-1]]['elapsed_seconds']]}
    base_folder,base_manifest=inputs('primary')
    image=cv2.imread(str(base_folder/base_manifest['images'][0]['file']))
    detector=cv2.FaceDetectorYN_create(str(ROOT/'models/yunet_2023mar.onnx'),'',(image.shape[1],image.shape[0]),.65)
    face=max(detector.detect(image)[1],key=lambda v:v[2]*v[3])
    with np.load(OUT/'primary/00000.npz') as data:reference=data['points']
    uv,_=project(reference,base_manifest['camera_calibration'])
    near=(abs(uv[:,0]-(face[0]+face[2]*.5))<face[2]*.25)&(abs(uv[:,1]-face[1])<15)
    if near.sum()<30:raise RuntimeError('没有足够深度确定头顶与脸部的分界，停止合并。')
    record['crown_only_frame_indices']=sorted(set(indices)-set(direct))
    record['crown_cut_y_reference_m']=float(np.median(reference[near,1]))
    record['face_preservation']='New recovered views only contribute geometry and colour above the first-frame upper face bounding box.'
    save_json(OUT/'supplement_top/recovery_candidate.json',record)
    print(json.dumps({'accepted':len(indices),'recovered':record['recovered_count'],'time_span':record['time_span_seconds']},ensure_ascii=False),flush=True)
    return record


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stop',type=int,default=0);p.add_argument('--apply',action='store_true');a=p.parse_args()
    data=recover(a.stop)
    if a.apply:save_json(OUT/'supplement_top/registration.json',data)
