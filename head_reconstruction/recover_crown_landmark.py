"""Candidate-only crown recovery from agreeing facial anchors and temporal ICP."""
import os
os.environ.setdefault('OMP_NUM_THREADS','8')
import argparse
import json
from pathlib import Path
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


def recover(config):
    os.environ['HEAD_RGBD_CONFIG']=str(Path(config).resolve())
    from reconstruct_rgbd import OUT,inputs,load_frame,refine,chin_cut_y,project
    source_rows=json.loads((OUT/'supplement_top/geometry_probe_00000.json').read_text())['candidates']
    anchors=[r for r in source_rows if r['initial_yaw']=='landmarks' and r['landmark_error_mm']<15
             and r['fitness']>.75 and r['rmse_mm']<3.5 and r['median_colour_error']<.16]
    if len(anchors)<2:raise RuntimeError('Less than two independently initialized facial anchors agree.')
    transforms=np.array([r['transformation'] for r in anchors]);rotation=Rotation.from_matrix(transforms[:,:3,:3]).mean().as_matrix()
    seed=np.eye(4);seed[:3,:3]=rotation;seed[:3,3]=np.median(transforms[:,:3,3],axis=0)
    src,_=load_frame('supplement_top',0);centre=np.median(np.asarray(src.points),axis=0)
    positions=np.array([r[:3,:3]@centre+r[:3,3] for r in transforms])
    shift=float(np.max(np.linalg.norm(positions-positions.mean(axis=0),axis=1)))
    angle=float(np.max(Rotation.from_matrix(transforms[:,:3,:3]@rotation.T).magnitude()))
    if shift>.003 or angle>np.deg2rad(3):raise RuntimeError('Facial anchor consensus is not sufficiently stable.')
    main=json.loads((OUT/'primary/registration.json').read_text())
    points=[];colours=[]
    for i,p in enumerate(main['poses_camera_to_first_head']):
        if i%4:continue
        p=np.asarray(p)
        with np.load(OUT/'primary'/f'{i:05d}.npz') as data:
            points.append(data['points']@p[:3,:3].T+p[:3,3]);colours.append(data['colours']/255)
    xyz=np.concatenate(points);rgb=np.concatenate(colours);keep=xyz[:,1]<chin_cut_y()
    target=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz[keep]));target.colors=o3d.utility.Vector3dVector(rgb[keep])
    target=target.voxel_down_sample(.003);target.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=.015,max_nn=40));target.orient_normals_towards_camera_location(np.zeros(3))
    tree=cKDTree(np.asarray(target.points));target_rgb=np.asarray(target.colors)
    folder,manifest=inputs('supplement_top')
    available=json.loads((OUT/'supplement_top/segmentation.json').read_text())['frames']
    poses={0:seed};accepted=[{'index':0,'method':'three frontal facial initializations with geometric and colour agreement'}];rejected=[]
    for row in available:
        j=row['index']
        if j==0:continue
        previous=max(poses);dt=manifest['images'][j]['elapsed_seconds']-manifest['images'][previous]['elapsed_seconds']
        if dt>2.5:rejected.append({'index':j,'reason':'temporal gap'});continue
        source,_=load_frame('supplement_top',j);q=np.asarray(source.points)
        source=source.select_by_index(np.flatnonzero(q[:,1]<np.quantile(q[:,1],.01)+.205))
        p=poses[previous];predicted=p.copy();earlier=sorted(poses)
        if len(earlier)>1:
            k=earlier[-2];delta=p@np.linalg.inv(poses[k]);prior=manifest['images'][previous]['elapsed_seconds']-manifest['images'][k]['elapsed_seconds']
            factor=min(2,dt/max(prior,.05));motion=np.eye(4)
            motion[:3,:3]=Rotation.from_rotvec(Rotation.from_matrix(delta[:3,:3]).as_rotvec()*factor).as_matrix()
            motion[:3,3]=delta[:3,3]*factor;predicted=motion@p
        candidates=[]
        for initial in (predicted,p):
            result=refine(source,target,initial);t=result.transformation
            moved=np.asarray(source.points)@t[:3,:3].T+t[:3,3];distance,near=tree.query(moved);visible=distance<.007
            color_error=float(np.median(np.linalg.norm(np.asarray(source.colors)[visible]-target_rgb[near[visible]],axis=1))) if visible.any() else 1.
            motion_angle=float(np.rad2deg(Rotation.from_matrix((t@np.linalg.inv(p))[:3,:3]).magnitude()))
            moved_old=np.asarray(source.points)@p[:3,:3].T+p[:3,3]
            motion_shift=float(np.linalg.norm(moved.mean(0)-moved_old.mean(0)))
            if result.fitness>.7 and result.inlier_rmse<.0045 and color_error<.16 and motion_angle<max(18,35*dt) and motion_shift<max(.045,.12*dt):
                candidates.append((result.fitness-result.inlier_rmse*30-color_error*.6,t,
                    {'index':j,'fitness':result.fitness,'rmse_mm':result.inlier_rmse*1000,'median_colour_error':color_error,'motion_angle_degrees':motion_angle,'time_gap_seconds':dt}))
        if candidates:
            _,t,diagnostic=max(candidates,key=lambda x:x[0]);poses[j]=t;accepted.append(diagnostic)
        else:rejected.append({'index':j,'reason':'geometry/colour/motion consistency'})
        if j%10==0:print(json.dumps({'frame':j,'accepted':len(poses),'total':len(available)}),flush=True)
    if len(poses)<5:raise RuntimeError('Insufficient crown frames pass temporal validation.')
    base_folder,base_manifest=inputs('primary');first=base_manifest['images'][0]
    image=cv2.imread(str(base_folder/first['file']))
    detector=cv2.FaceDetectorYN_create(str(Path(__file__).resolve().parent/'models/yunet_2023mar.onnx'),'',(image.shape[1],image.shape[0]),.65)
    face=max(detector.detect(image)[1],key=lambda x:x[2]*x[3])
    with np.load(OUT/'primary/00000.npz') as data:reference=data['points']
    uv,_=project(reference,base_manifest['camera_calibration'])
    near=(abs(uv[:,0]-(face[0]+face[2]*.5))<face[2]*.25)&(abs(uv[:,1]-face[1])<15)
    if near.sum()<30:raise RuntimeError('Cannot protect the original face boundary.')
    indices=sorted(poses)
    result={'role':'supplement_top','poses_camera_to_first_head':[poses[i].tolist() for i in indices],
        'frame_indices':indices,'poses_are_in_primary_camera_frame':True,'quality_passed':True,
        'accepted':accepted,'rejected_diagnostics':rejected,'rejected_indices':[x['index'] for x in rejected],
        'method':'facial depth landmark consensus + temporally checked geometric and colour ICP; candidate only until fusion validation',
        'crown_only_frame_indices':indices,'crown_cut_y_reference_m':float(np.median(reference[near,1])),
        'facial_anchor_count':len(anchors),'anchor_agreement_shift_mm':shift*1000,'anchor_agreement_angle_degrees':float(np.rad2deg(angle)),
        'time_span_seconds':[manifest['images'][indices[0]]['elapsed_seconds'],manifest['images'][indices[-1]]['elapsed_seconds']],
        'face_preservation':'Every recovered view contributes only above the first-frame upper face boundary.'}
    destination=OUT/'supplement_top/landmark_recovery_candidate.json';destination.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({'candidate':str(destination),'accepted':len(indices),'span':result['time_span_seconds'],'anchor_shift_mm':shift*1000,'anchor_angle_degrees':float(np.rad2deg(angle))}),flush=True)


if __name__=='__main__':
    import cv2
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args();recover(a.config)
