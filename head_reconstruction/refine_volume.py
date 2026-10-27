"""Fuse calibrated silhouettes with reliable local MVS depths into a closed surface.

Unobserved interior detail falls back to the visual hull. This is an explicit
geometric completion, not a claim of measured sensor depth.
"""
from pathlib import Path
import argparse
import json
import cv2
import numpy as np
import torch
from scipy.ndimage import distance_transform_edt, gaussian_filter
from skimage.measure import marching_cubes
import trimesh
from colmap_data import read_text_model, canonical_frame, intrinsics

ROOT=Path(__file__).resolve().parent
SESSION=None


def read_map(path):
    with Path(path).open('rb') as f:
        header=b''
        while header.count(b'&')<3:
            header+=f.read(1)
        w,h,c=map(int,header[:-1].split(b'&'))
        values=np.fromfile(f,np.float32)
    return values.reshape((w,h,c),order='F').transpose(1,0,2).squeeze()


def main():
    p=argparse.ArgumentParser();p.add_argument('--session',required=True);a=p.parse_args()
    src=ROOT/'inputs'/a.session
    output=ROOT/'outputs'/a.session/'colmap_aliked'
    dense=output/'dense'
    cameras,images,points,_,_=read_text_model(output/'sparse_text')
    dcam,dimgs,_,_,_=read_text_model(dense/'sparse_text')
    basis,origin=canonical_frame(images,points)
    front=images[min(images)]
    center_cam=front['R']@origin+front['t']
    reference_depth=np.median((points@front['R'].T+front['t'])[:,2])
    floor=center_cam[1]-(615-cameras[front['camera_id']]['params'][2])/cameras[front['camera_id']]['params'][0]*reference_depth
    # Include a generous crown margin. Silhouettes determine the final shape.
    local=(points-origin)@basis.T
    lo=np.quantile(local,.01,axis=0)-[.28,.12,.28]
    hi=np.quantile(local,.99,axis=0)+[.28,.45,.28]
    lo[1]=floor-.03
    step=float(max(hi-lo)/200)
    axes=[np.arange(lo[i],hi[i]+step,step,dtype=np.float32) for i in range(3)]
    shape=tuple(len(a) for a in axes)
    grid=np.stack(np.meshgrid(*axes,indexing='ij'),axis=-1).reshape(-1,3)
    world=grid@basis+origin
    device='cuda' if torch.cuda.is_available() else 'cpu'
    pts=torch.tensor(world,dtype=torch.float32,device=device)
    votes=torch.zeros(len(grid),dtype=torch.int16,device=device)
    print('Volume',shape,'voxels',len(grid),'device',device,flush=True)
    # Conservative dilation accommodates small pose and segmentation errors.
    for index,(name,im) in enumerate(sorted(images.items())):
        camera=cameras[im['camera_id']]
        k=torch.tensor(intrinsics(camera),dtype=torch.float32,device=device)
        r=torch.tensor(im['R'],dtype=torch.float32,device=device)
        t=torch.tensor(im['t'],dtype=torch.float32,device=device)
        projected=pts@r.T+t
        uv=projected@k.T
        uv=torch.round(uv[:,:2]/uv[:,2:3]).long()
        u,v=uv[:,0],uv[:,1]
        valid=(u>=0)&(u<camera['width'])&(v>=0)&(v<camera['height'])&(projected[:,2]>0)
        mask=cv2.imread(str(src/'foreground_masks'/(name+'.png')),0)
        mask=cv2.dilate(mask,np.ones((9,9),np.uint8))
        m=torch.tensor(mask,device=device)
        inside=m[v.clamp(0,camera['height']-1),u.clamp(0,camera['width']-1)]>127
        votes+=(inside&valid).to(torch.int16)
        if index%15==0:print('Silhouettes',index+1,'/',len(images),flush=True)
    occupied=(votes.cpu().numpy().reshape(shape)>=int(np.ceil(len(images)*.96)))
    # Slice via a full column index, then surround the volume by known empty space.
    occupied[:,axes[1]<floor,:]=False
    occupied[[0,-1],:,:]=False;occupied[:,[0,-1],:]=False;occupied[:,:,[0,-1]]=False
    hull_sdf=(distance_transform_edt(~occupied)-distance_transform_edt(occupied))*step
    weight_sum=torch.zeros(len(grid),device=device)
    signed_sum=torch.zeros(len(grid),device=device)
    names=(output/'dense_images.txt').read_text().splitlines()
    trunc=.055
    for index,name in enumerate(names):
        im=dimgs[name];camera=dcam[im['camera_id']]
        path=dense/'stereo/depth_maps'/(name+'.geometric.bin')
        depth=read_map(path)
        h,w=depth.shape
        k=intrinsics(camera).copy();k[0]*=w/camera['width'];k[1]*=h/camera['height']
        r=torch.tensor(im['R'],dtype=torch.float32,device=device)
        t=torch.tensor(im['t'],dtype=torch.float32,device=device)
        projected=pts@r.T+t
        uv=projected@torch.tensor(k,dtype=torch.float32,device=device).T
        uv=torch.round(uv[:,:2]/uv[:,2:3]).long();u,v=uv[:,0],uv[:,1]
        valid=(u>=0)&(u<w)&(v>=0)&(v<h)&(projected[:,2]>0)
        # Filter isolated matches. Median filtering is only used where it has data.
        filtered=cv2.medianBlur(depth,5)
        stable=(depth>0)&(filtered>0)&(np.abs(depth-filtered)<.025)
        value=torch.tensor(filtered,device=device)[v.clamp(0,h-1),u.clamp(0,w-1)]
        stable_t=torch.tensor(stable,device=device)[v.clamp(0,h-1),u.clamp(0,w-1)]
        signed=value-projected[:,2]
        valid&=stable_t&(signed>=-trunc)&(signed<=trunc*2)
        weight=valid.to(torch.float32)
        signed_sum+=signed.clamp(-trunc,trunc)*weight
        weight_sum+=weight
        if index%8==0:print('Depth fusion',index+1,'/',len(names),flush=True)
    weights=weight_sum.cpu().numpy().reshape(shape)
    tsdf=(signed_sum/weight_sum.clamp(min=1)).cpu().numpy().reshape(shape)
    reliable=weights>=5
    sdf=hull_sdf.copy()
    # Carve only repeatably observed surface recesses; preserve the silhouette
    # where MVS is missing or unstable, including the crown and mirror lenses.
    observed=np.maximum(sdf[reliable],tsdf[reliable]-.012)
    sdf[reliable]=sdf[reliable]*.65+observed*.35
    sdf=gaussian_filter(sdf,1.25)
    verts,faces,_,_=marching_cubes(sdf,0,spacing=(step,step,step),allow_degenerate=False)
    verts+=lo
    mesh=trimesh.Trimesh(vertices=verts@basis+origin,faces=faces,process=True)
    components=mesh.split(only_watertight=False)
    mesh=max(components,key=lambda m:len(m.faces))
    mesh.fix_normals()
    mesh.export(dense/'mesh_refined.ply')
    (dense/'volume_refinement.json').write_text(json.dumps({'method':'Calibrated 77-view silhouette hull intersected with stable 37-view MVS TSDF',
        'voxel_size_scene_units':step,'grid_shape':shape,'mask_dilation_pixels':9,
        'silhouette_support_required':.96,'minimum_depth_support':5,'depth_carving_fraction':.35,
        'completion_note':'Missing or unstable MVS detail uses silhouette-derived geometry; not sensor depth.',
        'vertices':len(mesh.vertices),'faces':len(mesh.faces),'is_watertight':bool(mesh.is_watertight)},indent=2),encoding='utf-8')
    print('Refined',len(mesh.vertices),len(mesh.faces),'closed',mesh.is_watertight,flush=True)


if __name__=='__main__':main()
