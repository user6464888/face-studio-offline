"""Bake captured facial skin patches onto the estimated bare scalp, offline."""
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree
from scalp_fit import smoothstep

try:
    from numba import njit
except ImportError:
    def njit(*args,**kwargs):
        return lambda function:function


def to_linear(rgb):
    a=np.asarray(rgb,dtype=np.float32)/255
    return np.where(a<=.04045,a/12.92,((a+.055)/1.055)**2.4)


def to_srgb(linear):
    a=np.clip(linear,0,1)
    return np.clip(np.rint(np.where(a<=.0031308,a*12.92,1.055*a**(1/2.4)-.055)*255),0,255).astype(np.uint8)


def captured_frame(assets):
    from rgbd_context import inputs,OUT
    run=assets.parents[2]
    config_path=run/'config.json'
    if config_path.exists():
        config=json.loads(config_path.read_text(encoding='utf-8'))
        folder=Path(config['inputs']['primary']);work=Path(config['work'])
        manifest=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
    else:
        folder,manifest=inputs('primary');work=OUT
    metadata=json.loads((assets/'metadata.json').read_text(encoding='utf-8'))
    index=metadata['preferred_frontal_frame']
    registration=json.loads((work/'primary/registration.json').read_text(encoding='utf-8'))
    indices=registration.get('frame_indices',list(range(len(registration['poses_camera_to_first_head']))))
    pose=np.asarray(registration['poses_camera_to_first_head'][indices.index(index)])
    path=folder/manifest['images'][index]['file']
    image=cv2.cvtColor(cv2.imread(str(path)),cv2.COLOR_BGR2RGB)
    return image,path,manifest['camera_calibration'],pose,np.asarray(metadata['reference_camera_centre_m'])


def project_model(vertices,calibration,pose,centre):
    world=vertices*np.array([1,-1,-1])+centre
    camera=(world-pose[:3,3])@pose[:3,:3]
    xyz=camera@np.asarray(calibration['transform']['rot']).reshape(3,3).T+np.asarray(calibration['transform']['trans'])/1000
    intr=calibration['rgbIntrinsic'];d=calibration['rgbDistortion']
    matrix=np.array([[intr['fx'],0,intr['cx']],[0,intr['fy'],intr['cy']],[0,0,1.]])
    coefficients=np.array([d[n] for n in ['k1','k2','p1','p2','k3','k4','k5','k6']])
    uv=cv2.projectPoints(xyz,np.zeros(3),np.zeros(3),matrix,coefficients)[0].reshape(-1,2)
    return uv,xyz


def rgb_vertices_to_world(vertices,projection):
    local=vertices/(projection['canonical_scale']*projection.get('display_scale',1.))+np.asarray(projection['local_centre'])
    return local@np.asarray(projection['world_basis'])+np.asarray(projection['world_origin'])


def rgb_skin_frame(assets,vertices,metadata):
    """Use this RGB reconstruction's own cameras and photos, never RGB-D defaults."""
    from colmap_data import read_text_model,intrinsics
    root=Path(__file__).resolve().parent
    projection=metadata.get('rgb_projection')
    if not projection:raise RuntimeError('旧 RGB 结果缺少照片投影记录，请从两段录像重新计算。')
    session=projection['session'];folder=root/'inputs'/session
    cameras,images,_,_,_=read_text_model(root/'outputs'/session/'colmap_aliked/sparse_text')
    main=[name for name in images if name.startswith('primary_')] or sorted(images)
    first=images[min(main)];candidates=[name for name in main if np.linalg.norm(images[name]['R']-first['R'])<.12]
    manifest=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
    sharpness={Path(row['file']).name:row.get('sharpness_reference',0) for row in manifest['images']}
    name=max(candidates,key=lambda x:sharpness.get(x,0));im=images[name];cam=cameras[im['camera_id']]
    xyz=rgb_vertices_to_world(vertices,projection)@im['R'].T+im['t']
    k=intrinsics(cam);uv=xyz@k.T;uv=uv[:,:2]/uv[:,2:3]
    path=folder/'images'/name;image=cv2.imread(str(path))
    if image is None:raise RuntimeError('无法读取本次 RGB 重建的正面照片。')
    fx=k[0,0]/(projection['canonical_scale']*projection.get('display_scale',1.))
    return cv2.cvtColor(image,cv2.COLOR_BGR2RGB),path,uv,xyz,float(fx)


def extract_patches(assets,source,weights,c):
    v=np.asarray(source.vertices);x=v[:,0]-c['centerX'];y=v[:,1];z=v[:,2]-c['centerZ']
    metadata=json.loads((Path(assets)/'metadata.json').read_text(encoding='utf-8'))
    if metadata.get('sensor_depth') is False:
        image,path,uv,camera,fx=rgb_skin_frame(assets,v,metadata)
    else:
        image,path,cal,pose,centre=captured_frame(assets)
        uv,camera=project_model(v,cal,pose,centre);fx=cal['rgbIntrinsic']['fx']
    skin=(weights==0)&(z>.05)&(np.asarray(source.vertex_normals)[:,2]>.35)
    regions={
        'forehead':skin&(abs(x)<.038)&(y>c['frontLine']-.039)&(y<c['frontLine']-.021),
        'left_cheek':skin&(x>.033)&(x<.059)&(y>-.044)&(y<-.016),
        'right_cheek':skin&(x<-.033)&(x>-.059)&(y>-.044)&(y<-.016),
    }
    patches=[];records=[];audit=image.copy();h,w=image.shape[:2]
    for name,region in regions.items():
        pixels=np.rint(uv[region]).astype(np.int32)
        pixels=pixels[(pixels[:,0]>=0)&(pixels[:,0]<w)&(pixels[:,1]>=0)&(pixels[:,1]<h)]
        if len(pixels)<6:continue
        mask=np.zeros((h,w),np.uint8)
        cv2.fillConvexPoly(mask,cv2.convexHull(pixels),255)
        mask=cv2.erode(mask,np.ones((3,3),np.uint8))
        distance=cv2.distanceTransform(mask,cv2.DIST_L2,5)
        for _ in range(3):
            cy,cx=np.unravel_index(np.argmax(distance),distance.shape)
            radius=min(12,int(distance[cy,cx])-1)
            if radius<7:break
            bounds=[int(cx-radius),int(cy-radius),int(cx+radius),int(cy+radius)]
            patch=image[bounds[1]:bounds[3],bounds[0]:bounds[2]].copy()
            # The geometric regions already exclude facial features. Reject
            # any remaining very dark crop rather than clone a hair/shadow.
            if float(np.median(patch.mean(axis=2)))>40:
                patches.append(patch);records.append({'region':name,'bounds_xyxy':bounds,'pixels':list(patch.shape[:2])})
                cv2.rectangle(audit,tuple(bounds[:2]),tuple(bounds[2:]),(255,210,60),2)
            cv2.circle(distance,(int(cx),int(cy)),radius+3,0,-1)
    if len(patches)<2:raise RuntimeError('未找到足够干净的额头或脸颊纹理，请使用更清晰的正面采集。')
    forehead=[p for p,r in zip(patches,records) if r['region']=='forehead']
    base=np.median(np.concatenate([p.reshape(-1,3) for p in forehead or patches]),axis=0)
    # Keep a cropped, local audit image, rather than duplicating the full room.
    face=uv[(abs(x)<.085)&(y>-.11)&(y<.12)&(z>.015)]
    lo=np.maximum(np.floor(face.min(axis=0)).astype(int)-8,0);hi=np.minimum(np.ceil(face.max(axis=0)).astype(int)+8,[w,h])
    cv2.imwrite(str(assets/'skin_donor_regions.jpg'),cv2.cvtColor(audit[lo[1]:hi[1],lo[0]:hi[0]],cv2.COLOR_RGB2BGR))
    density=float(fx/np.median(camera[skin,2]))
    return patches,base,density,{'source_frame':str(path),'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'patches':records,'pixels_per_metre':density}


def quilt_detail(patches,size=512):
    """Stochastic overlapping real-photo patches, with low-frequency light removed."""
    rng=np.random.default_rng(6491);block=24;overlap=8;step=block-overlap
    exemplars=[]
    for patch in patches:
        linear=to_linear(patch)
        # Preserve pores/skin marks from the photograph while removing the
        # patch's directional lighting, which would form repeating rectangles.
        detail=linear-cv2.GaussianBlur(linear,(0,0),4)
        for turn in range(4):
            exemplars.append(cv2.resize(np.rot90(detail,turn).copy(),(block,block),interpolation=cv2.INTER_LINEAR))
    acc=np.zeros((size+block,size+block,3),np.float32);weight=np.zeros(acc.shape[:2],np.float32)
    ramp=np.ones(block,np.float32);ramp[:overlap]=smoothstep(np.linspace(.05,1,overlap));ramp[-overlap:]=ramp[:overlap][::-1]
    window=ramp[:,None]*ramp[None,:]
    for yy in range(0,size,step):
        for xx in range(0,size,step):
            patch=exemplars[int(rng.integers(len(exemplars)))].copy()
            if rng.integers(2):patch=patch[:,::-1]
            acc[yy:yy+block,xx:xx+block]+=patch*window[:,:,None]
            weight[yy:yy+block,xx:xx+block]+=window
    detail=acc[:size,:size]/np.maximum(weight[:size,:size,None],1e-8)
    detail-=detail.mean(axis=(0,1),keepdims=True)
    # Limit outliers without flattening the measured small-scale contrast.
    limit=np.maximum(np.std(detail,axis=(0,1))*3.5,.001)
    return np.clip(detail,-limit,limit)


def spherical_uv(points,c):
    centre=np.array([c['centerX'],c['centerY'],c['centerZ']])
    radii=np.array([c['radiusX'],c['radiusY'],c['radiusZ']])
    q=(points-centre)/radii
    r=np.maximum(np.linalg.norm(q,axis=1),1e-8)
    uv=np.column_stack([(np.arctan2(q[:,0],q[:,2])+np.pi)/(2*np.pi),1-np.arccos(np.clip(q[:,1]/r,-1,1))/np.pi])
    return uv,r


def unwrap_seam(points,faces,uv):
    mapping=list(range(len(points)));coords=list(uv);triangles=faces.copy();duplicates={}
    for i in np.flatnonzero(np.ptp(uv[faces,0],axis=1)>.5):
        for j in range(3):
            old=int(faces[i,j])
            if uv[old,0]<.5:
                if old not in duplicates:
                    duplicates[old]=len(mapping);mapping.append(old);coords.append(uv[old]+[1,0])
                triangles[i,j]=duplicates[old]
    return np.asarray(mapping,np.int32),triangles,np.asarray(coords,np.float32)


@njit(cache=True)
def raster_atlas(faces,uv,positions,normals,colours,amount,radius,width,height):
    rgb=np.zeros((height,width,3),np.float32);xyz=np.zeros_like(rgb);normal=np.zeros_like(rgb)
    alpha=np.zeros((height,width),np.float32);depth=np.full((height,width),-1e10,np.float32)
    for row in faces:
        a,b,c=row
        ax=uv[a,0]*width-.5;ay=(1-uv[a,1])*height-.5
        bx=uv[b,0]*width-.5;by=(1-uv[b,1])*height-.5
        cx=uv[c,0]*width-.5;cy=(1-uv[c,1])*height-.5
        det=(by-cy)*(ax-cx)+(cx-bx)*(ay-cy)
        if abs(det)<1e-9:continue
        xmin=int(np.floor(min(ax,bx,cx)));xmax=int(np.ceil(max(ax,bx,cx)))
        ymin=max(0,int(np.floor(min(ay,by,cy))));ymax=min(height-1,int(np.ceil(max(ay,by,cy))))
        for yy in range(ymin,ymax+1):
            for xx in range(xmin,xmax+1):
                wa=((by-cy)*(xx-cx)+(cx-bx)*(yy-cy))/det
                wb=((cy-ay)*(xx-cx)+(ax-cx)*(yy-cy))/det;wc=1-wa-wb
                if min(wa,wb,wc)<-1e-5:continue
                x=xx%width;z=radius[a]*wa+radius[b]*wb+radius[c]*wc
                if z<depth[yy,x]:continue
                depth[yy,x]=z;alpha[yy,x]=amount[a]*wa+amount[b]*wb+amount[c]*wc
                for k in range(3):
                    rgb[yy,x,k]=colours[a,k]*wa+colours[b,k]*wb+colours[c,k]*wc
                    xyz[yy,x,k]=positions[a,k]*wa+positions[b,k]*wb+positions[c,k]*wc
                    normal[yy,x,k]=normals[a,k]*wa+normals[b,k]*wb+normals[c,k]*wc
    return rgb,xyz,normal,alpha,depth


def write_uv_ply(path,points,faces,uv,colours):
    vertex=np.empty(len(points),dtype=[('x','<f4'),('y','<f4'),('z','<f4'),('s','<f4'),('t','<f4'),('red','u1'),('green','u1'),('blue','u1')])
    for k,name in enumerate(('x','y','z')):vertex[name]=points[:,k]
    vertex['s']=uv[:,0];vertex['t']=uv[:,1]
    for k,name in enumerate(('red','green','blue')):vertex[name]=colours[:,k]
    triangle=np.empty(len(faces),dtype=[('count','u1'),('index','<i4',(3,))]);triangle['count']=3;triangle['index']=faces
    header=f'ply\nformat binary_little_endian 1.0\nelement vertex {len(points)}\nproperty float x\nproperty float y\nproperty float z\nproperty float s\nproperty float t\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nelement face {len(faces)}\nproperty list uchar int vertex_indices\nend_header\n'
    with path.open('wb') as stream:stream.write(header.encode('ascii'));stream.write(vertex.tobytes());stream.write(triangle.tobytes())


def build_skin_texture(assets,source,bald,weights,protected,c,ear_guard=None):
    assets=Path(assets);points=np.asarray(bald.vertices);hair=weights>0
    patches,skin_mean,density,provenance=extract_patches(assets,source,weights,c)
    detail=quilt_detail(patches)
    to_hair,_=cKDTree(source.vertices[hair]).query(source.vertices,workers=4)
    repair=(hair|(to_hair<.012))&~protected
    distance,_=cKDTree(points[~repair]).query(points,workers=4)
    amount=smoothstep(distance/.018)*repair
    normals=np.asarray(bald.vertex_normals)
    light=np.array([-.15,.55,.82]);light/=np.linalg.norm(light)
    shading=(.9+.1*np.maximum(normals@light,0))/(.9+.1*.8)
    original=to_linear(np.asarray(bald.visual.vertex_colors)[:,:3])
    target=to_linear(skin_mean)[None,:]*shading[:,None]
    base=original*(1-amount[:,None])+target*amount[:,None]
    uv,radius=spherical_uv(points,c)
    mapping,faces,coords=unwrap_seam(points,np.asarray(bald.faces),uv)
    width,height=2048,1024
    # Retained face/neck triangles use vertex colours in the viewer. Baking
    # them into this scalp-only atlas lets folded scan edges overwrite ear
    # cleanup texels with background pixels at the same spherical UV.
    textured_faces=repair[np.asarray(bald.faces)].any(axis=1).astype(np.uint8)
    if ear_guard is not None:
        ear_faces=np.asarray(ear_guard)[np.asarray(bald.faces)].any(axis=1)
        if textured_faces[ear_faces].any():raise RuntimeError('头皮贴图侵入了保留的耳廓。')
    atlas_faces=faces[textured_faces>0].astype(np.int32)
    rgb,xyz,normal,alpha,depth=raster_atlas(atlas_faces,coords,points[mapping].astype(np.float32),normals[mapping].astype(np.float32),base[mapping].astype(np.float32),amount[mapping].astype(np.float32),radius[mapping].astype(np.float32),width,height)
    valid=depth>-1e9
    # World-space, three-plane sampling maintains the captured millimetre scale
    # on both the sides and crown, without stretching pores at the UV poles.
    contribution=np.zeros_like(rgb);weights3=abs(normal)**8;weights3/=np.maximum(weights3.sum(axis=2,keepdims=True),1e-9)
    for axis,axes in enumerate(((2,1),(0,2),(0,1))):
        sx=np.ascontiguousarray(xyz[:,:,axes[0]]*density+137,dtype=np.float32)
        sy=np.ascontiguousarray(xyz[:,:,axes[1]]*density+251,dtype=np.float32)
        sampled=cv2.remap(detail,sx,sy,cv2.INTER_LINEAR,borderMode=cv2.BORDER_WRAP)
        contribution+=sampled*weights3[:,:,axis,None]
    rgb+=contribution*alpha[:,:,None]
    if not valid.all():
        nearest=distance_transform_edt(~valid,return_distances=False,return_indices=True)
        rgb[~valid]=rgb[nearest[0][~valid],nearest[1][~valid]]
    texture=to_srgb(rgb)
    cv2.imwrite(str(assets/'head_bald_texture.png'),cv2.cvtColor(texture,cv2.COLOR_RGB2BGR))
    write_uv_ply(assets/'head_bald_uv.ply',points[mapping],faces,coords,np.asarray(bald.visual.vertex_colors)[mapping,:3])
    (assets/'head_bald_texture_faces.bin').write_bytes(textured_faces.tobytes())
    np.savez_compressed(assets/'head_bald_uv_mapping.npz',source_indices=mapping,uv=coords)
    stats={'method':'captured facial skin patches, frequency separation, stochastic quilting, metric triplanar projection and UV bake',
           **provenance,'texture_size':[width,height],'skin_base_rgb':skin_mean.tolist(),
           'uv_vertices':len(mapping),'source_vertices':len(points),'faces':len(faces),
           'texture_detail_rms_linear':float(np.sqrt(np.mean(contribution[alpha>.95]**2))),
           'photo_detail_rms_linear':float(np.sqrt(np.mean(detail**2))),
           'scalp_is_estimated':True,'texture_placement_is_synthesised':True,
           'external_skin_images_used':False,'retained_face_uses_original_vertex_colours':True,
           'textured_scalp_faces':int(textured_faces.sum()),'retained_photo_faces':int((textured_faces==0).sum())}
    if ear_guard is not None:
        stats['ear_faces_use_scan_colours_with_outer_rim_correction']=True
        stats['ear_rim_colour_repair_vertices']=int((bald.metadata.get('ear_rim_mask',np.zeros(len(points)))>0).sum())
    (assets/'skin_texture_sources.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2),encoding='utf-8')
    return stats
