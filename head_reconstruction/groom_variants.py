"""Offline, explicitly labelled cuts and grooming of the local strand assets."""
import hashlib
import numpy as np
import trimesh
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import cKDTree
from strand_library import SOURCE,read_curves
from scalp_fit import smoothstep

PRESETS={
 'straight':('中分直发','wStraight.hair',28),
 'bob':('中分短波波','wStraight.hair',28),
 'lob':('中分长直发','wStraight.hair',40),
 'fringe':('齐刘海短发','wStraight.hair',28),
 'part':('侧分直发','wStraight.hair',32),
 'wavy':('中分微波浪','wWavy.hair',40),
 'waves':('中分长波浪','wWavy.hair',48),
 'curly':('后梳蓬松卷发','wCurly.hair',52),
 'ponytail':('低马尾','wStraight.hair',48),
}
VARIANTS={'straight':'基础素材适配','wavy':'基础素材适配','curly':'基础素材适配',
 'bob':'直发素材剪短','lob':'直发素材延长','fringe':'直发剪短并增加刘海',
 'part':'直发素材侧分梳理','waves':'波浪素材延长梳理','ponytail':'发根沿头皮汇拢的马尾梳理'}


def unit(v):
    return v/np.maximum(np.linalg.norm(v,axis=-1,keepdims=True),1e-12)


def resample(curves,samples):
    result=np.empty((len(curves),samples,3))
    for i,p in enumerate(curves):
        arc=np.r_[0,np.cumsum(np.linalg.norm(np.diff(p,axis=0),axis=1))]
        for k in range(3):result[i,:,k]=np.interp(np.linspace(0,arc[-1],samples),arc,p[:,k])
    return result


def trim(curves,bottom):
    clipped=[]
    for i,p in enumerate(curves):
        cut=np.flatnonzero((p[:,1]<bottom[i])&(np.arange(len(p))>2))
        if len(cut):
            j=cut[0];t=(bottom[i]-p[j-1,1])/(p[j,1]-p[j-1,1])
            p=np.vstack([p[:j],p[j-1]+t*(p[j]-p[j-1])])
        clipped.append(p)
    return resample(clipped,curves.shape[1])


def ponytail(atlas,roots,samples,rng):
    n=len(roots);s=np.linspace(0,1,samples)
    a=unit((roots-atlas.centre)/atlas.radii)
    b=np.tile(unit(np.array([0.,-.25,-1.])),(n,1))
    angle=np.arccos(np.clip(np.sum(a*b,axis=1),-.999,.999))
    u=np.minimum(s/.58,1)
    direction=(np.sin((1-u[None,:])*angle[:,None])[:,:,None]*a[:,None]+
               np.sin(u[None,:]*angle[:,None])[:,:,None]*b[:,None])/np.sin(angle)[:,None,None]
    direction=unit(direction)
    curve=atlas.centre+direction*(atlas.surface(direction)+.014)[:,:,None]*atlas.radii
    phase=rng.uniform(0,2*np.pi,n);spread=np.sqrt(rng.uniform(0,1,n))
    tail=np.clip((s-.58)/.42,0,1)
    tie=atlas.centre+unit(np.array([0.,-.25,-1.]))*atlas.radii*1.17
    radial=.009+.009*np.sin(np.pi*tail)
    x=tie[0]+np.cos(phase)[:,None]*spread[:,None]*radial+.008*np.sin(tail*4)[None,:]
    y=tie[1]-(.23+rng.uniform(-.025,.025,n))[:,None]*tail
    z=tie[2]+np.sin(phase)[:,None]*spread[:,None]*radial-.018*np.sin(np.pi*tail)
    hanging=np.stack([x,y,z],axis=-1)
    blend=smoothstep((s-.42)/.17)
    return curve*(1-blend[None,:,None])+hanging*blend[None,:,None]


def project_and_root(atlas,curves):
    q=(curves-atlas.centre)/atlas.radii;r=np.linalg.norm(q,axis=2);direction=unit(q)
    surface=gaussian_filter1d(atlas.surface(direction),.8,axis=1)
    arc=np.column_stack([np.zeros(len(curves)),np.cumsum(np.linalg.norm(np.diff(curves,axis=1),axis=2),axis=1)])
    emergence=smoothstep(arc/.006)
    # First samples emerge through the scalp, becoming gradually thicker and
    # lifting tangentially; the rest of the groom cannot burrow into the head.
    clearance=(-.00025+.0011*emergence)/np.mean(atlas.radii)
    near=(curves[:,:,1]>atlas.c['sideLine']-.020)&(r<1.5)
    r=np.where(near,np.maximum(r,surface+clearance),r)
    r[:,0]=surface[:,0]-.00025/np.mean(atlas.radii)
    return atlas.centre+direction*r[:,:,None]*atlas.radii


def density_curves(atlas,key,guides,rng,count=12000,samples=14):
    """Area-sampled crown fibres follow the existing groom, away from the brow.

    The former independent V-shaped forehead strip is intentionally absent.
    Denser short fibres close the wide part and cover the crown with geometry.
    """
    mesh=atlas.skin_mesh;tri=np.asarray(mesh.vertices)[mesh.faces]
    centre=tri.mean(axis=1);q=(centre-atlas.centre)/atlas.radii
    angle=np.arctan2(q[:,0],q[:,2])
    floor=atlas.c['frontLine']+.010-.010*np.sin(angle)**2
    eligible=centre[:,1]>floor
    area=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)/2
    part_x=atlas.c['centerX']+(.020 if key=='part' else 0)
    parted=key not in ('curly','ponytail','fringe')
    if parted:eligible&=abs(centre[:,0]-part_x)>.0005
    priority=1+5*np.exp(-((centre[:,0]-part_x)/.014)**2)
    priority+=1.5*smoothstep((centre[:,1]-atlas.c['frontLine']-.025)/.025)
    probability=area*eligible*priority
    probability/=probability.sum()
    face_ids=rng.choice(len(tri),size=count,p=probability)
    u=np.sqrt(rng.uniform(0,1,count));w=rng.uniform(0,1,count)
    selected=tri[face_ids]
    roots=selected[:,0]*(1-u[:,None])+selected[:,1]*(u*(1-w))[:,None]+selected[:,2]*(u*w)[:,None]
    if parted:
        side=np.where(centre[face_ids,0]>=part_x,1,-1)
        roots[:,0]=part_x+side*np.maximum(abs(roots[:,0]-part_x),.0005)
    # Start at the closest point of a nearby guide, then continue in its
    # authored direction rather than adding a separate frontal hairline.
    guide_ids,sample_ids=np.indices(guides.shape[:2])
    valid=(guides[:,:,1]>atlas.c['frontLine']+.002)&(sample_ids<guides.shape[1]-3)
    points=guides[valid];gi=guide_ids[valid];si=sample_ids[valid]
    nearest=np.empty(count,int)
    for side in ((-1,1) if parted else (0,)):
        ids=np.arange(count) if not parted else np.flatnonzero((roots[:,0]>=part_x)==(side==1))
        candidates=np.arange(len(points)) if not parted else np.flatnonzero((points[:,0]>=part_x)==(side==1))
        _,near=cKDTree(points[candidates]).query(roots[ids],workers=4)
        nearest[ids]=candidates[near]
    result=np.empty((count,samples,3));lengths=rng.uniform(.025,.060,count)
    for i,j in enumerate(nearest):
        path=guides[gi[j],si[j]:]
        arc=np.r_[0,np.cumsum(np.linalg.norm(np.diff(path,axis=0),axis=1))]
        distance=np.linspace(0,min(lengths[i],arc[-1]),samples)
        for axis in range(3):result[i,:,axis]=np.interp(distance,arc,path[:,axis])
        offset=roots[i]-path[0]
        result[i]+=offset[None,:]*np.exp(-distance/.025)[:,None]
    return project_and_root(atlas,result)


def fringe_curves(atlas,rng,count=2200):
    t=np.linspace(0,1,24);x=rng.uniform(-.059,.059,count)+atlas.c['centerX']
    y0=atlas.c['frontLine']+.027+rng.uniform(0,.023,count)
    y1=atlas.c['frontLine']-.038+rng.normal(0,.002,count)
    y=y0[:,None]*(1-t)+y1[:,None]*t
    xx=x[:,None]+.002*np.sin(t*np.pi)[None,:]*rng.uniform(-1,1,count)[:,None]
    yy=(y-atlas.c['centerY'])/atlas.c['radiusY'];nx=(xx-atlas.c['centerX'])/atlas.c['radiusX']
    nz=np.sqrt(np.maximum(.10,1-nx*nx-yy*yy))
    direction=unit(np.stack([nx,yy,nz],axis=-1))
    r=atlas.surface(direction)+(.0005+.004*np.sin(t*np.pi))/np.mean(atlas.radii)
    return project_and_root(atlas,atlas.centre+direction*r[:,:,None]*atlas.radii)


def tubes(atlas,curves,rng,fine=False,density=False):
    curves=atlas.anchor_roots(curves)
    if density:curves=atlas.conform_density(curves)
    n,samples,_=curves.shape;t=np.linspace(0,1,samples)
    tangent=unit(np.gradient(curves,axis=1));ref=np.zeros_like(tangent);ref[:,:,2]=1;ref[abs(tangent[:,:,2])>.9]=[1,0,0]
    a=unit(np.cross(tangent,ref));b=np.cross(tangent,a)
    arc=np.column_stack([np.zeros(n),np.cumsum(np.linalg.norm(np.diff(curves,axis=1),axis=2),axis=1)])
    emergence=.055+.945*smoothstep(arc/(.001 if density else (.002 if fine else .005)))
    width=rng.uniform(.00010,.00017,n) if density else (rng.uniform(.000045,.00009,n) if fine else rng.uniform(.00016,.00027,n))
    radius=width[:,None]*emergence*np.maximum(.025,(1-t)**.4)
    phase=np.arange(3)*2*np.pi/3
    normal=a[:,:,None]*np.cos(phase)[None,None,:,None]+b[:,:,None]*np.sin(phase)[None,None,:,None]
    vertices=curves[:,:,None]+radius[:,:,None,None]*normal
    ring=np.arange(n*samples*3).reshape(n,samples,3);aa=ring[:,:-1];bb=np.roll(aa,-1,axis=2);dd=ring[:,1:];ee=np.roll(dd,-1,axis=2)
    faces=np.concatenate([np.stack([aa,bb,dd],axis=-1).reshape(-1,3),np.stack([bb,ee,dd],axis=-1).reshape(-1,3)])
    rgb=atlas.colours(curves[:,0]).astype(float)
    shade=rng.uniform(.82,1.22,n)[:,None]*(.78+.22*np.sin(t*np.pi*.8))[None,:]
    colours=np.repeat(np.clip(rgb[:,None]*shade[:,:,None],0,255).astype(np.uint8)[:,:,None],3,axis=2).reshape(-1,3)
    mesh=trimesh.Trimesh(vertices.reshape(-1,3),faces,process=False)
    mesh.visual.vertex_colors=np.column_stack([colours,np.full(len(colours),255,np.uint8)])
    mesh.metadata['volume_weights']=np.repeat((smoothstep(arc/.04)*255).astype(np.uint8),3,axis=1).ravel()
    return mesh


def fitted_strands(atlas,key):
    label,filename,samples=PRESETS[key];path=SOURCE/filename
    raw,source_count=read_curves(path,count=6000,samples=samples)
    rng=np.random.default_rng(143)
    canonical=np.stack([-raw[:,:,1]/28,(raw[:,:,2]-15)/45,(raw[:,:,0]+10)/34],axis=-1)
    curves=atlas.centre+canonical*atlas.radii
    root=unit(canonical[:,0]);theta=np.arctan2(root[:,0],root[:,2]);phi=np.arccos(np.clip(root[:,1],-1,1))
    phi*=1+.52*np.maximum(np.cos(theta),0)**2
    direction=np.column_stack([np.sin(phi)*np.sin(theta),np.cos(phi),np.sin(phi)*np.cos(theta)])
    root=atlas.centre+direction*atlas.surface(direction)[:,None]*atlas.radii
    arc=np.column_stack([np.zeros(len(curves)),np.cumsum(np.linalg.norm(np.diff(curves,axis=1),axis=2),axis=1)])
    curves+=(root-curves[:,0])[:,None]*np.exp(-arc/.08)[:,:,None]
    t=np.linspace(0,1,samples)
    if key in ('bob','fringe'):
        curves=trim(curves,np.full(len(curves),atlas.c['sideLine']-.075)+rng.normal(0,.002,len(curves)))
    if key in ('lob','waves'):
        tail=smoothstep((t-.38)/.62)
        curves[:,:,1]-=.125*tail
        if key=='waves':
            phase=rng.uniform(-.4,.4,len(curves))[:,None]
            curves[:,:,0]+=.012*np.sin(t*3.5*np.pi+phase)*tail
            curves[:,:,2]+=.008*np.cos(t*3.5*np.pi+phase)*tail
    if key=='part':
        top=smoothstep((curves[:,:,1]-atlas.c['frontLine']+.01)/.08)
        curves[:,:,0]+=.024*top
        curves[:,:,1]+=.007*top*np.clip(-(curves[:,:,0]-atlas.c['centerX'])/.07,0,1)
    if key=='ponytail':curves=ponytail(atlas,root,samples,rng)
    curves=project_and_root(atlas,curves)
    pieces=[tubes(atlas,curves,rng)]
    density=density_curves(atlas,key,curves,rng);pieces.append(tubes(atlas,density,rng,density=True))
    fringe_count=0
    if key=='fringe':
        bangs=fringe_curves(atlas,rng);fringe_count=len(bangs);pieces.append(tubes(atlas,bangs,rng))
    mesh=trimesh.util.concatenate(pieces)
    mesh.metadata['volume_weights']=np.concatenate([p.metadata['volume_weights'] for p in pieces])
    return mesh,{'name':label,'source_file':filename,'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
       'author':'Cem Yuksel','source_url':'https://www.cemyuksel.com/research/hairmodels/',
       'variant':VARIANTS[key],'original_strands':source_count,'display_strands':len(curves)+len(density)+fringe_count,
       'main_strands':len(curves),'transition_strands':0,'fringe_strands':fringe_count,
       'density_strands':len(density),'density_samples':density.shape[1],
       'separate_forehead_strip':False,'crown_density_area_sampled':True,'density_follows_bald_surface':True,
       'samples_per_strand':samples,'vertices':len(mesh.vertices),'faces':len(mesh.faces),
       'embedded_tapered_roots':True,'dimensions_mm':(mesh.extents*1000).tolist(),'generated_hair_is_measured':False}
