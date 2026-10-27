"""Offline authored groom recipes and depth-surface retargeting for 19 cuts."""
import json
import hashlib
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
import trimesh
from scipy.spatial import cKDTree

ROOT=Path(__file__).resolve().parent
RECON=ROOT.parent
sys.path.insert(0,str(RECON))
from salon.math_utils import unit
from salon.catalog import STYLES

VERSION=23
LONG_DENSITY=5
MEDIUM_ROOT_COUNT=20000
MEDIUM_STEPS=10
SCENES={}
EAR_SCENES={}


def ear_surface_distance(head,points):
    """Distance to continuous ear triangles, not isolated mesh vertices."""
    key=head['fingerprint']
    if key not in EAR_SCENES:
        mesh=trimesh.load(ROOT/head['bald_file'],process=False)
        ear=np.load(ROOT/head['masks_file'])['ear'].astype(bool)
        faces=np.asarray(mesh.faces)[ear[np.asarray(mesh.faces)].any(axis=1)]
        EAR_SCENES[key]=scene_for(trimesh.Trimesh(mesh.vertices,faces,process=False)) if len(faces) else None
    scene=EAR_SCENES[key]
    if scene is None:return np.full(len(points),np.inf)
    return scene.compute_distance(o3d.core.Tensor(np.asarray(points,np.float32))).numpy()


def smooth(t):
    t=np.clip(t,0,1);return t*t*(3-2*t)


def scene_for(mesh):
    scene=o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.core.Tensor(np.asarray(mesh.vertices,np.float32)),o3d.core.Tensor(np.asarray(mesh.faces,np.uint32)))
    return scene


def radial_field(mesh,centre,radii,width=192,height=96):
    angle=np.linspace(-np.pi,np.pi,width,endpoint=False)
    y=np.linspace(-.85,1,height)
    theta,yy=np.meshgrid(angle,y);xz=np.sqrt(np.maximum(0,1-yy*yy))
    d=np.stack([xz*np.sin(theta),yy,xz*np.cos(theta)],-1)
    vectors=d*np.asarray(radii);length=np.linalg.norm(vectors,axis=-1)
    rays=np.column_stack([np.tile(centre,(width*height,1)),unit(vectors).reshape(-1,3)])
    result=scene_for(mesh).cast_rays(o3d.core.Tensor(rays.astype(np.float32)))
    radius=result['t_hit'].numpy().reshape(height,width)/length
    missing=~np.isfinite(radius)|(radius<.3)|(radius>2.5)
    if missing.any():
        p=np.asarray(centre)+vectors.reshape(-1,3)
        hit=scene_for(mesh).compute_closest_points(o3d.core.Tensor(p.astype(np.float32)))
        fallback=np.linalg.norm((hit['points'].numpy()-centre)/radii,axis=1).reshape(height,width)
        radius[missing]=fallback[missing]
    return radius.astype(np.float32),float(missing.mean())


def sample_radius(field,d):
    h,w=field.shape
    theta=np.arctan2(d[...,0],d[...,2])
    x=(theta+np.pi)/(2*np.pi)*w
    y=np.clip((d[...,1]+.85)/1.85*(h-1),0,h-1)
    x0=np.floor(x).astype(int)%w;x1=(x0+1)%w
    y0=np.floor(y).astype(int);y1=np.minimum(y0+1,h-1)
    a=x-np.floor(x);b=y-y0
    return (field[y0,x0]*(1-a)+field[y0,x1]*a)*(1-b)+(field[y1,x0]*(1-a)+field[y1,x1]*a)*b


def roots(count,seed):
    rng=np.random.default_rng(seed)
    theta=rng.uniform(-np.pi,np.pi,count)
    floor=-.11+.48*np.maximum(np.cos(theta),0)**1.4-.50*np.maximum(-np.cos(theta),0)**.7
    y=floor+(1-floor)*rng.uniform(0,1,count)
    # Extra independent roots on the crown; every style inherits this coverage.
    n=count//4;y[:n]=rng.uniform(.78,.998,n);theta[:n]=rng.uniform(-np.pi,np.pi,n)
    xz=np.sqrt(1-y*y)
    return np.column_stack([xz*np.sin(theta),y,xz*np.cos(theta)])


def medium_roots(count):
    """Area-stratified roots over the hairline, with extra crown coverage.

    Golden-angle sampling avoids random empty patches. These independent
    roots are shared by all medium cuts, including their side/nape regions.
    """
    n=int(count*.85);i=np.arange(n)
    theta=(i*np.pi*(3-np.sqrt(5)))%(2*np.pi)-np.pi
    co=np.cos(theta)
    floor=-.30+.68*np.maximum(co,0)**1.4-.44*np.maximum(-co,0)**.65
    # Keep the hairline soft, without roots below the existing short layer.
    y=floor+.012+(1-floor-.014)*(i+.5)/n
    k=np.arange(count-n)
    theta=np.r_[theta,(k*np.pi*(3-np.sqrt(5))+.73)%(2*np.pi)-np.pi]
    y=np.r_[y,.82+.178*(k+.5)/max(1,len(k))]
    xz=np.sqrt(np.maximum(0,1-y*y))
    return np.column_stack([xz*np.sin(theta),y,xz*np.cos(theta)])


def long_roots(count):
    """Even angular/area samples within the original outer-hair region."""
    n=int(count*.85);i=np.arange(n);gold=np.pi*(3-np.sqrt(5))
    theta=(i*gold+.31)%(2*np.pi)-np.pi
    floor=-.11+.48*np.maximum(np.cos(theta),0)**1.4-.50*np.maximum(-np.cos(theta),0)**.7
    y=floor+(1-floor)*(i+.5)/n
    k=np.arange(count-n)
    theta=np.r_[theta,(k*gold+1.13)%(2*np.pi)-np.pi]
    y=np.r_[y,.78+.218*(k+.5)/max(1,len(k))]
    xz=np.sqrt(np.maximum(0,1-y*y))
    return np.column_stack([xz*np.sin(theta),y,xz*np.cos(theta)])


def authored_curves(style,count=7600,steps=28,minimum_length=.009,medium=False,dense_long=False):
    p=style['design'];family=style['family'];rng=np.random.default_rng(style['number']*1553)
    if family=='buzz':return np.empty((0,steps,3)),np.empty(0)
    if family=='locs':count=78;steps=48
    d=medium_roots(count) if medium else long_roots(count) if dense_long else roots(count,319)
    side=1-smooth((d[:,1]-.24)/.46)
    back=smooth((-d[:,2]-.10)/.60)
    front=smooth((d[:,2]-.30)/.50)
    length=p['length']*(1-side)+p['side']*side
    length=length*(1-back*.68)+p['nape']*back*.68
    length=length*(1-front*.55)+p['fringe']*front*.55
    # Shared lock lengths and lift create visible clumps and an irregular
    # silhouette, while the short surface supplies continuous scalp coverage.
    theta=np.arctan2(d[:,0],d[:,2])
    cell_angle,cell_y=(.065,.045) if medium else (.22,.16)
    lock_id=np.floor((theta+np.pi)/cell_angle)+np.floor(d[:,1]/cell_y)*(100 if medium else 40)
    lock_random=np.mod(np.sin(lock_id*17.123+style['number'])*43758.5453,1)
    length*=(.72+.52*lock_random)*rng.uniform(.97,1.03,len(d))
    eligible=length>minimum_length
    if not medium and family in ('crop','spike','caesar'):eligible&=d[:,1]>.22
    if dense_long:
        # The side silhouette includes protruding ears in the depth scan.
        # Redistribute these candidates into actual hair-bearing regions,
        # rather than growing the fivefold layer from the ear surface.
        ear_zone=(abs(d[:,0])>.84)&(d[:,1]<.25)&(abs(d[:,2])<.38)
        eligible&=~ear_zone
        eligible&=~((abs(d[:,0])<.80)&(d[:,2]>.40)&(d[:,1]<.26))
    if family=='locs':eligible=d[:,1]>.28
    d=d[eligible];length=length[eligible];n=len(d)
    q=np.empty((n,steps,3));q[:,0]=d
    phase=np.arctan2(d[:,0],d[:,2]);part=p.get('part',0)
    lock_id=lock_id[eligible];lock_random=lock_random[eligible]
    side_sign=np.where(d[:,0]>=part,1.,-1.)
    cluster=np.floor((phase+np.pi)/.17)+np.floor(d[:,1]/.14)*40
    lock_phase=np.mod(np.sin(cluster*12.9898)*43758.5453,1)*np.pi*2
    lift=p['volume']*(.65+.35*smooth((d[:,1]-.35)/.5))
    lift*=(.70+.65*lock_random)*rng.uniform(.97,1.03,n)
    for j in range(1,steps):
        t=j/(steps-1);prev=unit(q[:,j-1])
        if family in ('part','mushroom'):
            flow=np.column_stack([side_sign,np.full(n,-.20),np.full(n,.12)])
            if family=='part':
                bangs=d[:,2]>.42
                flow[bangs]=np.column_stack([side_sign[bangs]*.8,-np.full(bangs.sum(),.72),np.full(bangs.sum(),.35)])
        elif family=='sideback':flow=np.column_stack([side_sign*.90,np.full(n,-.08),np.full(n,-.55)])
        elif family=='back':flow=np.tile([0.,.08,-1.],(n,1))
        else:
            flow=np.column_stack([.20*np.sin(phase*3),np.full(n,-.12),np.ones(n)])
            rear=d[:,2]<-.15;flow[rear]=np.column_stack([.2*np.sin(phase[rear]),-np.ones(rear.sum()),-np.full(rear.sum(),.25)])
        if family in ('mullet','mushroom'):
            rear=d[:,2]<-.10;flow[rear]=np.column_stack([.2*np.sin(phase[rear]),-np.ones(rear.sum()),-np.full(rear.sum(),.20)])
        if (medium or dense_long) and family not in ('part','mushroom','sideback','back','locs'):
            blend=smooth((-d[:,2]+.02)/.40)
            front_flow=np.column_stack([.20*np.sin(phase*3),np.full(n,-.12),np.ones(n)])
            rear_flow=np.column_stack([.2*np.sin(phase),-np.ones(n),-np.full(n,.25)])
            flow=front_flow*(1-blend[:,None])+rear_flow*blend[:,None]
            flow[:,0]+=.35*np.tanh(d[:,0]/.22)*4*blend*(1-blend)
        if family=='locs':flow=np.column_stack([.35*np.sin(phase),np.full(n,.5),-np.ones(n)])
        # A small crown vortex blends into the authored overall flow.
        distance=np.linalg.norm(prev[:,[0,2]]-np.array([.08,-.24]),axis=1)
        radial=np.column_stack([prev[:,0]-.08,np.zeros(n),prev[:,2]+.24])
        whirl=unit(radial+.35*np.cross(prev,radial))
        mix=.8*np.exp(-(distance/.19)**2)
        flow=unit(flow*(1-mix[:,None])+whirl*mix[:,None])
        tangent=unit(flow-prev*np.sum(flow*prev,axis=1)[:,None])
        on_head=unit(prev+tangent*(np.minimum(length,.13)/(.092*(steps-1)))[:,None])
        rho=1+lift/.095*np.sin(np.pi*t)**.7*smooth(t/.12)
        q[:,j]=on_head*rho[:,None]
        # Medium and long ends hang outside the ears and nape rather than
        # wrapping around the face or continuing down through the skull.
        hang=smooth((t-.52)/.48)*np.maximum(length-.085,0)
        if family=='locs':hang*=.12
        q[:,j,1]-=hang/.095
        q[:,j,0]+=side_sign*hang/.095*.16
        if family in ('spike','crop'):
            q[:,j,1]+=p.get('spike',.4)*t**1.5*length/.095*.42
            q[:,j,2]+=p.get('spike',.4)*t**1.4*length/.095*.16
        if family=='caesar':q[:,j,1]-=front[eligible]*t**2*.13
    # Correlated locks produce coherent waves/foil crimp, not random spaghetti.
    amplitude=p.get('curl',0)
    if amplitude:
        t=np.linspace(0,1,steps);arc=length[:,None]*t
        wave=2*np.pi*arc/p.get('wave',.06)+lock_phase[:,None]
        env=smooth(t/.18)[None,:]
        a=unit(np.cross(unit(q),np.array([0.,1.,0.])))
        b=unit(np.cross(unit(q),a))
        displacement=(a*np.sin(wave)[:,:,None]+b*np.cos(wave)[:,:,None]*.55)
        q+=displacement*(amplitude/.095)*env[:,:,None]
    if family!='locs' and len(q):
        # Fibres start at separate actual roots, then follow common guides.
        # Unlike uniform extra strands this changes the spatial distribution.
        guide_id=lock_id
        if dense_long:
            # Many small, overlapping bundles retain texture without pulling
            # thousands of independent hairs into the same sparse corridors.
            guide_id=np.floor((phase+np.pi)/.11)+np.floor(d[:,1]/.08)*64
        _,groups=np.unique(guide_id,return_inverse=True)
        guide=np.zeros((groups.max()+1,steps,3));np.add.at(guide,groups,q)
        guide/=np.bincount(groups)[:,None,None]
        t=np.linspace(0,1,steps)
        gather=(.25 if medium else .60 if dense_long else .88)*smooth(t/.75)
        q=q*(1-gather[None,:,None])+guide[groups]*gather[None,:,None]
    # Fine independent variation is bounded, and goes to zero at every root.
    noise=rng.normal(0,.0015,q.shape)
    noise*=np.linspace(0,1,steps)[None,:,None]
    q+=noise
    # Bound the central fringe above the brows. Longer curtain/curl ends may
    # hang beside the temples, but cannot drape across the eyes or nose.
    if family not in ('back','sideback','locs'):
        short_front=family in ('fringe','messy','foil','caesar','crop','spike')
        central=(q[...,2]>(.32 if dense_long and short_front else .58))&(abs(q[...,0])<(1.12 if dense_long and short_front else .94 if short_front else .68))
        lower=-.10 if family in ('part','curl') else .035
        edge=lower+.055*np.sin(lock_phase)[:,None]
        q[...,1]=np.where(central,np.maximum(q[...,1],edge),q[...,1])
    if dense_long and family in ('back','sideback'):
        # Slicked-back styles must not acquire cheek-length curtains when
        # sparse outlier guides become dense. Constrain the frontal flow to
        # its hair-bearing scalp, retaining the radial styling volume.
        rho=np.linalg.norm(q,axis=2);direction=unit(q)
        angle=np.arctan2(direction[...,0],direction[...,2]);co=np.cos(angle)
        hairline=-.11+.48*np.maximum(co,0)**1.4-.50*np.maximum(-co,0)**.7-.025
        yy=np.where(direction[...,2]>.16,np.maximum(direction[...,1],hairline),direction[...,1])
        xz=np.sqrt(np.maximum(0,1-yy*yy))
        q=np.stack([xz*np.sin(angle),yy,xz*np.cos(angle)],axis=2)*rho[:,:,None]
    q[:,0]=d*(1-.00025/.095)
    radii=np.full(n,p.get('loc_radius',.00019))
    if family!='locs':radii*=rng.uniform(.8,1.14,n)
    return q.astype(np.float32),radii.astype(np.float32)


def long_curves(style):
    """Exactly five times each cut's old eligible count, uniformly rooted.

    Short sides retain the cut's eligibility/length; braids gain internal
    fibres rather than five times as many full-size dreadlocks.
    """
    if style['design'].get('review_revision') and style['family']!='locs':
        from salon.review_groom import guides
        steps=44 if style['design']['flow'] in ('curl','foil') else 32
        return guides(style,style['design'].get('long_count',38000),steps,'long')
    base,radius=authored_curves(style)
    target=len(base)*LONG_DENSITY
    if not target:return base,radius
    if style['family']=='locs':
        tangent=unit(np.gradient(base,axis=1));radial=unit(base)
        a=unit(np.cross(tangent,radial));b=unit(np.cross(tangent,a))
        t=np.linspace(0,1,base.shape[1]);pieces=[]
        for i in range(LONG_DENSITY):
            angle=2*np.pi*i/LONG_DENSITY+t[None,:]*np.pi*2
            offset=(a*np.cos(angle)[:,:,None]+b*np.sin(angle)[:,:,None])*(radius[:,None,None]/.095)*.50
            q=base+offset;q[:,0]=unit(q[:,0])*(1-.00025/.095)
            pieces.append(q)
        return np.concatenate(pieces).astype(np.float32),np.tile(radius*.50,LONG_DENSITY)
    count=7600*LONG_DENSITY
    for _ in range(4):
        q,width=authored_curves(style,count=count,dense_long=True)
        if len(q)>=target:
            ids=np.linspace(0,len(q)-1,target).round().astype(int)
            return q[ids],width[ids]
        count=int(np.ceil(count*target/max(1,len(q))*1.04))
    raise RuntimeError('长发层均匀采样未达到目标数量')


def retarget(q,head,style=None):
    if not len(q):return q.copy()
    c=np.asarray(head['centre']);r=np.asarray(head['radii'])
    rho=np.linalg.norm(q,axis=-1);d=unit(q)
    field=np.load(ROOT/head['radial_file'])['radius']
    foot=sample_radius(field,d)
    # Root-to-tip preservation of the actual depth scalp. Hanging hair stays
    # outside its own underlying scalp while original trajectory is retained.
    t=np.linspace(0,1,q.shape[1])[None,:]
    emergence=smooth(t/.13)
    minimum=.00055*emergence-.00025*(1-emergence)
    if style is not None and style['family']!='locs':
        minimum=minimum+style['design']['volume']*.46*smooth((d[...,1]-.28)/.52)*emergence
    shifted=foot+np.maximum(rho-1,minimum/np.mean(r))
    curves=c+d*shifted[:,:,None]*r
    # Final roots use actual triangles, beyond the directional lookup's
    # interpolation accuracy. All subsequent points receive a smooth tether.
    key=head['fingerprint']
    if key not in SCENES:SCENES[key]=scene_for(trimesh.load(ROOT/head['bald_file'],process=False))
    hit=SCENES[key].compute_closest_points(o3d.core.Tensor(curves[:,0].astype(np.float32)))
    near=hit['points'].numpy();normal=hit['primitive_normals'].numpy()
    normal*=np.where(np.sum(normal*(near-c),axis=1)<0,-1,1)[:,None]
    target=near-normal*.00025
    arc=np.column_stack([np.zeros(len(curves)),np.cumsum(np.linalg.norm(np.diff(curves,axis=1),axis=2),axis=1)])
    curves+=(target-curves[:,0])[:,None]*np.exp(-arc/.015)[:,:,None]
    return curves.astype(np.float32)


def medium_profile(style):
    from copy import deepcopy
    middle=deepcopy(style);p=middle['design']
    for region,maximum in [('length',.035),('side',.022),('nape',.028),('fringe',.028)]:
        p[region]=np.clip(p[region]*.48,.005,maximum)
    p['volume']=min(p['volume']*.44,.011)
    p['curl']=min(p.get('curl',0)*.28,.0035)
    p['wave']=min(p.get('wave',.045),.045)
    if middle['family']=='locs':
        middle['family']='back'
        p.update(length=.012,fringe=.010,side=.005,nape=.005,volume=.004,curl=.001)
    return middle


def fitted_long_curves(q,head,style):
    """Attach roots and end strands before they cross the central face.

    Clip continuous segments, not just guide vertices: moving isolated
    points leaves straight connecting edges across the eyes and cheeks.
    Resample the retained trajectory, preserving curls without bent fringes.
    """
    revised=bool(style['design'].get('review_revision'))
    curves=fit_review_curves(q,head,style,'long') if revised else retarget(q,head,style)
    if not len(curves):return curves
    c=np.asarray(head['centre']);r=np.asarray(head['radii']);v=(curves-c)/r
    start=v[:,:-1];delta=np.diff(v,axis=1)
    lower=np.array([-.78,-1.60,.40]);upper=np.array([.78,.0 if revised else .16,10.])
    parallel=abs(delta)<1e-10;safe=np.where(parallel,1.,delta)
    t0=(lower-start)/safe;t1=(upper-start)/safe
    near=np.where(parallel,-np.inf,np.minimum(t0,t1)).max(axis=2)
    far=np.where(parallel,np.inf,np.maximum(t0,t1)).min(axis=2)
    outside=(parallel&((start<lower)|(start>upper))).any(axis=2)
    hit=(np.maximum(near,0)<=np.minimum(far,1))&~outside
    for i in np.flatnonzero(hit.any(axis=1)):
        j=np.flatnonzero(hit[i])[0];curve=curves[i]
        arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(curve,axis=0),axis=1))]
        end=arc[j]+np.clip(near[i,j],0,1)*(arc[j+1]-arc[j])
        end=max(.00025,end-.0008)
        target=np.linspace(0,min(end,arc[-1]),len(curve))
        for axis in range(3):curves[i,:,axis]=np.interp(target,arc,curve[:,axis])
    return curves


def fit_review_curves(q,head,style,layer):
    """Use measured scalp once, without a second style-volume floor."""
    if not len(q):return q.copy()
    curves=retarget(q,head)
    hit=SCENES[head['fingerprint']].compute_closest_points(o3d.core.Tensor(curves.reshape(-1,3)))
    near=hit['points'].numpy().reshape(curves.shape);normal=hit['primitive_normals'].numpy().reshape(curves.shape)
    normal*=np.where(np.sum(normal*(near-np.asarray(head['centre'])),axis=2)<0,-1,1)[:,:,None]
    emergence=smooth(np.linspace(0,1,q.shape[1])/.13)
    clearance=(.00035 if layer=='long' else .00065)*emergence-.00025*(1-emergence)
    if style['design'].get('smooth_transition'):
        # The fitted short surface is itself up to 1.85 mm above the scalp.
        # A clearance measured only from bare skin buried the rear fibres
        # beneath that surface and exposed a sharp grey "cap" boundary.
        local=(near-np.asarray(head['centre']))/np.asarray(head['radii'])
        p=style['design'];top=smooth((local[:,:,1]-.28)/.52)
        visual=np.clip(p['side']*1000+(min(p['length'],.006)*1000-p['side']*1000)*top,.12,6)
        support=.00015+np.minimum(visual*.00028,.0017)
        gap=.00115 if layer=='long' else .00010
        clearance=(support+gap)*emergence[None,:]-.00025*(1-emergence[None,:])
    distance=np.sum((curves-near)*normal,axis=2)
    push=np.maximum(clearance-distance if clearance.ndim==2 else clearance[None,:]-distance,0);push[:,0]=0
    curves+=normal*push[:,:,None]
    if layer=='medium':
        limit={9:.005,10:.010}.get(style['number'],.040)
        for i in range(len(curves)):
            arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(curves[i],axis=0),axis=1))]
            if arc[-1]<=limit:continue
            target=np.linspace(0,limit,len(arc))
            for axis in range(3):curves[i,:,axis]=np.interp(target,arc,curves[i,:,axis])
    return curves.astype(np.float32)


def medium_curves(style,count=MEDIUM_ROOT_COUNT,steps=MEDIUM_STEPS):
    """Independent shorter guides, following the cut's parting and curl flow.

    Near-bald and buzz cuts keep their intended length: no artificial
    medium growth. Other cuts use roughly 5--35 mm guides between the short
    surface and the existing outer strands.
    """
    if style['design'].get('no_medium'):return np.empty((0,steps,3),np.float32),np.empty(0,np.float32)
    if style['design'].get('review_revision'):
        from salon.review_groom import guides
        count=style['design'].get('medium_count',MEDIUM_ROOT_COUNT) if count==MEDIUM_ROOT_COUNT else count
        return guides(style,count,steps,'medium')
    if style['family']=='buzz':return np.empty((0,steps,3)),np.empty(0)
    middle=medium_profile(style)
    q,_=authored_curves(middle,count=count,steps=steps,minimum_length=.003,medium=True)
    # Width is a bundle cross-section, not an oversized single hair fibre.
    rng=np.random.default_rng(style['number']*331+17)
    width=rng.uniform(.00110,.00200,len(q)).astype(np.float32)
    return q,width


def fitted_medium_curves(style,head,count=MEDIUM_ROOT_COUNT,steps=MEDIUM_STEPS):
    q,width=medium_curves(style,count,steps)
    if not len(q):return q,width
    if style['design'].get('review_revision'):return fit_review_curves(q,head,style,'medium'),width
    curves=retarget(q,head);p=medium_profile(style)['design'];d=unit(q[:,0])
    # Thin locks must have real volume above the fitted scalp. A directional
    # lookup alone can leave parts of a guide buried at triangle boundaries.
    hit=SCENES[head['fingerprint']].compute_closest_points(o3d.core.Tensor(curves.reshape(-1,3)))
    near=hit['points'].numpy().reshape(curves.shape)
    normal=hit['primitive_normals'].numpy().reshape(curves.shape)
    normal*=np.where(np.sum(normal*(near-np.asarray(head['centre'])),axis=2)<0,-1,1)[:,:,None]
    t=np.linspace(0,1,steps);emergence=smooth(t/.18)
    clearance=(.0008+.004*np.sin(np.pi*t)**.8)*emergence-.00025*(1-emergence)
    signed=np.sum((curves-near)*normal,axis=2)
    push=np.maximum(clearance[None,:]-signed,0);push[:,0]=0
    curves+=normal*push[:,:,None]
    side=1-smooth((d[:,1]-.24)/.46);back=smooth((-d[:,2]-.10)/.60);front=smooth((d[:,2]-.30)/.50)
    length=p['length']*(1-side)+p['side']*side
    length=length*(1-back*.68)+p['nape']*back*.68
    length=length*(1-front*.55)+p['fringe']*front*.55
    length=np.clip(length,.0045,.035)
    # Enforce actual arc length on the fitted head. Recipe length alone is
    # insufficient because root tethering, lift and clumping add distance.
    fitted=np.empty_like(curves)
    for i,curve in enumerate(curves):
        arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(curve,axis=0),axis=1))]
        target=np.linspace(0,min(length[i],arc[-1]),steps)
        for axis in range(3):fitted[i,:,axis]=np.interp(target,arc,curve[:,axis])
    width*=np.clip(length/.025,.75,1.)
    # Do not grow roots on the ears, even where a broad side region meets
    # an irregular personal scan. Keep all other roots at their actual scalp.
    mesh=trimesh.load(ROOT/head['bald_file'],process=False)
    ear=np.load(ROOT/head['masks_file'])['ear'].astype(bool)
    ear_points=np.asarray(mesh.vertices)[ear]
    if len(ear_points):
        distance,_=cKDTree(ear_points).query(fitted[:,0])
        keep=distance>.0045
        fitted=fitted[keep];width=width[keep]
    return fitted,width


def medium_mesh(head,style):
    """Tapered elliptical 3-D locks with embedded roots and free tips.

    The old scalp shell is not reused. Every lock has its own curved centre,
    varying cross-section and directional fibre shading attributes.
    """
    from salon.surface import write_surface
    curves,half_width=fitted_medium_curves(style,head)
    if not len(curves):return None
    # Eight rings and four elliptical sides keep the dense layer compact;
    # extra locks improve coverage rather than multiplying tube tessellation.
    n,steps,_=curves.shape;sides=4
    tangent=unit(np.gradient(curves,axis=1))
    radial=unit(curves-np.asarray(head['centre']))
    width_axis=unit(np.cross(tangent,radial))
    degenerate=np.linalg.norm(width_axis,axis=2)<.1
    ref=np.zeros_like(tangent);ref[:,:,2]=1;ref[abs(tangent[:,:,2])>.90]=[1.,0.,0.]
    fallback=unit(np.cross(tangent,ref))
    width_axis[degenerate]=fallback[degenerate]
    depth_axis=unit(np.cross(tangent,width_axis))
    phase=np.arange(sides)*2*np.pi/sides;t=np.linspace(0,1,steps)
    # A bundle has many neighbouring roots, so its basal cross-section must
    # not pinch to a single thin filament. The thick base bridges opposing
    # flows at the crown without a separate scalp shell or painted fallback.
    taper=(1.35-.35*smooth(t/.19))*np.maximum(.012,(1-t)**.65)
    width=half_width[:,None]*taper[None,:]
    thickness=width*(.25+.55*(1-smooth(t/.30)))[None,:]
    # Surface roots start narrow and inside the actual head triangles.
    points=curves[:,:,None]+width[:,:,None,None]*width_axis[:,:,None]*np.cos(phase)[None,None,:,None]+thickness[:,:,None,None]*depth_axis[:,:,None]*np.sin(phase)[None,None,:,None]
    ring=np.arange(n*steps*sides).reshape(n,steps,sides)
    a=ring[:,:-1];b=np.roll(a,-1,2);c=ring[:,1:];d=np.roll(c,-1,2)
    faces=np.concatenate([np.stack([a,b,c],-1).reshape(-1,3),np.stack([b,d,c],-1).reshape(-1,3)])
    mesh=trimesh.Trimesh(points.reshape(-1,3),faces,process=False)
    rng=np.random.default_rng(style['number']*733)
    coverage=(.58+.42*smooth(t/.12))*smooth((1-t)/.14)
    coverage=np.broadcast_to(coverage[None,:,None]*rng.uniform(.86,1.,n)[:,None,None],(n,steps,sides)).reshape(-1)
    lengths=np.sum(np.linalg.norm(np.diff(curves,axis=1),axis=2),axis=1)
    length_attribute=np.broadcast_to(lengths[:,None,None]*1000,(n,steps,sides)).reshape(-1)
    flow=np.broadcast_to(tangent[:,:,None],(n,steps,sides,3)).reshape(-1,3)
    centres=np.broadcast_to(curves[:,:,None],(n,steps,sides,3)).reshape(-1,3)
    target=ROOT/'cache'/f"{head['id']}_{style['id']}_medium.ply"
    write_surface(target,mesh.vertices,mesh.faces,mesh.vertex_normals,np.zeros((len(mesh.vertices),2)),coverage,length_attribute,flow,centres)
    return {'file':target.relative_to(ROOT).as_posix(),'triangles':len(mesh.faces),'locks':n,
            'root_candidates':style['design'].get('medium_count',MEDIUM_ROOT_COUNT),'sampling':'area_stratified_with_crown_boost',
            'length_mm_p10_p50_p90':(np.quantile(lengths,[.1,.5,.9])*1000).round(2).tolist(),
            'structure':'independent_curved_tapered_3d_locks','root_embed_mm':.25}


def tube_mesh(curves,radii,seed=11):
    if not len(curves):return trimesh.Trimesh(np.empty((0,3)),np.empty((0,3),int),process=False)
    rng=np.random.default_rng(seed);n,steps,_=curves.shape
    tangent=unit(np.gradient(curves,axis=1));ref=np.zeros_like(tangent);ref[:,:,2]=1;ref[abs(tangent[:,:,2])>.90]=[1,0,0]
    a=unit(np.cross(tangent,ref));b=np.cross(tangent,a)
    sides=6 if np.median(radii)>.001 else 3
    phase=np.arange(sides)*2*np.pi/sides
    taper=np.maximum(.06,(1-np.linspace(0,1,steps))**.40)
    width=radii[:,None]*taper[None,:]
    v=curves[:,:,None]+width[:,:,None,None]*(a[:,:,None]*np.cos(phase)[None,None,:,None]+b[:,:,None]*np.sin(phase)[None,None,:,None])
    ring=np.arange(n*steps*sides).reshape(n,steps,sides);aa=ring[:,:-1];bb=np.roll(aa,-1,2);cc=ring[:,1:];dd=np.roll(cc,-1,2)
    faces=np.concatenate([np.stack([aa,bb,cc],-1).reshape(-1,3),np.stack([bb,dd,cc],-1).reshape(-1,3)])
    mesh=trimesh.Trimesh(v.reshape(-1,3),faces,process=False)
    tone=rng.uniform(.72,1.18,n)[:,None,None]*np.array([33,30,29])
    colors=np.broadcast_to(tone[:,None],(n,steps,sides,3)).reshape(-1,3)
    mesh.visual.vertex_colors=np.column_stack([np.clip(colors,0,255).astype(np.uint8),np.full(len(v.reshape(-1,3)),255,np.uint8)])
    return mesh


def write_long_ribbons(path,curves,radii,centre,seed,head_radii=None):
    """Individual tapered 3-D fibre strips, with full 28-point trajectories.

    Fine hairs use two-sided strips rather than three-sided tubes so the
    fivefold density does not also require fivefold triangle cost. Thick
    dreadlock fibres still use volumetric tubes.
    """
    n,steps,_=curves.shape;tangent=unit(np.gradient(curves,axis=1))
    radial=unit(curves-np.asarray(centre));axis=unit(np.cross(tangent,radial))
    ref=np.zeros_like(tangent);ref[:,:,2]=1;ref[abs(tangent[:,:,2])>.9]=[1.,0.,0.]
    fallback=unit(np.cross(tangent,ref));bad=np.linalg.norm(axis,axis=2)<.1;axis[bad]=fallback[bad]
    # Keep neighbouring rings oriented consistently, avoiding flipped strips.
    for j in range(1,steps):
        axis[:,j]*=np.where(np.sum(axis[:,j]*axis[:,j-1],axis=1)<0,-1,1)[:,None]
    taper=np.maximum(.06,(1-np.linspace(0,1,steps))**.40)
    offset=axis*(radii[:,None]*taper[None,:])[:,:,None]
    points=np.stack([curves-offset,curves+offset],axis=2).reshape(-1,3)
    normal=unit(np.cross(tangent,axis));normal*=np.where(np.sum(normal*radial,axis=2)<0,-1,1)[:,:,None]
    normal=np.repeat(normal[:,:,None],2,axis=2).reshape(-1,3)
    ring=np.arange(n*steps*2,dtype=np.int32).reshape(n,steps,2)
    a=ring[:,:-1,0];b=ring[:,:-1,1];c=ring[:,1:,0];d=ring[:,1:,1]
    faces=np.concatenate([np.stack([a,b,c],-1).reshape(-1,3),np.stack([b,d,c],-1).reshape(-1,3)])
    rng=np.random.default_rng(seed)
    root=unit((curves[:,0]-np.asarray(centre))/(np.asarray(head_radii) if head_radii is not None else 1.))
    theta=np.arctan2(root[:,0],root[:,2]);bundle=np.floor((theta+np.pi)/.12)+np.floor(root[:,1]/.085)*64
    shade=.65+.90*np.mod(np.sin((bundle+seed*79)*12.9898)*43758.5453,1.)
    tone=(shade*rng.uniform(.90,1.10,n))[:,None]*np.array([36,33,31])
    colours=np.repeat(np.clip(tone,0,255).astype(np.uint8),steps*2,axis=0)
    vertex=np.empty(len(points),dtype=[(k,'<f4') for k in ['x','y','z','nx','ny','nz']]+[(k,'u1') for k in ['red','green','blue','alpha']])
    for i,k in enumerate(['x','y','z']):vertex[k]=points[:,i]
    for i,k in enumerate(['nx','ny','nz']):vertex[k]=normal[:,i]
    for i,k in enumerate(['red','green','blue']):vertex[k]=colours[:,i]
    vertex['alpha']=255
    triangles=np.empty(len(faces),dtype=[('count','u1'),('ids','<i4',(3,))]);triangles['count']=3;triangles['ids']=faces
    header=['ply','format binary_little_endian 1.0',f'element vertex {len(vertex)}']
    header +=[f'property float {k}' for k in ['x','y','z','nx','ny','nz']]+[f'property uchar {k}' for k in ['red','green','blue','alpha']]
    header +=[f'element face {len(faces)}','property list uchar int vertex_indices','end_header','']
    temporary=path.with_suffix('.tmp')
    with temporary.open('wb') as f:
        f.write('\n'.join(header).encode('ascii'));f.write(vertex.tobytes());f.write(triangles.tobytes())
    temporary.replace(path)
    return len(faces)


def surface_mesh(head,style,volume=False):
    from salon.surface import write_surface
    mesh=trimesh.load(ROOT/head['bald_file'],process=False)
    if head['is_standard']:
        base=mesh
        v,f=trimesh.remesh.subdivide_to_size(base.vertices,base.faces,max_edge=.004,max_iter=6)
        mesh=trimesh.Trimesh(v,f,process=False)
        hit=scene_for(base).compute_closest_points(o3d.core.Tensor(v.astype(np.float32)))
        bc=hit['primitive_uvs'].numpy();norm=np.asarray(base.vertex_normals)[base.faces[hit['primitive_ids'].numpy()]]
        surface_normals=unit(norm[:,0]*(1-bc.sum(axis=1))[:,None]+norm[:,1]*bc[:,:1]+norm[:,2]*bc[:,1:])
    else:surface_normals=np.asarray(mesh.vertex_normals).copy()
    vertices=np.asarray(mesh.vertices);faces=np.asarray(mesh.faces)
    c=np.asarray(head['centre']);r=np.asarray(head['radii']);q=(vertices-c)/r
    theta=np.arctan2(q[:,0],q[:,2]);co=np.cos(theta)
    floor=-.30+.68*np.maximum(co,0)**1.4-.44*np.maximum(-co,0)**.65
    coverage=smooth((q[:,1]-floor)/.055)
    distance=ear_surface_distance(head,vertices)
    coverage*=smooth((distance-.00075)/.00225)
    p=style['design'];top=smooth((q[:,1]-.28)/.52)
    visual=p['side']*1000+(min(p['length'],.006)*1000-p['side']*1000)*top
    visual=np.clip(visual,.12,6)
    n=surface_normals;n*=np.where(np.sum(n*(vertices-c),axis=1)<0,-1,1)[:,None]
    height=(.00015+np.minimum(visual*.00028,.0017))*coverage
    if volume:
        coverage*=top
        # Styling-specific inner support, below outer fibres, with real
        # vertices rather than a flat texture filling visible scalp gaps.
        height+=p['volume']*.42*top
        visual=np.full(len(vertices),14.)
    points=vertices+n*height[:,None]
    # Do not leave a skin-coloured duplicate shell on the ear itself.
    used=faces[(coverage[faces].max(axis=1)>.001)&(distance[faces].min(axis=1)>.00060)]
    uv=np.asarray(mesh.visual.uv) if getattr(mesh.visual,'uv',None) is not None else np.zeros((len(vertices),2))
    flow=np.tile([0.,0.,-1.],(len(vertices),1))
    if style['family'] not in ('back','sideback'):flow[:,2]=1
    destination=ROOT/'cache'/f"{head['id']}_{style['id']}_{'volume' if volume else 'short'}.ply"
    write_surface(destination,points,used,n,uv,coverage,visual,flow,vertices)
    return destination,len(used)


def style_assets(head,style):
    cache=ROOT/'cache';cache.mkdir(exist_ok=True)
    manifest=cache/f"{head['id']}_{style['id']}.json"
    identity=hashlib.sha256(json.dumps({'v':VERSION,'head':head['fingerprint'],'style':style},sort_keys=True).encode()).hexdigest()
    if manifest.exists():
        old=json.loads(manifest.read_text(encoding='utf-8'))
        if old.get('identity')==identity and all((ROOT/x['file']).exists() for x in old['layers'].values()):return old
    q,width=long_curves(style)
    curves=fitted_long_curves(q,head,style)
    layers={}
    short,faces=surface_mesh(head,style)
    layers['short']={'file':short.relative_to(ROOT).as_posix(),'triangles':faces}
    medium=medium_mesh(head,style)
    if medium:layers['medium']=medium
    if len(curves):
        file=cache/f"{head['id']}_{style['id']}_long.ply"
        if style['family']=='locs':
            hair=tube_mesh(curves,width,style['number']);hair.export(file);triangles=len(hair.faces)
        else:triangles=write_long_ribbons(file,curves,width,head['centre'],style['number'],head['radii'])
        layers['long']={'file':file.relative_to(ROOT).as_posix(),'triangles':triangles,'strands':len(curves),
                        'previous_strands':len(curves)//LONG_DENSITY if not style['design'].get('review_revision') else None,'density_multiplier':LONG_DENSITY if not style['design'].get('review_revision') else None,
                        'density_policy':'per_style_after_user_review' if style['design'].get('review_revision') else 'fivefold',
                        'root_sampling':'braid_internal_fibres' if style['family']=='locs' else 'area_stratified_with_crown_boost',
                        'structure':'volumetric_braid_fibres' if style['family']=='locs' else 'individual_curved_tapered_fibre_strips'}
    result={'identity':identity,'head':head['id'],'style':style['id'],'layers':layers,
            'layer_schema':'short_medium_long_v2','layer_order':['short','medium','long'],
            'revision_summary':style.get('revision_summary',''),'transition_revision':style.get('transition_revision',0),'ear_cleanup_revision':1,
            'inactive_layers':({'medium':'本款按当前选择只保留短发底层','long':'本款无需长发层'} if style['design'].get('no_medium') else {'long':'本款按寸头长度使用短层与中层，无需长发层'}) if style['family']=='buzz' else {},
            'roughness':style['design'].get('roughness',.72),'status':'procedural_reference_draft'}
    manifest.write_text(json.dumps(result,ensure_ascii=False),encoding='utf-8')
    return result
