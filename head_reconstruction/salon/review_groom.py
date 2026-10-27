"""Style-dependent growth fields for the first user inspection revision.

All coordinates are canonical head units (0.095 m per unit). Medium and
outer fibres share the same parting, directional field and bundle phases.
Curl is added after routing, never averaged back into flat shared guides.
"""
import numpy as np
from salon.math_utils import unit


def smooth(x):
    x=np.clip(x,0,1);return x*x*(3-2*x)


def hash01(x):return np.mod(np.sin(x*12.9898+4.1414)*43758.5453,1.)


def transition_noise(d,seed):
    """Continuous length variation; no steps at hashed bundle boundaries."""
    theta=np.arctan2(d[:,0],d[:,2]);y=d[:,1]
    return .5+.27*np.sin(theta*7+seed*.73+y*9)+.14*np.sin(theta*13-y*17+seed)


def front_weight(d):
    return smooth((d[:,2]+.05)/.85)*smooth((d[:,1]+.08)/.42)


def transition_height(d,p,layer):
    """One spatial envelope spanning the crown, parietal ridge and occiput.

    The rear remains close to the scalp, but no per-strand rear clamp cuts
    the forelock volume off at one meridian. Both layers use this field.
    """
    rear=smooth((.65-d[:,2])/1.40)
    crown=smooth((d[:,1]+.32)/1.28)
    rear_height=.0048 if p['flow'] in ('back','sideback') else .0065
    height=(p['volume']*(1-rear)+rear_height*rear)*crown
    if layer=='medium':height*=.27
    return height


def root_samples(count):
    n=max(1,int(count*.85));i=np.arange(n);gold=np.pi*(3-np.sqrt(5))
    theta=(i*gold+.37)%(2*np.pi)-np.pi;co=np.cos(theta)
    floor=-.22+.60*np.maximum(co,0)**1.4-.45*np.maximum(-co,0)**.65
    y=floor+.015+(1-floor-.017)*(i+.5)/n
    k=np.arange(count-n);theta=np.r_[theta,(k*gold+1.17)%(2*np.pi)-np.pi]
    y=np.r_[y,.81+.188*(k+.5)/max(1,len(k))]
    r=np.sqrt(np.maximum(0,1-y*y))
    return np.column_stack([r*np.sin(theta),y,r*np.cos(theta)])


def regional_lengths(d,p,layer):
    top=smooth((d[:,1]-.10)/.62)
    front=smooth((d[:,2]-.30)/.45)*smooth((d[:,1]-.02)/.35)
    back=smooth((-d[:,2]-.04)/.60)
    length=p['side']+(p['length']-p['side'])*top
    length=length*(1-front*.80)+p['fringe']*front*.80
    if p['flow'] in ('mullet','radial_mullet','front_part'):
        inner=1-smooth((d[:,1]+.05)/.70)
        rear=p.get('outer_back',p['length']*.67)*(1-inner)+p['nape']*inner
    else:
        rear=p['nape']+(p['length']*.70-p['nape'])*top
    length=length*(1-back*.92)+rear*back*.92
    if p.get('smooth_transition'):
        top=smooth((d[:,1]+.04)/.95)
        front=front_weight(d)
        back=smooth((.12-d[:,2])/.98)
        length=p['side']+(p['length']-p['side'])*top
        length=length*(1-front*.80)+p['fringe']*front*.80
        # Crown roots retain overlapping length as they turn towards the
        # back; only lower roots gradually acquire the nape length.
        inner=1-smooth((d[:,1]+.15)/1.05)
        rear=p['length']*(1-inner)+p['nape']*inner
        length=length*(1-back)+rear*back
    if layer=='medium':
        length=np.minimum(length*p.get('medium_factor',.65),.035)
        if p['flow']=='locs':length=.0012+(.011-.0012)*top
    return length


def guides(style,count,steps,layer='long',roots=None):
    p=style['design'];flow_type=p['flow']
    if layer=='medium' and p.get('no_medium'):return np.empty((0,steps,3),np.float32),np.empty(0,np.float32)
    if layer=='long' and p.get('no_long'):return np.empty((0,steps,3),np.float32),np.empty(0,np.float32)
    if roots is not None:
        d=unit(np.asarray(roots,dtype=float))
        theta=np.arctan2(d[:,0],d[:,2]);bundle=np.floor((theta+np.pi)/.12)+np.floor(d[:,1]/.085)*64
        random=transition_noise(d,style['number']) if p.get('smooth_transition') else hash01(bundle+style['number']*79)
        length=regional_lengths(d,p,layer)*(1+p.get('broken',.15)*(random*2-1))
        return trace_routes(style,d,length,bundle,random,steps,layer)
    candidate_count=max(512,int(count*1.35));d=None
    for _ in range(5):
        raw=root_samples(candidate_count);theta=np.arctan2(raw[:,0],raw[:,2])
        bundle=np.floor((theta+np.pi)/.12)+np.floor(raw[:,1]/.085)*64
        random=hash01(bundle+style['number']*79)
        if p.get('smooth_transition'):random=transition_noise(raw,style['number'])
        length=regional_lengths(raw,p,layer)
        variation=p.get('broken',.15)
        factor=1+variation*(random*2-1)
        if not p.get('smooth_transition'):factor+=.045*(hash01(theta*113+raw[:,1]*337)-.5)
        length*=factor
        ear=(abs(raw[:,0])>.88)&(raw[:,1]<.06)&(abs(raw[:,2])<.38)
        minimum=.00035 if layer=='medium' else .0015
        density=smooth((length-minimum)/(minimum*2.0))
        back=smooth((-raw[:,2]-.05)/.60)
        if layer=='long':density*=1-back*(1-p.get('back_density',1.))
        if layer=='medium' and p.get('medium_top_only'):density*=smooth((raw[:,1]-.22)/.35)
        eligible=(length>minimum)&~ear&(hash01(theta*229+raw[:,1]*721)<density)
        ids=np.flatnonzero(eligible)
        if len(ids)>=count:
            ids=ids[np.linspace(0,len(ids)-1,count).round().astype(int)]
            d=raw[ids];length=length[ids];bundle=bundle[ids];random=random[ids];break
        candidate_count=int(candidate_count*count/max(1,len(ids))*1.08)
    if d is None:raise RuntimeError('发型区域取样不足')
    return trace_routes(style,d,length,bundle,random,steps,layer)


def trace_routes(style,d,length,bundle,random,steps,layer):
    p=style['design'];flow_type=p['flow'];continuous=bool(p.get('smooth_transition'))
    n=len(d);q=np.empty((n,steps,3));on=d.copy();q[:,0]=d
    theta=np.arctan2(d[:,0],d[:,2])
    phase=hash01(bundle*.91+style['number'])*2*np.pi
    if continuous:phase=2*np.pi*(np.sin(theta)*2.8+d[:,1]*2.1+style['number']*.17)
    sign=np.where(d[:,0]>=p.get('part',0),1.,-1.)
    frontal=(d[:,2]>.36)&(d[:,1]>.20)
    frontal_blend=front_weight(d) if continuous else frontal.astype(float)
    broken=p.get('broken',.15)
    fringe_floor=.13+(random-.5)*(.04+broken*.48)
    if layer=='medium' and flow_type in ('curl','foil','fringe','mullet','radial_mullet'):
        fringe_floor+=.12
    # Different rear root heights create proper under-long/over-short tails.
    rear_root=smooth((-d[:,2]-.02)/.55)
    top_root=smooth((d[:,1]-.20)/.60)
    volume=p['volume']*(.75+.40*random)
    if layer=='medium':volume=np.minimum(volume*.38,.006)
    if flow_type=='buzz':volume=np.zeros(n)
    if flow_type in ('back','sideback'):
        volume=volume*(1-rear_root)+p.get('rear_lift',.002)*rear_root
    elif flow_type not in ('spike','foil','curl'):
        volume*=1-rear_root*.65
    volume*=.4+.6*top_root
    release=np.zeros(n,bool)
    release_weight=np.zeros(n)
    for j in range(1,steps):
        t=j/(steps-1)
        # Most crown and side regions grow radially away from one whorl.
        whorl=np.array([.08,.94,-.29])
        away=on-whorl
        flow=away-on*np.sum(away*on,axis=1)[:,None]
        flow[:,1]-=.13
        rear=smooth((-on[:,2]-.04)/.38)
        if continuous:rear=smooth((.35-on[:,2])/1.10)
        down=np.column_stack([.08*np.sin(theta),-np.ones(n),-np.full(n,.12)])
        flow=flow*(1-rear[:,None])+down*rear[:,None]
        if flow_type in ('fringe','crop','caesar','foil','curl','mullet'):
            front_flow=np.column_stack([.10*np.sin(phase),-np.full(n,.36),np.ones(n)])
            w=smooth((on[:,2]+.02)/.45)
            if continuous:w=smooth((on[:,2]+.30)/1.15)
            flow=flow*(1-w[:,None])+front_flow*w[:,None]
        if flow_type=='front_part':
            swoop=np.column_stack([sign*(.95-.30*t),np.full(n,-.08-.65*t),np.full(n,.32*(1-t))])
            flow=flow*(1-frontal_blend[:,None])+swoop*frontal_blend[:,None]
        elif flow_type in ('back','sideback'):
            sweep=sign
            if continuous:sweep=np.full(n,1. if p.get('part',.42)>=0 else -1.) if flow_type=='sideback' else np.tanh(d[:,0]/.15)
            swept=np.column_stack([sweep*(.62 if flow_type=='sideback' else .04),np.full(n,.10),-np.ones(n)])
            flow=swept*(1-rear[:,None])+down*rear[:,None]
        elif flow_type=='spike':
            front_flow=np.column_stack([.08*np.sin(phase),np.full(n,.25),np.ones(n)])
            flow=front_flow*(1-rear[:,None])+down*rear[:,None]
        # A coherent, small per-bundle lateral variation creates texture,
        # without attracting all neighbouring tips to one shared polyline.
        flow[:,0]+=(random-.5)*p.get('clump',.22)*(1-rear)
        tangent=unit(flow-on*np.sum(flow*on,axis=1)[:,None])
        candidate=unit(on+tangent*(length/(.095*(steps-1)))[:,None])
        central=(candidate[:,2]>.44)&(abs(candidate[:,0])<.86)
        floor=fringe_floor.copy()
        if flow_type in ('back','sideback'):
            angle=np.arctan2(candidate[:,0],candidate[:,2])
            floor=-.22+.60*np.maximum(np.cos(angle),0)**1.4-.025
        need=central&(candidate[:,1]<floor)
        if continuous:
            # A hard rectangular forehead guard also cuts a visible step
            # into the temple. Fade its influence across the whole corner.
            guard=smooth((candidate[:,2]-.22)/.40)*(1-smooth((abs(candidate[:,0])-.72)/.32))
            yy=candidate[:,1]+np.maximum(floor-candidate[:,1],0)*guard
            angle=np.arctan2(candidate[:,0],candidate[:,2]);xz=np.sqrt(np.maximum(0,1-yy*yy))
            candidate=np.column_stack([xz*np.sin(angle),yy,xz*np.cos(angle)])
        elif need.any():
            angle=np.arctan2(candidate[need,0],candidate[need,2]);yy=floor[need]
            xz=np.sqrt(1-yy*yy);candidate[need]=np.column_stack([xz*np.sin(angle),yy,xz*np.cos(angle)])
        on=candidate
        lift=volume*np.sin(np.pi*t)**.75*smooth(t/.16)
        if continuous:lift=transition_height(on,p,layer)*smooth(t/.15)*(1-.18*t**4)
        if flow_type=='front_part':lift+=p.get('fringe_arc',.012)*np.sin(np.pi*t)*frontal_blend*(.4 if layer=='medium' else 1.)
        if flow_type in ('back','sideback') and not continuous:
            actual_rear=smooth((-on[:,2]-.05)/.40)
            lift=lift*(1-actual_rear)+np.minimum(lift,.002 if layer=='long' else .001)*actual_rear
        q[:,j]=on*(1+lift/.095)[:,None]
        if flow_type=='spike' and not continuous:
            emergence=smooth(t/.16)*t**.7
            strength=(.55 if layer=='medium' else 1.)*top_root*emergence
            q[:,j,1]+=.025/.095*strength
            q[:,j,2]+=.017/.095*strength
        if flow_type in ('mullet','radial_mullet','front_part'):
            release|=(on[:,1]<-.58)&(on[:,2]<-.32)&(rear_root>.25)
            # Once below the occiput, continue down rather than wrapping
            # under the skull and back up through the neck or face.
            if continuous and j>1:
                blend=smooth((-.42-on[:,1])/.32)*smooth((-on[:,2]-.20)/.25)*rear_root
                release_weight=np.maximum(release_weight,blend)
                down=q[:,j-1]+np.array([0.,-1.,-.10])*length[:,None]/(.095*(steps-1))
                q[:,j]=q[:,j]*(1-release_weight[:,None])+down*release_weight[:,None]
            elif j>1 and release.any():
                q[release,j]=q[release,j-1]+np.array([0.,-1.,-.10])*length[release,None]/(.095*(steps-1))
    amplitude=p.get('curl',0)
    if layer=='medium':amplitude=min(amplitude*.32,.0025)
    if flow_type in ('curl','foil'):
        # Gather the uncurled routes first. Averaging already-curled strands
        # destroys their coils; leaving routes fully separate makes a solid
        # fuzzy shell instead of recognisable curl locks.
        _,group=np.unique(bundle,return_inverse=True)
        guide=np.zeros((group.max()+1,steps,3));np.add.at(guide,group,q)
        guide/=np.bincount(group)[:,None,None]
        t=np.linspace(0,1,steps)
        gather=(.80 if layer=='long' else .40)*smooth(t/.26)
        q=q*(1-gather[None,:,None])+guide[group]*gather[None,:,None]
    if flow_type in ('back','sideback'):curl_region=1-rear_root*.96
    else:curl_region=np.ones(n)
    if amplitude:
        t=np.linspace(0,1,steps)[None,:];arc=length[:,None]*t
        waves=2*np.pi*arc/p.get('wave',.055)+phase[:,None]
        frame=guide[group] if flow_type in ('curl','foil') else q
        tangent=unit(np.gradient(frame,axis=1));normal=unit(frame)
        side=unit(np.cross(tangent,normal));out=unit(np.cross(side,tangent))
        out*=np.where(np.sum(out*normal,axis=2)<0,-1,1)[:,:,None]
        if continuous:
            # Frenet frames flip where a guard makes the route locally
            # stationary. Use a continuous scalp frame for these small waves.
            side=np.broadcast_to(np.column_stack([np.cos(theta),np.zeros(n),-np.sin(theta)])[:,None],q.shape)
            out=normal
        env=smooth(t/.16)
        # Shared phases retain visible coils; they are not averaged away.
        transverse=np.sin(waves)-np.sin(phase[:,None])
        outward=.55*(1-np.cos(waves-phase[:,None]))
        q+=(side*transverse[:,:,None]+out*outward[:,:,None])*(amplitude/.095)*env[:,:,None]*curl_region[:,None,None]
    if flow_type=='buzz':
        q=d[:,None]*(1+length[:,None]*np.linspace(0,1,steps)[None,:]/.095)[:,:,None]
    if flow_type=='spike':
        t=np.linspace(0,1,steps)[None,:]
        direction=unit(np.column_stack([.10*np.sin(phase),np.full(n,.82),np.ones(n)]))
        extruded=d[:,None]+direction[:,None]*(length[:,None]*t/.095)[:,:,None]
        # Front spikes leave the scalp forwards and upwards. A tangent-only
        # spherical walk would comb them down the forehead instead.
        weight=smooth((d[:,1]-.16)/.30)*(1-smooth((-d[:,2]-.02)/.30))
        if continuous:weight=smooth((d[:,1]+.05)/.85)*smooth((d[:,2]+.55)/1.25)
        q=q*(1-weight[:,None,None])+extruded*weight[:,None,None]
    q[:,0]=d*(1-.00025/.095)
    rng=np.random.default_rng(style['number']*1907+(0 if layer=='long' else 1))
    if layer=='medium':
        width=rng.uniform(.0010,.0016,n)*p.get('medium_width',1.)
        width*=np.clip(length/.015,.3,1.)
    else:width=rng.uniform(.00016,.00023,n)
    return q.astype(np.float32),width.astype(np.float32)
