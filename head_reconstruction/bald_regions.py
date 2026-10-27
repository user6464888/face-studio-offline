"""Separate complete measured pinnae from adjacent sideburns for bald cleanup."""
import numpy as np
from scipy.spatial import cKDTree,ConvexHull,QhullError
from scalp_fit import smoothstep


def bald_regions(source,c):
    v=np.asarray(source.vertices);rgb=np.asarray(source.visual.vertex_colors)[:,:3].astype(float)
    x=abs(v[:,0]-c['centerX']);y=v[:,1]-c['sideLine'];z=v[:,2]-c['centerZ']
    # The upper helix slopes backwards. The old upright, narrow ellipse missed
    # most of its posterior arc, but also kept the dark sideburn in front.
    vertical=y+.033
    envelope=(x>c['radiusX']*.65)&(((z-(.001-.38*vertical))/.041)**2+(vertical/.040)**2<1)
    ear=np.zeros(len(v),bool)
    for sign in (-1,1):
        side=sign*(v[:,0]-c['centerX'])>0
        warm=envelope&side&(x>c['radiusX']*.70)&(z<.028)&(y>-.067)
        warm&=(rgb[:,0]-rgb[:,2]>22)&(rgb.mean(axis=1)>45)
        if warm.sum()<30:
            ear|=envelope&side
            continue
        # Fit each pinna to its own lateral skin silhouette. A fixed ellipse
        # alone preserves a ring of captured hair around the real helix.
        lateral=np.quantile(x[warm],.55)
        seeds=warm&(x>=lateral)
        try:hull=ConvexHull(np.column_stack([z[seeds],y[seeds]]))
        except QhullError:
            ear|=envelope&side
            continue
        yz=np.column_stack([z,y])
        inside=(yz@hull.equations[:,:2].T+hull.equations[:,2]<=.001).all(axis=1)
        ear|=envelope&side&inside
    ear_support=np.zeros(len(v),bool)
    ear_support[np.unique(source.faces[ear[source.faces].any(axis=1)])]=True
    temple=(x>c['radiusX']*.67)&(z>.026)&(z<.070)&(y>-.050)&(y<.038)&~ear_support
    dark=temple&(rgb.mean(axis=1)<80)&((rgb[:,0]-rgb[:,2])<32)
    sideburn=dark.copy()
    if dark.any():
        distance,_=cKDTree(v[dark]).query(v,workers=4)
        sideburn|=temple&(distance<.004)
    # Keep the existing central face protection, replacing its old ear mask
    # with the independently measured ear silhouette.
    face=(x<c['radiusX']*.82)&(z>.025)&(v[:,1]<c['frontLine']-.044)
    protected=(face|ear_support)&~sideburn
    return protected,ear,sideburn


def repair_ear_rim_colours(source,c,ear,hair_weights):
    """Remove baked dark hair around the helix; never smooth ear geometry.

    Only the narrow posterior/superior silhouette band is eligible. Deep ear
    cavities and the warm photographed cartilage remain the source colours.
    """
    v=np.asarray(source.vertices);rgb=np.asarray(source.visual.vertex_colors)[:,:3].astype(float)
    x=v[:,0]-c['centerX'];y=v[:,1]-c['sideLine'];z=v[:,2]-c['centerZ']
    value=rgb.mean(axis=1);warmth=rgb[:,0]-rgb[:,2]
    support=np.zeros(len(v),bool);support[np.unique(source.faces[ear[source.faces].any(axis=1)])]=True
    alpha=np.zeros(len(v));corrected=rgb.copy()
    linear=rgb/255;linear=np.where(linear<=.04045,linear/12.92,((linear+.055)/1.055)**2.4)
    for sign in (-1,1):
        side=sign*x>0
        if (ear&side).sum()<30:continue
        try:hull=ConvexHull(np.column_stack([z[ear&side],y[ear&side]]))
        except QhullError:continue
        signed=(np.column_stack([z,y])@hull.equations[:,:2].T+hull.equations[:,2]).max(axis=1)
        band=smoothstep((signed+.008)/.005)*(1-smoothstep((signed-.003)/.009))
        outer_arc=np.maximum(smoothstep((.005-z)/.016),smoothstep((y+.020)/.012))
        dark=smoothstep((82-value)/25)*smoothstep((34-warmth)/17)
        strength=band*outer_arc*dark*support*side
        strength=np.rint(strength*255)/255
        ids=np.flatnonzero(strength>0)
        donors=side&(abs(x)>c['radiusX']*.50)&(z<.060)&(z>-.055)&(y>-.095)&(y<.020)
        donors&=(value>60)&(warmth>24)&(np.asarray(hair_weights)==0)&(strength==0)
        if not len(ids) or donors.sum()<16:continue
        donor_ids=np.flatnonzero(donors)
        distance,near=cKDTree(v[donor_ids]).query(v[ids],k=min(48,len(donor_ids)),workers=4)
        blend=np.exp(-(distance-distance[:,:1])/.008);blend/=blend.sum(axis=1,keepdims=True)
        skin=(linear[donor_ids[near]]*blend[:,:,None]).sum(axis=1)
        result=linear[ids]*(1-strength[ids,None])+skin*strength[ids,None]
        srgb=np.where(result<=.0031308,result*12.92,1.055*np.maximum(result,0)**(1/2.4)-.055)
        corrected[ids]=np.clip(np.rint(srgb*255),0,255)
        alpha[ids]=strength[ids]
    return corrected,(alpha*255).astype(np.uint8)
