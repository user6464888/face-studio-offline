"""Constrained bare-scalp estimate; hidden skull dimensions are not measured."""
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scalp_fit import smoothstep


DEFAULT_REDUCTION_MM={'crown':14.,'each_side':6.,'back':10.}


def fit_bare_scalp(source,baseline,weights,protected,c,skin_weights=None):
    original=np.asarray(source.vertices);hair=weights>0
    rgb=np.asarray(source.visual.vertex_colors)[:,:3].astype(float)
    x=original[:,0]-c['centerX'];y=original[:,1];z=original[:,2]-c['centerZ']
    measured_skin=(~hair) if skin_weights is None else np.asarray(skin_weights)==0
    clean=measured_skin&(rgb.mean(axis=1)>60)
    forehead=clean&(abs(x)<c['radiusX']*.58)&(y>c['frontLine']-.040)&(y<c['frontLine']-.014)&(z>.05)
    temple=clean&(abs(x)>c['radiusX']*.68)&(y>c['sideLine']+.010)&(y<c['frontLine']-.020)&(z>-.015)&(z<.075)
    normals=np.asarray(source.vertex_normals)
    forehead&=normals[:,2]>.45
    temple&=(abs(normals[:,0])>.45)&~protected
    groups=[np.flatnonzero(forehead),np.flatnonzero(temple)]
    # Cap the sample count so denser forehead sampling cannot determine the
    # hidden crown/back without the explicit, bounded shape prior.
    samples=[]
    for ids in groups:
        if len(ids):samples.append(original[ids[np.linspace(0,len(ids)-1,min(160,len(ids))).astype(int)]])
    if not samples:raise RuntimeError('缺少额头或太阳穴的可靠皮肤，不能估计光头轮廓。')
    anchors=np.vstack(samples)
    region=hair&(baseline[:,1]>c['sideLine']+.012)
    low,high=np.quantile(baseline[region],[.005,.995],axis=0)
    cx=(low[0]+high[0])/2;cy=c['centerY'];cz=c['centerZ']
    rx=max(.055,(high[0]-low[0])/2-DEFAULT_REDUCTION_MM['each_side']/1000)
    ry=max(.075,high[1]-cy-DEFAULT_REDUCTION_MM['crown']/1000)
    rb=max(.070,cz-low[2]-DEFAULT_REDUCTION_MM['back']/1000)
    front=original[forehead]
    if len(front):
        denominator=np.sqrt(np.maximum(.25,1-((front[:,0]-cx)/rx)**2-((front[:,1]-cy)/ry)**2))
        rf=float(np.median((front[:,2]-cz)/denominator))
    else:rf=c['radiusZ']
    prior=np.array([cx,cy,cz,rx,ry,np.clip(rf,.075,.13),rb])
    limits=np.array([.002,.003,.003,.0025,.004,.005,.003])
    def residual(parameters):
        centre=parameters[:3];radius=np.tile(parameters[3:6],(len(anchors),1))
        radius[:,2]=np.where(anchors[:,2]>=centre[2],parameters[5],parameters[6])
        rho=np.linalg.norm((anchors-centre)/radius,axis=1)
        observed=(rho-1)*.085/.002
        regularity=(parameters-prior)/limits*np.sqrt(len(anchors))*.55
        return np.r_[observed,regularity]
    fit=least_squares(residual,prior,bounds=(prior-limits,prior+limits),loss='soft_l1',f_scale=1.5,max_nfev=120)
    centre=fit.x[:3]
    radii=np.tile(fit.x[3:6],(len(baseline),1))
    radii[:,2]=np.where(baseline[:,2]>=centre[2],fit.x[5],fit.x[6])
    direction=baseline-centre
    rho=np.maximum(np.linalg.norm(direction/radii,axis=1),1e-8)
    # Radial projection is inward-only: a fitted scalp cannot grow outside the
    # previous hair-envelope estimate. Ellipsoid fitting removes hair spikes.
    # Retain a small radial residual so originally folded scan triangles do
    # not all collapse onto exactly the same analytic surface.
    target=centre+direction*(.02+.98/np.maximum(rho,1))[:,None]
    edges=source.edges_unique
    ring=np.unique(edges[hair[edges[:,0]]!=hair[edges[:,1]]])
    distance,_=cKDTree(original[ring]).query(original,workers=4)
    blend=smoothstep(distance/.022)*hair
    points=baseline+(target-baseline)*blend[:,None]
    points[~hair]=original[~hair]
    points[protected]=original[protected]
    # Report dimensions on the same original upper-scalp vertex set. Including
    # the fixed ears would mask a real scalp-width reduction.
    upper=hair&(baseline[:,1]>c['frontLine'])
    def dimensions(v):
        lo,hi=np.quantile(v[upper],[.005,.995],axis=0)
        return {'upper_scalp_width_mm':float((hi[0]-lo[0])*1000),
                'crown_y_mm':float(hi[1]*1000),'rear_z_mm':float(lo[2]*1000)}
    before=dimensions(baseline);after=dimensions(points)
    delta=np.linalg.norm(points-baseline,axis=1)*1000
    stats={'method':'bounded asymmetric ellipsoid fit to exposed forehead/temples, inward-only projection with fixed scan boundary',
           'estimated_not_measured':True,'default_reductions_mm':DEFAULT_REDUCTION_MM,
           'forehead_anchor_vertices':int(forehead.sum()),'temple_anchor_vertices':int(temple.sum()),
           'fit_success':bool(fit.success),'fit_centre_m':centre.tolist(),
           'fit_radii_m':{'side':float(fit.x[3]),'crown':float(fit.x[4]),'front':float(fit.x[5]),'back':float(fit.x[6])},
           'before':before,'after':after,
           'change_mm':{'upper_scalp_width_reduction':before['upper_scalp_width_mm']-after['upper_scalp_width_mm'],
                        'crown_lowering':before['crown_y_mm']-after['crown_y_mm'],
                        'rear_retraction':after['rear_z_mm']-before['rear_z_mm']},
           'maximum_vertex_movement_mm':float(delta.max()),'face_ears_neck_positions_exact':bool(np.array_equal(points[~hair],original[~hair])),
           'baseline_asset':'head_bald_before_shape.ply','corrected_asset':'head_bald.ply'}
    return points,stats
