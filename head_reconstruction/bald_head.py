"""Estimate a bare scalp, including the mixed-colour captured hairline."""
import numpy as np
import trimesh
from scipy.sparse import coo_matrix
from scipy.spatial import cKDTree
from scalp_fit import extend_skin_colours,smoothstep
from bald_shape import fit_bare_scalp


def build_bald_head(source,scalp,weights,protected,c,sideburn=None,ear=None):
    original=np.asarray(source.vertices)
    hair=weights>0
    edges=source.edges_unique
    boundary=np.unique(edges[hair[edges[:,0]]!=hair[edges[:,1]]])
    distance,_=cKDTree(original[boundary]).query(original,workers=4)
    points=np.asarray(scalp.vertices).copy()
    paired=np.vstack([edges,edges[:,::-1]])
    degree=np.bincount(paired[:,0],minlength=len(points))
    average=coo_matrix((1/np.maximum(degree[paired[:,0]],1),(paired[:,0],paired[:,1])),shape=(len(points),len(points))).tocsr()
    freedom=smoothstep(distance/.014)*hair
    for _ in range(100):points+=(average@points-points)*(.4*freedom[:,None])
    points[~hair]=original[~hair]
    baseline=points.copy()
    # Some estimated crown pixels have a skin-like RGB tint and were excluded
    # by hair colour segmentation. They are still hidden scalp, not measured
    # exposed skin; otherwise those islands would pin the old hair height.
    to_original_hair,_=cKDTree(original[hair]).query(original,workers=4)
    forehead_band=(to_original_hair<.010)&(original[:,1]>c['frontLine']-.028)&~protected
    # Short hairs behind the pinna are often mixed with warm skin pixels and
    # fall below the angular hairline. Grow only near connected source hair,
    # on its posterior side; the measured ear recesses remain protected.
    rgb=np.asarray(source.visual.vertex_colors)[:,:3].astype(float)
    x=original[:,0]-c['centerX'];y=original[:,1];z=original[:,2]-c['centerZ']
    posterior=(z<.010)&(abs(x)>c['radiusX']*.50)
    posterior&=(y>c['sideLine']-.085)&(y<c['sideLine']+.040)&~protected
    mixed_hair=(rgb.mean(axis=1)<90)&((rgb[:,0]-rgb[:,2])<40)
    ear_cleanup=posterior&(to_original_hair<.020)&(mixed_hair|(to_original_hair<.006))
    if sideburn is None:sideburn=np.zeros(len(original),bool)
    hair=(hair|((original[:,1]>c['frontLine']+.015)&~protected)|forehead_band|ear_cleanup|sideburn)&~protected
    shape_weights=hair.astype(np.uint8)*255
    points,shape_stats=fit_bare_scalp(source,baseline,shape_weights,protected,c,skin_weights=weights)
    boundary=np.unique(edges[hair[edges[:,0]]!=hair[edges[:,1]]])
    distance,_=cKDTree(original[boundary]).query(original,workers=4)
    ear_distance,_=cKDTree(original[protected]).query(original,workers=4)
    freedom=smoothstep(distance/.010)*smoothstep(ear_distance/.006)*hair
    freedom*=smoothstep((y-c['sideLine']+.085)/.018)
    freedom*=1-smoothstep((y-c['sideLine']-.010)/.025)
    freedom*=smoothstep((.014-z)/.018)
    freedom*=smoothstep((abs(x)-c['radiusX']*.30)/.020)
    # Remove the remaining short-hair relief without changing topology or
    # smoothing the ear itself. Keep the inward-only size constraint.
    centre=np.asarray(shape_stats['fit_centre_m'])
    maximum_radius=np.linalg.norm(baseline-centre,axis=1)
    for _ in range(80):
        points+=(average@points-points)*(.4*freedom[:,None])
        delta=points-centre;radius=np.linalg.norm(delta,axis=1)
        scale=np.minimum(1,maximum_radius/np.maximum(radius,1e-12))
        points=centre+delta*scale[:,None]
        points[~hair|protected]=original[~hair|protected]
    shape_stats['maximum_vertex_movement_mm']=float(np.linalg.norm(points-baseline,axis=1).max()*1000)
    shape_stats['posterior_cleanup_vertices']=int((freedom>0).sum())
    shape_stats['posterior_mask_added_vertices']=int((ear_cleanup&(weights==0)).sum())
    # Mixed fringe pixels extend into the nominal skin mask. Continue skin
    # from a clean outer ring, preserving eyes, brows and protected ear folds.
    to_hair,_=cKDTree(original[hair]).query(original,workers=4)
    repair=(hair|(to_hair<.012))&~protected
    from bald_regions import repair_ear_rim_colours
    if ear is None:ear=np.zeros(len(original),bool)
    adjusted_rgb,rim_mask=repair_ear_rim_colours(source,c,ear,weights)
    colour=extend_skin_colours(original,points,adjusted_rgb,repair,edges)
    head=trimesh.Trimesh(points,np.asarray(source.faces).copy(),process=False)
    # Broad, low-contrast baked shading makes the estimated dome readable with
    # the same unlit photographic material as the retained scanned face.
    light=np.array([-.15,.55,.82]);light/=np.linalg.norm(light)
    diffuse=np.maximum(0,np.asarray(head.vertex_normals)@light)
    shade=(.78+.22*diffuse)/(.78+.22*.8)
    amount=smoothstep(distance/.025)*hair
    factor=1+(shade-1)*amount
    linear=colour/255;linear=np.where(linear<=.04045,linear/12.92,((linear+.055)/1.055)**2.4)
    linear*=factor[:,None]
    srgb=np.where(linear<=.0031308,linear*12.92,1.055*np.maximum(linear,0)**(1/2.4)-.055)
    colour=np.clip(np.rint(srgb*255),0,255).astype(np.uint8)
    colour[~repair]=adjusted_rgb[~repair].astype(np.uint8)
    head.visual.vertex_colors=np.column_stack([colour,np.full(len(points),255,np.uint8)])
    head.metadata['shape_reference_vertices']=baseline
    head.metadata['shape_fit']=shape_stats
    head.metadata['shape_weights']=shape_weights
    head.metadata['ear_rim_mask']=rim_mask
    shape_stats['sideburn_cleanup_vertices']=int(sideburn.sum())
    if not np.isfinite(points).all():raise RuntimeError('光头补全包含无效曲面。')
    if not np.array_equal(points[protected],original[protected]):raise RuntimeError('光头补全移动了脸部或耳部。')
    return head,{'strand_count':0,'estimated_scalp':True,'skin_repair_vertices':int(repair.sum()),
                 'hairline_blend_skin_vertices':int((repair&~hair).sum()),
                 'protected_face_and_ear_interior_geometry_and_rgb_exact':bool(np.array_equal(colour[protected&(rim_mask==0)],rgb[protected&(rim_mask==0)])),
                 'ear_rim_colour_repair_vertices':int((rim_mask>0).sum()),
                 'shape_fit':shape_stats,
                 'vertices':len(points),'faces':len(head.faces)}
