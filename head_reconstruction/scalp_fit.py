"""A scalp replacement sharing the scan's exact face boundary and topology."""
import numpy as np
import trimesh
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix,diags
from scipy.sparse.linalg import spsolve


def smoothstep(value):
    t=np.clip(value,0,1)
    return t*t*(3-2*t)


def extend_skin_colours(original,points,rgb,hair,edges):
    """Continue captured skin lighting with a screened Poisson solve.

    The visible scan boundary is a Dirichlet constraint. A broad clean-skin
    estimate supplies the interior, while a smooth residual matches every
    boundary pixel. This avoids nearest-neighbour colour islands and a narrow
    painted feather band. Solve in linear light and retain source skin exactly.
    """
    value=rgb.mean(axis=1)
    retained=(~hair)&(rgb[:,0]-rgb[:,2]>15)
    threshold=np.quantile(value[retained],.65) if retained.any() else 90.
    clean=retained&(value>threshold*.8)
    ids=np.flatnonzero(clean)
    if len(ids)<16:ids=np.flatnonzero(~hair)
    encoded=rgb/255
    linear=np.where(encoded<=.04045,encoded/12.92,((encoded+.055)/1.055)**2.4)
    d,near=cKDTree(original[ids]).query(points,k=min(64,len(ids)),workers=4)
    blend=np.exp(-(d-d[:,:1])/.025)
    blend/=blend.sum(axis=1,keepdims=True)
    prior=(linear[ids[near]]*blend[:,:,None]).sum(axis=1)
    # Only non-hair vertices occur in retained body faces. The hair-side ring
    # is not shared with them and contains baked dark fringe; locking it would
    # preserve an artificial dark outline inside the replacement.
    fixed=~hair
    unknown=np.flatnonzero(~fixed)
    paired=np.vstack([edges,edges[:,::-1]])
    length=np.linalg.norm(points[paired[:,0]]-points[paired[:,1]],axis=1)
    scale=np.median(length[length>1e-8])
    conductance=np.clip(scale/np.maximum(length,scale*.2),.2,5.)
    adjacency=coo_matrix((conductance,(paired[:,0],paired[:,1])),shape=(len(points),len(points))).tocsr()
    degree=np.asarray(adjacency.sum(axis=1)).ravel()
    laplacian=diags(degree)-adjacency
    residual=np.zeros_like(linear);residual[fixed]=linear[fixed]-prior[fixed]
    screen=np.maximum(degree[unknown],1)*.002
    matrix=laplacian[unknown][:,unknown]+diags(screen)
    rhs=adjacency[unknown][:,fixed]@residual[fixed]
    residual[unknown]=spsolve(matrix.tocsc(),rhs)
    solved=np.clip(prior+residual,0,1)
    srgb=np.where(solved<=.0031308,solved*12.92,1.055*solved**(1/2.4)-.055)
    result=np.clip(np.rint(srgb*255),0,255).astype(np.uint8)
    result[fixed]=rgb[fixed].astype(np.uint8)
    if not np.isfinite(solved).all():raise RuntimeError('头皮肤色延续计算失败。')
    return result


def fit_scalp(mesh,weights):
    original=np.asarray(mesh.vertices);rgba=np.asarray(mesh.visual.vertex_colors)
    hair=weights>0;edges=mesh.edges_unique
    boundary=np.unique(edges[hair[edges[:,0]]!=hair[edges[:,1]]])
    distance,_=cKDTree(original[boundary]).query(original,workers=4)
    freedom=smoothstep(distance/.012)*hair
    # Removing the thickness of the short captured hair is an estimate. The
    # zero-displacement boundary is shared with the measured face triangles.
    points=original-np.asarray(mesh.vertex_normals)*(.0045*freedom[:,None])
    edge=np.vstack([edges,edges[:,::-1]])
    degree=np.bincount(edge[:,0],minlength=len(original))
    average=coo_matrix((1/np.maximum(degree[edge[:,0]],1),(edge[:,0],edge[:,1])),shape=(len(original),len(original))).tocsr()
    for _ in range(28):
        points+=(average@points-points)*(.42*freedom[:,None])
    points[~hair]=original[~hair]
    keep=~hair[np.asarray(mesh.faces)].any(axis=1)
    body=trimesh.Trimesh(original.copy(),mesh.faces[keep].copy(),process=False)
    body.visual.vertex_colors=rgba.copy()
    scalp=trimesh.Trimesh(points,mesh.faces[~keep].copy(),process=False)
    # Preserve source RGB on the shared boundary and continue its lighting
    # smoothly across the replacement. Hair colour never paints this skin.
    rgb=rgba[:,:3].astype(float)
    skin_rgb=extend_skin_colours(original,points,rgb,hair,edges)
    scalp.visual.vertex_colors=np.column_stack([skin_rgb,np.full(len(points),255)]).astype(np.uint8)
    if not np.array_equal(scalp.vertices[~hair],original[~hair]):
        raise RuntimeError('头皮替换移动了原始脸部边界。')
    return body,scalp,distance,keep
