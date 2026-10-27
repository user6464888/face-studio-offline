"""Offline fitted hair geometry. New hair is simulated, not sensor measurement."""
import numpy as np
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter1d
from scalp_fit import fit_scalp
from groom_variants import PRESETS,fitted_strands

LIBRARY={key:spec[0] for key,spec in PRESETS.items()}

class ScanAtlas:
    def __init__(self,mesh,c,weights,measured_count):
        self.c=c;self.centre=np.array([c['centerX'],c['centerY'],c['centerZ']]);self.radii=np.array([c['radiusX'],c['radiusY'],c['radiusZ']])
        ids=np.flatnonzero((weights>200)&(np.arange(len(weights))<measured_count))
        if len(ids)<100:raise RuntimeError('实测头发点不足，无法适配发型。')
        self.xyz=np.asarray(mesh.vertices)[ids];self.rgb=np.asarray(mesh.visual.vertex_colors)[ids,:3]
        q=(self.xyz-self.centre)/self.radii;self.radius=np.linalg.norm(q,axis=1);self.directions=q/self.radius[:,None];self.tree=cKDTree(self.directions)
        theta=np.arctan2(q[:,0],q[:,2])%(2*np.pi);bins=128;index=(theta*bins/(2*np.pi)).astype(int)%bins
        roots=np.full((bins,3),np.nan)
        for i in range(bins):
            points=self.xyz[index==i]
            if len(points):
                floor=np.quantile(points[:,1],.03);roots[i]=np.median(points[points[:,1]<floor+.004],axis=0)
        known=np.flatnonzero(np.isfinite(roots[:,0]))
        for k in range(3):roots[:,k]=np.interp(np.arange(bins),np.r_[known-bins,known,known+bins],np.tile(roots[known,k],3))
        self.roots=gaussian_filter1d(roots,1.8,axis=0,mode='wrap')

    def root(self,theta):
        phase=(theta%(2*np.pi))/(2*np.pi)*len(self.roots)
        a=np.floor(phase).astype(int)%len(self.roots);b=(a+1)%len(self.roots);t=phase-a
        return self.roots[a]*(1-t[:,None])+self.roots[b]*t[:,None]

    def sample(self,directions):
        distance,neighbor=self.tree.query(directions,k=32,workers=4)
        radial=np.median(self.radius[neighbor],axis=1)
        # Source colour variation is transferred, while new locks and lengths
        # are explicitly generated geometry. No new photos are invented.
        colours=self.rgb[neighbor[:,0]].copy()
        return radial,colours

    def colours(self,positions):
        q=(positions-self.centre)/self.radii;q/=np.maximum(np.linalg.norm(q,axis=1,keepdims=True),1e-9)
        return self.sample(q)[1]

    def attach_scalp(self,scalp,weights):
        self.skin_mesh=scalp;self.root_scene=None
        q=(scalp.vertices[weights>0]-self.centre)/self.radii
        self.skin_radius=np.linalg.norm(q,axis=1)
        self.skin_tree=cKDTree(q/self.skin_radius[:,None])

    def anchor_roots(self,curves):
        # Directional interpolation guides the groom; final roots use actual
        # triangles of the common bare scalp, including its repaired hairline.
        import open3d as o3d
        if self.root_scene is None:
            self.root_scene=o3d.t.geometry.RaycastingScene()
            self.root_scene.add_triangles(o3d.core.Tensor(np.asarray(self.skin_mesh.vertices,dtype=np.float32)),
                                          o3d.core.Tensor(np.asarray(self.skin_mesh.faces,dtype=np.uint32)))
        hit=self.root_scene.compute_closest_points(o3d.core.Tensor(np.asarray(curves[:,0],dtype=np.float32)))
        point=hit['points'].numpy().astype(float);normal=hit['primitive_normals'].numpy().astype(float)
        normal*=np.where(np.sum(normal*(point-self.centre),axis=1)<0,-1,1)[:,None]
        target=point-normal*.00025
        arc=np.column_stack([np.zeros(len(curves)),np.cumsum(np.linalg.norm(np.diff(curves,axis=1),axis=2),axis=1)])
        return curves+(target-curves[:,0])[:,None]*np.exp(-arc/.012)[:,:,None]

    def conform_density(self,curves):
        import open3d as o3d
        shape=curves.shape
        hit=self.root_scene.compute_closest_points(o3d.core.Tensor(np.asarray(curves.reshape(-1,3),dtype=np.float32)))
        point=hit['points'].numpy().astype(float);normal=hit['primitive_normals'].numpy().astype(float)
        normal*=np.where(np.sum(normal*(point-self.centre),axis=1)<0,-1,1)[:,None]
        point=point.reshape(shape);normal=normal.reshape(shape)
        arc=np.column_stack([np.zeros(len(curves)),np.cumsum(np.linalg.norm(np.diff(point,axis=1),axis=2),axis=1)])
        emergence=np.clip(arc/.0015,0,1);emergence=emergence*emergence*(3-2*emergence)
        height=-.00025*(1-emergence)+(.00035+.0006*np.sin(np.linspace(0,np.pi,shape[1])))[None,:]*emergence
        return point+normal*height[:,:,None]

    def surface(self,direction):
        _,near=self.skin_tree.query(np.asarray(direction).reshape(-1,3),k=16,workers=4)
        return np.median(self.skin_radius[near],axis=1).reshape(np.asarray(direction).shape[:-1])


def build_library(mesh,c,weights,measured_count,generate_styles=True):
    atlas=ScanAtlas(mesh,c,weights,measured_count)
    body,scalp,boundary_distance,keep=fit_scalp(mesh,weights)
    atlas.attach_scalp(scalp,weights)
    skin=np.asarray(scalp.visual.vertex_colors)[:,:3].copy()
    hair_rgb=atlas.colours(scalp.vertices)
    styles={};statistics={};masks={key:np.zeros(len(scalp.vertices),dtype=np.uint8) for key in LIBRARY}
    # Coloured coverage comes from strands. The skin continuation stays intact
    # under every hairstyle and every dye choice.
    for key in (LIBRARY if generate_styles else ()):
        hair,stat=fitted_strands(atlas,key)
        if not np.isfinite(hair.vertices).all():raise RuntimeError(f'{key} 发型包含无效几何。')
        styles[key]=hair;statistics[key]=stat
        # Compatibility masks remain on disk, but scalp is never dyed. Strand
        # geometry supplies all coloured coverage, including near the hairline.
        masks[key]=np.zeros(len(scalp.vertices),dtype=np.uint8)
    scalp.metadata.update(hair_weights=masks['straight'],style_hair_weights=masks,hair_rgb=hair_rgb,skin_rgb=skin)
    return body,scalp,styles,{'presets':statistics,'body_faces':len(body.faces),'removed_original_hair_faces':int((~keep).sum()),'scalp_faces':len(scalp.faces),'source_face_and_ear_vertices_unchanged':True,'scalp_face_boundary_continuous':True,'source_hair_samples':len(atlas.xyz),'generation':'authored strands with labelled cut/groom variants; continuous scan-topology scalp, tapered embedded roots and baby-hair transition'}
