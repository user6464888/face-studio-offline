"""Close only measured crown boundary loops; retain source geometry and face.

Patch interiors are estimates, recorded separately from Gemini depth observations.
No lower head, ear or neck boundary is completed by this operation.
"""
import json
import argparse
from pathlib import Path
from collections import defaultdict
import numpy as np
import trimesh
import mapbox_earcut
from shapely.geometry import Polygon
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import spsolve
from reconstruct_rgbd import preview_mesh,save_json

from rgbd_context import ROOT,OUT,ASSETS

def refine_patch(points,faces,boundary_count):
    points=list(points)
    for iteration in range(8):
        edges=np.unique(np.sort(np.vstack([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]]),axis=1),axis=0)
        p=np.asarray(points);length=np.linalg.norm(p[edges[:,0]]-p[edges[:,1]],axis=1)
        mid={}
        for a,b in edges[length>.003]:
            if a<boundary_count and b<boundary_count and (abs(int(a)-int(b))==1 or {int(a),int(b)}=={0,boundary_count-1}):continue
            mid[(int(a),int(b))]=len(points);points.append((p[a]+p[b])/2)
        if not mid:break
        def edge(a,b):return mid.get(tuple(sorted((int(a),int(b)))))
        triangles=[]
        for a,b,c in faces:
            ab,bc,ca=edge(a,b),edge(b,c),edge(c,a)
            count=sum(x is not None for x in (ab,bc,ca))
            if count==0:triangles.append([a,b,c])
            elif count==3:triangles.extend([[a,ab,ca],[ab,b,bc],[ca,bc,c],[ab,bc,ca]])
            elif count==1:
                if ab is not None:triangles.extend([[a,ab,c],[ab,b,c]])
                elif bc is not None:triangles.extend([[b,bc,a],[bc,c,a]])
                else:triangles.extend([[c,ca,b],[ca,a,b]])
            else:
                if ab is None:triangles.extend([[c,ca,bc],[ca,a,b],[ca,b,bc]])
                elif bc is None:triangles.extend([[a,ab,ca],[ab,b,c],[ab,c,ca]])
                else:triangles.extend([[b,bc,ab],[bc,c,a],[bc,a,ab]])
        faces=np.asarray(triangles,dtype=np.int64)
    return np.asarray(points),faces

def main(refresh_source=False):
    # Always read an unpatched source to make the operation repeatable.
    source=OUT/'sensor_recovered_head.ply'
    if refresh_source:
        metadata=json.loads((ASSETS/'metadata.json').read_text(encoding='utf-8'))
        if metadata.get('crown_surface_completion'):raise RuntimeError('Cannot replace measured source with an already patched mesh')
    if refresh_source or not source.exists():source.write_bytes((ASSETS/'head.ply').read_bytes())
    m=trimesh.load(source,process=False);v=np.asarray(m.vertices);col=np.asarray(m.visual.vertex_colors)
    crown_floor=float(v[:,1].max()-.05)
    boundary=m.edges_unique[np.bincount(m.edges_unique_inverse,minlength=len(m.edges_unique))==1]
    graph=coo_matrix((np.ones(len(boundary)*2),(np.r_[boundary[:,0],boundary[:,1]],np.r_[boundary[:,1],boundary[:,0]])),shape=(len(v),len(v)))
    _,labels=connected_components(graph)
    vertices=[v];colours=[col];faces=[m.faces];records=[];offset=len(v)
    for component in np.unique(labels[boundary.ravel()]):
        edges=boundary[labels[boundary[:,0]]==component];ids=np.unique(edges)
        if len(ids)<3 or v[ids,1].min()<crown_floor:continue
        adj=defaultdict(list)
        for a,b in edges:adj[int(a)].append(int(b));adj[int(b)].append(int(a))
        if any(len(adj[i])!=2 for i in ids):continue
        ordered=[int(ids[0])];last=-1
        while True:
            following=next(i for i in adj[ordered[-1]] if i!=last)
            if following==ordered[0]:break
            last=ordered[-1];ordered.append(following)
            if len(ordered)>len(ids):raise RuntimeError('Boundary loop traversal failed')
        if len(ordered)!=len(ids):continue
        p=v[ordered];poly=Polygon(p[:,[0,2]])
        record={'boundary_vertices':len(p),'area_projection_mm2':poly.area*1e6,'valid_polygon':bool(poly.is_valid)}
        if poly.area<1e-8 or poly.area>.006:
            record['patched']=False;records.append(record);continue
        if poly.is_valid:
            chart=p.copy();record['parameterization']='XZ projection'
        else:
            # Folded scan edges can cross in top projection. Use a simple disk
            # chart while retaining every measured boundary point in 3D.
            lengths=np.linalg.norm(np.roll(p,-1,axis=0)-p,axis=1)
            theta=np.r_[0,np.cumsum(lengths[:-1])]/lengths.sum()*2*np.pi
            radius=np.sqrt(poly.area/np.pi)
            chart=np.column_stack([radius*np.cos(theta),np.zeros(len(p)),radius*np.sin(theta)])
            record['parameterization']='arc-length disk for folded boundary'
        t=mapbox_earcut.triangulate_float64(np.ascontiguousarray(chart[:,[0,2]]),np.array([len(p)],np.uint32)).reshape(-1,3)
        q,t=refine_patch(chart,t,len(p));n=len(q);nb=len(p)
        edge=np.unique(np.sort(np.vstack([t[:,[0,1]],t[:,[1,2]],t[:,[2,0]]]),axis=1),axis=0)
        # Cotangent weights retain linear precision on the disk chart; equal
        # graph weights distort thin triangulations into visible radial fans.
        uv=q[:,[0,2]];rows=[];columns=[];weights=[]
        for a,b,c in [(0,1,2),(1,2,0),(2,0,1)]:
            u=uv[t[:,a]]-uv[t[:,c]];w=uv[t[:,b]]-uv[t[:,c]]
            area=np.maximum(abs(u[:,0]*w[:,1]-u[:,1]*w[:,0]),1e-18)
            cot=(u*w).sum(1)/area*.5
            rows.extend([t[:,a],t[:,b]]);columns.extend([t[:,b],t[:,a]]);weights.extend([cot,cot])
        rows=np.concatenate(rows);columns=np.concatenate(columns);weights=np.concatenate(weights)
        adjacency=coo_matrix((weights,(rows,columns)),shape=(n,n)).tocsr()
        degree=np.asarray(adjacency.sum(1)).ravel()
        lap=coo_matrix((degree,(np.arange(n),np.arange(n))),shape=(n,n)).tocsr()-adjacency
        if not poly.is_valid:
            q[nb:]=spsolve(lap[nb:,nb:],-lap[nb:,:nb]@p)
        q[:nb]=p
        # A quadratic surrounding crown followed by harmonic boundary residuals.
        tree=cKDTree(p[:,[0,2]])
        near=tree.query(v[:,[0,2]])[0]<.014
        near&=(v[:,1]>crown_floor)
        sample=v[near];origin=p[:,[0,2]].mean(0)
        def basis(points):
            x,z=(points[:,[0,2]]-origin).T
            return np.column_stack([np.ones(len(x)),x,z,x*x,x*z,z*z])
        coefficient=np.linalg.lstsq(basis(sample),sample[:,1],rcond=None)[0]
        trend=basis(q)@coefficient
        fixed=np.column_stack([p[:,1]-trend[:nb],col[ordered,:3].astype(float)])
        interior=spsolve(lap[nb:,nb:],-lap[nb:,:nb]@fixed) if n>nb else np.empty((0,4))
        q[nb:,1]=np.clip(trend[nb:]+interior[:,0],p[:,1].min()-.002,p[:,1].max()+.004)
        # Blend interpolated edge colour towards surrounding dark hair away
        # from the seam. No invented individual hair texture is generated.
        dark=col[near,:3].astype(float)
        dark=dark[dark.mean(1)<65]
        median=np.median(dark,axis=0) if len(dark) else np.median(col[ordered,:3],axis=0)
        edge_distance=tree.query(q[nb:,[0,2]])[0]
        blend=np.clip(edge_distance/.008,0,1)[:,None]*.85
        rgb=interior[:,1:]*(1-blend)+median*blend
        patch_col=np.column_stack([np.clip(rgb,0,255),np.full(n-nb,255)]).astype(np.uint8)
        lookup=np.r_[ordered,np.arange(offset,offset+n-nb)]
        triangles=lookup[t]
        cross=np.cross(q[t[:,1]]-q[t[:,0]],q[t[:,2]]-q[t[:,0]])
        triangles[cross[:,1]<0]=triangles[cross[:,1]<0][:,::-1]
        vertices.append(q[nb:]);colours.append(patch_col);faces.append(triangles);offset+=n-nb
        record.update(patched=True,added_vertices=n-nb,added_faces=len(t));records.append(record)
    result=trimesh.Trimesh(np.concatenate(vertices),np.concatenate(faces),vertex_colors=np.concatenate(colours),process=False)
    result.fix_normals()
    assert np.array_equal(result.vertices[:len(v)],v),'Measured vertices changed'
    result.export(ASSETS/'head.ply');result.export(ASSETS/'head.glb');result.export(OUT/'head_metric.ply')
    preview_mesh(result,ASSETS/'model_overview.jpg')
    patch_report={'method':'local crown loop triangulation, quadratic surrounding surface + harmonic boundary interpolation',
                  'geometry_is_estimated_inside_patches':True,'measured_vertices_unchanged':True,
                  'source_mesh':'sensor_recovered_head.ply','patches':records,'added_vertices':len(result.vertices)-len(v),
                  'crown_floor_model_y_m':crown_floor,
                  'added_faces':len(result.faces)-len(m.faces),'neck_and_face_completed':False}
    save_json(OUT/'crown_patch.json',patch_report)
    meta=json.loads((ASSETS/'metadata.json').read_text(encoding='utf-8'))
    meta.update(vertices=len(result.vertices),faces=len(result.faces),is_watertight=bool(result.is_watertight),
                dimensions_mm=(np.ptp(result.vertices,axis=0)*1000).tolist(),
                crown_surface_completion=patch_report,summary='主录像深度重建改进版；利用可靠高位画面，并对头顶剩余缺口做局部曲面估计。补面不是实测深度，脖子底部保留开口。')
    save_json(ASSETS/'metadata.json',meta);save_json(OUT/'result.json',meta)
    print(json.dumps(patch_report,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--refresh-source',action='store_true')
    main(parser.parse_args().refresh_source)
