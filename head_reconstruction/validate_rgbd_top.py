"""Compare the crown projection and the retained front of local mesh versions."""
import json
from pathlib import Path
import cv2
import numpy as np
import trimesh
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree

from rgbd_context import ROOT,OUT,ASSETS

def projection(mesh, top=True):
    v=np.asarray(mesh.vertices); f=np.asarray(mesh.faces)
    axes=[0,2] if top else [0,1]
    uv=(v[:,axes]*(1900 if top else 1450)+[230,240]).astype(np.int32)
    if not top:uv[:,1]=480-uv[:,1]
    depth=v[:,1] if top else v[:,2]
    pic=np.full((480,460,3),[237,243,246],np.uint8)
    col=np.asarray(mesh.visual.vertex_colors)[f,:3].mean(1)
    for i in np.argsort(depth[f].mean(1)):
        cv2.fillConvexPoly(pic,uv[f[i]],tuple(map(int,col[i])),lineType=cv2.LINE_AA)
    return Image.fromarray(pic)

def roof_mask(mesh):
    v=np.asarray(mesh.vertices);f=np.asarray(mesh.faces)
    # Same X/Z grid in metres; only actual upper surface triangles cover a cell.
    uv=np.round(v[:,[0,2]]*1000+[150,180]).astype(np.int32)
    selected=(mesh.triangles_center[:,1]>v[:,1].max()-.095)&(mesh.face_normals[:,1]>.25)
    mask=np.zeros((360,300),np.uint8)
    for face in f[selected]:cv2.fillConvexPoly(mask,uv[face],1)
    return mask

def main():
    old=trimesh.load(OUT/'before_top_recovery/head.ply',process=False)
    new=trimesh.load(ASSETS/'head.ply',process=False)
    measured=trimesh.load(OUT/'sensor_recovered_head.ply',process=False)
    meta=json.loads((ASSETS/'metadata.json').read_text(encoding='utf-8'))
    before_meta=json.loads((OUT/'before_top_recovery/metadata.json').read_text(encoding='utf-8'))
    retained=np.array_equal(new.vertices[:len(measured.vertices)],measured.vertices)
    boundary=new.edges_unique[np.bincount(new.edges_unique_inverse,minlength=len(new.edges_unique))==1]
    crown_open=int((new.vertices[boundary,1].min(1)>measured.vertices[:,1].max()-.05).sum())
    # Canonical exports use their own bounding-box centre; align using the common neck cut.
    if 'reference_camera_centre_m' in meta and 'reference_camera_centre_m' in before_meta:
        shift=(np.asarray(meta['reference_camera_centre_m'])-np.asarray(before_meta['reference_camera_centre_m']))*[1,-1,-1]
    else:shift=np.array([0,old.bounds[0,1]-new.bounds[0,1],0])
    new.apply_translation(shift)
    measured.apply_translation(shift)
    roi=lambda v:(abs(v[:,0])<.06)&(v[:,2]>.035)&(v[:,1]<.025)&(v[:,1]>-.09)
    a=old.vertices[roi(old.vertices)]; b=new.vertices[roi(new.vertices)]
    if not len(a) or not len(b):raise RuntimeError('No front surface available for geometry validation')
    distance=cKDTree(a).query(b)[0]*1000
    masks=[roof_mask(m) for m in (old,measured,new)]
    union=(masks[0]|masks[1]|masks[2])*255
    hull=cv2.convexHull(cv2.findNonZero(union))
    domain=np.zeros_like(union);cv2.fillConvexPoly(domain,hull,1)
    domain=cv2.erode(domain,np.ones((9,9),np.uint8))>0
    gaps=[int(((mask==0)&domain).sum()) for mask in masks]
    scene=trimesh.load(ASSETS/'head.glb',force='scene',process=False)
    glb=scene.to_geometry()
    glb.apply_translation(shift)
    match=len(glb.vertices)==len(new.vertices) and len(glb.faces)==len(new.faces) and np.allclose(glb.vertices,new.vertices,atol=1e-7) and np.array_equal(glb.faces,new.faces)
    result={'primary_views':meta['primary_views'],'top_views_before':before_meta['top_views'],'top_views_after':meta['top_views'],
            'front_nearest_vertex_distance_mm':{'median':float(np.median(distance)),'p95':float(np.percentile(distance,95)),
            'max':float(distance.max()),'note':'mesh comparison, not physical scan accuracy'},
            'crown_projection_missing_area_mm2':{'before':gaps[0],'after_recovered_depth_only':gaps[1],'after_local_completion':gaps[2],
            'reduction_fraction_measured_only':float(1-gaps[1]/gaps[0]) if gaps[0] else None,'reduction_fraction_with_estimated_completion':float(1-gaps[2]/gaps[0]) if gaps[0] else None,'method':'1 mm X/Z grid, upward facing triangles in upper 95 mm; common eroded convex hull'},
            'measured_source_vertices_retained_exactly':retained,'remaining_crown_boundary_edges':crown_open,
            'ply_glb_counts_match':match,'finite_geometry':bool(np.isfinite(new.vertices).all()),'mesh_closed':bool(new.is_watertight),
            'exported_vertices':len(new.vertices),'exported_faces':len(new.faces),'new_to_old_centre_translation_m':shift.tolist()}
    (OUT/'top_improvement_validation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    sheet=Image.new('RGB',(1380,1020),'#edf3f6');draw=ImageDraw.Draw(sheet)
    for row,top in enumerate([True,False]):
        for col,mesh in enumerate([old,measured,new]):
            sheet.paste(projection(mesh,top),(col*460,row*510+30))
            labels=[f"Before | {before_meta['top_views']} high views",f"Recovered depth | {meta['top_views']} high views",'Final | local estimated completion']
            draw.text((col*460+16,row*510+10),labels[col]+(' | crown' if top else ' | front'),fill='#203944')
    sheet.save(ASSETS/'top_comparison.jpg',quality=94)
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
