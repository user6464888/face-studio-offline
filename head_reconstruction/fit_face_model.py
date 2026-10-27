"""Infer a shared face prior from local video frames and fit it to calibrated views."""
from pathlib import Path
import argparse
import importlib.util
import json
import pickle
import cv2
import numpy as np
import torch
import trimesh
from PIL import Image, ImageDraw, ImageFont
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from colmap_data import read_text_model, intrinsics, canonical_frame

ROOT = Path(__file__).resolve().parent
SESSION = None
MODELS = ROOT / 'tools/3ddfa_v2'

def load_prior():
    spec = importlib.util.spec_from_file_location('face_mobilenet', MODELS/'models/mobilenet_v1.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    model = module.mobilenet(num_classes=62)
    checkpoint = torch.load(MODELS/'weights/mb1_120x120.pth', map_location='cpu', weights_only=True)['state_dict']
    state = {k.replace('module.', '').replace('fc_param.', 'fc.'): v for k,v in checkpoint.items()
             if not k.replace('module.', '').startswith('fc_lm.')}
    model.load_state_dict(state, strict=True)
    model.eval().to('cuda' if torch.cuda.is_available() else 'cpu')
    with open(MODELS/'configs/param_mean_std_62d_120x120.pkl','rb') as f: normal = pickle.load(f)
    with open(MODELS/'configs/bfm_noneck_v3.pkl','rb') as f: prior = pickle.load(f)
    with open(MODELS/'configs/tri.pkl','rb') as f: faces = pickle.load(f).T.astype(np.int32)
    return model,normal,prior,faces

def crop_square(picture, roi):
    x0,y0,x1,y1 = np.rint(roi).astype(int)
    result = np.zeros((y1-y0,x1-x0,3),np.uint8)
    h,w = picture.shape[:2]
    xa,ya,xb,yb = max(0,x0),max(0,y0),min(w,x1),min(h,y1)
    result[ya-y0:yb-y0,xa-x0:xb-x0] = picture[ya:yb,xa:xb]
    return cv2.resize(result,(120,120))

def predict(model, normal, prior, image, bbox):
    x,y,w,h = bbox; size = int((w+h)*.5*1.58)
    cx,cy = x+w/2,y+h/2+(w+h)*.5*.14
    roi = np.array([cx-size/2,cy-size/2,cx+size/2,cy+size/2])
    def infer(roi):
        crop = crop_square(image,roi)
        tensor = torch.from_numpy(crop.transpose(2,0,1).copy()).float()[None]
        tensor = ((tensor-127.5)/128).to(next(model.parameters()).device)
        with torch.inference_mode(): param = model(tensor)[0].cpu().numpy()
        param = param*normal['std']+normal['mean']
        a = param[:12].reshape(3,4)
        canonical = (prior['u'] + prior['w_shp']@param[12:52,None] + prior['w_exp']@param[52:,None]).reshape(-1,3)
        posed = canonical@a[:,:3].T+a[:,3]
        projected = posed.copy(); projected[:,0]-=1;projected[:,1]=120-projected[:,1];projected[:,2]-=1
        projected *= (roi[2]-roi[0])/120; projected[:,:2] += roi[:2]
        projected[:,2] -= projected[:,2].min()
        return param,projected,canonical
    param,projected,canonical = infer(roi)
    # The official landmark refinement crop reduces detector-box sensitivity.
    key = np.unique(prior['keypoints']//3)
    lo,hi = projected[key,:2].min(axis=0),projected[key,:2].max(axis=0)
    side = (hi-lo).max()*np.sqrt(2)/3*2
    center = (lo+hi)/2
    roi = np.r_[center-side/2,center+side/2]
    param,projected,canonical = infer(roi)
    return param,projected,canonical,roi,key

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--session',required=True);args=parser.parse_args()
    src = ROOT/'inputs'/args.session
    out = ROOT/'outputs'/args.session/'face_prior';out.mkdir(parents=True,exist_ok=True)
    cameras,images,sparse,_,_ = read_text_model(ROOT/'outputs'/args.session/'colmap_aliked/sparse_text')
    basis,origin = canonical_frame(images,sparse)
    poses = json.loads((ROOT/'outputs'/args.session/'colmap_aliked/pose_review.json').read_text())['cameras']
    yaw = {r['file']:r['yaw_degrees'] for r in poses}
    manifest = json.loads((src/'manifest.json').read_text(encoding='utf-8'))
    sharpness = {Path(r['file']).name:r['sharpness_reference'] for r in manifest['images']}
    selected=[]
    for target in range(-60,61,15):
        candidates = [name for name in images if abs(yaw[name]-target)<10]
        if candidates:
            name = max(candidates,key=lambda n:sharpness[n]/(1+abs(yaw[n]-target)*.06))
            if name not in selected:selected.append(name)
    model,normal,prior,faces=load_prior()
    detector=cv2.FaceDetectorYN.create(str(ROOT/'models/yunet_2023mar.onnx'),'',(640,360),.8,.3,5000)
    observations=[]
    for name in selected:
        image=cv2.imread(str(src/'images'/name))
        _,detected=detector.detect(cv2.resize(image,(640,360)))
        boxes=[] if detected is None else [row[:4]*2 for row in detected if row[2]>80 and row[3]>90]
        if not len(boxes):
            print('No frontal detector result:',name,flush=True);continue
        bbox=max(boxes,key=lambda b:b[2]*b[3])
        param,projected,canonical,roi,key=predict(model,normal,prior,image,bbox)
        im=images[name]
        observations.append({'name':name,'param':param,'landmarks':projected[key,:2],
            'canonical':canonical,'roi':roi,'R':im['R'],'t':im['t'],'K':intrinsics(cameras[im['camera_id']])})
        debug=image.copy()
        for p in projected[key,:2]:cv2.circle(debug,tuple(np.rint(p).astype(int)),2,(80,220,30),-1)
        cv2.imwrite(str(out/(name.replace('.jpg','_landmarks.jpg'))),debug)
        print(name,'yaw',round(yaw[name],1),'box',bbox.tolist(),flush=True)
    if len(observations)<3:raise RuntimeError('Not enough reliable face detections for multiview fitting')
    front=min(observations,key=lambda r:abs(yaw[r['name']]))
    key=np.unique(prior['keypoints']//3)
    u=prior['u'].reshape(-1,3).astype(float)
    shape=prior['w_shp'].reshape(-1,3,40).astype(float)
    expression=prior['w_exp'].reshape(-1,3,10).astype(float)
    alpha=np.median([r['param'][12:52] for r in observations if abs(yaw[r['name']])<40],axis=0)
    beta=np.median([r['param'][52:] for r in observations if abs(yaw[r['name']])<40],axis=0)
    # Initialize the model in the reference camera from its weak-perspective pose.
    camera_depth=np.median((sparse@front['R'].T+front['t'])[:,2])
    a=front['param'][:12].reshape(3,4)
    flip=np.diag([1,-1,-1])
    b=flip@a[:,:3]
    U,sv,Vt=np.linalg.svd(b);rot=U@Vt
    if np.linalg.det(rot)<0:U[:,-1]*=-1;rot=U@Vt
    focal=front['K'][0,0]
    scale=sv.mean()*(front['roi'][2]-front['roi'][0])/120*camera_depth/focal
    center_face=u.mean(axis=0)
    base=u+np.einsum('vck,k->vc',shape,alpha)+np.einsum('vck,k->vc',expression,beta)
    pixel_center=front['landmarks'].mean(axis=0)
    cam_center=np.array([(pixel_center[0]-front['K'][0,2])*camera_depth/focal,
                         (pixel_center[1]-front['K'][1,2])*camera_depth/focal,camera_depth])
    trans=front['R'].T@(cam_center-front['t'])-scale*(front['R'].T@rot)@base[key].mean(axis=0)
    worldrot=front['R'].T@rot
    initial=np.r_[np.log(scale),Rotation.from_matrix(worldrot).as_rotvec(),trans,alpha,beta]
    alpha_std=np.maximum(np.std([r['param'][12:52] for r in observations],axis=0),50000)
    beta_std=np.maximum(np.std([r['param'][52:] for r in observations],axis=0),20000)
    def vertices(parameters, subset=None):
        ids=np.arange(len(u)) if subset is None else subset
        surface=u[ids]+np.einsum('vck,k->vc',shape[ids],parameters[7:47])+np.einsum('vck,k->vc',expression[ids],parameters[47:])
        return np.exp(parameters[0])*surface@Rotation.from_rotvec(parameters[1:4]).as_matrix().T+parameters[4:7]
    def residual(parameters):
        verts=vertices(parameters,key);parts=[]
        for row in observations:
            xyz=verts@row['R'].T+row['t'];pix=xyz@row['K'].T;pix=pix[:,:2]/pix[:,2:]
            parts.append((pix-row['landmarks']).ravel())
        # Keep the shared identity near the inferred shape when landmarks do not constrain it.
        parts += [(parameters[7:47]-alpha)/alpha_std*3,(parameters[47:]-beta)/beta_std*3]
        return np.concatenate(parts)
    fit=least_squares(residual,initial,loss='soft_l1',f_scale=5,x_scale='jac',max_nfev=100)
    world=vertices(fit.x)
    canonical=(world-origin)@basis.T
    mesh=trimesh.Trimesh(canonical,faces,process=False);mesh.fix_normals()
    mesh.export(out/'face_world_canonical.ply')
    np.savez_compressed(out/'face_fit.npz',world=world,faces=faces,keypoints=key,parameters=fit.x,
        observations=np.stack([r['landmarks'] for r in observations]),names=np.array([r['name'] for r in observations]))
    errs={}
    for row in observations:
        xyz=world[key]@row['R'].T+row['t'];pix=xyz@row['K'].T;pix=pix[:,:2]/pix[:,2:]
        errs[row['name']]=float(np.linalg.norm(pix-row['landmarks'],axis=1).mean())
    stats={'source_session':args.session,'method':'3DDFA_V2 face prior with shared multiview shape fitting',
        'model_inferred':True,'metric_scale':False,'prior_is_full_head':False,'source_frames':list(errs),
        'landmark_fit_error_pixels':errs,'mean_landmark_fit_error_pixels':float(np.mean(list(errs.values()))),
        'success':bool(fit.success),'optimization_message':fit.message,'vertices':len(world),'faces':len(faces),
        'reference_frame':front['name'],'license_note':'BFM prior for academic use; see tools/3ddfa_v2/bfm/readme.md',
        'note':'Fit error measures agreement with inferred landmarks, not physical reconstruction accuracy.'}
    (out/'face_fit.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(stats,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
