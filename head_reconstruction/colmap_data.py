from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation


def read_text_model(folder):
    folder = Path(folder)
    cameras = {}
    for line in (folder / 'cameras.txt').read_text().splitlines():
        if not line or line.startswith('#'):
            continue
        t = line.split()
        cameras[int(t[0])] = dict(model=t[1], width=int(t[2]), height=int(t[3]), params=np.array(t[4:], float))
    images = {}
    lines = iter((folder / 'images.txt').read_text().splitlines())
    for line in lines:
        if not line or line.startswith('#'):
            continue
        t = line.split()
        q = np.array(t[1:5], float)
        r = Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()
        trans = np.array(t[5:8], float)
        obs = np.array(next(lines).split(), float).reshape(-1, 3)
        images[t[9]] = dict(id=int(t[0]), R=r, t=trans, center=-r.T @ trans, camera_id=int(t[8]), observations=obs)
    points, colors, errors = [], [], []
    for line in (folder / 'points3D.txt').read_text().splitlines():
        if not line or line.startswith('#'):
            continue
        t = line.split()
        points.append(list(map(float, t[1:4])))
        colors.append(list(map(int, t[4:7])))
        errors.append(float(t[7]))
    return cameras, images, np.asarray(points), np.asarray(colors, np.uint8), np.asarray(errors)


def intrinsics(camera):
    params = camera['params']
    if camera['model'] in ('SIMPLE_RADIAL', 'SIMPLE_PINHOLE'):
        f, cx, cy = params[:3]
        return np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])
    if camera['model'] == 'PINHOLE':
        fx, fy, cx, cy = params[:4]
        return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    raise ValueError(camera['model'])


def canonical_frame(images, points):
    front = images[min(images)]
    # Y-up, front toward +Z, while preserving the photographed left/right.
    basis = np.diag([1, -1, -1]) @ front['R']
    origin = np.median(points, axis=0)
    return basis, origin
