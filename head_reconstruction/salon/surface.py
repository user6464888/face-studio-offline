"""Binary surface attributes used by the local hybrid material."""
import numpy as np


def write_surface(path,points,faces,normals,uv,coverage,length,flow,scalp_points):
    names=['x','y','z','nx','ny','nz','s','t','coverage','length_mm','flow_x','flow_y','flow_z','scalp_x','scalp_y','scalp_z']
    values=np.column_stack([points,normals,uv,coverage,length,flow,scalp_points])
    if not np.isfinite(values).all():raise ValueError('发层存在无效坐标')
    vertex=np.empty(len(points),dtype=[(n,'<f4') for n in names])
    for i,n in enumerate(names):vertex[n]=values[:,i]
    triangles=np.empty(len(faces),dtype=[('count','u1'),('ids','<i4',(3,))]);triangles['count']=3;triangles['ids']=faces
    header=['ply','format binary_little_endian 1.0',f'element vertex {len(vertex)}']
    header +=[f'property float {n}' for n in names]+[f'element face {len(faces)}','property list uchar int vertex_indices','end_header','']
    path.parent.mkdir(exist_ok=True,parents=True)
    temporary=path.with_suffix('.tmp')
    with temporary.open('wb') as f:f.write('\n'.join(header).encode('ascii'));f.write(vertex.tobytes());f.write(triangles.tobytes())
    temporary.replace(path)
