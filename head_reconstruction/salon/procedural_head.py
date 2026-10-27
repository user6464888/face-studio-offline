"""An original analytic mannequin; no scan, photograph or downloaded head input."""
import numpy as np
import trimesh

PROCEDURAL_REVISION=1


def neutral_head():
    centre=np.zeros(3)
    radii=np.array([.078,.102,.086])
    sides=128
    levels=np.linspace(-1.54,.998,112)
    theta=np.arange(sides)*2*np.pi/sides
    vertices=[]
    for y in levels:
        if y>=0:
            width=depth=np.sqrt(1-y*y)
        else:
            width=np.interp(y,[-1.54,-1.40,-1.15,-.90,-.50,0],[.12,.37,.61,.80,.94,1])
            depth=np.interp(y,[-1.54,-1.40,-1.15,-.90,-.50,0],[.28,.52,.70,.87,.96,1])
        x=width*np.sin(theta)
        z=depth*np.cos(theta)
        front=np.maximum(np.cos(theta),0)**10
        # Analytic brow, nose and mouth relief, intentionally stylised.
        relief=.20*np.exp(-((y+.58)/.16)**2)*np.exp(-(x/.14)**2)
        relief+=.035*np.exp(-((y+.25)/.10)**2)
        relief-=.022*np.exp(-((y+.99)/.035)**2)*np.exp(-(x/.36)**4)
        relief-=.035*np.exp(-((y+.40)/.075)**2)*(np.exp(-((x-.36)/.12)**2)+np.exp(-((x+.36)/.12)**2))
        z+=front*relief
        vertices.extend(np.column_stack([x,np.full(sides,y),z])*radii)
    faces=[]
    for row in range(len(levels)-1):
        for j in range(sides):
            a=row*sides+j;b=row*sides+(j+1)%sides;c=a+sides;d=b+sides
            faces.extend([(a,c,b),(b,c,d)])
    vertices=np.asarray(vertices).tolist()
    for row,y in [(0,-1.55),(len(levels)-1,1.)]:
        cap=len(vertices);vertices.append([0,y*radii[1],0])
        for j in range(sides):
            a=row*sides+j;b=row*sides+(j+1)%sides
            faces.append((cap,b,a) if row==0 else (cap,a,b))
    mesh=trimesh.Trimesh(vertices,faces,process=True)
    mesh.fix_normals()
    mesh.visual.vertex_colors=np.tile([184,170,151,255],(len(mesh.vertices),1))
    parts=[mesh]
    # Ears, neck and eye marks are also generated from primitives.
    for sign in [-1,1]:
        ear=trimesh.creation.uv_sphere(radius=1,count=[20,24])
        ear.vertices=ear.vertices*np.array([.012,.026,.016])+np.array([sign*.080,-.036,-.002])
        ear.visual.vertex_colors=np.tile([184,170,151,255],(len(ear.vertices),1));parts.append(ear)
        eye=trimesh.creation.uv_sphere(radius=1,count=[12,16])
        eye.vertices=eye.vertices*np.array([.008,.002,.002])+np.array([sign*.028,-.040,.076])
        eye.visual.vertex_colors=np.tile([65,61,57,255],(len(eye.vertices),1));parts.append(eye)
    neck=trimesh.creation.cylinder(radius=.033,height=.068,sections=64)
    neck.apply_transform(trimesh.transformations.rotation_matrix(np.pi/2,[1,0,0]))
    neck.apply_translation([0,-.162,-.012])
    neck.visual.vertex_colors=np.tile([184,170,151,255],(len(neck.vertices),1));parts.append(neck)
    return trimesh.util.concatenate(parts),centre,radii
