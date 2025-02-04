
## This DEMO is a 3D ray tracing example [x,y,z] with velocity gradient and with one shot
# 
#  COPYRIGHT: Yangkang Chen, 2022, The University of Texas at Austin

import pyekfmm as fmm
import numpy as np
import matplotlib.pyplot as plt

### 3D grid (currently only allow dx=dy=dz)
v1=3;
v2=4;
v3=5;
nz=301;
nx=51;
ny=51;
dx=0.01;
dz=0.01;
dy=0.01;
sx=0;
sy=0.25;
sz=0;

z=np.arange(nz)*dz;
y=np.arange(ny)*dy;
x=np.arange(nx)*dx;

## create or load velocity model [z,x,y]
vel3d=v1*np.ones([nz,nx,ny],dtype='float32');
vel3d[100:200, :, :]=v2
vel3d[200:301, :, :]=v3

## velocity dimension swap [z,x,y] -> [x,y,z]
vxyz=np.swapaxes(np.swapaxes(vel3d,0,1),1,2);
#or
vxyz=np.transpose(vel3d,[1,2,0]);

## FMM calculation
t=fmm.eikonal(vxyz.flatten(order='F'),xyz=np.array([sx,sy,sz]),ax=[0,dx,nx],ay=[0,dy,ny],az=[0,dz,nz],order=2);
time=t.reshape(nx,ny,nz,order='F');#first axis (vertical) is x, second is y, third is z
tzxy=np.swapaxes(np.swapaxes(time,1,2),0,1);
#or
tzxy=np.transpose(time,[2,0,1]);

## plot 3D velocity model
from pyekfmm import plot3d
# plot3d(tzxy,cmap=plt.cm.jet,barlabel='Traveltime (s)',figname='vel3d.png',format='png',dpi=300)
plot3d(tzxy,dz=dz,dx=dx,dy=dy,cmap=plt.cm.jet,barlabel='Traveltime (s)',showf=False,close=False);
plt.gca().set_xlabel("X (km)",fontsize='large', fontweight='normal')
plt.gca().set_ylabel("Y (km)",fontsize='large', fontweight='normal')
plt.gca().set_zlabel("Z (km)",fontsize='large', fontweight='normal')
plt.show()

# plot3d(vel3d,cmap=plt.cm.jet,barlabel='Velocity (km/s)',figname='vel3d.png',format='png',dpi=300)#,figname='time3d.png',format='png',dpi=300)
plot3d(vel3d,x=x,y=y,z=z,cmap=plt.cm.jet,barlabel='Velocity (km/s)',showf=False,close=False);
plt.gca().set_xlabel("X (km)",fontsize='large', fontweight='normal')
plt.gca().set_ylabel("Y (km)",fontsize='large', fontweight='normal')
plt.gca().set_zlabel("Z (km)",fontsize='large', fontweight='normal')
plt.show()

# Ray tracing and plotting
from pyekfmm import ray3d
plot3d(tzxy,x=x,y=y,z=z,cmap=plt.cm.jet,barlabel='Traveltime (s)',showf=False,close=False,alpha=0.7);
rz=np.linspace(0,3,4);
rx=0.5*dx/0.01;ry=0.25*dy/0.01;
for z in rz:
	paths=ray3d(time,source=[sx,sy,sz],receiver=[rx,ry,z],trim=0.1,ax=[0,dx,nx],ay=[0,dy,ny],az=[0,dz,nz])
	plt.plot(rx,ry,z,'vb',markersize=10);
	## plot rays
	plt.plot(paths[0,:],paths[1,:],paths[2,:],'g--',markersize=20);

plt.savefig('test_pyekfmm_raytracing3d-0.png',format='png',dpi=300)
# Show Figure
plt.show()





## Combined figure
from pyseistr import plot3d #remember, this is not from pyekfmm

plt.figure(figsize=(18,18))
ax=plt.subplot(2,2,1,projection='3d')
z=np.arange(nz)*dz;
y=np.arange(ny)*dy;
x=np.arange(nx)*dx;

plot3d(vel3d,x=x,y=y,z=z,cmap=plt.cm.jet,barlabel='Velocity (km/s)',ifnewfig=False,showf=False,close=False);
plt.gca().set_xlabel("X (km)",fontsize='large', fontweight='normal')
plt.gca().set_ylabel("Y (km)",fontsize='large', fontweight='normal')
plt.gca().set_zlabel("Z (km)",fontsize='large', fontweight='normal')

ax=plt.subplot(2,2,2,projection='3d')

# Ray tracing and plotting
from pyekfmm import ray3d
[nz,nx,ny]=tzxy.shape
z=np.arange(nz)*dz;
y=np.arange(ny)*dy;
x=np.arange(nx)*dx;

plot3d(tzxy,x=x,y=y,z=z,cmap=plt.cm.jet,barlabel='Traveltime (s)',ifnewfig=False,showf=False,close=False,alpha=0.7);
rz=np.linspace(0,3,4);
rx=0.5*dx/0.01;ry=0.25*dy/0.01;
for z in rz:
	paths=ray3d(time,source=[sx,sy,sz],receiver=[rx,ry,z],step=0.1,trim=0.1,ax=[0,dx,nx],ay=[0,dy,ny],az=[0,dz,nz])
	plt.plot(rx,ry,z,'vb',markersize=10);
	## plot rays
	plt.plot(paths[0,:],paths[1,:],paths[2,:],'g--',markersize=20);

############################################################
# Much larger model
############################################################
ax=plt.subplot(2,2,3,projection='3d')
nx=301;
ny=301;
vel3d=v1*np.ones([nz,nx,ny],dtype='float32');
vel3d[100:200, :, :]=v2
vel3d[200:301, :, :]=v3

## velocity dimension swap [z,x,y] -> [x,y,z]
vxyz=np.swapaxes(np.swapaxes(vel3d,0,1),1,2);
#or
vxyz=np.transpose(vel3d,[1,2,0]);

## FMM calculation
t=fmm.eikonal(vxyz.flatten(order='F'),xyz=np.array([0,0.5*((ny-1)*dy),0]),ax=[0,dx,nx],ay=[0,dy,ny],az=[0,dz,nz],order=2);
time=t.reshape(nx,ny,nz,order='F');#first axis (vertical) is x, second is y, third is z
tzxy=np.swapaxes(np.swapaxes(time,1,2),0,1);
#or
tzxy=np.transpose(time,[2,0,1]);

z=np.arange(nz)*dz;
y=np.arange(ny)*dy;
x=np.arange(nx)*dx;

plot3d(vel3d,x=x,y=y,z=z,cmap=plt.cm.jet,barlabel='Velocity (km/s)',ifnewfig=False,showf=False,close=False);
plt.gca().set_xlabel("X (km)",fontsize='large', fontweight='normal')
plt.gca().set_ylabel("Y (km)",fontsize='large', fontweight='normal')
plt.gca().set_zlabel("Z (km)",fontsize='large', fontweight='normal')

ax=plt.subplot(2,2,4,projection='3d')

[nz,nx,ny]=tzxy.shape
z=np.arange(nz)*dz;
y=np.arange(ny)*dy;
x=np.arange(nx)*dx;

plot3d(tzxy,x=x,y=y,z=z,cmap=plt.cm.jet,barlabel='Traveltime (s)',ifnewfig=False,showf=False,close=False,alpha=0.7);
rz=np.linspace(0,3,4);
rx=(dx*(nx-1));ry=0.5*(dy*(ny-1))
for z in rz:
	paths=ray3d(time,source=[0,0.5*((ny-1)*dy),0],receiver=[rx,ry,z],step=0.1,trim=0.1,ax=[0,dx,nx],ay=[0,dy,ny],az=[0,dz,nz])
	plt.plot(rx,ry,z,'vb',markersize=10);
	## plot rays
	plt.plot(paths[0,:],paths[1,:],paths[2,:],'g--',markersize=20);

plt.savefig('test_pyekfmm_raytracing3d.png',format='png',dpi=300)
plt.show()









