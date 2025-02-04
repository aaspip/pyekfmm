
## This DEMO is a 2D example [x,z] with velocity gradient and with one shot
# 
#  COPYRIGHT: Yangkang Chen, 2022, The University of Texas at Austin

import pyekfmm as fmm
import numpy as np
import matplotlib.pyplot as plt

import pyekfmm as fmm
import numpy as np

### 1D
v1=1;
v2=3;
nz=501;
nx=501;
dx=0.01;
dz=0.01;
# vel=3.0*np.ones([501*501,1],dtype='float32'); #velocity axis must be x,y,z respectively
v=np.linspace(v1,v2,nz);
v=np.expand_dims(v,1);
h=np.ones([1,nx])
vel=np.multiply(v,h,dtype='float32'); #z,x
# plt.imshow(vel);plt.jet();plt.show()

t=fmm.eikonal(vel.transpose().flatten(order='F'),xyz=np.array([0,0,0]),ax=[0,dx,nx],ay=[0,0.01,1],az=[0,dz,nz],order=2);
time=t.reshape(nx,nz,order='F');#first axis (vertical) is x, second is z
time=time.transpose(); #z,x

# tz=np.gradient(time,axis=1);
# tx=np.gradient(time,axis=0);
# # or
tz,tx = np.gradient(time)


# fig = plt.figure(figsize=(16, 8))
fig, ax = plt.subplots(1, 2, gridspec_kw={'width_ratios': [1, 1.5]})
plt.subplot(1,2,1);
fig.tight_layout();

plt.imshow(time,cmap=plt.cm.jet, interpolation='none', extent=[0,5,5,0]); #transpose so that first axis is z, second is x
plt.plot(0,0.0,'*r',markersize=20);
plt.xlabel('X (km)');plt.ylabel('Z (km)');
plt.jet()
plt.colorbar(orientation='horizontal',shrink=0.6,label='Traveltime (s)');

for ii in range(1,502,50):
	paths,nrays=fmm.stream2d(-tx,-tz, 501, ii, step=0.1, maxvert=10000)
	plt.plot(500*dx,(ii-1)*dz,'vb',markersize=15);
	## plot rays
	plt.plot((paths[0,:]-1)*dx,(paths[1,:]-1)*dz,'g--',markersize=20);


plt.gca().text(-0.5, -0.66, "a)", fontsize=28, color='k')


plt.subplot(1,2,2);
plt.axis('off')


### 3D 
v1=1;
v2=3;
nz=101;
nx=101;
ny=101;
dx=0.01;
dz=0.01;
dy=0.01;
# vel=3.0*np.ones([101*101,1],dtype='float32'); #velocity axis must be x,y,z respectively
v=np.linspace(v1,v2,nz);
v=np.expand_dims(v,1);
h=np.ones([1,nx])
vel=np.multiply(v,h,dtype='float32'); #z,x

vel3d=np.zeros([nz,nx,ny],dtype='float32');
for ii in range(ny):
	vel3d[:,:,ii]=vel

vxyz=np.swapaxes(np.swapaxes(vel3d,0,1),1,2);
t=fmm.eikonal(vxyz.flatten(order='F'),xyz=np.array([0,0,0]),ax=[0,dx,nx],ay=[0,dy,ny],az=[0,dz,nz],order=2);
time=t.reshape(nx,ny,nz,order='F');#first axis (vertical) is x, second is y, third is z

# tz=np.gradient(time,axis=1);
# tx=np.gradient(time,axis=0);

tx,ty,tz = np.gradient(time)

receiverx=101.0
receivery=101.0
receiverz=101.0
paths,nrays=fmm.stream3d(-tx,-ty, -tz, dx, dy, dz, receiverx, receivery, receiverz, step=0.1, maxvert=10000)
print('Before trim',paths.shape)
## trim the rays and add the source point
paths=fmm.trimrays(paths,start_points=np.array([1,1,1]),T=0.5)
print('After trim',paths.shape)

import matplotlib.pyplot as plt
import numpy as np

# Define dimensions
Nx, Ny, Nz = 101, 101, 101
X, Y, Z = np.meshgrid(np.arange(Nx)*0.01, np.arange(Ny)*0.01, np.arange(Nz)*0.01)

# Specify the 3D data
data=np.transpose(time,(1,0,2)); ## data requires [y,x,z] so tranpose the first and second axis
# data=np.transpose(time,(2,1,0)); #[z,x,y] -> [y,x,z]

kw = {
    'vmin': data.min(),
    'vmax': data.max(),
    'levels': np.linspace(data.min(), data.max(), 10),
}

ax = fig.add_subplot(122, projection='3d')
plt.jet()
# Plot contour surfaces
_ = ax.contourf(
    X[:, :, -1], Y[:, :, -1], data[:, :, 0],
    zdir='z', offset=0, alpha=0.7, **kw
)
_ = ax.contourf(
    X[0, :, :], data[0, :, :], Z[0, :, :],
    zdir='y', offset=0, alpha=0.7, **kw
)
C = ax.contourf(
    data[:, -1, :], Y[:, -1, :], Z[:, -1, :],
    zdir='x', offset=X.max(), alpha=0.7, **kw
)
# --

# Set limits of the plot from coord limits
xmin, xmax = X.min(), X.max()
ymin, ymax = Y.min(), Y.max()
zmin, zmax = Z.min(), Z.max()
ax.set(xlim=[xmin, xmax], ylim=[ymin, ymax], zlim=[zmin, zmax])

# Set labels and zticks
ax.set(
    xlabel='X (km)',
    ylabel='Y (km)',
    zlabel='Z (km)',
)

# Colorbar
cbar=fig.colorbar(C, ax=ax, orientation='horizontal', fraction=0.02, pad=0.1, format= "%.2f", label='Traveltime (s)')
cbar.ax.locator_params(nbins=5)

plt.gca().scatter(0.0,0,0,s=200,marker='*',color='r')
plt.gca().set_xlim(0,1);
plt.gca().set_ylim(0,1);
plt.gca().set_zlim(0,1);

plt.plot((receivery-1)*dy,(receiverx-1)*dx,(receiverz-1)*dz,'vb',markersize=15);
## plot rays
plt.plot((paths[1,:]-1)*dy,(paths[0,:]-1)*dx,(paths[2,:]-1)*dz,'g--',markersize=20);


for ii in range(1,102,10):
	paths,nrays=fmm.stream3d(-tx,-ty, -tz, dx, dy, dz, 101, 101, ii, step=0.1, maxvert=10000)
	plt.plot((101-1)*dy,(101-1)*dx,(ii-1)*dz,'vb',markersize=10);
	## plot rays
	plt.plot((paths[1,:]-1)*dy,(paths[0,:]-1)*dx,(paths[2,:]-1)*dz,'g--',markersize=20);
	
	
plt.gca().invert_zaxis()
plt.gca().text(-0.124, 0, -0.66, "b)", fontsize=28, color='k')

plt.savefig('test_pyekfmm_fig4.png',format='png',dpi=300,bbox_inches='tight', pad_inches=0)
plt.savefig('test_pyekfmm_fig4.pdf',format='pdf',dpi=300,bbox_inches='tight', pad_inches=0)


# Show Figure
plt.show()

##########################################################################################
# Below is X != Y test
##########################################################################################
import pyekfmm as fmm
import numpy as np
import matplotlib.pyplot as plt

import pyekfmm as fmm
import numpy as np

### 1D
v1=1;
v2=3;
nz=501;
nx=501;
dx=0.01;
dz=0.01;
# vel=3.0*np.ones([501*501,1],dtype='float32'); #velocity axis must be x,y,z respectively
v=np.linspace(v1,v2,nz);
v=np.expand_dims(v,1);
h=np.ones([1,nx])
vel=np.multiply(v,h,dtype='float32'); #z,x
# plt.imshow(vel);plt.jet();plt.show()

t=fmm.eikonal(vel.transpose().flatten(order='F'),xyz=np.array([0,0,0]),ax=[0,dx,nx],ay=[0,0.01,1],az=[0,dz,nz],order=2);
time=t.reshape(nx,nz,order='F');#first axis (vertical) is x, second is z
time=time.transpose(); #z,x

# tz=np.gradient(time,axis=1);
# tx=np.gradient(time,axis=0);
# # or
tz,tx = np.gradient(time)


# fig = plt.figure(figsize=(16, 8))
fig, ax = plt.subplots(1, 2, gridspec_kw={'width_ratios': [1, 1.5]})
plt.subplot(1,2,1);
fig.tight_layout();

plt.imshow(time,cmap=plt.cm.jet, interpolation='none', extent=[0,5,5,0]); #transpose so that first axis is z, second is x
plt.plot(0,0.0,'*r',markersize=20);
plt.xlabel('X (km)');plt.ylabel('Z (km)');
plt.jet()
plt.colorbar(orientation='horizontal',shrink=0.6,label='Traveltime (s)');

for ii in range(1,502,50):
	paths,nrays=fmm.stream2d(-tx,-tz, 301, ii, step=0.1, maxvert=10000)
	plt.plot(300*dx,(ii-1)*dz,'vb',markersize=15);
	## plot rays
	plt.plot((paths[0,:]-1)*dx,(paths[1,:]-1)*dz,'g--',markersize=20);


plt.gca().text(-0.5, -0.66, "a)", fontsize=28, color='k')


plt.subplot(1,2,2);
plt.axis('off')


### 3D 
v1=1;
v2=3;
nz=101;
nx=101;
ny=101;
dx=0.01;
dz=0.01;
dy=0.01;
# vel=3.0*np.ones([101*101,1],dtype='float32'); #velocity axis must be x,y,z respectively
v=np.linspace(v1,v2,nz);
v=np.expand_dims(v,1);
h=np.ones([1,nx])
vel=np.multiply(v,h,dtype='float32'); #z,x

vel3d=np.zeros([nz,nx,ny],dtype='float32');
for ii in range(ny):
	vel3d[:,:,ii]=vel

vxyz=np.swapaxes(np.swapaxes(vel3d,0,1),1,2);
t=fmm.eikonal(vxyz.flatten(order='F'),xyz=np.array([0,0,0]),ax=[0,dx,nx],ay=[0,dy,ny],az=[0,dz,nz],order=2);
time=t.reshape(nx,ny,nz,order='F');#first axis (vertical) is x, second is y, third is z

# tz=np.gradient(time,axis=1);
# tx=np.gradient(time,axis=0);

tx,ty,tz = np.gradient(time)

receiverx=101.0
receivery=101.0
receiverz=101.0
paths,nrays=fmm.stream3d(-tx,-ty, -tz, dx, dy, dz, receiverx, receivery, receiverz, step=0.1, maxvert=10000)
print('Before trim',paths.shape)
## trim the rays and add the source point
paths=fmm.trimrays(paths,start_points=np.array([1,1,1]),T=0.5)
print('After trim',paths.shape)

import matplotlib.pyplot as plt
import numpy as np

# Define dimensions
Nx, Ny, Nz = nx,ny,nz
X, Y, Z = np.meshgrid(np.arange(Nx)*dx, np.arange(Ny)*dy, np.arange(Nz)*dz)

# Specify the 3D data
data=np.transpose(time,(1,0,2)); ## data requires [y,x,z] so tranpose the first and second axis
# data=np.transpose(time,(2,1,0)); #[z,x,y] -> [y,x,z]

kw = {
    'vmin': data.min(),
    'vmax': data.max(),
    'levels': np.linspace(data.min(), data.max(), 10),
}

ax = fig.add_subplot(122, projection='3d')
plt.jet()
# Plot contour surfaces
_ = ax.contourf(
    X[:, :, -1], Y[:, :, -1], data[:, :, 0],
    zdir='z', offset=0, alpha=0.7, **kw
)
_ = ax.contourf(
    X[0, :, :], data[0, :, :], Z[0, :, :],
    zdir='y', offset=0, alpha=0.7, **kw
)
C = ax.contourf(
    data[:, -1, :], Y[:, -1, :], Z[:, -1, :],
    zdir='x', offset=X.max(), alpha=0.7, **kw
)
# --

# Set limits of the plot from coord limits
xmin, xmax = X.min(), X.max()
ymin, ymax = Y.min(), Y.max()
zmin, zmax = Z.min(), Z.max()
ax.set(xlim=[xmin, xmax], ylim=[ymin, ymax], zlim=[zmin, zmax])

# Set labels and zticks
ax.set(
    xlabel='X (km)',
    ylabel='Y (km)',
    zlabel='Z (km)',
)

# Colorbar
cbar=fig.colorbar(C, ax=ax, orientation='horizontal', fraction=0.02, pad=0.1, format= "%.2f", label='Traveltime (s)')
cbar.ax.locator_params(nbins=5)

plt.gca().scatter(0.0,0,0,s=200,marker='*',color='r')
plt.gca().set_xlim(0,xmax);
plt.gca().set_ylim(0,ymax);
plt.gca().set_zlim(0,zmax);

# plt.plot((receivery-1)*dy,(receiverx-1)*dx,(receiverz-1)*dz,'vb',markersize=15);
# ## plot rays
# plt.plot((paths[1,:]-1)*dy,(paths[0,:]-1)*dx,(paths[2,:]-1)*dz,'g--',markersize=20);
# 

for ii in range(1,102,10):
	paths,nrays=fmm.stream3d(-tx,-ty, -tz, dx, dy, dz, 101, 51, ii, step=0.1, maxvert=10000)
	plt.plot((101-1)*dx,(51-1)*dy,(ii-1)*dz,'vb',markersize=10);
	## plot rays
	plt.plot((paths[0,:]-1)*dx,(paths[1,:]-1)*dy,(paths[2,:]-1)*dz,'g--',markersize=20);
	
	
plt.gca().invert_zaxis()
plt.gca().text(-0.124, 0, -0.66, "b)", fontsize=28, color='k')

plt.savefig('test_pyekfmm_fig4_xnoty.png',format='png',dpi=300,bbox_inches='tight', pad_inches=0)
plt.savefig('test_pyekfmm_fig4_xnoty.pdf',format='pdf',dpi=300,bbox_inches='tight', pad_inches=0)


# Show Figure
plt.show()


######
#Below is a more integrated way
######
import pyekfmm as fmm
import numpy as np
import matplotlib.pyplot as plt

### 3D grid (currently only allow dx=dy=dz)
v1=1;
v2=3;
nz=101;
nx=101;
ny=101;
dx=0.01;
dz=0.01;
dy=0.01;
sx=0;
sy=0.5;
sz=0;

## create or load velocity model [z,x,y]
v=np.linspace(v1,v2,nz);
v=np.expand_dims(v,1);
h=np.ones([1,nx])
vel=np.multiply(v,h,dtype='float32'); #z,x
vel3d=np.zeros([nz,nx,ny],dtype='float32');
for ii in range(ny):
	vel3d[:,:,ii]=vel

## velocity dimension swap [z,x,y] -> [x,y,z]
vxyz=np.swapaxes(np.swapaxes(vel3d,0,1),1,2);

## FMM calculation
t=fmm.eikonal(vxyz.flatten(order='F'),xyz=np.array([sx,sy,sz]),ax=[0,dx,nx],ay=[0,dy,ny],az=[0,dz,nz],order=2);
time=t.reshape(nx,ny,nz,order='F');#first axis (vertical) is x, second is y, third is z
tzxy=np.swapaxes(np.swapaxes(time,1,2),0,1);

## plot 3D velocity model
from pyekfmm import plot3d
# plot3d(tzxy,cmap=plt.cm.jet,barlabel='Traveltime (s)',figname='vel3d.png',format='png',dpi=300)
plot3d(tzxy,nlevel=10,cmap=plt.cm.jet,barlabel='Traveltime (s)',showf=False,close=False);
plt.gca().set_xlabel("X (km)",fontsize='large', fontweight='normal')
plt.gca().set_ylabel("Y (km)",fontsize='large', fontweight='normal')
plt.gca().set_zlabel("Z (km)",fontsize='large', fontweight='normal')
plt.show()

# plot3d(vel3d,cmap=plt.cm.jet,barlabel='Velocity (km/s)',figname='vel3d.png',format='png',dpi=300)#,figname='time3d.png',format='png',dpi=300)
plot3d(vel3d,nlevel=100,cmap=plt.cm.jet,barlabel='Velocity',showf=False,close=False);
plt.gca().set_xlabel("X (km)",fontsize='large', fontweight='normal')
plt.gca().set_ylabel("Y (km)",fontsize='large', fontweight='normal')
plt.gca().set_zlabel("Z (km)",fontsize='large', fontweight='normal')
plt.show()

# Ray tracing and plotting
from pyekfmm import ray3d
plot3d(tzxy,nlevel=10,cmap=plt.cm.jet,barlabel='Traveltime (s)',showf=False,close=False,alpha=0.7);
rx=1;
ry=0.5;
z=np.linspace(0,1.0,11);
for zi in z:
	paths=ray3d(time,source=[sx,sy,sz],receiver=[rx,ry,zi],trim=0.5,ax=[0,dx,nx],ay=[0,dy,ny],az=[0,dz,nz])
	plt.plot(rx,ry,zi,'vb',markersize=10);
	## plot rays
	plt.plot(paths[0,:],paths[1,:],paths[2,:],'g--',markersize=20);

plt.savefig('raytracing3d.png',format='png',dpi=300)
# Show Figure
plt.show()





