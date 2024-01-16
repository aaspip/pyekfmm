## This is a script pending further investigations and tests
# 
# ON Jan, 2024
# 
##########################################################################################
# Below is X != Y test
##########################################################################################
import pyekfmm as fmm
import numpy as np
import matplotlib.pyplot as plt

import pyekfmm as fmm
import numpy as np



fig = plt.figure(figsize=(16, 8))
# fig, ax = plt.subplots(1, 2, gridspec_kw={'width_ratios': [1, 1.5]})


### 3D 
v1=3;
v2=4;
v3=5;
nz=301;
nx=51;
ny=51;
dx=0.01;
dz=0.01;
dy=0.01;

vel3d=v1*np.ones([nz,nx,ny],dtype='float32');
vel3d[100:200, :, :]=v2
vel3d[200:301, :, :]=v3

vxyz=np.swapaxes(np.swapaxes(vel3d,0,1),1,2);
t=fmm.eikonal(vxyz.flatten(order='F'),xyz=np.array([0,0.25,0]),ax=[0,dx,nx],ay=[0,dy,ny],az=[0,dz,nz],order=2);
time=t.reshape(nx,ny,nz,order='F');#first axis (vertical) is x, second is y, third is z

# tz=np.gradient(time,axis=1);
# tx=np.gradient(time,axis=0);

tx,ty,tz = np.gradient(time)

receiverx=101.0
receivery=101.0
receiverz=101.0
paths,nrays=fmm.stream3d(-tx,-ty, -tz, receiverx, receivery, receiverz, step=0.1, maxvert=10000)
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

ax = fig.add_subplot(111, projection='3d')
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

# for ii in range(1,102,10):
for ii in [0]:
	paths,nrays=fmm.stream3d(-tx,-ty, -tz, 51, 26, ii, step=0.1, maxvert=10000)
	plt.plot((51-1)*dx,(26-1)*dy,(ii-1)*dz,'vb',markersize=10);
	## plot rays
	plt.plot((paths[0,:]-1)*dx,(paths[1,:]-1)*dy,(paths[2,:]-1)*dz,'g--',markersize=20);
	
# for ii in range(0,301,100):
# # for ii in [0]:
# 	paths,nrays=fmm.stream3d(-tx,-ty, -tz, 51, 26, ii+1, step=0.5, maxvert=10000)
# 	plt.plot((51-1)*dx,(26-1)*dy,(ii-1)*dz,'vb',markersize=10);
# 	## plot rays
# 	plt.plot((paths[0,:]-1)*dx,(paths[1,:]-1)*dy,(paths[2,:]-1)*dz,'g--',markersize=20);
	
	
plt.gca().invert_zaxis()
plt.gca().text(-0.124, 0, -0.66, "b)", fontsize=28, color='k')

plt.savefig('test_pyekfmm_fig4_xnoty.png',format='png',dpi=300,bbox_inches='tight', pad_inches=0)
plt.savefig('test_pyekfmm_fig4_xnoty.pdf',format='pdf',dpi=300,bbox_inches='tight', pad_inches=0)


# Show Figure
plt.show()

