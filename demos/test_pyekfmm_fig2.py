
## This DEMO is a 3D example [x,y,z] with constant velocity and with one shot
# 
#  COPYRIGHT: Yangkang Chen, 2022, The University of Texas at Austin

import pyekfmm as fmm
import numpy as np

vel=3.09354*np.ones([101*101*101,1],dtype='float32');
t=fmm.eikonal(vel,xyz=np.array([0.5,0,0]),ax=[0,0.01,101],ay=[0,0.01,101],az=[0,0.01,101],order=2);
time=t.reshape(101,101,101,order='F'); #[x,y,z]

velx=3.80395*np.ones([101*101*101,1],dtype='float32'); #velocity axis must be x,y,z respectively
# velx=3.09354*np.ones([101*101*101,1],dtype='float32'); #velocity axis must be x,y,z respectively

eta=0.340859*np.ones([101*101*101,1],dtype='float32'); #velocity axis must be x,y,z respectively
eta=0.340859*np.ones([101*101*101,1],dtype='float32');
t=fmm.eikonalvti(velx,vel,eta,xyz=np.array([0.5,0,0]),ax=[0,0.01,101],ay=[0,0.01,101],az=[0,0.01,101],order=1);
time2=t.reshape(101,101,101,order='F');#first axis (vertical) is x, second is z


## Verify
print(['Testing result:',time.max(),time.min(),time.std(),time.var()])
print(['Correct result:',0.4845428, 0.0, 0.08635751, 0.00745762])


import matplotlib.pyplot as plt
import numpy as np

# Define dimensions
Nx, Ny, Nz = 101, 101, 101
X, Y, Z = np.meshgrid(np.arange(Nx)*0.01, np.arange(Ny)*0.01, np.arange(Nz)*0.01)

# Specify the 3D data
data=np.transpose(time,(1,0,2)); ## data requires [y,x,z] so tranpose the first and second axis

kw = {
    'vmin': data.min(),
    'vmax': data.max(),
    'levels': np.linspace(data.min(), data.max(), 10),
}

# Create a figure with 3D ax
fig = plt.figure(figsize=(16, 8))
# plt.subplot(1,2,1)
plt.jet()
ax = fig.add_subplot(121, projection='3d')

# Plot contour surfaces
_ = ax.contourf(
    X[:, :, -1], Y[:, :, -1], data[:, :, 0],
    zdir='z', offset=0, **kw
)
_ = ax.contourf(
    X[0, :, :], data[0, :, :], Z[0, :, :],
    zdir='y', offset=0, **kw
)
C = ax.contourf(
    data[:, -1, :], Y[:, -1, :], Z[:, -1, :],
    zdir='x', offset=X.max(), **kw
)
# --


# Set limits of the plot from coord limits
xmin, xmax = X.min(), X.max()
ymin, ymax = Y.min(), Y.max()
zmin, zmax = Z.min(), Z.max()
ax.set(xlim=[xmin, xmax], ylim=[ymin, ymax], zlim=[zmin, zmax])

# Plot edges
# edges_kw = dict(color='0.4', linewidth=1, zorder=1e3)
# ax.plot([xmax, xmax], [ymin, ymax], 0, **edges_kw)
# ax.plot([xmin, xmax], [ymin, ymin], 0, **edges_kw)
# ax.plot([xmax, xmax], [ymin, ymin], [zmin, zmax], **edges_kw)

# Set labels and zticks
ax.set(
    xlabel='X (km)',
    ylabel='Y (km)',
    zlabel='Z (km)',
#     zticks=[0, -150, -300, -450],
)

# Set zoom and angle view
# ax.view_init(40, -30, 0)
# ax.set_box_aspect(None, zoom=0.9)

# Colorbar
# fig.colorbar(C, ax=ax, fraction=0.02, pad=0.1, format= "%4.2f", label='Traveltime (s)')

plt.gca().scatter(0.5,0,0,s=500,marker='*',color='r')
plt.gca().set_xlim(0,1);
plt.gca().set_ylim(0,1);
plt.gca().set_zlim(0,1);
plt.gca().text(-0.124, 0, -0.66, "a)", fontsize=28, color='k')
plt.title('Isotropic',color='k', fontsize=20)
plt.gca().invert_zaxis()

ax = fig.add_subplot(122, projection='3d')
data=np.transpose(time2,(1,0,2)); ## data requires [y,x,z] so tranpose the first and second axis

# Plot contour surfaces
_ = ax.contourf(
    X[:, :, -1], Y[:, :, -1], data[:, :, 0],
    zdir='z', offset=0, **kw
)
_ = ax.contourf(
    X[0, :, :], data[0, :, :], Z[0, :, :],
    zdir='y', offset=0, **kw
)
C = ax.contourf(
    data[:, -1, :], Y[:, -1, :], Z[:, -1, :],
    zdir='x', offset=X.max(), **kw
)
# --


# Set limits of the plot from coord limits
xmin, xmax = X.min(), X.max()
ymin, ymax = Y.min(), Y.max()
zmin, zmax = Z.min(), Z.max()
ax.set(xlim=[xmin, xmax], ylim=[ymin, ymax], zlim=[zmin, zmax])

# Plot edges
# edges_kw = dict(color='0.4', linewidth=1, zorder=1e3)
# ax.plot([xmax, xmax], [ymin, ymax], 0, **edges_kw)
# ax.plot([xmin, xmax], [ymin, ymin], 0, **edges_kw)
# ax.plot([xmax, xmax], [ymin, ymin], [zmin, zmax], **edges_kw)

# Set labels and zticks
ax.set(
    xlabel='X (km)',
    ylabel='Y (km)',
    zlabel='Z (km)',
#     zticks=[0, -150, -300, -450],
)

# Set zoom and angle view
# ax.view_init(40, -30, 0)
# ax.set_box_aspect(None, zoom=0.9)

# Colorbar
# fig.colorbar(C, ax=ax, fraction=0.02, pad=0.1, orientation='horizontal', format= "%4.2f", label='Traveltime (s)')

# fig.colorbar(orientation='horizontal',cax=fig.add_axes([0.37,0.07,0.3,0.02]),shrink=1,label='Traveltime (s)');
# plt.colorbar(orientation='horizontal',cax=fig.add_axes([0.37,0.07,0.3,0.02]),shrink=1,label='Traveltime (s)');

plt.gca().scatter(0.5,0,0,s=500,marker='*',color='r')
plt.gca().set_xlim(0,1);
plt.gca().set_ylim(0,1);
plt.gca().set_zlim(0,1);
plt.gca().text(-0.124, 0, -0.66, "b)", fontsize=28, color='k')
plt.title('Anisotropic',color='k', fontsize=20)
plt.gca().invert_zaxis()

# position for the colorbar
cb = plt.colorbar(C, cax = fig.add_axes([0.37,0.1,0.3,0.02]), format= "%4.2f", orientation='horizontal',label='Traveltime (s)')

plt.savefig('test_pyekfmm_fig2.png',format='png',dpi=300,bbox_inches='tight', pad_inches=0)
plt.savefig('test_pyekfmm_fig2.pdf',format='pdf',dpi=300,bbox_inches='tight', pad_inches=0)

# Show Figure
plt.show()


## for other usage
# time=np.float32(time)
# fid = open ("time3d.bin", "wb") #binary file format, int
# fid.write(time.flatten(order='F'))


