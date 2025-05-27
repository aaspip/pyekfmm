#This script reproduces figure 8 in the Pykonal paper; the original Pykonal script is in the Pykonal folder
#
# NOTE: This test was not successful (traveltime calculation for reflection waves are not currently supported)

import matplotlib.pyplot as plt
import numpy as np
import os
import pkg_resources
import pykonal
import scipy.ndimage


# In[2]:


velocity = pykonal.fields.ScalarField3D(coord_sys="cartesian")
velocity.min_coords = 0, 0, 0
velocity.node_intervals = 0.1, 0.1, 0.1
velocity.npts = 512, 128, 1
velocity.values = scipy.ndimage.gaussian_filter(20. * np.random.randn(*velocity.npts) + 6., 10)

np.random.seed(202314)
vv0=scipy.ndimage.gaussian_filter(20. * np.random.randn(*velocity.npts) + 6., 10);
min_coords = np.array([0, 0, 0])
node_intervals = np.array([0.1, 0.1, 0.1])
npts=np.array([512, 128, 1])
# In[3]:


solver_dg = pykonal.EikonalSolver(coord_sys="cartesian")
solver_dg.vv.min_coords = velocity.min_coords
solver_dg.vv.node_intervals = velocity.node_intervals
solver_dg.vv.npts = velocity.npts
solver_dg.vv.values = velocity.values

src_idx = (127, 32, 0)
solver_dg.tt.values[src_idx] = 0
solver_dg.unknown[src_idx] = False
solver_dg.trial.push(*src_idx)
solver_dg.solve()

################################################################################################
# Using pyekfmm
## Isotropic case (Cartesian)
################################################################################################
import pyekfmm as fmm
x0=min_coords[0];y0=min_coords[1];z0=min_coords[2];
dx=node_intervals[0];dy=node_intervals[1];dz=node_intervals[2];
nx=npts[0];ny=npts[1];nz=npts[2];

x=x0+np.arange(nx)*dx;
y=y0+np.arange(ny)*dy;
z=z0+np.arange(nz)*dz;

vel=vv0.flatten(order='F') #default: (6801, 1401, 1), x,y,z
t=fmm.eikonal(vel,xyz=np.array([x[src_idx[0]],y[src_idx[1]],z[src_idx[2]]]),ax=[x0,dx,nx],ay=[y0,dy,ny],az=[z0,dz,nz],order=2);
time_c=t.reshape(nx,ny,nz,order='F');	#first axis (vertical) is x, second is z

plt.imshow(np.concatenate([solver_dg.traveltime.values[:,:,-1].transpose(),time_c[:,:,-1].transpose(),solver_dg.traveltime.values[:,:,-1].transpose()-time_c[:,:,-1].transpose()],axis=1),aspect='auto')
plt.show()

plt.imshow(np.concatenate([solver_dg.tt.values[:,:,-1].transpose(),time_c[:,:,-1].transpose(),solver_dg.traveltime.values[:,:,-1].transpose()-time_c[:,:,-1].transpose()],axis=1),aspect='auto')
plt.show()

print('np.linalg.norm(solver_dg.traveltime.values[:,:,-1]-solver_dg.tt.values[:,:,-1])',np.linalg.norm(solver_dg.traveltime.values[:,:,-1]-solver_dg.tt.values[:,:,-1])
)

time_dg = time_c;
# plt.imshow(time_dg[:,:,0].transpose());plt.colorbar();
# plt.show()

solver_ug = pykonal.EikonalSolver(coord_sys="cartesian")
solver_ug.vv.min_coords = solver_dg.vv.min_coords
solver_ug.vv.node_intervals = solver_dg.vv.node_intervals
solver_ug.vv.npts = solver_dg.vv.npts
solver_ug.vv.values = solver_dg.vv.values

#linesource
# tt1=solver_ug.tt.values[:,:,-1].copy();
for ix in range(solver_ug.tt.npts[0]):
    idx = (ix, solver_ug.tt.npts[1]-1, 0)
    solver_ug.tt.values[idx] = solver_dg.tt.values[idx]
    solver_ug.unknown[idx] = False
    solver_ug.trial.push(*idx)
# tt2=solver_ug.tt.values[:,:,-1].copy();
solver_ug.solve()
# tt3=solver_ug.tt.values[:,:,-1].copy();

# plt.imshow(np.concatenate([tt2.transpose(),tt3.transpose(),tt2.transpose()-tt3.transpose()],axis=1),aspect='auto')
# plt.show()
# 
# plt.imshow(tt2)
# plt.show()

indx=np.argmin(time_dg[:,-1,0])
t=fmm.eikonal_plane(vel,xyz=np.array([x[indx],y[-1],0]),ax=[x0,dx,nx],ay=[y0,dy,ny],az=[z0,dz,nz],order=2, planedir=0);
time_c2=t.reshape(nx,ny,nz,order='F');	#first axis (vertical) is x, second is z
# time_ug=time_dg+time_c2;
# time_ug = time_c2 + np.matmul(time_dg[:,-1,:],np.ones([1,128]))[:,:,np.newaxis]
time_ug=time_dg[:,-1,0].min()+time_c2;

plt.imshow(np.concatenate([solver_ug.traveltime.values[:,:,-1].transpose(),time_ug[:,:,-1].transpose(),solver_ug.traveltime.values[:,:,-1].transpose()-time_ug[:,:,-1].transpose()],axis=1),aspect='auto')
plt.show()

plt.imshow(np.concatenate([solver_ug.traveltime.values[:,:,-1].transpose(),time_c2[:,:,-1].transpose(),solver_ug.traveltime.values[:,:,-1].transpose()-time_c2[:,:,-1].transpose()],axis=1),aspect='auto')
plt.colorbar()
plt.show()


# time_ug=time_ug.transpose()	   #NOW: first axis (vertical) is z, second is x



# In[4]:

# Define nodes as required by visualization
xx,yy=np.meshgrid(x,y) #for plotting
xx=xx.transpose() #for plotting
yy=yy.transpose() #for plotting


plt.close("all")
fig = plt.figure(figsize=(6, 2.5))

ax1 = fig.add_subplot(2, 1, 1)
ax2 = fig.add_subplot(2, 1, 2)

ax = fig.add_subplot(1, 1, 1, frameon=False)
plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
ax.set_ylabel("Depth [km]")
ax2.set_xlabel("Horizontal offset [km]")
ax1.set_xticklabels([])

for time, ax, panel in (
    (time_dg, ax1, f"(a)"), 
    (time_ug, ax2, f"(b)")
):
    ax.text(-0.025, 1.05, panel, va="bottom", ha="right", transform=ax.transAxes)
    qmesh = ax.pcolormesh(
        xx, 
        yy, 
        vv0[:,:,0],
        cmap=plt.get_cmap("hot")
    )
    ax.contour(
        xx, 
        yy, 
        time[:,:,0],
        colors="k",
        linestyles="--",
        linewidths=1,
        levels=np.arange(0, time.max(), 0.25)
    )
    ax.scatter(
        x[src_idx[0] + 0],
        y[src_idx[1] + 1],
        marker="*",
        facecolor="w",
        edgecolor="k",
        s=256
    )
    ax.invert_yaxis()
cbar = fig.colorbar(qmesh, ax=(ax1, ax2))
cbar.set_label("Velocity [km/s]")

plt.savefig('test_pykonal_figure_8.png',bbox_inches='tight')
plt.show()
# In[ ]:




