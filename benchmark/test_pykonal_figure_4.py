#This script reproduces figure 4 in Pykonal paper; the original Pykonal script is in the Pykonal folder
#
#NOTE: The Spherical Pyekfmm currently does not support plane/line source case. For such case, pykonal is recommended
#If there is a strong need for the spherical Pyekfmm to support plane/line source case, please contact Yangkang Chen (chenyk2016@gmail.com), and we can adjust the code to make it work.

import matplotlib.gridspec as gs
import matplotlib.pyplot as plt
import numpy as np
import pykonal


# ## Initialize a solver in Cartesian coordinates

# In[2]:

solver_c = pykonal.EikonalSolver(coord_sys='cartesian')
solver_c.velocity.min_coords = -25, -25, 0
solver_c.velocity.node_intervals = 1, 1, 1
solver_c.velocity.npts = 51, 51, 1
solver_c.velocity.values = np.ones(solver_c.velocity.npts) # Uniform velocity model

# Add a line source at y = 0.
for ix in range(solver_c.velocity.npts[0]):
    src_idx = (ix, 25, 0)
    solver_c.traveltime.values[src_idx] = 0
    solver_c.unknown[src_idx] = False
    solver_c.trial.push(*src_idx)

# Solve the system
# get_ipython().run_line_magic('time', 'solver_c.solve()')
solver_c.solve()

plt.imshow(solver_c.traveltime.values)
plt.show()

################################################################################################
# Using pyekfmm
## Isotropic case (Cartesian)
################################################################################################
import pyekfmm as fmm
nz,nx=51,51
x=-25+np.arange(nx)*1
z=-25+np.arange(nz)*1
xx_c,yy_c=np.meshgrid(x,z) #for plotting
xx_c=xx_c.transpose() #for plotting
yy_c=yy_c.transpose() #for plotting

vel=np.ones(nz*nx)
t=fmm.eikonal_plane(vel,xyz=np.array([0,0,0]),ax=[-25,1,nx],ay=[0,0.05,1],az=[-25,1,nz],order=2,planedir=0);
time_c=t.reshape(nx,nz,order='F');	#first axis (vertical) is x, second is z
time_c=time_c.transpose()	   #NOW: first axis (vertical) is z, second is x

plt.imshow(time_c)
plt.show()

# ## Initialize a solver in Spherical coordinates
solver_s = pykonal.EikonalSolver(coord_sys='spherical')
solver_s.velocity.min_coords = 0.01, np.pi/2, 0
solver_s.velocity.node_intervals = 1, 1, np.pi/10
solver_s.velocity.npts = 26, 1, 21
solver_s.velocity.values = np.ones(solver_s.velocity.npts) # Uniform velocity model

# Add a line source at y = 0.
for ir in range(solver_s.velocity.npts[0]):
    for it in range(solver_s.velocity.npts[1]):
        for ip in np.argwhere(solver_s.traveltime.nodes[0,0,:,2] % np.pi  == 0).flatten():
            src_idx = (ir, it, ip)
            solver_s.traveltime.values[src_idx] = 0
            solver_s.unknown[src_idx] = False
            solver_s.trial.push(*src_idx)

# Solve the system
# get_ipython().run_line_magic('time', 'solver_s.solve()')
solver_s.solve()

plt.imshow(solver_s.traveltime.values[:,0,:])
plt.show()

################################################################################################
## Isotropic case (Spherical)
# Spherical Pyekfmm does not support plane/line source, so in this case, we use pykonal result
################################################################################################


iz, it = 0, 0
plot_kwargs = dict(
    shading='gouraud'
)
plt.close('all')

nodes = solver_s.traveltime.nodes
xx = nodes[...,0] * np.sin(nodes[...,1]) * np.cos(nodes[...,2])
yy = nodes[...,0] * np.sin(nodes[...,1]) * np.sin(nodes[...,2])

fig = plt.figure(figsize=(5, 4))

gridspec = gs.GridSpec(nrows=2, ncols=4, width_ratios=[0.08, 1, 1, 0.08])
ax1 = fig.add_subplot(gridspec[0, 1], aspect=1)
ax2 = fig.add_subplot(gridspec[0, 2], aspect=1)
ax3 = fig.add_subplot(gridspec[1, 1], aspect=1)
ax4 = fig.add_subplot(gridspec[1, 2], aspect=1)
cax1 = fig.add_subplot(gridspec[0, 3])
cax2 = fig.add_subplot(gridspec[1, 3])
ax1.set_title('Cartesian')
ax2.set_title('Spherical')
qmesh = ax1.pcolormesh(
    xx_c,
    yy_c,
    time_c,
    cmap=plt.get_cmap('jet_r'),
    **plot_kwargs
)
qmesh = ax2.pcolormesh(
    xx[:,it,:],
    yy[:,it,:],
    solver_s.traveltime.values[:,it,:],
    vmin=qmesh.get_array().min(),
    vmax=qmesh.get_array().max(),
    cmap=plt.get_cmap('jet_r'),
    **plot_kwargs
)
cbar1 =  fig.colorbar(qmesh, cax=cax1)
cbar1.set_label('Travel time [s]')
qmesh = ax3.pcolormesh(
    xx_c,
    yy_c,
    time_c-np.abs(yy_c),
    vmin=0.0,
    vmax=0.35,
    cmap=plt.get_cmap('bone_r'),
    **plot_kwargs
)
qmesh = ax4.pcolormesh(
    xx[:,it,:],
    yy[:,it,:],
    np.abs(solver_s.traveltime.values[:,it,:]-np.abs(yy[:,it,:])),
    vmin=0.0,
    vmax=0.35,
    cmap=plt.get_cmap('bone_r'),
    **plot_kwargs
)
cbar2 =  fig.colorbar(qmesh, cax=cax2)
cbar2.set_label('Absolute error [s]')

for ax in (ax1, ax2):
    ax.set_xticklabels([])
    
for ax in (ax2, ax4):
    ax.set_yticklabels([])
    
panel = ord("a")
for ax in (ax1, ax2, ax3, ax4):
    ax.text(
        0, 1.025, f"({chr(panel)})",
        ha="right",
        va="bottom",
        transform=ax.transAxes
    )
    panel += 1

fig.text(0.5, 0, "x [km]", ha="center", va="bottom")
fig.text(0.05, 0.5, "y [km]", ha="left", va="center", rotation=90)

plt.savefig('test_pykonal_figure_4.png',bbox_inches='tight')
plt.show()

# In[ ]:




