#!/usr/bin/env python
# coding: utf-8

# # Figure 3
# Compatible with PyKonal Version 0.2.0

# In[2]:


# %matplotlib ipympl

import matplotlib.gridspec as gs
import matplotlib.pyplot as plt
import numpy as np
import pykonal


# ## Initialize a solver in Cartesian coordinates

# In[3]:


solver_c = pykonal.EikonalSolver(coord_sys='cartesian')
solver_c.velocity.min_coords = -25, -25, 0
solver_c.velocity.node_intervals = 1, 1, 1
solver_c.velocity.npts = 51, 51, 1
solver_c.velocity.values = np.ones(solver_c.velocity.npts) # Uniform velocity model

# Add a point source at the origin of the grid.
src_idx = (25, 25, 0)
solver_c.traveltime.values[src_idx] = 0
solver_c.unknown[src_idx] = False
solver_c.trial.push(*src_idx)

# Solve the system.
# get_ipython().run_line_magic('time', 'solver_c.solve()')
solver_c.solve()

# ## Initialize a solver in Spherical coordinates

# In[4]:


solver_s = pykonal.EikonalSolver(coord_sys='spherical')
solver_s.velocity.min_coords = 0.01, np.pi/2, 0
solver_s.velocity.node_intervals = 1, 1, np.pi/10
solver_s.velocity.npts = 25, 1, 21
solver_s.velocity.values = np.ones(solver_s.velocity.npts) # Uniform velocity model

# Add a point source at the origin of the grid.
for it in range(solver_s.velocity.npts[1]):
    for ip in range(solver_s.velocity.npts[2]):
        src_idx = (0, it, ip)
        vv = solver_s.velocity.values[src_idx]
        solver_s.traveltime.values[src_idx] = solver_s.velocity.nodes[src_idx + (0,)] / vv
        solver_s.unknown[src_idx] = False
        solver_s.trial.push(*src_idx)
        
# Solve the system
# get_ipython().run_line_magic('time', 'solver_s.solve()')
solver_s.solve()

# ## Plot the results

# In[5]:


iz, it = 0, 0
plot_kwargs = dict(
    shading='gouraud'
)
plt.close('all')

nodes = solver_s.traveltime.nodes
xx = nodes[...,0] * np.sin(nodes[...,1]) * np.cos(nodes[...,2])
yy = nodes[...,0] * np.sin(nodes[...,1]) * np.sin(nodes[...,2])

fig = plt.figure(figsize=(4.5, 4))

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
    solver_c.traveltime.nodes[:,:,iz,0],
    solver_c.traveltime.nodes[:,:,iz,1],
    solver_c.traveltime.values[:,:,iz],
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
    solver_c.traveltime.nodes[:,:,iz,0],
    solver_c.traveltime.nodes[:,:,iz,1],
    solver_c.traveltime.values[:,:,iz]-np.sqrt(np.sum(np.square(solver_c.traveltime.nodes[:,:,iz]), axis=-1)),
    vmin=0.0,
    vmax=0.35,
    cmap=plt.get_cmap('bone_r'),
    **plot_kwargs
)
qmesh = ax4.pcolormesh(
    xx[:,it,:],
    yy[:,it,:],
    solver_s.traveltime.values[:,it,:]-np.sqrt(xx[:,it,:]**2 + yy[:,it,:]**2),
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

plt.savefig('pykonal_figure_3.png')
plt.show()
# In[ ]: