#!/usr/bin/env python
# coding: utf-8

# # Figure 5
# Compatible with PyKonal Version 0.2.0

# In[1]:


# %matplotlib ipympl
import matplotlib
import matplotlib.gridspec as gs
import matplotlib.pyplot as plt
import numpy as np
import os
import pkg_resources
import pykonal


# ### Load Marmousi 2D velocity model

# In[2]:


fname = pkg_resources.resource_filename(
    'pykonal',
    os.path.join('data', "marmousi2", 'marmousi2.npz')
)
with np.load(fname) as infile:
    vv = infile['vv']


# ### Initialize the EikonalSolver

# In[3]:


velocity  = pykonal.fields.ScalarField3D(coord_sys="cartesian")
velocity.min_coords     = 0, 0, 0
velocity.node_intervals = 0.004, 0.004, 1
velocity.npts           = vv.shape
velocity.values         = vv


# In[4]:


traveltime_fields = dict()
for decimation_factor in range(7, -1, -1):
    decimation_factor = 2**decimation_factor
    
    vv = velocity.values[::decimation_factor, ::decimation_factor]

    solver = pykonal.EikonalSolver(coord_sys="cartesian")

    solver.velocity.min_coords     = 0, 0, 0
    solver.velocity.node_intervals = velocity.node_intervals * decimation_factor
    solver.velocity.npts = vv.shape
    solver.velocity.values = vv

    src_idx = 0, 0, 0
    solver.traveltime.values[src_idx] = 0
    solver.unknown[src_idx] = False
    solver.trial.push(*src_idx)

    get_ipython().run_line_magic('time', 'solver.solve()')
    traveltime_fields[decimation_factor] = solver.traveltime


# ### Plot the results

# In[5]:


plt.close('all')
fig = plt.figure(figsize=(5, 4.25))
ax = fig.add_subplot(1, 1, 1, frameon=False)
plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
ax.set_xlabel("Horizontal offset [km]")
ax.set_ylabel("Depth [km]")

gridspec = gs.GridSpec(nrows=8, ncols=2, height_ratios=[0.08, 1, 1, 1, 1, 0.2, 0.08, 0.72], width_ratios=[1, 1])
cax00 = fig.add_subplot(gridspec[0, 0])
cax01 = fig.add_subplot(gridspec[0, 1])
cax51 = fig.add_subplot(gridspec[6, 1])
ax10 = fig.add_subplot(gridspec[1, 0])
ax11 = fig.add_subplot(gridspec[1, 1])
ax20 = fig.add_subplot(gridspec[2, 0])
ax21 = fig.add_subplot(gridspec[2, 1])
ax30 = fig.add_subplot(gridspec[3, 0])
ax31 = fig.add_subplot(gridspec[3, 1])
ax40 = fig.add_subplot(gridspec[4, 0])
ax41 = fig.add_subplot(gridspec[4, 1])
ax50 = fig.add_subplot(gridspec[5:, 0])

panel = ord("a")
for ax in (ax10, ax11, ax20, ax21, ax30, ax31, ax40, ax41, ax50):
    ax.text(-0.05, 0.8, f"({chr(panel)})", ha="right", va="top", transform=ax.transAxes)
    panel += 1

qmesh = ax10.pcolormesh(
    velocity.nodes[:,:,0,0],
    velocity.nodes[:,:,0,1],
    velocity.values[:,:,0],
    cmap=plt.get_cmap("hot")
)
cbar = fig.colorbar(qmesh, cax=cax00, orientation="horizontal")
cbar.set_label("Velocity [km/s]")
cbar.ax.xaxis.tick_top()
cbar.ax.xaxis.set_label_position("top")

tt0 = traveltime_fields[1]
qmesh = ax11.pcolormesh(
    tt0.nodes[:,:,0,0],
    tt0.nodes[:,:,0,1],
    tt0.values[:,:,0],
    cmap=plt.get_cmap("jet"),
)
ax11.contour(
    tt0.nodes[:,:,0,0],
    tt0.nodes[:,:,0,1],
    tt0.values[:,:,0],
    colors="k",
    levels=np.arange(0, tt0.values.max(), 0.25),
    linewidths=0.5,
    linestyles="--"
)
cbar = fig.colorbar(qmesh, cax=cax01, orientation="horizontal")
cbar.set_label("Traveltime [s]")
cbar.ax.xaxis.tick_top()
cbar.ax.xaxis.set_label_position("top")

for ax, decimation_factor in ((ax20, 128), (ax21, 64), (ax30, 32), (ax31, 16), (ax40, 8), (ax41, 4), (ax50, 2)):
    tt = traveltime_fields[decimation_factor]
    qmesh = ax.pcolormesh(
        tt.nodes[:,:,0,0],
        tt.nodes[:,:,0,1],
        np.abs(tt.values[:,:,0] - tt0.values[::decimation_factor, ::decimation_factor, 0]),
        cmap=plt.get_cmap("bone_r"),
        vmin=0,
        vmax=0.62
    )
    ax.text(
        0.05, 0.95,
        f"$d={decimation_factor}$",
        ha="left",
        va="top",
        transform=ax.transAxes
    )
cbar = fig.colorbar(qmesh, cax=cax51, orientation="horizontal")
cbar.set_label("$\Delta t$ [s]")

for ax in (ax10, ax11, ax20, ax21, ax30, ax31, ax40):
    ax.set_xticklabels([])
for ax in (ax11, ax21, ax31, ax41):
    ax.yaxis.tick_right()
for ax in (ax10, ax11, ax20, ax21, ax30, ax31, ax40, ax41, ax50):
    ax.set_yticks([0, 4])
    ax.set_xticks([0, 8, 16, 24])
    ax.set_xlim(0, 27.136)
    ax.set_ylim(0, 5.12)
    ax.invert_yaxis()

plt.savefig('pykonal_figure_5.png')
plt.show()
# In[ ]:




