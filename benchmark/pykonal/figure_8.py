#!/usr/bin/env python
# coding: utf-8

# # Figure 8
# Compatible with PyKonal Version 0.2.0

# In[1]:


# %matplotlib ipympl

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


solver_ug = pykonal.EikonalSolver(coord_sys="cartesian")
solver_ug.vv.min_coords = solver_dg.vv.min_coords
solver_ug.vv.node_intervals = solver_dg.vv.node_intervals
solver_ug.vv.npts = solver_dg.vv.npts
solver_ug.vv.values = solver_dg.vv.values

for ix in range(solver_ug.tt.npts[0]):
    idx = (ix, solver_ug.tt.npts[1]-1, 0)
    solver_ug.tt.values[idx] = solver_dg.tt.values[idx]
    solver_ug.unknown[idx] = False
    solver_ug.trial.push(*idx)
solver_ug.solve()


# In[4]:


plt.close("all")
fig = plt.figure(figsize=(6, 2.5))

ax1 = fig.add_subplot(2, 1, 1)
ax2 = fig.add_subplot(2, 1, 2)

ax = fig.add_subplot(1, 1, 1, frameon=False)
plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
ax.set_ylabel("Depth [km]")
ax2.set_xlabel("Horizontal offset [km]")
ax1.set_xticklabels([])

for solver, ax, panel in (
    (solver_dg, ax1, f"(a)"), 
    (solver_ug, ax2, f"(b)")
):
    ax.text(-0.025, 1.05, panel, va="bottom", ha="right", transform=ax.transAxes)
    qmesh = ax.pcolormesh(
        solver.vv.nodes[:,:,0,0], 
        solver.vv.nodes[:,:,0,1], 
        solver.vv.values[:,:,0],
        cmap=plt.get_cmap("hot")
    )
    ax.contour(
        solver.tt.nodes[:,:,0,0], 
        solver.tt.nodes[:,:,0,1], 
        solver.tt.values[:,:,0],
        colors="k",
        linestyles="--",
        linewidths=1,
        levels=np.arange(0, solver.tt.values.max(), 0.25)
    )
    ax.scatter(
        solver.vv.nodes[src_idx + (0,)],
        solver.vv.nodes[src_idx + (1,)],
        marker="*",
        facecolor="w",
        edgecolor="k",
        s=256
    )
    ax.invert_yaxis()
cbar = fig.colorbar(qmesh, ax=(ax1, ax2))
cbar.set_label("Velocity [km/s]")


# In[ ]:




