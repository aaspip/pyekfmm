#!/usr/bin/env python
# coding: utf-8

# # Figure 7
# Compatible with PyKonal Version 0.2.0

# In[1]:


# %matplotlib ipympl

import matplotlib.gridspec as gs
import matplotlib.pyplot as plt
import numpy as np
import os
import pkg_resources
import pykonal


# In[2]:


fname = pkg_resources.resource_filename(
    "pykonal",
    os.path.join("data", "mitp2008.npz")
)

with np.load(fname) as npz:
    vv = pykonal.fields.ScalarField3D(coord_sys='spherical')
    dv = pykonal.fields.ScalarField3D(coord_sys='spherical')
    vv.min_coords = dv.min_coords = npz['min_coords_2d'] + [0, 0, np.pi/4]
    vv.node_intervals = dv.node_intervals = npz['node_intervals_2d']
    vv.npts = dv.npts = npz['npts_2d'][0], npz['npts_2d'][1], npz['npts_2d'][2]//4
    vv.values = npz['vv_2d'][:,:,128:256]
    dv.values = npz["dv_2d"][:,:,128:256]

# Resample the velocity model (64, 1, 128) --> (1024, 1, 2048)
rho_min, theta_min, phi_min = vv.min_coords
rho_max, theta_max, phi_max = vv.max_coords
nrho, ntheta, nphi = 1024, 1, 2048

drho = (rho_max - rho_min) / (nrho - 1)
rho = np.linspace(rho_min, rho_max, nrho)

# dtheta = (theta_max - theta_min) / (ntheta - 1)
dtheta = 1
theta = np.linspace(theta_min, theta_max, ntheta)

dphi = (phi_max - phi_min) / (nphi - 1)
phi = np.linspace(phi_min, phi_max, nphi)

rtp = np.moveaxis(
    np.stack(np.meshgrid(rho, theta, phi, indexing="ij")),
    0, 
    -1
)
vv_new = vv.resample(rtp.reshape(-1, 3)).reshape(rtp.shape[:-1])
dv_new = dv.resample(rtp.reshape(-1, 3)).reshape(rtp.shape[:-1])

vv = pykonal.fields.ScalarField3D(coord_sys="spherical")
dv = pykonal.fields.ScalarField3D(coord_sys="spherical")
vv.min_coords = dv.min_coords = rho_min, theta_min, phi_min
vv.node_intervals = dv.node_intervals = drho, dtheta, dphi
vv.npts = dv.npts = nrho, ntheta, nphi
vv.values = vv_new
dv.values = dv_new

velocity = vv


# In[3]:


SRC_IDX = np.array([512, 0, 1024])

traveltime_fields = dict()
for decimation_factor in range(7, -1, -1):
    decimation_factor = 2**decimation_factor
    
    vv = velocity.values[::decimation_factor, :, ::decimation_factor]

    solver = pykonal.EikonalSolver(coord_sys="spherical")

    solver.vv.min_coords = velocity.min_coords
    solver.vv.node_intervals = velocity.node_intervals * decimation_factor
    solver.vv.npts = vv.shape
    solver.vv.values = vv

    src_idx = tuple(SRC_IDX // decimation_factor - [1, 0, 1])
    print(src_idx)
    solver.traveltime.values[src_idx] = 0
    solver.unknown[src_idx] = False
    solver.trial.push(*src_idx)

    get_ipython().run_line_magic('time', 'solver.solve()')
    traveltime_fields[decimation_factor] = solver.traveltime


# In[4]:


plt.close('all')
fig = plt.figure(figsize=(4.5, 6.5))

fig.text(0, 0.5, "$y$ [km]", ha="left", va="center", rotation=90)
fig.text(0.5, 0.05, "$x$ [km]", ha="center", va="bottom")

gridspec = gs.GridSpec(nrows=6, ncols=2, height_ratios=[0.08, 1, 1, 1, 1, 1], width_ratios=[1, 1])
cax00 = fig.add_subplot(gridspec[0, 0])
cax01 = fig.add_subplot(gridspec[0, 1])
ax10 = fig.add_subplot(gridspec[1, 0], aspect=1)
ax11 = fig.add_subplot(gridspec[1, 1], aspect=1)
ax20 = fig.add_subplot(gridspec[2, 0], aspect=1)
ax21 = fig.add_subplot(gridspec[2, 1], aspect=1)
ax30 = fig.add_subplot(gridspec[3, 0], aspect=1)
ax31 = fig.add_subplot(gridspec[3, 1], aspect=1)
ax40 = fig.add_subplot(gridspec[4, 0], aspect=1)
ax41 = fig.add_subplot(gridspec[4, 1], aspect=1)
ax50 = fig.add_subplot(gridspec[5, 0], aspect=1)

gridspec = gs.GridSpec(nrows=8, ncols=2, height_ratios=[0.08, 1, 1, 1, 1, 0.2, 0.08, 0.72], width_ratios=[1, 1])
cax51 = fig.add_subplot(gridspec[6, 1])

panel = ord("a")
for ax in (ax10, ax11, ax20, ax21, ax30, ax31, ax40, ax41, ax50):
    ax.text(-0.05, 1.1, f"({chr(panel)})", ha="right", va="top", transform=ax.transAxes)
    panel += 1

nodes = dv.nodes
xx = nodes[...,0] * np.sin(nodes[...,1]) * np.cos(nodes[...,2])
yy = nodes[...,0] * np.sin(nodes[...,1]) * np.sin(nodes[...,2])
qmesh = ax10.pcolormesh(
    xx[:,0],
    yy[:,0],
    dv.values[:,0],
    cmap=plt.get_cmap("seismic_r"),
    vmin=-1.25,
    vmax=1.25,
    shading="gouraud"
)
cbar = fig.colorbar(qmesh, cax=cax00, orientation="horizontal")
cbar.set_label("$dV_P/V_P$ [\%]")
cbar.ax.xaxis.tick_top()
cbar.ax.xaxis.set_label_position("top")

tt0 = traveltime_fields[1]
nodes = tt0.nodes
xx = nodes[...,0] * np.sin(nodes[...,1]) * np.cos(nodes[...,2])
yy = nodes[...,0] * np.sin(nodes[...,1]) * np.sin(nodes[...,2])
qmesh = ax11.pcolormesh(
    xx[:,0],
    yy[:,0],
    tt0.values[:,0],
    cmap=plt.get_cmap("jet"),
    shading="gouraud"
)
ax11.contour(
    xx[:,0],
    yy[:,0],
    tt0.values[:,0],
    colors="k",
    levels=np.arange(0, tt0.values.max(), 20),
    linewidths=0.5,
    linestyles="--"
)
ax11.scatter(
    xx[511, 0, 1023], yy[511, 0, 1023],
    marker="*",
    s=250,
    facecolor="w",
    edgecolor="k"
)
cbar = fig.colorbar(qmesh, cax=cax01, orientation="horizontal")
cbar.set_label("Traveltime [s]")
cbar.ax.xaxis.tick_top()
cbar.ax.xaxis.set_label_position("top")

for ax, decimation_factor in ((ax20, 128), (ax21, 64), (ax30, 32), (ax31, 16), (ax40, 8), (ax41, 4), (ax50, 2)):
    tt = traveltime_fields[decimation_factor]
    nodes = tt.nodes
    xx = nodes[...,0] * np.sin(nodes[...,1]) * np.cos(nodes[...,2])
    yy = nodes[...,0] * np.sin(nodes[...,1]) * np.sin(nodes[...,2])
    qmesh = ax.pcolormesh(
        xx[:,0],
        yy[:,0],
        np.abs(tt.values[:,0] - tt0.values[::decimation_factor, 0, ::decimation_factor]),
        cmap=plt.get_cmap("bone_r"),
        vmax=50,
        shading="gouraud"
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
    ax.set_yticks([3500, 5500])
    ax.set_xlim(-4500, 4500)
    ax.set_ylim(2500, 6500)

plt.savefig('pykonal_figure_7.png')
plt.show()
# In[ ]:




