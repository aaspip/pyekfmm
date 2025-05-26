#!/usr/bin/env python
# coding: utf-8

# # Figure 6
# Compatible with PyKonal Version 0.2.0

# In[1]:


# %matplotlib ipympl

import matplotlib.gridspec as gs
import matplotlib.pyplot as plt
import numpy as np
import pykonal

from matplotlib import markers
from matplotlib.path import Path
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset


# In[2]:


def align_marker(marker, halign='center', valign='middle',):
    """
    create markers with specified alignment.

    Parameters
    ----------

    marker : a valid marker specification.
      See mpl.markers

    halign : string, float {'left', 'center', 'right'}
      Specifies the horizontal alignment of the marker. *float* values
      specify the alignment in units of the markersize/2 (0 is 'center',
      -1 is 'right', 1 is 'left').

    valign : string, float {'top', 'middle', 'bottom'}
      Specifies the vertical alignment of the marker. *float* values
      specify the alignment in units of the markersize/2 (0 is 'middle',
      -1 is 'top', 1 is 'bottom').

    Returns
    -------

    marker_array : numpy.ndarray
      A Nx2 array that specifies the marker path relative to the
      plot target point at (0, 0).

    Notes
    -----
    The mark_array can be passed directly to ax.plot and ax.scatter, e.g.::

        ax.plot(1, 1, marker=align_marker('>', 'left'))

    """

    halign = {'right': -1.,
              'middle': 0.,
              'center': 0.,
              'left': 1.,
              }[halign]

    valign = {'top': -1.,
              'middle': 0.,
              'center': 0.,
              'bottom': 1.,
              }[valign]

    # Define the base marker
    bm = markers.MarkerStyle(marker)

    # Get the marker path and apply the marker transform to get the
    # actual marker vertices (they should all be in a unit-square
    # centered at (0, 0))
    m_arr = bm.get_path().transformed(bm.get_transform()).vertices

    # Shift the marker vertices for the specified alignment.
    m_arr[:, 0] += halign / 2
    m_arr[:, 1] += valign / 2

    return Path(m_arr, bm.get_path().codes)


# # Define the velocity model

# In[3]:


# Set the velocity gradient in 1/s
velocity_gradient = 0.25

velocity = pykonal.fields.ScalarField3D(coord_sys="cartesian")
velocity.min_coords = 0, 0, 0
velocity.npts = 4096, 1024, 1
velocity.node_intervals = [40, 10, 1] / velocity.npts
velocity.values = np.full(velocity.npts, fill_value=4.5)

for iy in range(velocity.npts[1]):
    velocity.values[:,iy] += velocity_gradient * velocity.nodes[0,iy,0,1]


# # Determine the analytical solution

# In[4]:


# Set the take-off angle in radians.
takeoff_angle = np.radians(50)

# Set the index of the node corresponding to the source location.
src_idx = np.array([128, 128, 0])

# Get the source coordinates.
src_coords = velocity.nodes[tuple(src_idx)]

# Compute the radius of the circle defining the raypath.
radius = velocity.value(src_coords) / (velocity_gradient * np.sin(takeoff_angle))

# Compute the coordinates of the center of the circle.
center_coords = src_coords + radius*np.array([np.cos(takeoff_angle), np.sin(takeoff_angle), 0])

# Define function mapping horizontal coordinate to vertical
def vertical_coords(horizontal_coord, src_coords=src_coords, radius=radius, takeoff_angle=takeoff_angle):
    sqrt = np.sqrt(radius**2 - (horizontal_coord - src_coords[0] - radius*np.cos(takeoff_angle))**2)
    first_two_terms = src_coords[1]  -  radius * np.sin(takeoff_angle)
    return (first_two_terms + sqrt, first_two_terms - sqrt)

# Define function mapping vertical coordinate to horizontal
def horizontal_coords(vertical_coord, src_coords=src_coords, radius=radius, takeoff_angle=takeoff_angle):
    sqrt = np.sqrt(radius**2 - (src_coords[1] - vertical_coord - radius*np.sin(takeoff_angle))**2)
    first_two_terms = src_coords[0]  +  radius * np.cos(takeoff_angle)
    return (first_two_terms + sqrt, first_two_terms - sqrt)

rec_coords = [horizontal_coords(0)[0], 0]


# # Compute numerical solutions

# In[6]:


rays = dict()

for decimation_factor in range(6, 1, -1):
    decimation_factor = 2**decimation_factor
    
    vv = velocity.values[::decimation_factor, ::decimation_factor]

    solver = pykonal.EikonalSolver(coord_sys="cartesian")

    solver.velocity.min_coords = 0, 0, 0
    solver.velocity.node_intervals = velocity.node_intervals * decimation_factor
    solver.velocity.npts = vv.shape
    solver.velocity.values = vv

    idx = tuple((src_idx / decimation_factor).astype(np.int_) - [1, 1, 0])
    solver.traveltime.values[idx] = 0
    solver.unknown[idx] = False
    solver.trial.push(*idx)

    get_ipython().run_line_magic('time', 'solver.solve()')
    rays[decimation_factor] = solver.trace_ray(np.array([rec_coords[0], 0, 0]))


# # Plot the results

# In[7]:


zoom = 22
dx = 0.5
dy = 0.5
anchor_y = 1.1


plt.close("all")
fig = plt.figure(figsize=(6, 3))

# Set up the main Axes.
ax0 = fig.add_subplot(2, 1, 2, aspect=1)
ax0.set_xlabel("Horizontal offset [km]")
ax0.set_ylabel("Depth [km]")
ax0.set_xlim(0, rec_coords[0]+src_coords[0])
ax0.set_ylim(-2, 10)
ax0.invert_yaxis()

# Set up the first inset Axes.
axins1 = zoomed_inset_axes(
    ax0,
    zoom=zoom,
    loc="lower left",
    bbox_to_anchor=(0, anchor_y),
    bbox_transform=ax0.transAxes
)
axins1.set_xlim(src_coords[0]-dx/2, src_coords[0]+dx/2)
axins1.set_ylim(src_coords[1]+dy/2, src_coords[1]-dy/2)

# Set up the second inset Axes.
axins2 = zoomed_inset_axes(
    ax0,
    zoom=zoom,
    loc="lower center",
    bbox_to_anchor=(0.5, anchor_y),
    bbox_transform=ax0.transAxes
)
turning_point = (src_coords[0] + rec_coords[0]) / 2
axins2.set_xlim(turning_point-dx/2, turning_point+dx/2)
axins2.set_ylim(vertical_coords(turning_point)[0]+dy/2, vertical_coords(turning_point)[0]-dy/2)

# Set up the third inset Axes.
axins3 = zoomed_inset_axes(
    ax0,
    zoom=zoom,
    loc="lower right",
    bbox_to_anchor=(1, anchor_y),
    bbox_transform=ax0.transAxes
)
axins3.set_xlim(horizontal_coords(0)[0]-3*dx/4, horizontal_coords(0)[0]+dx/4)
axins3.set_ylim(3*dy/4, -dy/4)

for ax in (ax0, axins1):
    # Plot the source location.
    ax.scatter(
        src_coords[0], src_coords[1], 
        marker="*",
        s=250,
        facecolor="w",
        edgecolor="k"
    )

for ax in (ax0, axins3):
    # Plot the receiver location.
    ax.scatter(
        rec_coords[0], rec_coords[1], 
        marker=align_marker("v", valign="bottom"),
        s=250,
        facecolor="w",
        edgecolor="k"
    )
    
xx = np.linspace(src_coords[0], rec_coords[0])
yy = vertical_coords(xx)[0]
label = True
for ax in (ax0, axins1, axins2, axins3):
    # Plot the synthetic raypaths.
    for decimation_factor in rays:
        ax.plot(
            rays[decimation_factor][:,0], 
            rays[decimation_factor][:,1],
            linewidth=1,
            label=f"d={decimation_factor}" if label is True else None
        )
    # Plot the analytic raypath.
    ax.plot(xx, yy, "k--", label="Analytical" if label is True else None)
    label = False
    
for ax in (axins1, axins2, axins3):
    mark_inset(ax0, ax, loc1=3, loc2=4, fc="none", ec="0.5", linestyle="-.")
    ax.text(0.5, 0.95, f"{zoom}x", ha="center", va="top", transform=ax.transAxes)
    ax.set_xticks(np.arange(*ax.get_xlim(), 0.2))
    ax.set_yticks(np.arange(*ax.get_ylim(), -0.2))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(direction="in")

label = ord("a")
for ax in (axins1, axins2, axins3, ax0):
    ax.text(
        0, 1.05, f"{chr(label)})",
        ha="center",
        va="bottom",
        transform=ax.transAxes
    )
    label += 1
fig.legend(loc="center right")
fig.tight_layout()

plt.savefig('pykonal_figure_6.png')
plt.show()
# In[ ]:




